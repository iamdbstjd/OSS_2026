"""Resumable Stage 10 query-by-plan profiler over the final scheduler path."""

from __future__ import annotations

import asyncio
import csv
import fcntl
import hashlib
import io
import os
import random
import time
from collections import defaultdict
from collections.abc import Awaitable, Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from pydantic import Field, model_validator

from ragplan.benchmark.aggregate import percentile_type7
from ragplan.benchmark.artifacts import write_bytes, write_json, write_json_model
from ragplan.benchmark.contracts import SplitName, canonical_json_bytes, canonical_sha256
from ragplan.benchmark.metrics import mrr_at_10, ndcg_at_10, recall_at_5, recall_at_10
from ragplan.benchmark.profile_records import (
    FINAL_ENGINE_PATH,
    ProfilePlanFeatures,
    ProfileProtocolConfig,
    ProfileRunManifest,
    ProfileTrialRecord,
    QualityScope,
    TrainingMatrixRow,
    profile_records_sha256,
    profile_trial_identity,
)
from ragplan.benchmark.records import (
    BenchmarkQueryIdentity,
    BranchTrialRecord,
    EnvironmentManifest,
    TrialPhase,
    TrialStatus,
    benchmark_query_identities_sha256,
)
from ragplan.benchmark.runner import BenchmarkCase, RawEvidenceError
from ragplan.core.engine import SearchEngine
from ragplan.core.errors import ErrorCode, RAGPlanError
from ragplan.core.models import (
    FrozenModel,
    PlannerMode,
    QueryFeatures,
    RequestState,
    SearchRequest,
    SearchResponse,
    SearchStatus,
)


@dataclass(frozen=True, slots=True)
class ProfileInvocation:
    run_id: str
    trial_id: str
    query_id: str
    query: str
    plan_features: ProfilePlanFeatures
    latency_budget_ms: int
    final_top_k: int
    phase: TrialPhase
    repetition: int


class ProfileTrialObservation(FrozenModel):
    status: TrialStatus
    ranked_chunk_ids: tuple[str, ...] = ()
    selected_plan_id: str | None = None
    query_features: QueryFeatures | None = None
    analyzer_latency_ms: float = Field(default=0.0, ge=0.0)
    planner_latency_ms: float = Field(default=0.0, ge=0.0)
    embedding_latency_ms: float = Field(default=0.0, ge=0.0)
    execution_latency_ms: float | None = Field(default=None, ge=0.0)
    vector_latency_ms: float | None = Field(default=None, ge=0.0)
    graph_latency_ms: float | None = Field(default=None, ge=0.0)
    fusion_latency_ms: float = Field(default=0.0, ge=0.0)
    rerank_latency_ms: float = Field(default=0.0, ge=0.0)
    total_latency_ms: float = Field(ge=0.0)
    branch_results: tuple[BranchTrialRecord, ...] = ()
    timeout: bool = False
    fallback: bool = False
    budget_violated: bool = False
    error_code: ErrorCode | None = None
    scheduler_trace_present: bool = False
    scheduler_runtime_semantics_version: str | None = None
    trace_config_version: str | None = None
    trace_model_version: str | None = None

    @model_validator(mode="after")
    def _consistent_status(self) -> ProfileTrialObservation:
        if self.timeout is not (self.status is TrialStatus.TIMEOUT):
            raise ValueError("profile observation timeout flag must match status")
        if self.fallback is not (self.status is TrialStatus.PARTIAL):
            raise ValueError("profile observation fallback flag must match partial status")
        if self.status in {TrialStatus.TIMEOUT, TrialStatus.ERROR}:
            if self.ranked_chunk_ids or self.error_code is None:
                raise ValueError("failed profile observations require zero results and an error")
        elif self.error_code is not None:
            raise ValueError("successful profile observations cannot have a top-level error")
        return self


@dataclass(frozen=True, slots=True)
class ProfileRunnerSummary:
    run_id: str
    expected_rows: int
    preexisting_rows: int
    executed_rows: int
    total_rows: int
    complete: bool
    raw_path: Path


@runtime_checkable
class ProfileTrialExecutor(Protocol):
    async def prepare_trial(self, invocation: ProfileInvocation) -> None: ...

    async def execute(self, invocation: ProfileInvocation) -> ProfileTrialObservation: ...


@runtime_checkable
class PlanProfileSearchEngine(Protocol):
    async def benchmark_plan_search(
        self,
        request: SearchRequest,
        *,
        request_id: str,
        plan_id: str,
    ) -> SearchResponse: ...


class SearchEngineProfileTrialExecutor:
    """Adapt one final engine response into profiler evidence without raw query leakage."""

    def __init__(self, engine: SearchEngine) -> None:
        self._engine = engine

    async def prepare_trial(self, invocation: ProfileInvocation) -> None:
        del invocation

    async def execute(self, invocation: ProfileInvocation) -> ProfileTrialObservation:
        request = _request_for_invocation(invocation)
        request_id = f"profile-{invocation.trial_id}"
        started_ns = time.perf_counter_ns()
        try:
            if not isinstance(self._engine, PlanProfileSearchEngine):
                raise RAGPlanError(
                    ErrorCode.MODE_UNAVAILABLE,
                    "engine does not expose final-path plan profiling",
                )
            response = await self._engine.benchmark_plan_search(
                request,
                request_id=request_id,
                plan_id=invocation.plan_features.plan_id,
            )
            elapsed_ms = (time.perf_counter_ns() - started_ns) / 1_000_000
            return _observation_from_response(response, total_latency_ms=elapsed_ms)
        except asyncio.CancelledError:
            raise
        except RAGPlanError as exc:
            elapsed_ms = (time.perf_counter_ns() - started_ns) / 1_000_000
            timeout = exc.code is ErrorCode.DEADLINE_EXCEEDED
            return ProfileTrialObservation(
                status=TrialStatus.TIMEOUT if timeout else TrialStatus.ERROR,
                total_latency_ms=elapsed_ms,
                timeout=timeout,
                budget_violated=elapsed_ms > invocation.latency_budget_ms,
                error_code=exc.code,
            )
        except Exception:
            elapsed_ms = (time.perf_counter_ns() - started_ns) / 1_000_000
            return ProfileTrialObservation(
                status=TrialStatus.ERROR,
                total_latency_ms=elapsed_ms,
                budget_violated=elapsed_ms > invocation.latency_budget_ms,
                error_code=ErrorCode.INTERNAL_ERROR,
            )


ProfileEngineFactory = Callable[[], Awaitable[SearchEngine]]


class ManagedSearchEngineProfileTrialExecutor:
    """Own a single final engine for the deterministic cold/warm/measured sweep."""

    def __init__(self, engine_factory: ProfileEngineFactory) -> None:
        self._engine_factory = engine_factory
        self._engine: SearchEngine | None = None

    async def prepare_trial(self, invocation: ProfileInvocation) -> None:
        del invocation
        if self._engine is None:
            self._engine = await self._engine_factory()

    async def execute(self, invocation: ProfileInvocation) -> ProfileTrialObservation:
        if self._engine is None:
            raise RuntimeError("profile engine was not prepared")
        return await SearchEngineProfileTrialExecutor(self._engine).execute(invocation)

    async def close(self) -> None:
        if self._engine is None:
            return
        engine, self._engine = self._engine, None
        await engine.close()


class ProfileRepository:
    """Append-only profile evidence under ``profile_<run_id>`` with immutable identities."""

    def __init__(
        self,
        output_root: Path,
        *,
        protocol: ProfileProtocolConfig,
        environment: EnvironmentManifest,
        manifest: ProfileRunManifest,
    ) -> None:
        _require_identity_bundle(protocol, environment, manifest)
        self.run_dir = output_root / f"profile_{manifest.run_id}"
        self.raw_path = self.run_dir / "raw_trials.jsonl"
        self.protocol_path = self.run_dir / "profile_protocol.json"
        self.environment_path = self.run_dir / "environment.json"
        self.manifest_path = self.run_dir / "run_manifest.json"
        self.lock_path = self.run_dir / ".run.lock"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        with self.exclusive_lock():
            self._initialize(self.protocol_path, protocol)
            self._initialize(self.environment_path, environment)
            self._initialize(self.manifest_path, manifest)

    @staticmethod
    def _initialize(path: Path, model: FrozenModel) -> None:
        if path.exists():
            try:
                observed = type(model).model_validate_json(path.read_text(encoding="utf-8"))
            except Exception as exc:
                raise RawEvidenceError(
                    f"existing profile evidence is invalid: {path.name}"
                ) from exc
            if observed != model:
                raise RawEvidenceError(f"profile run already exists with different {path.name}")
            return
        write_json_model(path, model)

    def load_records(self) -> tuple[ProfileTrialRecord, ...]:
        if not self.raw_path.exists():
            return ()
        records: list[ProfileTrialRecord] = []
        seen: set[str] = set()
        with self.raw_path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.endswith("\n"):
                    raise RawEvidenceError("profile evidence contains a truncated final row")
                if not line.strip():
                    raise RawEvidenceError("profile evidence contains an empty row")
                try:
                    record = ProfileTrialRecord.model_validate_json(line)
                except ValueError as exc:
                    raise RawEvidenceError(
                        f"profile evidence row {line_number} is invalid"
                    ) from exc
                if record.trial_id in seen:
                    raise RawEvidenceError("profile evidence contains a duplicate trial ID")
                seen.add(record.trial_id)
                records.append(record)
        return tuple(records)

    def append(self, record: ProfileTrialRecord) -> None:
        payload = canonical_json_bytes(record.model_dump(mode="json")) + b"\n"
        descriptor = os.open(self.raw_path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
        try:
            offset = 0
            while offset < len(payload):
                written = os.write(descriptor, payload[offset:])
                if written <= 0:
                    raise OSError("short append while writing profile evidence")
                offset += written
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    @contextmanager
    def exclusive_lock(self) -> Iterator[None]:
        descriptor = os.open(self.lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RawEvidenceError("profile run is already being written") from exc
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)


class PlanProfiler:
    """Execute the complete query-plan-budget trial schedule with resumability."""

    def __init__(
        self,
        *,
        protocol: ProfileProtocolConfig,
        run_manifest: ProfileRunManifest,
        cases: Sequence[BenchmarkCase],
        executor: ProfileTrialExecutor,
        repository: ProfileRepository,
        progress: Callable[[int, int], None] | None = None,
    ) -> None:
        if protocol.concurrency != 1:
            raise ValueError("Stage 10 primary profiler supports concurrency 1 only")
        self._protocol = protocol
        self._manifest = run_manifest
        self._cases = tuple(sorted(cases, key=lambda item: item.query.query_id))
        self._executor = executor
        self._repository = repository
        self._progress = progress
        self._validate_cases()
        _require_identity_bundle_from_manifest(protocol, run_manifest)

    async def run(self) -> ProfileRunnerSummary:
        with self._repository.exclusive_lock():
            return await self._run_locked()

    async def _run_locked(self) -> ProfileRunnerSummary:
        existing = self._repository.load_records()
        self._validate_existing(existing)
        completed_ids = {item.trial_id for item in existing}
        executed = 0
        for invocation, case in self._schedule():
            if invocation.trial_id in completed_ids:
                continue
            await self._executor.prepare_trial(invocation)
            started_ns = time.perf_counter_ns()
            try:
                observation = await self._executor.execute(invocation)
            except asyncio.CancelledError:
                raise
            except Exception:
                elapsed_ms = (time.perf_counter_ns() - started_ns) / 1_000_000
                observation = ProfileTrialObservation(
                    status=TrialStatus.ERROR,
                    total_latency_ms=elapsed_ms,
                    budget_violated=elapsed_ms > invocation.latency_budget_ms,
                    error_code=ErrorCode.INTERNAL_ERROR,
                )
            record = self._record(invocation, case, observation)
            self._repository.append(record)
            completed_ids.add(record.trial_id)
            executed += 1
            completed = len(existing) + executed
            if self._progress is not None and (
                completed % 100 == 0 or completed == self._manifest.expected_raw_row_count
            ):
                self._progress(completed, self._manifest.expected_raw_row_count)
        total = len(self._repository.load_records())
        return ProfileRunnerSummary(
            run_id=self._manifest.run_id,
            expected_rows=self._manifest.expected_raw_row_count,
            preexisting_rows=len(existing),
            executed_rows=executed,
            total_rows=total,
            complete=total == self._manifest.expected_raw_row_count,
            raw_path=self._repository.raw_path,
        )

    def _schedule(self) -> tuple[tuple[ProfileInvocation, BenchmarkCase], ...]:
        pairs = tuple(
            (case, features) for case in self._cases for features in self._protocol.plan_features
        )
        blocks: list[tuple[TrialPhase, int]] = [(TrialPhase.COLD, 0)]
        blocks.extend((TrialPhase.WARMUP, index) for index in range(self._protocol.warmup_runs))
        blocks.extend((TrialPhase.MEASURED, index) for index in range(self._protocol.measured_runs))
        scheduled: list[tuple[ProfileInvocation, BenchmarkCase]] = []
        for phase, repetition in blocks:
            for budget in self._protocol.latency_budgets_ms:
                shuffled = list(pairs)
                seed_payload = (
                    f"{self._protocol.random_seed}:{phase.value}:{repetition}:{budget}:profile"
                ).encode()
                seed = int.from_bytes(hashlib.sha256(seed_payload).digest()[:8], "big")
                random.Random(seed).shuffle(shuffled)
                for case, features in shuffled:
                    trial_id = profile_trial_identity(
                        run_id=self._manifest.run_id,
                        query_id=case.query.query_id,
                        plan_id=features.plan_id,
                        latency_budget_ms=budget,
                        phase=phase,
                        repetition=repetition,
                    )
                    scheduled.append(
                        (
                            ProfileInvocation(
                                run_id=self._manifest.run_id,
                                trial_id=trial_id,
                                query_id=case.query.query_id,
                                query=case.query.question,
                                plan_features=features,
                                latency_budget_ms=budget,
                                final_top_k=self._protocol.final_top_k,
                                phase=phase,
                                repetition=repetition,
                            ),
                            case,
                        )
                    )
        return tuple(scheduled)

    def _record(
        self,
        invocation: ProfileInvocation,
        case: BenchmarkCase,
        observation: ProfileTrialObservation,
    ) -> ProfileTrialRecord:
        ranked = observation.ranked_chunk_ids
        if observation.status in {TrialStatus.TIMEOUT, TrialStatus.ERROR} or not ranked:
            metrics = (0.0, 0.0, 0.0, 0.0)
        else:
            metrics = (
                recall_at_5(ranked, case.relevance),
                recall_at_10(ranked, case.relevance),
                mrr_at_10(ranked, case.relevance),
                ndcg_at_10(ranked, case.relevance),
            )
        quality_valid, latency_valid, exclusion = self._label_validity(
            invocation,
            observation,
        )
        return ProfileTrialRecord(
            run_id=self._manifest.run_id,
            trial_id=invocation.trial_id,
            query_id=case.query.query_id,
            split=case.split,
            source_dataset=case.query.source_dataset,
            query_tags=case.query.query_tags,
            plan_id=invocation.plan_features.plan_id,
            plan_features=invocation.plan_features,
            query_features=observation.query_features,
            latency_budget_ms=invocation.latency_budget_ms,
            trial_phase=invocation.phase,
            repetition=invocation.repetition,
            status=observation.status,
            quality_scope={
                TrialStatus.COMPLETE: QualityScope.FULL,
                TrialStatus.PARTIAL: QualityScope.PARTIAL,
                TrialStatus.TIMEOUT: QualityScope.NONE,
                TrialStatus.ERROR: QualityScope.NONE,
            }[observation.status],
            recall_at_5=metrics[0],
            recall_at_10=metrics[1],
            mrr_at_10=metrics[2],
            ndcg_at_10=metrics[3],
            analyzer_latency_ms=observation.analyzer_latency_ms,
            planner_latency_ms=observation.planner_latency_ms,
            embedding_latency_ms=observation.embedding_latency_ms,
            execution_latency_ms=observation.execution_latency_ms,
            vector_latency_ms=observation.vector_latency_ms,
            graph_latency_ms=observation.graph_latency_ms,
            fusion_latency_ms=observation.fusion_latency_ms,
            rerank_latency_ms=observation.rerank_latency_ms,
            total_latency_ms=observation.total_latency_ms,
            branch_results=observation.branch_results,
            timeout=observation.timeout,
            fallback=observation.fallback,
            error=observation.status is TrialStatus.ERROR,
            budget_violated=observation.budget_violated,
            error_code=observation.error_code,
            result_count=len(ranked),
            quality_label_valid=quality_valid,
            execution_latency_label_valid=latency_valid,
            invalid_exclusion_reason=exclusion,
            final_engine_path=FINAL_ENGINE_PATH,
            scheduler_trace_present=observation.scheduler_trace_present,
            scheduler_runtime_semantics_version=(observation.scheduler_runtime_semantics_version),
            trace_config_version=observation.trace_config_version,
            trace_model_version=observation.trace_model_version,
            profile_protocol_sha256=self._manifest.profile_protocol_sha256,
            environment_manifest_sha256=self._manifest.environment_manifest_sha256,
            benchmark_manifest_sha256=self._manifest.benchmark_manifest_sha256,
            split_hash=self._manifest.split_hash,
            qrels_sha256=self._manifest.qrels_sha256,
            corpus_version=self._manifest.corpus_version,
            corpus_chunk_ids_sha256=self._manifest.corpus_chunk_ids_sha256,
            plan_catalog_sha256=self._manifest.plan_catalog_sha256,
            query_feature_config_sha256=self._manifest.query_feature_config_sha256,
            runtime_semantics_version=self._manifest.runtime_semantics_version,
        )

    def _label_validity(
        self,
        invocation: ProfileInvocation,
        observation: ProfileTrialObservation,
    ) -> tuple[bool, bool, str | None]:
        if observation.status is TrialStatus.TIMEOUT:
            return True, False, "execution_trace_unavailable:timeout"
        if observation.status is TrialStatus.ERROR:
            code = observation.error_code.value if observation.error_code is not None else "unknown"
            return True, False, f"execution_trace_unavailable:error:{code}"
        issues: list[str] = []
        quality_valid = True
        latency_valid = True
        if observation.selected_plan_id != invocation.plan_features.plan_id:
            issues.append("selected_plan_mismatch")
            quality_valid = False
            latency_valid = False
        if observation.query_features is None:
            issues.append("query_features_missing")
            quality_valid = False
            latency_valid = False
        if observation.trace_config_version != self._manifest.plan_catalog_sha256:
            issues.append("plan_catalog_trace_mismatch")
            quality_valid = False
            latency_valid = False
        expected_model = (
            f"{self._manifest.embedding_model_revision}:{self._manifest.extractor_version}"
        )
        if observation.trace_model_version != expected_model:
            issues.append("runtime_model_trace_mismatch")
            quality_valid = False
            latency_valid = False
        if not observation.scheduler_trace_present:
            issues.append("scheduler_trace_missing")
            latency_valid = False
        if (
            observation.scheduler_runtime_semantics_version
            != self._manifest.runtime_semantics_version
        ):
            issues.append("runtime_semantics_trace_mismatch")
            latency_valid = False
        if observation.execution_latency_ms is None:
            issues.append("execution_latency_missing")
            latency_valid = False
        return quality_valid, latency_valid, ";".join(issues) if issues else None

    def _validate_cases(self) -> None:
        ids = tuple(item.query.query_id for item in self._cases)
        if len(ids) != len(set(ids)):
            raise ValueError("profile cases contain duplicate query IDs")
        if any(case.split is SplitName.TEST for case in self._cases):
            raise ValueError("Stage 10 profiler refuses the held-out test split")
        if any(case.split not in self._protocol.primary_splits for case in self._cases):
            raise ValueError("profile case split is outside the frozen protocol")
        if any(not any(grade >= 1 for grade in case.relevance.values()) for case in self._cases):
            raise ValueError("every profile query requires at least one relevant chunk")
        if len(self._cases) != self._manifest.query_count:
            raise ValueError("profile case count does not match the run manifest")
        if canonical_sha256(list(sorted(ids))) != self._manifest.query_ids_sha256:
            raise ValueError("profile query IDs do not match the run manifest")
        identities = tuple(
            BenchmarkQueryIdentity(
                query_id=case.query.query_id,
                split=case.split,
                source_dataset=case.query.source_dataset,
                query_tags=case.query.query_tags,
            )
            for case in self._cases
        )
        if benchmark_query_identities_sha256(identities) != self._manifest.query_identities_sha256:
            raise ValueError("profile query identities do not match the run manifest")

    def _validate_existing(self, records: Sequence[ProfileTrialRecord]) -> None:
        expected = {
            invocation.trial_id: (invocation, case) for invocation, case in self._schedule()
        }
        for record in records:
            scheduled = expected.get(record.trial_id)
            if scheduled is None:
                raise RawEvidenceError("profile row is outside the deterministic schedule")
            invocation, case = scheduled
            observed = (
                record.run_id,
                record.query_id,
                record.split,
                record.source_dataset,
                record.query_tags,
                record.plan_id,
                record.plan_features,
                record.latency_budget_ms,
                record.trial_phase,
                record.repetition,
            )
            wanted = (
                self._manifest.run_id,
                case.query.query_id,
                case.split,
                case.query.source_dataset,
                case.query.query_tags,
                invocation.plan_features.plan_id,
                invocation.plan_features,
                invocation.latency_budget_ms,
                invocation.phase,
                invocation.repetition,
            )
            if observed != wanted:
                raise RawEvidenceError("profile row fields do not match its trial identity")
            _require_record_versions(record, self._manifest)
        if len(records) > self._manifest.expected_raw_row_count:
            raise RawEvidenceError("profile row count exceeds the immutable schedule")


def build_training_matrix(
    records: Sequence[ProfileTrialRecord],
    *,
    manifest: ProfileRunManifest,
    fallback_query_features: Mapping[str, QueryFeatures] | None = None,
) -> tuple[TrainingMatrixRow, ...]:
    """Validate the exact raw matrix and derive one measured query-plan-budget row."""

    rows = tuple(records)
    _validate_complete_raw_matrix(rows, manifest=manifest)
    query_features = _consistent_query_features(
        rows,
        fallback_query_features=fallback_query_features,
    )
    grouped: dict[tuple[str, str, int], list[ProfileTrialRecord]] = defaultdict(list)
    for record in rows:
        if record.trial_phase is TrialPhase.MEASURED:
            grouped[(record.query_id, record.plan_id, record.latency_budget_ms)].append(record)
    matrix: list[TrainingMatrixRow] = []
    for key in sorted(grouped, key=lambda item: (item[0], int(item[1][1:]), item[2])):
        subset = tuple(sorted(grouped[key], key=lambda item: item.repetition))
        first = subset[0]
        quality_count = sum(item.quality_label_valid for item in subset)
        latency_values = tuple(
            item.execution_latency_ms
            for item in subset
            if item.execution_latency_label_valid and item.execution_latency_ms is not None
        )
        reasons = tuple(
            sorted(
                {
                    item.invalid_exclusion_reason
                    for item in subset
                    if item.invalid_exclusion_reason is not None
                }
            )
        )
        quality_valid = quality_count == manifest.measured_runs
        latency_valid = len(latency_values) == manifest.measured_runs
        usable = quality_valid and latency_valid
        if not usable and not reasons:
            reasons = ("incomplete_training_labels",)
        full = tuple(
            item.recall_at_10 for item in subset if item.quality_scope is QualityScope.FULL
        )
        partial = tuple(
            item.recall_at_10 for item in subset if item.quality_scope is QualityScope.PARTIAL
        )
        matrix.append(
            TrainingMatrixRow(
                run_id=manifest.run_id,
                query_id=first.query_id,
                split=first.split,
                source_dataset=first.source_dataset,
                query_tags=first.query_tags,
                plan_id=first.plan_id,
                query_features=query_features[first.query_id],
                plan_features=first.plan_features,
                latency_budget_ms=first.latency_budget_ms,
                quality_label_trial_count=quality_count,
                execution_latency_trial_count=len(latency_values),
                complete_trial_count=sum(item.status is TrialStatus.COMPLETE for item in subset),
                partial_trial_count=sum(item.status is TrialStatus.PARTIAL for item in subset),
                timeout_trial_count=sum(item.status is TrialStatus.TIMEOUT for item in subset),
                error_trial_count=sum(item.status is TrialStatus.ERROR for item in subset),
                fallback_trial_count=sum(item.fallback for item in subset),
                recall_at_5=_mean(tuple(item.recall_at_5 for item in subset)),
                recall_at_10=_mean(tuple(item.recall_at_10 for item in subset)),
                mrr_at_10=_mean(tuple(item.mrr_at_10 for item in subset)),
                ndcg_at_10=_mean(tuple(item.ndcg_at_10 for item in subset)),
                full_result_recall_at_10=_mean(full) if full else None,
                partial_result_recall_at_10=_mean(partial) if partial else None,
                p95_execution_latency_ms=(
                    percentile_type7(latency_values, 0.95) if latency_valid else None
                ),
                fallback_rate=sum(item.fallback for item in subset) / len(subset),
                budget_violation_rate=(sum(item.budget_violated for item in subset) / len(subset)),
                quality_label_valid=quality_valid,
                execution_latency_label_valid=latency_valid,
                usable_for_model_training=usable,
                invalid_exclusion_reasons=reasons,
                source_trial_ids_sha256=canonical_sha256(
                    [item.trial_id for item in sorted(subset, key=lambda item: item.trial_id)]
                ),
                profile_protocol_sha256=manifest.profile_protocol_sha256,
                environment_manifest_sha256=manifest.environment_manifest_sha256,
                benchmark_manifest_sha256=manifest.benchmark_manifest_sha256,
                split_hash=manifest.split_hash,
                qrels_sha256=manifest.qrels_sha256,
                corpus_version=manifest.corpus_version,
                corpus_chunk_ids_sha256=manifest.corpus_chunk_ids_sha256,
                plan_catalog_sha256=manifest.plan_catalog_sha256,
                query_feature_config_sha256=manifest.query_feature_config_sha256,
                runtime_semantics_version=manifest.runtime_semantics_version,
            )
        )
    expected = manifest.query_count * len(manifest.plan_ids) * len(manifest.latency_budgets_ms)
    if len(matrix) != expected:
        raise RawEvidenceError("training matrix is missing query-plan-budget rows")
    return tuple(matrix)


def write_training_matrix_artifacts(
    run_dir: Path,
    *,
    records: Sequence[ProfileTrialRecord],
    matrix: Sequence[TrainingMatrixRow],
    recovered_query_feature_count: int = 0,
) -> dict[str, str]:
    """Write deterministic matrix JSONL/CSV plus logical and file checksums."""

    ordered = tuple(
        sorted(
            matrix,
            key=lambda item: (item.query_id, int(item.plan_id[1:]), item.latency_budget_ms),
        )
    )
    jsonl = (
        b"\n".join(canonical_json_bytes(item.model_dump(mode="json")) for item in ordered) + b"\n"
    )
    csv_payload = _matrix_csv(ordered)
    write_bytes(run_dir / "training_matrix.jsonl", jsonl)
    write_bytes(run_dir / "training_matrix.csv", csv_payload)
    checksums = {
        "environment.json": _file_sha256(run_dir / "environment.json"),
        "profile_protocol.json": _file_sha256(run_dir / "profile_protocol.json"),
        "raw_trials.jsonl": _file_sha256(run_dir / "raw_trials.jsonl"),
        "run_manifest.json": _file_sha256(run_dir / "run_manifest.json"),
        "training_matrix.csv": hashlib.sha256(csv_payload).hexdigest(),
        "training_matrix.jsonl": hashlib.sha256(jsonl).hexdigest(),
        "raw_logical_sha256": profile_records_sha256(tuple(records)),
        "recovered_query_feature_count": str(recovered_query_feature_count),
    }
    recovery_path = run_dir / "query_feature_recovery.json"
    if recovery_path.is_file():
        checksums[recovery_path.name] = _file_sha256(recovery_path)
    write_json(run_dir / "checksums.json", checksums)
    return checksums


def _validate_complete_raw_matrix(
    records: tuple[ProfileTrialRecord, ...],
    *,
    manifest: ProfileRunManifest,
) -> None:
    if len(records) != manifest.expected_raw_row_count:
        raise RawEvidenceError("profile raw matrix is incomplete")
    if len({item.trial_id for item in records}) != len(records):
        raise RawEvidenceError("profile raw matrix contains duplicate trials")
    query_ids = tuple(sorted({item.query_id for item in records}))
    if len(query_ids) != manifest.query_count:
        raise RawEvidenceError("profile raw matrix query count mismatch")
    if canonical_sha256(list(query_ids)) != manifest.query_ids_sha256:
        raise RawEvidenceError("profile raw matrix query identity mismatch")
    identities_by_query = {
        item.query_id: BenchmarkQueryIdentity(
            query_id=item.query_id,
            split=item.split,
            source_dataset=item.source_dataset,
            query_tags=item.query_tags,
        )
        for item in records
    }
    if benchmark_query_identities_sha256(tuple(identities_by_query.values())) != (
        manifest.query_identities_sha256
    ):
        raise RawEvidenceError("profile raw query metadata mismatch")
    expected_repetitions = {
        TrialPhase.COLD: {0},
        TrialPhase.WARMUP: set(range(manifest.warmup_runs)),
        TrialPhase.MEASURED: set(range(manifest.measured_runs)),
    }
    grouped: dict[tuple[str, str, int, TrialPhase], set[int]] = defaultdict(set)
    plan_features: dict[str, ProfilePlanFeatures] = {}
    for record in records:
        _require_record_versions(record, manifest)
        expected_id = profile_trial_identity(
            run_id=record.run_id,
            query_id=record.query_id,
            plan_id=record.plan_id,
            latency_budget_ms=record.latency_budget_ms,
            phase=record.trial_phase,
            repetition=record.repetition,
        )
        if record.trial_id != expected_id:
            raise RawEvidenceError("profile row has a forged trial identity")
        if record.plan_id not in manifest.plan_ids:
            raise RawEvidenceError("profile row plan is outside the manifest")
        existing = plan_features.setdefault(record.plan_id, record.plan_features)
        if existing != record.plan_features:
            raise RawEvidenceError("profile plan feature columns changed within a run")
        key = (
            record.query_id,
            record.plan_id,
            record.latency_budget_ms,
            record.trial_phase,
        )
        if record.repetition in grouped[key]:
            raise RawEvidenceError("profile matrix repeats a phase repetition")
        grouped[key].add(record.repetition)
    expected_group_count = (
        manifest.query_count
        * len(manifest.plan_ids)
        * len(manifest.latency_budgets_ms)
        * len(expected_repetitions)
    )
    if len(grouped) != expected_group_count:
        raise RawEvidenceError("profile matrix is missing a query-plan phase")
    if any(repetitions != expected_repetitions[key[3]] for key, repetitions in grouped.items()):
        raise RawEvidenceError("profile matrix phase trial count is incomplete")
    payload = [plan_features[plan_id].model_dump(mode="json") for plan_id in manifest.plan_ids]
    if hashlib.sha256(canonical_json_bytes(payload)).hexdigest() != manifest.plan_features_sha256:
        raise RawEvidenceError("profile plan features do not match the manifest")


def _consistent_query_features(
    records: Sequence[ProfileTrialRecord],
    *,
    fallback_query_features: Mapping[str, QueryFeatures] | None = None,
) -> dict[str, QueryFeatures]:
    features: dict[str, QueryFeatures] = {}
    for record in records:
        if record.query_features is None:
            continue
        existing = features.setdefault(record.query_id, record.query_features)
        if existing != record.query_features:
            raise RawEvidenceError("query features changed across plan trials")
    query_ids = {item.query_id for item in records}
    fallback = {} if fallback_query_features is None else dict(fallback_query_features)
    if not set(fallback) <= query_ids:
        raise RawEvidenceError("fallback query features contain IDs outside the raw matrix")
    for query_id, recovered in fallback.items():
        existing = features.setdefault(query_id, recovered)
        if existing != recovered:
            raise RawEvidenceError("recovered query features differ from trace evidence")
    if set(features) != query_ids:
        raise RawEvidenceError("one or more queries have no profiler feature evidence")
    return features


def _require_identity_bundle(
    protocol: ProfileProtocolConfig,
    environment: EnvironmentManifest,
    manifest: ProfileRunManifest,
) -> None:
    _require_identity_bundle_from_manifest(protocol, manifest)
    if manifest.environment_manifest_sha256 != environment.sha256:
        raise RawEvidenceError("profile manifest does not identify its environment")
    if environment.concurrency != protocol.concurrency:
        raise RawEvidenceError("profile environment concurrency differs from protocol")


def _require_identity_bundle_from_manifest(
    protocol: ProfileProtocolConfig,
    manifest: ProfileRunManifest,
) -> None:
    observed = (
        manifest.profile_protocol_sha256,
        manifest.baseline_protocol_sha256,
        manifest.benchmark_manifest_sha256,
        manifest.split_hash,
        manifest.qrels_sha256,
        manifest.corpus_version,
        manifest.corpus_chunk_count,
        manifest.corpus_chunk_ids_sha256,
        manifest.embedding_model_revision,
        manifest.extractor_version,
        manifest.plan_catalog_sha256,
        manifest.query_feature_config_sha256,
        manifest.runtime_semantics_version,
        manifest.cold_runs,
        manifest.warmup_runs,
        manifest.measured_runs,
        manifest.concurrency,
        manifest.plan_ids,
        manifest.latency_budgets_ms,
    )
    expected = (
        protocol.sha256,
        protocol.baseline_protocol_sha256,
        protocol.benchmark_manifest_sha256,
        protocol.split_hash,
        protocol.qrels_sha256,
        protocol.corpus_version,
        protocol.corpus_chunk_count,
        protocol.corpus_chunk_ids_sha256,
        protocol.embedding_model_revision,
        protocol.extractor_version,
        protocol.plan_catalog_sha256,
        protocol.query_feature_config_sha256,
        protocol.runtime_semantics_version,
        protocol.cold_runs,
        protocol.warmup_runs,
        protocol.measured_runs,
        protocol.concurrency,
        protocol.plan_ids,
        protocol.latency_budgets_ms,
    )
    if observed != expected:
        raise RawEvidenceError("profile manifest version/hash bundle differs from protocol")


def _require_record_versions(
    record: ProfileTrialRecord,
    manifest: ProfileRunManifest,
) -> None:
    observed = (
        record.run_id,
        record.profile_protocol_sha256,
        record.environment_manifest_sha256,
        record.benchmark_manifest_sha256,
        record.split_hash,
        record.qrels_sha256,
        record.corpus_version,
        record.corpus_chunk_ids_sha256,
        record.plan_catalog_sha256,
        record.query_feature_config_sha256,
        record.runtime_semantics_version,
    )
    expected = (
        manifest.run_id,
        manifest.profile_protocol_sha256,
        manifest.environment_manifest_sha256,
        manifest.benchmark_manifest_sha256,
        manifest.split_hash,
        manifest.qrels_sha256,
        manifest.corpus_version,
        manifest.corpus_chunk_ids_sha256,
        manifest.plan_catalog_sha256,
        manifest.query_feature_config_sha256,
        manifest.runtime_semantics_version,
    )
    if observed != expected:
        raise RawEvidenceError("profile raw row version/hash mismatch")


def _request_for_invocation(invocation: ProfileInvocation) -> SearchRequest:
    mode = invocation.plan_features.planner_mode
    return SearchRequest(
        query=invocation.query,
        top_k=invocation.final_top_k,
        latency_budget_ms=invocation.latency_budget_ms,
        planner=mode,
        plan_id=(invocation.plan_features.plan_id if mode is PlannerMode.FIXED_HYBRID else None),
    )


def _observation_from_response(
    response: SearchResponse,
    *,
    total_latency_ms: float,
) -> ProfileTrialObservation:
    trace = response.trace
    scheduler = trace.scheduler_trace
    execution_latency_ms: float | None = None
    runtime_version: str | None = None
    if scheduler is not None:
        execution_start = next(
            (
                event.elapsed_ms
                for event in scheduler.state_events
                if event.state is RequestState.EXECUTING
            ),
            None,
        )
        terminal = scheduler.state_events[-1].elapsed_ms
        if execution_start is not None and terminal >= execution_start:
            execution_latency_ms = terminal - execution_start
        runtime_version = scheduler.runtime_semantics_version
    return ProfileTrialObservation(
        status=(
            TrialStatus.PARTIAL if response.status is SearchStatus.PARTIAL else TrialStatus.COMPLETE
        ),
        ranked_chunk_ids=tuple(item.canonical_chunk_id for item in response.results),
        selected_plan_id=response.planner_decision.selected_plan_id,
        query_features=trace.features,
        analyzer_latency_ms=trace.analyzer_latency_ms,
        planner_latency_ms=trace.planner_latency_ms,
        embedding_latency_ms=trace.embedding_latency_ms,
        execution_latency_ms=execution_latency_ms,
        vector_latency_ms=trace.vector_latency_ms,
        graph_latency_ms=trace.graph_latency_ms,
        fusion_latency_ms=trace.fusion_latency_ms,
        rerank_latency_ms=trace.rerank_latency_ms,
        total_latency_ms=total_latency_ms,
        branch_results=tuple(
            BranchTrialRecord(
                branch=item.branch,
                status=item.status,
                latency_ms=item.latency_ms,
                error_code=item.error_code,
            )
            for item in trace.branch_results
        ),
        fallback=response.fallback,
        budget_violated=(trace.budget_violated or total_latency_ms > trace.latency_budget_ms),
        scheduler_trace_present=scheduler is not None,
        scheduler_runtime_semantics_version=runtime_version,
        trace_config_version=trace.config_version,
        trace_model_version=trace.model_version,
    )


def _matrix_csv(rows: Sequence[TrainingMatrixRow]) -> bytes:
    query_fields = tuple(QueryFeatures.model_fields)
    plan_fields = tuple(ProfilePlanFeatures.model_fields)
    nested = {"query_features", "plan_features", "query_tags", "invalid_exclusion_reasons"}
    fields = tuple(field for field in TrainingMatrixRow.model_fields if field not in nested)
    header = (
        *fields,
        "query_tags",
        *(f"query_{field}" for field in query_fields),
        *(f"plan_{field}" for field in plan_fields if field != "plan_id"),
        "invalid_exclusion_reasons",
    )
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=header, lineterminator="\n")
    writer.writeheader()
    for item in rows:
        raw = item.model_dump(mode="json")
        row = {field: raw[field] for field in fields}
        row["query_tags"] = "|".join(raw["query_tags"])
        for field in query_fields:
            row[f"query_{field}"] = raw["query_features"][field]
        for field in plan_fields:
            if field != "plan_id":
                row[f"plan_{field}"] = raw["plan_features"][field]
        row["invalid_exclusion_reasons"] = "|".join(raw["invalid_exclusion_reasons"])
        writer.writerow(row)
    return output.getvalue().encode("utf-8")


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("mean requires at least one value")
    return sum(values) / len(values)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "ManagedSearchEngineProfileTrialExecutor",
    "PlanProfiler",
    "PlanProfileSearchEngine",
    "ProfileInvocation",
    "ProfileRepository",
    "ProfileRunnerSummary",
    "ProfileTrialExecutor",
    "ProfileTrialObservation",
    "SearchEngineProfileTrialExecutor",
    "build_training_matrix",
    "write_training_matrix_artifacts",
]
