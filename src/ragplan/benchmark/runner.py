"""Resumable, single-concurrency Stage 9 benchmark execution."""

from __future__ import annotations

import asyncio
import fcntl
import hashlib
import os
import random
import time
from collections.abc import Awaitable, Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from ragplan.benchmark.artifacts import (
    load_benchmark_manifest,
    load_qrels,
    load_split_manifest,
    write_json_model,
    write_yaml_model,
)
from ragplan.benchmark.contracts import (
    BenchmarkQuery,
    SplitName,
    canonical_json_bytes,
    canonical_sha256,
)
from ragplan.benchmark.metrics import mrr_at_10, ndcg_at_10, recall_at_5, recall_at_10
from ragplan.benchmark.qrels import aggregate_relevance
from ragplan.benchmark.records import (
    BenchmarkMethod,
    BenchmarkProtocolConfig,
    BenchmarkQueryIdentity,
    BenchmarkRunManifest,
    BranchTrialRecord,
    EnvironmentManifest,
    MethodDefinition,
    RawTrialRecord,
    TrialObservation,
    TrialPhase,
    TrialStatus,
    benchmark_query_identities_sha256,
    parse_raw_record,
    trial_identity,
)
from ragplan.core.engine import SearchEngine, benchmark_graph_depth_2_config_version
from ragplan.core.errors import ErrorCode, RAGPlanError
from ragplan.core.models import SearchRequest, SearchResponse, SearchStatus


class DuplicateRunError(ValueError):
    """Raised when a run ID is reused with different immutable evidence."""


class RawEvidenceError(ValueError):
    """Raised when existing raw evidence is malformed, duplicated, or incompatible."""


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    query: BenchmarkQuery
    split: SplitName
    relevance: Mapping[str, int]


@dataclass(frozen=True, slots=True)
class BenchmarkInvocation:
    run_id: str
    trial_id: str
    query_id: str
    query: str
    method: MethodDefinition
    latency_budget_ms: int
    final_top_k: int
    phase: TrialPhase
    repetition: int


@dataclass(frozen=True, slots=True)
class RunnerSummary:
    run_id: str
    expected_rows: int
    preexisting_rows: int
    executed_rows: int
    total_rows: int
    complete: bool
    raw_path: Path


@runtime_checkable
class TrialExecutor(Protocol):
    async def prepare_trial(self, invocation: BenchmarkInvocation) -> None: ...

    async def execute(self, invocation: BenchmarkInvocation) -> TrialObservation: ...


@runtime_checkable
class BenchmarkDepthSearchEngine(Protocol):
    async def benchmark_graph_search(
        self,
        request: SearchRequest,
        *,
        request_id: str,
        graph_depth: int,
    ) -> SearchResponse: ...


class SearchEngineTrialExecutor:
    """Convert one already-owned engine's responses into raw-trial observations."""

    def __init__(self, engine: SearchEngine) -> None:
        self._engine = engine

    async def prepare_trial(self, invocation: BenchmarkInvocation) -> None:
        # The managed executor below owns normative cold resets. This adapter is useful for
        # non-cold probes and tests where engine lifecycle is controlled by the caller.
        del invocation

    async def execute(self, invocation: BenchmarkInvocation) -> TrialObservation:
        request = _request_for_invocation(invocation)
        request_id = f"bench-{invocation.trial_id}"
        started_ns = time.perf_counter_ns()
        try:
            if invocation.method.graph_depth is not None:
                if not isinstance(self._engine, BenchmarkDepthSearchEngine):
                    raise RAGPlanError(
                        ErrorCode.MODE_UNAVAILABLE,
                        "engine does not expose benchmark graph-depth execution",
                    )
                response = await self._engine.benchmark_graph_search(
                    request,
                    request_id=request_id,
                    graph_depth=invocation.method.graph_depth,
                )
            else:
                response = await self._engine.search(request, request_id=request_id)
            elapsed_ms = (time.perf_counter_ns() - started_ns) / 1_000_000
            return _observation_from_response(response, total_latency_ms=elapsed_ms)
        except asyncio.CancelledError:
            raise
        except RAGPlanError as exc:
            error_response = exc.response(request_id)
            elapsed_ms = (time.perf_counter_ns() - started_ns) / 1_000_000
            timeout = exc.code is ErrorCode.DEADLINE_EXCEEDED
            return TrialObservation(
                status=TrialStatus.TIMEOUT if timeout else TrialStatus.ERROR,
                total_latency_ms=elapsed_ms,
                timeout=timeout,
                budget_violated=elapsed_ms > invocation.latency_budget_ms,
                error_code=error_response.code,
            )
        except Exception:
            error_response = RAGPlanError(
                ErrorCode.INTERNAL_ERROR,
                "benchmark trial failed internally",
                retryable=False,
            ).response(request_id)
            elapsed_ms = (time.perf_counter_ns() - started_ns) / 1_000_000
            return TrialObservation(
                status=TrialStatus.ERROR,
                total_latency_ms=elapsed_ms,
                budget_violated=elapsed_ms > invocation.latency_budget_ms,
                error_code=error_response.code,
            )


EngineFactory = Callable[[], Awaitable[SearchEngine]]


class ManagedSearchEngineTrialExecutor:
    """Own one fresh engine for the cold sweep and all later warm/measured sweeps."""

    def __init__(self, engine_factory: EngineFactory) -> None:
        self._engine_factory = engine_factory
        self._engine: SearchEngine | None = None

    async def prepare_trial(self, invocation: BenchmarkInvocation) -> None:
        del invocation
        if self._engine is None:
            self._engine = await self._engine_factory()

    async def execute(self, invocation: BenchmarkInvocation) -> TrialObservation:
        if self._engine is None:
            raise RuntimeError("benchmark engine was not prepared")
        return await SearchEngineTrialExecutor(self._engine).execute(invocation)

    async def close(self) -> None:
        if self._engine is None:
            return
        engine, self._engine = self._engine, None
        await engine.close()


class RunRepository:
    """Append-only raw evidence plus immutable run/environment/config identities."""

    def __init__(
        self,
        output_root: Path,
        *,
        protocol: BenchmarkProtocolConfig,
        environment: EnvironmentManifest,
        manifest: BenchmarkRunManifest,
    ) -> None:
        self._validate_identity_bundle(protocol, environment, manifest)
        self.run_dir = output_root / manifest.run_id
        self.raw_path = self.run_dir / "raw_trials.jsonl"
        self.protocol_path = self.run_dir / "protocol.yaml"
        self.environment_path = self.run_dir / "environment.json"
        self.manifest_path = self.run_dir / "run_manifest.json"
        self.lock_path = self.run_dir / ".run.lock"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        with self.exclusive_lock():
            self._initialize_model(self.manifest_path, manifest)
            self._initialize_model(self.environment_path, environment)
            self._initialize_yaml(self.protocol_path, protocol)

    @staticmethod
    def _validate_identity_bundle(
        protocol: BenchmarkProtocolConfig,
        environment: EnvironmentManifest,
        manifest: BenchmarkRunManifest,
    ) -> None:
        if manifest.protocol_config_sha256 != protocol.sha256:
            raise DuplicateRunError("run manifest does not identify its protocol")
        if manifest.environment_manifest_sha256 != environment.sha256:
            raise DuplicateRunError("run manifest does not identify its environment")
        observed = (
            manifest.benchmark_manifest_sha256,
            manifest.split_hash,
            manifest.qrels_sha256,
            manifest.corpus_version,
            manifest.corpus_chunk_count,
            manifest.corpus_chunk_ids_sha256,
            manifest.embedding_model_revision,
            manifest.extractor_version,
            manifest.plan_catalog_sha256,
            manifest.planner_config_sha256,
            manifest.query_feature_config_sha256,
            manifest.graph_tier_policy_sha256,
            manifest.rule_runtime_config_version,
            manifest.stage2_artifact_set_sha256,
            manifest.runtime_semantics_version,
            manifest.random_seed,
            manifest.cold_runs,
            manifest.warmup_runs,
            manifest.measured_runs,
            manifest.concurrency,
            manifest.method_count,
            manifest.latency_budgets_ms,
        )
        expected = (
            protocol.benchmark_manifest_sha256,
            protocol.split_hash,
            protocol.qrels_sha256,
            protocol.corpus_version,
            protocol.corpus_chunk_count,
            protocol.corpus_chunk_ids_sha256,
            protocol.embedding_model_revision,
            protocol.extractor_version,
            protocol.plan_catalog_sha256,
            protocol.planner_config_sha256,
            protocol.query_feature_config_sha256,
            protocol.graph_tier_policy_sha256,
            protocol.rule_runtime_config_version,
            protocol.stage2_artifact_set_sha256,
            protocol.runtime_semantics_version,
            protocol.random_seed,
            protocol.cold_runs,
            protocol.warmup_runs,
            protocol.measured_runs,
            protocol.concurrency,
            len(protocol.methods),
            protocol.latency_budgets_ms,
        )
        if observed != expected:
            raise DuplicateRunError("run manifest version bundle differs from its protocol")

    @staticmethod
    def _initialize_model(path: Path, model: BenchmarkRunManifest | EnvironmentManifest) -> None:
        if path.exists():
            expected_type = type(model)
            try:
                observed = expected_type.model_validate_json(path.read_text(encoding="utf-8"))
            except Exception as exc:
                raise DuplicateRunError(f"existing run evidence is invalid: {path.name}") from exc
            if observed != model:
                raise DuplicateRunError(f"run ID already exists with different {path.name}")
            return
        write_json_model(path, model)

    @staticmethod
    def _initialize_yaml(path: Path, protocol: BenchmarkProtocolConfig) -> None:
        if path.exists():
            from ragplan.benchmark.config import load_benchmark_protocol

            if load_benchmark_protocol(path) != protocol:
                raise DuplicateRunError("run ID already exists with a different protocol")
            return
        write_yaml_model(path, protocol)

    def load_records(self) -> tuple[RawTrialRecord, ...]:
        if not self.raw_path.exists():
            return ()
        records: list[RawTrialRecord] = []
        seen_ids: set[str] = set()
        with self.raw_path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.endswith("\n"):
                    raise RawEvidenceError("raw evidence contains a truncated final row")
                if not line.strip():
                    raise RawEvidenceError("raw evidence contains an empty row")
                try:
                    record = parse_raw_record(line)
                except ValueError as exc:
                    raise RawEvidenceError(f"raw evidence row {line_number} is invalid") from exc
                if record.trial_id in seen_ids:
                    raise RawEvidenceError("raw evidence contains a duplicate trial ID")
                seen_ids.add(record.trial_id)
                records.append(record)
        return tuple(records)

    def append(self, record: RawTrialRecord) -> None:
        payload = canonical_json_bytes(record.model_dump(mode="json")) + b"\n"
        descriptor = os.open(
            self.raw_path,
            os.O_APPEND | os.O_CREAT | os.O_WRONLY,
            0o600,
        )
        try:
            offset = 0
            while offset < len(payload):
                written = os.write(descriptor, payload[offset:])
                if written <= 0:
                    raise OSError("short append while writing raw benchmark evidence")
                offset += written
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    @contextmanager
    def exclusive_lock(self) -> Iterator[None]:
        """Reject simultaneous writers for the same immutable run directory."""

        descriptor = os.open(self.lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise DuplicateRunError("benchmark run is already being written") from exc
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)


class BenchmarkRunner:
    """Execute deterministic trial blocks, resume by trial ID, and retain every outcome."""

    def __init__(
        self,
        *,
        protocol: BenchmarkProtocolConfig,
        run_manifest: BenchmarkRunManifest,
        cases: Sequence[BenchmarkCase],
        executor: TrialExecutor,
        repository: RunRepository,
        progress: Callable[[int, int], None] | None = None,
    ) -> None:
        if protocol.concurrency != 1:
            raise ValueError("Stage 9 primary runner supports concurrency 1 only")
        self._protocol = protocol
        self._manifest = run_manifest
        self._cases = tuple(sorted(cases, key=lambda item: item.query.query_id))
        self._executor = executor
        self._repository = repository
        self._progress = progress
        self._validate_cases()

    async def run(self) -> RunnerSummary:
        with self._repository.exclusive_lock():
            return await self._run_locked()

    async def _run_locked(self) -> RunnerSummary:
        existing = self._repository.load_records()
        self._validate_existing(existing)
        completed_ids = {record.trial_id for record in existing}
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
                RAGPlanError(
                    ErrorCode.INTERNAL_ERROR,
                    "benchmark trial failed internally",
                    retryable=False,
                ).response(f"bench-{invocation.trial_id}")
                elapsed_ms = (time.perf_counter_ns() - started_ns) / 1_000_000
                observation = TrialObservation(
                    status=TrialStatus.ERROR,
                    total_latency_ms=elapsed_ms,
                    budget_violated=elapsed_ms > invocation.latency_budget_ms,
                    error_code=ErrorCode.INTERNAL_ERROR,
                )
            record = self._record(invocation, case, observation)
            self._repository.append(record)
            completed_ids.add(record.trial_id)
            executed += 1
            total_completed = len(existing) + executed
            if self._progress is not None and (
                total_completed % 100 == 0
                or total_completed == self._manifest.expected_raw_row_count
            ):
                self._progress(total_completed, self._manifest.expected_raw_row_count)
        total = len(self._repository.load_records())
        return RunnerSummary(
            run_id=self._manifest.run_id,
            expected_rows=self._manifest.expected_raw_row_count,
            preexisting_rows=len(existing),
            executed_rows=executed,
            total_rows=total,
            complete=total == self._manifest.expected_raw_row_count,
            raw_path=self._repository.raw_path,
        )

    def _schedule(self) -> tuple[tuple[BenchmarkInvocation, BenchmarkCase], ...]:
        pairs = tuple((case, method) for case in self._cases for method in self._protocol.methods)
        blocks: list[tuple[TrialPhase, int]] = [(TrialPhase.COLD, 0)]
        blocks.extend((TrialPhase.WARMUP, index) for index in range(self._protocol.warmup_runs))
        blocks.extend((TrialPhase.MEASURED, index) for index in range(self._protocol.measured_runs))
        scheduled: list[tuple[BenchmarkInvocation, BenchmarkCase]] = []
        for phase, repetition in blocks:
            for budget in self._protocol.latency_budgets_ms:
                shuffled = list(pairs)
                seed_payload = (
                    f"{self._protocol.random_seed}:{phase.value}:{repetition}:{budget}"
                ).encode()
                block_seed = int.from_bytes(hashlib.sha256(seed_payload).digest()[:8], "big")
                random.Random(block_seed).shuffle(shuffled)
                for case, method in shuffled:
                    trial_id = trial_identity(
                        run_id=self._manifest.run_id,
                        query_id=case.query.query_id,
                        method=method.method,
                        latency_budget_ms=budget,
                        phase=phase,
                        repetition=repetition,
                    )
                    scheduled.append(
                        (
                            BenchmarkInvocation(
                                run_id=self._manifest.run_id,
                                trial_id=trial_id,
                                query_id=case.query.query_id,
                                query=case.query.question,
                                method=method,
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
        invocation: BenchmarkInvocation,
        case: BenchmarkCase,
        observation: TrialObservation,
    ) -> RawTrialRecord:
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
        runtime_version = (
            observation.runtime_semantics_version or self._manifest.runtime_semantics_version
        )
        if runtime_version != self._manifest.runtime_semantics_version:
            raise RawEvidenceError("executor runtime semantics do not match the run manifest")
        if observation.status in {TrialStatus.COMPLETE, TrialStatus.PARTIAL}:
            expected_config_version = self._expected_trace_config(invocation.method.method)
            if observation.trace_config_version != expected_config_version:
                raise RawEvidenceError("executor trace config does not match the benchmark method")
            expected_model_version = (
                f"{self._manifest.embedding_model_revision}:{self._manifest.extractor_version}"
            )
            if observation.trace_model_version != expected_model_version:
                raise RawEvidenceError("executor trace model does not match the run manifest")
        return RawTrialRecord(
            run_id=self._manifest.run_id,
            trial_id=invocation.trial_id,
            query_id=case.query.query_id,
            split=case.split,
            source_dataset=case.query.source_dataset,
            query_tags=case.query.query_tags,
            method=invocation.method.method,
            planner=invocation.method.planner,
            effective_planner=observation.effective_planner,
            configured_plan_id=invocation.method.plan_id,
            selected_plan_id=observation.selected_plan_id,
            graph_depth=invocation.method.graph_depth,
            latency_budget_ms=invocation.latency_budget_ms,
            trial_phase=invocation.phase,
            repetition=invocation.repetition,
            status=observation.status,
            recall_at_5=metrics[0],
            recall_at_10=metrics[1],
            mrr_at_10=metrics[2],
            ndcg_at_10=metrics[3],
            analyzer_latency_ms=observation.analyzer_latency_ms,
            planner_latency_ms=observation.planner_latency_ms,
            embedding_latency_ms=observation.embedding_latency_ms,
            vector_latency_ms=observation.vector_latency_ms,
            graph_latency_ms=observation.graph_latency_ms,
            fusion_latency_ms=observation.fusion_latency_ms,
            rerank_latency_ms=observation.rerank_latency_ms,
            total_latency_ms=observation.total_latency_ms,
            branch_results=observation.branch_results,
            timeout=observation.timeout,
            fallback=observation.fallback,
            error=observation.status is TrialStatus.ERROR,
            no_result=not ranked,
            budget_violated=observation.budget_violated,
            error_code=observation.error_code,
            result_count=len(ranked),
            protocol_config_sha256=self._manifest.protocol_config_sha256,
            environment_manifest_sha256=self._manifest.environment_manifest_sha256,
            benchmark_manifest_sha256=self._manifest.benchmark_manifest_sha256,
            split_hash=self._manifest.split_hash,
            qrels_sha256=self._manifest.qrels_sha256,
            corpus_version=self._manifest.corpus_version,
            corpus_chunk_count=self._manifest.corpus_chunk_count,
            corpus_chunk_ids_sha256=self._manifest.corpus_chunk_ids_sha256,
            embedding_model_revision=self._manifest.embedding_model_revision,
            extractor_version=self._manifest.extractor_version,
            plan_catalog_sha256=self._manifest.plan_catalog_sha256,
            planner_config_sha256=self._manifest.planner_config_sha256,
            query_feature_config_sha256=self._manifest.query_feature_config_sha256,
            graph_tier_policy_sha256=self._manifest.graph_tier_policy_sha256,
            rule_runtime_config_version=self._manifest.rule_runtime_config_version,
            stage2_artifact_set_sha256=self._manifest.stage2_artifact_set_sha256,
            runtime_semantics_version=self._manifest.runtime_semantics_version,
            trace_config_version=observation.trace_config_version,
            trace_model_version=observation.trace_model_version,
        )

    def _expected_trace_config(self, method: BenchmarkMethod) -> str:
        if method is BenchmarkMethod.RULE:
            return self._manifest.rule_runtime_config_version
        if method is BenchmarkMethod.GRAPH_DEPTH_2:
            return benchmark_graph_depth_2_config_version(self._manifest.plan_catalog_sha256)
        return self._manifest.plan_catalog_sha256

    def _validate_cases(self) -> None:
        ids = tuple(item.query.query_id for item in self._cases)
        if len(ids) != len(set(ids)):
            raise ValueError("benchmark cases contain duplicate query IDs")
        if any(case.split not in self._protocol.primary_splits for case in self._cases):
            raise ValueError("Stage 9 runner refuses the held-out test split")
        if any(not any(grade >= 1 for grade in case.relevance.values()) for case in self._cases):
            raise ValueError("every benchmark case requires a relevant chunk")
        if len(self._cases) != self._manifest.query_count:
            raise ValueError("benchmark case count does not match the run manifest")
        if canonical_sha256(list(sorted(ids))) != self._manifest.query_ids_sha256:
            raise ValueError("benchmark query IDs do not match the run manifest")
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
            raise ValueError("benchmark query identities do not match the run manifest")

    def _validate_existing(self, records: Sequence[RawTrialRecord]) -> None:
        expected = {
            invocation.trial_id: (invocation, case) for invocation, case in self._schedule()
        }
        for record in records:
            if record.run_id != self._manifest.run_id:
                raise RawEvidenceError("raw record run ID mismatch")
            scheduled = expected.get(record.trial_id)
            if scheduled is None:
                raise RawEvidenceError("raw record is not part of the deterministic schedule")
            invocation, case = scheduled
            observed_identity = (
                record.query_id,
                record.split,
                record.source_dataset,
                record.query_tags,
                record.method,
                record.planner,
                record.configured_plan_id,
                record.graph_depth,
                record.latency_budget_ms,
                record.trial_phase,
                record.repetition,
            )
            expected_identity = (
                case.query.query_id,
                case.split,
                case.query.source_dataset,
                case.query.query_tags,
                invocation.method.method,
                invocation.method.planner,
                invocation.method.plan_id,
                invocation.method.graph_depth,
                invocation.latency_budget_ms,
                invocation.phase,
                invocation.repetition,
            )
            if observed_identity != expected_identity:
                raise RawEvidenceError("raw record fields do not match its deterministic trial ID")
            _require_record_versions(record, self._manifest)
        if len(records) > self._manifest.expected_raw_row_count:
            raise RawEvidenceError("raw row count exceeds the immutable schedule")


def load_stage9_cases(
    *,
    manifest_path: Path,
    split_path: Path,
    qrels_path: Path,
    protocol: BenchmarkProtocolConfig,
) -> tuple[BenchmarkCase, ...]:
    manifest = load_benchmark_manifest(manifest_path)
    splits = load_split_manifest(split_path)
    qrels = load_qrels(qrels_path)
    if manifest.manifest_sha256 != protocol.benchmark_manifest_sha256:
        raise ValueError("benchmark manifest does not match the Stage 9 protocol")
    if splits.split_hash != protocol.split_hash:
        raise ValueError("split manifest does not match the Stage 9 protocol")
    qrels_sha256 = canonical_sha256([item.model_dump(mode="json") for item in qrels])
    if qrels_sha256 != protocol.qrels_sha256:
        raise ValueError("qrels do not match the Stage 9 protocol")
    split_by_id = {item.query_id: item.split for item in splits.assignments}
    if len(split_by_id) != len(splits.assignments):
        raise ValueError("split manifest contains duplicate query IDs")
    relevance = aggregate_relevance(qrels)
    cases: list[BenchmarkCase] = []
    for query in manifest.queries:
        try:
            split = split_by_id[query.query_id]
        except KeyError as exc:
            raise ValueError("benchmark query is missing from the split manifest") from exc
        if split not in protocol.primary_splits:
            continue
        try:
            query_relevance = relevance[query.query_id]
        except KeyError as exc:
            raise ValueError("benchmark query has no qrels") from exc
        cases.append(BenchmarkCase(query=query, split=split, relevance=query_relevance))
    if len(cases) != 480:
        raise ValueError("Stage 9 requires exactly 480 train/validation queries")
    return tuple(cases)


def _request_for_invocation(invocation: BenchmarkInvocation) -> SearchRequest:
    return SearchRequest(
        query=invocation.query,
        top_k=invocation.final_top_k,
        latency_budget_ms=invocation.latency_budget_ms,
        planner=invocation.method.planner,
        plan_id=invocation.method.plan_id,
    )


def _observation_from_response(
    response: SearchResponse,
    *,
    total_latency_ms: float,
) -> TrialObservation:
    trace = response.trace
    runtime_version = (
        trace.scheduler_trace.runtime_semantics_version
        if trace.scheduler_trace is not None
        else None
    )
    return TrialObservation(
        status=(
            TrialStatus.PARTIAL if response.status is SearchStatus.PARTIAL else TrialStatus.COMPLETE
        ),
        ranked_chunk_ids=tuple(item.canonical_chunk_id for item in response.results),
        selected_plan_id=response.planner_decision.selected_plan_id,
        effective_planner=response.planner_decision.effective_mode,
        analyzer_latency_ms=trace.analyzer_latency_ms,
        planner_latency_ms=trace.planner_latency_ms,
        embedding_latency_ms=trace.embedding_latency_ms,
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
        trace_config_version=trace.config_version,
        trace_model_version=trace.model_version,
        runtime_semantics_version=runtime_version,
    )


def _require_record_versions(
    record: RawTrialRecord,
    manifest: BenchmarkRunManifest,
) -> None:
    observed = (
        record.protocol_config_sha256,
        record.environment_manifest_sha256,
        record.benchmark_manifest_sha256,
        record.split_hash,
        record.qrels_sha256,
        record.corpus_version,
        record.corpus_chunk_count,
        record.corpus_chunk_ids_sha256,
        record.embedding_model_revision,
        record.extractor_version,
        record.plan_catalog_sha256,
        record.planner_config_sha256,
        record.query_feature_config_sha256,
        record.graph_tier_policy_sha256,
        record.rule_runtime_config_version,
        record.stage2_artifact_set_sha256,
        record.runtime_semantics_version,
    )
    expected = (
        manifest.protocol_config_sha256,
        manifest.environment_manifest_sha256,
        manifest.benchmark_manifest_sha256,
        manifest.split_hash,
        manifest.qrels_sha256,
        manifest.corpus_version,
        manifest.corpus_chunk_count,
        manifest.corpus_chunk_ids_sha256,
        manifest.embedding_model_revision,
        manifest.extractor_version,
        manifest.plan_catalog_sha256,
        manifest.planner_config_sha256,
        manifest.query_feature_config_sha256,
        manifest.graph_tier_policy_sha256,
        manifest.rule_runtime_config_version,
        manifest.stage2_artifact_set_sha256,
        manifest.runtime_semantics_version,
    )
    if observed != expected:
        raise RawEvidenceError("raw record version bundle differs from the run manifest")
