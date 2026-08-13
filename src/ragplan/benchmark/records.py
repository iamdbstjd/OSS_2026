"""Frozen Stage 9 benchmark protocol, raw-row, and evidence contracts."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from enum import StrEnum
from typing import Annotated, Final, Literal, Self

from pydantic import Field, StringConstraints, field_validator, model_validator

from ragplan.benchmark.contracts import (
    CORPUS_VERSION,
    QRELS_VERSION,
    SPLIT_SEED,
    QueryTag,
    SourceDataset,
    SplitName,
    canonical_json_bytes,
)
from ragplan.core.errors import ErrorCode
from ragplan.core.models import (
    BranchKind,
    BranchStatus,
    FrozenModel,
    NonEmptyString,
    PlannerMode,
    Sha256Hex,
)

BENCHMARK_PROTOCOL_VERSION: Final[Literal["baseline_v1"]] = "baseline_v1"
BENCHMARK_RUN_SCHEMA_VERSION: Final[Literal["benchmark_run_v1"]] = "benchmark_run_v1"
RAW_RECORD_SCHEMA_VERSION: Final[Literal["benchmark_raw_v1"]] = "benchmark_raw_v1"
AGGREGATE_SCHEMA_VERSION: Final[Literal["benchmark_aggregate_v1"]] = "benchmark_aggregate_v1"
RUNTIME_SEMANTICS_VERSION: Final[Literal["v1"]] = "v1"
BENCHMARK_RANDOM_SEED: Final[Literal[20260809]] = SPLIT_SEED
BOOTSTRAP_SAMPLES: Final[Literal[10000]] = 10_000

RunId = Annotated[str, StringConstraints(pattern=r"^[a-z0-9][a-z0-9._-]{2,95}$")]
TrialId = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]


class BenchmarkMethod(StrEnum):
    VECTOR = "vector"
    GRAPH_DEPTH_1 = "graph_depth_1"
    GRAPH_DEPTH_2 = "graph_depth_2"
    GRAPH_DEPTH_3 = "graph_depth_3"
    FIXED_P4 = "fixed_p4"
    FIXED_P5 = "fixed_p5"
    FIXED_P6 = "fixed_p6"
    FIXED_P8 = "fixed_p8"
    RULE = "rule"
    BEST_FIXED = "best_fixed"


EXECUTED_METHODS: Final = (
    BenchmarkMethod.VECTOR,
    BenchmarkMethod.GRAPH_DEPTH_1,
    BenchmarkMethod.GRAPH_DEPTH_2,
    BenchmarkMethod.GRAPH_DEPTH_3,
    BenchmarkMethod.FIXED_P4,
    BenchmarkMethod.FIXED_P5,
    BenchmarkMethod.FIXED_P6,
    BenchmarkMethod.FIXED_P8,
    BenchmarkMethod.RULE,
)
FIXED_METHODS: Final = (
    BenchmarkMethod.FIXED_P4,
    BenchmarkMethod.FIXED_P5,
    BenchmarkMethod.FIXED_P6,
    BenchmarkMethod.FIXED_P8,
)


class TrialPhase(StrEnum):
    COLD = "cold"
    WARMUP = "warmup"
    MEASURED = "measured"


class TrialStatus(StrEnum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    TIMEOUT = "timeout"
    ERROR = "error"


class AggregateDimension(StrEnum):
    OVERALL = "overall"
    QUERY_TYPE = "query_type"
    DATASET_SOURCE = "dataset_source"


class RuntimeRequirements(FrozenModel):
    cpu_only: Literal[True] = True
    single_host: Literal[True] = True
    local_docker_network: Literal[True] = True
    concurrency: Literal[1] = 1
    competing_workload_allowed: Literal[False] = False
    cpu_governor_must_be_recorded: Literal[True] = True
    container_limits_must_be_recorded: Literal[True] = True


class MethodDefinition(FrozenModel):
    method: BenchmarkMethod
    planner: PlannerMode
    plan_id: str | None = None
    graph_depth: Annotated[int, Field(ge=1, le=3)] | None = None

    @model_validator(mode="after")
    def _match_method(self) -> Self:
        expected: dict[BenchmarkMethod, tuple[PlannerMode, str | None, int | None]] = {
            BenchmarkMethod.VECTOR: (PlannerMode.VECTOR, None, None),
            BenchmarkMethod.GRAPH_DEPTH_1: (PlannerMode.GRAPH, None, 1),
            BenchmarkMethod.GRAPH_DEPTH_2: (PlannerMode.GRAPH, None, 2),
            BenchmarkMethod.GRAPH_DEPTH_3: (PlannerMode.GRAPH, None, 3),
            BenchmarkMethod.FIXED_P4: (PlannerMode.FIXED_HYBRID, "P4", None),
            BenchmarkMethod.FIXED_P5: (PlannerMode.FIXED_HYBRID, "P5", None),
            BenchmarkMethod.FIXED_P6: (PlannerMode.FIXED_HYBRID, "P6", None),
            BenchmarkMethod.FIXED_P8: (PlannerMode.FIXED_HYBRID, "P8", None),
            BenchmarkMethod.RULE: (PlannerMode.RULE, None, None),
        }
        if self.method is BenchmarkMethod.BEST_FIXED:
            raise ValueError("best_fixed is derived from validation and is not an executed method")
        if (self.planner, self.plan_id, self.graph_depth) != expected[self.method]:
            raise ValueError("benchmark method semantics do not match baseline_v1")
        return self


class BenchmarkProtocolConfig(FrozenModel):
    schema_version: Literal["baseline_v1"] = BENCHMARK_PROTOCOL_VERSION
    benchmark_id: Literal["adaptive_rag_bench_v1"] = "adaptive_rag_bench_v1"
    dataset_version: Literal["adaptive_rag_bench_v1"] = "adaptive_rag_bench_v1"
    split_version: Literal["splits_v1"] = "splits_v1"
    qrels_version: Literal["qrels_v1"] = QRELS_VERSION
    corpus_version: Literal["adaptive_rag_bench_v1-corpus-v1"] = CORPUS_VERSION
    corpus_chunk_count: Literal[8604] = 8604
    corpus_chunk_ids_sha256: Sha256Hex
    benchmark_manifest_sha256: Sha256Hex
    split_hash: Sha256Hex
    qrels_sha256: Sha256Hex
    embedding_model_revision: NonEmptyString
    extractor_version: NonEmptyString
    plan_catalog_sha256: Sha256Hex
    planner_config_sha256: Sha256Hex
    query_feature_config_sha256: Sha256Hex
    graph_tier_policy_sha256: Sha256Hex
    rule_runtime_config_version: Sha256Hex
    stage2_artifact_set_sha256: Sha256Hex
    runtime_semantics_version: Literal["v1"] = RUNTIME_SEMANTICS_VERSION
    latency_budgets_ms: tuple[int, ...] = Field(min_length=1)
    final_top_k: Literal[10] = 10
    cold_runs: Literal[1] = 1
    warmup_runs: Literal[2] = 2
    measured_runs: Literal[10] = 10
    concurrency: Literal[1] = 1
    random_seed: Literal[20260809] = BENCHMARK_RANDOM_SEED
    bootstrap_samples: Literal[10000] = BOOTSTRAP_SAMPLES
    primary_splits: tuple[SplitName, ...] = (SplitName.TRAIN, SplitName.VALIDATION)
    methods: tuple[MethodDefinition, ...] = Field(min_length=9, max_length=9)
    derive_best_fixed_on_validation: Literal[True] = True
    percentile_method: Literal["hyndman_fan_type_7_linear"] = "hyndman_fan_type_7_linear"
    timeout_error_quality: Literal["zero"] = "zero"
    outlier_policy: Literal["retain_all"] = "retain_all"
    runtime_requirements: RuntimeRequirements

    @field_validator("latency_budgets_ms")
    @classmethod
    def _valid_budgets(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if value != (50, 100, 200, 500):
            raise ValueError("baseline_v1 latency budgets are fixed at 50/100/200/500 ms")
        return value

    @model_validator(mode="after")
    def _frozen_protocol(self) -> Self:
        if self.primary_splits != (SplitName.TRAIN, SplitName.VALIDATION):
            raise ValueError("Stage 9 may execute only train and validation splits")
        if tuple(item.method for item in self.methods) != EXECUTED_METHODS:
            raise ValueError("baseline_v1 methods and their order are immutable")
        if self.concurrency != self.runtime_requirements.concurrency:
            raise ValueError("runner and environment concurrency must match")
        return self

    @property
    def sha256(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.model_dump(mode="json"))).hexdigest()

    @property
    def trials_per_query_method_budget(self) -> int:
        return self.cold_runs + self.warmup_runs + self.measured_runs


class EnvironmentManifest(FrozenModel):
    schema_version: Literal["environment_v1"] = "environment_v1"
    captured_at_utc: NonEmptyString
    cpu_only: Literal[True] = True
    single_host: Literal[True] = True
    local_docker_network: Literal[True] = True
    competing_workload: Literal[False] = False
    concurrency: Literal[1] = 1
    cold_reset_strategy: Literal["fresh_engine_before_cold_sweep"] = (
        "fresh_engine_before_cold_sweep"
    )
    force_vector_only: Literal[False] = False
    os_name: NonEmptyString
    os_release: NonEmptyString
    machine: NonEmptyString
    cpu_model: NonEmptyString
    logical_cpu_count: Annotated[int, Field(ge=1)]
    cpu_governor: NonEmptyString
    python_version: NonEmptyString
    qdrant_image: NonEmptyString
    neo4j_image: NonEmptyString
    api_image: NonEmptyString
    container_resource_limits: NonEmptyString
    runtime_source_sha256: Sha256Hex
    dependency_lock_sha256: Sha256Hex
    docker_compose_sha256: Sha256Hex
    db_tuning_sha256: Sha256Hex
    notes: NonEmptyString

    @model_validator(mode="after")
    def _pinned_database_images(self) -> Self:
        for image in (self.qdrant_image, self.neo4j_image):
            reference, separator, digest = image.rpartition("@sha256:")
            if (
                not reference
                or not separator
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ValueError("benchmark database images must be pinned by SHA-256 digest")
        return self

    @property
    def sha256(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.model_dump(mode="json"))).hexdigest()


class BenchmarkQueryIdentity(FrozenModel):
    query_id: NonEmptyString
    split: SplitName
    source_dataset: SourceDataset
    query_tags: tuple[QueryTag, ...] = Field(min_length=1)


def benchmark_query_identities_sha256(
    identities: Sequence[BenchmarkQueryIdentity],
) -> str:
    ordered = tuple(sorted(identities, key=lambda item: item.query_id))
    ids = tuple(item.query_id for item in ordered)
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("benchmark query identities must have unique query IDs")
    return hashlib.sha256(
        canonical_json_bytes([item.model_dump(mode="json") for item in ordered])
    ).hexdigest()


class BenchmarkRunManifest(FrozenModel):
    schema_version: Literal["benchmark_run_v1"] = BENCHMARK_RUN_SCHEMA_VERSION
    run_id: RunId
    created_at_utc: NonEmptyString
    protocol_config_sha256: Sha256Hex
    environment_manifest_sha256: Sha256Hex
    benchmark_manifest_sha256: Sha256Hex
    split_hash: Sha256Hex
    qrels_sha256: Sha256Hex
    corpus_version: NonEmptyString
    corpus_chunk_count: Annotated[int, Field(ge=1)]
    corpus_chunk_ids_sha256: Sha256Hex
    embedding_model_revision: NonEmptyString
    extractor_version: NonEmptyString
    plan_catalog_sha256: Sha256Hex
    planner_config_sha256: Sha256Hex
    query_feature_config_sha256: Sha256Hex
    graph_tier_policy_sha256: Sha256Hex
    rule_runtime_config_version: Sha256Hex
    stage2_artifact_set_sha256: Sha256Hex
    runtime_semantics_version: Literal["v1"] = RUNTIME_SEMANTICS_VERSION
    random_seed: Literal[20260809] = BENCHMARK_RANDOM_SEED
    cold_runs: Literal[1] = 1
    warmup_runs: Literal[2] = 2
    measured_runs: Literal[10] = 10
    concurrency: Literal[1] = 1
    query_count: Annotated[int, Field(ge=1)]
    query_ids_sha256: Sha256Hex
    query_identities_sha256: Sha256Hex
    method_count: Literal[9] = 9
    latency_budgets_ms: tuple[int, ...] = Field(min_length=1)
    expected_raw_row_count: Annotated[int, Field(ge=1)]

    @field_validator("latency_budgets_ms")
    @classmethod
    def _valid_run_budgets(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if value != (50, 100, 200, 500):
            raise ValueError("run manifest latency budgets must match baseline_v1")
        return value

    @model_validator(mode="after")
    def _row_count_matches(self) -> Self:
        expected = (
            self.query_count
            * self.method_count
            * len(self.latency_budgets_ms)
            * (self.cold_runs + self.warmup_runs + self.measured_runs)
        )
        if self.expected_raw_row_count != expected:
            raise ValueError("expected raw row count does not match the frozen protocol")
        return self


class BranchTrialRecord(FrozenModel):
    branch: BranchKind
    status: BranchStatus
    latency_ms: float | None = Field(default=None, ge=0.0)
    error_code: ErrorCode | None = None

    @model_validator(mode="after")
    def _terminal_branch_shape(self) -> Self:
        if self.status not in {
            BranchStatus.SUCCEEDED,
            BranchStatus.TIMED_OUT,
            BranchStatus.FAILED,
            BranchStatus.CANCELLED,
        }:
            raise ValueError("raw branch status must be terminal")
        if self.latency_ms is None:
            raise ValueError("raw terminal branch requires latency")
        if (self.status is BranchStatus.FAILED) is not (self.error_code is not None):
            raise ValueError("only a failed raw branch requires an error code")
        return self


class TrialObservation(FrozenModel):
    """Executor output before benchmark identity and qrels metrics are attached."""

    status: TrialStatus
    ranked_chunk_ids: tuple[str, ...] = ()
    selected_plan_id: str | None = None
    effective_planner: PlannerMode | None = None
    analyzer_latency_ms: float = Field(default=0.0, ge=0.0)
    planner_latency_ms: float = Field(default=0.0, ge=0.0)
    embedding_latency_ms: float = Field(default=0.0, ge=0.0)
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
    trace_config_version: str | None = None
    trace_model_version: str | None = None
    runtime_semantics_version: str | None = None

    @model_validator(mode="after")
    def _status_flags_match(self) -> Self:
        if self.timeout is not (self.status is TrialStatus.TIMEOUT):
            raise ValueError("timeout flag must match timeout trial status")
        if self.fallback is not (self.status is TrialStatus.PARTIAL):
            raise ValueError("fallback flag must match partial trial status")
        if self.status in {TrialStatus.TIMEOUT, TrialStatus.ERROR} and self.ranked_chunk_ids:
            raise ValueError("timeout/error observations cannot expose ranked results")
        if self.status in {TrialStatus.TIMEOUT, TrialStatus.ERROR} and self.error_code is None:
            raise ValueError("timeout/error observations require an error code")
        return self


class RawTrialRecord(FrozenModel):
    schema_version: Literal["benchmark_raw_v1"] = RAW_RECORD_SCHEMA_VERSION
    run_id: RunId
    trial_id: TrialId
    query_id: NonEmptyString
    split: SplitName
    source_dataset: SourceDataset
    query_tags: tuple[QueryTag, ...] = Field(min_length=1)
    method: BenchmarkMethod
    planner: PlannerMode
    effective_planner: PlannerMode | None = None
    configured_plan_id: str | None = None
    selected_plan_id: str | None = None
    graph_depth: Annotated[int, Field(ge=1, le=3)] | None = None
    latency_budget_ms: Annotated[int, Field(ge=25, le=5000)]
    trial_phase: TrialPhase
    repetition: Annotated[int, Field(ge=0)]
    status: TrialStatus
    recall_at_5: float = Field(ge=0.0, le=1.0)
    recall_at_10: float = Field(ge=0.0, le=1.0)
    mrr_at_10: float = Field(ge=0.0, le=1.0)
    ndcg_at_10: float = Field(ge=0.0, le=1.0)
    analyzer_latency_ms: float = Field(ge=0.0)
    planner_latency_ms: float = Field(ge=0.0)
    embedding_latency_ms: float = Field(ge=0.0)
    vector_latency_ms: float | None = Field(default=None, ge=0.0)
    graph_latency_ms: float | None = Field(default=None, ge=0.0)
    fusion_latency_ms: float = Field(ge=0.0)
    rerank_latency_ms: float = Field(ge=0.0)
    total_latency_ms: float = Field(ge=0.0)
    branch_results: tuple[BranchTrialRecord, ...] = ()
    timeout: bool
    fallback: bool
    error: bool
    no_result: bool
    budget_violated: bool
    error_code: ErrorCode | None = None
    result_count: Annotated[int, Field(ge=0, le=10)]
    protocol_config_sha256: Sha256Hex
    environment_manifest_sha256: Sha256Hex
    benchmark_manifest_sha256: Sha256Hex
    split_hash: Sha256Hex
    qrels_sha256: Sha256Hex
    corpus_version: NonEmptyString
    corpus_chunk_count: Annotated[int, Field(ge=1)]
    corpus_chunk_ids_sha256: Sha256Hex
    embedding_model_revision: NonEmptyString
    extractor_version: NonEmptyString
    plan_catalog_sha256: Sha256Hex
    planner_config_sha256: Sha256Hex
    query_feature_config_sha256: Sha256Hex
    graph_tier_policy_sha256: Sha256Hex
    rule_runtime_config_version: Sha256Hex
    stage2_artifact_set_sha256: Sha256Hex
    runtime_semantics_version: Literal["v1"] = RUNTIME_SEMANTICS_VERSION
    trace_config_version: str | None = None
    trace_model_version: str | None = None

    @model_validator(mode="after")
    def _record_consistency(self) -> Self:
        if self.method is BenchmarkMethod.BEST_FIXED:
            raise ValueError("best_fixed rows are derived and cannot enter raw evidence")
        method_contract = {
            BenchmarkMethod.VECTOR: (PlannerMode.VECTOR, None, None, "P0"),
            BenchmarkMethod.GRAPH_DEPTH_1: (PlannerMode.GRAPH, None, 1, "P2"),
            BenchmarkMethod.GRAPH_DEPTH_2: (PlannerMode.GRAPH, None, 2, "P3"),
            BenchmarkMethod.GRAPH_DEPTH_3: (PlannerMode.GRAPH, None, 3, "P3"),
            BenchmarkMethod.FIXED_P4: (PlannerMode.FIXED_HYBRID, "P4", None, "P4"),
            BenchmarkMethod.FIXED_P5: (PlannerMode.FIXED_HYBRID, "P5", None, "P5"),
            BenchmarkMethod.FIXED_P6: (PlannerMode.FIXED_HYBRID, "P6", None, "P6"),
            BenchmarkMethod.FIXED_P8: (PlannerMode.FIXED_HYBRID, "P8", None, "P8"),
            BenchmarkMethod.RULE: (PlannerMode.RULE, None, None, None),
        }
        expected_planner, expected_plan, expected_depth, expected_selected = method_contract[
            self.method
        ]
        if (self.planner, self.configured_plan_id, self.graph_depth) != (
            expected_planner,
            expected_plan,
            expected_depth,
        ):
            raise ValueError("raw method fields do not match baseline_v1")
        if self.timeout is not (self.status is TrialStatus.TIMEOUT):
            raise ValueError("raw timeout flag must match status")
        if self.fallback is not (self.status is TrialStatus.PARTIAL):
            raise ValueError("raw fallback flag must match status")
        if self.error is not (self.status is TrialStatus.ERROR):
            raise ValueError("raw error flag must match status")
        if self.no_result is not (self.result_count == 0):
            raise ValueError("no_result must reflect result_count")
        if self.no_result and any(
            value != 0.0
            for value in (self.recall_at_5, self.recall_at_10, self.mrr_at_10, self.ndcg_at_10)
        ):
            raise ValueError("no-result quality must be zero")
        if self.status in {TrialStatus.TIMEOUT, TrialStatus.ERROR}:
            if any(
                value != 0.0
                for value in (
                    self.recall_at_5,
                    self.recall_at_10,
                    self.mrr_at_10,
                    self.ndcg_at_10,
                )
            ):
                raise ValueError("timeout/error quality must be zero")
            if self.result_count != 0 or self.error_code is None:
                raise ValueError("timeout/error rows require zero results and an error code")
        else:
            if self.error_code is not None:
                raise ValueError("complete/partial rows cannot have a top-level error code")
            if self.selected_plan_id is None or self.effective_planner is None:
                raise ValueError("complete/partial rows require the executed plan and planner")
            if self.trace_config_version is None or self.trace_model_version is None:
                raise ValueError("complete/partial rows require trace config and model identities")
            if not self.branch_results:
                raise ValueError("complete/partial rows require terminal branch evidence")
            expected_branches = {
                PlannerMode.VECTOR: {BranchKind.VECTOR},
                PlannerMode.GRAPH: {BranchKind.GRAPH},
                PlannerMode.FIXED_HYBRID: {BranchKind.VECTOR, BranchKind.GRAPH},
            }.get(self.effective_planner)
            observed_branches = {item.branch for item in self.branch_results}
            if len(observed_branches) != len(self.branch_results):
                raise ValueError("raw branch evidence cannot contain duplicate branches")
            if expected_branches is None or observed_branches != expected_branches:
                raise ValueError("raw branch evidence does not match the effective planner")
            statuses = {item.status for item in self.branch_results}
            failed_statuses = {
                BranchStatus.TIMED_OUT,
                BranchStatus.FAILED,
                BranchStatus.CANCELLED,
            }
            if self.status is TrialStatus.COMPLETE and statuses != {BranchStatus.SUCCEEDED}:
                raise ValueError("complete raw rows require every branch to succeed")
            if self.status is TrialStatus.PARTIAL and not (
                BranchStatus.SUCCEEDED in statuses and bool(statuses & failed_statuses)
            ):
                raise ValueError("partial raw rows require a successful and failed branch")
            if expected_selected is not None and self.selected_plan_id != expected_selected:
                raise ValueError("raw selected plan does not match the explicit baseline")
            if (
                self.method is not BenchmarkMethod.RULE
                and self.effective_planner is not self.planner
            ):
                raise ValueError("explicit baseline effective planner must match its request")
        if not math.isfinite(self.total_latency_ms):
            raise ValueError("trial latency must be finite")
        return self


def trial_identity(
    *,
    run_id: str,
    query_id: str,
    method: BenchmarkMethod,
    latency_budget_ms: int,
    phase: TrialPhase,
    repetition: int,
) -> str:
    payload = (
        f"{RAW_RECORD_SCHEMA_VERSION}\0{run_id}\0{query_id}\0{method.value}\0"
        f"{latency_budget_ms}\0{phase.value}\0{repetition}"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def raw_records_sha256(records: tuple[RawTrialRecord, ...]) -> str:
    rows = b"\n".join(
        canonical_json_bytes(item.model_dump(mode="json"))
        for item in sorted(records, key=lambda record: record.trial_id)
    )
    return hashlib.sha256(rows + (b"\n" if rows else b"")).hexdigest()


def parse_raw_record(line: str) -> RawTrialRecord:
    try:
        return RawTrialRecord.model_validate_json(line)
    except (ValueError, json.JSONDecodeError) as exc:
        raise ValueError("raw benchmark row is invalid") from exc
