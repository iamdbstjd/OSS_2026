"""Frozen Stage 10 profiler, training-matrix, and Oracle evidence contracts."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from enum import StrEnum
from typing import Annotated, Final, Literal, Self

from pydantic import Field, model_validator

from ragplan.benchmark.contracts import QueryTag, SourceDataset, SplitName, canonical_json_bytes
from ragplan.benchmark.records import (
    BenchmarkProtocolConfig,
    BenchmarkQueryIdentity,
    BranchTrialRecord,
    EnvironmentManifest,
    RunId,
    TrialId,
    TrialPhase,
    TrialStatus,
    benchmark_query_identities_sha256,
)
from ragplan.core.errors import ErrorCode
from ragplan.core.models import (
    BranchKind,
    BranchStatus,
    FrozenModel,
    NonEmptyString,
    PlanDefinition,
    PlanId,
    PlannerMode,
    QueryFeatures,
    Sha256Hex,
)
from ragplan.planner.catalog import PlanCatalog

PROFILE_PROTOCOL_VERSION: Final[Literal["profile_v1"]] = "profile_v1"
PROFILE_RUN_SCHEMA_VERSION: Final[Literal["profile_run_v1"]] = "profile_run_v1"
PROFILE_RAW_SCHEMA_VERSION: Final[Literal["profile_raw_v1"]] = "profile_raw_v1"
TRAINING_MATRIX_SCHEMA_VERSION: Final[Literal["training_matrix_v1"]] = "training_matrix_v1"
P0_PROFILE_PLAN_IDS: Final = ("P0", "P1", "P2", "P3", "P4", "P5", "P6", "P8")
FINAL_ENGINE_PATH: Final[Literal["BaselineSearchEngine.benchmark_plan_search"]] = (
    "BaselineSearchEngine.benchmark_plan_search"
)


class QualityScope(StrEnum):
    FULL = "full"
    PARTIAL = "partial"
    NONE = "none"


class ProfilePlanFeatures(FrozenModel):
    """Static plan columns copied from the immutable catalog, never inferred from results."""

    plan_id: PlanId
    vector_enabled: bool
    graph_enabled: bool
    vector_top_k: Annotated[int, Field(ge=0)]
    graph_top_k: Annotated[int, Field(ge=0)]
    graph_depth: Annotated[int, Field(ge=0, le=3)]
    vector_weight: float = Field(ge=0.0, le=1.0)
    graph_weight: float = Field(ge=0.0, le=1.0)
    rerank_enabled: bool
    rerank_top_k: Annotated[int, Field(ge=0)]

    @classmethod
    def from_plan(cls, plan: PlanDefinition) -> ProfilePlanFeatures:
        return cls(
            plan_id=plan.id,
            vector_enabled=plan.vector_enabled,
            graph_enabled=plan.graph_enabled,
            vector_top_k=plan.vector_top_k,
            graph_top_k=plan.graph_top_k,
            graph_depth=plan.graph_depth,
            vector_weight=plan.vector_weight,
            graph_weight=plan.graph_weight,
            rerank_enabled=plan.rerank_enabled,
            rerank_top_k=plan.rerank_top_k,
        )

    @property
    def planner_mode(self) -> PlannerMode:
        if self.vector_enabled and self.graph_enabled:
            return PlannerMode.FIXED_HYBRID
        if self.vector_enabled:
            return PlannerMode.VECTOR
        if self.graph_enabled:
            return PlannerMode.GRAPH
        raise ValueError("profile plan has no enabled retrieval branch")


class ProfileProtocolConfig(FrozenModel):
    schema_version: Literal["profile_v1"] = PROFILE_PROTOCOL_VERSION
    baseline_protocol_sha256: Sha256Hex
    benchmark_manifest_sha256: Sha256Hex
    split_hash: Sha256Hex
    qrels_sha256: Sha256Hex
    corpus_version: NonEmptyString
    corpus_chunk_count: Annotated[int, Field(ge=1)]
    corpus_chunk_ids_sha256: Sha256Hex
    embedding_model_revision: NonEmptyString
    extractor_version: NonEmptyString
    plan_catalog_sha256: Sha256Hex
    query_feature_config_sha256: Sha256Hex
    runtime_semantics_version: Literal["v1"] = "v1"
    final_top_k: Literal[10] = 10
    latency_budgets_ms: tuple[Annotated[int, Field(ge=25, le=5000)], ...] = Field(min_length=1)
    cold_runs: Literal[1] = 1
    warmup_runs: Literal[2] = 2
    measured_runs: Literal[10] = 10
    concurrency: Literal[1] = 1
    random_seed: Literal[20260809] = 20260809
    primary_splits: tuple[SplitName, ...] = (SplitName.TRAIN, SplitName.VALIDATION)
    plan_features: tuple[ProfilePlanFeatures, ...] = Field(min_length=1)
    percentile_method: Literal["hyndman_fan_type_7_linear"] = "hyndman_fan_type_7_linear"

    @model_validator(mode="after")
    def _frozen_matrix(self) -> Self:
        if self.primary_splits != (SplitName.TRAIN, SplitName.VALIDATION):
            raise ValueError("profiler protocol may contain only train and validation")
        budgets = self.latency_budgets_ms
        if budgets != tuple(sorted(set(budgets))):
            raise ValueError("profiler budgets must be unique and ascending")
        plan_ids = tuple(item.plan_id for item in self.plan_features)
        expected_order = tuple(sorted(plan_ids, key=lambda item: int(item[1:])))
        if plan_ids != expected_order or len(plan_ids) != len(set(plan_ids)):
            raise ValueError("profiler plans must be unique and naturally ordered")
        if any(item.rerank_enabled for item in self.plan_features):
            raise ValueError("P0 profiler cannot execute reranker plans")
        return self

    @property
    def sha256(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.model_dump(mode="json"))).hexdigest()

    @property
    def trials_per_query_plan_budget(self) -> int:
        return self.cold_runs + self.warmup_runs + self.measured_runs

    @property
    def plan_ids(self) -> tuple[str, ...]:
        return tuple(item.plan_id for item in self.plan_features)

    def features_for(self, plan_id: str) -> ProfilePlanFeatures:
        for features in self.plan_features:
            if features.plan_id == plan_id:
                return features
        raise KeyError(plan_id)


def create_profile_protocol(
    baseline: BenchmarkProtocolConfig,
    catalog: PlanCatalog,
    *,
    plan_ids: Sequence[str] = P0_PROFILE_PLAN_IDS,
    latency_budgets_ms: Sequence[int] | None = None,
) -> ProfileProtocolConfig:
    """Bind the profiler matrix to the already-frozen Stage 9 evidence identities."""

    if catalog.sha256() != baseline.plan_catalog_sha256:
        raise ValueError("profile catalog does not match the baseline protocol")
    ids = tuple(plan_ids)
    features: list[ProfilePlanFeatures] = []
    for plan_id in ids:
        try:
            plan = catalog.plan_for_id(plan_id)
        except KeyError as exc:
            raise ValueError(f"unknown profiler plan: {plan_id}") from exc
        if not plan.enabled_in_p0:
            raise ValueError(f"profiler plan is disabled in P0: {plan_id}")
        features.append(ProfilePlanFeatures.from_plan(plan))
    return ProfileProtocolConfig(
        baseline_protocol_sha256=baseline.sha256,
        benchmark_manifest_sha256=baseline.benchmark_manifest_sha256,
        split_hash=baseline.split_hash,
        qrels_sha256=baseline.qrels_sha256,
        corpus_version=baseline.corpus_version,
        corpus_chunk_count=baseline.corpus_chunk_count,
        corpus_chunk_ids_sha256=baseline.corpus_chunk_ids_sha256,
        embedding_model_revision=baseline.embedding_model_revision,
        extractor_version=baseline.extractor_version,
        plan_catalog_sha256=baseline.plan_catalog_sha256,
        query_feature_config_sha256=baseline.query_feature_config_sha256,
        runtime_semantics_version=baseline.runtime_semantics_version,
        latency_budgets_ms=tuple(
            baseline.latency_budgets_ms if latency_budgets_ms is None else latency_budgets_ms
        ),
        plan_features=tuple(features),
    )


class ProfileRunManifest(FrozenModel):
    schema_version: Literal["profile_run_v1"] = PROFILE_RUN_SCHEMA_VERSION
    run_id: RunId
    created_at_utc: NonEmptyString
    profile_protocol_sha256: Sha256Hex
    baseline_protocol_sha256: Sha256Hex
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
    query_feature_config_sha256: Sha256Hex
    runtime_semantics_version: Literal["v1"] = "v1"
    cold_runs: Literal[1] = 1
    warmup_runs: Literal[2] = 2
    measured_runs: Literal[10] = 10
    concurrency: Literal[1] = 1
    query_count: Annotated[int, Field(ge=1)]
    query_ids_sha256: Sha256Hex
    query_identities_sha256: Sha256Hex
    plan_ids: tuple[PlanId, ...] = Field(min_length=1)
    plan_features_sha256: Sha256Hex
    latency_budgets_ms: tuple[Annotated[int, Field(ge=25, le=5000)], ...] = Field(min_length=1)
    expected_raw_row_count: Annotated[int, Field(ge=1)]

    @model_validator(mode="after")
    def _count_matches(self) -> Self:
        expected = (
            self.query_count
            * len(self.plan_ids)
            * len(self.latency_budgets_ms)
            * (self.cold_runs + self.warmup_runs + self.measured_runs)
        )
        if self.expected_raw_row_count != expected:
            raise ValueError("profile raw row count does not match the immutable matrix")
        return self


def create_profile_run_manifest(
    *,
    run_id: str,
    protocol: ProfileProtocolConfig,
    environment: EnvironmentManifest,
    query_identities: Sequence[BenchmarkQueryIdentity],
    created_at_utc: str,
) -> ProfileRunManifest:
    ordered = tuple(sorted(query_identities, key=lambda item: item.query_id))
    query_ids = tuple(item.query_id for item in ordered)
    if not query_ids or len(query_ids) != len(set(query_ids)):
        raise ValueError("profile query identities must be non-empty and unique")
    plan_payload = [item.model_dump(mode="json") for item in protocol.plan_features]
    expected_rows = (
        len(query_ids)
        * len(protocol.plan_features)
        * len(protocol.latency_budgets_ms)
        * protocol.trials_per_query_plan_budget
    )
    return ProfileRunManifest(
        run_id=run_id,
        created_at_utc=created_at_utc,
        profile_protocol_sha256=protocol.sha256,
        baseline_protocol_sha256=protocol.baseline_protocol_sha256,
        environment_manifest_sha256=environment.sha256,
        benchmark_manifest_sha256=protocol.benchmark_manifest_sha256,
        split_hash=protocol.split_hash,
        qrels_sha256=protocol.qrels_sha256,
        corpus_version=protocol.corpus_version,
        corpus_chunk_count=protocol.corpus_chunk_count,
        corpus_chunk_ids_sha256=protocol.corpus_chunk_ids_sha256,
        embedding_model_revision=protocol.embedding_model_revision,
        extractor_version=protocol.extractor_version,
        plan_catalog_sha256=protocol.plan_catalog_sha256,
        query_feature_config_sha256=protocol.query_feature_config_sha256,
        runtime_semantics_version=protocol.runtime_semantics_version,
        query_count=len(query_ids),
        query_ids_sha256=hashlib.sha256(canonical_json_bytes(list(query_ids))).hexdigest(),
        query_identities_sha256=benchmark_query_identities_sha256(ordered),
        plan_ids=protocol.plan_ids,
        plan_features_sha256=hashlib.sha256(canonical_json_bytes(plan_payload)).hexdigest(),
        latency_budgets_ms=protocol.latency_budgets_ms,
        expected_raw_row_count=expected_rows,
    )


class ProfileTrialRecord(FrozenModel):
    schema_version: Literal["profile_raw_v1"] = PROFILE_RAW_SCHEMA_VERSION
    run_id: RunId
    trial_id: TrialId
    query_id: NonEmptyString
    split: SplitName
    source_dataset: SourceDataset
    query_tags: tuple[QueryTag, ...] = Field(min_length=1)
    plan_id: PlanId
    plan_features: ProfilePlanFeatures
    query_features: QueryFeatures | None = None
    latency_budget_ms: Annotated[int, Field(ge=25, le=5000)]
    trial_phase: TrialPhase
    repetition: Annotated[int, Field(ge=0)]
    status: TrialStatus
    quality_scope: QualityScope
    recall_at_5: float = Field(ge=0.0, le=1.0)
    recall_at_10: float = Field(ge=0.0, le=1.0)
    mrr_at_10: float = Field(ge=0.0, le=1.0)
    ndcg_at_10: float = Field(ge=0.0, le=1.0)
    analyzer_latency_ms: float = Field(ge=0.0)
    planner_latency_ms: float = Field(ge=0.0)
    embedding_latency_ms: float = Field(ge=0.0)
    execution_latency_ms: float | None = Field(default=None, ge=0.0)
    vector_latency_ms: float | None = Field(default=None, ge=0.0)
    graph_latency_ms: float | None = Field(default=None, ge=0.0)
    fusion_latency_ms: float = Field(ge=0.0)
    rerank_latency_ms: float = Field(ge=0.0)
    total_latency_ms: float = Field(ge=0.0)
    branch_results: tuple[BranchTrialRecord, ...] = ()
    timeout: bool
    fallback: bool
    error: bool
    budget_violated: bool
    error_code: ErrorCode | None = None
    result_count: Annotated[int, Field(ge=0, le=10)]
    quality_label_valid: bool
    execution_latency_label_valid: bool
    invalid_exclusion_reason: str | None = None
    final_engine_path: Literal["BaselineSearchEngine.benchmark_plan_search"] = FINAL_ENGINE_PATH
    scheduler_trace_present: bool
    scheduler_runtime_semantics_version: str | None = None
    trace_config_version: str | None = None
    trace_model_version: str | None = None
    profile_protocol_sha256: Sha256Hex
    environment_manifest_sha256: Sha256Hex
    benchmark_manifest_sha256: Sha256Hex
    split_hash: Sha256Hex
    qrels_sha256: Sha256Hex
    corpus_version: NonEmptyString
    corpus_chunk_ids_sha256: Sha256Hex
    plan_catalog_sha256: Sha256Hex
    query_feature_config_sha256: Sha256Hex
    runtime_semantics_version: Literal["v1"] = "v1"

    @model_validator(mode="after")
    def _consistent_trial(self) -> Self:
        if self.split is SplitName.TEST:
            raise ValueError("profile raw evidence cannot contain held-out test queries")
        if self.plan_features.plan_id != self.plan_id:
            raise ValueError("profile plan columns do not match plan_id")
        if self.query_features is not None and self.query_features.final_top_k != 10:
            raise ValueError("profile query features require final_top_k=10")
        expected_scope = {
            TrialStatus.COMPLETE: QualityScope.FULL,
            TrialStatus.PARTIAL: QualityScope.PARTIAL,
            TrialStatus.TIMEOUT: QualityScope.NONE,
            TrialStatus.ERROR: QualityScope.NONE,
        }[self.status]
        if self.quality_scope is not expected_scope:
            raise ValueError("quality scope must distinguish full, partial, and failed trials")
        if self.timeout is not (self.status is TrialStatus.TIMEOUT):
            raise ValueError("profile timeout flag must match status")
        if self.fallback is not (self.status is TrialStatus.PARTIAL):
            raise ValueError("profile fallback flag must match partial status")
        if self.error is not (self.status is TrialStatus.ERROR):
            raise ValueError("profile error flag must match status")
        if self.status in {TrialStatus.TIMEOUT, TrialStatus.ERROR}:
            if self.result_count != 0 or self.error_code is None:
                raise ValueError("failed profile trials require an error code and zero results")
            if any(
                value != 0.0
                for value in (
                    self.recall_at_5,
                    self.recall_at_10,
                    self.mrr_at_10,
                    self.ndcg_at_10,
                )
            ):
                raise ValueError("failed profile trial quality must be zero")
        elif self.error_code is not None:
            raise ValueError("successful profile trials cannot have a top-level error code")
        if self.status in {TrialStatus.COMPLETE, TrialStatus.PARTIAL}:
            expected_branches = {
                branch
                for branch, enabled in (
                    (BranchKind.VECTOR, self.plan_features.vector_enabled),
                    (BranchKind.GRAPH, self.plan_features.graph_enabled),
                )
                if enabled
            }
            observed_branches = {item.branch for item in self.branch_results}
            if len(observed_branches) != len(self.branch_results):
                raise ValueError("profile branch evidence cannot contain duplicates")
            if observed_branches != expected_branches:
                raise ValueError("profile branch evidence must match the static plan")
            statuses = {item.status for item in self.branch_results}
            failed = {
                BranchStatus.TIMED_OUT,
                BranchStatus.FAILED,
                BranchStatus.CANCELLED,
            }
            if self.status is TrialStatus.COMPLETE and statuses != {BranchStatus.SUCCEEDED}:
                raise ValueError("complete profile trials require every branch to succeed")
            if self.status is TrialStatus.PARTIAL and not (
                BranchStatus.SUCCEEDED in statuses and bool(statuses & failed)
            ):
                raise ValueError("partial profile trials require one usable and one failed branch")
        if self.execution_latency_label_valid:
            required = (
                self.execution_latency_ms is not None
                and self.scheduler_trace_present
                and self.scheduler_runtime_semantics_version == self.runtime_semantics_version
                and self.trace_config_version == self.plan_catalog_sha256
                and self.query_features is not None
            )
            if not required:
                raise ValueError("valid execution labels require final scheduler trace evidence")
        if (
            self.quality_label_valid
            and self.status
            in {
                TrialStatus.COMPLETE,
                TrialStatus.PARTIAL,
            }
            and self.query_features is None
        ):
            raise ValueError("successful quality labels require query feature evidence")
        if (self.quality_label_valid and self.execution_latency_label_valid) is not (
            self.invalid_exclusion_reason is None
        ):
            raise ValueError("invalid label rows require one exclusion reason")
        if not math.isfinite(self.total_latency_ms):
            raise ValueError("profile total latency must be finite")
        return self


def profile_trial_identity(
    *,
    run_id: str,
    query_id: str,
    plan_id: str,
    latency_budget_ms: int,
    phase: TrialPhase,
    repetition: int,
) -> str:
    payload = (
        f"{PROFILE_RAW_SCHEMA_VERSION}\0{run_id}\0{query_id}\0{plan_id}\0"
        f"{latency_budget_ms}\0{phase.value}\0{repetition}"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def profile_records_sha256(records: Sequence[ProfileTrialRecord]) -> str:
    rows = b"\n".join(
        canonical_json_bytes(item.model_dump(mode="json"))
        for item in sorted(records, key=lambda record: record.trial_id)
    )
    return hashlib.sha256(rows + (b"\n" if rows else b"")).hexdigest()


class TrainingMatrixRow(FrozenModel):
    schema_version: Literal["training_matrix_v1"] = TRAINING_MATRIX_SCHEMA_VERSION
    run_id: RunId
    query_id: NonEmptyString
    split: SplitName
    source_dataset: SourceDataset
    query_tags: tuple[QueryTag, ...] = Field(min_length=1)
    plan_id: PlanId
    query_features: QueryFeatures
    plan_features: ProfilePlanFeatures
    latency_budget_ms: Annotated[int, Field(ge=25, le=5000)]
    measured_trial_count: Literal[10] = 10
    quality_label_trial_count: Annotated[int, Field(ge=0, le=10)]
    execution_latency_trial_count: Annotated[int, Field(ge=0, le=10)]
    complete_trial_count: Annotated[int, Field(ge=0, le=10)]
    partial_trial_count: Annotated[int, Field(ge=0, le=10)]
    timeout_trial_count: Annotated[int, Field(ge=0, le=10)]
    error_trial_count: Annotated[int, Field(ge=0, le=10)]
    fallback_trial_count: Annotated[int, Field(ge=0, le=10)]
    recall_at_5: float = Field(ge=0.0, le=1.0)
    recall_at_10: float = Field(ge=0.0, le=1.0)
    mrr_at_10: float = Field(ge=0.0, le=1.0)
    ndcg_at_10: float = Field(ge=0.0, le=1.0)
    full_result_recall_at_10: float | None = Field(default=None, ge=0.0, le=1.0)
    partial_result_recall_at_10: float | None = Field(default=None, ge=0.0, le=1.0)
    p95_execution_latency_ms: float | None = Field(default=None, ge=0.0)
    fallback_rate: float = Field(ge=0.0, le=1.0)
    budget_violation_rate: float = Field(ge=0.0, le=1.0)
    quality_label_valid: bool
    execution_latency_label_valid: bool
    usable_for_model_training: bool
    invalid_exclusion_reasons: tuple[str, ...] = ()
    source_trial_ids_sha256: Sha256Hex
    profile_protocol_sha256: Sha256Hex
    environment_manifest_sha256: Sha256Hex
    benchmark_manifest_sha256: Sha256Hex
    split_hash: Sha256Hex
    qrels_sha256: Sha256Hex
    corpus_version: NonEmptyString
    corpus_chunk_ids_sha256: Sha256Hex
    plan_catalog_sha256: Sha256Hex
    query_feature_config_sha256: Sha256Hex
    runtime_semantics_version: Literal["v1"] = "v1"

    @model_validator(mode="after")
    def _matrix_counts(self) -> Self:
        status_total = (
            self.complete_trial_count
            + self.partial_trial_count
            + self.timeout_trial_count
            + self.error_trial_count
        )
        if status_total != self.measured_trial_count:
            raise ValueError("training matrix status counts must cover all measured trials")
        if self.fallback_trial_count != self.partial_trial_count:
            raise ValueError("training matrix fallback count must equal partial count")
        if self.quality_label_valid is not (
            self.quality_label_trial_count == self.measured_trial_count
        ):
            raise ValueError("quality validity must reflect complete label coverage")
        if self.execution_latency_label_valid is not (
            self.execution_latency_trial_count == self.measured_trial_count
        ):
            raise ValueError("latency validity must reflect complete label coverage")
        if self.usable_for_model_training is not (
            self.quality_label_valid and self.execution_latency_label_valid
        ):
            raise ValueError("model-training usability must require both labels")
        if self.execution_latency_label_valid is not (self.p95_execution_latency_ms is not None):
            raise ValueError("p95 execution latency requires all measured execution labels")
        if self.usable_for_model_training is not (not self.invalid_exclusion_reasons):
            raise ValueError("invalid reasons must describe every unusable matrix row")
        return self


__all__ = [
    "FINAL_ENGINE_PATH",
    "P0_PROFILE_PLAN_IDS",
    "ProfilePlanFeatures",
    "ProfileProtocolConfig",
    "ProfileRunManifest",
    "ProfileTrialRecord",
    "QualityScope",
    "TrainingMatrixRow",
    "create_profile_protocol",
    "create_profile_run_manifest",
    "profile_records_sha256",
    "profile_trial_identity",
]
