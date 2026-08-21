"""Research-only Stage 12 cost-aware candidate scoring.

This module intentionally has no dependency on the API or retrieval engine.  The
frozen Stage 11 R2 models require benchmark-only source/tag metadata, so they may
be used for offline comparison but can never be activated as an online planner.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, cast

import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import HistGradientBoostingRegressor  # type: ignore[import-untyped]

from ragplan.benchmark.contracts import QueryTag, SourceDataset, canonical_sha256
from ragplan.benchmark.profile_records import ProfilePlanFeatures
from ragplan.core.deadline import Deadline, MonotonicClock, PerfCounterClock
from ragplan.core.errors import ErrorCode, RAGPlanError
from ragplan.core.models import (
    FrozenModel,
    PlanEstimate,
    PlannerDecision,
    PlannerMode,
    QueryAnalysis,
)
from ragplan.planner.artifacts import (
    ArtifactStatus,
    CompatibilityContext,
    CostModelArtifactManifest,
    ModelKind,
    installed_dependency_versions,
    load_cost_model,
    manifest_path_for,
)
from ragplan.planner.catalog import PlanCatalog, stable_tie_break_key
from ragplan.planner.training import (
    COST_FEATURE_SCHEMA_VERSION,
    FEATURE_NAMES,
    OfflineCostModelInput,
    encode_offline_inputs,
)

OFFLINE_OPTIMIZER_VERSION: Literal["cost_aware_offline_v1"] = "cost_aware_offline_v1"


class Predictor(Protocol):
    def predict(self, features: NDArray[np.float64]) -> object: ...


class OfflineResearchContext(FrozenModel):
    """Metadata present in the benchmark matrix but absent from online requests."""

    source_dataset: SourceDataset
    query_tags: tuple[QueryTag, ...]


class OfflineOptimizationResult(FrozenModel):
    execution_mode: Literal["research_only_offline"] = "research_only_offline"
    public_api_enabled: Literal[False] = False
    decision: PlannerDecision
    planner_overhead_ms: float
    artifact_status: Literal["research_only"] = "research_only"


@dataclass(frozen=True, slots=True)
class HistoricalResearchBundle:
    """Safely loaded pair of checksum-bound, historical research estimators."""

    quality_model: HistGradientBoostingRegressor
    latency_model: HistGradientBoostingRegressor
    quality_manifest: CostModelArtifactManifest
    latency_manifest: CostModelArtifactManifest

    @property
    def model_version(self) -> str:
        return (
            f"{self.quality_manifest.model.model_version}:"
            f"{self.latency_manifest.model.model_version}"
        )


def load_historical_research_bundle(
    *,
    quality_artifact: Path,
    latency_artifact: Path,
    catalog: PlanCatalog,
) -> HistoricalResearchBundle:
    """Load R2 for historical replay without claiming online runtime compatibility."""

    quality_manifest = _read_manifest(quality_artifact)
    latency_manifest = _read_manifest(latency_artifact)
    _validate_research_pair(quality_manifest, latency_manifest, catalog=catalog)
    dependencies = installed_dependency_versions()
    quality, loaded_quality_manifest, _ = load_cost_model(
        quality_artifact,
        compatibility=_historical_context(quality_manifest, dependencies),
        require_serving_eligible=False,
    )
    latency, loaded_latency_manifest, _ = load_cost_model(
        latency_artifact,
        compatibility=_historical_context(latency_manifest, dependencies),
        require_serving_eligible=False,
    )
    return HistoricalResearchBundle(
        quality_model=quality,
        latency_model=latency,
        quality_manifest=loaded_quality_manifest,
        latency_manifest=loaded_latency_manifest,
    )


class OfflineCostAwareOptimizer:
    """Score every P0 plan while remaining physically isolated from serving."""

    def __init__(
        self,
        *,
        catalog: PlanCatalog,
        quality_model: Predictor,
        latency_model: Predictor,
        quality_manifest: CostModelArtifactManifest,
        latency_manifest: CostModelArtifactManifest,
        clock: MonotonicClock | None = None,
    ) -> None:
        _validate_research_pair(quality_manifest, latency_manifest, catalog=catalog)
        self._catalog = catalog
        self._quality_model = quality_model
        self._latency_model = latency_model
        self._quality_manifest = quality_manifest
        self._latency_manifest = latency_manifest
        self._clock = clock if clock is not None else PerfCounterClock()
        identity = {
            "optimizer_version": OFFLINE_OPTIMIZER_VERSION,
            "catalog_sha256": catalog.sha256(),
            "quality_manifest_sha256": quality_manifest.sha256,
            "latency_manifest_sha256": latency_manifest.sha256,
        }
        self._config_version = canonical_sha256(identity)
        self._model_version = (
            f"{quality_manifest.model.model_version}:{latency_manifest.model.model_version}"
        )

    @classmethod
    def from_bundle(
        cls,
        *,
        catalog: PlanCatalog,
        bundle: HistoricalResearchBundle,
        clock: MonotonicClock | None = None,
    ) -> OfflineCostAwareOptimizer:
        return cls(
            catalog=catalog,
            quality_model=bundle.quality_model,
            latency_model=bundle.latency_model,
            quality_manifest=bundle.quality_manifest,
            latency_manifest=bundle.latency_manifest,
            clock=clock,
        )

    @property
    def config_version(self) -> str:
        return self._config_version

    @property
    def model_version(self) -> str:
        return self._model_version

    def select(
        self,
        analysis: QueryAnalysis,
        *,
        context: OfflineResearchContext,
        deadline: Deadline,
    ) -> OfflineOptimizationResult:
        if analysis.features.final_top_k != 10:
            raise RAGPlanError(
                ErrorCode.INVALID_REQUEST,
                "offline cost-aware comparison supports only top_k=10",
                retryable=False,
            )
        started_ns = self._clock.now_ns()
        plans = self._catalog.p0_enabled_plans
        inputs = tuple(
            OfflineCostModelInput(
                query_features=analysis.features,
                source_dataset=context.source_dataset,
                query_tags=context.query_tags,
                plan_features=ProfilePlanFeatures.from_plan(plan),
            )
            for plan in plans
        )
        features = encode_offline_inputs(inputs)
        quality = _predict(self._quality_model, features, kind="quality")
        latency = _predict(self._latency_model, features, kind="latency")
        snapshot = deadline.snapshot()
        remaining_ms = snapshot.remaining_ms
        reserve_ms = deadline.finalization_reserve_ms
        estimates = tuple(
            self._estimate(
                plan_index=index,
                inputs=item,
                predicted_quality=float(quality[index]),
                predicted_latency=float(latency[index]),
                language_supported=analysis.language_supported,
                remaining_ms=remaining_ms,
                reserve_ms=reserve_ms,
            )
            for index, item in enumerate(inputs)
        )
        feasible = tuple(estimate for estimate in estimates if estimate.feasible)
        if feasible:
            selected_estimate = min(
                feasible,
                key=lambda estimate: (
                    -cast(float, estimate.predicted_quality),
                    *stable_tie_break_key(
                        self._catalog.plan_for_id(estimate.plan_id),
                        cast(float, estimate.predicted_p95_latency_ms),
                    ),
                ),
            )
            fallback_reason = None
            selection_reason = (
                "research-only offline optimizer selected maximum predicted Recall@10 "
                "among deadline-feasible P0 plans"
            )
        else:
            selected_estimate = next(item for item in estimates if item.plan_id == "P0")
            fallback_reason = "no_feasible_candidate_p0_best_effort"
            selection_reason = (
                "research-only offline optimizer found no feasible candidate; "
                "selected P0 best effort"
            )
        plan = self._catalog.plan_for_id(selected_estimate.plan_id)
        decision = PlannerDecision(
            mode=PlannerMode.COST_AWARE,
            effective_mode=_mode_for_plan(plan),
            selected_plan_id=plan.id,
            selected_plan=plan,
            executed_vector_top_k=plan.vector_top_k if plan.vector_enabled else None,
            executed_graph_top_k=plan.graph_top_k if plan.graph_enabled else None,
            matched_rules=("research_only_offline", "all_p0_candidates_scored"),
            remaining_budget_ms=remaining_ms,
            candidate_estimates=estimates,
            budget_feasible=selected_estimate.feasible,
            selection_reason=selection_reason,
            fallback_reason=fallback_reason,
            feature_version=COST_FEATURE_SCHEMA_VERSION,
            config_version=self._config_version,
            model_version=self._model_version,
        )
        overhead_ms = max(0.0, (self._clock.now_ns() - started_ns) / 1_000_000)
        return OfflineOptimizationResult(
            decision=decision,
            planner_overhead_ms=overhead_ms,
        )

    def _estimate(
        self,
        *,
        plan_index: int,
        inputs: OfflineCostModelInput,
        predicted_quality: float,
        predicted_latency: float,
        language_supported: bool,
        remaining_ms: float,
        reserve_ms: float,
    ) -> PlanEstimate:
        plan = self._catalog.p0_enabled_plans[plan_index]
        invalid_reason: str | None = None
        quality_value: float | None = predicted_quality
        latency_value: float | None = predicted_latency
        if not math.isfinite(predicted_quality):
            quality_value = None
            invalid_reason = "quality_prediction_non_finite"
        elif not 0.0 <= predicted_quality <= 1.0:
            quality_value = None
            invalid_reason = "quality_prediction_out_of_range"
        if not math.isfinite(predicted_latency):
            latency_value = None
            invalid_reason = invalid_reason or "latency_prediction_non_finite"
        elif predicted_latency < 0.0:
            latency_value = None
            invalid_reason = invalid_reason or "latency_prediction_negative"
        if invalid_reason is None and plan.graph_enabled and not language_supported:
            invalid_reason = "unsupported_language_graph_plan"
        if (
            invalid_reason is None
            and latency_value is not None
            and latency_value + reserve_ms > remaining_ms
        ):
            invalid_reason = "predicted_p95_plus_reserve_exceeds_remaining_budget"
        return PlanEstimate(
            plan_id=plan.id,
            predicted_quality=quality_value,
            predicted_p95_latency_ms=latency_value,
            feasible=invalid_reason is None,
            infeasible_reason=invalid_reason,
            model_version=self._model_version,
            inputs_hash=canonical_sha256(
                {
                    "feature_schema_version": COST_FEATURE_SCHEMA_VERSION,
                    "offline_input_sha256": inputs.sha256,
                    "quality_model_version": self._quality_manifest.model.model_version,
                    "latency_model_version": self._latency_manifest.model.model_version,
                }
            ),
        )


def _predict(
    estimator: Predictor,
    features: NDArray[np.float64],
    *,
    kind: str,
) -> NDArray[np.float64]:
    try:
        result = np.asarray(estimator.predict(features), dtype=np.float64)
    except Exception as exc:
        raise RAGPlanError(
            ErrorCode.MODEL_INCOMPATIBLE,
            f"offline {kind} model prediction failed",
            retryable=False,
        ) from exc
    if result.shape != (features.shape[0],):
        raise RAGPlanError(
            ErrorCode.MODEL_INCOMPATIBLE,
            f"offline {kind} model returned an invalid prediction shape",
            retryable=False,
        )
    return result


def _mode_for_plan(plan: object) -> PlannerMode:
    vector_enabled = bool(getattr(plan, "vector_enabled"))
    graph_enabled = bool(getattr(plan, "graph_enabled"))
    if vector_enabled and graph_enabled:
        return PlannerMode.FIXED_HYBRID
    if vector_enabled:
        return PlannerMode.VECTOR
    if graph_enabled:
        return PlannerMode.GRAPH
    raise RAGPlanError(
        ErrorCode.PLAN_INVARIANT_VIOLATION,
        "candidate plan has no enabled retrieval branch",
        retryable=False,
    )


def _read_manifest(path: Path) -> CostModelArtifactManifest:
    try:
        return CostModelArtifactManifest.model_validate_json(
            manifest_path_for(path).read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise RAGPlanError(
            ErrorCode.MODEL_INCOMPATIBLE,
            "offline cost-model manifest is missing or invalid",
            retryable=False,
        ) from exc


def _historical_context(
    manifest: CostModelArtifactManifest,
    dependencies: dict[str, str],
) -> CompatibilityContext:
    model = manifest.model
    return CompatibilityContext(
        feature_schema_version=model.feature_schema_version,
        plan_catalog_hash=model.plan_catalog_hash,
        corpus_version=model.corpus_version,
        qrels_version=model.qrels_version,
        embedding_model_revision=model.embedding_model_revision,
        extractor_version=model.extractor_version,
        qdrant_version=model.qdrant_version,
        neo4j_version=model.neo4j_version,
        qdrant_client_version=model.qdrant_client_version,
        runtime_fingerprint=model.runtime_fingerprint,
        runtime_semantics_version=model.runtime_semantics_version,
        hardware_fingerprint=manifest.hardware_fingerprint,
        dependency_versions=dependencies,
    )


def _validate_research_pair(
    quality: CostModelArtifactManifest,
    latency: CostModelArtifactManifest,
    *,
    catalog: PlanCatalog,
) -> None:
    if quality.kind is not ModelKind.QUALITY or latency.kind is not ModelKind.LATENCY_P95:
        raise RAGPlanError(
            ErrorCode.MODEL_INCOMPATIBLE,
            "offline cost-model bundle kinds are invalid",
            retryable=False,
        )
    if (
        quality.status is not ArtifactStatus.RESEARCH_ONLY
        or latency.status is not ArtifactStatus.RESEARCH_ONLY
    ):
        raise RAGPlanError(
            ErrorCode.MODE_UNAVAILABLE,
            "this Stage 12 path accepts research-only artifacts only",
            retryable=False,
        )
    if (
        tuple(quality.feature_names) != FEATURE_NAMES
        or tuple(latency.feature_names) != FEATURE_NAMES
    ):
        raise RAGPlanError(
            ErrorCode.MODEL_INCOMPATIBLE,
            "offline cost-model feature ordering differs from the frozen schema",
            retryable=False,
        )
    shared = (
        quality.model.feature_schema_version
        == latency.model.feature_schema_version
        == COST_FEATURE_SCHEMA_VERSION
        and quality.model.plan_catalog_hash == latency.model.plan_catalog_hash == catalog.sha256()
        and quality.training_matrix_sha256 == latency.training_matrix_sha256
        and quality.model.corpus_version == latency.model.corpus_version
        and quality.model.qrels_version == latency.model.qrels_version
        and quality.model.embedding_model_revision == latency.model.embedding_model_revision
        and quality.model.extractor_version == latency.model.extractor_version
        and quality.model.runtime_semantics_version == latency.model.runtime_semantics_version
    )
    if not shared:
        raise RAGPlanError(
            ErrorCode.MODEL_INCOMPATIBLE,
            "offline quality and latency artifacts do not form one compatible bundle",
            retryable=False,
        )


__all__ = [
    "HistoricalResearchBundle",
    "OFFLINE_OPTIMIZER_VERSION",
    "OfflineCostAwareOptimizer",
    "OfflineOptimizationResult",
    "OfflineResearchContext",
    "load_historical_research_bundle",
]
