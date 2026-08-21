from __future__ import annotations

import math

import numpy as np
import pytest
from numpy.typing import NDArray

from ragplan.benchmark.contracts import QueryTag, SourceDataset
from ragplan.core.deadline import Deadline, ManualClock
from ragplan.core.errors import ErrorCode, RAGPlanError
from ragplan.core.models import ModelManifest, PlannerMode, QueryAnalysis, QueryFeatures
from ragplan.planner.artifacts import (
    ArtifactStatus,
    CostModelArtifactManifest,
    ModelKind,
)
from ragplan.planner.catalog import PlanCatalog, load_default_plan_catalog
from ragplan.planner.optimizer import OfflineCostAwareOptimizer, OfflineResearchContext
from ragplan.planner.training import COST_FEATURE_SCHEMA_VERSION, FEATURE_NAMES

pytestmark = pytest.mark.unit


class FixedPredictor:
    def __init__(self, values: list[float]) -> None:
        self.values = np.asarray(values, dtype=np.float64)

    def predict(self, features: NDArray[np.float64]) -> object:
        assert features.shape == (8, len(FEATURE_NAMES))
        return self.values.copy()


def _manifest(kind: ModelKind, catalog: PlanCatalog) -> CostModelArtifactManifest:
    name = "quality" if kind is ModelKind.QUALITY else "latency"
    return CostModelArtifactManifest(
        kind=kind,
        status=ArtifactStatus.RESEARCH_ONLY,
        model=ModelManifest(
            model_name=name,
            model_version=f"{name}-fixture-v1",
            artifact_version="cost_model_artifact_v1",
            artifact_sha256=("1" if kind is ModelKind.QUALITY else "2") * 64,
            feature_schema_version=COST_FEATURE_SCHEMA_VERSION,
            plan_catalog_hash=catalog.sha256(),
            corpus_version="fixture-corpus-v1",
            qrels_version="qrels-v1",
            embedding_model_revision="embedding-v1",
            extractor_version="extractor-v1",
            qdrant_version="1.18.2",
            neo4j_version="5.26.28",
            qdrant_client_version="1.18.0",
            training_config_hash="3" * 64,
            train_validation_split_hash="4" * 64,
            runtime_fingerprint="fixture-runtime-v1",
            runtime_semantics_version="v1",
            validation_metrics={"metric": 0.0},
        ),
        feature_names=FEATURE_NAMES,
        training_matrix_sha256="5" * 64,
        training_row_count=100,
        validation_row_count=20,
        hardware_fingerprint="6" * 64,
        dependency_versions={},
        gate_failures=("fixture_research_gate",),
    )


def _optimizer(
    quality: list[float],
    latency: list[float],
    *,
    clock: ManualClock | None = None,
) -> OfflineCostAwareOptimizer:
    catalog = load_default_plan_catalog()
    return OfflineCostAwareOptimizer(
        catalog=catalog,
        quality_model=FixedPredictor(quality),
        latency_model=FixedPredictor(latency),
        quality_manifest=_manifest(ModelKind.QUALITY, catalog),
        latency_manifest=_manifest(ModelKind.LATENCY_P95, catalog),
        clock=clock,
    )


def _analysis(*, top_k: int = 10, supported: bool = True) -> QueryAnalysis:
    return QueryAnalysis(
        normalized_query="offline fixture",
        language_supported=supported,
        token_count=8,
        query_embedding=(),
        features=QueryFeatures(
            token_count=8,
            entity_count=2,
            entity_density=0.25,
            relation_signal=0.5,
            multi_hop_signal=0.2,
            comparison_signal=0.0,
            aggregation_signal=0.0,
            global_signal=0.0,
            final_top_k=top_k,
        ),
        analyzer_version="fixture-qf-v1",
        analysis_latency_ms=1.0,
    )


def _context() -> OfflineResearchContext:
    return OfflineResearchContext(
        source_dataset=SourceDataset.HOTPOT_BRIDGE,
        query_tags=(QueryTag.TWO_HOP, QueryTag.RELATIONSHIP),
    )


def test_scores_all_candidates_and_selects_highest_quality_feasible() -> None:
    optimizer = _optimizer(
        [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80],
        [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0],
    )
    result = optimizer.select(
        _analysis(),
        context=_context(),
        deadline=Deadline.start(100, clock=ManualClock()),
    )

    decision = result.decision
    assert decision.mode is PlannerMode.COST_AWARE
    assert decision.selected_plan_id == "P8"
    assert decision.budget_feasible is True
    assert tuple(item.plan_id for item in decision.candidate_estimates) == (
        "P0",
        "P1",
        "P2",
        "P3",
        "P4",
        "P5",
        "P6",
        "P8",
    )
    assert all(
        item.model_version == optimizer.model_version for item in decision.candidate_estimates
    )
    assert len({item.inputs_hash for item in decision.candidate_estimates}) == 8
    assert result.public_api_enabled is False
    assert result.planner_overhead_ms >= 0.0


def test_feasibility_uses_remaining_budget_plus_reserve_and_p0_best_effort() -> None:
    clock = ManualClock()
    deadline = Deadline.start(100, clock=clock)
    clock.advance_ms(96)
    optimizer = _optimizer([0.5] * 8, [1.0] * 8)

    result = optimizer.select(_analysis(), context=_context(), deadline=deadline)

    assert result.decision.selected_plan_id == "P0"
    assert result.decision.budget_feasible is False
    assert result.decision.fallback_reason == "no_feasible_candidate_p0_best_effort"
    assert all(not item.feasible for item in result.decision.candidate_estimates)
    assert {item.infeasible_reason for item in result.decision.candidate_estimates} == {
        "predicted_p95_plus_reserve_exceeds_remaining_budget"
    }


def test_equal_quality_uses_latency_depth_and_plan_id_tie_break() -> None:
    optimizer = _optimizer(
        [0.1, 0.1, 0.9, 0.9, 0.2, 0.2, 0.2, 0.2],
        [10.0, 11.0, 40.0, 40.0, 20.0, 20.0, 20.0, 20.0],
    )

    result = optimizer.select(
        _analysis(),
        context=_context(),
        deadline=Deadline.start(200, clock=ManualClock()),
    )

    assert result.decision.selected_plan_id == "P2"


def test_invalid_predictions_are_visible_and_never_selected() -> None:
    quality = [0.1, 0.2, math.nan, 1.2, 0.5, 0.6, 0.7, 0.8]
    latency = [10.0, 20.0, 30.0, 40.0, math.inf, -1.0, 70.0, 80.0]
    result = _optimizer(quality, latency).select(
        _analysis(),
        context=_context(),
        deadline=Deadline.start(200, clock=ManualClock()),
    )
    estimates = {item.plan_id: item for item in result.decision.candidate_estimates}

    assert estimates["P2"].infeasible_reason == "quality_prediction_non_finite"
    assert estimates["P3"].infeasible_reason == "quality_prediction_out_of_range"
    assert estimates["P4"].infeasible_reason == "latency_prediction_non_finite"
    assert estimates["P5"].infeasible_reason == "latency_prediction_negative"
    assert result.decision.selected_plan_id == "P8"


def test_non_english_marks_every_graph_candidate_infeasible() -> None:
    result = _optimizer([0.5] * 8, [10.0] * 8).select(
        _analysis(supported=False),
        context=_context(),
        deadline=Deadline.start(100, clock=ManualClock()),
    )

    assert result.decision.selected_plan_id == "P0"
    assert all(
        item.infeasible_reason == "unsupported_language_graph_plan"
        for item in result.decision.candidate_estimates
        if item.plan_id in {"P2", "P3", "P4", "P5", "P6", "P8"}
    )


def test_unsupported_top_k_and_incompatible_bundle_fail_closed() -> None:
    optimizer = _optimizer([0.5] * 8, [10.0] * 8)
    with pytest.raises(RAGPlanError) as top_k:
        optimizer.select(
            _analysis(top_k=5),
            context=_context(),
            deadline=Deadline.start(100, clock=ManualClock()),
        )
    assert top_k.value.code is ErrorCode.INVALID_REQUEST

    catalog = load_default_plan_catalog()
    bad_latency = _manifest(ModelKind.LATENCY_P95, catalog)
    bad_model = bad_latency.model.model_copy(update={"plan_catalog_hash": "f" * 64})
    with pytest.raises(RAGPlanError) as incompatible:
        OfflineCostAwareOptimizer(
            catalog=catalog,
            quality_model=FixedPredictor([0.5] * 8),
            latency_model=FixedPredictor([10.0] * 8),
            quality_manifest=_manifest(ModelKind.QUALITY, catalog),
            latency_manifest=bad_latency.model_copy(update={"model": bad_model}),
        )
    assert incompatible.value.code is ErrorCode.MODEL_INCOMPATIBLE
