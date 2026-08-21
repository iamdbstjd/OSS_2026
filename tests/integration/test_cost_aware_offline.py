from __future__ import annotations

from pathlib import Path

import pytest

from ragplan.benchmark.contracts import SplitName
from ragplan.core.deadline import Deadline, ManualClock
from ragplan.core.models import QueryAnalysis
from ragplan.planner.catalog import load_default_plan_catalog
from ragplan.planner.optimizer import (
    OfflineCostAwareOptimizer,
    OfflineResearchContext,
    load_historical_research_bundle,
)
from ragplan.planner.training import load_training_matrix

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.integration


def test_actual_r2_bundle_runs_one_offline_decision_without_serving() -> None:
    model_dir = ROOT / "artifacts/cost_models/stage11_r2"
    matrix_path = (
        ROOT / "benchmark/results/profile_plans_audit_bypass_20260820_r2/training_matrix.jsonl"
    )
    if not model_dir.is_dir() or not matrix_path.is_file():
        pytest.skip("generated Stage 11/10 artifacts are intentionally not committed")
    catalog = load_default_plan_catalog()
    bundle = load_historical_research_bundle(
        quality_artifact=model_dir / "quality_model.skops",
        latency_artifact=model_dir / "latency_model.skops",
        catalog=catalog,
    )
    optimizer = OfflineCostAwareOptimizer.from_bundle(catalog=catalog, bundle=bundle)
    validation = tuple(
        row for row in load_training_matrix(matrix_path) if row.split is SplitName.VALIDATION
    )
    first = validation[0]
    analysis = QueryAnalysis(
        normalized_query="offline-redacted-validation-query",
        language_supported=True,
        token_count=first.query_features.token_count,
        query_embedding=(),
        features=first.query_features,
        analyzer_version="stage10-qf-v1-replay",
        analysis_latency_ms=0.0,
    )

    result = optimizer.select(
        analysis,
        context=OfflineResearchContext(
            source_dataset=first.source_dataset,
            query_tags=first.query_tags,
        ),
        deadline=Deadline.start(first.latency_budget_ms, clock=ManualClock()),
    )

    assert result.execution_mode == "research_only_offline"
    assert result.public_api_enabled is False
    assert len(result.decision.candidate_estimates) == 8
    assert result.decision.model_version == bundle.model_version
