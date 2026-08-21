from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

from ragplan.core.models import (
    BranchKind,
    BranchResult,
    BranchStatus,
    PlanDefinition,
    PlannerDecision,
    PlannerMode,
    QueryFeatures,
    RetrievalHit,
    SearchResponse,
    SearchStatus,
    SearchTrace,
)

pytestmark = pytest.mark.unit


def _module() -> ModuleType:
    path = Path(__file__).resolve().parents[2] / "examples/llm_handoff.py"
    spec = importlib.util.spec_from_file_location("ragplan_example_llm_handoff", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _response() -> SearchResponse:
    plan = PlanDefinition(
        id="P0",
        name="VECTOR_FAST",
        vector_enabled=True,
        graph_enabled=False,
        vector_top_k=10,
        graph_top_k=0,
        graph_depth=0,
        vector_weight=1.0,
        graph_weight=0.0,
        rerank_enabled=False,
        enabled_in_p0=True,
    )
    decision = PlannerDecision(
        mode=PlannerMode.VECTOR,
        effective_mode=PlannerMode.VECTOR,
        selected_plan_id="P0",
        selected_plan=plan,
        executed_vector_top_k=10,
        remaining_budget_ms=100.0,
        feature_version="v1",
        config_version="v1",
    )
    features = QueryFeatures(
        token_count=2,
        entity_count=0,
        entity_density=0.0,
        relation_signal=0.0,
        multi_hop_signal=0.0,
        comparison_signal=0.0,
        aggregation_signal=0.0,
        global_signal=0.0,
        final_top_k=10,
    )
    hit = RetrievalHit(
        canonical_chunk_id="v1:chunk:example:1",
        text="Ada wrote notes about the Analytical Engine.",
        score=1.0,
        source="vector",
        rank=1,
    )
    trace = SearchTrace(
        request_id="example-request",
        query_hash="a" * 64,
        query_length=8,
        language_supported=True,
        features=features,
        planner_decision=decision,
        branch_results=(
            BranchResult(
                branch=BranchKind.VECTOR,
                status=BranchStatus.SUCCEEDED,
                latency_ms=1.0,
                hits=(hit,),
            ),
        ),
        analyzer_latency_ms=0.0,
        planner_latency_ms=0.0,
        vector_latency_ms=1.0,
        total_latency_ms=1.0,
        latency_budget_ms=200,
        finalization_reserve_ms=10.0,
        budget_feasible=True,
        budget_violated=False,
        fallback=False,
        result_count=1,
        corpus_version="example-v1",
        config_version="v1",
        model_version="model-v1",
    )
    return SearchResponse(
        status=SearchStatus.COMPLETE,
        results=(hit,),
        planner_decision=decision,
        trace=trace,
        fallback=False,
        request_id="example-request",
    )


def test_generic_handoff_preserves_ranked_evidence_without_provider_dependency() -> None:
    module = _module()
    messages = module.build_llm_messages("What did Ada write?", _response())

    assert messages[0]["role"] == "system"
    assert "untrusted data" in messages[0]["content"]
    assert "chunk=v1:chunk:example:1" in messages[1]["content"]
    assert "Analytical Engine" in messages[1]["content"]
    assert "openai" not in Path(module.__file__).read_text(encoding="utf-8").casefold()
