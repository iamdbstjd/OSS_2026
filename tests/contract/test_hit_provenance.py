"""Public Stage 6 hit and request provenance contract."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ragplan.core.models import (
    BranchKind,
    PlannerMode,
    RetrievalContribution,
    RetrievalHit,
    SearchRequest,
)

pytestmark = pytest.mark.contract


def test_fused_hit_provenance_survives_public_json_serialization() -> None:
    contribution = RetrievalContribution(
        source=BranchKind.VECTOR,
        original_rank=2,
        original_score=0.87,
        weight=0.5,
        rrf_contribution=0.5 / 62,
        metadata={"native_metric": "cosine"},
    )
    hit = RetrievalHit(
        canonical_chunk_id="v1:chunk:doc:0:digest",
        document_id="v1:document:doc",
        text="ranked evidence",
        score=0.5 / 62,
        source="fusion",
        rank=1,
        sources=(BranchKind.VECTOR,),
        source_contributions=(contribution,),
    )

    payload = hit.model_dump(mode="json")
    assert payload["sources"] == ["vector"]
    assert payload["source_contributions"] == [
        {
            "source": "vector",
            "original_rank": 2,
            "original_score": 0.87,
            "weight": 0.5,
            "rrf_contribution": 0.5 / 62,
            "metadata": {"native_metric": "cosine"},
        }
    ]


def test_fused_hit_rejects_misaligned_source_summary() -> None:
    with pytest.raises(ValidationError):
        RetrievalHit(
            canonical_chunk_id="chunk",
            text="evidence",
            score=1.0,
            source="fusion",
            sources=(BranchKind.GRAPH,),
            source_contributions=(
                RetrievalContribution(
                    source=BranchKind.VECTOR,
                    original_rank=1,
                    original_score=1.0,
                    weight=1.0,
                    rrf_contribution=1 / 61,
                ),
            ),
        )


def test_fixed_hybrid_plan_request_contract_is_closed() -> None:
    for plan_id in (None, "P4", "P5", "P6", "P8"):
        request = SearchRequest(
            query="question",
            planner=PlannerMode.FIXED_HYBRID,
            plan_id=plan_id,
        )
        assert request.plan_id == plan_id

    with pytest.raises(ValidationError):
        SearchRequest(query="question", planner=PlannerMode.FIXED_HYBRID, plan_id="P7")
    with pytest.raises(ValidationError):
        SearchRequest(query="question", planner=PlannerMode.VECTOR, plan_id="P4")
