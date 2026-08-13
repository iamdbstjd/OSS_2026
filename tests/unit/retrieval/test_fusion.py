"""Exact Stage 6 weighted-RRF and canonical deduplication tests."""

from __future__ import annotations

import pytest

from ragplan.core.errors import ErrorCode, RAGPlanError
from ragplan.core.models import BranchKind, GraphPath, Relation, RetrievalHit
from ragplan.retrieval.fusion import MAX_GRAPH_PATHS_PER_HIT, RRF_K, weighted_rrf_v1

pytestmark = pytest.mark.unit


def _hit(
    canonical_id: str,
    *,
    source: str,
    score: float,
    text: str | None = None,
    document_id: str = "doc-1",
) -> RetrievalHit:
    return RetrievalHit(
        canonical_chunk_id=canonical_id,
        text=text or canonical_id,
        score=score,
        source=source,
        document_id=document_id,
        metadata={"native_score": score},
    )


def test_weighted_rrf_matches_hand_calculation_and_merges_duplicate() -> None:
    vector = (
        _hit("shared", source="vector", score=0.91, text="same"),
        _hit("vector-only", source="vector", score=0.80),
    )
    graph = (
        _hit("graph-only", source="graph", score=0.95),
        _hit("shared", source="graph", score=0.70, text="same"),
    )

    result = weighted_rrf_v1(
        vector_hits=vector,
        graph_hits=graph,
        vector_weight=0.5,
        graph_weight=0.5,
        top_k=3,
    )

    assert [hit.canonical_chunk_id for hit in result.hits] == [
        "shared",
        "graph-only",
        "vector-only",
    ]
    assert result.hits[0].score == pytest.approx(0.5 / 61 + 0.5 / 62)
    assert result.hits[0].sources == (BranchKind.VECTOR, BranchKind.GRAPH)
    assert [item.original_rank for item in result.hits[0].source_contributions] == [1, 2]
    assert [item.original_score for item in result.hits[0].source_contributions] == [
        0.91,
        0.70,
    ]
    assert result.trace.rrf_k == RRF_K
    assert result.trace.duplicate_count == 1
    assert result.trace.output_count == 3


@pytest.mark.parametrize(
    ("vector_weight", "graph_weight", "expected"),
    [
        (1.0, 0.0, ["v", "g"]),
        (0.0, 1.0, ["g", "v"]),
        (0.5, 0.5, ["g", "v"]),
    ],
)
def test_weights_and_canonical_tie_break_are_deterministic(
    vector_weight: float,
    graph_weight: float,
    expected: list[str],
) -> None:
    result = weighted_rrf_v1(
        vector_hits=(_hit("v", source="vector", score=99.0),),
        graph_hits=(_hit("g", source="graph", score=-99.0),),
        vector_weight=vector_weight,
        graph_weight=graph_weight,
        top_k=2,
    )

    assert [hit.canonical_chunk_id for hit in result.hits] == expected


def test_missing_branch_preserves_successful_branch_order_even_at_zero_weight() -> None:
    vector = (
        _hit("z", source="vector", score=0.1),
        _hit("a", source="vector", score=100.0),
    )

    result = weighted_rrf_v1(
        vector_hits=vector,
        graph_hits=None,
        vector_weight=0.0,
        graph_weight=1.0,
        top_k=2,
    )

    assert [hit.canonical_chunk_id for hit in result.hits] == ["z", "a"]
    assert result.trace.missing_branches == (BranchKind.GRAPH,)
    assert all(hit.score == 0.0 for hit in result.hits)


def test_successful_zero_hit_branches_are_not_marked_missing() -> None:
    result = weighted_rrf_v1(
        vector_hits=(),
        graph_hits=(),
        vector_weight=0.5,
        graph_weight=0.5,
        top_k=10,
    )

    assert result.hits == ()
    assert result.trace.missing_branches == ()


@pytest.mark.parametrize(
    ("graph_text", "graph_document_id"),
    [("different", "doc-1"), ("same", "doc-2")],
)
def test_same_id_with_conflicting_evidence_fails_closed(
    graph_text: str,
    graph_document_id: str,
) -> None:
    vector = _hit("shared", source="vector", score=1.0, text="same")
    graph = _hit(
        "shared",
        source="graph",
        score=1.0,
        text=graph_text,
        document_id=graph_document_id,
    )

    with pytest.raises(RAGPlanError) as error:
        weighted_rrf_v1(
            vector_hits=(vector,),
            graph_hits=(graph,),
            vector_weight=0.5,
            graph_weight=0.5,
            top_k=1,
        )

    assert error.value.code is ErrorCode.CORPUS_INCONSISTENT


def test_same_text_with_distinct_ids_is_not_merged_and_provenance_serializes() -> None:
    result = weighted_rrf_v1(
        vector_hits=(_hit("v", source="vector", score=1.0, text="same"),),
        graph_hits=(_hit("g", source="graph", score=1.0, text="same"),),
        vector_weight=0.5,
        graph_weight=0.5,
        top_k=2,
    )

    payload = result.hits[0].model_dump(mode="json")
    assert len(result.hits) == 2
    assert payload["sources"]
    assert payload["source_contributions"][0]["original_rank"] == 1
    assert "rrf_contribution" in payload["source_contributions"][0]


def test_duplicate_inside_one_branch_is_a_consistency_error() -> None:
    duplicate = _hit("same", source="vector", score=1.0)
    with pytest.raises(RAGPlanError) as error:
        weighted_rrf_v1(
            vector_hits=(duplicate, duplicate),
            graph_hits=(),
            vector_weight=0.5,
            graph_weight=0.5,
            top_k=2,
        )
    assert error.value.code is ErrorCode.CORPUS_INCONSISTENT


def test_conflicting_shared_document_metadata_fails_closed() -> None:
    vector = _hit("shared", source="vector", score=1.0).model_copy(
        update={"metadata": {"title": "first"}}
    )
    graph = _hit("shared", source="graph", score=1.0).model_copy(
        update={"metadata": {"title": "different"}}
    )

    with pytest.raises(RAGPlanError) as error:
        weighted_rrf_v1(
            vector_hits=(vector,),
            graph_hits=(graph,),
            vector_weight=0.5,
            graph_weight=0.5,
            top_k=1,
        )
    assert error.value.code is ErrorCode.CORPUS_INCONSISTENT


def test_graph_paths_are_deduplicated_sorted_and_capped() -> None:
    paths = tuple(
        GraphPath(
            entity_ids=(f"source-{index}", f"target-{index}"),
            relations=(
                Relation(
                    source_entity_id=f"source-{index}",
                    target_entity_id=f"target-{index}",
                    predicate=f"predicate-{index}",
                    confidence=0.9,
                    source_chunk_id="shared",
                    extractor_version="extractor-v1",
                ),
            ),
            score=float(index),
        )
        for index in range(55)
    )
    graph = _hit("shared", source="graph", score=1.0).model_copy(
        update={"paths": (*paths, paths[-1])}
    )

    result = weighted_rrf_v1(
        vector_hits=None,
        graph_hits=(graph,),
        vector_weight=0.5,
        graph_weight=0.5,
        top_k=1,
    )

    retained = result.hits[0].paths
    assert len(retained) == MAX_GRAPH_PATHS_PER_HIT
    assert [path.score for path in retained] == list(reversed([float(i) for i in range(5, 55)]))


def test_both_missing_branches_produce_empty_auditable_result() -> None:
    result = weighted_rrf_v1(
        vector_hits=None,
        graph_hits=None,
        vector_weight=0.5,
        graph_weight=0.5,
        top_k=10,
    )
    assert result.hits == ()
    assert result.trace.missing_branches == (BranchKind.VECTOR, BranchKind.GRAPH)


def test_declared_rank_must_match_branch_order() -> None:
    hit = _hit("ranked", source="vector", score=1.0).model_copy(update={"rank": 2})
    with pytest.raises(RAGPlanError) as error:
        weighted_rrf_v1(
            vector_hits=(hit,),
            graph_hits=(),
            vector_weight=0.5,
            graph_weight=0.5,
            top_k=1,
        )
    assert error.value.code is ErrorCode.CORPUS_INCONSISTENT
