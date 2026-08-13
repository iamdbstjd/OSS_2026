from __future__ import annotations

from pathlib import Path

import pytest

from ragplan.core.deadline import ManualClock
from ragplan.core.models import Relation, RelationExtractionRule
from ragplan.ingestion.entities import EntityExtractor
from ragplan.retrieval.graph import (
    MAX_PATHS_PER_SEED,
    MAX_VISITED_ENTITIES,
    EdgeBatch,
    GraphQueryAnalyzer,
    GraphTraversalEdge,
    GraphTraversalPath,
    RecoveredGraphChunk,
    rank_graph_chunks,
    traverse_bounded,
)

pytestmark = pytest.mark.unit


def _relation(source: str, target: str, *, confidence: float = 0.9) -> Relation:
    return Relation(
        source_entity_id=source,
        target_entity_id=target,
        predicate=f"rel-{source}-{target}",
        confidence=confidence,
        source_chunk_id=f"v1:chunk:{source}-{target}",
        extractor_version="fixture-v1",
        extraction_rule=RelationExtractionRule.DIRECT_SVO,
    )


def _path(
    seed: str,
    entities: tuple[str, ...],
    confidences: tuple[float, ...],
) -> GraphTraversalPath:
    return GraphTraversalPath(
        seed_entity_id=seed,
        entity_ids=entities,
        relations=tuple(
            _relation(source, target, confidence=confidence)
            for source, target, confidence in zip(
                entities[:-1], entities[1:], confidences, strict=True
            )
        ),
    )


def test_graph_score_v0_is_exact_and_ties_use_canonical_chunk_id() -> None:
    direct = _path("seed", ("seed", "answer"), (0.8,))
    chunks = (
        RecoveredGraphChunk(
            canonical_chunk_id="v1:chunk:b",
            document_id="doc-b",
            text="second canonical ID",
            entity_ids=("answer",),
            paths=(direct,),
        ),
        RecoveredGraphChunk(
            canonical_chunk_id="v1:chunk:a",
            document_id="doc-a",
            text="first canonical ID",
            entity_ids=("answer",),
            paths=(direct,),
        ),
    )

    hits = rank_graph_chunks(chunks, matched_seed_ids=("seed",), top_k=10)

    assert [hit.canonical_chunk_id for hit in hits] == ["v1:chunk:a", "v1:chunk:b"]
    assert hits[0].score == pytest.approx(0.45 + 0.35 + 0.20 * 0.8)
    assert hits[0].metadata["seed_overlap_contribution"] == pytest.approx(0.45)
    assert hits[0].metadata["hop_contribution"] == pytest.approx(0.35)
    assert hits[0].metadata["confidence_contribution"] == pytest.approx(0.16)
    assert [hit.rank for hit in hits] == [1, 2]


def test_graph_score_uses_seed_overlap_hop_and_mean_confidence() -> None:
    two_hop = _path("seed-a", ("seed-a", "middle", "answer"), (0.8, 1.0))
    chunk = RecoveredGraphChunk(
        canonical_chunk_id="v1:chunk:answer",
        document_id="doc-answer",
        text="two hop evidence",
        entity_ids=("answer",),
        paths=(two_hop,),
    )

    hit = rank_graph_chunks(
        (chunk,),
        matched_seed_ids=("seed-a", "seed-b"),
        top_k=1,
    )[0]

    assert hit.score == pytest.approx(0.45 * 0.5 + 0.35 * 0.5 + 0.20 * 0.9)
    assert hit.paths[0].entity_ids == ("seed-a", "middle", "answer")
    assert hit.paths[0].score == pytest.approx(hit.score)


def test_query_analyzer_reuses_pinned_ingestion_entity_pipeline() -> None:
    extractor = EntityExtractor.load_pinned(lockfile=Path("uv.lock"))
    analyzer = GraphQueryAnalyzer(extractor, clock=ManualClock())

    analysis = analyzer.analyze(
        "How is Apple related to Beats Electronics?",
        final_top_k=10,
    )

    assert analysis.seed_entity_mentions == ("apple", "beats electronics")
    assert len(analysis.seed_entity_ids) == 2
    assert analysis.features.entity_count == 2
    assert analysis.query_embedding == ()
    assert extractor.extractor_version in analysis.analyzer_version


@pytest.mark.asyncio
async def test_bounded_traversal_finds_one_two_three_hops_and_preserves_direction() -> None:
    relations = (
        _relation("a", "b"),
        _relation("c", "b"),
        _relation("c", "d"),
        _relation("d", "a"),
    )

    async def load_edges(frontier: tuple[str, ...]) -> EdgeBatch:
        edges = tuple(
            GraphTraversalEdge(relation)
            for relation in relations
            if relation.source_entity_id in frontier or relation.target_entity_id in frontier
        )
        return EdgeBatch(edges)

    outcome = await traverse_bounded(("a",), requested_depth=3, load_edges=load_edges)

    entities = {path.entity_ids for path in outcome.paths}
    assert ("a", "b") in entities
    assert ("a", "b", "c") in entities
    assert ("a", "b", "c", "d") in entities
    assert outcome.actual_depth == 3
    assert outcome.visited_entity_count == 4
    reverse_discovery = next(path for path in outcome.paths if path.entity_ids == ("a", "b", "c"))
    assert reverse_discovery.relations[1].source_entity_id == "c"
    assert reverse_discovery.relations[1].target_entity_id == "b"
    assert all(len(set(path.entity_ids)) == len(path.entity_ids) for path in outcome.paths)


@pytest.mark.asyncio
async def test_hub_traversal_respects_path_and_visited_caps() -> None:
    relations = tuple(_relation("seed", f"neighbor-{index:03d}") for index in range(600))

    async def load_edges(frontier: tuple[str, ...]) -> EdgeBatch:
        assert frontier == ("seed",)
        return EdgeBatch(tuple(GraphTraversalEdge(item) for item in relations), truncated=True)

    outcome = await traverse_bounded(("seed",), requested_depth=1, load_edges=load_edges)

    assert len(outcome.paths) == MAX_PATHS_PER_SEED
    assert outcome.visited_entity_count == MAX_VISITED_ENTITIES
    assert outcome.path_limit_hit is True
    assert outcome.visited_limit_hit is True


@pytest.mark.asyncio
async def test_no_seed_does_not_call_storage() -> None:
    calls = 0

    async def load_edges(frontier: tuple[str, ...]) -> EdgeBatch:
        nonlocal calls
        calls += 1
        return EdgeBatch(())

    outcome = await traverse_bounded((), requested_depth=3, load_edges=load_edges)

    assert outcome.paths == ()
    assert outcome.visited_entity_count == 0
    assert calls == 0
