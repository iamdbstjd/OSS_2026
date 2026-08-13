from __future__ import annotations

import pytest

from ragplan.benchmark.synthetic import build_synthetic_graph_fixture
from ragplan.core.models import Relation, RelationExtractionRule
from ragplan.retrieval.graph import (
    MAX_PATHS_PER_SEED,
    MAX_VISITED_ENTITIES,
    EdgeBatch,
    GraphTraversalEdge,
    traverse_bounded,
)

pytestmark = pytest.mark.benchmark


@pytest.mark.asyncio
async def test_all_supported_synthetic_gold_paths_are_recalled() -> None:
    fixture = build_synthetic_graph_fixture()
    gold_chunk_by_edge = {
        (edge.source_entity, edge.relation, edge.target_entity): chunk_id
        for query in fixture.queries
        for edge, chunk_id in zip(query.gold_path, query.relevant_chunk_ids, strict=True)
    }
    relations = tuple(
        Relation(
            source_entity_id=edge.source_entity,
            target_entity_id=edge.target_entity,
            predicate=edge.relation,
            confidence=0.9,
            source_chunk_id=gold_chunk_by_edge.get(
                (edge.source_entity, edge.relation, edge.target_entity),
                f"synthetic:nuisance:{index:04d}",
            ),
            extractor_version="synthetic-v1",
            extraction_rule=RelationExtractionRule.DIRECT_SVO,
        )
        for index, edge in enumerate(fixture.edges)
    )

    recalled = 0
    for query in fixture.queries:

        async def load_edges(frontier: tuple[str, ...]) -> EdgeBatch:
            return EdgeBatch(
                tuple(
                    GraphTraversalEdge(relation)
                    for relation in relations
                    if relation.source_entity_id in frontier
                    or relation.target_entity_id in frontier
                )
            )

        outcome = await traverse_bounded(
            (query.start_entity,),
            requested_depth=query.hop_count,
            load_edges=load_edges,
        )
        recovered_provenance = {
            relation.source_chunk_id for path in outcome.paths for relation in path.relations
        }
        if set(query.relevant_chunk_ids) <= recovered_provenance:
            recalled += 1
        assert outcome.visited_entity_count <= MAX_VISITED_ENTITIES
        assert len(outcome.paths) <= MAX_PATHS_PER_SEED
        assert all(len(set(path.entity_ids)) == len(path.entity_ids) for path in outcome.paths)

    assert recalled / len(fixture.queries) == 1.0
