from __future__ import annotations

import os
from pathlib import Path
from uuid import uuid4

import pytest

from ragplan.backends.graph.neo4j import (
    Neo4jGraphBackend,
    Neo4jGraphConfig,
    Neo4jGraphWriter,
)
from ragplan.core.deadline import PerfCounterClock
from ragplan.core.engine import GraphSearchEngine
from ragplan.core.ids import (
    canonical_chunk_id,
    canonical_document_id,
    entity_id,
    entity_mention_id,
)
from ragplan.core.models import (
    ActivationStatus,
    Chunk,
    Entity,
    EntityMention,
    EntityType,
    IngestionManifest,
    IngestionStoreStatus,
    PlannerMode,
    Relation,
    RelationExtractionRule,
    SearchRequest,
)
from ragplan.ingestion.entities import EntityExtractor
from ragplan.planner.catalog import load_default_plan_catalog
from ragplan.retrieval.graph import GraphQueryAnalyzer

pytestmark = pytest.mark.integration


def _neo4j_config() -> Neo4jGraphConfig:
    uri = os.environ.get("RAGPLAN_TEST_NEO4J_URI")
    password = os.environ.get("RAGPLAN_TEST_NEO4J_PASSWORD")
    if not uri or not password:
        pytest.skip("RAGPLAN_TEST_NEO4J_URI and RAGPLAN_TEST_NEO4J_PASSWORD are required")
    return Neo4jGraphConfig(
        uri=uri,
        user=os.environ.get("RAGPLAN_TEST_NEO4J_USER", "neo4j"),
        password=password,
        batch_size=2,
        transaction_timeout_seconds=10.0,
    )


def _fixture_graph(
    corpus_version: str,
    extractor_version: str,
) -> tuple[
    tuple[Chunk, ...],
    tuple[Entity, ...],
    tuple[EntityMention, ...],
    tuple[Relation, ...],
]:
    names = ("Alice", "Bob", "Carol", "Dave")
    entities = tuple(
        Entity(
            id=str(entity_id(EntityType.PERSON.value, name)),
            name=name,
            entity_type=EntityType.PERSON,
            normalized_name=name.casefold(),
            aliases=(name,),
        )
        for name in names
    )
    chunks: list[Chunk] = []
    mentions: list[EntityMention] = []
    relations: list[Relation] = []
    for index, (source, target) in enumerate(zip(entities, entities[1:], strict=False)):
        text = f"{source.name} mentors {target.name}."
        document_id = canonical_document_id("stage5_fixture", str(index))
        chunk = Chunk(
            id=canonical_chunk_id(document_id, 0, text),
            document_id=document_id,
            corpus_version=corpus_version,
            position=0,
            text=text,
            token_count=3,
        )
        chunks.append(chunk)
        source_start = 0
        source_end = len(source.name)
        target_start = text.index(target.name)
        target_end = target_start + len(target.name)
        for token_index, (entity, start, end) in enumerate(
            (
                (source, source_start, source_end),
                (target, target_start, target_end),
            )
        ):
            mentions.append(
                EntityMention(
                    id=str(entity_mention_id(chunk.id, entity.id, start, end)),
                    entity_id=entity.id,
                    entity_type=entity.entity_type,
                    raw_text=entity.name,
                    normalized_name=entity.normalized_name,
                    source_chunk_id=chunk.id,
                    start_char=start,
                    end_char=end,
                    sentence_start_char=0,
                    sentence_end_char=len(text),
                    token_start=token_index * 2,
                    token_end=token_index * 2 + 1,
                    root_token=token_index * 2,
                )
            )
        relations.append(
            Relation(
                source_entity_id=source.id,
                target_entity_id=target.id,
                predicate="mentor",
                confidence=0.9,
                source_chunk_id=chunk.id,
                extractor_version=extractor_version,
                extraction_rule=RelationExtractionRule.DIRECT_SVO,
            )
        )
    return tuple(chunks), entities, tuple(mentions), tuple(relations)


@pytest.mark.asyncio
async def test_real_neo4j_graph_only_search_recalls_three_hop_fixture() -> None:
    config = _neo4j_config()
    corpus_version = f"stage5-integration-{uuid4()}"
    extractor = EntityExtractor.load_pinned(lockfile=Path("uv.lock"))
    chunks, entities, mentions, relations = _fixture_graph(
        corpus_version,
        extractor.extractor_version,
    )
    writer = Neo4jGraphWriter.connect(config)
    backend = Neo4jGraphBackend.connect(config)
    try:
        graph_stage = await writer.stage_graph(
            chunks,
            entities,
            mentions,
            relations,
            corpus_version,
            extractor_version=extractor.extractor_version,
        )
        await backend.require_active_corpus(
            corpus_version=corpus_version,
            chunk_count=graph_stage.chunk_count,
            canonical_id_checksum=graph_stage.canonical_id_checksum,
            extractor_version=graph_stage.extractor_version,
        )
        active = IngestionManifest(
            ingestion_run_id="stage5-real-run-v1",
            corpus_version=corpus_version,
            source_dataset="stage5-fixture",
            source_version="v1",
            source_sha256="a" * 64,
            chunker_version="fixture-v1",
            embedding_model_revision="embedding-fixture-v1",
            extractor_version=extractor.extractor_version,
            document_count=3,
            chunk_count=3,
            qdrant_count=3,
            qdrant_id_checksum=graph_stage.canonical_id_checksum,
            qdrant_status=IngestionStoreStatus.SUCCEEDED,
            neo4j_count=3,
            neo4j_id_checksum=graph_stage.canonical_id_checksum,
            neo4j_status=IngestionStoreStatus.SUCCEEDED,
            activation_status=ActivationStatus.ACTIVE,
        )
        clock = PerfCounterClock()
        engine = GraphSearchEngine(
            analyzer=GraphQueryAnalyzer(extractor, clock=clock),
            graph_backend=backend,
            plan_catalog=load_default_plan_catalog(),
            active_manifest=active,
            clock=clock,
        )

        response = await engine.search(
            SearchRequest(
                query="How is Alice connected to the final person?",
                planner=PlannerMode.GRAPH,
                top_k=21,
                latency_budget_ms=5000,
            ),
            request_id="stage5-real-graph-request",
        )

        assert {hit.canonical_chunk_id for hit in response.results} == {
            chunk.canonical_chunk_id for chunk in chunks
        }
        assert response.trace.graph_trace is not None
        assert response.trace.graph_trace.actual_depth == 3
        assert response.trace.graph_trace.visited_entity_count == 4
        deepest_paths = [
            path for hit in response.results for path in hit.paths if path.hop_count == 3
        ]
        assert deepest_paths
        assert deepest_paths[0].relations[-1].source_entity_id == entities[2].id
        assert deepest_paths[0].relations[-1].target_entity_id == entities[3].id
    finally:
        await backend.close()
        await writer.discard_version(corpus_version)
        await writer.close()
