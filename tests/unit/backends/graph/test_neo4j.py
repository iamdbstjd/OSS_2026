from __future__ import annotations

from typing import Any

import pytest

from ragplan.backends.graph.neo4j import Neo4jGraphConfig, Neo4jGraphWriter
from ragplan.core.errors import ErrorCode, RAGPlanError
from ragplan.core.ids import (
    canonical_chunk_id,
    canonical_document_id,
    entity_id,
    entity_mention_id,
)
from ragplan.core.models import (
    Chunk,
    Entity,
    EntityMention,
    EntityType,
    Relation,
    RelationExtractionRule,
)

pytestmark = pytest.mark.unit


class _FakeNeo4jDriver:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object], str, float | None]] = []
        self.marker: dict[str, object] | None = None
        self.documents: set[str] = set()
        self.chunks: set[str] = set()
        self.entities: set[str] = set()
        self.mentions: set[str] = set()
        self.relations: set[str] = set()
        self.closed = False

    async def execute_query(
        self,
        query: Any,
        *,
        parameters_: dict[str, object],
        database_: str,
    ) -> tuple[list[dict[str, object]], None, list[str]]:
        statement = str(query)
        self.calls.append((statement, parameters_, database_, query.timeout))
        rows = parameters_.get("rows")
        row_list = rows if isinstance(rows, list) else []
        if statement.startswith("CREATE "):
            return [], None, []
        if "MERGE (version:CorpusVersion" in statement:
            if self.marker is None:
                self.marker = {
                    "status": "writing",
                    "graph_content_checksum": parameters_["graph_content_checksum"],
                    "extractor_version": parameters_["extractor_version"],
                }
            return [dict(self.marker)], None, []
        if "MERGE (document:Document" in statement:
            self.documents.update(str(item["id"]) for item in row_list if isinstance(item, dict))
            return [], None, []
        if "MERGE (chunk:Chunk" in statement:
            self.chunks.update(str(item["id"]) for item in row_list if isinstance(item, dict))
            return [], None, []
        if "MERGE (entity:Entity" in statement:
            self.entities.update(str(item["id"]) for item in row_list if isinstance(item, dict))
            return [], None, []
        if "MERGE (chunk)-[mention:MENTIONS" in statement:
            self.mentions.update(str(item["id"]) for item in row_list if isinstance(item, dict))
            return [], None, []
        if "MERGE (source)-[relation:RELATES_TO" in statement:
            self.relations.update(str(item["id"]) for item in row_list if isinstance(item, dict))
            return [], None, []
        if "SET version.status = 'graph_staged'" in statement:
            assert self.marker is not None
            self.marker.update(parameters_)
            self.marker["status"] = "graph_staged"
            return [{"status": "graph_staged"}], None, []
        if "RETURN version.status AS status" in statement and "MATCH" in statement:
            return ([] if self.marker is None else [dict(self.marker)]), None, []
        if "RETURN chunk.id AS canonical_chunk_id" in statement:
            return (
                [{"canonical_chunk_id": item} for item in sorted(self.chunks)],
                None,
                [],
            )
        if "count(document)" in statement:
            return [{"count": len(self.documents)}], None, []
        if "count(DISTINCT entity)" in statement:
            return [{"count": len(self.entities)}], None, []
        if "count(mention)" in statement:
            return [{"count": len(self.mentions)}], None, []
        if "count(relation)" in statement:
            return [{"count": len(self.relations)}], None, []
        if "SET version.status = 'failed'" in statement:
            if self.marker is not None and self.marker.get("status") != "graph_staged":
                self.marker["status"] = "failed"
            return [], None, []
        return [], None, []

    async def verify_connectivity(self) -> None:
        return None

    async def close(self) -> None:
        self.closed = True


def _graph() -> tuple[
    tuple[Chunk, ...],
    tuple[Entity, ...],
    tuple[EntityMention, ...],
    tuple[Relation, ...],
]:
    text = "Apple acquired Beats."
    document_id = canonical_document_id("fixture", "one")
    chunk = Chunk(
        id=canonical_chunk_id(document_id, 0, text),
        document_id=document_id,
        corpus_version="fixture-v1",
        position=0,
        text=text,
        token_count=4,
    )
    entity_specs = (("Apple", EntityType.ORG, 0, 5), ("Beats", EntityType.ORG, 15, 20))
    entities: list[Entity] = []
    mentions: list[EntityMention] = []
    for token_index, (name, entity_type, start, end) in enumerate(entity_specs):
        normalized = name.casefold()
        entity_uuid = str(entity_id(entity_type.value, normalized))
        entities.append(
            Entity(
                id=entity_uuid,
                name=name,
                entity_type=entity_type,
                normalized_name=normalized,
                aliases=(name,),
            )
        )
        mentions.append(
            EntityMention(
                id=str(entity_mention_id(chunk.id, entity_uuid, start, end)),
                entity_id=entity_uuid,
                entity_type=entity_type,
                raw_text=name,
                normalized_name=normalized,
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
    relation = Relation(
        source_entity_id=entities[0].id,
        target_entity_id=entities[1].id,
        predicate="acquire",
        confidence=0.9,
        source_chunk_id=chunk.id,
        extractor_version="graph-extractor-v1-fixture",
        extraction_rule=RelationExtractionRule.DIRECT_SVO,
    )
    return (chunk,), tuple(entities), tuple(mentions), (relation,)


@pytest.mark.asyncio
async def test_graph_writer_is_parameterized_idempotent_and_timeout_bounded() -> None:
    driver = _FakeNeo4jDriver()
    writer = Neo4jGraphWriter(
        driver,
        Neo4jGraphConfig(
            password="test-only",
            batch_size=1,
            transaction_timeout_seconds=2.5,
        ),
    )
    chunks, entities, mentions, relations = _graph()

    first = await writer.stage_graph(
        chunks,
        entities,
        mentions,
        relations,
        "fixture-v1",
        extractor_version="graph-extractor-v1-fixture",
    )
    write_calls_after_first = sum("UNWIND $rows" in statement for statement, *_ in driver.calls)
    second = await writer.stage_graph(
        chunks,
        entities,
        mentions,
        relations,
        "fixture-v1",
        extractor_version="graph-extractor-v1-fixture",
    )

    assert first == second
    assert first.chunk_count == 1
    assert first.entity_count == 2
    assert first.mention_count == 2
    assert first.relation_count == 1
    assert sum("UNWIND $rows" in statement for statement, *_ in driver.calls) == (
        write_calls_after_first
    )
    assert all(timeout == 2.5 for _, _, _, timeout in driver.calls)
    assert all(database == "neo4j" for _, _, database, _ in driver.calls)
    assert all(chunks[0].text not in statement for statement, *_ in driver.calls)
    assert any(
        chunks[0].text in str(parameters)
        for statement, parameters, *_ in driver.calls
        if "MERGE (chunk:Chunk" in statement
    )
    assert all(
        "$rows" in statement
        for statement, parameters, *_ in driver.calls
        if isinstance(parameters.get("rows"), list)
    )

    recovered = await writer.recover_stage("fixture-v1")
    assert recovered == first


@pytest.mark.asyncio
async def test_same_version_with_changed_graph_content_is_rejected_before_write() -> None:
    driver = _FakeNeo4jDriver()
    writer = Neo4jGraphWriter(driver, Neo4jGraphConfig(password="test-only"))
    chunks, entities, mentions, relations = _graph()
    await writer.stage_graph(
        chunks,
        entities,
        mentions,
        relations,
        "fixture-v1",
        extractor_version="graph-extractor-v1-fixture",
    )
    changed = (relations[0].model_copy(update={"confidence": 0.8}),)
    before = sum("UNWIND $rows" in statement for statement, *_ in driver.calls)

    with pytest.raises(RAGPlanError) as caught:
        await writer.stage_graph(
            chunks,
            entities,
            mentions,
            changed,
            "fixture-v1",
            extractor_version="graph-extractor-v1-fixture",
        )

    assert caught.value.code is ErrorCode.CORPUS_INCONSISTENT
    assert sum("UNWIND $rows" in statement for statement, *_ in driver.calls) == before
