from __future__ import annotations

import os
from pathlib import Path
from uuid import uuid4

import pytest

from ragplan.backends.graph.neo4j import Neo4jGraphConfig, Neo4jGraphWriter
from ragplan.core.ids import canonical_chunk_id, canonical_document_id
from ragplan.core.models import Chunk
from ragplan.ingestion.entities import EntityExtractor
from ragplan.ingestion.pipeline import extract_graph

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


def _chunks(corpus_version: str) -> tuple[Chunk, ...]:
    texts = (
        "Apple acquired Beats Electronics.",
        "Paris is the capital of France.",
    )
    chunks = []
    for index, text in enumerate(texts):
        document_id = canonical_document_id("stage4_fixture", str(index))
        chunks.append(
            Chunk(
                id=canonical_chunk_id(document_id, 0, text),
                document_id=document_id,
                corpus_version=corpus_version,
                position=0,
                text=text,
                token_count=len(text.split()),
            )
        )
    return tuple(chunks)


@pytest.mark.asyncio
async def test_real_neo4j_ingestion_is_verified_and_idempotent() -> None:
    corpus_version = f"stage4-integration-{uuid4()}"
    extraction = extract_graph(
        _chunks(corpus_version),
        EntityExtractor.load_pinned(lockfile=Path("uv.lock")),
    )
    writer = Neo4jGraphWriter.connect(_neo4j_config())
    try:
        first = await writer.stage_graph(
            extraction.chunks,
            extraction.entities,
            extraction.mentions,
            extraction.relations,
            corpus_version,
            extractor_version=extraction.extractor_version,
        )
        second = await writer.stage_graph(
            extraction.chunks,
            extraction.entities,
            extraction.mentions,
            extraction.relations,
            corpus_version,
            extractor_version=extraction.extractor_version,
        )

        assert first == second
        assert first.chunk_count == len(extraction.chunks)
        assert first.canonical_id_checksum
        assert first.relation_count == 2
    finally:
        await writer.discard_version(corpus_version)
        await writer.close()
