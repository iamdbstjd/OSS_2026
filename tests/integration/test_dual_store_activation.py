from __future__ import annotations

import os
from pathlib import Path
from uuid import uuid4

import pytest
from qdrant_client import AsyncQdrantClient

from ragplan.backends.graph.neo4j import Neo4jGraphConfig, Neo4jGraphWriter
from ragplan.backends.vector.qdrant import (
    QdrantCollectionManager,
    QdrantVectorConfig,
    QdrantVectorWriter,
)
from ragplan.core.ids import canonical_chunk_id, canonical_document_id
from ragplan.core.models import Chunk
from ragplan.ingestion.entities import EntityExtractor
from ragplan.ingestion.manifest import ActiveCorpusResolver, ManifestRepository
from ragplan.ingestion.pipeline import extract_graph
from ragplan.ingestion.reconcile import ActivationCoordinator, IngestionSource

pytestmark = pytest.mark.integration


def _configuration() -> tuple[str, Neo4jGraphConfig]:
    qdrant_url = os.environ.get("RAGPLAN_TEST_QDRANT_URL")
    neo4j_uri = os.environ.get("RAGPLAN_TEST_NEO4J_URI")
    neo4j_password = os.environ.get("RAGPLAN_TEST_NEO4J_PASSWORD")
    if not qdrant_url or not neo4j_uri or not neo4j_password:
        pytest.skip("real Qdrant and Neo4j integration environment is required")
    return qdrant_url, Neo4jGraphConfig(
        uri=neo4j_uri,
        user=os.environ.get("RAGPLAN_TEST_NEO4J_USER", "neo4j"),
        password=neo4j_password,
        batch_size=2,
        transaction_timeout_seconds=10.0,
    )


def _chunks(corpus_version: str) -> tuple[Chunk, ...]:
    texts = ("Apple acquired Beats Electronics.", "Paris is the capital of France.")
    return tuple(
        Chunk(
            id=canonical_chunk_id(
                document_id := canonical_document_id("stage4_dual_fixture", str(index)),
                0,
                text,
            ),
            document_id=document_id,
            corpus_version=corpus_version,
            position=0,
            text=text,
            token_count=len(text.split()),
        )
        for index, text in enumerate(texts)
    )


def _embedding(axis: int) -> tuple[float, ...]:
    return tuple(1.0 if index == axis else 0.0 for index in range(384))


@pytest.mark.asyncio
async def test_real_dual_store_reconciliation_atomically_activates(tmp_path: Path) -> None:
    qdrant_url, neo4j_config = _configuration()
    corpus_version = f"stage4-dual-{uuid4()}"
    chunks = _chunks(corpus_version)
    qdrant = AsyncQdrantClient(url=qdrant_url)
    vector_manager = QdrantCollectionManager(
        qdrant,
        QdrantVectorConfig(collection_prefix="stage4_dual", batch_size=2),
    )
    vector_writer = QdrantVectorWriter(vector_manager)
    graph_writer = Neo4jGraphWriter.connect(neo4j_config)
    try:
        vector_stage = await vector_writer.stage_chunks(
            chunks,
            (_embedding(0), _embedding(1)),
            corpus_version,
            embedding_artifact_manifest_sha256="a" * 64,
        )
        extraction = extract_graph(
            chunks,
            EntityExtractor.load_pinned(lockfile=Path("uv.lock")),
        )
        graph_stage = await graph_writer.stage_graph(
            extraction.chunks,
            extraction.entities,
            extraction.mentions,
            extraction.relations,
            corpus_version,
            extractor_version=extraction.extractor_version,
        )
        repository = ManifestRepository(tmp_path / "ingestion")
        coordinator = ActivationCoordinator(
            vector_verifier=vector_manager,
            graph_verifier=graph_writer,
            repository=repository,
        )

        pointer, manifest = await coordinator.activate(
            ingestion_run_id="real-dual-run-v1",
            source=IngestionSource(
                source_dataset="stage4-dual-fixture",
                source_version="v1",
                source_sha256="b" * 64,
                chunker_version="fixture-v1",
            ),
            vector=vector_stage,
            graph=graph_stage,
        )

        assert pointer.corpus_version == corpus_version
        assert manifest.qdrant_id_checksum == manifest.neo4j_id_checksum
        assert ActiveCorpusResolver(repository).resolve() == corpus_version
    finally:
        await vector_manager.discard_version(corpus_version)
        await graph_writer.discard_version(corpus_version)
        await graph_writer.close()
        await qdrant.close()
