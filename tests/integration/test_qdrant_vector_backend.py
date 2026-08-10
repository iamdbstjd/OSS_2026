"""Real-Qdrant integration coverage for the Stage 3 vector backend."""

from __future__ import annotations

import os
from uuid import UUID, uuid4

import pytest
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models

from ragplan.backends.vector.qdrant import (
    VECTOR_SIZE,
    QdrantCollectionManager,
    QdrantVectorBackend,
    QdrantVectorConfig,
    QdrantVectorWriter,
)
from ragplan.core.deadline import Deadline
from ragplan.core.errors import ErrorCode, RAGPlanError
from ragplan.core.ids import canonical_chunk_id, canonical_document_id, qdrant_point_id
from ragplan.core.models import Chunk

pytestmark = pytest.mark.integration


def _vector(axis: int) -> list[float]:
    result = [0.0] * VECTOR_SIZE
    result[axis] = 1.0
    return result


def _chunk(position: int, text: str) -> Chunk:
    document_id = canonical_document_id("integration", "qdrant-document")
    return Chunk(
        id=canonical_chunk_id(document_id, position, text),
        document_id=document_id,
        corpus_version="integration-v1",
        position=position,
        text=text,
        token_count=len(text.split()),
    )


@pytest.mark.asyncio
async def test_real_qdrant_idempotent_ingest_schema_and_relevant_search() -> None:
    qdrant_url = os.getenv("RAGPLAN_TEST_QDRANT_URL")
    if not qdrant_url:
        pytest.skip("set RAGPLAN_TEST_QDRANT_URL to run against real Qdrant")

    prefix = f"ragplan_s3_{uuid4().hex[:12]}"
    client = AsyncQdrantClient(url=qdrant_url, timeout=60)
    manager = QdrantCollectionManager(
        client,
        QdrantVectorConfig(collection_prefix=prefix, batch_size=1),
    )
    writer = QdrantVectorWriter(manager)
    backend = QdrantVectorBackend(manager)
    chunks = (
        _chunk(0, "Ada Lovelace wrote the first algorithm."),
        _chunk(1, "Oceans cover Earth."),
    )
    collection_name = manager.collection_name("integration-v1")
    try:
        first = await writer.write_chunks(
            chunks,
            (_vector(0), _vector(1)),
            "integration-v1",
            embedding_artifact_manifest_sha256="a" * 64,
        )
        second = await writer.write_chunks(
            chunks,
            (_vector(0), _vector(1)),
            "integration-v1",
            embedding_artifact_manifest_sha256="a" * 64,
        )
        stage = await writer.stage_chunks(
            chunks,
            (_vector(0), _vector(1)),
            "integration-v1",
            embedding_artifact_manifest_sha256="a" * 64,
        )
        info = await client.get_collection(collection_name)
        records, _ = await client.scroll(
            collection_name=collection_name,
            limit=10,
            with_payload=True,
            with_vectors=False,
        )
        hits = await backend.search(
            _vector(0),
            top_k=2,
            corpus_version="integration-v1",
            deadline=Deadline.start(5000),
        )

        vectors = info.config.params.vectors
        assert isinstance(vectors, models.VectorParams)
        assert vectors.size == VECTOR_SIZE
        assert vectors.distance is models.Distance.COSINE
        assert {
            field: info.payload_schema[field].data_type
            for field in (
                "corpus_version",
                "canonical_chunk_id",
                "document_id",
                "position",
                "embedding_artifact_manifest_sha256",
                "embedding_checksum",
            )
        } == {
            "corpus_version": models.PayloadSchemaType.KEYWORD,
            "canonical_chunk_id": models.PayloadSchemaType.KEYWORD,
            "document_id": models.PayloadSchemaType.KEYWORD,
            "position": models.PayloadSchemaType.INTEGER,
            "embedding_artifact_manifest_sha256": models.PayloadSchemaType.KEYWORD,
            "embedding_checksum": models.PayloadSchemaType.KEYWORD,
        }
        assert first == second
        assert first.written_count == 2
        assert len(records) == 2
        assert all(isinstance(UUID(str(record.id)), UUID) for record in records)
        assert {UUID(str(record.id)) for record in records} == {
            qdrant_point_id(chunk.canonical_chunk_id) for chunk in chunks
        }
        assert all(
            record.payload and record.payload.get("canonical_chunk_id") for record in records
        )
        assert all(
            record.payload
            and record.payload.get("embedding_artifact_manifest_sha256") == "a" * 64
            and isinstance(record.payload.get("embedding_checksum"), str)
            for record in records
        )
        assert hits[0].canonical_chunk_id == chunks[0].canonical_chunk_id
        assert hits[0].text == chunks[0].text
        assert hits[0].score > hits[1].score

        first_record = next(
            record
            for record in records
            if record.payload
            and record.payload["canonical_chunk_id"] == chunks[0].canonical_chunk_id
        )
        assert first_record.payload
        await client.upsert(
            collection_name=collection_name,
            points=[
                models.PointStruct(
                    id=qdrant_point_id(chunks[0].canonical_chunk_id),
                    vector=_vector(2),
                    payload=first_record.payload,
                )
            ],
            wait=True,
        )
        with pytest.raises(RAGPlanError) as tampering:
            await manager.verify_stage(stage)
        assert tampering.value.code is ErrorCode.CORPUS_INCONSISTENT
    finally:
        if await client.collection_exists(collection_name):
            await client.delete_collection(collection_name)
        await client.close()
