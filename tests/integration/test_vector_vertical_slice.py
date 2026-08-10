"""End-to-end Stage 3 slice using the pinned model and real Docker Qdrant."""

from __future__ import annotations

import os
from pathlib import Path
from uuid import uuid4

import httpx
import pytest
from qdrant_client import AsyncQdrantClient

from ragplan.api.runtime import Stage3RuntimeConfig, build_search_engine
from ragplan.api.server import create_app
from ragplan.backends.vector.qdrant import (
    QdrantCollectionManager,
    QdrantVectorConfig,
    QdrantVectorWriter,
)
from ragplan.ingestion.chunker import chunk_document
from ragplan.ingestion.embedder import SentenceTransformerEmbedder
from ragplan.ingestion.model_manifest import load_default_model_artifact_manifest

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_pinned_model_ingest_and_runtime_api_search_returns_relevant_chunk(
    tmp_path: Path,
) -> None:
    qdrant_url = os.getenv("RAGPLAN_TEST_QDRANT_URL")
    snapshot_value = os.getenv("RAGPLAN_TEST_MODEL_SNAPSHOT")
    if not qdrant_url or not snapshot_value:
        pytest.skip("set Qdrant URL and exact local model snapshot integration variables")

    manifest = load_default_model_artifact_manifest()
    embedder = SentenceTransformerEmbedder.from_local_snapshot(
        snapshot_path=Path(snapshot_value),
        manifest=manifest,
        device="cpu",
    )
    corpus_version = f"stage3-real-{uuid4().hex}"
    prefix = f"ragplan_s3_e2e_{uuid4().hex[:12]}"
    documents = (
        (
            "ada",
            "Ada Lovelace wrote the first published algorithm for Charles Babbage's "
            "Analytical Engine and is often called the first computer programmer.",
        ),
        (
            "ocean",
            "The Pacific Ocean is the largest and deepest ocean on Earth.",
        ),
    )
    chunks = tuple(
        chunk
        for source_document_id, text in documents
        for chunk in chunk_document(
            source_dataset="ragplan-stage3-sample",
            source_document_id=source_document_id,
            corpus_version=corpus_version,
            text=text,
            tokenizer=embedder.tokenizer,
        )
    )
    embeddings = await embedder.embed_documents(tuple(chunk.text for chunk in chunks))
    client = AsyncQdrantClient(url=qdrant_url, timeout=60)
    manager = QdrantCollectionManager(
        client,
        QdrantVectorConfig(collection_prefix=prefix),
    )
    writer = QdrantVectorWriter(manager)
    collection_name = manager.collection_name(corpus_version)
    try:
        vector_stage = await writer.stage_chunks(
            chunks,
            embeddings,
            corpus_version,
            embedding_artifact_manifest_sha256=manifest.sha256,
        )
        stage_path = tmp_path / "vector-stage.json"
        stage_path.write_text(vector_stage.model_dump_json(), encoding="utf-8")
        await writer.close()

        runtime_config = Stage3RuntimeConfig(
            model_snapshot=Path(snapshot_value),
            vector_stage_manifest=stage_path,
            qdrant_url=qdrant_url,
            collection_prefix=prefix,
        )
        app = create_app(runtime_factory=lambda: build_search_engine(runtime_config))
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
                api_response = await http.post(
                    "/v1/search",
                    json={
                        "query": "Who wrote the first computer algorithm?",
                        "top_k": 2,
                        "latency_budget_ms": 5000,
                        "planner": "vector",
                    },
                    headers={"x-request-id": "stage3-real-integration"},
                )

        assert api_response.status_code == 200
        response = api_response.json()
        assert len(chunks) == 2
        assert vector_stage.status == "vector_staged"
        assert response["results"][0]["canonical_chunk_id"] == chunks[0].canonical_chunk_id
        assert "ada lovelace" in response["results"][0]["text"].casefold()
        assert response["trace"]["embedding_latency_ms"] > 0
        assert response["trace"]["vector_latency_ms"] is not None
        assert response["trace"]["total_latency_ms"] >= response["trace"]["vector_latency_ms"]
        assert response["trace"]["corpus_version"] == corpus_version
        assert response["request_id"] == "stage3-real-integration"
    finally:
        await writer.close()
        cleanup_client = AsyncQdrantClient(url=qdrant_url, timeout=60)
        try:
            if await cleanup_client.collection_exists(collection_name):
                await cleanup_client.delete_collection(collection_name)
        finally:
            await cleanup_client.close()
