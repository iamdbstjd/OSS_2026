"""HTTP wiring for the executable Stage 3 vector engine."""

from __future__ import annotations

import httpx
import pytest
from qdrant_client import AsyncQdrantClient

from ragplan.api.server import create_app
from ragplan.backends.vector.qdrant import (
    VECTOR_SIZE,
    QdrantCollectionManager,
    QdrantVectorBackend,
    QdrantVectorConfig,
    QdrantVectorWriter,
)
from ragplan.core.engine import VectorSearchEngine
from ragplan.core.ids import canonical_chunk_id, canonical_document_id
from ragplan.core.models import Chunk
from ragplan.ingestion.chunker import TokenEncoding
from ragplan.planner.catalog import load_default_plan_catalog

pytestmark = pytest.mark.unit


class _Encoding:
    @property
    def token_count(self) -> int:
        return 2

    def decode(self, start: int, end: int) -> str:
        return "query"


class _Tokenizer:
    def encode(self, text: str) -> TokenEncoding:
        del text
        return _Encoding()


class _Embedder:
    tokenizer = _Tokenizer()

    def __init__(self) -> None:
        self.calls = 0

    async def embed_query(self, query: str) -> tuple[float, ...]:
        del query
        self.calls += 1
        return (1.0, *([0.0] * (VECTOR_SIZE - 1)))


@pytest.mark.asyncio
async def test_injected_engine_serves_vector_search_and_lifespan_closes_it() -> None:
    client = AsyncQdrantClient(location=":memory:")
    manager = QdrantCollectionManager(
        client,
        QdrantVectorConfig(collection_prefix="api_stage3", require_payload_indexes=False),
    )
    writer = QdrantVectorWriter(manager)
    backend = QdrantVectorBackend(manager)
    document_id = canonical_document_id("api", "document")
    chunk = Chunk(
        id=canonical_chunk_id(document_id, 0, "executable evidence"),
        document_id=document_id,
        corpus_version="api-v1",
        position=0,
        text="executable evidence",
        token_count=2,
    )
    vector = (1.0, *([0.0] * (VECTOR_SIZE - 1)))
    stage = await writer.stage_chunks(
        (chunk,),
        (vector,),
        "api-v1",
        embedding_artifact_manifest_sha256="a" * 64,
    )
    embedder = _Embedder()
    engine = VectorSearchEngine(
        embedder=embedder,
        vector_backend=backend,
        plan_catalog=load_default_plan_catalog(),
        vector_stage=stage,
    )
    app = create_app(search_engine=engine)

    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
            response = await http.post(
                "/v1/search",
                json={"query": "find evidence", "planner": "vector", "top_k": 1},
                headers={"x-request-id": "stage3-http-request"},
            )

        assert response.status_code == 200
        body = response.json()
        assert body["request_id"] == "stage3-http-request"
        assert body["results"][0]["text"] == "executable evidence"
        assert body["trace"]["embedding_latency_ms"] >= 0
        assert body["trace"]["vector_latency_ms"] >= 0
        assert embedder.calls == 1

    health_after_shutdown = await backend.health()
    assert health_after_shutdown.available is False
