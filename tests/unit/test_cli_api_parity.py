from __future__ import annotations

from collections.abc import Sequence

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
from ragplan.cli import services
from ragplan.core.engine import VectorSearchEngine
from ragplan.core.ids import canonical_chunk_id, canonical_document_id
from ragplan.core.models import Chunk, PlannerMode, SearchRequest, SearchResponse
from ragplan.ingestion.chunker import TokenEncoding
from ragplan.planner.catalog import load_default_plan_catalog

pytestmark = pytest.mark.unit


class _Encoding:
    @property
    def token_count(self) -> int:
        return 2

    def decode(self, start: int, end: int) -> str:
        del start, end
        return "parity query"


class _Tokenizer:
    def encode(self, text: str) -> TokenEncoding:
        del text
        return _Encoding()


class _Embedder:
    tokenizer = _Tokenizer()

    async def embed_query(self, query: str) -> Sequence[float]:
        del query
        return (1.0, *([0.0] * (VECTOR_SIZE - 1)))


@pytest.mark.asyncio
async def test_cli_service_and_api_return_the_same_plan_and_result_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = AsyncQdrantClient(location=":memory:")
    manager = QdrantCollectionManager(
        client,
        QdrantVectorConfig(collection_prefix="parity", require_payload_indexes=False),
    )
    writer = QdrantVectorWriter(manager)
    backend = QdrantVectorBackend(manager)
    document_id = canonical_document_id("parity", "document")
    chunk = Chunk(
        id=canonical_chunk_id(document_id, 0, "same evidence"),
        document_id=document_id,
        corpus_version="parity-v1",
        position=0,
        text="same evidence",
        token_count=2,
    )
    stage = await writer.stage_chunks(
        (chunk,),
        ((1.0, *([0.0] * (VECTOR_SIZE - 1))),),
        "parity-v1",
        embedding_artifact_manifest_sha256="a" * 64,
    )
    engine = VectorSearchEngine(
        embedder=_Embedder(),
        vector_backend=backend,
        plan_catalog=load_default_plan_catalog(),
        vector_stage=stage,
    )
    request = SearchRequest(query="find the same evidence", planner=PlannerMode.VECTOR, top_k=1)
    app = create_app(search_engine=engine)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
        api_raw = await http.post(
            "/v1/search",
            content=request.model_dump_json(),
            headers={"content-type": "application/json", "x-request-id": "api-parity"},
        )
    api_response = SearchResponse.model_validate_json(api_raw.content)

    async def configured_engine(environment: object = None) -> VectorSearchEngine:
        del environment
        return engine

    monkeypatch.setattr(services, "build_search_engine_from_environment", configured_engine)
    cli_response = await services.search_configured_runtime(request, request_id="cli-parity")

    assert api_response.planner_decision.selected_plan_id == (
        cli_response.planner_decision.selected_plan_id
    )
    assert tuple(item.canonical_chunk_id for item in api_response.results) == tuple(
        item.canonical_chunk_id for item in cli_response.results
    )
