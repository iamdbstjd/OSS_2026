"""Real Qdrant + Neo4j Stage 6 vertical integration."""

from __future__ import annotations

import os
from collections.abc import Sequence
from contextlib import suppress
from pathlib import Path
from uuid import uuid4

import pytest
from qdrant_client import AsyncQdrantClient

from ragplan.backends.graph.neo4j import (
    Neo4jGraphBackend,
    Neo4jGraphConfig,
    Neo4jGraphWriter,
)
from ragplan.backends.vector.qdrant import (
    VECTOR_SIZE,
    QdrantCollectionManager,
    QdrantVectorBackend,
    QdrantVectorConfig,
    QdrantVectorWriter,
)
from ragplan.core.deadline import PerfCounterClock
from ragplan.core.engine import BaselineSearchEngine
from ragplan.core.errors import RAGPlanError
from ragplan.core.ids import canonical_chunk_id, canonical_document_id
from ragplan.core.models import (
    ActivationStatus,
    BranchKind,
    Chunk,
    IngestionManifest,
    IngestionStoreStatus,
    PlannerMode,
    SearchRequest,
)
from ragplan.ingestion.chunker import TokenEncoding
from ragplan.ingestion.entities import EntityExtractor
from ragplan.ingestion.pipeline import extract_graph
from ragplan.planner.catalog import load_default_plan_catalog
from ragplan.retrieval.graph import GraphQueryAnalyzer

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
                document_id := canonical_document_id("stage6_fixture", str(index)),
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


def _vector(axis: int) -> tuple[float, ...]:
    return tuple(1.0 if index == axis else 0.0 for index in range(VECTOR_SIZE))


class _Encoding:
    @property
    def token_count(self) -> int:
        return 1

    def decode(self, start: int, end: int) -> str:
        return "Apple"


class _Tokenizer:
    def encode(self, text: str) -> TokenEncoding:
        return _Encoding()


class _Embedder:
    tokenizer = _Tokenizer()

    async def embed_query(self, query: str) -> Sequence[float]:
        return _vector(0)


@pytest.mark.asyncio
async def test_real_dual_store_fixed_hybrid_deduplicates_shared_chunk() -> None:
    qdrant_url, neo4j_config = _configuration()
    corpus_version = f"stage6-hybrid-{uuid4()}"
    extractor = EntityExtractor.load_pinned(lockfile=Path("uv.lock"))
    extraction = extract_graph(_chunks(corpus_version), extractor)
    qdrant_client = AsyncQdrantClient(url=qdrant_url, timeout=60)
    vector_manager = QdrantCollectionManager(
        qdrant_client,
        QdrantVectorConfig(
            collection_prefix=f"stage6_{uuid4().hex[:12]}",
            batch_size=2,
        ),
    )
    vector_writer = QdrantVectorWriter(vector_manager)
    vector_backend = QdrantVectorBackend(vector_manager)
    graph_writer = Neo4jGraphWriter.connect(neo4j_config)
    graph_backend = Neo4jGraphBackend.connect(neo4j_config)
    engine: BaselineSearchEngine | None = None
    try:
        vector_stage = await vector_writer.stage_chunks(
            extraction.chunks,
            (_vector(0), _vector(1)),
            corpus_version,
            embedding_artifact_manifest_sha256="a" * 64,
        )
        graph_stage = await graph_writer.stage_graph(
            extraction.chunks,
            extraction.entities,
            extraction.mentions,
            extraction.relations,
            corpus_version,
            extractor_version=extractor.extractor_version,
        )
        assert vector_stage.canonical_id_checksum == graph_stage.canonical_id_checksum
        active = IngestionManifest(
            ingestion_run_id="stage6-real-run-v1",
            corpus_version=corpus_version,
            source_dataset="stage6-fixture",
            source_version="v1",
            source_sha256="b" * 64,
            chunker_version="fixture-v1",
            embedding_model_revision=vector_stage.embedding_model_revision,
            extractor_version=extractor.extractor_version,
            document_count=2,
            chunk_count=2,
            qdrant_count=2,
            qdrant_id_checksum=vector_stage.canonical_id_checksum,
            qdrant_status=IngestionStoreStatus.SUCCEEDED,
            neo4j_count=2,
            neo4j_id_checksum=graph_stage.canonical_id_checksum,
            neo4j_status=IngestionStoreStatus.SUCCEEDED,
            activation_status=ActivationStatus.ACTIVE,
        )
        await graph_backend.require_active_corpus(
            corpus_version=corpus_version,
            chunk_count=2,
            canonical_id_checksum=graph_stage.canonical_id_checksum,
            extractor_version=extractor.extractor_version,
        )
        clock = PerfCounterClock()
        engine = BaselineSearchEngine(
            embedder=_Embedder(),
            vector_backend=vector_backend,
            analyzer=GraphQueryAnalyzer(extractor, clock=clock),
            graph_backend=graph_backend,
            plan_catalog=load_default_plan_catalog(),
            active_manifest=active,
            vector_stage=vector_stage,
            graph_stage=graph_stage,
            clock=clock,
        )

        response = await engine.search(
            SearchRequest(
                query="Apple",
                planner=PlannerMode.FIXED_HYBRID,
                top_k=2,
                latency_budget_ms=5000,
            ),
            request_id="stage6-real-hybrid-request",
        )

        shared = next(
            hit
            for hit in response.results
            if hit.canonical_chunk_id == extraction.chunks[0].canonical_chunk_id
        )
        assert shared.sources == (BranchKind.VECTOR, BranchKind.GRAPH)
        assert len(shared.source_contributions) == 2
        assert shared.paths
        assert response.trace.fusion_trace is not None
        assert response.trace.fusion_trace.duplicate_count >= 1
        assert response.planner_decision.selected_plan_id == "P5"
        assert response.trace.scheduler_trace is not None
        assert response.trace.scheduler_trace.runtime_semantics_version == "v1"
        assert response.trace.scheduler_trace.backend_task_count == 2
        assert all(
            branch.started_at_ms is not None and branch.ended_at_ms is not None
            for branch in response.trace.branch_results
        )
    finally:
        with suppress(RAGPlanError):
            await vector_manager.discard_version(corpus_version)
        with suppress(RAGPlanError):
            await graph_writer.discard_version(corpus_version)
        if engine is not None:
            await engine.close()
        else:
            await vector_backend.close()
            await graph_backend.close()
        await graph_writer.close()
