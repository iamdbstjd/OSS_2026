"""Stage 7 engine integration over independently delayed async backend adapters."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence

import pytest

from ragplan.core.deadline import Deadline, PerfCounterClock
from ragplan.core.engine import BaselineSearchEngine
from ragplan.core.models import (
    ActivationStatus,
    BranchKind,
    BranchStatus,
    GraphStageManifest,
    GraphTrace,
    IngestionManifest,
    IngestionStoreStatus,
    PlannerMode,
    QueryAnalysis,
    QueryFeatures,
    RequestState,
    RetrievalHit,
    SearchRequest,
    SearchStatus,
    VectorStageManifest,
)
from ragplan.ingestion.chunker import TokenEncoding
from ragplan.planner.catalog import load_default_plan_catalog
from ragplan.retrieval.graph import GraphBackendExecution

pytestmark = pytest.mark.integration


class _Encoding:
    @property
    def token_count(self) -> int:
        return 2

    def decode(self, start: int, end: int) -> str:
        del start, end
        return "parallel query"


class _Tokenizer:
    def encode(self, text: str) -> TokenEncoding:
        del text
        return _Encoding()


class _Embedder:
    tokenizer = _Tokenizer()

    async def embed_query(self, query: str) -> Sequence[float]:
        del query
        return (1.0, 0.0, 0.0)


class _Analyzer:
    extractor_version = "extractor-v1"

    def analyze(self, query: str, *, final_top_k: int) -> QueryAnalysis:
        return QueryAnalysis(
            normalized_query=query,
            language_supported=True,
            token_count=2,
            query_embedding=(),
            features=QueryFeatures(
                token_count=2,
                entity_count=0,
                entity_density=0.0,
                relation_signal=0.0,
                multi_hop_signal=0.0,
                comparison_signal=0.0,
                aggregation_signal=0.0,
                global_signal=0.0,
                final_top_k=final_top_k,
            ),
            analyzer_version="integration-analyzer-v1",
            analysis_latency_ms=0.0,
        )


def _hit(branch: BranchKind) -> RetrievalHit:
    return RetrievalHit(
        canonical_chunk_id=f"{branch.value}-chunk",
        document_id=f"{branch.value}-document",
        text=f"{branch.value} evidence",
        score=1.0,
        source=branch.value,
        rank=1,
    )


class _VectorBackend:
    def __init__(self, delay_seconds: float, *, hang: bool = False) -> None:
        self.delay_seconds = delay_seconds
        self.hang = hang
        self.started = asyncio.Event()
        self.cleaned = asyncio.Event()

    async def search(
        self,
        embedding: Sequence[float],
        top_k: int,
        corpus_version: str,
        deadline: Deadline,
    ) -> Sequence[RetrievalHit]:
        del embedding, top_k, corpus_version, deadline
        self.started.set()
        try:
            if self.hang:
                await asyncio.Event().wait()
            else:
                await asyncio.sleep(self.delay_seconds)
            return (_hit(BranchKind.VECTOR),)
        finally:
            self.cleaned.set()

    async def close(self) -> None: ...


class _GraphBackend:
    def __init__(self, delay_seconds: float) -> None:
        self.delay_seconds = delay_seconds

    async def search(
        self,
        query_analysis: QueryAnalysis,
        plan: object,
        corpus_version: str,
        deadline: Deadline,
    ) -> GraphBackendExecution:
        del query_analysis, corpus_version, deadline
        await asyncio.sleep(self.delay_seconds)
        return GraphBackendExecution(
            hits=(_hit(BranchKind.GRAPH),),
            trace=GraphTrace(
                seed_matches=(),
                requested_depth=plan.graph_depth,  # type: ignore[attr-defined]
                actual_depth=0,
                visited_entity_count=0,
                path_count=0,
                recovered_chunk_count=0,
                seed_lookup_latency_ms=0.0,
                traversal_latency_ms=self.delay_seconds * 1000,
                recovery_latency_ms=0.0,
                ranking_latency_ms=0.0,
            ),
        )

    async def close(self) -> None: ...


def _engine(vector: _VectorBackend, graph: _GraphBackend) -> BaselineSearchEngine:
    active = IngestionManifest(
        ingestion_run_id="scheduler-integration",
        corpus_version="scheduler-v1",
        source_dataset="fixture",
        source_version="v1",
        source_sha256="1" * 64,
        chunker_version="chunker-v1",
        embedding_model_revision="b8903db39f65d93ae28d49a37c4f3fa90c5f94e0",
        extractor_version="extractor-v1",
        document_count=1,
        chunk_count=1,
        qdrant_count=1,
        qdrant_id_checksum="2" * 64,
        qdrant_status=IngestionStoreStatus.SUCCEEDED,
        neo4j_count=1,
        neo4j_id_checksum="2" * 64,
        neo4j_status=IngestionStoreStatus.SUCCEEDED,
        activation_status=ActivationStatus.ACTIVE,
    )
    return BaselineSearchEngine(
        embedder=_Embedder(),
        vector_backend=vector,
        analyzer=_Analyzer(),  # type: ignore[arg-type]
        graph_backend=graph,
        plan_catalog=load_default_plan_catalog(),
        active_manifest=active,
        vector_stage=VectorStageManifest(
            corpus_version="scheduler-v1",
            collection_name="scheduler-vectors",
            chunk_count=1,
            canonical_id_checksum="2" * 64,
            embedding_set_checksum="3" * 64,
            embedding_artifact_manifest_sha256="4" * 64,
        ),
        graph_stage=GraphStageManifest(
            corpus_version="scheduler-v1",
            database="neo4j",
            document_count=1,
            chunk_count=1,
            entity_count=0,
            mention_count=0,
            relation_count=0,
            canonical_id_checksum="2" * 64,
            graph_content_checksum="5" * 64,
            extractor_version="extractor-v1",
        ),
        clock=PerfCounterClock(),
    )


@pytest.mark.asyncio
async def test_fixed_hybrid_runs_30ms_and_80ms_branches_in_parallel() -> None:
    engine = _engine(_VectorBackend(0.03), _GraphBackend(0.08))

    started = time.perf_counter()
    response = await engine.search(
        SearchRequest(
            query="parallel query",
            planner=PlannerMode.FIXED_HYBRID,
            latency_budget_ms=500,
        ),
        request_id="parallel-integration",
    )
    elapsed_ms = (time.perf_counter() - started) * 1000

    branch_sum_ms = sum(item.latency_ms or 0.0 for item in response.trace.branch_results)
    assert 70 <= elapsed_ms < 140
    assert elapsed_ms < branch_sum_ms - 15
    assert response.status is SearchStatus.COMPLETE
    assert response.trace.scheduler_trace is not None
    assert response.trace.scheduler_trace.branch_start_skew_ms < 20
    assert response.trace.scheduler_trace.actual_terminal_state is RequestState.COMPLETE


@pytest.mark.asyncio
async def test_deadline_cancels_hanging_vector_and_preserves_graph_result() -> None:
    vector = _VectorBackend(0.0, hang=True)
    engine = _engine(vector, _GraphBackend(0.002))

    response = await engine.search(
        SearchRequest(
            query="parallel query",
            planner=PlannerMode.FIXED_HYBRID,
            latency_budget_ms=25,
        ),
        request_id="partial-integration",
    )

    assert response.status is SearchStatus.PARTIAL
    assert response.results[0].sources == (BranchKind.GRAPH,)
    vector_result = next(
        item for item in response.trace.branch_results if item.branch is BranchKind.VECTOR
    )
    assert vector_result.status is BranchStatus.TIMED_OUT
    assert vector.cleaned.is_set()


@pytest.mark.asyncio
async def test_engine_shutdown_cancels_and_awaits_active_backend_children() -> None:
    vector = _VectorBackend(0.0, hang=True)
    engine = _engine(vector, _GraphBackend(0.0))
    search_task = asyncio.create_task(
        engine.search(
            SearchRequest(
                query="parallel query",
                planner=PlannerMode.VECTOR,
                latency_budget_ms=5000,
            ),
            request_id="engine-shutdown",
        )
    )
    await vector.started.wait()

    await engine.close()

    with pytest.raises(asyncio.CancelledError):
        await search_task
    assert vector.cleaned.is_set()
    assert not [
        task
        for task in asyncio.all_tasks()
        if task is not asyncio.current_task()
        and not task.done()
        and task.get_name().startswith("ragplan-")
    ]
