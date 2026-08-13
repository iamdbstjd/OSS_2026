"""Stage 3 vector engine tests with a deterministic monotonic clock."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import pytest

from ragplan.backends.base import BackendHealth, BackendHealthStatus
from ragplan.core.deadline import Deadline, ManualClock
from ragplan.core.engine import GraphSearchEngine, VectorSearchEngine
from ragplan.core.errors import ErrorCode, RAGPlanError
from ragplan.core.models import (
    ActivationStatus,
    GraphSeedMatch,
    GraphTrace,
    IngestionManifest,
    IngestionStoreStatus,
    PlannerMode,
    QueryAnalysis,
    QueryFeatures,
    RetrievalHit,
    SearchRequest,
    SearchStatus,
    VectorStageManifest,
)
from ragplan.ingestion.chunker import TokenEncoding
from ragplan.planner.catalog import load_default_plan_catalog
from ragplan.retrieval.graph import GraphBackendExecution

pytestmark = pytest.mark.unit


@dataclass(frozen=True)
class FakeEncoding:
    values: tuple[str, ...]

    @property
    def token_count(self) -> int:
        return len(self.values)

    def decode(self, start: int, end: int) -> str:
        return " ".join(self.values[start:end])


class FakeTokenizer:
    def encode(self, text: str) -> TokenEncoding:
        return FakeEncoding(tuple(text.split()))


class CountingEmbedder:
    tokenizer = FakeTokenizer()

    def __init__(self, clock: ManualClock, latency_ms: float = 2.0) -> None:
        self.clock = clock
        self.latency_ms = latency_ms
        self.queries: list[str] = []

    async def embed_query(self, query: str) -> Sequence[float]:
        self.queries.append(query)
        self.clock.advance_ms(self.latency_ms)
        return (1.0, 0.0, 0.0)


class RecordingVectorBackend:
    def __init__(
        self,
        clock: ManualClock,
        hits: tuple[RetrievalHit, ...] = (),
        latency_ms: float = 3.0,
    ) -> None:
        self.clock = clock
        self.hits = hits
        self.latency_ms = latency_ms
        self.calls: list[tuple[tuple[float, ...], int, str, Deadline]] = []
        self.closed = False

    async def search(
        self,
        embedding: Sequence[float],
        top_k: int,
        corpus_version: str,
        deadline: Deadline,
    ) -> Sequence[RetrievalHit]:
        self.calls.append((tuple(embedding), top_k, corpus_version, deadline))
        self.clock.advance_ms(self.latency_ms)
        return self.hits

    async def health(self) -> BackendHealth:
        return BackendHealth(BackendHealthStatus.HEALTHY)

    async def close(self) -> None:
        self.closed = True


def _engine(
    *,
    clock: ManualClock,
    embedder: CountingEmbedder,
    backend: RecordingVectorBackend,
) -> VectorSearchEngine:
    return VectorSearchEngine(
        embedder=embedder,
        vector_backend=backend,
        plan_catalog=load_default_plan_catalog(),
        vector_stage=VectorStageManifest(
            corpus_version="sample-v1",
            collection_name="ragplan_chunks_fixture",
            chunk_count=2,
            canonical_id_checksum="a" * 64,
            embedding_set_checksum="c" * 64,
            embedding_artifact_manifest_sha256="b" * 64,
        ),
        clock=clock,
    )


@pytest.mark.asyncio
async def test_vector_engine_embeds_once_preserves_order_and_records_phase_latency() -> None:
    clock = ManualClock()
    embedder = CountingEmbedder(clock)
    hits = (
        RetrievalHit(
            canonical_chunk_id="v1:chunk:doc:0:a",
            text="first",
            score=0.9,
            source="vector",
            rank=1,
        ),
        RetrievalHit(
            canonical_chunk_id="v1:chunk:doc:1:b",
            text="second",
            score=0.8,
            source="vector",
            rank=2,
        ),
    )
    backend = RecordingVectorBackend(clock, hits)

    response = await _engine(clock=clock, embedder=embedder, backend=backend).search(
        SearchRequest(query="  relevant evidence  ", planner=PlannerMode.VECTOR, top_k=1),
        request_id="request-vector-1",
    )

    assert response.status is SearchStatus.COMPLETE
    assert [hit.text for hit in response.results] == ["first"]
    assert embedder.queries == ["relevant evidence"]
    assert len(backend.calls) == 1
    assert backend.calls[0][1:3] == (10, "sample-v1")
    assert response.planner_decision.selected_plan_id == "P0"
    assert response.planner_decision.executed_vector_top_k == 10
    assert response.trace.embedding_latency_ms == 2.0
    assert response.trace.vector_latency_ms == 3.0
    assert response.trace.total_latency_ms == 5.0
    assert "query_embedding" not in response.model_dump_json()


@pytest.mark.asyncio
async def test_vector_engine_supports_zero_hits_non_english_and_top_k_fifty() -> None:
    clock = ManualClock()
    embedder = CountingEmbedder(clock)
    backend = RecordingVectorBackend(clock)

    response = await _engine(clock=clock, embedder=embedder, backend=backend).search(
        SearchRequest(query="한국어 질문", planner=PlannerMode.VECTOR, top_k=50),
        request_id="request-vector-2",
    )

    assert response.results == ()
    assert response.status is SearchStatus.COMPLETE
    assert response.trace.language_supported is False
    assert response.planner_decision.selected_plan_id == "P1"
    assert response.planner_decision.executed_vector_top_k == 50
    assert "request-floor override" in response.planner_decision.selection_reason
    assert backend.calls[0][1] == 50


@pytest.mark.asyncio
async def test_vector_only_runtime_serves_default_rule_and_non_english_safe_plan() -> None:
    for query, expected_plan in (("What is recursion?", "P1"), ("한국어 질문", "P0")):
        clock = ManualClock()
        embedder = CountingEmbedder(clock)
        backend = RecordingVectorBackend(clock)

        response = await _engine(clock=clock, embedder=embedder, backend=backend).search(
            SearchRequest(query=query, planner=PlannerMode.RULE),
            request_id=f"rule-{expected_plan}",
        )

        assert response.planner_decision.mode is PlannerMode.RULE
        assert response.planner_decision.effective_mode is PlannerMode.VECTOR
        assert response.planner_decision.selected_plan_id == expected_plan
        assert response.planner_decision.selection_reason
        assert all(
            not estimate.feasible
            for estimate in response.planner_decision.candidate_estimates
            if estimate.plan_id in {"P4", "P5", "P6", "P8"}
        )
        assert embedder.queries == [query]
        assert len(backend.calls) == 1


@pytest.mark.asyncio
async def test_unavailable_mode_is_rejected_before_embedding() -> None:
    clock = ManualClock()
    embedder = CountingEmbedder(clock)
    backend = RecordingVectorBackend(clock)

    with pytest.raises(RAGPlanError) as error:
        await _engine(clock=clock, embedder=embedder, backend=backend).search(
            SearchRequest(query="question", planner=PlannerMode.GRAPH),
            request_id="request-vector-3",
        )

    assert error.value.code is ErrorCode.MODE_UNAVAILABLE
    assert embedder.queries == []
    assert backend.calls == []


@pytest.mark.asyncio
async def test_embedding_crossing_branch_cutoff_never_calls_qdrant() -> None:
    clock = ManualClock()
    embedder = CountingEmbedder(clock, latency_ms=191.0)
    backend = RecordingVectorBackend(clock)

    with pytest.raises(RAGPlanError) as error:
        await _engine(clock=clock, embedder=embedder, backend=backend).search(
            SearchRequest(
                query="question",
                planner=PlannerMode.VECTOR,
                latency_budget_ms=200,
            ),
            request_id="request-vector-4",
        )

    assert error.value.code is ErrorCode.DEADLINE_EXCEEDED
    assert embedder.queries == ["question"]
    assert backend.calls == []


@pytest.mark.asyncio
async def test_response_dto_finalization_cannot_overrun_absolute_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ragplan.core import engine as engine_module

    clock = ManualClock()
    embedder = CountingEmbedder(clock, latency_ms=5.0)
    backend = RecordingVectorBackend(clock, latency_ms=180.0)
    response_model = engine_module.SearchResponse

    class DelayedSearchResponse(response_model):  # type: ignore[misc, valid-type]
        def __init__(self, **data: Any) -> None:
            clock.advance_ms(20.0)
            super().__init__(**data)

    monkeypatch.setattr(engine_module, "SearchResponse", DelayedSearchResponse)

    with pytest.raises(RAGPlanError) as error:
        await _engine(clock=clock, embedder=embedder, backend=backend).search(
            SearchRequest(
                query="question",
                planner=PlannerMode.VECTOR,
                latency_budget_ms=200,
            ),
            request_id="request-vector-finalization-overrun",
        )

    assert error.value.code is ErrorCode.DEADLINE_EXCEEDED
    assert clock.now_ns() == 205_000_000


class RecordingGraphAnalyzer:
    def __init__(self, clock: ManualClock, *, supported: bool = True) -> None:
        self.clock = clock
        self.supported = supported
        self.queries: list[str] = []

    def analyze(self, query: str, *, final_top_k: int) -> QueryAnalysis:
        self.queries.append(query)
        self.clock.advance_ms(1.0)
        return QueryAnalysis(
            normalized_query=query.strip(),
            language_supported=self.supported,
            token_count=2,
            query_embedding=(),
            seed_entity_mentions=("apple",),
            seed_entity_ids=("00000000-0000-5000-8000-000000000001",),
            features=QueryFeatures(
                token_count=2,
                entity_count=1,
                entity_density=0.5,
                relation_signal=0.0,
                multi_hop_signal=0.0,
                comparison_signal=0.0,
                aggregation_signal=0.0,
                global_signal=0.0,
                final_top_k=final_top_k,
            ),
            analyzer_version="stage5-fixture-v1",
            analysis_latency_ms=1.0,
        )


class RecordingGraphBackend:
    def __init__(self, clock: ManualClock, hits: tuple[RetrievalHit, ...] = ()) -> None:
        self.clock = clock
        self.hits = hits
        self.calls: list[tuple[QueryAnalysis, object, str, Deadline]] = []
        self.closed = False

    async def search(
        self,
        query_analysis: QueryAnalysis,
        plan: object,
        corpus_version: str,
        deadline: Deadline,
    ) -> GraphBackendExecution:
        self.calls.append((query_analysis, plan, corpus_version, deadline))
        self.clock.advance_ms(4.0)
        return GraphBackendExecution(
            hits=self.hits,
            trace=GraphTrace(
                seed_matches=(
                    GraphSeedMatch(
                        mention_sha256="a" * 64,
                        requested_entity_id="00000000-0000-5000-8000-000000000001",
                        matched_entity_id="00000000-0000-5000-8000-000000000001",
                        lookup_score=1.0,
                    ),
                ),
                requested_depth=1,
                actual_depth=1,
                visited_entity_count=2,
                path_count=1,
                recovered_chunk_count=len(self.hits),
                seed_lookup_latency_ms=1.0,
                traversal_latency_ms=1.0,
                recovery_latency_ms=1.0,
                ranking_latency_ms=1.0,
            ),
        )

    async def close(self) -> None:
        self.closed = True


def _active_manifest() -> IngestionManifest:
    return IngestionManifest(
        ingestion_run_id="fixture-run-v1",
        corpus_version="active-v1",
        source_dataset="fixture",
        source_version="v1",
        source_sha256="a" * 64,
        chunker_version="chunker-v1",
        embedding_model_revision="embedding-v1",
        extractor_version="extractor-v1",
        document_count=1,
        chunk_count=1,
        qdrant_count=1,
        qdrant_id_checksum="b" * 64,
        qdrant_status=IngestionStoreStatus.SUCCEEDED,
        neo4j_count=1,
        neo4j_id_checksum="b" * 64,
        neo4j_status=IngestionStoreStatus.SUCCEEDED,
        activation_status=ActivationStatus.ACTIVE,
    )


@pytest.mark.asyncio
async def test_graph_engine_runs_explicit_plan_and_keeps_bounded_trace() -> None:
    clock = ManualClock()
    analyzer = RecordingGraphAnalyzer(clock)
    hit = RetrievalHit(
        canonical_chunk_id="v1:chunk:evidence",
        document_id="v1:document:evidence",
        text="Apple evidence",
        score=0.98,
        source="graph",
        rank=1,
    )
    backend = RecordingGraphBackend(clock, (hit,))
    engine = GraphSearchEngine(
        analyzer=analyzer,  # type: ignore[arg-type]
        graph_backend=backend,
        plan_catalog=load_default_plan_catalog(),
        active_manifest=_active_manifest(),
        clock=clock,
    )

    response = await engine.search(
        SearchRequest(query="Apple relation", planner=PlannerMode.GRAPH, top_k=10),
        request_id="request-graph-1",
    )

    assert response.status is SearchStatus.COMPLETE
    assert response.results == (hit,)
    assert response.planner_decision.selected_plan_id == "P2"
    assert response.planner_decision.executed_graph_top_k == 20
    assert response.trace.graph_latency_ms == 4.0
    assert response.trace.graph_trace is not None
    assert response.trace.graph_trace.visited_entity_count == 2
    assert response.trace.total_latency_ms == 5.0
    assert backend.calls[0][2] == "active-v1"
    serialized_trace = response.trace.model_dump_json()
    assert "Apple relation" not in serialized_trace
    assert "apple" not in serialized_trace


@pytest.mark.asyncio
async def test_graph_engine_uses_p3_request_floor_and_rejects_non_english() -> None:
    clock = ManualClock()
    backend = RecordingGraphBackend(clock)
    engine = GraphSearchEngine(
        analyzer=RecordingGraphAnalyzer(clock),  # type: ignore[arg-type]
        graph_backend=backend,
        plan_catalog=load_default_plan_catalog(),
        active_manifest=_active_manifest(),
        clock=clock,
    )

    response = await engine.search(
        SearchRequest(query="deep graph", planner=PlannerMode.GRAPH, top_k=50),
        request_id="request-graph-2",
    )

    assert response.planner_decision.selected_plan_id == "P3"
    assert response.planner_decision.executed_graph_top_k == 50
    assert "request-floor override" in response.planner_decision.selection_reason

    unsupported = GraphSearchEngine(
        analyzer=RecordingGraphAnalyzer(clock, supported=False),  # type: ignore[arg-type]
        graph_backend=backend,
        plan_catalog=load_default_plan_catalog(),
        active_manifest=_active_manifest(),
        clock=clock,
    )
    with pytest.raises(RAGPlanError) as caught:
        await unsupported.search(
            SearchRequest(query="한국어 질문", planner=PlannerMode.GRAPH),
            request_id="request-graph-3",
        )
    assert caught.value.code is ErrorCode.MODE_UNAVAILABLE
