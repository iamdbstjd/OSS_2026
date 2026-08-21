"""Stage 6 shared-engine and fixed-hybrid execution tests."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from ragplan.backends.base import BackendHealth, BackendHealthStatus
from ragplan.core.deadline import Deadline, ManualClock
from ragplan.core.engine import BaselineSearchEngine
from ragplan.core.errors import ErrorCode, RAGPlanError
from ragplan.core.models import (
    ActivationStatus,
    BranchKind,
    BranchStatus,
    GraphStageManifest,
    GraphTrace,
    IngestionManifest,
    IngestionStoreStatus,
    KillSwitch,
    PlannerMode,
    QueryAnalysis,
    QueryFeatures,
    RequestState,
    RetrievalHit,
    SearchRequest,
    SearchResponse,
    SearchStatus,
    VectorStageManifest,
)
from ragplan.ingestion.audit import AuditStatus, RuleGraphTierPolicy
from ragplan.ingestion.chunker import TokenEncoding
from ragplan.planner.catalog import load_default_plan_catalog
from ragplan.planner.rule import RulePlanner, load_default_rule_planner_config
from ragplan.retrieval.graph import GraphBackendExecution
from ragplan.scheduler.states import KillSwitchSnapshot

pytestmark = pytest.mark.unit


class _Encoding:
    @property
    def token_count(self) -> int:
        return 1

    def decode(self, start: int, end: int) -> str:
        return "question"


class _Tokenizer:
    def encode(self, text: str) -> TokenEncoding:
        return _Encoding()


class _Embedder:
    tokenizer = _Tokenizer()

    def __init__(self, clock: ManualClock) -> None:
        self.clock = clock
        self.calls = 0

    async def embed_query(self, query: str) -> Sequence[float]:
        self.calls += 1
        self.clock.advance_ms(2.0)
        return (1.0, 0.0, 0.0)


class _Analyzer:
    extractor_version = "extractor-v1"

    def __init__(self, clock: ManualClock) -> None:
        self.clock = clock
        self.calls = 0

    def analyze(self, query: str, *, final_top_k: int) -> QueryAnalysis:
        self.calls += 1
        self.clock.advance_ms(1.0)
        return QueryAnalysis(
            normalized_query=query.strip(),
            language_supported=True,
            token_count=1,
            query_embedding=(),
            features=QueryFeatures(
                token_count=1,
                entity_count=0,
                entity_density=0.0,
                relation_signal=0.0,
                multi_hop_signal=0.0,
                comparison_signal=0.0,
                aggregation_signal=0.0,
                global_signal=0.0,
                final_top_k=final_top_k,
            ),
            analyzer_version="test-analyzer-v1",
            analysis_latency_ms=1.0,
        )


class _VectorBackend:
    def __init__(self, clock: ManualClock) -> None:
        self.clock = clock
        self.calls: list[tuple[int, str]] = []
        self.closed = False

    async def search(
        self,
        embedding: Sequence[float],
        top_k: int,
        corpus_version: str,
        deadline: Deadline,
    ) -> Sequence[RetrievalHit]:
        self.calls.append((top_k, corpus_version))
        self.clock.advance_ms(3.0)
        return (
            RetrievalHit(
                canonical_chunk_id="shared",
                document_id="doc",
                text="same evidence",
                score=0.9,
                source="vector",
                rank=1,
            ),
            RetrievalHit(
                canonical_chunk_id="vector-only",
                document_id="doc",
                text="vector evidence",
                score=0.8,
                source="vector",
                rank=2,
            ),
        )

    async def health(self) -> BackendHealth:
        return BackendHealth(BackendHealthStatus.HEALTHY)

    async def close(self) -> None:
        self.closed = True


class _GraphBackend:
    def __init__(self, clock: ManualClock) -> None:
        self.clock = clock
        self.calls: list[tuple[str, int, int]] = []
        self.closed = False

    async def search(
        self,
        query_analysis: QueryAnalysis,
        plan: object,
        corpus_version: str,
        deadline: Deadline,
    ) -> GraphBackendExecution:
        graph_plan = plan
        self.calls.append(
            (corpus_version, graph_plan.graph_top_k, graph_plan.graph_depth)  # type: ignore[attr-defined]
        )
        self.clock.advance_ms(4.0)
        return GraphBackendExecution(
            hits=(
                RetrievalHit(
                    canonical_chunk_id="shared",
                    document_id="doc",
                    text="same evidence",
                    score=0.7,
                    source="graph",
                    rank=1,
                    metadata={"graph": True},
                ),
            ),
            trace=GraphTrace(
                seed_matches=(),
                requested_depth=graph_plan.graph_depth,  # type: ignore[attr-defined]
                actual_depth=0,
                visited_entity_count=0,
                path_count=0,
                recovered_chunk_count=0,
                seed_lookup_latency_ms=0.0,
                traversal_latency_ms=0.0,
                recovery_latency_ms=0.0,
                ranking_latency_ms=0.0,
            ),
        )

    async def close(self) -> None:
        self.closed = True


def _active() -> IngestionManifest:
    return IngestionManifest(
        ingestion_run_id="run-v1",
        corpus_version="corpus-v1",
        source_dataset="fixture",
        source_version="v1",
        source_sha256="1" * 64,
        chunker_version="chunker-v1",
        embedding_model_revision="b8903db39f65d93ae28d49a37c4f3fa90c5f94e0",
        extractor_version="extractor-v1",
        document_count=1,
        chunk_count=2,
        qdrant_count=2,
        qdrant_id_checksum="2" * 64,
        qdrant_status=IngestionStoreStatus.SUCCEEDED,
        neo4j_count=2,
        neo4j_id_checksum="2" * 64,
        neo4j_status=IngestionStoreStatus.SUCCEEDED,
        activation_status=ActivationStatus.ACTIVE,
    )


def _vector_stage() -> VectorStageManifest:
    return VectorStageManifest(
        corpus_version="corpus-v1",
        collection_name="collection-v1",
        chunk_count=2,
        canonical_id_checksum="2" * 64,
        embedding_set_checksum="3" * 64,
        embedding_artifact_manifest_sha256="4" * 64,
    )


def _graph_stage() -> GraphStageManifest:
    return GraphStageManifest(
        corpus_version="corpus-v1",
        database="neo4j",
        document_count=1,
        chunk_count=2,
        entity_count=0,
        mention_count=0,
        relation_count=0,
        canonical_id_checksum="2" * 64,
        graph_content_checksum="5" * 64,
        extractor_version="extractor-v1",
    )


def _engine(clock: ManualClock) -> tuple[BaselineSearchEngine, _VectorBackend, _GraphBackend]:
    vector = _VectorBackend(clock)
    graph = _GraphBackend(clock)
    engine = BaselineSearchEngine(
        embedder=_Embedder(clock),
        vector_backend=vector,
        analyzer=_Analyzer(clock),  # type: ignore[arg-type]
        graph_backend=graph,
        plan_catalog=load_default_plan_catalog(),
        active_manifest=_active(),
        vector_stage=_vector_stage(),
        graph_stage=_graph_stage(),
        clock=clock,
    )
    return engine, vector, graph


def _enabled_rule_planner() -> RulePlanner:
    return RulePlanner(
        catalog=load_default_plan_catalog(),
        graph_policy=RuleGraphTierPolicy(
            audit_sample_checksum="a" * 64,
            audit_status=AuditStatus.COMPLETE,
            observed_entity_f1=0.80,
            observed_relation_precision=0.75,
            graph_tier_enabled=True,
            reason="gates passed",
        ),
        config=load_default_rule_planner_config(),
    )


class _FailingVectorBackend(_VectorBackend):
    def __init__(
        self,
        clock: ManualClock,
        code: ErrorCode = ErrorCode.DEPENDENCY_UNAVAILABLE,
    ) -> None:
        super().__init__(clock)
        self.code = code

    async def search(
        self,
        embedding: Sequence[float],
        top_k: int,
        corpus_version: str,
        deadline: Deadline,
    ) -> Sequence[RetrievalHit]:
        del embedding, deadline
        self.calls.append((top_k, corpus_version))
        raise RAGPlanError(self.code, "vector branch failed")


class _FailingGraphBackend(_GraphBackend):
    def __init__(
        self,
        clock: ManualClock,
        code: ErrorCode = ErrorCode.DEPENDENCY_UNAVAILABLE,
    ) -> None:
        super().__init__(clock)
        self.code = code

    async def search(
        self,
        query_analysis: QueryAnalysis,
        plan: object,
        corpus_version: str,
        deadline: Deadline,
    ) -> GraphBackendExecution:
        del query_analysis, deadline
        self.calls.append(
            (corpus_version, plan.graph_top_k, plan.graph_depth)  # type: ignore[attr-defined]
        )
        raise RAGPlanError(self.code, "graph branch failed")


class _EmptyVectorBackend(_VectorBackend):
    async def search(
        self,
        embedding: Sequence[float],
        top_k: int,
        corpus_version: str,
        deadline: Deadline,
    ) -> Sequence[RetrievalHit]:
        del embedding, deadline
        self.calls.append((top_k, corpus_version))
        return ()


class _EmptyGraphBackend(_GraphBackend):
    async def search(
        self,
        query_analysis: QueryAnalysis,
        plan: object,
        corpus_version: str,
        deadline: Deadline,
    ) -> GraphBackendExecution:
        del query_analysis, deadline
        self.calls.append(
            (corpus_version, plan.graph_top_k, plan.graph_depth)  # type: ignore[attr-defined]
        )
        return GraphBackendExecution(
            hits=(),
            trace=GraphTrace(
                seed_matches=(),
                requested_depth=plan.graph_depth,  # type: ignore[attr-defined]
                actual_depth=0,
                visited_entity_count=0,
                path_count=0,
                recovered_chunk_count=0,
                seed_lookup_latency_ms=0.0,
                traversal_latency_ms=0.0,
                recovery_latency_ms=0.0,
                ranking_latency_ms=0.0,
            ),
        )


def _custom_engine(
    clock: ManualClock,
    *,
    vector: _VectorBackend,
    graph: _GraphBackend,
    switches: KillSwitchSnapshot | None = None,
) -> BaselineSearchEngine:
    return BaselineSearchEngine(
        embedder=_Embedder(clock),
        vector_backend=vector,
        analyzer=_Analyzer(clock),  # type: ignore[arg-type]
        graph_backend=graph,
        plan_catalog=load_default_plan_catalog(),
        active_manifest=_active(),
        vector_stage=_vector_stage(),
        graph_stage=_graph_stage(),
        clock=clock,
        kill_switch_provider=(lambda: switches or KillSwitchSnapshot()),
    )


@pytest.mark.asyncio
async def test_fixed_hybrid_defaults_to_p5_and_emits_exact_provenance() -> None:
    engine, vector, graph = _engine(ManualClock())

    response = await engine.search(
        SearchRequest(query="question", planner=PlannerMode.FIXED_HYBRID, top_k=2),
        request_id="hybrid-request-1",
    )

    assert response.planner_decision.selected_plan_id == "P5"
    assert vector.calls == [(20, "corpus-v1")]
    assert graph.calls == [("corpus-v1", 20, 1)]
    assert [hit.canonical_chunk_id for hit in response.results] == [
        "shared",
        "vector-only",
    ]
    shared = response.results[0]
    assert shared.sources == (BranchKind.VECTOR, BranchKind.GRAPH)
    assert [item.original_rank for item in shared.source_contributions] == [1, 1]
    assert shared.score == pytest.approx(1.0 / 61)
    assert response.trace.fusion_trace is not None
    assert response.trace.fusion_trace.duplicate_count == 1
    assert response.trace.fusion_latency_ms == 0.0
    assert response.trace.total_latency_ms == 10.0
    round_trip = SearchResponse.model_validate_json(response.model_dump_json())
    assert round_trip.results[0].source_contributions == shared.source_contributions
    assert round_trip.trace.fusion_trace == response.trace.fusion_trace


@pytest.mark.asyncio
async def test_same_engine_serves_vector_graph_and_all_fixed_presets() -> None:
    engine, _, _ = _engine(ManualClock())
    vector_response = await engine.search(
        SearchRequest(query="question", planner=PlannerMode.VECTOR),
        request_id="vector-request",
    )
    graph_response = await engine.search(
        SearchRequest(query="question", planner=PlannerMode.GRAPH),
        request_id="graph-request",
    )
    assert vector_response.planner_decision.selected_plan_id == "P0"
    assert graph_response.planner_decision.selected_plan_id == "P2"
    assert vector_response.results[0].sources == (BranchKind.VECTOR,)
    assert graph_response.results[0].sources == (BranchKind.GRAPH,)

    for plan_id in ("P4", "P5", "P6", "P8"):
        response = await engine.search(
            SearchRequest(
                query="question",
                planner=PlannerMode.FIXED_HYBRID,
                plan_id=plan_id,
            ),
            request_id=f"hybrid-{plan_id}",
        )
        assert response.planner_decision.selected_plan_id == plan_id


@pytest.mark.asyncio
async def test_stage10_profiler_executes_p1_and_p3_through_final_scheduler_path() -> None:
    engine, vector, graph = _engine(ManualClock())

    p1 = await engine.benchmark_plan_search(
        SearchRequest(query="question", planner=PlannerMode.VECTOR),
        request_id="profile-p1",
        plan_id="P1",
    )
    p3 = await engine.benchmark_plan_search(
        SearchRequest(query="question", planner=PlannerMode.GRAPH),
        request_id="profile-p3",
        plan_id="P3",
    )

    assert p1.planner_decision.selected_plan_id == "P1"
    assert p3.planner_decision.selected_plan_id == "P3"
    assert vector.calls == [(30, "corpus-v1")]
    assert graph.calls == [("corpus-v1", 30, 3)]
    assert p1.trace.features == p3.trace.features
    assert p1.trace.scheduler_trace is not None
    assert p3.trace.scheduler_trace is not None
    assert p1.trace.config_version == load_default_plan_catalog().sha256()
    assert p3.trace.config_version == load_default_plan_catalog().sha256()


@pytest.mark.asyncio
async def test_stage10_profiler_refuses_mode_mismatch_and_p7() -> None:
    engine, _, _ = _engine(ManualClock())
    with pytest.raises(RAGPlanError) as mismatch:
        await engine.benchmark_plan_search(
            SearchRequest(query="question", planner=PlannerMode.GRAPH),
            request_id="profile-mismatch",
            plan_id="P1",
        )
    assert mismatch.value.code is ErrorCode.INVALID_REQUEST

    with pytest.raises(RAGPlanError) as disabled:
        await engine.benchmark_plan_search(
            SearchRequest(
                query="question",
                planner=PlannerMode.FIXED_HYBRID,
                plan_id="P4",
            ),
            request_id="profile-p7",
            plan_id="P7",
        )
    assert disabled.value.code is ErrorCode.PLAN_INVARIANT_VIOLATION


def test_mismatched_graph_stage_is_rejected_before_serving() -> None:
    clock = ManualClock()
    with pytest.raises(RAGPlanError) as error:
        BaselineSearchEngine(
            embedder=_Embedder(clock),
            vector_backend=_VectorBackend(clock),
            analyzer=_Analyzer(clock),  # type: ignore[arg-type]
            graph_backend=_GraphBackend(clock),
            plan_catalog=load_default_plan_catalog(),
            active_manifest=_active(),
            vector_stage=_vector_stage(),
            graph_stage=_graph_stage().model_copy(update={"corpus_version": "wrong-v1"}),
            clock=clock,
        )
    assert error.value.code is ErrorCode.CORPUS_INCONSISTENT


@pytest.mark.asyncio
async def test_shared_engine_closes_both_backends() -> None:
    engine, vector, graph = _engine(ManualClock())
    await engine.close()
    assert vector.closed is True
    assert graph.closed is True


@pytest.mark.asyncio
async def test_single_mode_provenance_finalization_cannot_escape_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ragplan.core import engine as engine_module

    clock = ManualClock()
    engine, _, _ = _engine(clock)
    original = engine_module.annotate_single_source

    def delayed_annotation(*args: object, **kwargs: object) -> object:
        result = original(*args, **kwargs)  # type: ignore[arg-type]
        clock.advance_ms(196.0)
        return result

    monkeypatch.setattr(engine_module, "annotate_single_source", delayed_annotation)
    with pytest.raises(RAGPlanError) as error:
        await engine.search(
            SearchRequest(
                query="question",
                planner=PlannerMode.VECTOR,
                latency_budget_ms=200,
            ),
            request_id="single-finalization-overrun",
        )
    assert error.value.code is ErrorCode.DEADLINE_EXCEEDED


@pytest.mark.asyncio
async def test_vector_backend_failure_returns_partial_graph_with_frozen_scheduler_trace() -> None:
    clock = ManualClock()
    vector = _FailingVectorBackend(clock)
    graph = _GraphBackend(clock)
    engine = _custom_engine(clock, vector=vector, graph=graph)

    response = await engine.search(
        SearchRequest(query="question", planner=PlannerMode.FIXED_HYBRID, top_k=2),
        request_id="partial-graph",
    )

    assert response.status is SearchStatus.PARTIAL
    assert response.fallback is True
    assert [hit.canonical_chunk_id for hit in response.results] == ["shared"]
    assert response.results[0].sources == (BranchKind.GRAPH,)
    assert response.trace.fusion_trace is not None
    assert response.trace.fusion_trace.missing_branches == (BranchKind.VECTOR,)
    by_branch = {item.branch: item for item in response.trace.branch_results}
    assert by_branch[BranchKind.VECTOR].status is BranchStatus.FAILED
    assert by_branch[BranchKind.VECTOR].error_code is ErrorCode.DEPENDENCY_UNAVAILABLE
    assert by_branch[BranchKind.GRAPH].status is BranchStatus.SUCCEEDED
    scheduler = response.trace.scheduler_trace
    assert scheduler is not None
    assert scheduler.runtime_semantics_version == "v1"
    assert tuple(event.state for event in scheduler.state_events) == (
        RequestState.RECEIVED,
        RequestState.ANALYZING,
        RequestState.PLANNING,
        RequestState.EXECUTING,
        RequestState.FUSING,
        RequestState.PARTIAL,
    )
    assert scheduler.actual_terminal_state is RequestState.PARTIAL
    assert scheduler.fallback_reason == "vector:DEPENDENCY_UNAVAILABLE"


@pytest.mark.asyncio
async def test_graph_backend_failure_returns_partial_vector_without_reordering() -> None:
    clock = ManualClock()
    vector = _VectorBackend(clock)
    graph = _FailingGraphBackend(clock)
    engine = _custom_engine(clock, vector=vector, graph=graph)

    response = await engine.search(
        SearchRequest(query="question", planner=PlannerMode.FIXED_HYBRID, top_k=2),
        request_id="partial-vector",
    )

    assert response.status is SearchStatus.PARTIAL
    assert [hit.canonical_chunk_id for hit in response.results] == [
        "shared",
        "vector-only",
    ]
    assert all(hit.sources == (BranchKind.VECTOR,) for hit in response.results)
    assert response.trace.fusion_trace is not None
    assert response.trace.fusion_trace.missing_branches == (BranchKind.GRAPH,)


@pytest.mark.asyncio
async def test_all_backend_errors_and_all_timeouts_have_distinct_public_failures() -> None:
    for code, expected in (
        (ErrorCode.DEPENDENCY_UNAVAILABLE, ErrorCode.DEPENDENCY_UNAVAILABLE),
        (ErrorCode.DEADLINE_EXCEEDED, ErrorCode.DEADLINE_EXCEEDED),
    ):
        clock = ManualClock()
        engine = _custom_engine(
            clock,
            vector=_FailingVectorBackend(clock, code),
            graph=_FailingGraphBackend(clock, code),
        )
        with pytest.raises(RAGPlanError) as captured:
            await engine.search(
                SearchRequest(query="question", planner=PlannerMode.FIXED_HYBRID),
                request_id=f"all-{code.value}",
            )
        assert captured.value.code is expected


@pytest.mark.asyncio
async def test_two_successful_zero_hit_branches_are_complete_empty_not_failure() -> None:
    clock = ManualClock()
    engine = _custom_engine(
        clock,
        vector=_EmptyVectorBackend(clock),
        graph=_EmptyGraphBackend(clock),
    )

    response = await engine.search(
        SearchRequest(query="question", planner=PlannerMode.FIXED_HYBRID),
        request_id="zero-hits",
    )

    assert response.status is SearchStatus.COMPLETE
    assert response.results == ()
    assert response.fallback is False
    assert {item.status for item in response.trace.branch_results} == {BranchStatus.SUCCEEDED}


@pytest.mark.asyncio
async def test_force_vector_kill_switch_bypasses_graph_for_hybrid_ingress() -> None:
    clock = ManualClock()
    vector = _VectorBackend(clock)
    graph = _GraphBackend(clock)
    engine = _custom_engine(
        clock,
        vector=vector,
        graph=graph,
        switches=KillSwitchSnapshot(force_vector_only=True),
    )

    response = await engine.search(
        SearchRequest(query="question", planner=PlannerMode.FIXED_HYBRID),
        request_id="forced-vector",
    )

    assert vector.calls
    assert graph.calls == []
    assert response.planner_decision.mode is PlannerMode.FIXED_HYBRID
    assert response.planner_decision.effective_mode is PlannerMode.VECTOR
    assert response.trace.scheduler_trace is not None
    assert response.trace.scheduler_trace.backend_task_count == 1
    assert response.trace.scheduler_trace.kill_switches == (KillSwitch.FORCE_VECTOR_ONLY,)


@pytest.mark.asyncio
async def test_disable_cost_aware_kill_switch_rejects_before_any_backend_call() -> None:
    clock = ManualClock()
    vector = _VectorBackend(clock)
    graph = _GraphBackend(clock)
    engine = _custom_engine(
        clock,
        vector=vector,
        graph=graph,
        switches=KillSwitchSnapshot(disable_cost_aware=True),
    )

    with pytest.raises(RAGPlanError) as captured:
        await engine.search(
            SearchRequest(query="question", planner=PlannerMode.COST_AWARE),
            request_id="cost-disabled",
        )

    assert captured.value.code is ErrorCode.MODE_UNAVAILABLE
    assert vector.calls == []
    assert graph.calls == []


@pytest.mark.asyncio
async def test_research_only_cost_aware_is_not_exposed_by_public_engine() -> None:
    clock = ManualClock()
    vector = _VectorBackend(clock)
    graph = _GraphBackend(clock)
    engine = _custom_engine(clock, vector=vector, graph=graph)

    with pytest.raises(RAGPlanError) as captured:
        await engine.search(
            SearchRequest(query="question", planner=PlannerMode.COST_AWARE),
            request_id="cost-research-only",
        )

    assert captured.value.code is ErrorCode.MODE_UNAVAILABLE
    assert vector.calls == []
    assert graph.calls == []


@pytest.mark.asyncio
async def test_rule_mode_fail_closed_policy_executes_vector_only() -> None:
    engine, vector, graph = _engine(ManualClock())

    response = await engine.search(
        SearchRequest(
            query="Who founded Acme and who acquired it?",
            planner=PlannerMode.RULE,
            latency_budget_ms=500,
        ),
        request_id="rule-gated",
    )

    assert response.planner_decision.mode is PlannerMode.RULE
    assert response.planner_decision.effective_mode is PlannerMode.VECTOR
    assert response.planner_decision.selected_plan_id == "P1"
    assert response.planner_decision.fallback_reason is not None
    assert vector.calls and graph.calls == []
    assert response.trace.config_version == response.planner_decision.config_version


@pytest.mark.asyncio
async def test_rule_mode_analyzes_once_and_executes_budget_feasible_p8() -> None:
    clock = ManualClock()
    embedder = _Embedder(clock)
    analyzer = _Analyzer(clock)
    vector = _VectorBackend(clock)
    graph = _GraphBackend(clock)
    engine = BaselineSearchEngine(
        embedder=embedder,
        vector_backend=vector,
        analyzer=analyzer,  # type: ignore[arg-type]
        graph_backend=graph,
        plan_catalog=load_default_plan_catalog(),
        active_manifest=_active(),
        vector_stage=_vector_stage(),
        graph_stage=_graph_stage(),
        clock=clock,
        rule_planner=_enabled_rule_planner(),
    )

    response = await engine.search(
        SearchRequest(
            query="Who founded the company that acquired Acme and who created it?",
            planner=PlannerMode.RULE,
            latency_budget_ms=500,
        ),
        request_id="rule-deep",
    )

    assert analyzer.calls == 1
    assert embedder.calls == 1
    assert response.planner_decision.selected_plan_id == "P8"
    assert response.planner_decision.matched_rules == ("multi_hop_signal",)
    assert vector.calls and graph.calls
    assert response.trace.features.multi_hop_signal >= 0.5
