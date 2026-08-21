"""Baseline retrieval engines plus deterministic rule adaptation shared by all clients."""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Callable, Sequence
from typing import Protocol, runtime_checkable

from ragplan.backends.graph.base import GraphBackend
from ragplan.backends.vector.base import VectorBackend
from ragplan.core.deadline import (
    NANOSECONDS_PER_MILLISECOND,
    Deadline,
    MonotonicClock,
    PerfCounterClock,
)
from ragplan.core.errors import ErrorCode, RAGPlanError, TimeoutOrigin
from ragplan.core.health import EngineReadinessSnapshot, RuntimeProfile
from ragplan.core.models import (
    BranchKind,
    BranchResult,
    BranchStatus,
    CancellationReason,
    CircuitState,
    FusionTrace,
    GraphStageManifest,
    IngestionManifest,
    PlanDefinition,
    PlannerDecision,
    PlannerMode,
    QueryAnalysis,
    QueryFeatures,
    RequestState,
    RetrievalHit,
    SchedulerTrace,
    SearchRequest,
    SearchResponse,
    SearchStatus,
    SearchTrace,
    VectorStageManifest,
)
from ragplan.ingestion.audit import load_graph_tier_policy
from ragplan.ingestion.chunker import Tokenizer
from ragplan.ingestion.normalize import normalize_text
from ragplan.planner.analyzer import QueryAnalyzer as AdaptiveQueryAnalyzer
from ragplan.planner.catalog import PlanCatalog
from ragplan.planner.features import (
    extract_query_features,
    load_default_query_feature_config,
)
from ragplan.planner.rule import RulePlanner, load_default_rule_planner_config
from ragplan.retrieval.fusion import annotate_single_source, weighted_rrf_v1
from ragplan.retrieval.graph import GraphQueryAnalyzer, execute_graph_search
from ragplan.retrieval.vector import execute_vector_search
from ragplan.scheduler.cancellation import cancel_and_await
from ragplan.scheduler.executor import (
    AdmissionController,
    BranchPayload,
    BranchWork,
    SchedulerExecution,
    SchedulerExecutor,
)
from ragplan.scheduler.states import CircuitBreaker, KillSwitchSnapshot, RequestStateMachine

ANALYZER_VERSION = "stage3-vector-v1"
FEATURE_VERSION = "v1"
GRAPH_DEPTH_2_BENCHMARK_VERSION = "stage9-graph-depth-2-v1"


def benchmark_graph_depth_2_config_version(plan_catalog_sha256: str) -> str:
    """Identify the internal depth-2 derivation without expanding the public catalog."""

    return hashlib.sha256(
        f"{plan_catalog_sha256}:{GRAPH_DEPTH_2_BENCHMARK_VERSION}".encode()
    ).hexdigest()


@runtime_checkable
class QueryEmbedder(Protocol):
    """Minimal online embedding surface; implementations load only pinned artifacts."""

    @property
    def tokenizer(self) -> Tokenizer: ...

    async def embed_query(self, query: str) -> Sequence[float]: ...


@runtime_checkable
class SearchEngine(Protocol):
    """API-facing engine lifecycle contract."""

    async def search(self, request: SearchRequest, *, request_id: str) -> SearchResponse: ...

    async def close(self) -> None: ...


@runtime_checkable
class ReadinessProvider(Protocol):
    """Optional operational surface implemented by production search engines."""

    async def readiness(self) -> EngineReadinessSnapshot: ...


class VectorSearchEngine:
    """Execute explicit vector retrieval against one vector-staged corpus version.

    A Stage 3 corpus is deliberately supplied explicitly and is not called an
    active corpus. Full dual-store activation is deferred until Stage 4 can
    reconcile Qdrant and Neo4j.
    """

    def __init__(
        self,
        *,
        embedder: QueryEmbedder,
        vector_backend: VectorBackend,
        plan_catalog: PlanCatalog,
        vector_stage: VectorStageManifest,
        clock: MonotonicClock | None = None,
        rule_planner: RulePlanner | None = None,
        active_corpus: bool = False,
    ) -> None:
        self._embedder = embedder
        self._vector_backend = vector_backend
        self._plan_catalog = plan_catalog
        self._vector_stage = vector_stage
        self._corpus_version = vector_stage.corpus_version
        self._model_revision = vector_stage.embedding_model_revision
        self._active_corpus = active_corpus
        self._clock = clock if clock is not None else PerfCounterClock()
        self._feature_config = load_default_query_feature_config()
        self._rule_planner = (
            rule_planner
            if rule_planner is not None
            else RulePlanner(
                catalog=plan_catalog,
                graph_policy=load_graph_tier_policy(),
                config=load_default_rule_planner_config(),
            )
        )
        if self._feature_config.sha256 != self._rule_planner.feature_config_sha256:
            raise ValueError("vector analyzer and rule planner feature configs do not match")

    async def search(self, request: SearchRequest, *, request_id: str) -> SearchResponse:
        """Analyze, embed once, retrieve, and return a redacted Stage 3 trace."""

        deadline = Deadline.start(request.latency_budget_ms, clock=self._clock)
        if request.planner not in {PlannerMode.VECTOR, PlannerMode.RULE}:
            raise RAGPlanError(
                ErrorCode.MODE_UNAVAILABLE,
                "the vector-only runtime supports vector and rule modes",
            )
        analysis_started_ns = self._clock.now_ns()
        normalized_query = normalize_text(request.query)
        token_count = self._embedder.tokenizer.encode(normalized_query).token_count
        language_supported = _is_p0_english(normalized_query)
        features = (
            extract_query_features(
                normalized_query,
                token_count=token_count,
                entity_count=0,
                final_top_k=request.top_k,
                config=self._feature_config,
            )
            if request.planner is PlannerMode.RULE
            else _stage3_features(token_count=token_count, final_top_k=request.top_k)
        )
        analysis_latency_ms = _elapsed_ms(analysis_started_ns, self._clock.now_ns())

        analysis = QueryAnalysis(
            normalized_query=normalized_query,
            language_supported=language_supported,
            token_count=token_count,
            query_embedding=(),
            features=features,
            analyzer_version=(
                f"stage8-vector:{self._feature_config.schema_version}"
                if request.planner is PlannerMode.RULE
                else ANALYZER_VERSION
            ),
            analysis_latency_ms=analysis_latency_ms,
        )

        embedding_started_ns = self._clock.now_ns()
        embedding = await _embed_before_cutoff(self._embedder, normalized_query, deadline)
        embedding_finished_ns = self._clock.now_ns()
        embedding_latency_ms = _elapsed_ms(embedding_started_ns, embedding_finished_ns)
        if embedding_finished_ns >= deadline.branch_cutoff_ns:
            raise RAGPlanError(ErrorCode.DEADLINE_EXCEEDED, "embedding deadline exceeded")

        analysis = analysis.model_copy(
            update={"query_embedding": tuple(float(value) for value in embedding)}
        )
        planner_started_ns = self._clock.now_ns()
        if (
            request.planner is PlannerMode.RULE
            and self._feature_config.sha256 != self._rule_planner.feature_config_sha256
        ):
            raise RAGPlanError(
                ErrorCode.PLAN_INVARIANT_VIOLATION,
                "query analyzer and rule planner feature configs do not match",
                retryable=False,
            )
        decision = self._select_vector_plan(request, deadline, analysis=analysis)
        planner_latency_ms = _elapsed_ms(planner_started_ns, self._clock.now_ns())
        assert decision.executed_vector_top_k is not None
        execution = await execute_vector_search(
            backend=self._vector_backend,
            embedding=analysis.query_embedding,
            top_k=decision.executed_vector_top_k,
            corpus_version=self._corpus_version,
            deadline=deadline,
        )
        results = execution.hits[: request.top_k]
        total_latency_ms = deadline.snapshot().elapsed_ms
        branch = BranchResult(
            branch=BranchKind.VECTOR,
            status=BranchStatus.SUCCEEDED,
            latency_ms=execution.latency_ms,
            hits=execution.hits,
        )
        trace = SearchTrace(
            request_id=request_id,
            query_hash=hashlib.sha256(request.query.encode("utf-8")).hexdigest(),
            query_length=len(request.query),
            language_supported=language_supported,
            features=features,
            planner_decision=decision,
            branch_results=(branch,),
            analyzer_latency_ms=analysis_latency_ms,
            planner_latency_ms=planner_latency_ms,
            embedding_latency_ms=embedding_latency_ms,
            vector_latency_ms=execution.latency_ms,
            total_latency_ms=total_latency_ms,
            latency_budget_ms=request.latency_budget_ms,
            finalization_reserve_ms=deadline.finalization_reserve_ms,
            budget_feasible=decision.budget_feasible,
            budget_violated=total_latency_ms > request.latency_budget_ms,
            fallback=False,
            result_count=len(results),
            corpus_version=self._corpus_version,
            config_version=decision.config_version,
            model_version=self._model_revision,
        )
        response = SearchResponse(
            status=SearchStatus.COMPLETE,
            results=results,
            planner_decision=decision,
            trace=trace,
            fallback=False,
            request_id=request_id,
        )
        # ADR-010 defines the engine boundary through response DTO completion.
        # The finalization reserve is not a grace interval: never return a DTO
        # that completed after the absolute request deadline.
        if deadline.snapshot().budget_violated:
            raise RAGPlanError(
                ErrorCode.DEADLINE_EXCEEDED,
                "response finalization deadline exceeded",
            )
        return response

    def _select_vector_plan(
        self,
        request: SearchRequest,
        deadline: Deadline,
        *,
        analysis: QueryAnalysis,
    ) -> PlannerDecision:
        if request.planner is PlannerMode.RULE:
            return self._rule_planner.select(
                analysis,
                deadline=deadline,
                graph_runtime_available=False,
            )
        if request.planner is not PlannerMode.VECTOR:
            raise RAGPlanError(
                ErrorCode.MODE_UNAVAILABLE,
                "only explicit vector mode is available in the Stage 3 engine",
            )
        plan = self._plan_catalog.plan_for_id("P0" if request.top_k <= 10 else "P1")
        executed_top_k = max(request.top_k, plan.vector_top_k)
        remaining_ms = deadline.snapshot().branch_remaining_ms
        if remaining_ms <= 0:
            raise RAGPlanError(ErrorCode.DEADLINE_EXCEEDED, "planning deadline exceeded")
        selection_reason = "explicit vector mode; smallest vector preset covering final top-k"
        if request.top_k > plan.vector_top_k:
            selection_reason = "explicit vector mode; request-floor override derived from P1"
        return PlannerDecision(
            mode=PlannerMode.VECTOR,
            effective_mode=PlannerMode.VECTOR,
            selected_plan_id=plan.id,
            selected_plan=plan,
            executed_vector_top_k=executed_top_k,
            remaining_budget_ms=remaining_ms,
            budget_feasible=True,
            selection_reason=selection_reason,
            feature_version=FEATURE_VERSION,
            config_version=self._plan_catalog.sha256(),
        )

    async def close(self) -> None:
        await self._vector_backend.close()

    async def readiness(self) -> EngineReadinessSnapshot:
        return EngineReadinessSnapshot(
            profile=(
                RuntimeProfile.VECTOR_ACTIVE
                if self._active_corpus
                else RuntimeProfile.VECTOR_STAGED
            ),
            corpus_version=self._corpus_version,
            active_corpus=self._active_corpus,
            supported_modes=(PlannerMode.VECTOR, PlannerMode.RULE),
            vector=await self._vector_backend.health(),
        )


class GraphSearchEngine:
    """Execute explicit graph retrieval against one reconciled active corpus."""

    def __init__(
        self,
        *,
        analyzer: GraphQueryAnalyzer,
        graph_backend: GraphBackend,
        plan_catalog: PlanCatalog,
        active_manifest: IngestionManifest,
        clock: MonotonicClock | None = None,
    ) -> None:
        if active_manifest.activation_status.value != "active":
            raise ValueError("graph search requires an active reconciled corpus manifest")
        self._analyzer = analyzer
        self._graph_backend = graph_backend
        self._plan_catalog = plan_catalog
        self._active_manifest = active_manifest
        self._corpus_version = active_manifest.corpus_version
        self._clock = clock if clock is not None else PerfCounterClock()

    async def search(self, request: SearchRequest, *, request_id: str) -> SearchResponse:
        if request.planner is not PlannerMode.GRAPH:
            raise RAGPlanError(
                ErrorCode.MODE_UNAVAILABLE,
                "only explicit graph mode is available in the Stage 5 engine",
            )

        deadline = Deadline.start(request.latency_budget_ms, clock=self._clock)
        analysis = self._analyzer.analyze(request.query, final_top_k=request.top_k)
        if not analysis.language_supported:
            raise RAGPlanError(
                ErrorCode.MODE_UNAVAILABLE,
                "unsupported language requires the safe vector-only rule plan",
                retryable=False,
            )
        planner_started_ns = self._clock.now_ns()
        decision = self._select_graph_plan(request, deadline)
        planner_latency_ms = _elapsed_ms(planner_started_ns, self._clock.now_ns())

        plan = decision.selected_plan
        assert plan is not None
        execution = await execute_graph_search(
            backend=self._graph_backend,
            query_analysis=analysis,
            plan=plan,
            corpus_version=self._corpus_version,
            deadline=deadline,
        )
        results = execution.hits[: request.top_k]
        total_latency_ms = deadline.snapshot().elapsed_ms
        branch = BranchResult(
            branch=BranchKind.GRAPH,
            status=BranchStatus.SUCCEEDED,
            latency_ms=execution.latency_ms,
            hits=execution.hits,
        )
        trace = SearchTrace(
            request_id=request_id,
            query_hash=hashlib.sha256(request.query.encode("utf-8")).hexdigest(),
            query_length=len(request.query),
            language_supported=True,
            features=analysis.features,
            planner_decision=decision,
            branch_results=(branch,),
            analyzer_latency_ms=analysis.analysis_latency_ms,
            planner_latency_ms=planner_latency_ms,
            graph_latency_ms=execution.latency_ms,
            graph_trace=execution.trace,
            total_latency_ms=total_latency_ms,
            latency_budget_ms=request.latency_budget_ms,
            finalization_reserve_ms=deadline.finalization_reserve_ms,
            budget_feasible=decision.budget_feasible,
            budget_violated=total_latency_ms > request.latency_budget_ms,
            fallback=False,
            result_count=len(results),
            corpus_version=self._corpus_version,
            config_version=self._plan_catalog.sha256(),
            model_version=self._active_manifest.extractor_version,
        )
        response = SearchResponse(
            status=SearchStatus.COMPLETE,
            results=results,
            planner_decision=decision,
            trace=trace,
            fallback=False,
            request_id=request_id,
        )
        if deadline.snapshot().budget_violated:
            raise RAGPlanError(
                ErrorCode.DEADLINE_EXCEEDED,
                "response finalization deadline exceeded",
            )
        return response

    def _select_graph_plan(self, request: SearchRequest, deadline: Deadline) -> PlannerDecision:
        plan = self._plan_catalog.plan_for_id("P2" if request.top_k <= 20 else "P3")
        executed_top_k = max(request.top_k, plan.graph_top_k)
        remaining_ms = deadline.snapshot().branch_remaining_ms
        if remaining_ms <= 0:
            raise RAGPlanError(ErrorCode.DEADLINE_EXCEEDED, "planning deadline exceeded")
        selection_reason = "explicit graph mode; smallest graph preset covering final top-k"
        if request.top_k > plan.graph_top_k:
            selection_reason = "explicit graph mode; request-floor override derived from P3"
        return PlannerDecision(
            mode=PlannerMode.GRAPH,
            effective_mode=PlannerMode.GRAPH,
            selected_plan_id=plan.id,
            selected_plan=plan,
            executed_graph_top_k=executed_top_k,
            remaining_budget_ms=remaining_ms,
            budget_feasible=True,
            selection_reason=selection_reason,
            feature_version=FEATURE_VERSION,
            config_version=self._plan_catalog.sha256(),
        )

    async def close(self) -> None:
        await self._graph_backend.close()

    async def readiness(self) -> EngineReadinessSnapshot:
        return EngineReadinessSnapshot(
            profile=RuntimeProfile.GRAPH_ACTIVE,
            corpus_version=self._corpus_version,
            active_corpus=True,
            supported_modes=(PlannerMode.GRAPH,),
            graph=await self._graph_backend.health(),
        )


class BaselineSearchEngine:
    """Serve explicit baselines and deterministic rule adaptation over one corpus."""

    _FIXED_PLAN_IDS = frozenset({"P4", "P5", "P6", "P8"})

    def __init__(
        self,
        *,
        embedder: QueryEmbedder,
        vector_backend: VectorBackend,
        analyzer: GraphQueryAnalyzer,
        graph_backend: GraphBackend,
        plan_catalog: PlanCatalog,
        active_manifest: IngestionManifest,
        vector_stage: VectorStageManifest,
        graph_stage: GraphStageManifest,
        clock: MonotonicClock | None = None,
        scheduler: SchedulerExecutor | None = None,
        admission: AdmissionController | None = None,
        vector_circuit: CircuitBreaker | None = None,
        graph_circuit: CircuitBreaker | None = None,
        kill_switch_provider: Callable[[], KillSwitchSnapshot] | None = None,
        query_analyzer: AdaptiveQueryAnalyzer | None = None,
        rule_planner: RulePlanner | None = None,
    ) -> None:
        self._require_shared_active_corpus(
            active_manifest=active_manifest,
            vector_stage=vector_stage,
            graph_stage=graph_stage,
            analyzer=analyzer,
        )
        self._clock = clock if clock is not None else PerfCounterClock()
        self._embedder = embedder
        self._vector_backend = vector_backend
        self._analyzer = analyzer
        self._graph_backend = graph_backend
        self._plan_catalog = plan_catalog
        self._query_analyzer = (
            query_analyzer
            if query_analyzer is not None
            else AdaptiveQueryAnalyzer(
                entity_analyzer=analyzer,
                embedder=embedder,
                feature_config=load_default_query_feature_config(),
                clock=self._clock,
            )
        )
        self._rule_planner = (
            rule_planner
            if rule_planner is not None
            else RulePlanner(
                catalog=plan_catalog,
                graph_policy=load_graph_tier_policy(),
                config=load_default_rule_planner_config(),
            )
        )
        if self._query_analyzer.feature_config_sha256 != self._rule_planner.feature_config_sha256:
            raise ValueError("query analyzer and rule planner feature configs do not match")
        self._active_manifest = active_manifest
        self._vector_stage = vector_stage
        self._graph_stage = graph_stage
        self._scheduler = scheduler if scheduler is not None else SchedulerExecutor()
        self._admission = admission if admission is not None else AdmissionController()
        self._vector_circuit = (
            vector_circuit if vector_circuit is not None else CircuitBreaker(clock=self._clock)
        )
        self._graph_circuit = (
            graph_circuit if graph_circuit is not None else CircuitBreaker(clock=self._clock)
        )
        self._kill_switch_provider = (
            kill_switch_provider
            if kill_switch_provider is not None
            else KillSwitchSnapshot.from_environment
        )
        self._request_tasks: set[asyncio.Task[object]] = set()
        self._lifecycle_lock = asyncio.Lock()
        self._closing = False
        self._closed = False
        self._close_complete = asyncio.Event()

    async def search(self, request: SearchRequest, *, request_id: str) -> SearchResponse:
        """Run every explicit baseline through the frozen Stage 7 scheduler."""

        return await self._run_search(
            request,
            request_id=request_id,
            graph_depth_override=None,
            profile_plan_id=None,
        )

    async def benchmark_graph_search(
        self,
        request: SearchRequest,
        *,
        request_id: str,
        graph_depth: int,
    ) -> SearchResponse:
        """Run a Stage 9 graph-depth baseline without expanding the public API contract."""

        if request.planner is not PlannerMode.GRAPH or graph_depth not in {1, 2, 3}:
            raise RAGPlanError(
                ErrorCode.INVALID_REQUEST,
                "benchmark graph execution requires graph mode and depth 1, 2, or 3",
                retryable=False,
            )
        return await self._run_search(
            request,
            request_id=request_id,
            graph_depth_override=graph_depth,
            profile_plan_id=None,
        )

    async def benchmark_plan_search(
        self,
        request: SearchRequest,
        *,
        request_id: str,
        plan_id: str,
    ) -> SearchResponse:
        """Execute one immutable P0 plan through the final scheduler path.

        This is an offline-profiler surface, not a public request alias.  It keeps
        P1/P3 directly measurable without weakening ``SearchRequest`` validation.
        """

        try:
            plan = self._plan_catalog.plan_for_id(plan_id)
        except KeyError as exc:
            raise RAGPlanError(
                ErrorCode.PLAN_INVARIANT_VIOLATION,
                "profiler plan is absent from the immutable catalog",
                retryable=False,
            ) from exc
        if not plan.enabled_in_p0 or plan.rerank_enabled:
            raise RAGPlanError(
                ErrorCode.PLAN_INVARIANT_VIOLATION,
                "profiler accepts only enabled P0 plans",
                retryable=False,
            )
        expected_mode = self._mode_for_plan(plan)
        if request.planner is not expected_mode:
            raise RAGPlanError(
                ErrorCode.INVALID_REQUEST,
                "profiler request mode does not match the selected plan",
                retryable=False,
            )
        if expected_mode is PlannerMode.FIXED_HYBRID and request.plan_id != plan.id:
            raise RAGPlanError(
                ErrorCode.INVALID_REQUEST,
                "profiler hybrid request must name the selected plan",
                retryable=False,
            )
        return await self._run_search(
            request,
            request_id=request_id,
            graph_depth_override=None,
            profile_plan_id=plan.id,
        )

    async def _run_search(
        self,
        request: SearchRequest,
        *,
        request_id: str,
        graph_depth_override: int | None,
        profile_plan_id: str | None,
    ) -> SearchResponse:
        """Share lifecycle, admission, scheduler, and deadline semantics across callers."""

        deadline = Deadline.start(request.latency_budget_ms, clock=self._clock)
        states = RequestStateMachine(deadline)
        switches = self._kill_switch_provider()
        request_task = asyncio.current_task()
        if request_task is None:
            raise RAGPlanError(ErrorCode.INTERNAL_ERROR, "request task is unavailable")
        async with self._lifecycle_lock:
            if self._closing:
                raise RAGPlanError(ErrorCode.NOT_READY, "search engine is shutting down")
            self._request_tasks.add(request_task)
        try:
            async with self._admission.slot():
                return await self._execute_request(
                    request,
                    request_id=request_id,
                    deadline=deadline,
                    states=states,
                    switches=switches,
                    graph_depth_override=graph_depth_override,
                    profile_plan_id=profile_plan_id,
                )
        except BaseException:
            states.fail_if_possible()
            raise
        finally:
            async with self._lifecycle_lock:
                self._request_tasks.discard(request_task)

    async def _execute_request(
        self,
        request: SearchRequest,
        *,
        request_id: str,
        deadline: Deadline,
        states: RequestStateMachine,
        switches: KillSwitchSnapshot,
        graph_depth_override: int | None,
        profile_plan_id: str | None,
    ) -> SearchResponse:
        force_vector = switches.force_vector_only
        if force_vector and (graph_depth_override is not None or profile_plan_id is not None):
            raise RAGPlanError(
                ErrorCode.MODE_UNAVAILABLE,
                "force-vector kill switch disables benchmark override execution",
            )
        if graph_depth_override is not None and profile_plan_id is not None:
            raise RAGPlanError(
                ErrorCode.PLAN_INVARIANT_VIOLATION,
                "benchmark depth and profiler plan overrides are mutually exclusive",
                retryable=False,
            )
        if (
            not force_vector
            and request.planner is PlannerMode.COST_AWARE
            and switches.disable_cost_aware
        ):
            raise RAGPlanError(
                ErrorCode.MODE_UNAVAILABLE,
                "cost-aware mode is disabled by the runtime kill switch",
            )
        supported_modes = {
            PlannerMode.VECTOR,
            PlannerMode.GRAPH,
            PlannerMode.FIXED_HYBRID,
            PlannerMode.RULE,
        }
        if request.planner not in supported_modes and not force_vector:
            raise RAGPlanError(
                ErrorCode.MODE_UNAVAILABLE,
                "this runtime serves explicit baselines and rule mode",
            )

        states.transition(RequestState.ANALYZING)
        effective_mode = PlannerMode.VECTOR if force_vector else request.planner
        if profile_plan_id is not None:
            profile_plan = self._plan_catalog.plan_for_id(profile_plan_id)
            effective_mode = self._mode_for_plan(profile_plan)
            analyzer_execution = await self._query_analyzer.analyze(
                request.query,
                final_top_k=request.top_k,
                deadline=deadline,
            )
            analysis = analyzer_execution.analysis
            embedding_latency_ms = analyzer_execution.embedding_latency_ms
            if analyzer_execution.feature_config_sha256 != self._rule_planner.feature_config_sha256:
                raise RAGPlanError(
                    ErrorCode.PLAN_INVARIANT_VIOLATION,
                    "query analyzer and profiler feature configs do not match",
                    retryable=False,
                )
        elif effective_mode is PlannerMode.RULE:
            analyzer_execution = await self._query_analyzer.analyze(
                request.query,
                final_top_k=request.top_k,
                deadline=deadline,
            )
            analysis = analyzer_execution.analysis
            embedding_latency_ms = analyzer_execution.embedding_latency_ms
            if analyzer_execution.feature_config_sha256 != self._rule_planner.feature_config_sha256:
                raise RAGPlanError(
                    ErrorCode.PLAN_INVARIANT_VIOLATION,
                    "query analyzer and rule planner feature configs do not match",
                    retryable=False,
                )
        else:
            analysis, embedding_latency_ms = await self._analyze(
                request,
                effective_mode=effective_mode,
                deadline=deadline,
            )
        if not analysis.language_supported and effective_mode is not PlannerMode.VECTOR:
            if effective_mode is not PlannerMode.RULE:
                raise RAGPlanError(
                    ErrorCode.MODE_UNAVAILABLE,
                    "unsupported language requires the safe vector-only rule plan",
                    retryable=False,
                )

        states.transition(RequestState.PLANNING)
        planner_started_ns = self._clock.now_ns()
        if profile_plan_id is not None:
            decision = self._select_profile_plan(
                request,
                deadline,
                plan_id=profile_plan_id,
            )
        elif effective_mode is PlannerMode.RULE:
            graph_circuit = await self._graph_circuit.snapshot()
            decision = self._rule_planner.select(
                analysis,
                deadline=deadline,
                graph_runtime_available=graph_circuit.state is not CircuitState.OPEN,
            )
            assert decision.effective_mode is not None
            effective_mode = decision.effective_mode
        elif effective_mode is PlannerMode.VECTOR:
            decision = self._select_vector_plan(
                request,
                deadline,
                forced=force_vector and request.planner is not PlannerMode.VECTOR,
            )
        elif effective_mode is PlannerMode.GRAPH:
            decision = self._select_graph_plan(
                request,
                deadline,
                graph_depth_override=graph_depth_override,
            )
        else:
            decision = self._select_fixed_plan(request, deadline)
        planner_latency_ms = _elapsed_ms(planner_started_ns, self._clock.now_ns())

        states.transition(RequestState.EXECUTING)
        works = self._branch_work(
            effective_mode=effective_mode,
            analysis=analysis,
            decision=decision,
            deadline=deadline,
        )
        execution = await self._scheduler.execute(works, deadline=deadline)
        self._raise_if_no_usable_branch(execution)

        states.transition(RequestState.FUSING)
        fusion_started_ns = self._clock.now_ns()
        results, fusion_trace, fallback_reason = self._rank_execution(
            execution,
            effective_mode=effective_mode,
            decision=decision,
            top_k=request.top_k,
        )
        fusion_latency_ms = _elapsed_ms(fusion_started_ns, self._clock.now_ns())
        fallback = fallback_reason is not None
        if fallback:
            decision = decision.model_copy(update={"fallback_reason": fallback_reason})
            states.transition(RequestState.PARTIAL)
        else:
            states.transition(RequestState.COMPLETE)

        total_latency_ms = deadline.snapshot().elapsed_ms
        status = SearchStatus.PARTIAL if fallback else SearchStatus.COMPLETE
        scheduler_trace = SchedulerTrace(
            state_events=states.events,
            actual_terminal_state=(RequestState.PARTIAL if fallback else RequestState.COMPLETE),
            backend_task_count=len(execution.branch_results),
            branch_start_skew_ms=execution.branch_start_skew_ms,
            deadline_overshoot_ms=max(0.0, total_latency_ms - request.latency_budget_ms),
            admission_limit=self._admission.limit,
            kill_switches=switches.active,
            vector_circuit_state=execution.vector_circuit_state,
            graph_circuit_state=execution.graph_circuit_state,
            fallback_reason=fallback_reason,
        )
        vector_result = execution.result_for(BranchKind.VECTOR)
        graph_result = execution.result_for(BranchKind.GRAPH)
        trace = SearchTrace(
            request_id=request_id,
            query_hash=hashlib.sha256(request.query.encode("utf-8")).hexdigest(),
            query_length=len(request.query),
            language_supported=analysis.language_supported,
            features=analysis.features,
            planner_decision=decision,
            branch_results=execution.branch_results,
            analyzer_latency_ms=analysis.analysis_latency_ms,
            planner_latency_ms=planner_latency_ms,
            embedding_latency_ms=embedding_latency_ms,
            vector_latency_ms=(vector_result.latency_ms if vector_result is not None else None),
            graph_latency_ms=graph_result.latency_ms if graph_result is not None else None,
            graph_trace=execution.graph_trace,
            fusion_latency_ms=fusion_latency_ms if fusion_trace is not None else 0.0,
            fusion_trace=fusion_trace,
            scheduler_trace=scheduler_trace,
            total_latency_ms=total_latency_ms,
            latency_budget_ms=request.latency_budget_ms,
            finalization_reserve_ms=deadline.finalization_reserve_ms,
            budget_feasible=decision.budget_feasible,
            budget_violated=total_latency_ms > request.latency_budget_ms,
            fallback=fallback,
            result_count=len(results),
            corpus_version=self._active_manifest.corpus_version,
            config_version=decision.config_version,
            model_version=(
                f"{self._vector_stage.embedding_model_revision}:"
                f"{self._graph_stage.extractor_version}"
            ),
        )
        response = SearchResponse(
            status=status,
            results=results,
            planner_decision=decision,
            trace=trace,
            fallback=fallback,
            request_id=request_id,
        )
        if deadline.snapshot().budget_violated:
            raise RAGPlanError(
                ErrorCode.DEADLINE_EXCEEDED,
                "response finalization deadline exceeded",
            )
        return response

    def _select_profile_plan(
        self,
        request: SearchRequest,
        deadline: Deadline,
        *,
        plan_id: str,
    ) -> PlannerDecision:
        """Build a decision for one catalog plan without bypassing runtime execution."""

        plan = self._plan_catalog.plan_for_id(plan_id)
        if not plan.enabled_in_p0 or plan.rerank_enabled:
            raise RAGPlanError(
                ErrorCode.PLAN_INVARIANT_VIOLATION,
                "profiler plan is not enabled in P0",
                retryable=False,
            )
        effective_mode = self._mode_for_plan(plan)
        if request.planner is not effective_mode:
            raise RAGPlanError(
                ErrorCode.PLAN_INVARIANT_VIOLATION,
                "profiler plan and request mode diverged",
                retryable=False,
            )
        remaining_ms = deadline.snapshot().branch_remaining_ms
        if remaining_ms <= 0:
            raise RAGPlanError(ErrorCode.DEADLINE_EXCEEDED, "planning deadline exceeded")
        return PlannerDecision(
            mode=request.planner,
            effective_mode=effective_mode,
            selected_plan_id=plan.id,
            selected_plan=plan,
            executed_vector_top_k=(
                max(request.top_k, plan.vector_top_k) if plan.vector_enabled else None
            ),
            executed_graph_top_k=(
                max(request.top_k, plan.graph_top_k) if plan.graph_enabled else None
            ),
            remaining_budget_ms=remaining_ms,
            budget_feasible=True,
            selection_reason=f"Stage 10 offline profiler; immutable {plan.id} preset",
            feature_version=FEATURE_VERSION,
            config_version=self._plan_catalog.sha256(),
        )

    @staticmethod
    def _mode_for_plan(plan: PlanDefinition) -> PlannerMode:
        if plan.vector_enabled and plan.graph_enabled:
            return PlannerMode.FIXED_HYBRID
        if plan.vector_enabled:
            return PlannerMode.VECTOR
        if plan.graph_enabled:
            return PlannerMode.GRAPH
        raise RAGPlanError(
            ErrorCode.PLAN_INVARIANT_VIOLATION,
            "profiler plan has no retrieval branch",
            retryable=False,
        )

    async def _analyze(
        self,
        request: SearchRequest,
        *,
        effective_mode: PlannerMode,
        deadline: Deadline,
    ) -> tuple[QueryAnalysis, float]:
        if effective_mode is PlannerMode.VECTOR:
            started_ns = self._clock.now_ns()
            normalized_query = normalize_text(request.query)
            token_count = self._embedder.tokenizer.encode(normalized_query).token_count
            language_supported = _is_p0_english(normalized_query)
            features = _stage3_features(token_count=token_count, final_top_k=request.top_k)
            analysis_latency_ms = _elapsed_ms(started_ns, self._clock.now_ns())
            analysis = QueryAnalysis(
                normalized_query=normalized_query,
                language_supported=language_supported,
                token_count=token_count,
                query_embedding=(),
                features=features,
                analyzer_version=ANALYZER_VERSION,
                analysis_latency_ms=analysis_latency_ms,
            )
        else:
            analysis = self._analyzer.analyze(request.query, final_top_k=request.top_k)

        embedding_latency_ms = 0.0
        if effective_mode in {PlannerMode.VECTOR, PlannerMode.FIXED_HYBRID}:
            embedding_started_ns = self._clock.now_ns()
            embedding = await _embed_before_cutoff(
                self._embedder,
                analysis.normalized_query,
                deadline,
            )
            embedding_latency_ms = _elapsed_ms(
                embedding_started_ns,
                self._clock.now_ns(),
            )
            analysis = analysis.model_copy(
                update={"query_embedding": tuple(float(value) for value in embedding)}
            )
        if deadline.snapshot().branch_remaining_ms <= 0:
            raise RAGPlanError(ErrorCode.DEADLINE_EXCEEDED, "analysis deadline exceeded")
        return analysis, embedding_latency_ms

    def _branch_work(
        self,
        *,
        effective_mode: PlannerMode,
        analysis: QueryAnalysis,
        decision: PlannerDecision,
        deadline: Deadline,
    ) -> tuple[BranchWork, ...]:
        works: list[BranchWork] = []
        if effective_mode in {PlannerMode.VECTOR, PlannerMode.FIXED_HYBRID}:
            assert decision.executed_vector_top_k is not None

            async def vector_operation() -> BranchPayload:
                execution = await execute_vector_search(
                    backend=self._vector_backend,
                    embedding=analysis.query_embedding,
                    top_k=decision.executed_vector_top_k or 1,
                    corpus_version=self._active_manifest.corpus_version,
                    deadline=deadline,
                )
                return BranchPayload(hits=execution.hits)

            works.append(
                BranchWork(
                    branch=BranchKind.VECTOR,
                    operation=vector_operation,
                    circuit=self._vector_circuit,
                )
            )
        if effective_mode in {PlannerMode.GRAPH, PlannerMode.FIXED_HYBRID}:
            assert decision.selected_plan is not None
            assert decision.executed_graph_top_k is not None
            execution_plan = decision.selected_plan.model_copy(
                update={"graph_top_k": decision.executed_graph_top_k}
            )

            async def graph_operation() -> BranchPayload:
                execution = await self._graph_backend.search(
                    analysis,
                    execution_plan,
                    self._active_manifest.corpus_version,
                    deadline,
                )
                return BranchPayload(hits=execution.hits, graph_trace=execution.trace)

            works.append(
                BranchWork(
                    branch=BranchKind.GRAPH,
                    operation=graph_operation,
                    circuit=self._graph_circuit,
                )
            )
        return tuple(works)

    @staticmethod
    def _raise_if_no_usable_branch(execution: SchedulerExecution) -> None:
        fatal_codes = {
            ErrorCode.CORPUS_INCONSISTENT,
            ErrorCode.MODEL_INCOMPATIBLE,
            ErrorCode.PLAN_INVARIANT_VIOLATION,
            ErrorCode.INTERNAL_ERROR,
        }
        fatal = next(
            (
                result.error_code
                for result in execution.branch_results
                if result.error_code in fatal_codes
            ),
            None,
        )
        if fatal is not None:
            raise RAGPlanError(fatal, "retrieval integrity validation failed", retryable=False)
        if any(result.status is BranchStatus.SUCCEEDED for result in execution.branch_results):
            return
        if any(
            result.status in {BranchStatus.TIMED_OUT, BranchStatus.CANCELLED}
            for result in execution.branch_results
        ):
            raise RAGPlanError(
                ErrorCode.DEADLINE_EXCEEDED,
                "no retrieval branch completed within the request deadline",
            )
        if all(
            result.error_code is ErrorCode.MODE_UNAVAILABLE for result in execution.branch_results
        ):
            raise RAGPlanError(ErrorCode.MODE_UNAVAILABLE, "retrieval mode is unavailable")
        raise RAGPlanError(
            ErrorCode.DEPENDENCY_UNAVAILABLE,
            "all scheduled retrieval backends failed",
        )

    @staticmethod
    def _rank_execution(
        execution: SchedulerExecution,
        *,
        effective_mode: PlannerMode,
        decision: PlannerDecision,
        top_k: int,
    ) -> tuple[tuple[RetrievalHit, ...], FusionTrace | None, str | None]:
        succeeded = {
            result.branch: result
            for result in execution.branch_results
            if result.status is BranchStatus.SUCCEEDED
        }
        if effective_mode is PlannerMode.VECTOR:
            vector = succeeded[BranchKind.VECTOR]
            return (
                annotate_single_source(
                    vector.hits,
                    source=BranchKind.VECTOR,
                    top_k=top_k,
                ),
                None,
                None,
            )
        if effective_mode is PlannerMode.GRAPH:
            graph = succeeded[BranchKind.GRAPH]
            return (
                annotate_single_source(
                    graph.hits,
                    source=BranchKind.GRAPH,
                    top_k=top_k,
                ),
                None,
                None,
            )

        plan = decision.selected_plan
        assert plan is not None
        vector_result = succeeded.get(BranchKind.VECTOR)
        graph_result = succeeded.get(BranchKind.GRAPH)
        fusion = weighted_rrf_v1(
            vector_hits=vector_result.hits if vector_result is not None else None,
            graph_hits=graph_result.hits if graph_result is not None else None,
            vector_weight=plan.vector_weight,
            graph_weight=plan.graph_weight,
            top_k=top_k,
        )
        failed = next(
            (
                result
                for result in execution.branch_results
                if result.status is not BranchStatus.SUCCEEDED
            ),
            None,
        )
        fallback_reason = None
        if failed is not None:
            detail = (
                failed.error_code.value if failed.error_code is not None else failed.status.value
            )
            fallback_reason = f"{failed.branch.value}:{detail}"
        return fusion.hits, fusion.trace, fallback_reason

    def _select_vector_plan(
        self,
        request: SearchRequest,
        deadline: Deadline,
        *,
        forced: bool = False,
    ) -> PlannerDecision:
        plan = self._plan_catalog.plan_for_id("P0" if request.top_k <= 10 else "P1")
        executed_top_k = max(request.top_k, plan.vector_top_k)
        remaining_ms = deadline.snapshot().branch_remaining_ms
        if remaining_ms <= 0:
            raise RAGPlanError(ErrorCode.DEADLINE_EXCEEDED, "planning deadline exceeded")
        selection_reason = "explicit vector mode; smallest vector preset covering final top-k"
        if request.top_k > plan.vector_top_k:
            selection_reason = "explicit vector mode; request-floor override derived from P1"
        fallback_reason = None
        if forced:
            selection_reason = "runtime force-vector kill switch"
            fallback_reason = "RAGPLAN_FORCE_VECTOR_ONLY"
        return PlannerDecision(
            mode=request.planner,
            effective_mode=PlannerMode.VECTOR,
            selected_plan_id=plan.id,
            selected_plan=plan,
            executed_vector_top_k=executed_top_k,
            remaining_budget_ms=remaining_ms,
            budget_feasible=True,
            selection_reason=selection_reason,
            fallback_reason=fallback_reason,
            feature_version=FEATURE_VERSION,
            config_version=self._plan_catalog.sha256(),
        )

    def _select_graph_plan(
        self,
        request: SearchRequest,
        deadline: Deadline,
        *,
        graph_depth_override: int | None = None,
    ) -> PlannerDecision:
        if graph_depth_override is None:
            plan = self._plan_catalog.plan_for_id("P2" if request.top_k <= 20 else "P3")
            config_version = self._plan_catalog.sha256()
            selection_reason = "explicit graph mode; smallest graph preset covering final top-k"
        elif graph_depth_override == 1:
            plan = self._plan_catalog.plan_for_id("P2")
            config_version = self._plan_catalog.sha256()
            selection_reason = "Stage 9 graph-only depth-1 baseline using immutable P2"
        elif graph_depth_override == 3:
            plan = self._plan_catalog.plan_for_id("P3")
            config_version = self._plan_catalog.sha256()
            selection_reason = "Stage 9 graph-only depth-3 baseline using immutable P3"
        elif graph_depth_override == 2:
            base = self._plan_catalog.plan_for_id("P3")
            plan = base.model_copy(update={"name": "GRAPH_DEPTH_2_BENCHMARK", "graph_depth": 2})
            config_version = benchmark_graph_depth_2_config_version(self._plan_catalog.sha256())
            selection_reason = (
                "Stage 9 graph-only depth-2 benchmark override; exact plan serialized in trace"
            )
        else:
            raise RAGPlanError(
                ErrorCode.PLAN_INVARIANT_VIOLATION,
                "graph depth override must be 1, 2, or 3",
                retryable=False,
            )
        executed_top_k = max(request.top_k, plan.graph_top_k)
        remaining_ms = deadline.snapshot().branch_remaining_ms
        if remaining_ms <= 0:
            raise RAGPlanError(ErrorCode.DEADLINE_EXCEEDED, "planning deadline exceeded")
        if graph_depth_override is None and request.top_k > plan.graph_top_k:
            selection_reason = "explicit graph mode; request-floor override derived from P3"
        return PlannerDecision(
            mode=PlannerMode.GRAPH,
            effective_mode=PlannerMode.GRAPH,
            selected_plan_id=plan.id,
            selected_plan=plan,
            executed_graph_top_k=executed_top_k,
            remaining_budget_ms=remaining_ms,
            budget_feasible=True,
            selection_reason=selection_reason,
            feature_version=FEATURE_VERSION,
            config_version=config_version,
        )

    def _select_fixed_plan(
        self,
        request: SearchRequest,
        deadline: Deadline,
    ) -> PlannerDecision:
        plan_id = request.plan_id or "P5"
        if plan_id not in self._FIXED_PLAN_IDS:
            raise RAGPlanError(
                ErrorCode.PLAN_INVARIANT_VIOLATION,
                "fixed_hybrid plan is not in the P0 catalog",
                retryable=False,
            )
        plan = self._plan_catalog.plan_for_id(plan_id)
        if not plan.enabled_in_p0 or not plan.vector_enabled or not plan.graph_enabled:
            raise RAGPlanError(
                ErrorCode.PLAN_INVARIANT_VIOLATION,
                "fixed_hybrid plan is not executable",
                retryable=False,
            )
        remaining_ms = deadline.snapshot().branch_remaining_ms
        if remaining_ms <= 0:
            raise RAGPlanError(ErrorCode.DEADLINE_EXCEEDED, "planning deadline exceeded")
        executed_vector_top_k = max(request.top_k, plan.vector_top_k)
        executed_graph_top_k = max(request.top_k, plan.graph_top_k)
        selection_reason = f"explicit fixed_hybrid mode; immutable {plan_id} preset"
        if request.top_k > min(plan.vector_top_k, plan.graph_top_k):
            selection_reason += "; request-floor candidate override"
        return PlannerDecision(
            mode=PlannerMode.FIXED_HYBRID,
            effective_mode=PlannerMode.FIXED_HYBRID,
            selected_plan_id=plan.id,
            selected_plan=plan,
            executed_vector_top_k=executed_vector_top_k,
            executed_graph_top_k=executed_graph_top_k,
            remaining_budget_ms=remaining_ms,
            budget_feasible=True,
            selection_reason=selection_reason,
            feature_version=FEATURE_VERSION,
            config_version=self._plan_catalog.sha256(),
        )

    @staticmethod
    def _require_shared_active_corpus(
        *,
        active_manifest: IngestionManifest,
        vector_stage: VectorStageManifest,
        graph_stage: GraphStageManifest,
        analyzer: GraphQueryAnalyzer,
    ) -> None:
        if active_manifest.activation_status.value != "active":
            raise ValueError("fixed hybrid search requires an active reconciled corpus")
        vector_matches = (
            vector_stage.corpus_version == active_manifest.corpus_version
            and vector_stage.chunk_count == active_manifest.qdrant_count
            and vector_stage.canonical_id_checksum == active_manifest.qdrant_id_checksum
            and vector_stage.embedding_model_revision == active_manifest.embedding_model_revision
        )
        graph_matches = (
            graph_stage.corpus_version == active_manifest.corpus_version
            and graph_stage.chunk_count == active_manifest.neo4j_count
            and graph_stage.canonical_id_checksum == active_manifest.neo4j_id_checksum
            and graph_stage.extractor_version == active_manifest.extractor_version
            and analyzer.extractor_version == active_manifest.extractor_version
        )
        if not vector_matches or not graph_matches:
            raise RAGPlanError(
                ErrorCode.CORPUS_INCONSISTENT,
                "vector, graph, and active corpus evidence do not match",
                retryable=False,
            )

    async def close(self) -> None:
        current = asyncio.current_task()
        async with self._lifecycle_lock:
            if self._closed:
                return
            close_owner = not self._closing
            if close_owner:
                self._closing = True
                active_requests = tuple(task for task in self._request_tasks if task is not current)
            else:
                active_requests = ()
        if not close_owner:
            await self._close_complete.wait()
            return
        await cancel_and_await(
            active_requests,
            cancel_message=CancellationReason.ENGINE_SHUTDOWN.value,
        )
        vector_error: BaseException | None = None
        try:
            await self._vector_backend.close()
        except BaseException as exc:
            vector_error = exc
        graph_error: BaseException | None = None
        try:
            await self._graph_backend.close()
        except BaseException as exc:
            graph_error = exc
        finally:
            async with self._lifecycle_lock:
                self._closed = True
                self._close_complete.set()
        if vector_error is not None:
            raise vector_error
        if graph_error is not None:
            raise graph_error

    async def readiness(self) -> EngineReadinessSnapshot:
        vector, graph = await asyncio.gather(
            self._vector_backend.health(),
            self._graph_backend.health(),
        )
        return EngineReadinessSnapshot(
            profile=RuntimeProfile.DUAL_STORE_ACTIVE,
            corpus_version=self._active_manifest.corpus_version,
            active_corpus=True,
            supported_modes=(
                PlannerMode.VECTOR,
                PlannerMode.GRAPH,
                PlannerMode.FIXED_HYBRID,
                PlannerMode.RULE,
            ),
            vector=vector,
            graph=graph,
        )


async def _embed_before_cutoff(
    embedder: QueryEmbedder, query: str, deadline: Deadline
) -> Sequence[float]:
    timeout_seconds = deadline.remaining_seconds(reserve_finalization=True)
    if timeout_seconds <= 0:
        raise RAGPlanError(
            ErrorCode.DEADLINE_EXCEEDED,
            "embedding deadline exceeded",
            timeout_origin=TimeoutOrigin.APPLICATION_DEADLINE,
        )
    try:
        async with asyncio.timeout(timeout_seconds):
            return await embedder.embed_query(query)
    except TimeoutError as exc:
        raise RAGPlanError(
            ErrorCode.DEADLINE_EXCEEDED,
            "embedding deadline exceeded",
            timeout_origin=TimeoutOrigin.APPLICATION_DEADLINE,
        ) from exc


def _stage3_features(*, token_count: int, final_top_k: int) -> QueryFeatures:
    return QueryFeatures(
        token_count=token_count,
        entity_count=0,
        entity_density=0.0,
        relation_signal=0.0,
        multi_hop_signal=0.0,
        comparison_signal=0.0,
        aggregation_signal=0.0,
        global_signal=0.0,
        final_top_k=final_top_k,
    )


def _is_p0_english(query: str) -> bool:
    """Conservatively mark non-ASCII-letter queries unsupported but still vector-safe."""

    return not any(character.isalpha() and not character.isascii() for character in query)


def _elapsed_ms(start_ns: int, end_ns: int) -> float:
    return max(0, end_ns - start_ns) / NANOSECONDS_PER_MILLISECOND
