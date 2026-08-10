"""Stage 3 vector-only engine path shared by API and local scripts."""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from ragplan.backends.vector.base import VectorBackend
from ragplan.core.deadline import (
    NANOSECONDS_PER_MILLISECOND,
    Deadline,
    MonotonicClock,
    PerfCounterClock,
)
from ragplan.core.errors import ErrorCode, RAGPlanError
from ragplan.core.models import (
    BranchKind,
    BranchResult,
    BranchStatus,
    PlannerDecision,
    PlannerMode,
    QueryAnalysis,
    QueryFeatures,
    SearchRequest,
    SearchResponse,
    SearchStatus,
    SearchTrace,
    VectorStageManifest,
)
from ragplan.ingestion.chunker import Tokenizer
from ragplan.ingestion.normalize import normalize_text
from ragplan.planner.catalog import PlanCatalog
from ragplan.retrieval.vector import execute_vector_search

ANALYZER_VERSION = "stage3-vector-v1"
FEATURE_VERSION = "v1"


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
    ) -> None:
        self._embedder = embedder
        self._vector_backend = vector_backend
        self._plan_catalog = plan_catalog
        self._vector_stage = vector_stage
        self._corpus_version = vector_stage.corpus_version
        self._model_revision = vector_stage.embedding_model_revision
        self._clock = clock if clock is not None else PerfCounterClock()

    async def search(self, request: SearchRequest, *, request_id: str) -> SearchResponse:
        """Analyze, embed once, retrieve, and return a redacted Stage 3 trace."""

        deadline = Deadline.start(request.latency_budget_ms, clock=self._clock)
        analysis_started_ns = self._clock.now_ns()
        normalized_query = normalize_text(request.query)
        token_count = self._embedder.tokenizer.encode(normalized_query).token_count
        language_supported = _is_p0_english(normalized_query)
        features = _stage3_features(token_count=token_count, final_top_k=request.top_k)
        analysis_latency_ms = _elapsed_ms(analysis_started_ns, self._clock.now_ns())

        planner_started_ns = self._clock.now_ns()
        decision = self._select_vector_plan(request, deadline)
        planner_latency_ms = _elapsed_ms(planner_started_ns, self._clock.now_ns())

        embedding_started_ns = self._clock.now_ns()
        embedding = await _embed_before_cutoff(self._embedder, normalized_query, deadline)
        embedding_finished_ns = self._clock.now_ns()
        embedding_latency_ms = _elapsed_ms(embedding_started_ns, embedding_finished_ns)
        if embedding_finished_ns >= deadline.branch_cutoff_ns:
            raise RAGPlanError(ErrorCode.DEADLINE_EXCEEDED, "embedding deadline exceeded")

        analysis = QueryAnalysis(
            normalized_query=normalized_query,
            language_supported=language_supported,
            token_count=token_count,
            query_embedding=tuple(float(value) for value in embedding),
            features=features,
            analyzer_version=ANALYZER_VERSION,
            analysis_latency_ms=analysis_latency_ms,
        )
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
            config_version=self._plan_catalog.sha256(),
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

    def _select_vector_plan(self, request: SearchRequest, deadline: Deadline) -> PlannerDecision:
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


async def _embed_before_cutoff(
    embedder: QueryEmbedder, query: str, deadline: Deadline
) -> Sequence[float]:
    timeout_seconds = deadline.remaining_seconds(reserve_finalization=True)
    if timeout_seconds <= 0:
        raise RAGPlanError(ErrorCode.DEADLINE_EXCEEDED, "embedding deadline exceeded")
    try:
        async with asyncio.timeout(timeout_seconds):
            return await embedder.embed_query(query)
    except TimeoutError as exc:
        raise RAGPlanError(ErrorCode.DEADLINE_EXCEEDED, "embedding deadline exceeded") from exc


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
