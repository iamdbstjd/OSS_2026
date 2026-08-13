"""Single-pass Stage 8 query analysis with one entity extraction and embedding."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, Protocol

from ragplan.core.deadline import Deadline, MonotonicClock
from ragplan.core.errors import ErrorCode, RAGPlanError
from ragplan.core.models import QueryAnalysis
from ragplan.planner.features import (
    FEATURE_SCHEMA_VERSION,
    QueryFeatureConfig,
    extract_query_features,
)

ANALYZER_VERSION: Final = f"stage8-query-analyzer:{FEATURE_SCHEMA_VERSION}"


class EntityQueryAnalyzer(Protocol):
    def analyze(self, query: str, *, final_top_k: int) -> QueryAnalysis: ...


class AsyncQueryEmbedder(Protocol):
    async def embed_query(self, query: str) -> Sequence[float]: ...


@dataclass(frozen=True, slots=True)
class AnalyzerExecution:
    analysis: QueryAnalysis
    embedding_latency_ms: float
    feature_config_sha256: str


class QueryAnalyzer:
    """Compose the pinned extractor and embedder without duplicate work."""

    def __init__(
        self,
        *,
        entity_analyzer: EntityQueryAnalyzer,
        embedder: AsyncQueryEmbedder,
        feature_config: QueryFeatureConfig,
        clock: MonotonicClock,
    ) -> None:
        self._entity_analyzer = entity_analyzer
        self._embedder = embedder
        self._feature_config = feature_config
        self._clock = clock

    @property
    def feature_config_sha256(self) -> str:
        return self._feature_config.sha256

    async def analyze(
        self,
        query: str,
        *,
        final_top_k: int,
        deadline: Deadline,
    ) -> AnalyzerExecution:
        """Produce reusable analysis; query text and embedding remain internal only."""

        feature_started_ns = self._clock.now_ns()
        base = self._entity_analyzer.analyze(query, final_top_k=final_top_k)
        features = extract_query_features(
            base.normalized_query,
            token_count=base.token_count,
            entity_count=len(base.seed_entity_ids),
            final_top_k=final_top_k,
            config=self._feature_config,
        )
        feature_finished_ns = self._clock.now_ns()
        analysis_latency_ms = max(
            base.analysis_latency_ms,
            (feature_finished_ns - feature_started_ns) / 1_000_000,
        )
        timeout_seconds = deadline.remaining_seconds(reserve_finalization=True)
        if timeout_seconds <= 0:
            raise RAGPlanError(ErrorCode.DEADLINE_EXCEEDED, "query analysis deadline exceeded")
        embedding_started_ns = self._clock.now_ns()
        try:
            async with asyncio.timeout(timeout_seconds):
                embedding = await self._embedder.embed_query(base.normalized_query)
        except TimeoutError as exc:
            raise RAGPlanError(ErrorCode.DEADLINE_EXCEEDED, "embedding deadline exceeded") from exc
        embedding_finished_ns = self._clock.now_ns()
        if embedding_finished_ns >= deadline.branch_cutoff_ns:
            raise RAGPlanError(ErrorCode.DEADLINE_EXCEEDED, "embedding deadline exceeded")
        analysis = base.model_copy(
            update={
                "features": features,
                "query_embedding": tuple(float(value) for value in embedding),
                "analyzer_version": f"{ANALYZER_VERSION}:{base.analyzer_version}",
                "analysis_latency_ms": analysis_latency_ms,
            }
        )
        return AnalyzerExecution(
            analysis=analysis,
            embedding_latency_ms=(embedding_finished_ns - embedding_started_ns) / 1_000_000,
            feature_config_sha256=self._feature_config.sha256,
        )
