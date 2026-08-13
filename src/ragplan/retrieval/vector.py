"""One deadline-bound vector retrieval execution."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ragplan.backends.vector.base import VectorBackend
from ragplan.core.deadline import NANOSECONDS_PER_MILLISECOND, Deadline
from ragplan.core.errors import ErrorCode, RAGPlanError, TimeoutOrigin
from ragplan.core.models import RetrievalHit


@dataclass(frozen=True, slots=True)
class VectorExecution:
    """Ordered vector hits and Qdrant-adapter latency for one branch."""

    hits: tuple[RetrievalHit, ...]
    latency_ms: float


async def execute_vector_search(
    *,
    backend: VectorBackend,
    embedding: Sequence[float],
    top_k: int,
    corpus_version: str,
    deadline: Deadline,
) -> VectorExecution:
    """Execute one vector call without refreshing the absolute deadline."""

    if deadline.remaining_seconds(reserve_finalization=True) <= 0:
        raise RAGPlanError(
            ErrorCode.DEADLINE_EXCEEDED,
            "retrieval deadline exceeded",
            timeout_origin=TimeoutOrigin.APPLICATION_DEADLINE,
        )

    started_ns = deadline.clock.now_ns()
    hits = await backend.search(embedding, top_k, corpus_version, deadline)
    finished_ns = deadline.clock.now_ns()
    latency_ms = max(0, finished_ns - started_ns) / NANOSECONDS_PER_MILLISECOND

    if finished_ns >= deadline.branch_cutoff_ns:
        raise RAGPlanError(
            ErrorCode.DEADLINE_EXCEEDED,
            "retrieval deadline exceeded",
            timeout_origin=TimeoutOrigin.APPLICATION_DEADLINE,
        )
    return VectorExecution(hits=tuple(hits), latency_ms=latency_ms)
