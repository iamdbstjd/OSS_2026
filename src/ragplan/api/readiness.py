"""Stable readiness response derived from backend-neutral engine health."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict

from ragplan.backends.base import BackendHealth, BackendHealthStatus
from ragplan.core.engine import ReadinessProvider, SearchEngine


class ServiceReadinessStatus(StrEnum):
    READY = "ready"
    DEGRADED = "degraded"
    NOT_READY = "not_ready"


class DependencyReadinessStatus(StrEnum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"
    NOT_CONFIGURED = "not_configured"


class DependencyReadiness(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    status: DependencyReadinessStatus


class ReadinessResponse(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    schema_version: Literal["readiness_v1"] = "readiness_v1"
    status: ServiceReadinessStatus
    runtime_profile: str | None
    corpus_version: str | None
    active_corpus: bool
    qdrant: DependencyReadiness
    neo4j: DependencyReadiness
    supported_modes: tuple[str, ...]
    default_planner: Literal["rule"] = "rule"
    graph_tier_enabled: bool
    graph_modes_available: bool
    cost_aware_status: Literal["research_only_disabled"] = "research_only_disabled"
    reason: str | None = None


async def inspect_readiness(
    engine: SearchEngine | None,
    *,
    graph_tier_enabled: bool,
) -> tuple[ReadinessResponse, int]:
    if engine is None:
        return _not_ready("search_engine_not_initialized", graph_tier_enabled), 503
    if not isinstance(engine, ReadinessProvider):
        return _not_ready("readiness_probe_unavailable", graph_tier_enabled), 503
    try:
        snapshot = await engine.readiness()
    except Exception:
        return _not_ready("readiness_probe_failed", graph_tier_enabled), 503

    vector = _dependency(snapshot.vector)
    graph = _dependency(snapshot.graph)
    vector_required = snapshot.vector is not None
    graph_required = snapshot.graph is not None and not vector_required
    vector_failed = vector_required and vector.status is DependencyReadinessStatus.UNAVAILABLE
    graph_failed = (
        snapshot.graph is not None and graph.status is DependencyReadinessStatus.UNAVAILABLE
    )
    required_failed = vector_failed or (graph_required and graph_failed)
    degraded = (
        graph_failed
        or vector.status is DependencyReadinessStatus.DEGRADED
        or graph.status is DependencyReadinessStatus.DEGRADED
    )
    status = (
        ServiceReadinessStatus.NOT_READY
        if required_failed
        else ServiceReadinessStatus.DEGRADED
        if degraded
        else ServiceReadinessStatus.READY
    )
    reason = None
    if vector_failed:
        reason = "qdrant_unavailable"
    elif graph_required and graph_failed:
        reason = "neo4j_unavailable"
    elif graph_failed:
        reason = "neo4j_unavailable_vector_modes_only"
    response = ReadinessResponse(
        status=status,
        runtime_profile=snapshot.profile.value,
        corpus_version=snapshot.corpus_version,
        active_corpus=snapshot.active_corpus,
        qdrant=vector,
        neo4j=graph,
        supported_modes=tuple(mode.value for mode in snapshot.supported_modes),
        graph_tier_enabled=graph_tier_enabled,
        graph_modes_available=(
            snapshot.graph is not None
            and graph.status
            in {DependencyReadinessStatus.HEALTHY, DependencyReadinessStatus.DEGRADED}
        ),
        reason=reason,
    )
    return response, 503 if status is ServiceReadinessStatus.NOT_READY else 200


def _dependency(health: BackendHealth | None) -> DependencyReadiness:
    if health is None:
        return DependencyReadiness(status=DependencyReadinessStatus.NOT_CONFIGURED)
    status = {
        BackendHealthStatus.HEALTHY: DependencyReadinessStatus.HEALTHY,
        BackendHealthStatus.DEGRADED: DependencyReadinessStatus.DEGRADED,
        BackendHealthStatus.UNAVAILABLE: DependencyReadinessStatus.UNAVAILABLE,
    }[health.status]
    return DependencyReadiness(status=status)


def _not_ready(reason: str, graph_tier_enabled: bool) -> ReadinessResponse:
    return ReadinessResponse(
        status=ServiceReadinessStatus.NOT_READY,
        runtime_profile=None,
        corpus_version=None,
        active_corpus=False,
        qdrant=DependencyReadiness(status=DependencyReadinessStatus.NOT_CONFIGURED),
        neo4j=DependencyReadiness(status=DependencyReadinessStatus.NOT_CONFIGURED),
        supported_modes=(),
        graph_tier_enabled=graph_tier_enabled,
        graph_modes_available=False,
        reason=reason,
    )


__all__ = [
    "DependencyReadiness",
    "DependencyReadinessStatus",
    "ReadinessResponse",
    "ServiceReadinessStatus",
    "inspect_readiness",
]
