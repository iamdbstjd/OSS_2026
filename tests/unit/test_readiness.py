from __future__ import annotations

import httpx
import pytest

from ragplan.api.server import create_app
from ragplan.backends.base import BackendHealth, BackendHealthStatus
from ragplan.core.health import EngineReadinessSnapshot, RuntimeProfile
from ragplan.core.models import PlannerMode, SearchRequest, SearchResponse

pytestmark = pytest.mark.unit


class _ReadinessEngine:
    def __init__(self, snapshot: EngineReadinessSnapshot) -> None:
        self._snapshot = snapshot

    async def readiness(self) -> EngineReadinessSnapshot:
        return self._snapshot

    async def search(self, request: SearchRequest, *, request_id: str) -> SearchResponse:
        del request, request_id
        raise AssertionError("search is not expected")

    async def close(self) -> None: ...


async def _ready(engine: object | None) -> httpx.Response:
    app = create_app(search_engine=engine)  # type: ignore[arg-type]
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.get("/ready")


@pytest.mark.asyncio
async def test_no_engine_is_not_ready_but_health_remains_liveness() -> None:
    app = create_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        health = await client.get("/health")
        ready = await client.get("/ready")

    assert health.status_code == 200
    assert ready.status_code == 503
    assert ready.json()["reason"] == "search_engine_not_initialized"


@pytest.mark.asyncio
async def test_dual_store_neo4j_outage_is_degraded_vector_only() -> None:
    response = await _ready(
        _ReadinessEngine(
            EngineReadinessSnapshot(
                profile=RuntimeProfile.DUAL_STORE_ACTIVE,
                corpus_version="fixture-v1",
                active_corpus=True,
                supported_modes=(
                    PlannerMode.VECTOR,
                    PlannerMode.GRAPH,
                    PlannerMode.FIXED_HYBRID,
                    PlannerMode.RULE,
                ),
                vector=BackendHealth(BackendHealthStatus.HEALTHY),
                graph=BackendHealth(BackendHealthStatus.UNAVAILABLE, "secret backend detail"),
            )
        )
    )

    assert response.status_code == 200
    assert response.json()["status"] == "degraded"
    assert response.json()["reason"] == "neo4j_unavailable_vector_modes_only"
    assert response.json()["graph_modes_available"] is False
    assert "secret" not in response.text


@pytest.mark.asyncio
async def test_dual_store_qdrant_outage_is_not_ready() -> None:
    response = await _ready(
        _ReadinessEngine(
            EngineReadinessSnapshot(
                profile=RuntimeProfile.DUAL_STORE_ACTIVE,
                corpus_version="fixture-v1",
                active_corpus=True,
                supported_modes=(PlannerMode.VECTOR, PlannerMode.RULE),
                vector=BackendHealth(BackendHealthStatus.UNAVAILABLE),
                graph=BackendHealth(BackendHealthStatus.HEALTHY),
            )
        )
    )

    assert response.status_code == 503
    assert response.json()["status"] == "not_ready"
    assert response.json()["reason"] == "qdrant_unavailable"
