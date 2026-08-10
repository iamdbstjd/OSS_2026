import asyncio

import httpx
import pytest

from ragplan.api.server import create_app

pytestmark = pytest.mark.unit


async def _get_health() -> httpx.Response:
    transport = httpx.ASGITransport(app=create_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.get("/health")


def test_health_is_a_liveness_endpoint() -> None:
    response = asyncio.run(_get_health())

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
