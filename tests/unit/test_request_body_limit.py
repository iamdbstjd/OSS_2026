from __future__ import annotations

import asyncio

import httpx
import pytest
from fastapi import FastAPI, Request

from ragplan.api.middleware import RequestBodyLimitMiddleware
from ragplan.core.config import MAX_REQUEST_BODY_BYTES

pytestmark = pytest.mark.unit


def _app() -> FastAPI:
    app = FastAPI()
    app.add_middleware(RequestBodyLimitMiddleware)

    @app.post("/body")
    async def body_size(request: Request) -> dict[str, int]:
        return {"size": len(await request.body())}

    return app


async def _post(body: bytes, *, request_id: str | None = None) -> httpx.Response:
    headers = {"x-request-id": request_id} if request_id is not None else None
    transport = httpx.ASGITransport(app=_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.post("/body", content=body, headers=headers)


def test_exact_body_limit_is_accepted() -> None:
    response = asyncio.run(_post(b"x" * MAX_REQUEST_BODY_BYTES))

    assert response.status_code == 200
    assert response.json() == {"size": MAX_REQUEST_BODY_BYTES}


def test_oversized_body_uses_stable_error_schema() -> None:
    response = asyncio.run(_post(b"x" * (MAX_REQUEST_BODY_BYTES + 1), request_id="request-123"))

    assert response.status_code == 422
    assert response.json() == {
        "code": "INVALID_REQUEST",
        "message": f"request body exceeds {MAX_REQUEST_BODY_BYTES} bytes",
        "request_id": "request-123",
        "retryable": False,
    }
