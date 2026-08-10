from __future__ import annotations

import asyncio

import httpx
import pytest

from ragplan.api.server import create_app

pytestmark = [pytest.mark.unit, pytest.mark.contract]


async def _post(payload: dict[str, object]) -> httpx.Response:
    transport = httpx.ASGITransport(app=create_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.post(
            "/v1/search",
            json=payload,
            headers={"x-request-id": "request-contract-1"},
        )


def test_search_schema_accepts_public_enum_strings_and_returns_stable_not_ready() -> None:
    response = asyncio.run(_post({"query": "Where was Ada born?", "planner": "graph"}))

    assert response.status_code == 503
    assert response.json() == {
        "code": "NOT_READY",
        "message": "search engine is not initialized",
        "request_id": "request-contract-1",
        "retryable": True,
    }


@pytest.mark.parametrize(
    ("payload", "expected_code"),
    [
        ({"query": "   "}, "INVALID_QUERY"),
        ({"query": "valid", "planner": "adaptive"}, "INVALID_REQUEST"),
        ({"query": "valid", "unknown": True}, "INVALID_REQUEST"),
        ({"query": "valid", "planner": "cost_aware", "top_k": 9}, "INVALID_REQUEST"),
    ],
)
def test_invalid_search_request_uses_stable_error_body(
    payload: dict[str, object], expected_code: str
) -> None:
    response = asyncio.run(_post(payload))

    assert response.status_code == 422
    assert response.json() == {
        "code": expected_code,
        "message": (
            "query is invalid" if expected_code == "INVALID_QUERY" else "request validation failed"
        ),
        "request_id": "request-contract-1",
        "retryable": False,
    }


def test_openapi_exposes_the_frozen_search_request_response_and_trace() -> None:
    schema = create_app().openapi()
    operation = schema["paths"]["/v1/search"]["post"]
    component_schemas = schema["components"]["schemas"]

    assert operation["requestBody"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/SearchRequest"
    }
    assert operation["responses"]["200"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/SearchResponse"
    }
    assert {"SearchRequest", "SearchResponse", "SearchTrace", "ErrorResponse"} <= set(
        component_schemas
    )


def test_unexpected_exception_is_redacted_to_stable_internal_error() -> None:
    app = create_app()

    @app.get("/boom")
    async def boom() -> None:
        raise RuntimeError("sensitive backend credential")

    async def request() -> httpx.Response:
        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.get("/boom", headers={"x-request-id": "request-error-1"})

    response = asyncio.run(request())

    assert response.status_code == 500
    assert response.json() == {
        "code": "INTERNAL_ERROR",
        "message": "internal server error",
        "request_id": "request-error-1",
        "retryable": False,
    }
    assert "credential" not in response.text
