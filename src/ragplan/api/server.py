"""FastAPI bootstrap for the shared contracts and optional Stage 3 vector runtime."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, cast

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from ragplan import __version__
from ragplan.api.middleware import RequestBodyLimitMiddleware, request_id_from_scope
from ragplan.api.runtime import build_search_engine_from_environment
from ragplan.core.engine import SearchEngine
from ragplan.core.errors import ErrorCode, ErrorResponse, RAGPlanError
from ragplan.core.models import SearchRequest, SearchResponse
from ragplan.planner.catalog import load_default_plan_catalog, load_plan_catalog

logger = logging.getLogger(__name__)


class HealthResponse(BaseModel):
    """Liveness response returned without checking external dependencies."""

    status: Literal["ok"]


def _error_response(error: RAGPlanError, request: Request) -> JSONResponse:
    request_id = request_id_from_scope(request.scope)
    return JSONResponse(
        status_code=error.http_status,
        content=error.response(request_id).model_dump(mode="json"),
    )


def create_app(
    *,
    plan_catalog_path: Path | None = None,
    search_engine: SearchEngine | None = None,
    runtime_factory: Callable[[], Awaitable[SearchEngine | None]] | None = None,
) -> FastAPI:
    """Create the local single-process RAGPlan API application."""

    plan_catalog = (
        load_plan_catalog(plan_catalog_path)
        if plan_catalog_path is not None
        else load_default_plan_catalog()
    )

    @asynccontextmanager
    async def lifespan(api: FastAPI) -> AsyncIterator[None]:
        owned_engine: SearchEngine | None = None
        try:
            if api.state.search_engine is None:
                if runtime_factory is not None:
                    owned_engine = await runtime_factory()
                else:
                    owned_engine = await build_search_engine_from_environment(
                        plan_catalog=plan_catalog
                    )
                api.state.search_engine = owned_engine
            yield
        finally:
            if owned_engine is not None:
                await owned_engine.close()
                api.state.search_engine = None
            elif search_engine is not None:
                await search_engine.close()

    api = FastAPI(title="RAGPlan", version=__version__, lifespan=lifespan)
    api.add_middleware(RequestBodyLimitMiddleware)
    api.state.plan_catalog = plan_catalog
    api.state.search_engine = search_engine

    @api.exception_handler(RAGPlanError)
    async def ragplan_error_handler(request: Request, error: RAGPlanError) -> JSONResponse:
        return _error_response(error, request)

    @api.exception_handler(RequestValidationError)
    async def request_validation_handler(
        request: Request, error: RequestValidationError
    ) -> JSONResponse:
        invalid_query = any("query" in item["loc"] for item in error.errors())
        code = ErrorCode.INVALID_QUERY if invalid_query else ErrorCode.INVALID_REQUEST
        message = "query is invalid" if invalid_query else "request validation failed"
        return _error_response(RAGPlanError(code, message), request)

    @api.exception_handler(Exception)
    async def internal_error_handler(request: Request, error: Exception) -> JSONResponse:
        request_id = request_id_from_scope(request.scope)
        logger.error(
            "Unhandled API error request_id=%s exception_type=%s",
            request_id,
            type(error).__name__,
        )
        internal_error = RAGPlanError(ErrorCode.INTERNAL_ERROR, "internal server error")
        return JSONResponse(
            status_code=internal_error.http_status,
            content=internal_error.response(request_id).model_dump(mode="json"),
        )

    @api.get("/health", response_model=HealthResponse, tags=["health"])
    async def health() -> HealthResponse:
        return HealthResponse(status="ok")

    @api.post(
        "/v1/search",
        response_model=SearchResponse,
        responses={
            422: {"model": ErrorResponse, "description": "Invalid request"},
            503: {"model": ErrorResponse, "description": "Engine not ready"},
            504: {"model": ErrorResponse, "description": "Deadline exceeded"},
        },
        tags=["search"],
    )
    async def search_contract(request: Request, search_request: SearchRequest) -> SearchResponse:
        """Run the configured engine while retaining a stable unready state."""

        engine = cast(SearchEngine | None, api.state.search_engine)
        if engine is None:
            raise RAGPlanError(ErrorCode.NOT_READY, "search engine is not initialized")
        return await engine.search(
            search_request,
            request_id=request_id_from_scope(request.scope),
        )

    return api


app = create_app()
