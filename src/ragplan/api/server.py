"""FastAPI bootstrap for shared contracts and optional verified retrieval runtimes."""

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
from ragplan.api.readiness import ReadinessResponse, inspect_readiness
from ragplan.api.runtime import build_search_engine_from_environment
from ragplan.core.engine import SearchEngine
from ragplan.core.errors import ErrorCode, ErrorResponse, RAGPlanError
from ragplan.core.models import SearchRequest, SearchResponse
from ragplan.ingestion.audit import load_graph_tier_policy
from ragplan.observability.metrics import MetricsRegistry, MetricsSnapshot
from ragplan.observability.tracing import (
    RedactedTraceWriter,
    TraceLoggingConfig,
    TraceWriter,
)
from ragplan.planner.catalog import load_default_plan_catalog, load_plan_catalog
from ragplan.scheduler.cancellation import run_until_disconnect

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


async def _wait_for_http_disconnect(request: Request) -> None:
    """Consume only post-body ASGI events until the peer disconnects."""

    while True:
        message = await request.receive()
        if message["type"] == "http.disconnect":
            return


def create_app(
    *,
    plan_catalog_path: Path | None = None,
    search_engine: SearchEngine | None = None,
    runtime_factory: Callable[[], Awaitable[SearchEngine | None]] | None = None,
    metrics_registry: MetricsRegistry | None = None,
    trace_writer: TraceWriter | None = None,
    trace_config: TraceLoggingConfig | None = None,
) -> FastAPI:
    """Create the local single-process RAGPlan API application."""

    plan_catalog = (
        load_plan_catalog(plan_catalog_path)
        if plan_catalog_path is not None
        else load_default_plan_catalog()
    )
    selected_metrics = metrics_registry if metrics_registry is not None else MetricsRegistry()
    selected_trace_config = (
        trace_config if trace_config is not None else TraceLoggingConfig.from_environment()
    )
    selected_trace_writer = (
        trace_writer
        if trace_writer is not None
        else RedactedTraceWriter(
            selected_trace_config,
            on_failure=selected_metrics.record_trace_write_failure,
            on_drop=selected_metrics.record_trace_drop,
        )
    )

    @asynccontextmanager
    async def lifespan(api: FastAPI) -> AsyncIterator[None]:
        owned_engine: SearchEngine | None = None
        await _safe_trace_start(api)
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
            try:
                if owned_engine is not None:
                    await owned_engine.close()
                    api.state.search_engine = None
                elif search_engine is not None:
                    await search_engine.close()
            finally:
                await _safe_trace_close(api)

    api = FastAPI(title="RAGPlan", version=__version__, lifespan=lifespan)
    api.add_middleware(RequestBodyLimitMiddleware)
    api.state.plan_catalog = plan_catalog
    api.state.search_engine = search_engine
    api.state.metrics = selected_metrics
    api.state.trace_writer = selected_trace_writer
    api.state.graph_tier_policy = load_graph_tier_policy()

    @api.exception_handler(RAGPlanError)
    async def ragplan_error_handler(request: Request, error: RAGPlanError) -> JSONResponse:
        _record_error_metric(request, error.code)
        return _error_response(error, request)

    @api.exception_handler(RequestValidationError)
    async def request_validation_handler(
        request: Request, error: RequestValidationError
    ) -> JSONResponse:
        invalid_query = any("query" in item["loc"] for item in error.errors())
        code = ErrorCode.INVALID_QUERY if invalid_query else ErrorCode.INVALID_REQUEST
        message = "query is invalid" if invalid_query else "request validation failed"
        _record_error_metric(request, code)
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
        _record_error_metric(request, internal_error.code)
        return JSONResponse(
            status_code=internal_error.http_status,
            content=internal_error.response(request_id).model_dump(mode="json"),
        )

    @api.get("/health", response_model=HealthResponse, tags=["health"])
    async def health() -> HealthResponse:
        return HealthResponse(status="ok")

    @api.get(
        "/ready",
        response_model=ReadinessResponse,
        responses={503: {"model": ReadinessResponse, "description": "Runtime not ready"}},
        tags=["health"],
    )
    async def ready() -> JSONResponse:
        engine = cast(SearchEngine | None, api.state.search_engine)
        response, status_code = await inspect_readiness(
            engine,
            graph_tier_enabled=api.state.graph_tier_policy.graph_tier_enabled,
        )
        return JSONResponse(
            status_code=status_code,
            content=response.model_dump(mode="json"),
        )

    @api.get("/metrics", response_model=MetricsSnapshot, tags=["observability"])
    async def metrics() -> MetricsSnapshot:
        registry = cast(MetricsRegistry, api.state.metrics)
        return registry.snapshot()

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

        registry = cast(MetricsRegistry, api.state.metrics)
        registry.record_request(search_request.planner)
        request.scope["ragplan_metrics_counted"] = True
        request.scope["ragplan_requested_planner"] = search_request.planner.value
        engine = cast(SearchEngine | None, api.state.search_engine)
        if engine is None:
            raise RAGPlanError(ErrorCode.NOT_READY, "search engine is not initialized")
        response = await run_until_disconnect(
            engine.search(
                search_request,
                request_id=request_id_from_scope(request.scope),
            ),
            wait_for_disconnect=lambda: _wait_for_http_disconnect(request),
        )
        registry.record_success(response)
        _safe_trace_search(api, response)
        return response

    return api


def _record_error_metric(request: Request, code: ErrorCode) -> None:
    if request.url.path != "/v1/search":
        return
    registry = cast(MetricsRegistry, request.app.state.metrics)
    if not request.scope.get("ragplan_metrics_counted"):
        registry.record_request(None)
        request.scope["ragplan_metrics_counted"] = True
    registry.record_error(code)
    if not request.scope.get("ragplan_trace_error_recorded"):
        writer = cast(TraceWriter, request.app.state.trace_writer)
        try:
            writer.record_error(
                request_id=request_id_from_scope(request.scope),
                error_code=code,
                requested_planner=request.scope.get("ragplan_requested_planner"),
            )
        except Exception:
            registry.record_trace_write_failure()
        request.scope["ragplan_trace_error_recorded"] = True


def _safe_trace_search(api: FastAPI, response: SearchResponse) -> None:
    try:
        cast(TraceWriter, api.state.trace_writer).record_search(response)
    except Exception:
        cast(MetricsRegistry, api.state.metrics).record_trace_write_failure()


async def _safe_trace_start(api: FastAPI) -> None:
    try:
        await cast(TraceWriter, api.state.trace_writer).start()
    except Exception:
        cast(MetricsRegistry, api.state.metrics).record_trace_write_failure()


async def _safe_trace_close(api: FastAPI) -> None:
    try:
        await cast(TraceWriter, api.state.trace_writer).close()
    except Exception:
        cast(MetricsRegistry, api.state.metrics).record_trace_write_failure()


app = create_app()
