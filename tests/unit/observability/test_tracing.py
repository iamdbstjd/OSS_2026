from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from ragplan.api.server import create_app
from ragplan.core.errors import ErrorCode, RAGPlanError
from ragplan.core.models import (
    BranchKind,
    BranchResult,
    BranchStatus,
    PlanDefinition,
    PlannerDecision,
    PlannerMode,
    QueryFeatures,
    RetrievalHit,
    SearchRequest,
    SearchResponse,
    SearchStatus,
    SearchTrace,
)
from ragplan.observability.tracing import (
    TRACE_FILE_COUNT,
    TRACE_MAX_BYTES,
    RedactedTraceWriter,
    TraceLoggingConfig,
    TraceWriterStats,
)

pytestmark = pytest.mark.unit


def _response() -> SearchResponse:
    plan = PlanDefinition(
        id="P0",
        name="VECTOR_FAST",
        vector_enabled=True,
        graph_enabled=False,
        vector_top_k=10,
        graph_top_k=0,
        graph_depth=0,
        vector_weight=1.0,
        graph_weight=0.0,
        rerank_enabled=False,
        enabled_in_p0=True,
    )
    decision = PlannerDecision(
        mode=PlannerMode.VECTOR,
        effective_mode=PlannerMode.VECTOR,
        selected_plan_id="P0",
        selected_plan=plan,
        executed_vector_top_k=10,
        remaining_budget_ms=100.0,
        feature_version="v1",
        config_version="config-v1",
    )
    features = QueryFeatures(
        token_count=2,
        entity_count=0,
        entity_density=0.0,
        relation_signal=0.0,
        multi_hop_signal=0.0,
        comparison_signal=0.0,
        aggregation_signal=0.0,
        global_signal=0.0,
        final_top_k=10,
    )
    hit = RetrievalHit(
        canonical_chunk_id="v1:chunk:trace:1",
        text="CONFIDENTIAL FULL DOCUMENT TEXT",
        score=1.0,
        source="vector",
        rank=1,
        metadata={"private": "SECRET METADATA"},
    )
    branch = BranchResult(
        branch=BranchKind.VECTOR,
        status=BranchStatus.SUCCEEDED,
        latency_ms=2.0,
        hits=(hit,),
    )
    trace = SearchTrace(
        request_id="trace-request",
        query_hash="a" * 64,
        query_length=18,
        language_supported=True,
        features=features,
        planner_decision=decision,
        branch_results=(branch,),
        analyzer_latency_ms=1.0,
        planner_latency_ms=1.0,
        vector_latency_ms=2.0,
        total_latency_ms=4.0,
        latency_budget_ms=200,
        finalization_reserve_ms=10.0,
        budget_feasible=True,
        budget_violated=False,
        fallback=False,
        result_count=1,
        corpus_version="trace-v1",
        config_version="config-v1",
        model_version="model-v1",
    )
    return SearchResponse(
        status=SearchStatus.COMPLETE,
        results=(hit,),
        planner_decision=decision,
        trace=trace,
        fallback=False,
        request_id="trace-request",
    )


@pytest.mark.asyncio
async def test_rotating_writer_serializes_only_redacted_bounded_trace(tmp_path: Path) -> None:
    path = tmp_path / "logs" / "trace.jsonl"
    writer = RedactedTraceWriter(TraceLoggingConfig(path=path))

    await writer.start()
    writer.record_search(_response())
    await writer.close()

    payload = json.loads(path.read_text(encoding="utf-8"))
    serialized = json.dumps(payload)
    assert payload["query_hash"] == "a" * 64
    assert payload["selected_plan_id"] == "P0"
    assert payload["branch_results"][0]["latency_ms"] == 2.0
    assert "CONFIDENTIAL" not in serialized
    assert "SECRET METADATA" not in serialized
    assert "query_embedding" not in serialized
    assert writer.config.max_bytes == TRACE_MAX_BYTES == 10 * 1024 * 1024
    assert writer.config.file_count == TRACE_FILE_COUNT == 5
    assert writer.stats().written == 1


@pytest.mark.asyncio
async def test_trace_path_failure_is_counted_and_never_raised(tmp_path: Path) -> None:
    blocked_parent = tmp_path / "not-a-directory"
    blocked_parent.write_text("file", encoding="utf-8")
    writer = RedactedTraceWriter(TraceLoggingConfig(path=blocked_parent / "trace.jsonl"))

    await writer.start()
    writer.record_search(_response())
    await writer.close()

    assert writer.stats().failures == 1


def test_service_rejects_every_non_redacted_logging_mode() -> None:
    for mode in ("raw", "benchmark", "full", "debug"):
        with pytest.raises(RAGPlanError) as captured:
            TraceLoggingConfig.from_environment({"RAGPLAN_LOGGING__MODE": mode})
        assert captured.value.code is ErrorCode.INVALID_REQUEST


def test_api_startup_rejects_raw_logging_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RAGPLAN_LOGGING__MODE", "raw")
    with pytest.raises(RAGPlanError, match="redacted mode only"):
        create_app()


class _ExplodingTraceWriter:
    async def start(self) -> None: ...

    async def close(self) -> None: ...

    def record_search(self, response: SearchResponse) -> None:
        del response
        raise OSError("trace disk failed")

    def record_error(
        self,
        *,
        request_id: str,
        error_code: ErrorCode,
        requested_planner: str | None,
    ) -> None:
        del request_id, error_code, requested_planner
        raise OSError("trace disk failed")

    def stats(self) -> TraceWriterStats:
        return TraceWriterStats(written=0, failures=1, dropped=0)


class _Engine:
    async def search(self, request: SearchRequest, *, request_id: str) -> SearchResponse:
        del request, request_id
        return _response()

    async def close(self) -> None: ...


@pytest.mark.asyncio
async def test_trace_write_failure_does_not_fail_search_and_updates_metrics(tmp_path: Path) -> None:
    app = create_app(
        search_engine=_Engine(),
        trace_writer=_ExplodingTraceWriter(),
        trace_config=TraceLoggingConfig(path=tmp_path / "unused.jsonl"),
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/v1/search",
            json={"query": "private question", "planner": "vector"},
        )
        metrics = await client.get("/metrics")

    assert response.status_code == 200
    assert metrics.json()["complete_count"] == 1
    assert metrics.json()["trace_write_failure_count"] == 1
