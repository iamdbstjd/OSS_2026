from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import httpx
import pytest

from ragplan.api.readiness import ReadinessResponse, ServiceReadinessStatus
from ragplan.core.errors import ErrorCode, ErrorResponse
from ragplan.core.models import BranchKind, BranchStatus, PlannerMode, SearchResponse, SearchStatus
from ragplan.observability.metrics import MetricsSnapshot

ROOT = Path(__file__).resolve().parents[2]
TRUE_VALUES = {"1", "true", "yes", "on"}
pytestmark = [pytest.mark.qa, pytest.mark.e2e]


def _enabled(name: str) -> bool:
    return os.environ.get(name, "").strip().casefold() in TRUE_VALUES


@pytest.fixture(scope="module")
def api_url() -> str:
    if not _enabled("RAGPLAN_QA_LIVE"):
        pytest.skip("set RAGPLAN_QA_LIVE=1 to run live Compose QA")
    return os.environ.get("RAGPLAN_QA_API_URL", "http://127.0.0.1:8000").rstrip("/")


def _get(api_url: str, path: str) -> httpx.Response:
    return httpx.get(f"{api_url}{path}", timeout=10.0)


def _search(api_url: str, payload: dict[str, object]) -> httpx.Response:
    return httpx.post(f"{api_url}/v1/search", json=payload, timeout=15.0)


def _wait_for_readiness(
    api_url: str,
    predicate: Callable[[ReadinessResponse], bool],
    *,
    timeout_seconds: float = 90.0,
) -> ReadinessResponse:
    deadline = time.monotonic() + timeout_seconds
    last_status = "no response"
    while time.monotonic() < deadline:
        try:
            response = _get(api_url, "/ready")
            readiness = ReadinessResponse.model_validate_json(response.content)
            last_status = readiness.model_dump_json()
            if predicate(readiness):
                return readiness
        except (httpx.HTTPError, ValueError) as exc:
            last_status = type(exc).__name__
        time.sleep(1.0)
    pytest.fail(f"readiness condition was not reached: {last_status}")


def _compose(*arguments: str) -> subprocess.CompletedProcess[str]:
    if shutil.which("docker") is None:
        pytest.fail("docker is required when failure injection is enabled")
    return subprocess.run(
        ["docker", "compose", *arguments],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )


def _assert_compose_success(result: subprocess.CompletedProcess[str]) -> None:
    assert result.returncode == 0, result.stderr[-2_000:]


def _all_keys(value: object) -> Iterator[str]:
    if isinstance(value, dict):
        for key, item in value.items():
            yield str(key)
            yield from _all_keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _all_keys(item)


def test_public_surface_is_ready_for_the_dual_store_demo(api_url: str) -> None:
    health = _get(api_url, "/health")
    assert health.status_code == 200
    assert health.json() == {"status": "ok"}

    ready_raw = _get(api_url, "/ready")
    assert ready_raw.status_code == 200, ready_raw.text
    ready = ReadinessResponse.model_validate_json(ready_raw.content)
    assert ready.status is ServiceReadinessStatus.READY
    assert ready.active_corpus is True
    assert ready.qdrant.status.value == "healthy"
    assert ready.neo4j.status.value == "healthy"
    assert ready.graph_modes_available is True
    assert ready.cost_aware_status == "research_only_disabled"

    metrics_raw = _get(api_url, "/metrics")
    assert metrics_raw.status_code == 200
    MetricsSnapshot.model_validate_json(metrics_raw.content)


def test_fixed_hybrid_p5_is_real_parallel_retrieval_and_updates_metrics(api_url: str) -> None:
    query = os.environ.get(
        "RAGPLAN_QA_HYBRID_QUERY",
        "Who collaborated with Ada Lovelace?",
    )
    payload: dict[str, object] = {
        "query": query,
        "planner": "fixed_hybrid",
        "plan_id": "P5",
        "latency_budget_ms": 500,
        "top_k": 3,
    }
    warmup_raw = _search(api_url, {**payload, "latency_budget_ms": 5_000})
    assert warmup_raw.status_code == 200, warmup_raw.text
    warmup = SearchResponse.model_validate_json(warmup_raw.content)
    assert warmup.status is SearchStatus.COMPLETE
    assert all(item.status is BranchStatus.SUCCEEDED for item in warmup.trace.branch_results)

    before = MetricsSnapshot.model_validate_json(_get(api_url, "/metrics").content)
    raw = _search(api_url, payload)

    assert raw.status_code == 200, raw.text
    response = SearchResponse.model_validate_json(raw.content)
    assert response.status is SearchStatus.COMPLETE
    assert response.planner_decision.mode is PlannerMode.FIXED_HYBRID
    assert response.planner_decision.effective_mode is PlannerMode.FIXED_HYBRID
    assert response.planner_decision.selected_plan_id == "P5"
    assert response.results
    assert {item.branch for item in response.trace.branch_results} == {
        BranchKind.VECTOR,
        BranchKind.GRAPH,
    }
    assert all(item.status is BranchStatus.SUCCEEDED for item in response.trace.branch_results)
    branch_results = {item.branch: item for item in response.trace.branch_results}
    assert branch_results[BranchKind.VECTOR].hit_count > 0
    assert branch_results[BranchKind.GRAPH].hit_count > 0
    assert response.trace.fusion_trace is not None
    assert response.trace.fusion_trace.fusion_version == "weighted_rrf_v1"
    assert response.trace.fusion_trace.vector_input_count > 0
    assert response.trace.fusion_trace.graph_input_count > 0
    assert any(BranchKind.GRAPH in item.sources for item in response.results)
    trace_keys = set(_all_keys(response.trace.model_dump(mode="json")))
    assert "raw_query" not in trace_keys
    assert "query_embedding" not in trace_keys

    after = MetricsSnapshot.model_validate_json(_get(api_url, "/metrics").content)
    assert after.request_count == before.request_count + 1
    assert after.planner_distribution.get("fixed_hybrid", 0) == (
        before.planner_distribution.get("fixed_hybrid", 0) + 1
    )
    assert after.total_latency.count == before.total_latency.count + 1


def test_public_cost_aware_serving_fails_closed(api_url: str) -> None:
    raw = _search(
        api_url,
        {
            "query": "Which retrieval plan should run?",
            "planner": "cost_aware",
            "latency_budget_ms": 500,
            "top_k": 10,
        },
    )

    assert raw.status_code == 503
    error = ErrorResponse.model_validate_json(raw.content)
    assert error.code is ErrorCode.MODE_UNAVAILABLE


def test_vector_quickstart_ranks_ada_evidence_first(api_url: str) -> None:
    del api_url
    if not _enabled("RAGPLAN_QA_RUN_QUICKSTART"):
        pytest.skip("set RAGPLAN_QA_RUN_QUICKSTART=1 to run the model-backed quickstart")
    result = subprocess.run(
        [sys.executable, "-m", "ragplan", "quickstart-vector"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
    )

    assert result.returncode == 0, result.stderr[-2_000:]
    payload = json.loads(result.stdout)
    assert payload["infrastructure"] == "qdrant_plus_minilm"
    assert payload["search"]["planner_decision"]["selected_plan_id"] == "P0"
    assert payload["search"]["results"]
    assert "ada lovelace" in payload["search"]["results"][0]["text"].casefold()


def test_neo4j_outage_degrades_to_vector_and_recovers(api_url: str) -> None:
    if not _enabled("RAGPLAN_QA_ALLOW_FAILURE_INJECTION"):
        pytest.skip("set RAGPLAN_QA_ALLOW_FAILURE_INJECTION=1 to stop and recover Neo4j")

    stopped = False
    try:
        stop = _compose("stop", "neo4j")
        _assert_compose_success(stop)
        stopped = True
        degraded = _wait_for_readiness(
            api_url,
            lambda item: (
                item.status is ServiceReadinessStatus.DEGRADED
                and item.qdrant.status.value == "healthy"
                and item.neo4j.status.value == "unavailable"
                and not item.graph_modes_available
            ),
        )
        assert degraded.reason == "neo4j_unavailable_vector_modes_only"

        rule_raw = _search(
            api_url,
            {
                "query": "What is a vector database?",
                "planner": "rule",
                "latency_budget_ms": 500,
                "top_k": 3,
            },
        )
        assert rule_raw.status_code == 200, rule_raw.text
        rule_response = SearchResponse.model_validate_json(rule_raw.content)
        assert rule_response.results
        assert {item.branch for item in rule_response.trace.branch_results} == {BranchKind.VECTOR}

        graph_raw = _search(
            api_url,
            {
                "query": "What did Ada Lovelace write about?",
                "planner": "graph",
                "latency_budget_ms": 5_000,
                "top_k": 3,
            },
        )
        assert graph_raw.status_code == 503
        graph_error = ErrorResponse.model_validate_json(graph_raw.content)
        assert graph_error.code in {ErrorCode.MODE_UNAVAILABLE, ErrorCode.DEPENDENCY_UNAVAILABLE}
    finally:
        if stopped:
            start = _compose("start", "neo4j")
            _assert_compose_success(start)
            recovered = _wait_for_readiness(
                api_url,
                lambda item: (
                    item.status is ServiceReadinessStatus.READY
                    and item.neo4j.status.value == "healthy"
                    and item.graph_modes_available
                ),
                timeout_seconds=120.0,
            )
            assert recovered.active_corpus is True
