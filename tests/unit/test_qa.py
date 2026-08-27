from __future__ import annotations

from types import SimpleNamespace

import pytest

from ragplan.api.readiness import (
    DependencyReadiness,
    DependencyReadinessStatus,
    ReadinessResponse,
    ServiceReadinessStatus,
)
from ragplan.cli import services
from ragplan.cli.services import QALevel, run_qa

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_vector_qa_uses_only_packaged_sample_and_fixed_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_quickstart(**kwargs: object) -> object:
        assert kwargs["input_path"] is None
        assert kwargs["query"] == "What did Ada Lovelace write about?"
        return SimpleNamespace(
            search=SimpleNamespace(
                results=(object(),),
                status=SimpleNamespace(value="complete"),
                planner_decision=SimpleNamespace(selected_plan_id="P0"),
            )
        )

    monkeypatch.setattr(services, "quickstart_vector", fake_quickstart)
    report = await run_qa(QALevel.VECTOR)

    assert report.status == "passed"
    assert report.held_out_test_accessed is False
    assert report.checks[-1].name == "vector_e2e"
    assert report.checks[-1].detail == "plan=P0;results=1"


@pytest.mark.asyncio
async def test_full_qa_uses_public_readiness_and_one_fixed_rule_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    readiness = ReadinessResponse(
        status=ServiceReadinessStatus.READY,
        runtime_profile="dual_store_active",
        corpus_version="fixture-v1",
        active_corpus=True,
        qdrant=DependencyReadiness(status=DependencyReadinessStatus.HEALTHY),
        neo4j=DependencyReadiness(status=DependencyReadinessStatus.HEALTHY),
        supported_modes=("vector", "graph", "fixed_hybrid", "rule"),
        graph_tier_enabled=False,
        graph_modes_available=True,
    )
    monkeypatch.setattr(services, "fetch_http_readiness", lambda url: readiness)

    def fake_search(request: object, **kwargs: object) -> object:
        assert getattr(request, "query") == "What is a vector database?"
        assert kwargs["api_url"] == "http://service:8000"
        return SimpleNamespace(
            results=(object(),),
            status=SimpleNamespace(value="complete"),
            planner_decision=SimpleNamespace(selected_plan_id="P1"),
        )

    monkeypatch.setattr(services, "search_http_api", fake_search)
    report = await run_qa(QALevel.FULL, api_url="http://service:8000")

    assert report.status == "passed"
    assert report.held_out_test_accessed is False
    assert report.checks[-1].detail == "profile=dual_store_active;plan=P1;results=1"


@pytest.mark.asyncio
async def test_qa_returns_failed_report_with_safe_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fail(**kwargs: object) -> object:
        del kwargs
        raise RuntimeError("sensitive backend detail")

    monkeypatch.setattr(services, "quickstart_vector", fail)
    report = await run_qa(QALevel.VECTOR)

    assert report.status == "failed"
    assert report.checks[-1].status == "failed"
    assert report.checks[-1].detail == "unexpected_failure"
    assert "sensitive" not in report.model_dump_json()
