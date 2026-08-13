from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

from ragplan.core.errors import ErrorCode
from ragplan.core.models import SearchRequest

pytestmark = pytest.mark.unit


def _load_command() -> ModuleType:
    path = Path(__file__).resolve().parents[3] / "scripts" / "search_hybrid.py"
    spec = importlib.util.spec_from_file_location("ragplan_script_search_hybrid", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


command = _load_command()


@pytest.mark.asyncio
async def test_command_builds_shared_runtime_and_executes_selected_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[object] = []
    requests: list[SearchRequest] = []
    response = object()

    class FakeEngine:
        async def search(self, request: SearchRequest, *, request_id: str) -> object:
            events.append(("search", request_id))
            requests.append(request)
            return response

        async def close(self) -> None:
            events.append("close")

    async def fake_build(config: object, **kwargs: object) -> FakeEngine:
        events.append(("build", config))
        return FakeEngine()

    monkeypatch.setenv("RAGPLAN_GRAPH__PASSWORD", "test-only")
    monkeypatch.setattr(command, "build_baseline_search_engine", fake_build)
    args = argparse.Namespace(
        query="How are Apple and Beats related?",
        mode="fixed_hybrid",
        plan_id="P6",
        model_snapshot=tmp_path / "model",
        vector_stage_manifest=tmp_path / "vector.json",
        graph_stage_manifest=tmp_path / "graph.json",
        manifest_root=tmp_path / "manifests",
        extractor_lockfile=Path("uv.lock"),
        plan_catalog=None,
        qdrant_url="http://127.0.0.1:6333",
        collection_prefix="ragplan_chunks",
        neo4j_uri="bolt://127.0.0.1:7687",
        neo4j_user="neo4j",
        neo4j_database="neo4j",
        top_k=10,
        latency_budget_ms=500,
    )

    result = await command._execute(args, request_id="hybrid-request-1")

    assert result is response
    assert events[0][0] == "build"
    assert events[-2:] == [("search", "hybrid-request-1"), "close"]
    assert requests[0].planner.value == "fixed_hybrid"
    assert requests[0].plan_id == "P6"


def test_parser_error_never_echoes_raw_query(capsys: pytest.CaptureFixture[str]) -> None:
    raw_query = "private hybrid query"

    exit_code = command.run(["--query", raw_query, "--unknown", "value"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    assert json.loads(captured.err)["code"] == ErrorCode.INVALID_REQUEST
    assert raw_query not in captured.err
