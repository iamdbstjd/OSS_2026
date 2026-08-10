from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

from ragplan.core.errors import ErrorCode, RAGPlanError
from ragplan.core.models import SearchRequest, VectorStageManifest
from ragplan.ingestion.model_manifest import load_default_model_artifact_manifest

pytestmark = pytest.mark.unit


def _load_command() -> ModuleType:
    path = Path(__file__).resolve().parents[3] / "scripts" / "search.py"
    spec = importlib.util.spec_from_file_location("ragplan_script_search", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


command = _load_command()


def _stage() -> VectorStageManifest:
    return VectorStageManifest(
        corpus_version="sample-v1",
        collection_name="ragplan_chunks_" + hashlib.sha256(b"sample-v1").hexdigest(),
        chunk_count=0,
        canonical_id_checksum=hashlib.sha256(b"").hexdigest(),
        embedding_set_checksum=hashlib.sha256(b"").hexdigest(),
        embedding_artifact_manifest_sha256=load_default_model_artifact_manifest().sha256,
    )


def test_stage_loader_accepts_only_vector_staged_schema(tmp_path: Path) -> None:
    path = tmp_path / "stage.json"
    path.write_text(_stage().model_dump_json(), encoding="utf-8")
    assert command.load_stage_manifest(path) == _stage()

    decoded = _stage().model_dump(mode="json")
    decoded["status"] = "active"
    path.write_text(json.dumps(decoded), encoding="utf-8")
    with pytest.raises(RAGPlanError) as captured:
        command.load_stage_manifest(path)
    assert captured.value.code is ErrorCode.CORPUS_INCONSISTENT


@pytest.mark.asyncio
async def test_execute_verifies_stage_then_calls_vector_search_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stage_path = tmp_path / "stage.json"
    stage_path.write_text(_stage().model_dump_json(), encoding="utf-8")
    events: list[str] = []
    observed_requests: list[SearchRequest] = []
    response = object()

    class FakeEmbedderFactory:
        @classmethod
        def from_local_snapshot(cls, **kwargs: object) -> object:
            events.append("load_model")
            return object()

    class FakeClient:
        def __init__(self, **kwargs: object) -> None:
            events.append("client")

    class FakeManager:
        def __init__(self, client: object, config: object) -> None:
            events.append("manager")

        async def verify_stage(self, stage: VectorStageManifest) -> VectorStageManifest:
            events.append("verify_stage")
            return stage

    class FakeBackend:
        def __init__(self, manager: object) -> None:
            events.append("backend")

        async def close(self) -> None:
            events.append("close")

    class FakeEngine:
        def __init__(self, **kwargs: object) -> None:
            events.append("engine")

        async def search(self, request: SearchRequest, *, request_id: str) -> object:
            events.append(f"search:{request_id}")
            observed_requests.append(request)
            return response

    monkeypatch.setattr(command, "SentenceTransformerEmbedder", FakeEmbedderFactory)
    monkeypatch.setattr(command, "AsyncQdrantClient", FakeClient)
    monkeypatch.setattr(command, "QdrantCollectionManager", FakeManager)
    monkeypatch.setattr(command, "QdrantVectorBackend", FakeBackend)
    monkeypatch.setattr(command, "VectorSearchEngine", FakeEngine)
    args = argparse.Namespace(
        stage_manifest=stage_path,
        model_manifest=None,
        model_snapshot=tmp_path / "snapshot",
        embedding_batch_size=32,
        query="Which engine did Ada Lovelace describe?",
        top_k=3,
        latency_budget_ms=200,
        plan_catalog=None,
        qdrant_url="http://127.0.0.1:6333",
        collection_prefix="ragplan_chunks",
    )

    result = await command._execute(args, request_id="request-1")

    assert result is response
    assert events[-4:] == ["verify_stage", "engine", "search:request-1", "close"]
    assert observed_requests[0].planner.value == "vector"
    assert observed_requests[0].top_k == 3


def test_parser_error_never_echoes_raw_query(capsys: pytest.CaptureFixture[str]) -> None:
    raw_query = "sensitive raw query that must not be logged"

    exit_code = command.run(["--query", raw_query, "--unknown", "value"])

    captured = capsys.readouterr()
    error = json.loads(captured.err)
    assert exit_code == 1
    assert captured.out == ""
    assert error["code"] == ErrorCode.INVALID_REQUEST
    assert raw_query not in captured.err


def test_success_output_uses_only_response_serializer(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    raw_query = "private semantic question"

    class FakeResponse:
        def model_dump(self, *, mode: str) -> dict[str, object]:
            assert mode == "json"
            return {"request_id": "request-1", "trace": {"query_hash": "a" * 64}}

    async def fake_execute(args: argparse.Namespace, *, request_id: str) -> FakeResponse:
        assert args.query == raw_query
        return FakeResponse()

    monkeypatch.setattr(command, "_execute", fake_execute)
    exit_code = command.run(
        [
            "--query",
            raw_query,
            "--stage-manifest",
            "stage.json",
            "--model-snapshot",
            "snapshot",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert raw_query not in captured.out
    assert raw_query not in captured.err
    assert json.loads(captured.out)["trace"] == {"query_hash": "a" * 64}
