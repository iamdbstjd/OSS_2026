from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

from ragplan.core.errors import ErrorCode
from ragplan.core.models import (
    ActivationStatus,
    GraphStageManifest,
    IngestionManifest,
    IngestionStoreStatus,
    SearchRequest,
)
from ragplan.ingestion.manifest import ManifestRepository

pytestmark = pytest.mark.unit


def _load_command() -> ModuleType:
    path = Path(__file__).resolve().parents[3] / "scripts" / "search_graph.py"
    spec = importlib.util.spec_from_file_location("ragplan_script_search_graph", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


command = _load_command()


def _artifacts(tmp_path: Path) -> tuple[Path, Path]:
    checksum = "a" * 64
    repository = ManifestRepository(tmp_path / "ingestion")
    repository.activate(
        IngestionManifest(
            ingestion_run_id="run-v1",
            corpus_version="active-v1",
            source_dataset="fixture",
            source_version="v1",
            source_sha256="b" * 64,
            chunker_version="v1",
            embedding_model_revision="embedding-v1",
            extractor_version="extractor-v1",
            document_count=1,
            chunk_count=1,
            qdrant_count=1,
            qdrant_id_checksum=checksum,
            qdrant_status=IngestionStoreStatus.SUCCEEDED,
            neo4j_count=1,
            neo4j_id_checksum=checksum,
            neo4j_status=IngestionStoreStatus.SUCCEEDED,
            activation_status=ActivationStatus.ACTIVE,
        )
    )
    stage = GraphStageManifest(
        corpus_version="active-v1",
        database="neo4j",
        document_count=1,
        chunk_count=1,
        entity_count=2,
        mention_count=2,
        relation_count=1,
        canonical_id_checksum=checksum,
        graph_content_checksum="c" * 64,
        extractor_version="extractor-v1",
    )
    stage_path = tmp_path / "graph-stage.json"
    stage_path.write_text(stage.model_dump_json(), encoding="utf-8")
    return tmp_path / "ingestion", stage_path


@pytest.mark.asyncio
async def test_graph_command_verifies_active_corpus_and_executes_explicit_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_root, stage_path = _artifacts(tmp_path)
    events: list[object] = []
    requests: list[SearchRequest] = []
    response = object()

    class FakeExtractor:
        extractor_version = "extractor-v1"

        @classmethod
        def load_pinned(cls, **kwargs: object) -> FakeExtractor:
            events.append("extractor")
            return cls()

    class FakeBackend:
        @classmethod
        def connect(cls, config: object) -> FakeBackend:
            events.append("connect")
            return cls()

        async def require_active_corpus(self, **kwargs: object) -> None:
            events.append(("verify", kwargs["corpus_version"]))

        async def close(self) -> None:
            events.append("close")

    class FakeEngine:
        def __init__(self, **kwargs: object) -> None:
            events.append("engine")

        async def search(self, request: SearchRequest, *, request_id: str) -> object:
            events.append(("search", request_id))
            requests.append(request)
            return response

    monkeypatch.setenv("RAGPLAN_GRAPH__PASSWORD", "test-only")
    monkeypatch.setattr(command, "EntityExtractor", FakeExtractor)
    monkeypatch.setattr(command, "Neo4jGraphBackend", FakeBackend)
    monkeypatch.setattr(command, "GraphSearchEngine", FakeEngine)
    args = argparse.Namespace(
        query="How is Apple related to Beats?",
        manifest_root=manifest_root,
        graph_stage_manifest=stage_path,
        extractor_lockfile=Path("uv.lock"),
        plan_catalog=None,
        neo4j_uri="bolt://127.0.0.1:7687",
        neo4j_user="neo4j",
        neo4j_database="neo4j",
        top_k=3,
        latency_budget_ms=200,
    )

    result = await command._execute(args, request_id="graph-request-1")

    assert result is response
    assert events == [
        "extractor",
        "connect",
        ("verify", "active-v1"),
        "engine",
        ("search", "graph-request-1"),
        "close",
    ]
    assert requests[0].planner.value == "graph"
    assert requests[0].top_k == 3


def test_graph_parser_error_does_not_echo_raw_query(
    capsys: pytest.CaptureFixture[str],
) -> None:
    raw_query = "private graph question"

    exit_code = command.run(["--query", raw_query, "--unknown", "value"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    assert json.loads(captured.err)["code"] == ErrorCode.INVALID_REQUEST
    assert raw_query not in captured.err
