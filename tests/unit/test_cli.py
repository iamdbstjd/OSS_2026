import hashlib
import json

import pytest
from typer.testing import CliRunner

from ragplan import __version__
from ragplan.cli import app as cli_module
from ragplan.cli.app import app
from ragplan.core.models import VectorStageManifest
from ragplan.ingestion.service import VectorIngestResult

pytestmark = pytest.mark.unit


def test_version_option_prints_package_version() -> None:
    result = CliRunner().invoke(app, ["--version"])

    assert result.exit_code == 0
    assert result.stdout == f"{__version__}\n"


def test_benchmark_subcommands_are_exposed() -> None:
    result = CliRunner().invoke(app, ["benchmark", "--help"])

    assert result.exit_code == 0
    assert "capture-environment" in result.stdout
    assert "aggregate" in result.stdout
    assert "profile" in result.stdout
    assert "profile-aggregate" in result.stdout
    assert "run" in result.stdout


def test_accessibility_commands_are_exposed() -> None:
    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0
    for command in ("demo-plan", "search", "ingest", "verify", "quickstart-vector"):
        assert command in result.stdout


def test_planner_demo_needs_no_backend_and_redacts_query() -> None:
    raw_query = "Who founded Acme and who acquired it?"
    result = CliRunner().invoke(
        app,
        ["demo-plan", "--query", raw_query, "--budget-ms", "100", "--entity-count", "2"],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["mode"] == "planner_only_no_embedding"
    assert payload["executes_retrieval"] is False
    assert payload["query_hash"] == hashlib.sha256(raw_query.encode()).hexdigest()
    assert raw_query not in result.stdout
    assert payload["decision"]["mode"] == "rule"


def test_code_verify_requires_no_model_or_database() -> None:
    result = CliRunner().invoke(app, ["verify"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["level"] == "code"
    assert payload["cost_aware_status"] == "research_only_disabled"


def test_search_command_uses_configured_runtime_service_and_redacts_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_query = "private CLI question"

    class FakeResponse:
        def model_dump(self, *, mode: str) -> dict[str, object]:
            assert mode == "json"
            return {"request_id": "cli-test", "trace": {"query_hash": "a" * 64}}

    async def fake_search(request: object, *, request_id: str) -> FakeResponse:
        assert getattr(request, "query") == raw_query
        assert request_id.startswith("cli-search-")
        return FakeResponse()

    monkeypatch.setattr(cli_module, "search_configured_runtime", fake_search)
    result = CliRunner().invoke(app, ["search", "--query", raw_query])

    assert result.exit_code == 0
    assert json.loads(result.stdout)["trace"]["query_hash"] == "a" * 64
    assert raw_query not in result.stdout
    assert raw_query not in result.stderr


def test_ingest_command_reuses_packaged_service(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,  # type: ignore[no-untyped-def]
) -> None:
    corpus = tmp_path / "corpus.json"
    corpus.write_text("{}", encoding="utf-8")
    stage_path = tmp_path / "stage.json"
    stage = VectorStageManifest(
        corpus_version="cli-v1",
        collection_name="cli_collection",
        chunk_count=0,
        canonical_id_checksum=hashlib.sha256(b"").hexdigest(),
        embedding_set_checksum=hashlib.sha256(b"").hexdigest(),
        embedding_artifact_manifest_sha256="b" * 64,
    )

    async def fake_ingest(**kwargs: object) -> VectorIngestResult:
        assert kwargs["input_path"] == corpus
        return VectorIngestResult(
            stage=stage,
            model_snapshot=tmp_path / "snapshot",
            stage_manifest_path=stage_path,
        )

    monkeypatch.setattr(cli_module, "ingest_vector_corpus", fake_ingest)
    result = CliRunner().invoke(
        app,
        [
            "ingest",
            "--input",
            str(corpus),
            "--corpus-version",
            "cli-v1",
            "--stage-manifest",
            str(stage_path),
        ],
    )

    assert result.exit_code == 0
    assert json.loads(result.stdout)["vector_stage"]["corpus_version"] == "cli-v1"


def test_quickstart_command_runs_one_packaged_orchestration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeResult:
        def model_dump(self, *, mode: str) -> dict[str, object]:
            assert mode == "json"
            return {
                "schema_version": "quickstart_vector_v1",
                "infrastructure": "qdrant_plus_minilm",
                "search": {"request_id": "quickstart-vector-v1"},
            }

    async def fake_quickstart(**kwargs: object) -> FakeResult:
        assert kwargs["input_path"].name == "sample_corpus.json"  # type: ignore[union-attr]
        return FakeResult()

    monkeypatch.setattr(cli_module, "quickstart_vector", fake_quickstart)
    result = CliRunner().invoke(app, ["quickstart-vector"])

    assert result.exit_code == 0
    assert json.loads(result.stdout)["infrastructure"] == "qdrant_plus_minilm"


def test_primary_benchmark_requires_dedicated_environment_confirmation(tmp_path) -> None:  # type: ignore[no-untyped-def]
    result = CliRunner().invoke(
        app,
        [
            "benchmark",
            "run",
            "--run-id",
            "stage9-test",
            "--environment-manifest",
            str(tmp_path / "missing.json"),
        ],
    )

    assert result.exit_code == 1
    assert "--confirm-dedicated" in result.stderr


def test_primary_profiler_requires_dedicated_environment_confirmation(tmp_path) -> None:  # type: ignore[no-untyped-def]
    result = CliRunner().invoke(
        app,
        [
            "benchmark",
            "profile",
            "--run-id",
            "stage10-test",
            "--environment-manifest",
            str(tmp_path / "missing.json"),
        ],
    )

    assert result.exit_code == 1
    assert "--confirm-dedicated" in result.stderr
