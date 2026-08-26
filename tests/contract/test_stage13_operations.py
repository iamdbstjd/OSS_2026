from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from ragplan.cli.app import app
from ragplan.observability.tracing import (
    TRACE_BACKUP_COUNT,
    TRACE_FILE_COUNT,
    TRACE_MAX_BYTES,
    TRACE_QUEUE_CAPACITY,
)

ROOT = Path(__file__).resolve().parents[2]
pytestmark = [pytest.mark.contract, pytest.mark.unit]


def test_operational_cli_and_clean_wheel_assets_are_present() -> None:
    help_result = CliRunner().invoke(app, ["--help"])

    assert help_result.exit_code == 0
    assert "download-model" in help_result.stdout
    assert "qa" in help_result.stdout
    assert (ROOT / "src/ragplan/resources/sample_corpus.json").is_file()
    assert (ROOT / "scripts/verify_clean_wheel.py").is_file()
    assert (ROOT / "tests/e2e/test_clean_wheel.py").is_file()


def test_trace_rotation_and_queue_contract_is_frozen() -> None:
    assert TRACE_MAX_BYTES == 10 * 1024 * 1024
    assert TRACE_FILE_COUNT == 5
    assert TRACE_BACKUP_COUNT == 4
    assert TRACE_QUEUE_CAPACITY == 1024
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    environment = (ROOT / ".env.example").read_text(encoding="utf-8")
    defaults = (ROOT / "configs/default.yaml").read_text(encoding="utf-8")
    assert "RAGPLAN_LOGGING__PATH: /tmp/ragplan/ragplan-trace.jsonl" in compose
    assert "RAGPLAN_LOGGING__PATH=/tmp/ragplan/ragplan-trace.jsonl" in environment
    assert "max_bytes: 10485760" in defaults


def test_qa_source_has_no_benchmark_split_or_query_loader_dependency() -> None:
    source = (ROOT / "src/ragplan/cli/services.py").read_text(encoding="utf-8")

    assert "held_out_test_accessed: Literal[False]" in source
    assert "load_training_matrix" not in source
    assert "SplitName.TEST" not in source
    assert "test_ids_v1" not in source


def test_submission_and_internal_planning_documents_are_not_public_artifacts() -> None:
    private_paths = (
        "QA.md",
        "adaptive_rag_query_optimizer_PRD.md",
        "adaptive_rag_query_optimizer_PRD_addendum.md",
        "stage.md",
    )

    assert all(not (ROOT / path).exists() for path in private_paths)
    assert "/submission/" in (ROOT / ".gitignore").read_text(encoding="utf-8")
