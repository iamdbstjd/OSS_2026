from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from ragplan.api.server import create_app
from ragplan.cli.app import app

ROOT = Path(__file__).resolve().parents[2]
pytestmark = [pytest.mark.contract, pytest.mark.unit]


def test_stage13_accessibility_files_are_present() -> None:
    required = (
        "src/ragplan/api/readiness.py",
        "src/ragplan/observability/metrics.py",
        "src/ragplan/cli/services.py",
        "src/ragplan/ingestion/service.py",
        "examples/llm_handoff.py",
        "examples/README.md",
        "tests/unit/test_readiness.py",
        "tests/unit/test_llm_handoff.py",
    )
    assert all((ROOT / path).is_file() for path in required)


def test_public_openapi_contains_search_liveness_readiness_and_metrics() -> None:
    schema = create_app().openapi()

    assert {"/v1/search", "/health", "/ready", "/metrics"} == set(schema["paths"])
    assert "ReadinessResponse" in schema["components"]["schemas"]
    assert "MetricsSnapshot" in schema["components"]["schemas"]


def test_openapi_v1_canonical_snapshot_is_approved() -> None:
    payload = json.dumps(
        create_app().openapi(),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    observed = hashlib.sha256(payload).hexdigest()
    expected = (ROOT / "tests/fixtures/openapi_v1.sha256").read_text(encoding="utf-8").strip()

    assert observed == expected


def test_cli_exposes_demo_search_ingest_verify_and_vector_quickstart() -> None:
    help_result = CliRunner().invoke(app, ["--help"])

    assert help_result.exit_code == 0
    for command in ("demo-plan", "search", "ingest", "verify", "quickstart-vector"):
        assert command in help_result.stdout


def test_readme_first_screen_matches_current_scope_and_infrastructure_levels() -> None:
    korean = (ROOT / "README.md").read_text(encoding="utf-8")
    english = (ROOT / "README_EN.md").read_text(encoding="utf-8")

    assert "Stage 0~12" in korean
    assert "Stages 0–12" in english
    assert "학습 기반 cost-aware planner는 이후 Stage" not in korean
    assert "learned cost-aware planner remains a later stage" not in english.casefold()
    for token in (
        "demo-plan",
        "quickstart-vector",
        "planner_only_no_embedding",
        "research_only",
        "/ready",
        "/metrics",
        "examples/llm_handoff.py",
    ):
        assert token in korean and token in english
    assert "Qdrant + MiniLM" in korean and "Qdrant + MiniLM" in english
    assert "Qdrant + Neo4j" in korean and "Qdrant + Neo4j" in english
    assert "spaCy" in korean and "spaCy" in english


def test_metrics_and_llm_example_do_not_add_query_or_entity_labels_or_sdk_dependencies() -> None:
    metrics = (ROOT / "src/ragplan/observability/metrics.py").read_text(encoding="utf-8")
    example = (ROOT / "examples/llm_handoff.py").read_text(encoding="utf-8")

    assert "raw_query" not in metrics
    assert "entity_id" not in metrics
    assert "openai" not in example.casefold()
    assert "anthropic" not in example.casefold()
    assert "SearchResponse" in example
