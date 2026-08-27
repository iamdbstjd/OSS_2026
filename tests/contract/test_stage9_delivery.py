from __future__ import annotations

import tomllib
from pathlib import Path

import pytest
import yaml

from ragplan.benchmark.config import load_benchmark_protocol
from ragplan.benchmark.records import EXECUTED_METHODS, RawTrialRecord

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
pytestmark = [pytest.mark.unit, pytest.mark.contract]


def test_stage9_protocol_and_delivery_files_are_present() -> None:
    protocol = load_benchmark_protocol()

    assert protocol.latency_budgets_ms == (50, 100, 200, 500)
    assert protocol.cold_runs == 1
    assert protocol.warmup_runs == 2
    assert protocol.measured_runs == 10
    assert protocol.concurrency == 1
    assert tuple(item.method for item in protocol.methods) == EXECUTED_METHODS
    assert protocol.primary_splits[0].value == "train"
    assert protocol.primary_splits[1].value == "validation"
    assert protocol.corpus_chunk_count == 8604

    required = (
        "src/ragplan/benchmark/records.py",
        "src/ragplan/benchmark/runner.py",
        "src/ragplan/benchmark/aggregate.py",
        "src/ragplan/benchmark/config.py",
        "benchmark/configs/baseline_v1.yaml",
        "scripts/benchmark.py",
        "docs/benchmark.md",
        "tests/benchmark/test_stage9_harness.py",
    )
    assert all((REPOSITORY_ROOT / path).is_file() for path in required)


def test_raw_schema_has_identity_bundle_and_no_query_text() -> None:
    fields = set(RawTrialRecord.model_fields)

    assert "query" not in fields
    assert "question" not in fields
    assert {
        "query_id",
        "method",
        "planner",
        "configured_plan_id",
        "selected_plan_id",
        "recall_at_5",
        "recall_at_10",
        "mrr_at_10",
        "ndcg_at_10",
        "total_latency_ms",
        "branch_results",
        "timeout",
        "fallback",
        "budget_violated",
        "protocol_config_sha256",
        "environment_manifest_sha256",
        "benchmark_manifest_sha256",
        "split_hash",
        "qrels_sha256",
        "corpus_chunk_ids_sha256",
        "plan_catalog_sha256",
        "planner_config_sha256",
        "query_feature_config_sha256",
        "graph_tier_policy_sha256",
        "rule_runtime_config_version",
        "stage2_artifact_set_sha256",
    } <= fields


def test_benchmark_protocol_is_in_wheel_and_results_are_ignored() -> None:
    pyproject = tomllib.loads((REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    force_include = pyproject["tool"]["hatch"]["build"]["targets"]["wheel"]["force-include"]
    assert (
        force_include["benchmark/configs/baseline_v1.yaml"]
        == "ragplan/resources/benchmark/baseline_v1.yaml"
    )
    assert (
        force_include["benchmark/configs/db_tuning_default_v1.json"]
        == "ragplan/resources/benchmark/db_tuning_default_v1.json"
    )
    ignored = (REPOSITORY_ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "benchmark/results/*" in ignored
    assert "!benchmark/results/.gitkeep" in ignored


def test_stage9_commands_are_documented_in_benchmark_operations() -> None:
    operations = (REPOSITORY_ROOT / "docs/benchmark.md").read_text(encoding="utf-8")
    commands = (
        "ragplan benchmark capture-environment",
        "benchmark run --rm benchmark run",
        "benchmark run --rm benchmark aggregate",
    )

    assert all(command in operations for command in commands)


def test_primary_runner_uses_the_local_compose_network() -> None:
    compose = yaml.safe_load((REPOSITORY_ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    benchmark = compose["services"]["benchmark"]

    assert benchmark["profiles"] == ["benchmark"]
    assert benchmark["entrypoint"] == [".venv/bin/ragplan", "benchmark"]
    assert set(benchmark["depends_on"]) == {"qdrant", "neo4j"}
    assert benchmark["environment"]["RAGPLAN_STAGE6_QDRANT_URL"] == "http://qdrant:6333"
    assert benchmark["environment"]["RAGPLAN_STAGE6_NEO4J_URI"] == "bolt://neo4j:7687"
    assert "ports" not in benchmark
    assert benchmark["read_only"] is True

    dockerfile = (REPOSITORY_ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "README.md LICENSE docker-compose.yml ./" in dockerfile
    assert "COPY benchmark/manifests ./benchmark/manifests" in dockerfile
    assert "COPY benchmark/qrels ./benchmark/qrels" in dockerfile
