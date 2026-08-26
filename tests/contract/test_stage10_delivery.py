from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from ragplan.benchmark.config import load_benchmark_protocol
from ragplan.benchmark.profile_records import P0_PROFILE_PLAN_IDS, create_profile_protocol
from ragplan.benchmark.profiler import PlanProfileSearchEngine
from ragplan.core.engine import BaselineSearchEngine
from ragplan.planner.catalog import load_default_plan_catalog

pytestmark = [pytest.mark.contract, pytest.mark.unit]
ROOT = Path(__file__).resolve().parents[2]


def test_stage10_plan_space_and_final_engine_path_are_frozen() -> None:
    assert P0_PROFILE_PLAN_IDS == ("P0", "P1", "P2", "P3", "P4", "P5", "P6", "P8")
    protocol = create_profile_protocol(load_benchmark_protocol(), load_default_plan_catalog())
    assert protocol.plan_ids == P0_PROFILE_PLAN_IDS
    assert protocol.cold_runs == 1
    assert protocol.warmup_runs == 2
    assert protocol.measured_runs == 10
    assert protocol.latency_budgets_ms == (50, 100, 200, 500)
    assert protocol.primary_splits[0].value == "train"
    assert protocol.primary_splits[1].value == "validation"
    assert hasattr(BaselineSearchEngine, "benchmark_plan_search")
    assert isinstance(BaselineSearchEngine, type)
    assert inspect.isclass(PlanProfileSearchEngine)


def test_stage10_expected_modules_script_and_tests_are_delivered() -> None:
    expected = (
        "src/ragplan/benchmark/profile_records.py",
        "src/ragplan/benchmark/profiler.py",
        "src/ragplan/benchmark/oracle.py",
        "src/ragplan/benchmark/profile_command.py",
        "scripts/profile_plans.py",
        "tests/benchmark/test_profiler_matrix.py",
        "tests/benchmark/test_oracle.py",
    )
    assert all((ROOT / path).is_file() for path in expected)


def test_stage10_profiler_has_no_test_split_or_raw_query_output_path() -> None:
    source = (ROOT / "src/ragplan/benchmark/profiler.py").read_text(encoding="utf-8")
    records = (ROOT / "src/ragplan/benchmark/profile_records.py").read_text(encoding="utf-8")
    assert "refuses the held-out test split" in source
    assert "raw_query" not in source
    assert "query_embedding" not in source
    assert "query: " not in records
    assert "scheduler_trace_present" in records
    assert "invalid_exclusion_reason" in records


def test_stage10_oracle_tie_break_is_explicit_and_ordered() -> None:
    source = (ROOT / "src/ragplan/benchmark/oracle.py").read_text(encoding="utf-8")
    recall_position = source.index("-item.recall_at_10")
    latency_position = source.index("item.p95_execution_latency_ms", recall_position)
    depth_position = source.index("item.plan_features.graph_depth", latency_position)
    plan_position = source.index("int(item.plan_id[1:])", depth_position)
    assert recall_position < latency_position < depth_position < plan_position


def test_stage10_commands_are_documented_in_benchmark_operations() -> None:
    operations = (ROOT / "docs/benchmark.md").read_text(encoding="utf-8")
    commands = (
        "benchmark run --rm benchmark profile",
        "benchmark run --rm benchmark profile-aggregate",
    )
    assert all(command in operations for command in commands)
