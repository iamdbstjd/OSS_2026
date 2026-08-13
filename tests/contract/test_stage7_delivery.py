"""Static runtime-freeze gates for Stage 7 scheduler semantics."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from ragplan.api.runtime import BACKEND_CLIENT_TIMEOUT_SECONDS
from ragplan.api.server import create_app
from ragplan.scheduler.executor import (
    DEFAULT_IN_FLIGHT_LIMIT,
    MAX_BACKEND_TASKS_PER_REQUEST,
)
from ragplan.scheduler.states import CIRCUIT_FAILURE_THRESHOLD, CIRCUIT_OPEN_SECONDS

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
pytestmark = [pytest.mark.unit, pytest.mark.contract]


def test_runtime_semantics_v1_is_public_and_contains_scheduler_boundaries() -> None:
    components = create_app().openapi()["components"]["schemas"]
    scheduler = components["SchedulerTrace"]["properties"]
    branch = components["BranchResult"]["properties"]

    assert scheduler["runtime_semantics_version"]["const"] == "v1"
    assert {
        "state_events",
        "backend_task_count",
        "branch_start_skew_ms",
        "deadline_overshoot_ms",
        "kill_switches",
        "fallback_reason",
    } <= set(scheduler)
    assert {
        "started_at_ms",
        "ended_at_ms",
        "remaining_budget_at_start_ms",
        "remaining_budget_at_end_ms",
        "cancellation_reason",
        "failure_origin",
        "timeout_origin",
    } <= set(branch)


def test_operational_limits_match_the_frozen_addendum() -> None:
    assert DEFAULT_IN_FLIGHT_LIMIT == 32
    assert MAX_BACKEND_TASKS_PER_REQUEST == 2
    assert CIRCUIT_FAILURE_THRESHOLD == 5
    assert CIRCUIT_OPEN_SECONDS == 30.0
    assert BACKEND_CLIENT_TIMEOUT_SECONDS == 30


def test_kill_switches_are_exposed_by_compose_and_example_environment() -> None:
    compose = yaml.safe_load((REPOSITORY_ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    environment = compose["services"]["api"]["environment"]
    env_example = (REPOSITORY_ROOT / ".env.example").read_text(encoding="utf-8")

    for key in ("RAGPLAN_FORCE_VECTOR_ONLY", "RAGPLAN_DISABLE_COST_AWARE"):
        assert key in environment
        assert f"{key}=false" in env_example


def test_ci_and_bilingual_docs_cover_the_stage7_freeze_gate() -> None:
    workflow = (REPOSITORY_ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    korean = (REPOSITORY_ROOT / "README.md").read_text(encoding="utf-8")
    english = (REPOSITORY_ROOT / "README_EN.md").read_text(encoding="utf-8")

    assert "tests/integration/test_scheduler_backends.py" in workflow
    assert "runtime_semantics_version=v1" in korean
    assert "runtime_semantics_version=v1" in english
    assert "RAGPLAN_FORCE_VECTOR_ONLY" in korean
    assert "RAGPLAN_FORCE_VECTOR_ONLY" in english
    assert "Until Stage 7" not in english
