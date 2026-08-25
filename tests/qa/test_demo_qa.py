from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from typer.testing import CliRunner

from ragplan.cli.app import app

ROOT = Path(__file__).resolve().parents[2]
QUERY = "Who founded Acme and who acquired it?"
pytestmark = [pytest.mark.qa, pytest.mark.unit]


@pytest.mark.parametrize(("budget_ms", "expected_plan_id"), [(50, "P0"), (500, "P1")])
def test_planner_demo_is_offline_redacted_and_fail_closed(
    budget_ms: int,
    expected_plan_id: str,
) -> None:
    result = CliRunner().invoke(
        app,
        [
            "demo-plan",
            "--query",
            QUERY,
            "--budget-ms",
            str(budget_ms),
            "--entity-count",
            "2",
        ],
    )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    decision = payload["decision"]
    assert payload["mode"] == "planner_only_no_embedding"
    assert payload["executes_retrieval"] is False
    assert payload["query_hash"] == hashlib.sha256(QUERY.encode()).hexdigest()
    assert QUERY not in result.stdout
    assert decision["mode"] == "rule"
    assert decision["effective_mode"] == "vector"
    assert decision["selected_plan_id"] == expected_plan_id
    assert decision["selected_plan"]["graph_enabled"] is False
    assert 0.0 <= decision["remaining_budget_ms"] <= budget_ms

    graph_candidates = {
        item["plan_id"]: item
        for item in decision["candidate_estimates"]
        if item["plan_id"] in {"P4", "P5", "P6", "P8"}
    }
    assert set(graph_candidates) == {"P4", "P5", "P6", "P8"}
    assert all(item["feasible"] is False for item in graph_candidates.values())
    assert all(
        item["infeasible_reason"] == "graph_tier_unavailable" for item in graph_candidates.values()
    )


def test_stage9_and_stage10_evidence_supports_the_trial_row_claim() -> None:
    evidence = json.loads(
        (ROOT / "benchmark/manifests/stage9_stage10_evidence_r2.json").read_text(encoding="utf-8")
    )

    assert evidence["stage9"]["query_count"] == 480
    assert evidence["stage10"]["query_count"] == 480
    assert evidence["stage9"]["raw_rows"] == 224_640
    assert evidence["stage10"]["raw_rows"] == 199_680
    assert evidence["stage9"]["raw_rows"] + evidence["stage10"]["raw_rows"] == 424_320
    assert evidence["stage9"]["raw_file_committed"] is False
    assert evidence["stage10"]["raw_file_committed"] is False


def test_stage12_evidence_is_research_only_and_fail_closed() -> None:
    evidence = json.loads(
        (ROOT / "benchmark/manifests/stage12_policy_evidence_r2.json").read_text(encoding="utf-8")
    )

    assert evidence["status"] == "research_only"
    assert evidence["execution_mode"] == "offline_shadow"
    assert evidence["public_api_cost_aware_enabled"] is False
    assert evidence["runtime_guard_disabled"] is True
    assert evidence["runtime_guard_disable_reason"] == "p95_underprediction_rate_gt_0.20"
    assert evidence["runtime_guard_first_disabled_after_observation"] == 319
    assert evidence["runtime_guard_routed_to_rule_group_count"] == 422
    assert evidence["test_split_used"] is False


def _load_llm_handoff() -> ModuleType:
    path = ROOT / "examples/llm_handoff.py"
    spec = importlib.util.spec_from_file_location("ragplan_qa_llm_handoff", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_llm_handoff_preserves_ranked_untrusted_evidence_without_provider_sdk() -> None:
    module = _load_llm_handoff()
    response = SimpleNamespace(
        results=(
            SimpleNamespace(
                rank=1,
                canonical_chunk_id="v1:chunk:qa:0:abcdef",
                source="vector",
                text="Ada Lovelace wrote notes about the Analytical Engine.",
            ),
        )
    )

    messages = module.build_llm_messages("What did Ada Lovelace write about?", response)

    assert tuple(item["role"] for item in messages) == ("system", "user")
    assert "untrusted data" in messages[0]["content"]
    assert "chunk=v1:chunk:qa:0:abcdef" in messages[1]["content"]
    assert "Analytical Engine" in messages[1]["content"]
    source = Path(module.__file__).read_text(encoding="utf-8").casefold()
    assert "import openai" not in source
    assert "import anthropic" not in source
