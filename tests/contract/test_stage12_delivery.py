from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from ragplan.benchmark.policy_evaluation import (
    Stage12EvidenceManifest,
    Stage12PolicyReport,
)

ROOT = Path(__file__).resolve().parents[2]
pytestmark = [pytest.mark.contract, pytest.mark.unit]


def test_stage12_research_only_delivery_files_are_present() -> None:
    required = (
        "src/ragplan/planner/optimizer.py",
        "src/ragplan/planner/runtime_guard.py",
        "src/ragplan/benchmark/policy_evaluation.py",
        "scripts/evaluate_cost_policy.py",
        "docs/offline_cost_policy.md",
        "tests/unit/planner/test_optimizer.py",
        "tests/unit/planner/test_runtime_guard.py",
        "tests/benchmark/test_policy_regret.py",
        "benchmark/manifests/stage12_policy_evidence_r2.json",
    )
    assert all((ROOT / path).is_file() for path in required)


def test_stage12_evidence_is_validation_only_and_never_claims_serving() -> None:
    evidence = Stage12EvidenceManifest.model_validate_json(
        (ROOT / "benchmark/manifests/stage12_policy_evidence_r2.json").read_text(encoding="utf-8")
    )

    assert evidence.status == "research_only"
    assert evidence.execution_mode == "offline_shadow"
    assert evidence.public_api_cost_aware_enabled is False
    assert evidence.test_split_used is False
    assert evidence.decision_record_count == 480
    assert evidence.candidate_estimate_count == 3_840
    assert evidence.invalid_prediction_count == 1_856
    assert evidence.no_feasible_candidate_count == 120
    assert evidence.runtime_guard_disabled is True
    assert evidence.runtime_guard_disable_reason == "p95_underprediction_rate_gt_0.20"
    assert evidence.runtime_guard_first_disabled_after_observation == 319
    assert evidence.runtime_guard_routed_to_rule_group_count == 422
    assert set(evidence.inherited_model_gate_failures) == {
        "quality_plan_pair_ranking_lt_0.70",
        "latency_plan_coverage_lt_0.85",
        "latency_pinball_improvement_lt_0.10",
    }


def test_local_stage12_outputs_match_their_evidence_when_available() -> None:
    output = ROOT / "artifacts/cost_models/stage12_r2"
    if not output.is_dir():
        pytest.skip("large generated Stage 12 outputs are intentionally not committed")
    evidence = Stage12EvidenceManifest.model_validate_json(
        (ROOT / "benchmark/manifests/stage12_policy_evidence_r2.json").read_text(encoding="utf-8")
    )
    report_path = output / "policy_report.json"
    decisions_path = output / "decisions.jsonl"
    report = Stage12PolicyReport.model_validate_json(report_path.read_text(encoding="utf-8"))

    assert _sha256(report_path) == evidence.report_file_sha256
    assert _sha256(decisions_path) == evidence.decision_records_file_sha256
    assert report.sha256 == evidence.report_sha256
    assert report.decision_records_sha256 == evidence.decision_records_file_sha256
    assert sum(1 for _ in decisions_path.open(encoding="utf-8")) == 480
    first = json.loads(decisions_path.read_text(encoding="utf-8").splitlines()[0])
    assert len(first["cost_aware_decision"]["candidate_estimates"]) == 8
    serialized = json.dumps(first, ensure_ascii=False)
    assert "query_embedding" not in serialized
    assert "normalized_query" not in serialized


def test_offline_policy_documentation_contains_command_and_public_boundary() -> None:
    documentation = (ROOT / "docs/offline_cost_policy.md").read_text(encoding="utf-8")
    command = "uv run python scripts/evaluate_cost_policy.py"
    assert command in documentation
    assert "MODE_UNAVAILABLE" in documentation


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
