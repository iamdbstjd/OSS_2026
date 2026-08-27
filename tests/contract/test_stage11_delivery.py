from __future__ import annotations

import json
import tomllib
from pathlib import Path

import pytest

from ragplan.planner.artifacts import ArtifactStatus, load_trusted_types
from ragplan.planner.training import COST_FEATURE_SCHEMA_VERSION, FEATURE_NAMES

ROOT = Path(__file__).resolve().parents[2]
pytestmark = [pytest.mark.contract, pytest.mark.unit]


def test_stage11_delivery_files_and_feature_contract_are_present() -> None:
    required = (
        "src/ragplan/planner/training.py",
        "src/ragplan/planner/quality_model.py",
        "src/ragplan/planner/latency_model.py",
        "src/ragplan/planner/artifacts.py",
        "src/ragplan/benchmark/model_report.py",
        "scripts/train_cost_models.py",
        "configs/skops_trusted_types_v1.json",
        "docs/model_training.md",
        "tests/unit/planner/test_artifacts.py",
        "tests/benchmark/test_model_training.py",
    )
    assert all((ROOT / path).is_file() for path in required)
    assert COST_FEATURE_SCHEMA_VERSION == "cost_model_features_v1:qf_v1"
    assert FEATURE_NAMES
    assert all("embedding" not in name for name in FEATURE_NAMES)


def test_skops_allowlist_is_packaged_and_no_unsafe_model_suffix_is_supported() -> None:
    assert load_trusted_types() == ()
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    included = pyproject["tool"]["hatch"]["build"]["targets"]["wheel"]["force-include"]
    assert (
        included["configs/skops_trusted_types_v1.json"]
        == "ragplan/resources/skops_trusted_types_v1.json"
    )
    source = (ROOT / "src/ragplan/planner/artifacts.py").read_text(encoding="utf-8")
    assert 'artifact_path.suffix != ".skops"' in source
    assert "sio.get_untrusted_types" in source


def test_r2_model_evidence_is_honestly_research_only() -> None:
    evidence = json.loads(
        (ROOT / "benchmark/manifests/stage11_model_evidence_r2.json").read_text(encoding="utf-8")
    )
    assert evidence["status"] == ArtifactStatus.RESEARCH_ONLY.value
    assert evidence["test_split_used"] is False
    assert evidence["raw_embeddings_used"] is False
    assert evidence["latency"]["missing_validation_plans"] == ["P0", "P1"]
    assert set(evidence["gate_failures"]) == {
        "quality_plan_pair_ranking_lt_0.70",
        "latency_plan_coverage_lt_0.85",
        "latency_pinball_improvement_lt_0.10",
    }


def test_stage11_training_command_is_documented() -> None:
    documentation = (ROOT / "docs/model_training.md").read_text(encoding="utf-8")
    marker = "uv run python scripts/train_cost_models.py"
    assert marker in documentation
