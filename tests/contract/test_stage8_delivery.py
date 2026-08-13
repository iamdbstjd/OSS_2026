"""Static Stage 8 feature/planner freeze gates."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from ragplan.api.server import create_app
from ragplan.planner.features import FEATURE_SCHEMA_VERSION, load_default_query_feature_config
from ragplan.planner.rule import RULE_CONFIG_VERSION, load_default_rule_planner_config

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
pytestmark = [pytest.mark.unit, pytest.mark.contract]


def test_stage8_configs_and_schema_identities_are_frozen() -> None:
    features = load_default_query_feature_config()
    rules = load_default_rule_planner_config()

    assert FEATURE_SCHEMA_VERSION == "qf_v1"
    assert RULE_CONFIG_VERSION == "rule_v1"
    assert features.sha256 == "8432e9c5fc80da61919cd1d8a3f5fc8020903496b9e57b9fe9d8f7b85e04910d"
    assert rules.sha256 == "ed5ab309f77b3f43f8550e06e1371d27d0e82f9e4e203a104bdd10346061a6f1"
    assert features.embedding_feature_enabled is False
    assert features.supported_language == "en"
    assert rules.threshold_tuning_split == "validation"


def test_default_config_points_to_versioned_stage8_inputs() -> None:
    config = yaml.safe_load((REPOSITORY_ROOT / "configs" / "default.yaml").read_text())

    assert config["planner"] == {
        "mode": "rule",
        "feature_config": "configs/query_features_v1.json",
        "rule_config": "configs/rule_planner_v1.json",
    }
    pyproject = (REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "ragplan/resources/query_features_v1.json" in pyproject
    assert "ragplan/resources/rule_planner_v1.json" in pyproject


def test_public_trace_exposes_features_and_deterministic_rule_explanation() -> None:
    schemas = create_app().openapi()["components"]["schemas"]

    assert set(schemas["QueryFeatures"]["properties"]) == {
        "token_count",
        "entity_count",
        "entity_density",
        "relation_signal",
        "multi_hop_signal",
        "comparison_signal",
        "aggregation_signal",
        "global_signal",
        "final_top_k",
    }
    decision = schemas["PlannerDecision"]["properties"]
    assert {
        "selected_plan",
        "matched_rules",
        "remaining_budget_ms",
        "candidate_estimates",
        "fallback_reason",
        "feature_version",
        "config_version",
        "selection_reason",
    } <= set(decision)


def test_bilingual_docs_describe_stage8_safety_boundary() -> None:
    korean = (REPOSITORY_ROOT / "README.md").read_text(encoding="utf-8")
    english = (REPOSITORY_ROOT / "README_EN.md").read_text(encoding="utf-8")

    for document in (korean, english):
        assert "Stage 8" in document
        assert "qf_v1" in document
        assert "rule_planner_v1.json" in document
        assert "vector-only" in document
