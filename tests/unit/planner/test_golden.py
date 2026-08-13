from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest

from ragplan.core.deadline import Deadline, ManualClock
from ragplan.core.models import QueryAnalysis, QueryFeatures
from ragplan.ingestion.audit import AuditStatus, RuleGraphTierPolicy
from ragplan.planner.catalog import load_default_plan_catalog
from ragplan.planner.features import extract_query_features, load_default_query_feature_config
from ragplan.planner.rule import RulePlanner, load_default_rule_planner_config

pytestmark = pytest.mark.unit
FIXTURE = Path(__file__).resolve().parents[2] / "fixtures" / "planner" / "qf_v1_golden.json"


def _load_fixture() -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(FIXTURE.read_text(encoding="utf-8")))


def test_representative_qf_v1_golden_features_and_plans_are_exact() -> None:
    fixture = _load_fixture()
    feature_config = load_default_query_feature_config()
    planner = RulePlanner(
        catalog=load_default_plan_catalog(),
        graph_policy=RuleGraphTierPolicy(
            audit_sample_checksum="a" * 64,
            audit_status=AuditStatus.COMPLETE,
            observed_entity_f1=0.80,
            observed_relation_precision=0.75,
            graph_tier_enabled=True,
            reason="gates passed",
        ),
        config=load_default_rule_planner_config(),
    )

    assert fixture["feature_schema_version"] == "qf_v1"
    assert fixture["feature_config_sha256"] == feature_config.sha256
    for raw_case in fixture["cases"]:
        case = cast(dict[str, Any], raw_case)
        features = extract_query_features(
            cast(str, case["query"]),
            token_count=cast(int, case["token_count"]),
            entity_count=cast(int, case["entity_count"]),
            final_top_k=10,
            config=feature_config,
        )
        assert features.model_dump(mode="json") == case["features"], case["name"]
        analysis = QueryAnalysis(
            normalized_query=cast(str, case["query"]),
            language_supported=True,
            token_count=features.token_count,
            query_embedding=(1.0,),
            features=QueryFeatures.model_validate(features),
            analyzer_version="golden",
            analysis_latency_ms=0.0,
        )
        first = planner.select(
            analysis,
            deadline=Deadline.start(cast(int, case["budget_ms"]), clock=ManualClock()),
        )
        second = planner.select(
            analysis,
            deadline=Deadline.start(cast(int, case["budget_ms"]), clock=ManualClock()),
        )
        assert first == second
        assert first.selected_plan_id == case["selected_plan"], case["name"]
        assert first.selection_reason


def test_max_length_feature_input_remains_bounded() -> None:
    features = extract_query_features(
        "a" * 4096,
        token_count=2048,
        entity_count=0,
        final_top_k=50,
        config=load_default_query_feature_config(),
    )

    assert features.token_count == 2048
    assert features.final_top_k == 50
    for value in (
        features.entity_density,
        features.relation_signal,
        features.multi_hop_signal,
        features.comparison_signal,
        features.aggregation_signal,
        features.global_signal,
    ):
        assert 0.0 <= value <= 1.0
