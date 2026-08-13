from __future__ import annotations

import pytest

from ragplan.core.deadline import Deadline, ManualClock
from ragplan.core.models import PlannerMode, QueryAnalysis, QueryFeatures
from ragplan.ingestion.audit import AuditStatus, RuleGraphTierPolicy
from ragplan.planner.catalog import load_default_plan_catalog
from ragplan.planner.rule import RulePlanner, load_default_rule_planner_config

pytestmark = pytest.mark.unit


def _policy(*, enabled: bool) -> RuleGraphTierPolicy:
    return RuleGraphTierPolicy(
        audit_sample_checksum="a" * 64,
        audit_status=AuditStatus.COMPLETE if enabled else AuditStatus.PENDING_HUMAN_REVIEW,
        observed_entity_f1=0.80 if enabled else None,
        observed_relation_precision=0.75 if enabled else None,
        graph_tier_enabled=enabled,
        reason="gates passed" if enabled else "human audit pending",
    )


def _analysis(
    *,
    supported: bool = True,
    relation: float = 0.0,
    multi_hop: float = 0.0,
) -> QueryAnalysis:
    return QueryAnalysis(
        normalized_query="test query",
        language_supported=supported,
        token_count=4,
        query_embedding=(1.0, 0.0),
        features=QueryFeatures(
            token_count=4,
            entity_count=2 if relation else 0,
            entity_density=0.5 if relation else 0.0,
            relation_signal=relation,
            multi_hop_signal=multi_hop,
            comparison_signal=0.0,
            aggregation_signal=0.0,
            global_signal=0.0,
            final_top_k=10,
        ),
        analyzer_version="stage8-test",
        analysis_latency_ms=1.0,
    )


def _planner(*, graph_enabled: bool) -> RulePlanner:
    return RulePlanner(
        catalog=load_default_plan_catalog(),
        graph_policy=_policy(enabled=graph_enabled),
        config=load_default_rule_planner_config(),
    )


@pytest.mark.parametrize(
    ("budget", "expected"),
    ((50, "P0"), (100, "P4"), (120, "P5"), (200, "P6")),
)
def test_relation_rule_changes_plan_with_remaining_budget(budget: int, expected: str) -> None:
    decision = _planner(graph_enabled=True).select(
        _analysis(relation=1.0),
        deadline=Deadline.start(budget, clock=ManualClock()),
    )

    assert decision.selected_plan_id == expected
    assert decision.mode is PlannerMode.RULE
    assert decision.selection_reason is not None
    assert decision.matched_rules


@pytest.mark.parametrize(("budget", "expected"), ((200, "P6"), (500, "P8")))
def test_multi_hop_uses_p6_or_p8_as_budget_allows(budget: int, expected: str) -> None:
    decision = _planner(graph_enabled=True).select(
        _analysis(relation=1.0, multi_hop=1.0),
        deadline=Deadline.start(budget, clock=ManualClock()),
    )

    assert decision.selected_plan_id == expected
    assert decision.effective_mode is PlannerMode.FIXED_HYBRID


def test_non_english_and_graph_gate_fail_closed_to_vector() -> None:
    planner = _planner(graph_enabled=False)
    unsupported = planner.select(
        _analysis(supported=False, relation=1.0),
        deadline=Deadline.start(500, clock=ManualClock()),
    )
    gated = planner.select(
        _analysis(relation=1.0),
        deadline=Deadline.start(500, clock=ManualClock()),
    )

    assert unsupported.selected_plan_id == "P0"
    assert unsupported.effective_mode is PlannerMode.VECTOR
    assert all(
        estimate.infeasible_reason == "unsupported_language"
        for estimate in unsupported.candidate_estimates
        if estimate.plan_id in {"P4", "P5", "P6", "P8"}
    )
    assert gated.effective_mode is PlannerMode.VECTOR
    assert gated.selected_plan is not None and not gated.selected_plan.graph_enabled
    assert gated.fallback_reason == "graph_audit_gate:human audit pending"
    assert all(
        not estimate.feasible
        for estimate in gated.candidate_estimates
        if estimate.plan_id in {"P4", "P5", "P6", "P8"}
    )


def test_open_graph_runtime_never_selects_a_graph_plan() -> None:
    decision = _planner(graph_enabled=True).select(
        _analysis(relation=1.0),
        deadline=Deadline.start(500, clock=ManualClock()),
        graph_runtime_available=False,
    )

    assert decision.effective_mode is PlannerMode.VECTOR
    assert decision.fallback_reason == "graph_runtime_unavailable"
    assert decision.config_version == _planner(graph_enabled=True).config_version


def test_rule_thresholds_are_validation_only_and_preference_ladders_are_frozen() -> None:
    config = load_default_rule_planner_config()

    assert config.threshold_tuning_split == "validation"
    assert config.simple_plan_order == ("P1", "P0")
    assert config.relation_plan_order == ("P6", "P5", "P4", "P0")
    assert config.multi_hop_plan_order == ("P8", "P6", "P5", "P4", "P0")
    assert [item.predicted_p95_latency_ms for item in config.candidate_profiles] == sorted(
        item.predicted_p95_latency_ms for item in config.candidate_profiles
    )
