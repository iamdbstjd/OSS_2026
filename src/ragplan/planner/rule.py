"""Model-free, deterministic, budget-aware Stage 8 rule planner."""

from __future__ import annotations

import hashlib
from importlib.resources import files
from pathlib import Path
from typing import Final, Literal, Self

from pydantic import Field, model_validator

from ragplan.core.deadline import Deadline
from ragplan.core.errors import ErrorCode, RAGPlanError
from ragplan.core.models import (
    FrozenModel,
    PlanEstimate,
    PlannerDecision,
    PlannerMode,
    QueryAnalysis,
)
from ragplan.ingestion.audit import RuleGraphTierPolicy
from ragplan.planner.catalog import PlanCatalog, canonical_json, stable_tie_break_key
from ragplan.planner.features import FEATURE_SCHEMA_VERSION, load_default_query_feature_config

RULE_CONFIG_VERSION: Final[Literal["rule_v1"]] = "rule_v1"
DEFAULT_RULE_CONFIG_PATH: Final = (
    Path(__file__).resolve().parents[3] / "configs" / "rule_planner_v1.json"
)
_CANDIDATE_IDS: Final = frozenset({"P0", "P1", "P4", "P5", "P6", "P8"})


class StaticPlanProfile(FrozenModel):
    plan_id: str = Field(pattern=r"^P(?:0|[1-9][0-9]*)$")
    predicted_p95_latency_ms: float = Field(gt=0.0)


class RulePlannerConfig(FrozenModel):
    schema_version: Literal["rule_v1"] = RULE_CONFIG_VERSION
    feature_schema_version: Literal["qf_v1"] = FEATURE_SCHEMA_VERSION
    threshold_tuning_split: Literal["validation"] = "validation"
    low_budget_threshold_ms: float = Field(gt=0.0)
    relation_threshold: float = Field(ge=0.0, le=1.0)
    multi_hop_threshold: float = Field(ge=0.0, le=1.0)
    comparison_threshold: float = Field(ge=0.0, le=1.0)
    aggregation_threshold: float = Field(ge=0.0, le=1.0)
    global_threshold: float = Field(ge=0.0, le=1.0)
    candidate_profiles: tuple[StaticPlanProfile, ...] = Field(min_length=1)
    simple_plan_order: tuple[str, ...] = Field(min_length=1)
    relation_plan_order: tuple[str, ...] = Field(min_length=1)
    multi_hop_plan_order: tuple[str, ...] = Field(min_length=1)
    top_k_surcharge_ms: float = Field(ge=0.0)

    @model_validator(mode="after")
    def _valid_plan_space(self) -> Self:
        profile_ids = tuple(profile.plan_id for profile in self.candidate_profiles)
        expected_profile_ids = ("P0", "P1", "P4", "P5", "P6", "P8")
        if profile_ids != expected_profile_ids:
            raise ValueError("rule planner profiles must be ordered P0,P1,P4,P5,P6,P8")
        latencies = tuple(item.predicted_p95_latency_ms for item in self.candidate_profiles)
        if any(current >= following for current, following in zip(latencies, latencies[1:])):
            raise ValueError("rule planner safe p95 profiles must be strictly increasing")
        expected_orders = (
            (self.simple_plan_order, ("P1", "P0")),
            (self.relation_plan_order, ("P6", "P5", "P4", "P0")),
            (self.multi_hop_plan_order, ("P8", "P6", "P5", "P4", "P0")),
        )
        if any(actual != expected for actual, expected in expected_orders):
            raise ValueError("rule planner preference ladders are immutable in rule_v1")
        return self

    @property
    def sha256(self) -> str:
        payload = canonical_json(self.model_dump(mode="json"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_rule_planner_config(path: Path) -> RulePlannerConfig:
    try:
        return RulePlannerConfig.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError("rule planner config is missing or invalid") from exc


def load_default_rule_planner_config() -> RulePlannerConfig:
    if DEFAULT_RULE_CONFIG_PATH.is_file():
        return load_rule_planner_config(DEFAULT_RULE_CONFIG_PATH)
    payload = (
        files("ragplan.resources").joinpath("rule_planner_v1.json").read_text(encoding="utf-8")
    )
    try:
        return RulePlannerConfig.model_validate_json(payload)
    except Exception as exc:
        raise ValueError("packaged rule planner config is invalid") from exc


class RulePlanner:
    """Select only immutable catalog plans and expose every safety/budget decision."""

    def __init__(
        self,
        *,
        catalog: PlanCatalog,
        graph_policy: RuleGraphTierPolicy,
        config: RulePlannerConfig,
        feature_config_sha256: str | None = None,
    ) -> None:
        self._catalog = catalog
        self._graph_policy = graph_policy
        self._config = config
        for plan_id in _CANDIDATE_IDS:
            plan = catalog.plan_for_id(plan_id)
            if not plan.enabled_in_p0:
                raise ValueError("rule candidate is disabled in P0")
        self._feature_config_sha256 = (
            feature_config_sha256
            if feature_config_sha256 is not None
            else load_default_query_feature_config().sha256
        )
        if len(self._feature_config_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self._feature_config_sha256
        ):
            raise ValueError("feature config identity must be a lowercase SHA-256")
        policy_json = canonical_json(graph_policy.model_dump(mode="json"))
        identity = f"{config.sha256}:{self._feature_config_sha256}:{catalog.sha256()}:{policy_json}"
        self._config_version = hashlib.sha256(identity.encode("utf-8")).hexdigest()

    @property
    def config_version(self) -> str:
        return self._config_version

    @property
    def feature_config_sha256(self) -> str:
        return self._feature_config_sha256

    def select(
        self,
        analysis: QueryAnalysis,
        *,
        deadline: Deadline,
        graph_runtime_available: bool = True,
    ) -> PlannerDecision:
        remaining_ms = deadline.snapshot().branch_remaining_ms
        if remaining_ms <= 0:
            raise RAGPlanError(ErrorCode.DEADLINE_EXCEEDED, "rule planning deadline exceeded")
        graph_allowed = self._graph_policy.graph_tier_enabled and graph_runtime_available
        profiles = {item.plan_id: item for item in self._config.candidate_profiles}
        estimates = tuple(
            self._estimate(
                plan_id,
                analysis=analysis,
                remaining_ms=remaining_ms,
                graph_allowed=graph_allowed,
                profile=profiles[plan_id],
            )
            for plan_id in sorted(_CANDIDATE_IDS, key=lambda value: int(value[1:]))
        )
        estimate_by_id = {item.plan_id: item for item in estimates}
        features = analysis.features
        graph_intent = any(
            (
                features.relation_signal >= self._config.relation_threshold,
                features.multi_hop_signal >= self._config.multi_hop_threshold,
                features.comparison_signal >= self._config.comparison_threshold,
                features.aggregation_signal >= self._config.aggregation_threshold,
                features.global_signal >= self._config.global_threshold,
            )
        )
        matched: list[str] = []
        fallback_reason: str | None = None
        order: tuple[str, ...]
        if not analysis.language_supported:
            matched.append("unsupported_language_vector_safe")
            order = ("P0",)
            fallback_reason = "unsupported_language"
        elif remaining_ms <= self._config.low_budget_threshold_ms:
            matched.append("low_remaining_budget")
            order = ("P0",)
        elif graph_intent and not graph_allowed:
            matched.append("graph_tier_unavailable")
            order = self._config.simple_plan_order
            fallback_reason = (
                "graph_runtime_unavailable"
                if self._graph_policy.graph_tier_enabled
                else f"graph_audit_gate:{self._graph_policy.reason}"
            )
        elif features.multi_hop_signal >= self._config.multi_hop_threshold:
            matched.append("multi_hop_signal")
            order = self._config.multi_hop_plan_order
        elif graph_intent:
            if features.comparison_signal >= self._config.comparison_threshold:
                matched.append("comparison_signal")
            if features.aggregation_signal >= self._config.aggregation_threshold:
                matched.append("aggregation_signal")
            if features.global_signal >= self._config.global_threshold:
                matched.append("global_signal")
            if features.relation_signal >= self._config.relation_threshold:
                matched.append("relation_signal")
            order = self._config.relation_plan_order
        else:
            matched.append("simple_vector_default")
            order = self._config.simple_plan_order

        preference = {plan_id: index for index, plan_id in enumerate(order)}
        feasible_ids = tuple(plan_id for plan_id in order if estimate_by_id[plan_id].feasible)
        selected_id = (
            min(
                feasible_ids,
                key=lambda plan_id: (
                    preference[plan_id],
                    *stable_tie_break_key(
                        self._catalog.plan_for_id(plan_id),
                        estimate_by_id[plan_id].predicted_p95_latency_ms or 0.0,
                    ),
                ),
            )
            if feasible_ids
            else "P0"
        )
        plan = self._catalog.plan_for_id(selected_id)
        selected_estimate = estimate_by_id[selected_id]
        if not selected_estimate.feasible:
            matched.append("no_profile_within_budget")
        effective_mode = PlannerMode.FIXED_HYBRID if plan.graph_enabled else PlannerMode.VECTOR
        return PlannerDecision(
            mode=PlannerMode.RULE,
            effective_mode=effective_mode,
            selected_plan_id=plan.id,
            selected_plan=plan,
            executed_vector_top_k=(
                max(features.final_top_k, plan.vector_top_k) if plan.vector_enabled else None
            ),
            executed_graph_top_k=(
                max(features.final_top_k, plan.graph_top_k) if plan.graph_enabled else None
            ),
            matched_rules=tuple(matched),
            remaining_budget_ms=remaining_ms,
            candidate_estimates=estimates,
            budget_feasible=selected_estimate.feasible,
            selection_reason=(
                f"{RULE_CONFIG_VERSION} selected {plan.id} from "
                f"{','.join(matched)} using remaining-budget feasibility"
            ),
            fallback_reason=fallback_reason,
            feature_version=FEATURE_SCHEMA_VERSION,
            config_version=self._config_version,
        )

    def _estimate(
        self,
        plan_id: str,
        *,
        analysis: QueryAnalysis,
        remaining_ms: float,
        graph_allowed: bool,
        profile: StaticPlanProfile,
    ) -> PlanEstimate:
        plan = self._catalog.plan_for_id(plan_id)
        baseline_candidates = max(plan.vector_top_k, plan.graph_top_k)
        extra = max(0, analysis.features.final_top_k - baseline_candidates)
        predicted_ms = profile.predicted_p95_latency_ms + extra * self._config.top_k_surcharge_ms
        infeasible_reason: str | None = None
        if plan.graph_enabled and not analysis.language_supported:
            infeasible_reason = "unsupported_language"
        elif plan.graph_enabled and not graph_allowed:
            infeasible_reason = "graph_tier_unavailable"
        elif predicted_ms > remaining_ms:
            infeasible_reason = "predicted_p95_exceeds_remaining_budget"
        return PlanEstimate(
            plan_id=plan.id,
            predicted_p95_latency_ms=predicted_ms,
            feasible=infeasible_reason is None,
            infeasible_reason=infeasible_reason,
        )
