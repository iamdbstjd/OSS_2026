"""Offline-only Stage 12 policy comparison and redacted decision evidence."""

from __future__ import annotations

import hashlib
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, model_validator

from ragplan.benchmark.aggregate import BestFixedSelection
from ragplan.benchmark.contracts import QueryTag, SourceDataset, SplitName, canonical_json_bytes
from ragplan.benchmark.oracle import OracleReport
from ragplan.benchmark.profile_records import ProfileTrialRecord, TrainingMatrixRow
from ragplan.benchmark.records import (
    BenchmarkMethod,
    RawTrialRecord,
    TrialPhase,
    parse_raw_record,
)
from ragplan.core.deadline import Deadline, ManualClock
from ragplan.core.models import (
    FrozenModel,
    NonEmptyString,
    NonNegativeFloat,
    PlanId,
    PlannerDecision,
    QueryAnalysis,
    Sha256Hex,
    UnitFloat,
)
from ragplan.planner.artifacts import CostModelArtifactManifest
from ragplan.planner.optimizer import (
    OfflineCostAwareOptimizer,
    OfflineResearchContext,
)
from ragplan.planner.rule import RulePlanner
from ragplan.planner.runtime_guard import (
    RuntimeGuardObservation,
    RuntimeGuardSnapshot,
    RuntimeModelGuard,
)


class PlanSelectionCount(FrozenModel):
    plan_id: NonEmptyString
    count: Annotated[int, Field(ge=1)]


class PolicyMetricSummary(FrozenModel):
    policy: NonEmptyString
    evaluated_group_count: Annotated[int, Field(ge=1)]
    mean_recall_at_10: UnitFloat
    mean_budget_violation_rate: UnitFloat
    selected_plan_distribution: tuple[PlanSelectionCount, ...] = Field(min_length=1)
    evidence_source: NonEmptyString


class PlannerOverheadSummary(FrozenModel):
    sample_count: Annotated[int, Field(ge=1)]
    mean_ms: NonNegativeFloat
    p50_ms: NonNegativeFloat
    p95_ms: NonNegativeFloat
    maximum_ms: NonNegativeFloat


class DecisionDisagreement(FrozenModel):
    rule_plan_id: PlanId
    cost_aware_plan_id: PlanId
    count: Annotated[int, Field(ge=1)]


class RuntimeGuardSimulation(FrozenModel):
    source: Literal["stage10_measured_execution_trials"] = "stage10_measured_execution_trials"
    final_snapshot: RuntimeGuardSnapshot
    evaluable_trial_count: Annotated[int, Field(ge=0)]
    missing_execution_label_group_count: Annotated[int, Field(ge=0)]
    first_disabled_after_observation: Annotated[int, Field(ge=20)] | None = None
    routed_to_rule_group_count: Annotated[int, Field(ge=0)]


class OfflinePolicyDecisionRecord(FrozenModel):
    schema_version: Literal["cost_policy_decision_v1"] = "cost_policy_decision_v1"
    execution_mode: Literal["research_only_offline"] = "research_only_offline"
    query_id: NonEmptyString
    split: Literal["validation"] = "validation"
    source_dataset: SourceDataset
    query_tags: tuple[QueryTag, ...] = Field(min_length=1)
    latency_budget_ms: Annotated[int, Field(ge=25, le=5000)]
    cost_aware_decision: PlannerDecision
    rule_plan_id: PlanId
    best_fixed_plan_id: PlanId
    oracle_plan_id: PlanId | None = None
    cost_aware_recall_at_10: UnitFloat
    cost_aware_budget_violation_rate: UnitFloat
    cost_aware_measured_p95_latency_ms: NonNegativeFloat | None = None
    rule_recall_at_10: UnitFloat
    rule_budget_violation_rate: UnitFloat
    best_fixed_recall_at_10: UnitFloat
    best_fixed_budget_violation_rate: UnitFloat
    oracle_recall_at_10: UnitFloat | None = None
    cost_aware_planner_overhead_ms: NonNegativeFloat
    rule_planner_overhead_ms: NonNegativeFloat
    guard_routed_to_rule: bool
    guard_observation_count_after: Annotated[int, Field(ge=0, le=100)]


class Stage12PolicyReport(FrozenModel):
    schema_version: Literal["stage12_policy_report_v1"] = "stage12_policy_report_v1"
    status: Literal["research_only"] = "research_only"
    execution_mode: Literal["offline_shadow"] = "offline_shadow"
    public_api_cost_aware_enabled: Literal[False] = False
    test_split_used: Literal[False] = False
    historical_runtime_replay: Literal[True] = True
    benchmark_metadata_required: Literal[True] = True
    training_matrix_sha256: Sha256Hex
    stage9_raw_sha256: Sha256Hex
    stage10_raw_sha256: Sha256Hex
    best_fixed_sha256: Sha256Hex
    oracle_sha256: Sha256Hex
    quality_manifest_sha256: Sha256Hex
    latency_manifest_sha256: Sha256Hex
    plan_catalog_sha256: Sha256Hex
    optimizer_config_version: Sha256Hex
    query_budget_count: Annotated[int, Field(ge=1)]
    candidate_plan_ids: tuple[PlanId, ...] = Field(min_length=1)
    candidate_estimate_count: Annotated[int, Field(ge=1)]
    invalid_prediction_count: Annotated[int, Field(ge=0)]
    no_feasible_candidate_count: Annotated[int, Field(ge=0)]
    decision_records_sha256: Sha256Hex
    cost_aware: PolicyMetricSummary
    rule: PolicyMetricSummary
    best_fixed: PolicyMetricSummary
    oracle: PolicyMetricSummary
    oracle_comparable_count: Annotated[int, Field(ge=1)]
    cost_recall_difference_vs_rule: float
    cost_violation_difference_vs_rule: float
    cost_recall_difference_vs_best_fixed: float
    cost_violation_difference_vs_best_fixed: float
    mean_cost_regret_vs_oracle: UnitFloat
    decision_disagreement_count: Annotated[int, Field(ge=0)]
    decision_disagreement_rate: UnitFloat
    disagreement_matrix: tuple[DecisionDisagreement, ...] = Field(min_length=1)
    cost_aware_overhead: PlannerOverheadSummary
    rule_overhead: PlannerOverheadSummary
    runtime_guard: RuntimeGuardSimulation
    inherited_model_gate_failures: tuple[NonEmptyString, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _closed_counts(self) -> Stage12PolicyReport:
        expected = self.query_budget_count * len(self.candidate_plan_ids)
        if self.candidate_estimate_count != expected:
            raise ValueError("candidate estimate count does not cover every decision")
        if sum(item.count for item in self.disagreement_matrix) != self.query_budget_count:
            raise ValueError("disagreement matrix does not cover every decision")
        return self

    @property
    def sha256(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.model_dump(mode="json"))).hexdigest()


class Stage12EvidenceManifest(FrozenModel):
    schema_version: Literal["stage12_policy_evidence_v1"] = "stage12_policy_evidence_v1"
    status: Literal["research_only"] = "research_only"
    execution_mode: Literal["offline_shadow"] = "offline_shadow"
    public_api_cost_aware_enabled: Literal[False] = False
    test_split_used: Literal[False] = False
    report_sha256: Sha256Hex
    report_file_sha256: Sha256Hex
    decision_records_file_sha256: Sha256Hex
    decision_record_count: Annotated[int, Field(ge=1)]
    candidate_estimate_count: Annotated[int, Field(ge=1)]
    invalid_prediction_count: Annotated[int, Field(ge=0)]
    no_feasible_candidate_count: Annotated[int, Field(ge=0)]
    quality_artifact_sha256: Sha256Hex
    latency_artifact_sha256: Sha256Hex
    cost_recall_difference_vs_rule: float
    cost_violation_difference_vs_rule: float
    mean_cost_regret_vs_oracle: UnitFloat
    planner_overhead_p95_ms: NonNegativeFloat
    runtime_guard_disabled: bool
    runtime_guard_disable_reason: str | None = None
    runtime_guard_first_disabled_after_observation: Annotated[int, Field(ge=20)] | None = None
    runtime_guard_routed_to_rule_group_count: Annotated[int, Field(ge=0)]
    inherited_model_gate_failures: tuple[NonEmptyString, ...] = Field(min_length=1)


class BaselineMetric(FrozenModel):
    recall_at_10: UnitFloat
    budget_violation_rate: UnitFloat


class GuardTrial(FrozenModel):
    actual_execution_latency_ms: NonNegativeFloat
    budget_violated: bool


def load_stage9_comparison_metrics(
    path: Path,
    *,
    best_fixed: BestFixedSelection,
    measured_runs: int = 10,
) -> tuple[
    dict[tuple[str, int], BaselineMetric],
    dict[tuple[str, int], BaselineMetric],
]:
    selected_methods = {item.latency_budget_ms: item.method for item in best_fixed.selections}
    rule_values: dict[tuple[str, int], list[RawTrialRecord]] = defaultdict(list)
    fixed_values: dict[tuple[str, int], list[RawTrialRecord]] = defaultdict(list)
    with path.open(encoding="utf-8") as source:
        for line in source:
            record = parse_raw_record(line)
            if (
                record.split is not SplitName.VALIDATION
                or record.trial_phase is not TrialPhase.MEASURED
            ):
                continue
            key = (record.query_id, record.latency_budget_ms)
            if record.method is BenchmarkMethod.RULE:
                rule_values[key].append(record)
            if record.method is selected_methods[record.latency_budget_ms]:
                fixed_values[key].append(record)
    return (
        _aggregate_baseline(rule_values, measured_runs=measured_runs),
        _aggregate_baseline(fixed_values, measured_runs=measured_runs),
    )


def load_stage10_guard_trials(
    path: Path,
) -> dict[tuple[str, int, str], tuple[GuardTrial, ...]]:
    grouped: dict[tuple[str, int, str], list[tuple[int, GuardTrial]]] = defaultdict(list)
    with path.open(encoding="utf-8") as source:
        for line in source:
            record = ProfileTrialRecord.model_validate_json(line)
            if (
                record.split is not SplitName.VALIDATION
                or record.trial_phase is not TrialPhase.MEASURED
                or not record.execution_latency_label_valid
                or record.execution_latency_ms is None
            ):
                continue
            grouped[(record.query_id, record.latency_budget_ms, record.plan_id)].append(
                (
                    record.repetition,
                    GuardTrial(
                        actual_execution_latency_ms=record.execution_latency_ms,
                        budget_violated=record.budget_violated,
                    ),
                )
            )
    result: dict[tuple[str, int, str], tuple[GuardTrial, ...]] = {}
    for key, values in grouped.items():
        ordered = sorted(values, key=lambda item: item[0])
        repetitions = tuple(item[0] for item in ordered)
        if len(repetitions) != len(set(repetitions)) or any(
            repetition not in range(10) for repetition in repetitions
        ):
            raise ValueError("Stage 10 guard evidence has invalid measured repetitions")
        result[key] = tuple(item[1] for item in ordered)
    return result


def evaluate_offline_policy(
    matrix: Sequence[TrainingMatrixRow],
    *,
    optimizer: OfflineCostAwareOptimizer,
    rule_planner: RulePlanner,
    oracle: OracleReport,
    best_fixed: BestFixedSelection,
    rule_metrics: Mapping[tuple[str, int], BaselineMetric],
    best_fixed_metrics: Mapping[tuple[str, int], BaselineMetric],
    guard_trials: Mapping[tuple[str, int, str], tuple[GuardTrial, ...]],
    quality_manifest: CostModelArtifactManifest,
    latency_manifest: CostModelArtifactManifest,
    training_matrix_sha256: str,
    stage9_raw_sha256: str,
    stage10_raw_sha256: str,
    catalog_sha256: str,
) -> tuple[tuple[OfflinePolicyDecisionRecord, ...], Stage12PolicyReport]:
    rows = tuple(row for row in matrix if row.split is SplitName.VALIDATION)
    if not rows or any(row.split is SplitName.TEST for row in rows):
        raise ValueError("offline policy comparison requires validation rows only")
    grouped: dict[tuple[str, int], list[TrainingMatrixRow]] = defaultdict(list)
    for row in rows:
        grouped[(row.query_id, row.latency_budget_ms)].append(row)
    first_group = sorted(grouped[next(iter(grouped))], key=_plan_key)
    candidate_ids = tuple(row.plan_id for row in first_group)
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("offline policy candidates must be unique")
    oracle_by_key = {(item.query_id, item.latency_budget_ms): item for item in oracle.selections}
    best_plan_by_budget = {item.latency_budget_ms: item.plan_id for item in best_fixed.selections}
    guard = RuntimeModelGuard(optimizer.model_version)
    records: list[OfflinePolicyDecisionRecord] = []
    cost_recall: list[float] = []
    cost_violation: list[float] = []
    rule_recall: list[float] = []
    rule_violation: list[float] = []
    fixed_recall: list[float] = []
    fixed_violation: list[float] = []
    oracle_recall: list[float] = []
    regrets: list[float] = []
    cost_distribution: Counter[str] = Counter()
    rule_distribution: Counter[str] = Counter()
    fixed_distribution: Counter[str] = Counter()
    oracle_distribution: Counter[str] = Counter()
    disagreement: Counter[tuple[str, str]] = Counter()
    cost_overhead: list[float] = []
    rule_overhead: list[float] = []
    invalid_predictions = 0
    no_feasible = 0
    guard_evaluable = 0
    guard_missing = 0
    guard_fallback_groups = 0
    first_disabled: int | None = None
    for key in sorted(grouped):
        candidates = sorted(grouped[key], key=_plan_key)
        if tuple(row.plan_id for row in candidates) != candidate_ids:
            raise ValueError("offline policy group does not contain the complete plan space")
        representative = candidates[0]
        if any(
            (row.query_features, row.source_dataset, row.query_tags)
            != (
                representative.query_features,
                representative.source_dataset,
                representative.query_tags,
            )
            for row in candidates
        ):
            raise ValueError("offline policy query metadata differs across candidate plans")
        analysis = _offline_analysis(representative)
        cost_result = optimizer.select(
            analysis,
            context=OfflineResearchContext(
                source_dataset=representative.source_dataset,
                query_tags=representative.query_tags,
            ),
            deadline=Deadline.start(key[1], clock=ManualClock()),
        )
        cost_decision = cost_result.decision
        cost_overhead.append(cost_result.planner_overhead_ms)
        invalid_predictions += sum(
            item.predicted_quality is None or item.predicted_p95_latency_ms is None
            for item in cost_decision.candidate_estimates
        )
        no_feasible += int(not cost_decision.budget_feasible)
        rule_started_ns = time.perf_counter_ns()
        rule_decision = rule_planner.select(
            analysis,
            deadline=Deadline.start(key[1], clock=ManualClock()),
            graph_runtime_available=True,
        )
        rule_overhead.append((time.perf_counter_ns() - rule_started_ns) / 1_000_000)
        selected = next(row for row in candidates if row.plan_id == cost_decision.selected_plan_id)
        fixed_plan_id = best_plan_by_budget[key[1]]
        observed_rule = _required_metric(rule_metrics, key, label="Rule")
        observed_fixed = _required_metric(best_fixed_metrics, key, label="BestFixed")
        oracle_selection = oracle_by_key.get(key)
        if oracle_selection is None:
            raise ValueError("Oracle evidence is missing one validation query-budget group")
        guard_routed = guard.disabled
        if guard_routed:
            guard_fallback_groups += 1
        else:
            trials = guard_trials.get((key[0], key[1], selected.plan_id), ())
            if not trials:
                guard_missing += 1
            predicted = next(
                item.predicted_p95_latency_ms
                for item in cost_decision.candidate_estimates
                if item.plan_id == selected.plan_id
            )
            if predicted is not None:
                for trial in trials:
                    snapshot = guard.observe(
                        RuntimeGuardObservation(
                            artifact_version=optimizer.model_version,
                            predicted_p95_latency_ms=predicted,
                            actual_execution_latency_ms=trial.actual_execution_latency_ms,
                            budget_violated=trial.budget_violated,
                        )
                    )
                    guard_evaluable += 1
                    if snapshot.disabled and first_disabled is None:
                        first_disabled = guard_evaluable
                        break
        cost_distribution[selected.plan_id] += 1
        rule_distribution[rule_decision.selected_plan_id] += 1
        fixed_distribution[fixed_plan_id] += 1
        oracle_id = oracle_selection.oracle_plan_id or "NO_FEASIBLE"
        oracle_distribution[oracle_id] += 1
        disagreement[(rule_decision.selected_plan_id, selected.plan_id)] += 1
        cost_recall.append(selected.recall_at_10)
        cost_violation.append(selected.budget_violation_rate)
        rule_recall.append(observed_rule.recall_at_10)
        rule_violation.append(observed_rule.budget_violation_rate)
        fixed_recall.append(observed_fixed.recall_at_10)
        fixed_violation.append(observed_fixed.budget_violation_rate)
        if oracle_selection.oracle_recall_at_10 is not None:
            oracle_recall.append(oracle_selection.oracle_recall_at_10)
            regrets.append(max(0.0, oracle_selection.oracle_recall_at_10 - selected.recall_at_10))
        records.append(
            OfflinePolicyDecisionRecord(
                query_id=key[0],
                source_dataset=representative.source_dataset,
                query_tags=representative.query_tags,
                latency_budget_ms=key[1],
                cost_aware_decision=cost_decision,
                rule_plan_id=rule_decision.selected_plan_id,
                best_fixed_plan_id=fixed_plan_id,
                oracle_plan_id=oracle_selection.oracle_plan_id,
                cost_aware_recall_at_10=selected.recall_at_10,
                cost_aware_budget_violation_rate=selected.budget_violation_rate,
                cost_aware_measured_p95_latency_ms=selected.p95_execution_latency_ms,
                rule_recall_at_10=observed_rule.recall_at_10,
                rule_budget_violation_rate=observed_rule.budget_violation_rate,
                best_fixed_recall_at_10=observed_fixed.recall_at_10,
                best_fixed_budget_violation_rate=observed_fixed.budget_violation_rate,
                oracle_recall_at_10=oracle_selection.oracle_recall_at_10,
                cost_aware_planner_overhead_ms=cost_result.planner_overhead_ms,
                rule_planner_overhead_ms=rule_overhead[-1],
                guard_routed_to_rule=guard_routed,
                guard_observation_count_after=guard.snapshot().observation_count,
            )
        )
    frozen_records = tuple(records)
    record_hash = hashlib.sha256(_decision_bytes(frozen_records)).hexdigest()
    mean_cost_recall = _mean(cost_recall)
    mean_cost_violation = _mean(cost_violation)
    mean_rule_recall = _mean(rule_recall)
    mean_rule_violation = _mean(rule_violation)
    mean_fixed_recall = _mean(fixed_recall)
    mean_fixed_violation = _mean(fixed_violation)
    if not oracle_recall or not regrets:
        raise ValueError("offline policy comparison has no Oracle-comparable groups")
    report = Stage12PolicyReport(
        training_matrix_sha256=training_matrix_sha256,
        stage9_raw_sha256=stage9_raw_sha256,
        stage10_raw_sha256=stage10_raw_sha256,
        best_fixed_sha256=best_fixed.sha256,
        oracle_sha256=hashlib.sha256(
            canonical_json_bytes(oracle.model_dump(mode="json"))
        ).hexdigest(),
        quality_manifest_sha256=quality_manifest.sha256,
        latency_manifest_sha256=latency_manifest.sha256,
        plan_catalog_sha256=catalog_sha256,
        optimizer_config_version=optimizer.config_version,
        query_budget_count=len(grouped),
        candidate_plan_ids=candidate_ids,
        candidate_estimate_count=len(grouped) * len(candidate_ids),
        invalid_prediction_count=invalid_predictions,
        no_feasible_candidate_count=no_feasible,
        decision_records_sha256=record_hash,
        cost_aware=_policy_summary(
            "cost_aware_research",
            cost_recall,
            cost_violation,
            cost_distribution,
            "stage10_training_matrix_validation",
        ),
        rule=_policy_summary(
            "rule",
            rule_recall,
            rule_violation,
            rule_distribution,
            "stage9_measured_validation",
        ),
        best_fixed=_policy_summary(
            "best_fixed",
            fixed_recall,
            fixed_violation,
            fixed_distribution,
            "stage9_measured_validation_selection",
        ),
        oracle=_policy_summary(
            "oracle_at_budget",
            oracle_recall,
            [0.0] * len(oracle_recall),
            oracle_distribution,
            "stage10_measured_validation_feasible_only",
        ),
        oracle_comparable_count=len(regrets),
        cost_recall_difference_vs_rule=mean_cost_recall - mean_rule_recall,
        cost_violation_difference_vs_rule=mean_cost_violation - mean_rule_violation,
        cost_recall_difference_vs_best_fixed=mean_cost_recall - mean_fixed_recall,
        cost_violation_difference_vs_best_fixed=mean_cost_violation - mean_fixed_violation,
        mean_cost_regret_vs_oracle=_mean(regrets),
        decision_disagreement_count=sum(
            count for (left, right), count in disagreement.items() if left != right
        ),
        decision_disagreement_rate=(
            sum(count for (left, right), count in disagreement.items() if left != right)
            / len(grouped)
        ),
        disagreement_matrix=tuple(
            DecisionDisagreement(rule_plan_id=left, cost_aware_plan_id=right, count=count)
            for (left, right), count in sorted(disagreement.items())
        ),
        cost_aware_overhead=_overhead(cost_overhead),
        rule_overhead=_overhead(rule_overhead),
        runtime_guard=RuntimeGuardSimulation(
            final_snapshot=guard.snapshot(),
            evaluable_trial_count=guard_evaluable,
            missing_execution_label_group_count=guard_missing,
            first_disabled_after_observation=first_disabled,
            routed_to_rule_group_count=guard_fallback_groups,
        ),
        inherited_model_gate_failures=quality_manifest.gate_failures,
    )
    return frozen_records, report


def _aggregate_baseline(
    grouped: Mapping[tuple[str, int], Sequence[RawTrialRecord]],
    *,
    measured_runs: int,
) -> dict[tuple[str, int], BaselineMetric]:
    result: dict[tuple[str, int], BaselineMetric] = {}
    for key, values in grouped.items():
        if len(values) != measured_runs:
            raise ValueError("baseline comparison group has an unexpected measured-run count")
        result[key] = BaselineMetric(
            recall_at_10=_mean([item.recall_at_10 for item in values]),
            budget_violation_rate=_mean([float(item.budget_violated) for item in values]),
        )
    if not result:
        raise ValueError("baseline comparison evidence is empty")
    return result


def _required_metric(
    metrics: Mapping[tuple[str, int], BaselineMetric],
    key: tuple[str, int],
    *,
    label: str,
) -> BaselineMetric:
    try:
        return metrics[key]
    except KeyError as exc:
        raise ValueError(f"{label} comparison evidence is missing a group") from exc


def _offline_analysis(row: TrainingMatrixRow) -> QueryAnalysis:
    return QueryAnalysis(
        normalized_query="offline-redacted-validation-query",
        language_supported=True,
        token_count=row.query_features.token_count,
        query_embedding=(),
        features=row.query_features,
        analyzer_version="stage10-qf-v1-replay",
        analysis_latency_ms=0.0,
    )


def _policy_summary(
    policy: str,
    recall: Sequence[float],
    violation: Sequence[float],
    distribution: Mapping[str, int],
    source: str,
) -> PolicyMetricSummary:
    return PolicyMetricSummary(
        policy=policy,
        evaluated_group_count=len(recall),
        mean_recall_at_10=_mean(recall),
        mean_budget_violation_rate=_mean(violation),
        selected_plan_distribution=tuple(
            PlanSelectionCount(plan_id=plan_id, count=count)
            for plan_id, count in sorted(
                distribution.items(), key=lambda item: _plan_id_key(item[0])
            )
        ),
        evidence_source=source,
    )


def _overhead(values: Sequence[float]) -> PlannerOverheadSummary:
    ordered = sorted(values)
    return PlannerOverheadSummary(
        sample_count=len(ordered),
        mean_ms=_mean(ordered),
        p50_ms=_quantile(ordered, 0.50),
        p95_ms=_quantile(ordered, 0.95),
        maximum_ms=ordered[-1],
    )


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("quantile requires observations")
    position = (len(values) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    fraction = position - lower
    return float(values[lower] + (values[upper] - values[lower]) * fraction)


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("mean requires observations")
    return sum(float(item) for item in values) / len(values)


def _plan_key(row: TrainingMatrixRow) -> int:
    return int(row.plan_id[1:])


def _plan_id_key(plan_id: str) -> int:
    return 999 if plan_id == "NO_FEASIBLE" else int(plan_id[1:])


def _decision_bytes(records: Sequence[OfflinePolicyDecisionRecord]) -> bytes:
    return (
        b"\n".join(canonical_json_bytes(item.model_dump(mode="json")) for item in records) + b"\n"
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


__all__ = [
    "BaselineMetric",
    "GuardTrial",
    "OfflinePolicyDecisionRecord",
    "PlannerOverheadSummary",
    "PolicyMetricSummary",
    "RuntimeGuardSimulation",
    "Stage12EvidenceManifest",
    "Stage12PolicyReport",
    "evaluate_offline_policy",
    "file_sha256",
    "load_stage10_guard_trials",
    "load_stage9_comparison_metrics",
]
