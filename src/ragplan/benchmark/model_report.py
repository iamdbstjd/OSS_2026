"""Stage 11 validation metrics, policy simulation, and model-gate decision."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Annotated, Literal, Self

import numpy as np
from numpy.typing import NDArray
from pydantic import Field, model_validator
from sklearn.ensemble import HistGradientBoostingRegressor  # type: ignore[import-untyped]
from sklearn.inspection import permutation_importance  # type: ignore[import-untyped]

from ragplan.benchmark.contracts import SplitName, canonical_json_bytes
from ragplan.benchmark.oracle import OracleReport
from ragplan.benchmark.profile_records import TrainingMatrixRow
from ragplan.core.models import FrozenFloatMapping, FrozenModel, NonEmptyString, Sha256Hex
from ragplan.planner.artifacts import ArtifactStatus
from ragplan.planner.latency_model import predict_p95_latency
from ragplan.planner.quality_model import predict_quality
from ragplan.planner.training import FEATURE_NAMES, PLAN_CATEGORIES, TrainingDatasets, encode_rows


class QualityEvaluation(FrozenModel):
    validation_rows: Annotated[int, Field(ge=1)]
    mae: float = Field(ge=0.0)
    rmse: float = Field(ge=0.0)
    plan_pair_ranking_accuracy: float = Field(ge=0.0, le=1.0)
    ranked_plan_pairs: Annotated[int, Field(ge=1)]
    predicted_best_plan_regret: float = Field(ge=0.0, le=1.0)
    residual_bias: float
    absolute_error_p50: float = Field(ge=0.0)
    absolute_error_p95: float = Field(ge=0.0)
    mae_by_plan: FrozenFloatMapping
    mae_by_query_tag: FrozenFloatMapping
    permutation_importance: FrozenFloatMapping


class LatencyEvaluation(FrozenModel):
    validation_rows: Annotated[int, Field(ge=1)]
    mae_ms: float = Field(ge=0.0)
    rmse_ms: float = Field(ge=0.0)
    coverage_overall: float = Field(ge=0.0, le=1.0)
    coverage_by_plan: FrozenFloatMapping
    missing_validation_plans: tuple[NonEmptyString, ...] = ()
    severe_underprediction_rate: float = Field(ge=0.0, le=1.0)
    pinball_loss: float = Field(ge=0.0)
    constant_plan_pinball_loss: float = Field(ge=0.0)
    pinball_improvement: float
    residual_p50_ms: float
    residual_p95_ms: float
    mae_by_plan: FrozenFloatMapping
    mae_by_query_tag: FrozenFloatMapping
    permutation_importance: FrozenFloatMapping


class PolicyEvaluation(FrozenModel):
    validation_query_budget_count: Annotated[int, Field(ge=1)]
    mean_recall_at_10: float = Field(ge=0.0, le=1.0)
    mean_budget_violation_rate: float = Field(ge=0.0, le=1.0)
    rule_mean_recall_at_10: float = Field(ge=0.0, le=1.0)
    rule_mean_budget_violation_rate: float = Field(ge=0.0, le=1.0)
    recall_difference_vs_rule: float
    violation_rate_difference_vs_rule: float
    oracle_comparable_count: Annotated[int, Field(ge=1)]
    mean_policy_regret_vs_oracle: float = Field(ge=0.0, le=1.0)
    no_predicted_feasible_count: Annotated[int, Field(ge=0)]
    selected_plan_distribution: FrozenFloatMapping


class CostModelReport(FrozenModel):
    schema_version: Literal["cost_model_report_v1"] = "cost_model_report_v1"
    status: ArtifactStatus
    training_matrix_sha256: Sha256Hex
    feature_schema_version: NonEmptyString
    feature_names_sha256: Sha256Hex
    raw_embeddings_used: Literal[False] = False
    train_query_count: Annotated[int, Field(ge=1)]
    validation_query_count: Annotated[int, Field(ge=1)]
    quality_train_rows: Annotated[int, Field(ge=1)]
    quality_validation_rows: Annotated[int, Field(ge=1)]
    latency_train_rows: Annotated[int, Field(ge=1)]
    latency_validation_rows: Annotated[int, Field(ge=1)]
    quality: QualityEvaluation
    latency: LatencyEvaluation
    policy: PolicyEvaluation
    gate_failures: tuple[NonEmptyString, ...] = ()

    @model_validator(mode="after")
    def _status_matches(self) -> Self:
        if (self.status is ArtifactStatus.SERVING_ELIGIBLE) is not (not self.gate_failures):
            raise ValueError("cost model report status must reflect all gate failures")
        return self

    @property
    def sha256(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.model_dump(mode="json"))).hexdigest()


def build_cost_model_report(
    datasets: TrainingDatasets,
    quality_model: HistGradientBoostingRegressor,
    latency_model: HistGradientBoostingRegressor,
    *,
    oracle: OracleReport,
    rule_baseline: Mapping[tuple[str, int], tuple[float, float]],
    training_matrix_sha256: str,
    feature_schema_version: str,
) -> CostModelReport:
    quality_prediction = predict_quality(
        quality_model,
        datasets.quality_validation.features,
    )
    latency_prediction = predict_p95_latency(
        latency_model,
        datasets.latency_validation.features,
    )
    quality = _quality_evaluation(
        datasets.quality_validation.rows,
        datasets.quality_validation.targets,
        quality_prediction,
        quality_model,
        datasets.quality_validation.features,
    )
    latency = _latency_evaluation(
        datasets,
        latency_model,
        latency_prediction,
    )
    policy = _policy_evaluation(
        datasets.all_rows,
        quality_model,
        latency_model,
        oracle=oracle,
        rule_baseline=rule_baseline,
    )
    gate_failures = _gate_failures(quality, latency, policy)
    query_splits = {row.query_id: row.split for row in datasets.all_rows}
    return CostModelReport(
        status=(
            ArtifactStatus.SERVING_ELIGIBLE if not gate_failures else ArtifactStatus.RESEARCH_ONLY
        ),
        training_matrix_sha256=training_matrix_sha256,
        feature_schema_version=feature_schema_version,
        feature_names_sha256=hashlib.sha256(canonical_json_bytes(list(FEATURE_NAMES))).hexdigest(),
        train_query_count=sum(split is SplitName.TRAIN for split in query_splits.values()),
        validation_query_count=sum(
            split is SplitName.VALIDATION for split in query_splits.values()
        ),
        quality_train_rows=len(datasets.quality_train.rows),
        quality_validation_rows=len(datasets.quality_validation.rows),
        latency_train_rows=len(datasets.latency_train.rows),
        latency_validation_rows=len(datasets.latency_validation.rows),
        quality=quality,
        latency=latency,
        policy=policy,
        gate_failures=gate_failures,
    )


def load_rule_baseline(path: Path) -> dict[tuple[str, int], tuple[float, float]]:
    grouped: dict[tuple[str, int], list[tuple[float, float]]] = defaultdict(list)
    with path.open(encoding="utf-8") as source:
        for line in source:
            record = json.loads(line)
            if (
                record.get("split") == "validation"
                and record.get("trial_phase") == "measured"
                and record.get("method") == "rule"
            ):
                grouped[(record["query_id"], int(record["latency_budget_ms"]))].append(
                    (
                        float(record["recall_at_10"]),
                        float(bool(record["budget_violated"])),
                    )
                )
    result: dict[tuple[str, int], tuple[float, float]] = {}
    for key, values in sorted(grouped.items()):
        if len(values) != 10:
            raise ValueError("Rule baseline does not contain ten measured trials per group")
        result[key] = (
            sum(value[0] for value in values) / len(values),
            sum(value[1] for value in values) / len(values),
        )
    if len(result) != 120 * 4:
        raise ValueError("Rule baseline is missing validation query-budget groups")
    return result


def _quality_evaluation(
    rows: tuple[TrainingMatrixRow, ...],
    actual: NDArray[np.float64],
    predicted: NDArray[np.float64],
    estimator: HistGradientBoostingRegressor,
    features: NDArray[np.float64],
) -> QualityEvaluation:
    residual = predicted - actual
    absolute = np.abs(residual)
    pair_correct, pair_count = _pair_ranking(rows, actual, predicted)
    if pair_count == 0:
        raise ValueError("quality ranking metric has no non-tied plan pairs")
    importance = permutation_importance(
        estimator,
        features,
        actual,
        scoring="neg_mean_absolute_error",
        n_repeats=3,
        random_state=20260809,
    ).importances_mean
    return QualityEvaluation(
        validation_rows=len(rows),
        mae=float(np.mean(absolute)),
        rmse=float(np.sqrt(np.mean(np.square(residual)))),
        plan_pair_ranking_accuracy=pair_correct / pair_count,
        ranked_plan_pairs=pair_count,
        predicted_best_plan_regret=_quality_policy_regret(rows, actual, predicted),
        residual_bias=float(np.mean(residual)),
        absolute_error_p50=float(np.quantile(absolute, 0.50, method="linear")),
        absolute_error_p95=float(np.quantile(absolute, 0.95, method="linear")),
        mae_by_plan=_segmented_mae(rows, actual, predicted, segment="plan"),
        mae_by_query_tag=_segmented_mae(rows, actual, predicted, segment="tag"),
        permutation_importance=_importance_mapping(importance),
    )


def _latency_evaluation(
    datasets: TrainingDatasets,
    estimator: HistGradientBoostingRegressor,
    predicted: NDArray[np.float64],
) -> LatencyEvaluation:
    rows = datasets.latency_validation.rows
    actual = datasets.latency_validation.targets
    residual = predicted - actual
    coverage = actual <= predicted
    baseline_by_plan = {
        plan_id: float(np.quantile(values, 0.95, method="linear"))
        for plan_id, values in _targets_by_plan(
            datasets.latency_train.rows,
            datasets.latency_train.targets,
        ).items()
    }
    baseline_prediction = np.asarray(
        [baseline_by_plan[row.plan_id] for row in rows],
        dtype=np.float64,
    )
    model_loss = _pinball_loss(actual, predicted, quantile=0.95)
    baseline_loss = _pinball_loss(actual, baseline_prediction, quantile=0.95)

    def pinball_scorer(
        candidate: HistGradientBoostingRegressor,
        x_values: NDArray[np.float64],
        y_values: NDArray[np.float64],
    ) -> float:
        candidate_prediction = np.maximum(candidate.predict(x_values), 0.0)
        return -_pinball_loss(y_values, candidate_prediction, quantile=0.95)

    importance = permutation_importance(
        estimator,
        datasets.latency_validation.features,
        actual,
        scoring=pinball_scorer,
        n_repeats=3,
        random_state=20260809,
    ).importances_mean
    coverage_by_plan: dict[str, float] = {}
    observed_plans = {row.plan_id for row in rows}
    for plan_id in PLAN_CATEGORIES:
        indexes = [index for index, row in enumerate(rows) if row.plan_id == plan_id]
        coverage_by_plan[plan_id] = float(np.mean(coverage[indexes])) if indexes else 0.0
    return LatencyEvaluation(
        validation_rows=len(rows),
        mae_ms=float(np.mean(np.abs(residual))),
        rmse_ms=float(np.sqrt(np.mean(np.square(residual)))),
        coverage_overall=float(np.mean(coverage)),
        coverage_by_plan=coverage_by_plan,
        missing_validation_plans=tuple(
            plan_id for plan_id in PLAN_CATEGORIES if plan_id not in observed_plans
        ),
        severe_underprediction_rate=float(np.mean(actual > predicted * 1.20)),
        pinball_loss=model_loss,
        constant_plan_pinball_loss=baseline_loss,
        pinball_improvement=(
            (baseline_loss - model_loss) / baseline_loss if baseline_loss > 0.0 else 0.0
        ),
        residual_p50_ms=float(np.quantile(residual, 0.50, method="linear")),
        residual_p95_ms=float(np.quantile(residual, 0.95, method="linear")),
        mae_by_plan=_segmented_mae(rows, actual, predicted, segment="plan"),
        mae_by_query_tag=_segmented_mae(rows, actual, predicted, segment="tag"),
        permutation_importance=_importance_mapping(importance),
    )


def _policy_evaluation(
    all_rows: Sequence[TrainingMatrixRow],
    quality_model: HistGradientBoostingRegressor,
    latency_model: HistGradientBoostingRegressor,
    *,
    oracle: OracleReport,
    rule_baseline: Mapping[tuple[str, int], tuple[float, float]],
) -> PolicyEvaluation:
    rows = tuple(row for row in all_rows if row.split is SplitName.VALIDATION)
    quality_prediction = predict_quality(quality_model, encode_rows(rows))
    latency_prediction = predict_p95_latency(latency_model, encode_rows(rows))
    grouped: dict[tuple[str, int], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[(row.query_id, row.latency_budget_ms)].append(index)
    oracle_by_key = {(item.query_id, item.latency_budget_ms): item for item in oracle.selections}
    selected_recall: list[float] = []
    selected_violation: list[float] = []
    rule_recall: list[float] = []
    rule_violation: list[float] = []
    regrets: list[float] = []
    distribution: Counter[str] = Counter()
    no_feasible = 0
    for key in sorted(grouped):
        indexes = grouped[key]
        budget = key[1]
        reserve = min(20.0, max(5.0, budget * 0.05))
        feasible = [index for index in indexes if latency_prediction[index] + reserve <= budget]
        if not feasible:
            no_feasible += 1
            selected_index = next(index for index in indexes if rows[index].plan_id == "P0")
        else:
            selected_index = min(
                feasible,
                key=lambda index: (
                    -quality_prediction[index],
                    latency_prediction[index],
                    rows[index].plan_features.graph_depth,
                    int(rows[index].plan_id[1:]),
                ),
            )
        selected = rows[selected_index]
        distribution[selected.plan_id] += 1
        selected_recall.append(selected.recall_at_10)
        selected_violation.append(selected.budget_violation_rate)
        try:
            rule = rule_baseline[key]
        except KeyError as exc:
            raise ValueError("Rule baseline is missing a policy comparison group") from exc
        rule_recall.append(rule[0])
        rule_violation.append(rule[1])
        oracle_selection = oracle_by_key.get(key)
        if oracle_selection is not None and oracle_selection.oracle_recall_at_10 is not None:
            regrets.append(max(0.0, oracle_selection.oracle_recall_at_10 - selected.recall_at_10))
    if not regrets:
        raise ValueError("policy simulation has no Oracle-comparable groups")
    mean_recall = _mean(selected_recall)
    mean_violation = _mean(selected_violation)
    mean_rule_recall = _mean(rule_recall)
    mean_rule_violation = _mean(rule_violation)
    return PolicyEvaluation(
        validation_query_budget_count=len(grouped),
        mean_recall_at_10=mean_recall,
        mean_budget_violation_rate=mean_violation,
        rule_mean_recall_at_10=mean_rule_recall,
        rule_mean_budget_violation_rate=mean_rule_violation,
        recall_difference_vs_rule=mean_recall - mean_rule_recall,
        violation_rate_difference_vs_rule=mean_violation - mean_rule_violation,
        oracle_comparable_count=len(regrets),
        mean_policy_regret_vs_oracle=_mean(regrets),
        no_predicted_feasible_count=no_feasible,
        selected_plan_distribution={
            key: float(value) for key, value in sorted(distribution.items())
        },
    )


def _gate_failures(
    quality: QualityEvaluation,
    latency: LatencyEvaluation,
    policy: PolicyEvaluation,
) -> tuple[str, ...]:
    failures: list[str] = []
    checks = (
        (quality.mae <= 0.10, "quality_mae_gt_0.10"),
        (
            quality.plan_pair_ranking_accuracy >= 0.70,
            "quality_plan_pair_ranking_lt_0.70",
        ),
        (
            quality.predicted_best_plan_regret <= 0.05,
            "quality_policy_regret_gt_0.05",
        ),
        (latency.coverage_overall >= 0.90, "latency_coverage_lt_0.90"),
        (
            min(latency.coverage_by_plan.values()) >= 0.85,
            "latency_plan_coverage_lt_0.85",
        ),
        (
            latency.severe_underprediction_rate <= 0.02,
            "latency_severe_underprediction_gt_0.02",
        ),
        (latency.pinball_improvement >= 0.10, "latency_pinball_improvement_lt_0.10"),
        (policy.recall_difference_vs_rule >= -0.01, "policy_recall_delta_lt_-0.01"),
        (
            policy.violation_rate_difference_vs_rule <= 0.02,
            "policy_violation_delta_gt_0.02",
        ),
        (
            policy.mean_policy_regret_vs_oracle <= 0.05,
            "policy_oracle_regret_gt_0.05",
        ),
    )
    failures.extend(name for passed, name in checks if not passed)
    return tuple(failures)


def _pair_ranking(
    rows: Sequence[TrainingMatrixRow],
    actual: NDArray[np.float64],
    predicted: NDArray[np.float64],
) -> tuple[int, int]:
    grouped: dict[tuple[str, int], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[(row.query_id, row.latency_budget_ms)].append(index)
    correct = 0
    count = 0
    for indexes in grouped.values():
        for offset, left in enumerate(indexes):
            for right in indexes[offset + 1 :]:
                difference = actual[left] - actual[right]
                if math.isclose(float(difference), 0.0, abs_tol=1e-12):
                    continue
                predicted_difference = predicted[left] - predicted[right]
                count += 1
                correct += int(difference * predicted_difference > 0.0)
    return correct, count


def _quality_policy_regret(
    rows: Sequence[TrainingMatrixRow],
    actual: NDArray[np.float64],
    predicted: NDArray[np.float64],
) -> float:
    grouped: dict[tuple[str, int], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[(row.query_id, row.latency_budget_ms)].append(index)
    regrets = []
    for indexes in grouped.values():
        selected = min(
            indexes,
            key=lambda index: (
                -predicted[index],
                rows[index].plan_features.graph_depth,
                int(rows[index].plan_id[1:]),
            ),
        )
        regrets.append(max(0.0, float(max(actual[index] for index in indexes) - actual[selected])))
    return _mean(regrets)


def _segmented_mae(
    rows: Sequence[TrainingMatrixRow],
    actual: NDArray[np.float64],
    predicted: NDArray[np.float64],
    *,
    segment: str,
) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for index, row in enumerate(rows):
        error = abs(float(predicted[index] - actual[index]))
        if segment == "plan":
            grouped[row.plan_id].append(error)
        elif segment == "tag":
            for tag in row.query_tags:
                grouped[tag.value].append(error)
        else:
            raise ValueError("unknown error segment")
    return {key: _mean(values) for key, values in sorted(grouped.items())}


def _targets_by_plan(
    rows: Sequence[TrainingMatrixRow],
    targets: NDArray[np.float64],
) -> dict[str, NDArray[np.float64]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row, target in zip(rows, targets, strict=True):
        grouped[row.plan_id].append(float(target))
    return {key: np.asarray(values, dtype=np.float64) for key, values in grouped.items()}


def _pinball_loss(
    actual: NDArray[np.float64],
    predicted: NDArray[np.float64],
    *,
    quantile: float,
) -> float:
    error = actual - predicted
    return float(np.mean(np.maximum(quantile * error, (quantile - 1.0) * error)))


def _importance_mapping(values: NDArray[np.float64]) -> dict[str, float]:
    if values.shape != (len(FEATURE_NAMES),) or not np.isfinite(values).all():
        raise ValueError("permutation importance does not match the feature schema")
    return {name: float(value) for name, value in zip(FEATURE_NAMES, values, strict=True)}


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("mean requires at least one value")
    return sum(float(value) for value in values) / len(values)


__all__ = [
    "CostModelReport",
    "LatencyEvaluation",
    "PolicyEvaluation",
    "QualityEvaluation",
    "build_cost_model_report",
    "load_rule_baseline",
]
