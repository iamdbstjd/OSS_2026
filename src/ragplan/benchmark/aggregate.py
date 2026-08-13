"""Deterministic Stage 9 aggregation, validation-only BestFixed, and evidence output."""

from __future__ import annotations

import csv
import hashlib
import io
import math
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from ragplan.benchmark.artifacts import write_bytes, write_json, write_json_model
from ragplan.benchmark.contracts import SplitName, canonical_json_bytes, canonical_sha256
from ragplan.benchmark.records import (
    AGGREGATE_SCHEMA_VERSION,
    BENCHMARK_RANDOM_SEED,
    BOOTSTRAP_SAMPLES,
    EXECUTED_METHODS,
    FIXED_METHODS,
    AggregateDimension,
    BenchmarkMethod,
    BenchmarkQueryIdentity,
    BenchmarkRunManifest,
    RawTrialRecord,
    TrialPhase,
    TrialStatus,
    benchmark_query_identities_sha256,
    raw_records_sha256,
    trial_identity,
)
from ragplan.core.models import FrozenModel, NonEmptyString, Sha256Hex


class BootstrapInterval(FrozenModel):
    confidence_level: float = Field(default=0.95, ge=0.0, le=1.0)
    lower: float
    point: float
    upper: float
    samples: Literal[10000] = BOOTSTRAP_SAMPLES
    seed: Literal[20260809] = BENCHMARK_RANDOM_SEED

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        if not math.isclose(self.confidence_level, 0.95, rel_tol=0.0, abs_tol=0.0):
            raise ValueError("baseline_v1 confidence level is fixed at 0.95")
        if not self.lower <= self.point <= self.upper:
            raise ValueError("bootstrap interval must contain its point estimate")
        return self


class BestFixedBudget(FrozenModel):
    latency_budget_ms: Annotated[int, Field(ge=25, le=5000)]
    method: BenchmarkMethod
    plan_id: Literal["P4", "P5", "P6", "P8"]
    feasible: bool
    validation_recall_at_10: float = Field(ge=0.0, le=1.0)
    validation_p95_latency_ms: float = Field(ge=0.0)

    @model_validator(mode="after")
    def _fixed_only(self) -> Self:
        expected = {
            BenchmarkMethod.FIXED_P4: "P4",
            BenchmarkMethod.FIXED_P5: "P5",
            BenchmarkMethod.FIXED_P6: "P6",
            BenchmarkMethod.FIXED_P8: "P8",
        }
        if self.method not in expected or self.plan_id != expected[self.method]:
            raise ValueError("BestFixed must select one immutable fixed-hybrid baseline")
        return self


class BestFixedSelection(FrozenModel):
    schema_version: Literal["best_fixed_validation_v1"] = "best_fixed_validation_v1"
    run_id: NonEmptyString
    selection_split: Literal["validation"] = "validation"
    split_hash: Sha256Hex
    source_raw_sha256: Sha256Hex
    validation_query_ids_sha256: Sha256Hex
    selections: tuple[BestFixedBudget, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_budgets(self) -> Self:
        budgets = tuple(item.latency_budget_ms for item in self.selections)
        if budgets != tuple(sorted(set(budgets))):
            raise ValueError("BestFixed selections must have unique ascending budgets")
        return self

    @property
    def sha256(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.model_dump(mode="json"))).hexdigest()

    def for_budget(self, budget_ms: int) -> BestFixedBudget:
        for selection in self.selections:
            if selection.latency_budget_ms == budget_ms:
                return selection
        raise KeyError(budget_ms)


class GroupAggregate(FrozenModel):
    phase: Literal["cold", "measured"]
    dimension: AggregateDimension
    group: NonEmptyString
    method: BenchmarkMethod
    selected_fixed_method: BenchmarkMethod | None = None
    latency_budget_ms: Annotated[int, Field(ge=25, le=5000)]
    query_count: Annotated[int, Field(ge=1)]
    trial_count: Annotated[int, Field(ge=1)]
    recall_at_5: float = Field(ge=0.0, le=1.0)
    recall_at_10: float = Field(ge=0.0, le=1.0)
    mrr_at_10: float = Field(ge=0.0, le=1.0)
    ndcg_at_10: float = Field(ge=0.0, le=1.0)
    p50_latency_ms: float = Field(ge=0.0)
    p95_latency_ms: float = Field(ge=0.0)
    p99_latency_ms: float = Field(ge=0.0)
    timeout_rate: float = Field(ge=0.0, le=1.0)
    fallback_rate: float = Field(ge=0.0, le=1.0)
    error_rate: float = Field(ge=0.0, le=1.0)
    budget_violation_rate: float = Field(ge=0.0, le=1.0)
    recall_at_10_ci: BootstrapInterval | None = None

    @model_validator(mode="after")
    def _best_fixed_provenance(self) -> Self:
        if (self.method is BenchmarkMethod.BEST_FIXED) is not (
            self.selected_fixed_method is not None
        ):
            raise ValueError("derived BestFixed rows require their validation-selected method")
        return self


class PairedComparison(FrozenModel):
    latency_budget_ms: Annotated[int, Field(ge=25, le=5000)]
    left_method: BenchmarkMethod
    right_method: Literal["best_fixed"] = "best_fixed"
    right_selected_method: BenchmarkMethod
    query_count: Annotated[int, Field(ge=1)]
    recall_at_10_difference: BootstrapInterval


class AggregateReport(FrozenModel):
    schema_version: Literal["benchmark_aggregate_v1"] = AGGREGATE_SCHEMA_VERSION
    run_id: NonEmptyString
    raw_record_count: Annotated[int, Field(ge=1)]
    raw_logical_sha256: Sha256Hex
    protocol_config_sha256: Sha256Hex
    environment_manifest_sha256: Sha256Hex
    benchmark_manifest_sha256: Sha256Hex
    split_hash: Sha256Hex
    qrels_sha256: Sha256Hex
    query_ids_sha256: Sha256Hex
    query_identities_sha256: Sha256Hex
    corpus_version: NonEmptyString
    corpus_chunk_count: Annotated[int, Field(ge=1)]
    corpus_chunk_ids_sha256: Sha256Hex
    embedding_model_revision: NonEmptyString
    extractor_version: NonEmptyString
    plan_catalog_sha256: Sha256Hex
    planner_config_sha256: Sha256Hex
    query_feature_config_sha256: Sha256Hex
    graph_tier_policy_sha256: Sha256Hex
    rule_runtime_config_version: Sha256Hex
    stage2_artifact_set_sha256: Sha256Hex
    runtime_semantics_version: Literal["v1"] = "v1"
    percentile_method: Literal["hyndman_fan_type_7_linear"] = "hyndman_fan_type_7_linear"
    bootstrap_samples: Literal[10000] = BOOTSTRAP_SAMPLES
    bootstrap_seed: Literal[20260809] = BENCHMARK_RANDOM_SEED
    outlier_policy: Literal["retain_all"] = "retain_all"
    best_fixed: BestFixedSelection
    summaries: tuple[GroupAggregate, ...] = Field(min_length=1)
    paired_comparisons: tuple[PairedComparison, ...]


def percentile_type7(values: Sequence[float], probability: float) -> float:
    """Hyndman–Fan type 7, equivalent to NumPy/Pandas ``linear`` percentile."""

    if not values:
        raise ValueError("percentile requires at least one value")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("percentile probability must be in [0, 1]")
    ordered = sorted(float(value) for value in values)
    if any(not math.isfinite(value) for value in ordered):
        raise ValueError("percentile values must be finite")
    index = (len(ordered) - 1) * probability
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    fraction = index - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def clustered_bootstrap_mean_interval(
    values_by_query: Mapping[str, float],
    *,
    seed: int = BENCHMARK_RANDOM_SEED,
    samples: int = BOOTSTRAP_SAMPLES,
) -> BootstrapInterval:
    """Resample query clusters, never individual repetitions, with deterministic RNG."""

    if samples != BOOTSTRAP_SAMPLES or seed != BENCHMARK_RANDOM_SEED:
        raise ValueError("baseline_v1 bootstrap count and seed are immutable")
    query_ids = tuple(sorted(values_by_query))
    if not query_ids:
        raise ValueError("bootstrap requires at least one query cluster")
    values = tuple(float(values_by_query[query_id]) for query_id in query_ids)
    if any(not math.isfinite(value) for value in values):
        raise ValueError("bootstrap values must be finite")
    point = sum(values) / len(values)
    rng = random.Random(seed)
    estimates = [
        sum(values[rng.randrange(len(values))] for _ in values) / len(values)
        for _ in range(samples)
    ]
    lower = percentile_type7(estimates, 0.025)
    upper = percentile_type7(estimates, 0.975)
    # Finite bootstrap samples can narrowly miss the exact point; include it explicitly.
    return BootstrapInterval(lower=min(lower, point), point=point, upper=max(upper, point))


def paired_bootstrap_difference(
    left_by_query: Mapping[str, float],
    right_by_query: Mapping[str, float],
    *,
    seed: int = BENCHMARK_RANDOM_SEED,
) -> BootstrapInterval:
    if set(left_by_query) != set(right_by_query) or not left_by_query:
        raise ValueError("paired bootstrap requires identical non-empty query clusters")
    differences = {
        query_id: left_by_query[query_id] - right_by_query[query_id]
        for query_id in sorted(left_by_query)
    }
    return clustered_bootstrap_mean_interval(differences, seed=seed)


def aggregate_run(
    records: Sequence[RawTrialRecord],
    *,
    manifest: BenchmarkRunManifest,
) -> AggregateReport:
    records_tuple = tuple(records)
    _validate_raw_matrix(records_tuple, manifest=manifest)
    raw_sha = raw_records_sha256(records_tuple)
    best_fixed = select_best_fixed(records_tuple, manifest=manifest, raw_sha256=raw_sha)
    summaries = _build_summaries(records_tuple, best_fixed=best_fixed)
    comparisons = _build_paired_comparisons(records_tuple, best_fixed=best_fixed)
    return AggregateReport(
        run_id=manifest.run_id,
        raw_record_count=len(records_tuple),
        raw_logical_sha256=raw_sha,
        protocol_config_sha256=manifest.protocol_config_sha256,
        environment_manifest_sha256=manifest.environment_manifest_sha256,
        benchmark_manifest_sha256=manifest.benchmark_manifest_sha256,
        split_hash=manifest.split_hash,
        qrels_sha256=manifest.qrels_sha256,
        query_ids_sha256=manifest.query_ids_sha256,
        query_identities_sha256=manifest.query_identities_sha256,
        corpus_version=manifest.corpus_version,
        corpus_chunk_count=manifest.corpus_chunk_count,
        corpus_chunk_ids_sha256=manifest.corpus_chunk_ids_sha256,
        embedding_model_revision=manifest.embedding_model_revision,
        extractor_version=manifest.extractor_version,
        plan_catalog_sha256=manifest.plan_catalog_sha256,
        planner_config_sha256=manifest.planner_config_sha256,
        query_feature_config_sha256=manifest.query_feature_config_sha256,
        graph_tier_policy_sha256=manifest.graph_tier_policy_sha256,
        rule_runtime_config_version=manifest.rule_runtime_config_version,
        stage2_artifact_set_sha256=manifest.stage2_artifact_set_sha256,
        runtime_semantics_version=manifest.runtime_semantics_version,
        best_fixed=best_fixed,
        summaries=summaries,
        paired_comparisons=comparisons,
    )


def select_best_fixed(
    records: Sequence[RawTrialRecord],
    *,
    manifest: BenchmarkRunManifest,
    raw_sha256: str,
) -> BestFixedSelection:
    validation = tuple(
        record
        for record in records
        if record.split is SplitName.VALIDATION and record.trial_phase is TrialPhase.MEASURED
    )
    validation_ids = tuple(sorted({record.query_id for record in validation}))
    if not validation_ids:
        raise ValueError("BestFixed selection requires validation records")
    depth = {
        BenchmarkMethod.FIXED_P4: 1,
        BenchmarkMethod.FIXED_P5: 1,
        BenchmarkMethod.FIXED_P6: 2,
        BenchmarkMethod.FIXED_P8: 3,
    }
    plan_id = {
        BenchmarkMethod.FIXED_P4: "P4",
        BenchmarkMethod.FIXED_P5: "P5",
        BenchmarkMethod.FIXED_P6: "P6",
        BenchmarkMethod.FIXED_P8: "P8",
    }
    selected: list[BestFixedBudget] = []
    for budget in manifest.latency_budgets_ms:
        candidates: list[tuple[BenchmarkMethod, float, float, bool]] = []
        for method in FIXED_METHODS:
            subset = tuple(
                record
                for record in validation
                if record.method is method and record.latency_budget_ms == budget
            )
            if not subset:
                raise ValueError("validation matrix is missing a fixed baseline")
            recall = _query_cluster_mean(subset, "recall_at_10")
            p95 = percentile_type7([record.total_latency_ms for record in subset], 0.95)
            candidates.append((method, recall, p95, p95 <= budget))
        feasible = tuple(item for item in candidates if item[3])
        if feasible:
            winner = min(
                feasible,
                key=lambda item: (-item[1], item[2], depth[item[0]], plan_id[item[0]]),
            )
        else:
            winner = min(
                candidates,
                key=lambda item: (item[2], -item[1], depth[item[0]], plan_id[item[0]]),
            )
        selected.append(
            BestFixedBudget(
                latency_budget_ms=budget,
                method=winner[0],
                plan_id=plan_id[winner[0]],  # type: ignore[arg-type]
                feasible=winner[3],
                validation_recall_at_10=winner[1],
                validation_p95_latency_ms=winner[2],
            )
        )
    return BestFixedSelection(
        run_id=manifest.run_id,
        split_hash=manifest.split_hash,
        source_raw_sha256=raw_sha256,
        validation_query_ids_sha256=canonical_sha256(list(validation_ids)),
        selections=tuple(selected),
    )


def require_locked_best_fixed_for_test(
    selection: BestFixedSelection,
    requested: Mapping[int, str],
) -> None:
    """Prevent a held-out test command from changing validation-selected fixed plans."""

    locked = {item.latency_budget_ms: item.plan_id for item in selection.selections}
    if dict(requested) != locked:
        raise ValueError("test BestFixed plans must exactly match the validation lock")


def write_aggregate_artifacts(
    run_dir: Path,
    *,
    records: Sequence[RawTrialRecord],
    report: AggregateReport,
) -> None:
    ordered_records = tuple(sorted(records, key=lambda item: item.trial_id))
    write_json_model(run_dir / "aggregate.json", report)
    write_json_model(run_dir / "best_fixed_validation.json", report.best_fixed)
    write_bytes(run_dir / "raw_trials.csv", _raw_csv(ordered_records))
    write_bytes(run_dir / "aggregate.csv", _aggregate_csv(report.summaries))
    checksums = {
        name: _file_sha256(run_dir / name)
        for name in (
            "raw_trials.jsonl",
            "raw_trials.csv",
            "aggregate.json",
            "aggregate.csv",
            "best_fixed_validation.json",
            "environment.json",
            "protocol.yaml",
            "run_manifest.json",
        )
    }
    checksums["raw_logical_sha256"] = report.raw_logical_sha256
    write_json(run_dir / "checksums.json", checksums)


def _validate_raw_matrix(
    records: tuple[RawTrialRecord, ...],
    *,
    manifest: BenchmarkRunManifest,
) -> None:
    if len(records) != manifest.expected_raw_row_count:
        raise ValueError("raw benchmark matrix is incomplete")
    trial_ids = tuple(item.trial_id for item in records)
    if len(set(trial_ids)) != len(trial_ids):
        raise ValueError("raw benchmark matrix contains duplicate trial IDs")
    version_fields = (
        "run_id",
        "protocol_config_sha256",
        "environment_manifest_sha256",
        "benchmark_manifest_sha256",
        "split_hash",
        "qrels_sha256",
        "corpus_version",
        "corpus_chunk_count",
        "corpus_chunk_ids_sha256",
        "embedding_model_revision",
        "extractor_version",
        "plan_catalog_sha256",
        "planner_config_sha256",
        "query_feature_config_sha256",
        "graph_tier_policy_sha256",
        "rule_runtime_config_version",
        "stage2_artifact_set_sha256",
        "runtime_semantics_version",
    )
    expected = (
        manifest.run_id,
        manifest.protocol_config_sha256,
        manifest.environment_manifest_sha256,
        manifest.benchmark_manifest_sha256,
        manifest.split_hash,
        manifest.qrels_sha256,
        manifest.corpus_version,
        manifest.corpus_chunk_count,
        manifest.corpus_chunk_ids_sha256,
        manifest.embedding_model_revision,
        manifest.extractor_version,
        manifest.plan_catalog_sha256,
        manifest.planner_config_sha256,
        manifest.query_feature_config_sha256,
        manifest.graph_tier_policy_sha256,
        manifest.rule_runtime_config_version,
        manifest.stage2_artifact_set_sha256,
        manifest.runtime_semantics_version,
    )
    for record in records:
        if tuple(getattr(record, field) for field in version_fields) != expected:
            raise ValueError("raw benchmark versions do not match the run manifest")
        expected_trial_id = trial_identity(
            run_id=record.run_id,
            query_id=record.query_id,
            method=record.method,
            latency_budget_ms=record.latency_budget_ms,
            phase=record.trial_phase,
            repetition=record.repetition,
        )
        if record.trial_id != expected_trial_id:
            raise ValueError("raw benchmark trial identity is inconsistent")
        if record.method not in EXECUTED_METHODS:
            raise ValueError("raw benchmark contains a non-executed method")
        if record.latency_budget_ms not in manifest.latency_budgets_ms:
            raise ValueError("raw benchmark contains an unexpected latency budget")
        repetition_limit = {
            TrialPhase.COLD: manifest.cold_runs,
            TrialPhase.WARMUP: manifest.warmup_runs,
            TrialPhase.MEASURED: manifest.measured_runs,
        }[record.trial_phase]
        if record.repetition >= repetition_limit:
            raise ValueError("raw benchmark repetition is outside its protocol phase")
    query_ids = tuple(sorted({record.query_id for record in records}))
    if canonical_sha256(list(query_ids)) != manifest.query_ids_sha256:
        raise ValueError("raw benchmark query IDs do not match the run manifest")
    identities_by_query: dict[str, BenchmarkQueryIdentity] = {}
    for record in records:
        identity = BenchmarkQueryIdentity(
            query_id=record.query_id,
            split=record.split,
            source_dataset=record.source_dataset,
            query_tags=record.query_tags,
        )
        previous = identities_by_query.setdefault(record.query_id, identity)
        if previous != identity:
            raise ValueError("raw benchmark query identity changes between trials")
    if (
        benchmark_query_identities_sha256(tuple(identities_by_query.values()))
        != manifest.query_identities_sha256
    ):
        raise ValueError("raw benchmark query identities do not match the run manifest")
    counts: dict[tuple[str, BenchmarkMethod, int, TrialPhase], int] = defaultdict(int)
    for record in records:
        counts[(record.query_id, record.method, record.latency_budget_ms, record.trial_phase)] += 1
    expected_phase_counts = {
        TrialPhase.COLD: manifest.cold_runs,
        TrialPhase.WARMUP: manifest.warmup_runs,
        TrialPhase.MEASURED: manifest.measured_runs,
    }
    if any(counts[key] != expected_phase_counts[key[3]] for key in counts):
        raise ValueError("raw benchmark trial phase counts are inconsistent")
    expected_groups = (
        manifest.query_count
        * manifest.method_count
        * len(manifest.latency_budgets_ms)
        * len(expected_phase_counts)
    )
    if len(counts) != expected_groups:
        raise ValueError("raw benchmark matrix is missing query/method/budget phases")


def _build_summaries(
    records: tuple[RawTrialRecord, ...],
    *,
    best_fixed: BestFixedSelection,
) -> tuple[GroupAggregate, ...]:
    results: list[GroupAggregate] = []
    methods = tuple(sorted({record.method for record in records}, key=lambda item: item.value))
    budgets = tuple(sorted({record.latency_budget_ms for record in records}))
    for phase in (TrialPhase.COLD, TrialPhase.MEASURED):
        for budget in budgets:
            phase_budget = tuple(
                record
                for record in records
                if record.trial_phase is phase and record.latency_budget_ms == budget
            )
            selected = best_fixed.for_budget(budget).method
            for display_method, source_method in (
                *((method, method) for method in methods),
                (BenchmarkMethod.BEST_FIXED, selected),
            ):
                method_records = tuple(
                    record for record in phase_budget if record.method is source_method
                )
                dimensions: list[tuple[AggregateDimension, str, tuple[RawTrialRecord, ...]]] = [
                    (AggregateDimension.OVERALL, "all", method_records)
                ]
                sources = {record.source_dataset for record in method_records}
                for source in sorted(sources, key=lambda item: item.value):
                    dimensions.append(
                        (
                            AggregateDimension.DATASET_SOURCE,
                            source.value,
                            tuple(
                                record
                                for record in method_records
                                if record.source_dataset is source
                            ),
                        )
                    )
                for tag in sorted(
                    {tag for record in method_records for tag in record.query_tags},
                    key=lambda item: item.value,
                ):
                    dimensions.append(
                        (
                            AggregateDimension.QUERY_TYPE,
                            tag.value,
                            tuple(record for record in method_records if tag in record.query_tags),
                        )
                    )
                for dimension, group, subset in dimensions:
                    if not subset:
                        continue
                    ci = None
                    if phase is TrialPhase.MEASURED and dimension is AggregateDimension.OVERALL:
                        ci = clustered_bootstrap_mean_interval(
                            _query_means(subset, "recall_at_10"),
                            seed=BENCHMARK_RANDOM_SEED,
                        )
                    results.append(
                        _aggregate_subset(
                            subset,
                            phase=phase,
                            dimension=dimension,
                            group=group,
                            method=display_method,
                            selected_fixed_method=(
                                source_method
                                if display_method is BenchmarkMethod.BEST_FIXED
                                else None
                            ),
                            recall_ci=ci,
                        )
                    )
    return tuple(
        sorted(
            results,
            key=lambda item: (
                item.phase,
                item.latency_budget_ms,
                item.method.value,
                item.dimension.value,
                item.group,
            ),
        )
    )


def _aggregate_subset(
    records: tuple[RawTrialRecord, ...],
    *,
    phase: TrialPhase,
    dimension: AggregateDimension,
    group: str,
    method: BenchmarkMethod,
    selected_fixed_method: BenchmarkMethod | None,
    recall_ci: BootstrapInterval | None,
) -> GroupAggregate:
    count = len(records)
    latencies = [record.total_latency_ms for record in records]
    return GroupAggregate(
        phase="cold" if phase is TrialPhase.COLD else "measured",
        dimension=dimension,
        group=group,
        method=method,
        selected_fixed_method=selected_fixed_method,
        latency_budget_ms=records[0].latency_budget_ms,
        query_count=len({record.query_id for record in records}),
        trial_count=count,
        recall_at_5=_query_cluster_mean(records, "recall_at_5"),
        recall_at_10=_query_cluster_mean(records, "recall_at_10"),
        mrr_at_10=_query_cluster_mean(records, "mrr_at_10"),
        ndcg_at_10=_query_cluster_mean(records, "ndcg_at_10"),
        p50_latency_ms=percentile_type7(latencies, 0.50),
        p95_latency_ms=percentile_type7(latencies, 0.95),
        p99_latency_ms=percentile_type7(latencies, 0.99),
        timeout_rate=sum(record.timeout for record in records) / count,
        fallback_rate=sum(record.fallback for record in records) / count,
        error_rate=sum(record.status is TrialStatus.ERROR for record in records) / count,
        budget_violation_rate=sum(record.budget_violated for record in records) / count,
        recall_at_10_ci=recall_ci,
    )


def _build_paired_comparisons(
    records: tuple[RawTrialRecord, ...],
    *,
    best_fixed: BestFixedSelection,
) -> tuple[PairedComparison, ...]:
    comparisons: list[PairedComparison] = []
    measured = tuple(record for record in records if record.trial_phase is TrialPhase.MEASURED)
    for selection in best_fixed.selections:
        budget_records = tuple(
            record for record in measured if record.latency_budget_ms == selection.latency_budget_ms
        )
        right = _query_means(
            tuple(record for record in budget_records if record.method is selection.method),
            "recall_at_10",
        )
        for left_method in EXECUTED_METHODS:
            left = _query_means(
                tuple(record for record in budget_records if record.method is left_method),
                "recall_at_10",
            )
            comparisons.append(
                PairedComparison(
                    latency_budget_ms=selection.latency_budget_ms,
                    left_method=left_method,
                    right_selected_method=selection.method,
                    query_count=len(left),
                    recall_at_10_difference=paired_bootstrap_difference(left, right),
                )
            )
    return tuple(comparisons)


def _query_means(
    records: Sequence[RawTrialRecord],
    field: Literal["recall_at_5", "recall_at_10", "mrr_at_10", "ndcg_at_10"],
) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for record in records:
        grouped[record.query_id].append(float(getattr(record, field)))
    return {query_id: sum(values) / len(values) for query_id, values in sorted(grouped.items())}


def _query_cluster_mean(
    records: Sequence[RawTrialRecord],
    field: Literal["recall_at_5", "recall_at_10", "mrr_at_10", "ndcg_at_10"],
) -> float:
    means = _query_means(records, field)
    return sum(means.values()) / len(means)


def _raw_csv(records: Sequence[RawTrialRecord]) -> bytes:
    fields = tuple(RawTrialRecord.model_fields)
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for record in records:
        row = record.model_dump(mode="json")
        row["query_tags"] = "|".join(row["query_tags"])
        row["branch_results"] = canonical_json_bytes(row["branch_results"]).decode("utf-8")
        writer.writerow(row)
    return output.getvalue().encode("utf-8")


def _aggregate_csv(summaries: Sequence[GroupAggregate]) -> bytes:
    fields = tuple(GroupAggregate.model_fields)
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for summary in summaries:
        row = summary.model_dump(mode="json")
        row["recall_at_10_ci"] = canonical_json_bytes(row["recall_at_10_ci"]).decode("utf-8")
        writer.writerow(row)
    return output.getvalue().encode("utf-8")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
