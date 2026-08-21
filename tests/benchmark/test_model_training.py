from __future__ import annotations

import math
from collections import Counter

import numpy as np
import pytest
from pydantic import ValidationError

from ragplan.benchmark.contracts import QueryTag, SourceDataset, SplitName, canonical_sha256
from ragplan.benchmark.model_report import build_cost_model_report
from ragplan.benchmark.oracle import OracleDistribution, OracleReport, OracleSelection
from ragplan.benchmark.profile_records import ProfilePlanFeatures, TrainingMatrixRow
from ragplan.core.models import QueryFeatures
from ragplan.planner.catalog import load_default_plan_catalog
from ragplan.planner.latency_model import predict_p95_latency, train_latency_model
from ragplan.planner.quality_model import predict_quality, train_quality_model
from ragplan.planner.training import (
    COST_FEATURE_SCHEMA_VERSION,
    FEATURE_NAMES,
    build_training_datasets,
)

pytestmark = [pytest.mark.benchmark, pytest.mark.unit]


def _rows() -> tuple[TrainingMatrixRow, ...]:
    catalog = load_default_plan_catalog()
    rows: list[TrainingMatrixRow] = []
    for query_index in range(40):
        split = SplitName.TRAIN if query_index < 30 else SplitName.VALIDATION
        signal = query_index / 39
        for budget in (100, 200):
            for plan_index, plan_id in enumerate(("P0", "P2", "P4")):
                plan = catalog.plan_for_id(plan_id)
                quality = min(1.0, 0.15 + signal * 0.5 + plan_index * 0.12)
                latency = 20.0 + plan_index * 35.0 + signal * 10.0
                trial_hash = canonical_sha256(
                    [f"q{query_index}:{plan_id}:{budget}:{trial}" for trial in range(10)]
                )
                rows.append(
                    TrainingMatrixRow(
                        run_id="stage11-fixture",
                        query_id=f"query-{query_index:03d}",
                        split=split,
                        source_dataset=(
                            SourceDataset.NQ
                            if query_index % 2 == 0
                            else SourceDataset.HOTPOT_BRIDGE
                        ),
                        query_tags=(
                            QueryTag.SEMANTIC if query_index % 2 == 0 else QueryTag.RELATIONSHIP,
                        ),
                        plan_id=plan_id,
                        query_features=QueryFeatures(
                            token_count=5 + query_index,
                            entity_count=query_index % 3,
                            entity_density=(query_index % 3) / (5 + query_index),
                            relation_signal=signal,
                            multi_hop_signal=signal / 2,
                            comparison_signal=0.0,
                            aggregation_signal=0.0,
                            global_signal=0.0,
                            final_top_k=10,
                        ),
                        plan_features=ProfilePlanFeatures.from_plan(plan),
                        latency_budget_ms=budget,
                        quality_label_trial_count=10,
                        execution_latency_trial_count=10,
                        complete_trial_count=10,
                        partial_trial_count=0,
                        timeout_trial_count=0,
                        error_trial_count=0,
                        fallback_trial_count=0,
                        recall_at_5=quality,
                        recall_at_10=quality,
                        mrr_at_10=quality,
                        ndcg_at_10=quality,
                        full_result_recall_at_10=quality,
                        p95_execution_latency_ms=latency,
                        fallback_rate=0.0,
                        budget_violation_rate=float(latency > budget),
                        quality_label_valid=True,
                        execution_latency_label_valid=True,
                        usable_for_model_training=True,
                        source_trial_ids_sha256=trial_hash,
                        profile_protocol_sha256="1" * 64,
                        environment_manifest_sha256="2" * 64,
                        benchmark_manifest_sha256="3" * 64,
                        split_hash="4" * 64,
                        qrels_sha256="5" * 64,
                        corpus_version="fixture-corpus-v1",
                        corpus_chunk_ids_sha256="6" * 64,
                        plan_catalog_sha256="7" * 64,
                        query_feature_config_sha256="8" * 64,
                    )
                )
    return tuple(rows)


def _oracle(rows: tuple[TrainingMatrixRow, ...]) -> OracleReport:
    validation = tuple(row for row in rows if row.split is SplitName.VALIDATION)
    grouped: dict[tuple[str, int], list[TrainingMatrixRow]] = {}
    for row in validation:
        grouped.setdefault((row.query_id, row.latency_budget_ms), []).append(row)
    selections = []
    for key in sorted(grouped):
        feasible = [
            row for row in grouped[key] if (row.p95_execution_latency_ms or math.inf) <= key[1]
        ]
        winner = max(feasible, key=lambda row: row.recall_at_10)
        selections.append(
            OracleSelection(
                query_id=key[0],
                split=SplitName.VALIDATION,
                latency_budget_ms=key[1],
                oracle_plan_id=winner.plan_id,
                oracle_recall_at_10=winner.recall_at_10,
                oracle_p95_execution_latency_ms=winner.p95_execution_latency_ms,
                feasible_plan_count=len(feasible),
            )
        )
    counts = Counter((item.latency_budget_ms, item.oracle_plan_id) for item in selections)
    return OracleReport(
        run_id="stage11-fixture",
        source_training_matrix_sha256="9" * 64,
        profile_protocol_sha256="1" * 64,
        environment_manifest_sha256="2" * 64,
        benchmark_manifest_sha256="3" * 64,
        split_hash="4" * 64,
        qrels_sha256="5" * 64,
        corpus_version="fixture-corpus-v1",
        plan_catalog_sha256="7" * 64,
        query_feature_config_sha256="8" * 64,
        selections=tuple(selections),
        distribution=tuple(
            OracleDistribution(latency_budget_ms=key[0], plan_id=key[1], query_count=value)
            for key, value in sorted(counts.items())
        ),
    )


def test_dataset_builder_is_leakage_safe_finite_and_embedding_free() -> None:
    rows = _rows()
    datasets = build_training_datasets(rows)

    assert datasets.quality_train.features.shape == (180, len(FEATURE_NAMES))
    assert datasets.quality_validation.features.shape == (60, len(FEATURE_NAMES))
    assert datasets.latency_train.features.shape == (180, len(FEATURE_NAMES))
    assert np.isfinite(datasets.quality_train.features).all()
    assert all("embedding" not in name for name in FEATURE_NAMES)
    assert {row.query_id for row in datasets.quality_train.rows}.isdisjoint(
        {row.query_id for row in datasets.quality_validation.rows}
    )

    target_query = rows[0].query_id
    bad_rows = []
    for row in rows:
        if row.query_id != target_query:
            bad_rows.append(row)
            continue
        bad_feature = row.query_features.model_construct(
            **{**row.query_features.__dict__, "entity_density": math.nan}
        )
        bad_values = {name: getattr(row, name) for name in TrainingMatrixRow.model_fields}
        bad_values["query_features"] = bad_feature
        bad_rows.append(TrainingMatrixRow.model_construct(**bad_values))
    with pytest.raises(ValueError, match="NaN|infinite"):
        build_training_datasets(bad_rows)

    with pytest.raises(ValidationError):
        rows[0].model_copy(update={"recall_at_10": math.nan})


def test_models_are_deterministic_and_report_is_finite() -> None:
    rows = _rows()
    datasets = build_training_datasets(rows)
    quality_a = train_quality_model(datasets.quality_train)
    quality_b = train_quality_model(datasets.quality_train)
    latency_a = train_latency_model(datasets.latency_train)
    latency_b = train_latency_model(datasets.latency_train)
    quality_prediction_a = predict_quality(quality_a, datasets.quality_validation.features)
    quality_prediction_b = predict_quality(quality_b, datasets.quality_validation.features)
    latency_prediction_a = predict_p95_latency(latency_a, datasets.latency_validation.features)
    latency_prediction_b = predict_p95_latency(latency_b, datasets.latency_validation.features)
    assert np.array_equal(quality_prediction_a, quality_prediction_b)
    assert np.array_equal(latency_prediction_a, latency_prediction_b)
    assert np.all((0.0 <= quality_prediction_a) & (quality_prediction_a <= 1.0))
    assert np.all(latency_prediction_a >= 0.0)

    rule = {
        (row.query_id, row.latency_budget_ms): (row.recall_at_10, row.budget_violation_rate)
        for row in rows
        if row.split is SplitName.VALIDATION and row.plan_id == "P0"
    }
    report = build_cost_model_report(
        datasets,
        quality_a,
        latency_a,
        oracle=_oracle(rows),
        rule_baseline=rule,
        training_matrix_sha256="9" * 64,
        feature_schema_version=COST_FEATURE_SCHEMA_VERSION,
    )
    assert report.quality.validation_rows == 60
    assert report.latency.validation_rows == 60
    assert math.isfinite(report.quality.mae)
    assert math.isfinite(report.latency.coverage_overall)
    assert report.policy.validation_query_budget_count == 20


def test_query_group_leakage_is_rejected() -> None:
    rows = list(_rows())
    target = next(index for index, row in enumerate(rows) if row.plan_id == "P2")
    rows[target] = rows[target].model_copy(update={"split": SplitName.VALIDATION})
    with pytest.raises(ValueError, match="multiple dataset splits"):
        build_training_datasets(rows)
