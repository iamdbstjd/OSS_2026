from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from ragplan.benchmark.aggregate import BestFixedBudget, BestFixedSelection
from ragplan.benchmark.contracts import QueryTag, SourceDataset, SplitName, canonical_sha256
from ragplan.benchmark.oracle import OracleDistribution, OracleReport, OracleSelection
from ragplan.benchmark.policy_evaluation import BaselineMetric, GuardTrial, evaluate_offline_policy
from ragplan.benchmark.profile_records import ProfilePlanFeatures, TrainingMatrixRow
from ragplan.benchmark.records import BenchmarkMethod
from ragplan.core.models import ModelManifest, QueryFeatures
from ragplan.ingestion.audit import AuditStatus, RuleGraphTierPolicy
from ragplan.planner.artifacts import ArtifactStatus, CostModelArtifactManifest, ModelKind
from ragplan.planner.catalog import PlanCatalog, load_default_plan_catalog
from ragplan.planner.optimizer import OfflineCostAwareOptimizer
from ragplan.planner.rule import RulePlanner, load_default_rule_planner_config
from ragplan.planner.training import COST_FEATURE_SCHEMA_VERSION, FEATURE_NAMES

pytestmark = [pytest.mark.benchmark, pytest.mark.unit]


class FixedPredictor:
    def __init__(self, values: tuple[float, ...]) -> None:
        self._values = np.asarray(values, dtype=np.float64)

    def predict(self, features: NDArray[np.float64]) -> object:
        return self._values.copy()


def _manifest(kind: ModelKind, catalog: PlanCatalog) -> CostModelArtifactManifest:
    name = kind.value
    return CostModelArtifactManifest(
        kind=kind,
        status=ArtifactStatus.RESEARCH_ONLY,
        model=ModelManifest(
            model_name=name,
            model_version=f"{name}-fixture-v1",
            artifact_version="cost_model_artifact_v1",
            artifact_sha256=("1" if kind is ModelKind.QUALITY else "2") * 64,
            feature_schema_version=COST_FEATURE_SCHEMA_VERSION,
            plan_catalog_hash=catalog.sha256(),
            corpus_version="fixture-corpus-v1",
            qrels_version="qrels-v1",
            embedding_model_revision="embedding-v1",
            extractor_version="extractor-v1",
            qdrant_version="1.18.2",
            neo4j_version="5.26.28",
            qdrant_client_version="1.18.0",
            training_config_hash="3" * 64,
            train_validation_split_hash="4" * 64,
            runtime_fingerprint="runtime-v1",
            runtime_semantics_version="v1",
            validation_metrics={"metric": 0.0},
        ),
        feature_names=FEATURE_NAMES,
        training_matrix_sha256="5" * 64,
        training_row_count=100,
        validation_row_count=20,
        hardware_fingerprint="6" * 64,
        dependency_versions={},
        gate_failures=("research_gate",),
    )


def _matrix(catalog: PlanCatalog) -> tuple[TrainingMatrixRow, ...]:
    rows = []
    for query_index in range(2):
        for plan_index, plan in enumerate(catalog.p0_enabled_plans):
            recall = 0.9 if plan.id == "P4" else 0.1 + plan_index * 0.01
            rows.append(
                TrainingMatrixRow(
                    run_id="stage12-fixture",
                    query_id=f"validation-query-{query_index}",
                    split=SplitName.VALIDATION,
                    source_dataset=SourceDataset.HOTPOT_BRIDGE,
                    query_tags=(QueryTag.RELATIONSHIP,),
                    plan_id=plan.id,
                    query_features=QueryFeatures(
                        token_count=8,
                        entity_count=2,
                        entity_density=0.25,
                        relation_signal=0.8,
                        multi_hop_signal=0.0,
                        comparison_signal=0.0,
                        aggregation_signal=0.0,
                        global_signal=0.0,
                        final_top_k=10,
                    ),
                    plan_features=ProfilePlanFeatures.from_plan(plan),
                    latency_budget_ms=100,
                    quality_label_trial_count=10,
                    execution_latency_trial_count=10,
                    complete_trial_count=10,
                    partial_trial_count=0,
                    timeout_trial_count=0,
                    error_trial_count=0,
                    fallback_trial_count=0,
                    recall_at_5=recall,
                    recall_at_10=recall,
                    mrr_at_10=recall,
                    ndcg_at_10=recall,
                    full_result_recall_at_10=recall,
                    p95_execution_latency_ms=20.0 + plan_index,
                    fallback_rate=0.0,
                    budget_violation_rate=0.0,
                    quality_label_valid=True,
                    execution_latency_label_valid=True,
                    usable_for_model_training=True,
                    source_trial_ids_sha256=canonical_sha256(
                        [f"q{query_index}:{plan.id}:{trial}" for trial in range(10)]
                    ),
                    profile_protocol_sha256="1" * 64,
                    environment_manifest_sha256="2" * 64,
                    benchmark_manifest_sha256="3" * 64,
                    split_hash="4" * 64,
                    qrels_sha256="5" * 64,
                    corpus_version="fixture-corpus-v1",
                    corpus_chunk_ids_sha256="6" * 64,
                    plan_catalog_sha256=catalog.sha256(),
                    query_feature_config_sha256="8" * 64,
                )
            )
    return tuple(rows)


def _oracle(rows: tuple[TrainingMatrixRow, ...], catalog: PlanCatalog) -> OracleReport:
    query_ids = sorted({row.query_id for row in rows})
    selections = tuple(
        OracleSelection(
            query_id=query_id,
            split=SplitName.VALIDATION,
            latency_budget_ms=100,
            oracle_plan_id="P4",
            oracle_recall_at_10=0.9,
            oracle_p95_execution_latency_ms=24.0,
            feasible_plan_count=8,
        )
        for query_id in query_ids
    )
    return OracleReport(
        run_id="stage12-fixture",
        source_training_matrix_sha256="5" * 64,
        profile_protocol_sha256="1" * 64,
        environment_manifest_sha256="2" * 64,
        benchmark_manifest_sha256="3" * 64,
        split_hash="4" * 64,
        qrels_sha256="5" * 64,
        corpus_version="fixture-corpus-v1",
        plan_catalog_sha256=catalog.sha256(),
        query_feature_config_sha256="8" * 64,
        selections=selections,
        distribution=(OracleDistribution(latency_budget_ms=100, plan_id="P4", query_count=2),),
    )


def test_offline_report_compares_every_policy_without_enabling_serving() -> None:
    catalog = load_default_plan_catalog()
    quality_manifest = _manifest(ModelKind.QUALITY, catalog)
    latency_manifest = _manifest(ModelKind.LATENCY_P95, catalog)
    optimizer = OfflineCostAwareOptimizer(
        catalog=catalog,
        quality_model=FixedPredictor((0.1, 0.2, 0.3, 0.4, 0.9, 0.6, 0.7, 0.8)),
        latency_model=FixedPredictor((10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0)),
        quality_manifest=quality_manifest,
        latency_manifest=latency_manifest,
    )
    rows = _matrix(catalog)
    query_ids = sorted({row.query_id for row in rows})
    keys = tuple((query_id, 100) for query_id in query_ids)
    metrics = {key: BaselineMetric(recall_at_10=0.5, budget_violation_rate=0.0) for key in keys}
    guard_trials = {
        (query_id, 100, "P4"): tuple(
            GuardTrial(actual_execution_latency_ms=45.0, budget_violated=False) for _ in range(10)
        )
        for query_id in query_ids
    }
    best_fixed = BestFixedSelection(
        run_id="stage12-fixture",
        split_hash="4" * 64,
        source_raw_sha256="9" * 64,
        validation_query_ids_sha256="a" * 64,
        selections=(
            BestFixedBudget(
                latency_budget_ms=100,
                method=BenchmarkMethod.FIXED_P4,
                plan_id="P4",
                feasible=True,
                validation_recall_at_10=0.9,
                validation_p95_latency_ms=24.0,
            ),
        ),
    )
    policy = RuleGraphTierPolicy(
        audit_sample_checksum="b" * 64,
        audit_status=AuditStatus.PENDING_HUMAN_REVIEW,
        graph_tier_enabled=False,
        reason="human audit pending",
    )
    rule = RulePlanner(
        catalog=catalog,
        graph_policy=policy,
        config=load_default_rule_planner_config(),
    )

    records, report = evaluate_offline_policy(
        rows,
        optimizer=optimizer,
        rule_planner=rule,
        oracle=_oracle(rows, catalog),
        best_fixed=best_fixed,
        rule_metrics=metrics,
        best_fixed_metrics=metrics,
        guard_trials=guard_trials,
        quality_manifest=quality_manifest,
        latency_manifest=latency_manifest,
        training_matrix_sha256="5" * 64,
        stage9_raw_sha256="9" * 64,
        stage10_raw_sha256="a" * 64,
        catalog_sha256=catalog.sha256(),
    )

    assert len(records) == 2
    assert report.status == "research_only"
    assert report.public_api_cost_aware_enabled is False
    assert report.test_split_used is False
    assert report.candidate_estimate_count == 16
    assert report.cost_aware.mean_recall_at_10 == 0.9
    assert report.mean_cost_regret_vs_oracle == 0.0
    assert report.runtime_guard.evaluable_trial_count == 20
    assert report.runtime_guard.final_snapshot.disabled is False
    assert all(record.cost_aware_decision.selected_plan_id == "P4" for record in records)
    assert all(len(record.cost_aware_decision.candidate_estimates) == 8 for record in records)
    serialized = records[0].model_dump_json()
    assert "offline-redacted-validation-query" not in serialized
    assert "query_embedding" not in serialized
    assert sum(item.count for item in report.disagreement_matrix) == 2
