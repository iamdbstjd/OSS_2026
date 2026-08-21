#!/usr/bin/env python3
"""Run the Stage 12 research-only cost policy on frozen validation evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ragplan.benchmark.aggregate import BestFixedSelection
from ragplan.benchmark.artifacts import write_json_model, write_jsonl_models
from ragplan.benchmark.oracle import OracleReport
from ragplan.benchmark.policy_evaluation import (
    Stage12EvidenceManifest,
    evaluate_offline_policy,
    file_sha256,
    load_stage9_comparison_metrics,
    load_stage10_guard_trials,
)
from ragplan.ingestion.audit import load_graph_tier_policy
from ragplan.planner.catalog import load_default_plan_catalog
from ragplan.planner.optimizer import (
    OfflineCostAwareOptimizer,
    load_historical_research_bundle,
)
from ragplan.planner.rule import RulePlanner, load_default_rule_planner_config
from ragplan.planner.training import load_training_matrix

DEFAULT_PROFILE_DIR = Path("benchmark/results/profile_plans_audit_bypass_20260820_r2")
DEFAULT_BASELINE_DIR = Path("benchmark/results/baseline_audit_bypass_20260820_r2")
DEFAULT_STAGE11_DIR = Path("artifacts/cost_models/stage11_r2")
DEFAULT_OUTPUT_DIR = Path("artifacts/cost_models/stage12_r2")
DEFAULT_EVIDENCE = Path("benchmark/manifests/stage12_policy_evidence_r2.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matrix",
        type=Path,
        default=DEFAULT_PROFILE_DIR / "training_matrix.jsonl",
    )
    parser.add_argument(
        "--stage10-raw",
        type=Path,
        default=DEFAULT_PROFILE_DIR / "raw_trials.jsonl",
    )
    parser.add_argument(
        "--oracle",
        type=Path,
        default=DEFAULT_PROFILE_DIR / "oracle_at_budget.json",
    )
    parser.add_argument(
        "--stage9-raw",
        type=Path,
        default=DEFAULT_BASELINE_DIR / "raw_trials.jsonl",
    )
    parser.add_argument(
        "--best-fixed",
        type=Path,
        default=DEFAULT_BASELINE_DIR / "best_fixed_validation.json",
    )
    parser.add_argument(
        "--quality-artifact",
        type=Path,
        default=DEFAULT_STAGE11_DIR / "quality_model.skops",
    )
    parser.add_argument(
        "--latency-artifact",
        type=Path,
        default=DEFAULT_STAGE11_DIR / "latency_model.skops",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--evidence-manifest", type=Path, default=DEFAULT_EVIDENCE)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    catalog = load_default_plan_catalog()
    bundle = load_historical_research_bundle(
        quality_artifact=args.quality_artifact,
        latency_artifact=args.latency_artifact,
        catalog=catalog,
    )
    optimizer = OfflineCostAwareOptimizer.from_bundle(catalog=catalog, bundle=bundle)
    matrix_sha256 = file_sha256(args.matrix)
    if matrix_sha256 != bundle.quality_manifest.training_matrix_sha256:
        raise ValueError("Stage 12 matrix does not match the Stage 11 research artifacts")
    matrix = load_training_matrix(args.matrix)
    oracle = OracleReport.model_validate_json(args.oracle.read_text(encoding="utf-8"))
    if oracle.source_training_matrix_sha256 != matrix_sha256:
        raise ValueError("Stage 12 Oracle does not match the training matrix")
    best_fixed = BestFixedSelection.model_validate_json(args.best_fixed.read_text(encoding="utf-8"))
    rule_metrics, best_fixed_metrics = load_stage9_comparison_metrics(
        args.stage9_raw,
        best_fixed=best_fixed,
    )
    guard_trials = load_stage10_guard_trials(args.stage10_raw)
    rule_planner = RulePlanner(
        catalog=catalog,
        graph_policy=load_graph_tier_policy(),
        config=load_default_rule_planner_config(),
    )
    records, report = evaluate_offline_policy(
        matrix,
        optimizer=optimizer,
        rule_planner=rule_planner,
        oracle=oracle,
        best_fixed=best_fixed,
        rule_metrics=rule_metrics,
        best_fixed_metrics=best_fixed_metrics,
        guard_trials=guard_trials,
        quality_manifest=bundle.quality_manifest,
        latency_manifest=bundle.latency_manifest,
        training_matrix_sha256=matrix_sha256,
        stage9_raw_sha256=file_sha256(args.stage9_raw),
        stage10_raw_sha256=file_sha256(args.stage10_raw),
        catalog_sha256=catalog.sha256(),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    decisions_path = args.output_dir / "decisions.jsonl"
    report_path = args.output_dir / "policy_report.json"
    write_jsonl_models(decisions_path, records)
    write_json_model(report_path, report)
    decisions_sha256 = file_sha256(decisions_path)
    if decisions_sha256 != report.decision_records_sha256:
        raise ValueError("Stage 12 decision artifact checksum does not match its report")
    evidence = Stage12EvidenceManifest(
        report_sha256=report.sha256,
        report_file_sha256=file_sha256(report_path),
        decision_records_file_sha256=decisions_sha256,
        decision_record_count=len(records),
        candidate_estimate_count=report.candidate_estimate_count,
        invalid_prediction_count=report.invalid_prediction_count,
        no_feasible_candidate_count=report.no_feasible_candidate_count,
        quality_artifact_sha256=bundle.quality_manifest.model.artifact_sha256,
        latency_artifact_sha256=bundle.latency_manifest.model.artifact_sha256,
        cost_recall_difference_vs_rule=report.cost_recall_difference_vs_rule,
        cost_violation_difference_vs_rule=report.cost_violation_difference_vs_rule,
        mean_cost_regret_vs_oracle=report.mean_cost_regret_vs_oracle,
        planner_overhead_p95_ms=report.cost_aware_overhead.p95_ms,
        runtime_guard_disabled=report.runtime_guard.final_snapshot.disabled,
        runtime_guard_disable_reason=report.runtime_guard.final_snapshot.disable_reason,
        runtime_guard_first_disabled_after_observation=(
            report.runtime_guard.first_disabled_after_observation
        ),
        runtime_guard_routed_to_rule_group_count=(report.runtime_guard.routed_to_rule_group_count),
        inherited_model_gate_failures=report.inherited_model_gate_failures,
    )
    write_json_model(args.evidence_manifest, evidence)
    print(
        json.dumps(
            {
                "status": report.status,
                "execution_mode": report.execution_mode,
                "public_api_cost_aware_enabled": report.public_api_cost_aware_enabled,
                "query_budget_count": report.query_budget_count,
                "candidate_estimate_count": report.candidate_estimate_count,
                "cost_recall_difference_vs_rule": report.cost_recall_difference_vs_rule,
                "mean_cost_regret_vs_oracle": report.mean_cost_regret_vs_oracle,
                "runtime_guard_disabled": report.runtime_guard.final_snapshot.disabled,
                "report_sha256": report.sha256,
            },
            separators=(",", ":"),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
