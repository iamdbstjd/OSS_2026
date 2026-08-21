#!/usr/bin/env python3
"""Train, validate, gate, and safely serialize the Stage 11 cost models."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from ragplan.benchmark.artifacts import write_json_model
from ragplan.benchmark.config import load_benchmark_protocol, load_environment_manifest
from ragplan.benchmark.model_report import build_cost_model_report, load_rule_baseline
from ragplan.benchmark.oracle import OracleReport
from ragplan.benchmark.profile_records import ProfileRunManifest
from ragplan.planner.artifacts import (
    ArtifactContext,
    CostModelBundleManifest,
    ModelKind,
    hardware_fingerprint,
    installed_dependency_versions,
    load_trusted_types,
    save_cost_model,
)
from ragplan.planner.latency_model import LatencyTrainingConfig, train_latency_model
from ragplan.planner.quality_model import QualityTrainingConfig, train_quality_model
from ragplan.planner.training import (
    COST_FEATURE_SCHEMA_VERSION,
    FEATURE_NAMES,
    build_training_datasets,
    load_training_matrix,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--profile-run-manifest", type=Path, required=True)
    parser.add_argument("--profile-environment", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--stage9-raw", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-config", type=Path)
    parser.add_argument("--trusted-types", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    baseline = load_benchmark_protocol(args.baseline_config)
    profile_manifest = ProfileRunManifest.model_validate_json(
        args.profile_run_manifest.read_text(encoding="utf-8")
    )
    environment = load_environment_manifest(args.profile_environment)
    if profile_manifest.environment_manifest_sha256 != environment.sha256:
        raise ValueError("profile manifest and environment identity differ")
    if profile_manifest.baseline_protocol_sha256 != baseline.sha256:
        raise ValueError("profile manifest and baseline protocol identity differ")
    matrix_sha256 = _file_sha256(args.matrix)
    oracle = OracleReport.model_validate_json(args.oracle.read_text(encoding="utf-8"))
    if oracle.source_training_matrix_sha256 != matrix_sha256:
        raise ValueError("Oracle labels do not identify the training matrix")
    rows = load_training_matrix(args.matrix)
    datasets = build_training_datasets(rows)
    quality_config = QualityTrainingConfig()
    latency_config = LatencyTrainingConfig()
    quality_model = train_quality_model(datasets.quality_train, config=quality_config)
    latency_model = train_latency_model(datasets.latency_train, config=latency_config)
    rule_baseline = load_rule_baseline(args.stage9_raw)
    report = build_cost_model_report(
        datasets,
        quality_model,
        latency_model,
        oracle=oracle,
        rule_baseline=rule_baseline,
        training_matrix_sha256=matrix_sha256,
        feature_schema_version=COST_FEATURE_SCHEMA_VERSION,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "model_report.json"
    write_json_model(report_path, report)
    dependencies = installed_dependency_versions()
    trusted_types = load_trusted_types(args.trusted_types)
    shared = {
        "feature_schema_version": COST_FEATURE_SCHEMA_VERSION,
        "plan_catalog_hash": profile_manifest.plan_catalog_sha256,
        "corpus_version": profile_manifest.corpus_version,
        "qrels_version": baseline.qrels_version,
        "embedding_model_revision": profile_manifest.embedding_model_revision,
        "extractor_version": profile_manifest.extractor_version,
        "qdrant_version": _image_version(environment.qdrant_image),
        "neo4j_version": _image_version(environment.neo4j_image),
        "qdrant_client_version": dependencies["qdrant-client"],
        "train_validation_split_hash": datasets.train_validation_split_hash,
        "runtime_fingerprint": environment.runtime_source_sha256,
        "runtime_semantics_version": profile_manifest.runtime_semantics_version,
        "hardware_fingerprint": hardware_fingerprint(environment),
        "dependency_versions": dependencies,
        "training_matrix_sha256": matrix_sha256,
    }
    quality_context = ArtifactContext(
        **shared,
        training_config_hash=quality_config.sha256,
    )
    latency_context = ArtifactContext(
        **shared,
        training_config_hash=latency_config.sha256,
    )
    quality_manifest = save_cost_model(
        quality_model,
        args.output_dir / "quality_model.skops",
        kind=ModelKind.QUALITY,
        context=quality_context,
        feature_names=FEATURE_NAMES,
        validation_metrics={
            "mae": report.quality.mae,
            "rmse": report.quality.rmse,
            "plan_pair_ranking_accuracy": report.quality.plan_pair_ranking_accuracy,
            "predicted_best_plan_regret": report.quality.predicted_best_plan_regret,
        },
        status=report.status,
        gate_failures=report.gate_failures,
        training_row_count=len(datasets.quality_train.rows),
        validation_row_count=len(datasets.quality_validation.rows),
        trusted_types=trusted_types,
    )
    latency_manifest = save_cost_model(
        latency_model,
        args.output_dir / "latency_model.skops",
        kind=ModelKind.LATENCY_P95,
        context=latency_context,
        feature_names=FEATURE_NAMES,
        validation_metrics={
            "mae_ms": report.latency.mae_ms,
            "rmse_ms": report.latency.rmse_ms,
            "coverage_overall": report.latency.coverage_overall,
            "severe_underprediction_rate": report.latency.severe_underprediction_rate,
            "pinball_improvement": report.latency.pinball_improvement,
        },
        status=report.status,
        gate_failures=report.gate_failures,
        training_row_count=len(datasets.latency_train.rows),
        validation_row_count=len(datasets.latency_validation.rows),
        trusted_types=trusted_types,
    )
    bundle = CostModelBundleManifest(
        status=report.status,
        quality_manifest_sha256=quality_manifest.sha256,
        latency_manifest_sha256=latency_manifest.sha256,
        model_report_sha256=_file_sha256(report_path),
        training_matrix_sha256=matrix_sha256,
        gate_failures=report.gate_failures,
    )
    write_json_model(args.output_dir / "bundle_manifest.json", bundle)
    print(
        json.dumps(
            {
                "status": report.status.value,
                "gate_failures": report.gate_failures,
                "output_dir": str(args.output_dir),
                "quality_mae": report.quality.mae,
                "latency_coverage": report.latency.coverage_overall,
                "policy_regret": report.policy.mean_policy_regret_vs_oracle,
            },
            separators=(",", ":"),
            sort_keys=True,
        )
    )


def _image_version(image: str) -> str:
    reference = image.split("@sha256:", 1)[0]
    if ":" not in reference:
        raise ValueError("database image has no explicit version tag")
    return reference.rsplit(":", 1)[1].removeprefix("v")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
