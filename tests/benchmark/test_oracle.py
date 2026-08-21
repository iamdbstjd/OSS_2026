from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from ragplan.benchmark.config import load_benchmark_protocol
from ragplan.benchmark.contracts import QueryTag, SourceDataset, SplitName, canonical_json_bytes
from ragplan.benchmark.oracle import build_oracle_report, choose_oracle_plan, write_oracle_artifact
from ragplan.benchmark.profile_records import (
    ProfilePlanFeatures,
    ProfileRunManifest,
    TrainingMatrixRow,
    create_profile_protocol,
    create_profile_run_manifest,
)
from ragplan.benchmark.records import BenchmarkQueryIdentity, EnvironmentManifest
from ragplan.core.models import QueryFeatures
from ragplan.planner.catalog import load_default_plan_catalog

pytestmark = [pytest.mark.benchmark, pytest.mark.unit]


def _environment() -> EnvironmentManifest:
    return EnvironmentManifest(
        captured_at_utc="2026-08-19T00:00:00+00:00",
        os_name="Linux",
        os_release="fixture",
        machine="x86_64",
        cpu_model="fixture CPU",
        logical_cpu_count=1,
        cpu_governor="performance",
        python_version="3.12.13",
        qdrant_image=f"qdrant@sha256:{'a' * 64}",
        neo4j_image=f"neo4j@sha256:{'b' * 64}",
        api_image="ragplan-api:fixture",
        container_resource_limits="cpu=1,memory=1GiB",
        runtime_source_sha256="0" * 64,
        dependency_lock_sha256="1" * 64,
        docker_compose_sha256="2" * 64,
        db_tuning_sha256="3" * 64,
        notes="Oracle fixture",
    )


def _manifest(plan_ids: tuple[str, ...]) -> tuple[object, ProfileRunManifest]:
    baseline = load_benchmark_protocol()
    protocol = create_profile_protocol(
        baseline,
        load_default_plan_catalog(),
        plan_ids=plan_ids,
        latency_budgets_ms=(100,),
    )
    identity = BenchmarkQueryIdentity(
        query_id="adaptive_rag_bench_v1:nq:oracle-fixture",
        split=SplitName.TRAIN,
        source_dataset=SourceDataset.NQ,
        query_tags=(QueryTag.SEMANTIC,),
    )
    manifest = create_profile_run_manifest(
        run_id="oracle-fixture",
        protocol=protocol,
        environment=_environment(),
        query_identities=(identity,),
        created_at_utc="2026-08-19T00:00:00+00:00",
    )
    return protocol, manifest


def _row(
    plan_id: str,
    *,
    recall: float,
    latency: float | None,
    manifest: ProfileRunManifest,
    usable: bool = True,
) -> TrainingMatrixRow:
    plan = load_default_plan_catalog().plan_for_id(plan_id)
    trial_ids = [hashlib.sha256(f"{plan_id}:{index}".encode()).hexdigest() for index in range(10)]
    return TrainingMatrixRow(
        run_id=manifest.run_id,
        query_id="adaptive_rag_bench_v1:nq:oracle-fixture",
        split=SplitName.TRAIN,
        source_dataset=SourceDataset.NQ,
        query_tags=(QueryTag.SEMANTIC,),
        plan_id=plan_id,
        query_features=QueryFeatures(
            token_count=3,
            entity_count=0,
            entity_density=0.0,
            relation_signal=0.0,
            multi_hop_signal=0.0,
            comparison_signal=0.0,
            aggregation_signal=0.0,
            global_signal=0.0,
            final_top_k=10,
        ),
        plan_features=ProfilePlanFeatures.from_plan(plan),
        latency_budget_ms=100,
        quality_label_trial_count=10,
        execution_latency_trial_count=10 if usable else 0,
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
        p95_execution_latency_ms=latency if usable else None,
        fallback_rate=0.0,
        budget_violation_rate=0.0,
        quality_label_valid=True,
        execution_latency_label_valid=usable,
        usable_for_model_training=usable,
        invalid_exclusion_reasons=() if usable else ("execution_trace_unavailable",),
        source_trial_ids_sha256=hashlib.sha256(canonical_json_bytes(trial_ids)).hexdigest(),
        profile_protocol_sha256=manifest.profile_protocol_sha256,
        environment_manifest_sha256=manifest.environment_manifest_sha256,
        benchmark_manifest_sha256=manifest.benchmark_manifest_sha256,
        split_hash=manifest.split_hash,
        qrels_sha256=manifest.qrels_sha256,
        corpus_version=manifest.corpus_version,
        corpus_chunk_ids_sha256=manifest.corpus_chunk_ids_sha256,
        plan_catalog_sha256=manifest.plan_catalog_sha256,
        query_feature_config_sha256=manifest.query_feature_config_sha256,
        runtime_semantics_version=manifest.runtime_semantics_version,
    )


def test_oracle_feasibility_and_every_tie_break_level() -> None:
    _, manifest = _manifest(("P0", "P1", "P2", "P3", "P4", "P5"))

    feasible = choose_oracle_plan(
        (
            _row("P0", recall=0.8, latency=20.0, manifest=manifest),
            _row("P1", recall=1.0, latency=101.0, manifest=manifest),
        )
    )
    assert feasible is not None and feasible.plan_id == "P0"

    higher_recall = choose_oracle_plan(
        (
            _row("P0", recall=0.8, latency=10.0, manifest=manifest),
            _row("P1", recall=0.9, latency=90.0, manifest=manifest),
        )
    )
    assert higher_recall is not None and higher_recall.plan_id == "P1"

    lower_latency = choose_oracle_plan(
        (
            _row("P0", recall=0.8, latency=20.0, manifest=manifest),
            _row("P1", recall=0.8, latency=10.0, manifest=manifest),
        )
    )
    assert lower_latency is not None and lower_latency.plan_id == "P1"

    lower_depth = choose_oracle_plan(
        (
            _row("P0", recall=0.8, latency=10.0, manifest=manifest),
            _row("P2", recall=0.8, latency=10.0, manifest=manifest),
        )
    )
    assert lower_depth is not None and lower_depth.plan_id == "P0"

    lower_plan_id = choose_oracle_plan(
        (
            _row("P4", recall=0.8, latency=10.0, manifest=manifest),
            _row("P5", recall=0.8, latency=10.0, manifest=manifest),
        )
    )
    assert lower_plan_id is not None and lower_plan_id.plan_id == "P4"


def test_oracle_report_records_distribution_and_no_feasible_queries(tmp_path: Path) -> None:
    _, manifest = _manifest(("P0", "P2", "P4"))
    matrix = (
        _row("P0", recall=0.8, latency=20.0, manifest=manifest),
        _row("P2", recall=0.9, latency=90.0, manifest=manifest),
        _row("P4", recall=1.0, latency=None, manifest=manifest, usable=False),
    )
    report = build_oracle_report(matrix, manifest=manifest)
    assert report.selections[0].oracle_plan_id == "P2"
    assert report.selections[0].feasible_plan_count == 2
    assert [(item.plan_id, item.query_count) for item in report.distribution] == [("P2", 1)]
    write_oracle_artifact(tmp_path, report)
    first = (tmp_path / "oracle_at_budget.json").read_bytes()
    write_oracle_artifact(tmp_path, report)
    assert (tmp_path / "oracle_at_budget.json").read_bytes() == first

    unusable = tuple(
        _row(plan_id, recall=1.0, latency=None, manifest=manifest, usable=False)
        for plan_id in ("P0", "P2", "P4")
    )
    empty = build_oracle_report(unusable, manifest=manifest)
    assert empty.selections[0].oracle_plan_id is None
    assert empty.selections[0].no_feasible_reason is not None
    assert empty.distribution[0].plan_id is None


def test_oracle_rejects_plan_corpus_and_runtime_identity_drift() -> None:
    _, manifest = _manifest(("P0",))
    row = _row("P0", recall=1.0, latency=10.0, manifest=manifest)
    for update in (
        {"plan_catalog_sha256": "f" * 64},
        {"corpus_version": "wrong-corpus"},
        {"environment_manifest_sha256": "e" * 64},
    ):
        with pytest.raises(ValueError, match="version/hash"):
            build_oracle_report((row.model_copy(update=update),), manifest=manifest)
