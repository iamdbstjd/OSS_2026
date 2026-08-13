"""Fail-closed Stage 9 CLI orchestration over the production Stage 6 engine."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from ragplan.api.runtime import Stage6RuntimeConfig, build_baseline_search_engine
from ragplan.backends.base import canonical_id_checksum
from ragplan.benchmark.aggregate import aggregate_run, write_aggregate_artifacts
from ragplan.benchmark.artifacts import load_chunk_index, write_json_model
from ragplan.benchmark.config import (
    capture_environment_manifest,
    create_run_manifest,
    file_sha256,
    load_benchmark_protocol,
    load_environment_manifest,
    verify_environment_manifest,
)
from ragplan.benchmark.contracts import canonical_sha256
from ragplan.benchmark.records import (
    BenchmarkProtocolConfig,
    BenchmarkQueryIdentity,
    BenchmarkRunManifest,
)
from ragplan.benchmark.runner import (
    BenchmarkRunner,
    ManagedSearchEngineTrialExecutor,
    RunnerSummary,
    RunRepository,
    load_stage9_cases,
)
from ragplan.core.engine import SearchEngine
from ragplan.core.errors import RAGPlanError
from ragplan.core.models import GraphStageManifest, VectorStageManifest
from ragplan.ingestion.audit import load_graph_tier_policy
from ragplan.ingestion.manifest import ManifestRepository, load_contract_json
from ragplan.planner.catalog import load_default_plan_catalog
from ragplan.planner.features import load_default_query_feature_config
from ragplan.planner.rule import RulePlanner, load_default_rule_planner_config

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_ROOT = REPOSITORY_ROOT / "benchmark" / "results"
DEFAULT_MANIFEST = REPOSITORY_ROOT / "benchmark/manifests/adaptive_rag_bench_v1.yaml"
DEFAULT_SPLITS = REPOSITORY_ROOT / "benchmark/configs/splits_v1.json"
DEFAULT_QRELS = REPOSITORY_ROOT / "benchmark/qrels/qrels_v1.jsonl"
DEFAULT_CHUNK_INDEX = REPOSITORY_ROOT / "benchmark/manifests/chunk_index_v1.jsonl"
DEFAULT_ARTIFACT_SET = REPOSITORY_ROOT / "benchmark/manifests/artifact_set_v1.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run or aggregate the frozen Stage 9 baseline.")
    subcommands = parser.add_subparsers(dest="command", required=True)

    capture = subcommands.add_parser("capture-environment")
    capture.add_argument("--output", type=Path, required=True)
    capture.add_argument("--container-resource-limits", required=True)
    capture.add_argument("--confirm-dedicated", action="store_true")

    run = subcommands.add_parser("run")
    run.add_argument("--run-id", required=True)
    run.add_argument("--environment-manifest", type=Path, required=True)
    run.add_argument("--config", type=Path)
    run.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    run.add_argument("--benchmark-manifest", type=Path, default=DEFAULT_MANIFEST)
    run.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    run.add_argument("--qrels", type=Path, default=DEFAULT_QRELS)
    run.add_argument("--chunk-index", type=Path, default=DEFAULT_CHUNK_INDEX)
    run.add_argument("--confirm-dedicated", action="store_true")

    aggregate = subcommands.add_parser("aggregate")
    aggregate.add_argument("--run-id", required=True)
    aggregate.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser


def run(argv: Sequence[str] | None = None, *, environment: Mapping[str, str] | None = None) -> int:
    try:
        args = build_parser().parse_args(argv)
        if args.command == "capture-environment":
            return _capture(args)
        if args.command == "aggregate":
            return _aggregate(args)
        return asyncio.run(_execute(args, environment=environment))
    except RAGPlanError as exc:
        print(
            _json_line({"error": exc.code.value, "message": exc.message}),
            file=sys.stderr,
        )
        return 1
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(_json_line({"error": type(exc).__name__, "message": str(exc)}), file=sys.stderr)
        return 1


def _capture(args: argparse.Namespace) -> int:
    if not args.confirm_dedicated:
        raise ValueError("--confirm-dedicated is required for primary environment capture")
    manifest = capture_environment_manifest(
        REPOSITORY_ROOT,
        container_resource_limits=args.container_resource_limits,
    )
    write_json_model(args.output, manifest)
    print(_json_line({"environment_manifest": str(args.output), "sha256": manifest.sha256}))
    return 0


async def _execute(
    args: argparse.Namespace,
    *,
    environment: Mapping[str, str] | None,
) -> int:
    if not args.confirm_dedicated:
        raise ValueError("--confirm-dedicated is required for a primary benchmark run")
    source_environment = os.environ if environment is None else environment
    if _truthy(source_environment.get("RAGPLAN_FORCE_VECTOR_ONLY", "")):
        raise ValueError("RAGPLAN_FORCE_VECTOR_ONLY must be disabled for Stage 9")
    protocol = load_benchmark_protocol(args.config)
    _verify_static_protocol(protocol)
    benchmark_environment = load_environment_manifest(args.environment_manifest)
    verify_environment_manifest(REPOSITORY_ROOT, benchmark_environment)
    cases = load_stage9_cases(
        manifest_path=args.benchmark_manifest,
        split_path=args.splits,
        qrels_path=args.qrels,
        protocol=protocol,
    )
    _verify_corpus_artifact(args.chunk_index, protocol=protocol)
    runtime = Stage6RuntimeConfig.from_environment(source_environment)
    if runtime is None:
        raise ValueError("complete RAGPLAN_STAGE6_* runtime configuration is required")
    _verify_runtime_evidence(runtime, protocol=protocol)

    existing_manifest_path = args.output_root / args.run_id / "run_manifest.json"
    created_at = None
    if existing_manifest_path.is_file():
        existing = BenchmarkRunManifest.model_validate_json(
            existing_manifest_path.read_text(encoding="utf-8")
        )
        created_at = existing.created_at_utc
    manifest = create_run_manifest(
        run_id=args.run_id,
        protocol=protocol,
        environment=benchmark_environment,
        query_identities=tuple(
            BenchmarkQueryIdentity(
                query_id=case.query.query_id,
                split=case.split,
                source_dataset=case.query.source_dataset,
                query_tags=case.query.query_tags,
            )
            for case in cases
        ),
        created_at_utc=created_at,
    )
    repository = RunRepository(
        args.output_root,
        protocol=protocol,
        environment=benchmark_environment,
        manifest=manifest,
    )

    async def engine_factory() -> SearchEngine:
        return await build_baseline_search_engine(runtime)

    executor = ManagedSearchEngineTrialExecutor(engine_factory)
    try:
        summary = await BenchmarkRunner(
            protocol=protocol,
            run_manifest=manifest,
            cases=cases,
            executor=executor,
            repository=repository,
            progress=_report_progress,
        ).run()
    finally:
        await executor.close()
    if summary.complete:
        records = repository.load_records()
        report = aggregate_run(records, manifest=manifest)
        write_aggregate_artifacts(repository.run_dir, records=records, report=report)
    print(_json_line(_summary_payload(summary)))
    return 0 if summary.complete else 2


def _aggregate(args: argparse.Namespace) -> int:
    run_dir = args.output_root / args.run_id
    manifest = BenchmarkRunManifest.model_validate_json(
        (run_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    protocol = load_benchmark_protocol(run_dir / "protocol.yaml")
    environment = load_environment_manifest(run_dir / "environment.json")
    repository = RunRepository(
        args.output_root,
        protocol=protocol,
        environment=environment,
        manifest=manifest,
    )
    with repository.exclusive_lock():
        records = repository.load_records()
        report = aggregate_run(records, manifest=manifest)
        write_aggregate_artifacts(run_dir, records=records, report=report)
    print(
        _json_line(
            {
                "run_id": args.run_id,
                "raw_record_count": len(records),
                "raw_logical_sha256": report.raw_logical_sha256,
            }
        )
    )
    return 0


def _verify_static_protocol(protocol: BenchmarkProtocolConfig) -> None:
    catalog = load_default_plan_catalog()
    if catalog.sha256() != protocol.plan_catalog_sha256:
        raise ValueError("plan catalog does not match the Stage 9 protocol")
    rule_config = load_default_rule_planner_config()
    if rule_config.sha256 != protocol.planner_config_sha256:
        raise ValueError("rule planner config does not match the Stage 9 protocol")
    features = load_default_query_feature_config()
    if features.sha256 != protocol.query_feature_config_sha256:
        raise ValueError("query feature config does not match the Stage 9 protocol")
    graph_policy = load_graph_tier_policy()
    policy_hash = canonical_sha256(graph_policy.model_dump(mode="json"))
    if policy_hash != protocol.graph_tier_policy_sha256:
        raise ValueError("graph tier policy does not match the Stage 9 protocol")
    rule_runtime = RulePlanner(
        catalog=catalog,
        graph_policy=graph_policy,
        config=rule_config,
        feature_config_sha256=features.sha256,
    )
    if rule_runtime.config_version != protocol.rule_runtime_config_version:
        raise ValueError("rule runtime identity does not match the Stage 9 protocol")
    if file_sha256(DEFAULT_ARTIFACT_SET) != protocol.stage2_artifact_set_sha256:
        raise ValueError("Stage 2 artifact set does not match the Stage 9 protocol")


def _verify_corpus_artifact(path: Path, *, protocol: BenchmarkProtocolConfig) -> None:
    chunks = load_chunk_index(path)
    if len(chunks) != protocol.corpus_chunk_count:
        raise ValueError("corpus chunk count does not match the Stage 9 protocol")
    observed_checksum = canonical_id_checksum(tuple(chunk.canonical_chunk_id for chunk in chunks))
    if observed_checksum != protocol.corpus_chunk_ids_sha256:
        raise ValueError("corpus canonical IDs do not match the Stage 9 protocol")


def _verify_runtime_evidence(
    runtime: Stage6RuntimeConfig,
    *,
    protocol: BenchmarkProtocolConfig,
) -> None:
    vector = load_contract_json(runtime.vector_stage_manifest, VectorStageManifest)
    graph = load_contract_json(runtime.graph_stage_manifest, GraphStageManifest)
    _, active = ManifestRepository(runtime.manifest_root).load_active()
    if any(
        item != protocol.corpus_version
        for item in (vector.corpus_version, graph.corpus_version, active.corpus_version)
    ):
        raise ValueError("active runtime corpus does not match the Stage 9 protocol")
    if vector.embedding_model_revision != protocol.embedding_model_revision:
        raise ValueError("runtime embedding revision does not match the Stage 9 protocol")
    if graph.extractor_version != protocol.extractor_version:
        raise ValueError("runtime extractor does not match the Stage 9 protocol")
    expected = (protocol.corpus_chunk_count, protocol.corpus_chunk_ids_sha256)
    observed = (
        (vector.chunk_count, vector.canonical_id_checksum),
        (graph.chunk_count, graph.canonical_id_checksum),
        (active.chunk_count, active.qdrant_id_checksum),
        (active.chunk_count, active.neo4j_id_checksum),
    )
    if any(item != expected for item in observed):
        raise ValueError("active backend evidence does not match the frozen corpus")


def _truthy(value: str) -> bool:
    return value.strip().casefold() in {"1", "true", "yes", "on"}


def _summary_payload(summary: RunnerSummary) -> dict[str, object]:
    return {
        "run_id": summary.run_id,
        "expected_rows": summary.expected_rows,
        "preexisting_rows": summary.preexisting_rows,
        "executed_rows": summary.executed_rows,
        "total_rows": summary.total_rows,
        "complete": summary.complete,
        "raw_path": str(summary.raw_path),
    }


def _json_line(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _report_progress(completed: int, expected: int) -> None:
    print(
        _json_line(
            {
                "event": "benchmark_progress",
                "completed_rows": completed,
                "expected_rows": expected,
            }
        ),
        file=sys.stderr,
        flush=True,
    )


__all__ = ["build_parser", "run"]
