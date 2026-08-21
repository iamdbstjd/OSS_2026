"""Fail-closed Stage 10 profiler CLI over the production dual-store engine."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

from ragplan.api.runtime import Stage6RuntimeConfig, build_baseline_search_engine
from ragplan.benchmark.artifacts import write_json
from ragplan.benchmark.command import (
    DEFAULT_CHUNK_INDEX,
    DEFAULT_MANIFEST,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_QRELS,
    DEFAULT_SPLITS,
    REPOSITORY_ROOT,
    _truthy,
    _verify_corpus_artifact,
    _verify_runtime_evidence,
    _verify_static_protocol,
)
from ragplan.benchmark.config import (
    load_benchmark_protocol,
    load_environment_manifest,
    runtime_source_sha256,
    verify_environment_manifest,
)
from ragplan.benchmark.oracle import build_oracle_report, write_oracle_artifact
from ragplan.benchmark.profile_records import (
    P0_PROFILE_PLAN_IDS,
    ProfileProtocolConfig,
    ProfileRunManifest,
    create_profile_protocol,
    create_profile_run_manifest,
)
from ragplan.benchmark.profiler import (
    ManagedSearchEngineProfileTrialExecutor,
    PlanProfiler,
    ProfileRepository,
    ProfileRunnerSummary,
    build_training_matrix,
    write_training_matrix_artifacts,
)
from ragplan.benchmark.records import BenchmarkQueryIdentity
from ragplan.benchmark.runner import BenchmarkCase, load_stage9_cases
from ragplan.core.deadline import PerfCounterClock
from ragplan.core.engine import SearchEngine
from ragplan.core.errors import RAGPlanError
from ragplan.core.models import QueryFeatures
from ragplan.ingestion.entities import EntityExtractor
from ragplan.planner.catalog import load_default_plan_catalog
from ragplan.planner.features import extract_query_features, load_default_query_feature_config
from ragplan.retrieval.graph import GraphQueryAnalyzer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run or rebuild the frozen Stage 10 query-by-plan profile."
    )
    subcommands = parser.add_subparsers(dest="command", required=True)
    run = subcommands.add_parser("run")
    run.add_argument("--run-id", required=True)
    run.add_argument("--environment-manifest", type=Path, required=True)
    run.add_argument("--baseline-config", type=Path)
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


async def _execute(
    args: argparse.Namespace,
    *,
    environment: Mapping[str, str] | None,
) -> int:
    if not args.confirm_dedicated:
        raise ValueError("--confirm-dedicated is required for a primary profiler run")
    source_environment = os.environ if environment is None else environment
    if _truthy(source_environment.get("RAGPLAN_FORCE_VECTOR_ONLY", "")):
        raise ValueError("RAGPLAN_FORCE_VECTOR_ONLY must be disabled for Stage 10")
    baseline = load_benchmark_protocol(args.baseline_config)
    _verify_static_protocol(baseline)
    catalog = load_default_plan_catalog()
    protocol = create_profile_protocol(baseline, catalog)
    if protocol.plan_ids != P0_PROFILE_PLAN_IDS:
        raise ValueError("Stage 10 production profiler requires exactly P0/P1/P2/P3/P4/P5/P6/P8")
    profile_environment = load_environment_manifest(args.environment_manifest)
    verify_environment_manifest(REPOSITORY_ROOT, profile_environment)
    cases = load_stage9_cases(
        manifest_path=args.benchmark_manifest,
        split_path=args.splits,
        qrels_path=args.qrels,
        protocol=baseline,
    )
    _verify_corpus_artifact(args.chunk_index, protocol=baseline)
    runtime = Stage6RuntimeConfig.from_environment(source_environment)
    if runtime is None:
        raise ValueError("complete RAGPLAN_STAGE6_* runtime configuration is required")
    _verify_runtime_evidence(runtime, protocol=baseline)

    run_dir = args.output_root / f"profile_{args.run_id}"
    existing_manifest_path = run_dir / "run_manifest.json"
    created_at = datetime.now(UTC).isoformat(timespec="seconds")
    if existing_manifest_path.is_file():
        existing = ProfileRunManifest.model_validate_json(
            existing_manifest_path.read_text(encoding="utf-8")
        )
        created_at = existing.created_at_utc
    manifest = create_profile_run_manifest(
        run_id=args.run_id,
        protocol=protocol,
        environment=profile_environment,
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
    repository = ProfileRepository(
        args.output_root,
        protocol=protocol,
        environment=profile_environment,
        manifest=manifest,
    )

    async def engine_factory() -> SearchEngine:
        return await build_baseline_search_engine(runtime, plan_catalog=catalog)

    executor = ManagedSearchEngineProfileTrialExecutor(engine_factory)
    try:
        summary = await PlanProfiler(
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
        _write_analysis(
            repository,
            manifest=manifest,
            cases=cases,
            raw_runtime_source_sha256=profile_environment.runtime_source_sha256,
        )
    print(_json_line(_summary_payload(summary)))
    return 0 if summary.complete else 2


def _aggregate(args: argparse.Namespace) -> int:
    run_dir = args.output_root / f"profile_{args.run_id}"
    protocol = ProfileProtocolConfig.model_validate_json(
        (run_dir / "profile_protocol.json").read_text(encoding="utf-8")
    )
    environment = load_environment_manifest(run_dir / "environment.json")
    manifest = ProfileRunManifest.model_validate_json(
        (run_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    repository = ProfileRepository(
        args.output_root,
        protocol=protocol,
        environment=environment,
        manifest=manifest,
    )
    baseline = load_benchmark_protocol()
    if baseline.sha256 != protocol.baseline_protocol_sha256:
        raise ValueError("installed baseline protocol differs from the profile run")
    cases = load_stage9_cases(
        manifest_path=DEFAULT_MANIFEST,
        split_path=DEFAULT_SPLITS,
        qrels_path=DEFAULT_QRELS,
        protocol=baseline,
    )
    with repository.exclusive_lock():
        _write_analysis(
            repository,
            manifest=manifest,
            cases=cases,
            raw_runtime_source_sha256=environment.runtime_source_sha256,
        )
    print(
        _json_line(
            {
                "run_id": manifest.run_id,
                "profile_directory": str(repository.run_dir),
                "status": "aggregated",
            }
        )
    )
    return 0


def _write_analysis(
    repository: ProfileRepository,
    *,
    manifest: ProfileRunManifest,
    cases: Sequence[BenchmarkCase],
    raw_runtime_source_sha256: str,
) -> None:
    records = repository.load_records()
    fallback_features = _derive_query_features(cases, manifest=manifest)
    traced_query_ids = {record.query_id for record in records if record.query_features is not None}
    recovered_query_ids = tuple(sorted(set(fallback_features) - traced_query_ids))
    write_json(
        repository.run_dir / "query_feature_recovery.json",
        {
            "schema_version": "query_feature_recovery_v1",
            "reason": "all runtime trials lacked trace features for these query IDs",
            "method": "qf_v1 from frozen query text and pinned graph extractor",
            "query_ids": recovered_query_ids,
            "query_count": len(recovered_query_ids),
            "query_feature_config_sha256": manifest.query_feature_config_sha256,
            "extractor_version": manifest.extractor_version,
            "raw_runtime_source_sha256": raw_runtime_source_sha256,
            "derivation_runtime_source_sha256": runtime_source_sha256(REPOSITORY_ROOT),
        },
    )
    matrix = build_training_matrix(
        records,
        manifest=manifest,
        fallback_query_features=fallback_features,
    )
    write_training_matrix_artifacts(
        repository.run_dir,
        records=records,
        matrix=matrix,
        recovered_query_feature_count=len(recovered_query_ids),
    )
    oracle = build_oracle_report(matrix, manifest=manifest)
    write_oracle_artifact(repository.run_dir, oracle)


def _derive_query_features(
    cases: Sequence[BenchmarkCase],
    *,
    manifest: ProfileRunManifest,
) -> dict[str, QueryFeatures]:
    extractor = EntityExtractor.load_pinned(
        lockfile=REPOSITORY_ROOT / "uv.lock",
        benchmark_mode=True,
    )
    if extractor.extractor_version != manifest.extractor_version:
        raise ValueError("feature recovery extractor differs from the profile run")
    feature_config = load_default_query_feature_config()
    if feature_config.sha256 != manifest.query_feature_config_sha256:
        raise ValueError("feature recovery config differs from the profile run")
    analyzer = GraphQueryAnalyzer(extractor, clock=PerfCounterClock())
    recovered: dict[str, QueryFeatures] = {}
    for case in cases:
        query = case.query
        analysis = analyzer.analyze(query.question, final_top_k=10)
        recovered[query.query_id] = extract_query_features(
            analysis.normalized_query,
            token_count=analysis.token_count,
            entity_count=len(analysis.seed_entity_ids),
            final_top_k=10,
            config=feature_config,
        )
    return recovered


def _summary_payload(summary: ProfileRunnerSummary) -> dict[str, object]:
    return {
        "run_id": summary.run_id,
        "expected_rows": summary.expected_rows,
        "preexisting_rows": summary.preexisting_rows,
        "executed_rows": summary.executed_rows,
        "total_rows": summary.total_rows,
        "complete": summary.complete,
        "raw_path": str(summary.raw_path),
    }


def _report_progress(completed: int, expected: int) -> None:
    print(
        _json_line(
            {
                "event": "profiler_progress",
                "completed_rows": completed,
                "expected_rows": expected,
            }
        ),
        file=sys.stderr,
        flush=True,
    )


def _json_line(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


__all__ = ["build_parser", "run"]
