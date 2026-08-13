#!/usr/bin/env python3
"""Search one active corpus through the shared Stage 6 baseline engine."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from uuid import uuid4

from ragplan.api.runtime import Stage6RuntimeConfig, build_baseline_search_engine
from ragplan.backends.vector.qdrant import DEFAULT_COLLECTION_PREFIX
from ragplan.core.errors import ErrorCode, RAGPlanError
from ragplan.core.models import PlannerMode, SearchRequest, SearchResponse
from ragplan.planner.catalog import load_default_plan_catalog, load_plan_catalog


class _SafeArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        del message
        raise RAGPlanError(ErrorCode.INVALID_REQUEST, "invalid hybrid search arguments")


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected a positive integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = _SafeArgumentParser(
        description="Search one reconciled active corpus through the Stage 6 baseline engine."
    )
    parser.add_argument("--query", required=True, help="Query text; omitted from trace and logs.")
    parser.add_argument(
        "--mode",
        choices=("vector", "graph", "fixed_hybrid"),
        default="fixed_hybrid",
    )
    parser.add_argument("--plan-id", choices=("P4", "P5", "P6", "P8"))
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument("--vector-stage-manifest", type=Path, required=True)
    parser.add_argument("--graph-stage-manifest", type=Path, required=True)
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--extractor-lockfile", type=Path, default=Path("uv.lock"))
    parser.add_argument("--plan-catalog", type=Path)
    parser.add_argument("--qdrant-url", default="http://127.0.0.1:6333")
    parser.add_argument("--collection-prefix", default=DEFAULT_COLLECTION_PREFIX)
    parser.add_argument(
        "--neo4j-uri",
        default=os.environ.get("RAGPLAN_GRAPH__URI", "bolt://127.0.0.1:7687"),
    )
    parser.add_argument(
        "--neo4j-user",
        default=os.environ.get("RAGPLAN_GRAPH__USER", "neo4j"),
    )
    parser.add_argument("--neo4j-database", default="neo4j")
    parser.add_argument("--top-k", type=_positive_int, default=10)
    parser.add_argument("--latency-budget-ms", type=_positive_int, default=200)
    return parser


async def _execute(args: argparse.Namespace, *, request_id: str) -> SearchResponse:
    password = os.environ.get("RAGPLAN_GRAPH__PASSWORD", "")
    if not password:
        raise RAGPlanError(
            ErrorCode.INVALID_REQUEST,
            "RAGPLAN_GRAPH__PASSWORD is required",
            retryable=False,
        )
    catalog = (
        load_default_plan_catalog()
        if args.plan_catalog is None
        else load_plan_catalog(args.plan_catalog)
    )
    config = Stage6RuntimeConfig(
        model_snapshot=args.model_snapshot,
        vector_stage_manifest=args.vector_stage_manifest,
        graph_stage_manifest=args.graph_stage_manifest,
        manifest_root=args.manifest_root,
        extractor_lockfile=args.extractor_lockfile,
        qdrant_url=args.qdrant_url,
        collection_prefix=args.collection_prefix,
        neo4j_uri=args.neo4j_uri,
        neo4j_user=args.neo4j_user,
        neo4j_password=password,
        neo4j_database=args.neo4j_database,
    )
    engine = await build_baseline_search_engine(config, plan_catalog=catalog)
    try:
        request = SearchRequest(
            query=args.query,
            top_k=args.top_k,
            latency_budget_ms=args.latency_budget_ms,
            planner=PlannerMode(args.mode),
            plan_id=args.plan_id,
        )
        return await engine.search(request, request_id=request_id)
    finally:
        await engine.close()


def _json_line(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def run(argv: Sequence[str] | None = None) -> int:
    request_id = f"baseline-search-{uuid4()}"
    try:
        args = build_parser().parse_args(argv)
        response = asyncio.run(_execute(args, request_id=request_id))
    except RAGPlanError as exc:
        print(_json_line(exc.response(request_id).model_dump(mode="json")), file=sys.stderr)
        return 1
    except (OSError, TypeError, ValueError):
        error = RAGPlanError(
            ErrorCode.INVALID_REQUEST,
            "hybrid search input is invalid",
            retryable=False,
        )
        print(_json_line(error.response(request_id).model_dump(mode="json")), file=sys.stderr)
        return 1
    print(_json_line(response.model_dump(mode="json")))
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
