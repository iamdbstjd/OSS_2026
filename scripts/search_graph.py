#!/usr/bin/env python3
"""Run one explicit graph-only query against an activated Stage 5 corpus."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from uuid import uuid4

from ragplan.backends.graph.neo4j import Neo4jGraphBackend, Neo4jGraphConfig
from ragplan.core.deadline import PerfCounterClock
from ragplan.core.engine import GraphSearchEngine
from ragplan.core.errors import ErrorCode, RAGPlanError
from ragplan.core.models import GraphStageManifest, PlannerMode, SearchRequest, SearchResponse
from ragplan.ingestion.entities import EntityExtractor
from ragplan.ingestion.manifest import ManifestRepository, load_contract_json
from ragplan.planner.catalog import load_default_plan_catalog, load_plan_catalog
from ragplan.retrieval.graph import GraphQueryAnalyzer


class _SafeArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        del message
        raise RAGPlanError(ErrorCode.INVALID_REQUEST, "invalid graph search arguments")


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
        description="Search one reconciled active corpus through bounded graph retrieval."
    )
    parser.add_argument("--query", required=True, help="Query text; omitted from logs and trace.")
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--graph-stage-manifest", type=Path, required=True)
    parser.add_argument("--extractor-lockfile", type=Path, default=Path("uv.lock"))
    parser.add_argument("--plan-catalog", type=Path)
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
    repository = ManifestRepository(args.manifest_root)
    _, active = repository.load_active()
    graph_stage = load_contract_json(args.graph_stage_manifest, GraphStageManifest)
    if (
        graph_stage.corpus_version != active.corpus_version
        or graph_stage.chunk_count != active.neo4j_count
        or graph_stage.canonical_id_checksum != active.neo4j_id_checksum
        or graph_stage.extractor_version != active.extractor_version
        or graph_stage.database != args.neo4j_database
    ):
        raise RAGPlanError(
            ErrorCode.CORPUS_INCONSISTENT,
            "active corpus and graph stage evidence do not match",
            retryable=False,
        )
    extractor = EntityExtractor.load_pinned(lockfile=args.extractor_lockfile)
    if extractor.extractor_version != active.extractor_version:
        raise RAGPlanError(
            ErrorCode.MODEL_INCOMPATIBLE,
            "runtime graph extractor does not match the active corpus",
            retryable=False,
        )
    backend = Neo4jGraphBackend.connect(
        Neo4jGraphConfig(
            uri=args.neo4j_uri,
            user=args.neo4j_user,
            password=password,
            database=args.neo4j_database,
        )
    )
    try:
        await backend.require_active_corpus(
            corpus_version=active.corpus_version,
            chunk_count=active.neo4j_count,
            canonical_id_checksum=active.neo4j_id_checksum,
            extractor_version=active.extractor_version,
        )
        clock = PerfCounterClock()
        catalog = (
            load_default_plan_catalog()
            if args.plan_catalog is None
            else load_plan_catalog(args.plan_catalog)
        )
        engine = GraphSearchEngine(
            analyzer=GraphQueryAnalyzer(extractor, clock=clock),
            graph_backend=backend,
            plan_catalog=catalog,
            active_manifest=active,
            clock=clock,
        )
        request = SearchRequest(
            query=args.query,
            top_k=args.top_k,
            latency_budget_ms=args.latency_budget_ms,
            planner=PlannerMode.GRAPH,
        )
        return await engine.search(request, request_id=request_id)
    finally:
        await backend.close()


def _json_line(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def run(argv: Sequence[str] | None = None) -> int:
    request_id = f"graph-search-{uuid4()}"
    try:
        args = build_parser().parse_args(argv)
        response = asyncio.run(_execute(args, request_id=request_id))
    except RAGPlanError as exc:
        print(_json_line(exc.response(request_id).model_dump(mode="json")), file=sys.stderr)
        return 1
    except (OSError, TypeError, ValueError):
        error = RAGPlanError(
            ErrorCode.INVALID_REQUEST,
            "graph search input is invalid",
            retryable=False,
        )
        print(_json_line(error.response(request_id).model_dump(mode="json")), file=sys.stderr)
        return 1
    print(_json_line(response.model_dump(mode="json")))
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
