#!/usr/bin/env python3
"""Rollback or explicitly discard inactive Stage 4 ingestion state."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from uuid import uuid4

from qdrant_client import AsyncQdrantClient

from ragplan.backends.graph.neo4j import Neo4jGraphConfig, Neo4jGraphWriter
from ragplan.backends.vector.qdrant import (
    DEFAULT_COLLECTION_PREFIX,
    QdrantCollectionManager,
    QdrantVectorConfig,
)
from ragplan.core.errors import ErrorCode, RAGPlanError
from ragplan.ingestion.manifest import ManifestRepository


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="operation", required=True)

    rollback = subparsers.add_parser("rollback")
    rollback.add_argument("--manifest-root", type=Path, required=True)
    rollback.add_argument("--ingestion-run-id", required=True)

    discard_record = subparsers.add_parser("discard-record")
    discard_record.add_argument("--manifest-root", type=Path, required=True)
    discard_record.add_argument("--ingestion-run-id", required=True)

    discard_stores = subparsers.add_parser("discard-stores")
    discard_stores.add_argument("--manifest-root", type=Path, required=True)
    discard_stores.add_argument("--corpus-version", required=True)
    discard_stores.add_argument(
        "--qdrant-url",
        default=os.environ.get("RAGPLAN_VECTOR__URL", "http://127.0.0.1:6333"),
    )
    discard_stores.add_argument("--collection-prefix", default=DEFAULT_COLLECTION_PREFIX)
    discard_stores.add_argument(
        "--neo4j-uri",
        default=os.environ.get("RAGPLAN_GRAPH__URI", "bolt://127.0.0.1:7687"),
    )
    discard_stores.add_argument(
        "--neo4j-user",
        default=os.environ.get("RAGPLAN_GRAPH__USER", "neo4j"),
    )
    discard_stores.add_argument("--neo4j-database", default="neo4j")
    return parser


def _ensure_not_active(repository: ManifestRepository, corpus_version: str) -> None:
    try:
        _, active = repository.load_active()
    except RAGPlanError as exc:
        if exc.code is ErrorCode.NOT_READY:
            return
        raise
    if active.corpus_version == corpus_version:
        raise RAGPlanError(
            ErrorCode.CORPUS_INCONSISTENT,
            "the active corpus cannot be discarded",
            retryable=False,
        )


async def _discard_stores(args: argparse.Namespace) -> None:
    repository = ManifestRepository(args.manifest_root)
    _ensure_not_active(repository, args.corpus_version)
    password = os.environ.get("RAGPLAN_GRAPH__PASSWORD") or os.environ.get("NEO4J_PASSWORD")
    if not password:
        raise RAGPlanError(
            ErrorCode.INVALID_REQUEST,
            "Neo4j password must be supplied through the environment",
            retryable=False,
        )
    qdrant = AsyncQdrantClient(url=args.qdrant_url)
    vector = QdrantCollectionManager(
        qdrant,
        QdrantVectorConfig(collection_prefix=args.collection_prefix),
    )
    graph = Neo4jGraphWriter.connect(
        Neo4jGraphConfig(
            uri=args.neo4j_uri,
            user=args.neo4j_user,
            password=password,
            database=args.neo4j_database,
        )
    )
    try:
        await vector.discard_version(args.corpus_version)
        await graph.discard_version(args.corpus_version)
    finally:
        await graph.close()
        await qdrant.close()


async def _execute(args: argparse.Namespace) -> dict[str, object]:
    repository = ManifestRepository(args.manifest_root)
    if args.operation == "rollback":
        pointer = repository.rollback(args.ingestion_run_id)
        return {
            "status": "active",
            "corpus_version": pointer.corpus_version,
            "ingestion_run_id": pointer.ingestion_run_id,
        }
    if args.operation == "discard-record":
        repository.discard(args.ingestion_run_id)
        return {"status": "discarded", "ingestion_run_id": args.ingestion_run_id}
    if args.operation == "discard-stores":
        await _discard_stores(args)
        return {"status": "discarded", "corpus_version": args.corpus_version}
    raise RAGPlanError(ErrorCode.INVALID_REQUEST, "unknown ingestion operation", retryable=False)


def run(argv: Sequence[str] | None = None) -> int:
    request_id = f"ingestion-manage-{uuid4()}"
    try:
        args = build_parser().parse_args(argv)
        result = asyncio.run(_execute(args))
    except RAGPlanError as exc:
        print(
            json.dumps(exc.response(request_id).model_dump(mode="json"), sort_keys=True),
            file=sys.stderr,
        )
        return 1
    except (SystemExit, ValueError) as exc:
        if isinstance(exc, SystemExit) and exc.code == 0:
            return 0
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
