#!/usr/bin/env python3
"""Live-verify Qdrant and Neo4j stages, then atomically activate one corpus."""

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
from ragplan.core.models import ChunkerVersion, GraphStageManifest, VectorStageManifest
from ragplan.ingestion.manifest import ManifestRepository, load_contract_json
from ragplan.ingestion.reconcile import ActivationCoordinator, IngestionSource


def _sha256(value: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise argparse.ArgumentTypeError("expected a lowercase SHA-256 digest")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vector-stage-manifest", type=Path, required=True)
    parser.add_argument("--graph-stage-manifest", type=Path, required=True)
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--ingestion-run-id", required=True)
    parser.add_argument("--source-dataset", required=True)
    parser.add_argument("--source-version", required=True)
    parser.add_argument("--source-sha256", type=_sha256, required=True)
    parser.add_argument(
        "--chunker-version",
        choices=tuple(item.value for item in ChunkerVersion),
        default=ChunkerVersion.TOKEN_DECODE_V1.value,
    )
    parser.add_argument(
        "--qdrant-url",
        default=os.environ.get("RAGPLAN_VECTOR__URL", "http://127.0.0.1:6333"),
    )
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
    return parser


async def _execute(args: argparse.Namespace) -> dict[str, object]:
    password = os.environ.get("RAGPLAN_GRAPH__PASSWORD") or os.environ.get("NEO4J_PASSWORD")
    if not password:
        raise RAGPlanError(
            ErrorCode.INVALID_REQUEST,
            "Neo4j password must be supplied through the environment",
            retryable=False,
        )
    vector_stage = load_contract_json(args.vector_stage_manifest, VectorStageManifest)
    graph_stage = load_contract_json(args.graph_stage_manifest, GraphStageManifest)
    qdrant_client = AsyncQdrantClient(url=args.qdrant_url)
    vector_verifier = QdrantCollectionManager(
        qdrant_client,
        QdrantVectorConfig(collection_prefix=args.collection_prefix),
    )
    graph_verifier = Neo4jGraphWriter.connect(
        Neo4jGraphConfig(
            uri=args.neo4j_uri,
            user=args.neo4j_user,
            password=password,
            database=args.neo4j_database,
        )
    )
    coordinator = ActivationCoordinator(
        vector_verifier=vector_verifier,
        graph_verifier=graph_verifier,
        repository=ManifestRepository(args.manifest_root),
    )
    try:
        pointer, manifest = await coordinator.activate(
            ingestion_run_id=args.ingestion_run_id,
            source=IngestionSource(
                source_dataset=args.source_dataset,
                source_version=args.source_version,
                source_sha256=args.source_sha256,
                chunker_version=args.chunker_version,
            ),
            vector=vector_stage,
            graph=graph_stage,
        )
        return {
            "status": "active",
            "corpus_version": manifest.corpus_version,
            "ingestion_run_id": manifest.ingestion_run_id,
            "manifest_sha256": pointer.ingestion_manifest_sha256,
        }
    finally:
        await graph_verifier.close()
        await qdrant_client.close()


def run(argv: Sequence[str] | None = None) -> int:
    request_id = f"activate-{uuid4()}"
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
        error = RAGPlanError(
            ErrorCode.INVALID_REQUEST,
            "invalid command arguments",
            retryable=False,
        )
        print(
            json.dumps(error.response(request_id).model_dump(mode="json"), sort_keys=True),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
