#!/usr/bin/env python3
"""Embed and stage the exact Stage 2 frozen chunks in Qdrant."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from qdrant_client import AsyncQdrantClient

from ragplan.backends.vector.qdrant import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_COLLECTION_PREFIX,
    QdrantCollectionManager,
    QdrantVectorConfig,
    QdrantVectorWriter,
)
from ragplan.benchmark.config import load_benchmark_protocol
from ragplan.ingestion.embedder import SentenceTransformerEmbedder
from ragplan.ingestion.frozen_benchmark import (
    load_verified_frozen_chunks,
    stage_frozen_chunks,
)
from ragplan.ingestion.manifest import write_contract_json
from ragplan.ingestion.model_manifest import load_default_model_artifact_manifest

ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--chunks",
        type=Path,
        default=ROOT / "benchmark/datasets/normalized/chunks_v1.jsonl",
    )
    parser.add_argument(
        "--chunk-index",
        type=Path,
        default=ROOT / "benchmark/manifests/chunk_index_v1.jsonl",
    )
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument("--stage-manifest", type=Path, required=True)
    parser.add_argument("--baseline-config", type=Path)
    parser.add_argument("--qdrant-url", default="http://127.0.0.1:6333")
    parser.add_argument("--collection-prefix", default=DEFAULT_COLLECTION_PREFIX)
    parser.add_argument("--model-batch-size", type=int, default=32)
    parser.add_argument("--embedding-call-size", type=int, default=512)
    parser.add_argument("--qdrant-batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    return parser


def _progress(completed: int, total: int) -> None:
    print(
        json.dumps(
            {"event": "embedding_progress", "completed": completed, "total": total},
            separators=(",", ":"),
            sort_keys=True,
        ),
        file=sys.stderr,
        flush=True,
    )


async def _execute(args: argparse.Namespace) -> object:
    if min(args.model_batch_size, args.embedding_call_size, args.qdrant_batch_size) < 1:
        raise ValueError("batch sizes must be positive")
    protocol = load_benchmark_protocol(args.baseline_config)
    chunks = load_verified_frozen_chunks(args.chunks, args.chunk_index, protocol=protocol)
    artifact_manifest = load_default_model_artifact_manifest()
    if artifact_manifest.revision != protocol.embedding_model_revision:
        raise ValueError("embedding manifest revision differs from the benchmark protocol")
    embedder = SentenceTransformerEmbedder.from_local_snapshot(
        snapshot_path=args.model_snapshot,
        manifest=artifact_manifest,
        batch_size=args.model_batch_size,
        device="cpu",
    )
    writer = QdrantVectorWriter(
        QdrantCollectionManager(
            AsyncQdrantClient(url=args.qdrant_url, timeout=60),
            QdrantVectorConfig(
                collection_prefix=args.collection_prefix,
                batch_size=args.qdrant_batch_size,
            ),
        )
    )
    try:
        stage = await stage_frozen_chunks(
            chunks=chunks,
            embedder=embedder,
            writer=writer,
            protocol=protocol,
            embedding_artifact_manifest_sha256=artifact_manifest.sha256,
            embedding_call_size=args.embedding_call_size,
            progress=_progress,
        )
        write_contract_json(args.stage_manifest, stage)
        return stage
    finally:
        await writer.close()


def main() -> None:
    args = build_parser().parse_args()
    stage = asyncio.run(_execute(args))
    print(stage.model_dump_json())


if __name__ == "__main__":
    main()
