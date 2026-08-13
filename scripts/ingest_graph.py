#!/usr/bin/env python3
"""Extract and stage the Neo4j graph from the exact vector-staged chunks."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from uuid import uuid4

from pydantic import ValidationError

from ragplan.backends.base import canonical_id_checksum
from ragplan.backends.graph.neo4j import Neo4jGraphConfig, Neo4jGraphWriter
from ragplan.core.errors import ErrorCode, RAGPlanError
from ragplan.core.models import Chunk, GraphStageManifest, VectorStageManifest
from ragplan.ingestion.entities import EntityExtractor
from ragplan.ingestion.manifest import load_contract_json, write_contract_json
from ragplan.ingestion.pipeline import extract_graph


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", type=Path, required=True, help="Canonical Chunk JSONL.")
    parser.add_argument("--vector-stage-manifest", type=Path, required=True)
    parser.add_argument("--graph-stage-manifest", type=Path, required=True)
    parser.add_argument("--uv-lock", type=Path, default=Path("uv.lock"))
    parser.add_argument(
        "--neo4j-uri",
        default=os.environ.get("RAGPLAN_GRAPH__URI", "bolt://127.0.0.1:7687"),
    )
    parser.add_argument(
        "--neo4j-user",
        default=os.environ.get("RAGPLAN_GRAPH__USER", "neo4j"),
    )
    parser.add_argument("--neo4j-database", default="neo4j")
    parser.add_argument("--batch-size", type=int, default=250)
    parser.add_argument("--transaction-timeout-seconds", type=float, default=30.0)
    return parser


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError("duplicate JSON key")
        payload[key] = value
    return payload


def load_chunks(path: Path) -> tuple[Chunk, ...]:
    chunks: list[Chunk] = []
    try:
        with path.open(encoding="utf-8") as source:
            for line in source:
                if not line.strip():
                    raise ValueError("blank JSONL record")
                payload = json.loads(line, object_pairs_hook=_reject_duplicate_keys)
                chunks.append(Chunk.model_validate_json(json.dumps(payload)))
    except (OSError, UnicodeError, json.JSONDecodeError, ValidationError, ValueError) as exc:
        raise RAGPlanError(
            ErrorCode.INVALID_REQUEST,
            "canonical chunk JSONL is invalid",
            retryable=False,
        ) from exc
    if not chunks:
        raise RAGPlanError(
            ErrorCode.CORPUS_INCONSISTENT,
            "canonical chunk JSONL is empty",
            retryable=False,
        )
    return tuple(chunks)


async def stage_graph(
    *,
    chunks: tuple[Chunk, ...],
    vector_stage: VectorStageManifest,
    extractor: EntityExtractor,
    writer: Neo4jGraphWriter,
) -> GraphStageManifest:
    chunk_ids = tuple(chunk.canonical_chunk_id for chunk in chunks)
    if (
        any(chunk.corpus_version != vector_stage.corpus_version for chunk in chunks)
        or len(set(chunk_ids)) != len(chunk_ids)
        or len(chunks) != vector_stage.chunk_count
        or canonical_id_checksum(chunk_ids) != vector_stage.canonical_id_checksum
    ):
        raise RAGPlanError(
            ErrorCode.CORPUS_INCONSISTENT,
            "graph chunks do not match the immutable vector stage",
            retryable=False,
        )
    graph = extract_graph(chunks, extractor)
    return await writer.stage_graph(
        graph.chunks,
        graph.entities,
        graph.mentions,
        graph.relations,
        vector_stage.corpus_version,
        extractor_version=graph.extractor_version,
    )


async def _execute(args: argparse.Namespace) -> GraphStageManifest:
    password = os.environ.get("RAGPLAN_GRAPH__PASSWORD") or os.environ.get("NEO4J_PASSWORD")
    if not password:
        raise RAGPlanError(
            ErrorCode.INVALID_REQUEST,
            "Neo4j password must be supplied through the environment",
            retryable=False,
        )
    chunks = load_chunks(args.chunks)
    vector_stage = load_contract_json(args.vector_stage_manifest, VectorStageManifest)
    extractor = EntityExtractor.load_pinned(lockfile=args.uv_lock, benchmark_mode=True)
    writer = Neo4jGraphWriter.connect(
        Neo4jGraphConfig(
            uri=args.neo4j_uri,
            user=args.neo4j_user,
            password=password,
            database=args.neo4j_database,
            batch_size=args.batch_size,
            transaction_timeout_seconds=args.transaction_timeout_seconds,
        )
    )
    try:
        manifest = await stage_graph(
            chunks=chunks,
            vector_stage=vector_stage,
            extractor=extractor,
            writer=writer,
        )
        write_contract_json(args.graph_stage_manifest, manifest)
        return manifest
    finally:
        await writer.close()


def run(argv: Sequence[str] | None = None) -> int:
    request_id = f"graph-ingest-{uuid4()}"
    try:
        args = build_parser().parse_args(argv)
        manifest = asyncio.run(_execute(args))
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
    print(
        json.dumps(
            {
                "status": manifest.status,
                "corpus_version": manifest.corpus_version,
                "document_count": manifest.document_count,
                "chunk_count": manifest.chunk_count,
                "entity_count": manifest.entity_count,
                "mention_count": manifest.mention_count,
                "relation_count": manifest.relation_count,
                "canonical_id_checksum": manifest.canonical_id_checksum,
                "graph_content_checksum": manifest.graph_content_checksum,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
