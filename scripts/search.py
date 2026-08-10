#!/usr/bin/env python3
"""Run one redacted Stage 3 vector query through the production engine path."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from uuid import uuid4

from pydantic import ValidationError
from qdrant_client import AsyncQdrantClient

from ragplan.backends.vector.qdrant import (
    DEFAULT_COLLECTION_PREFIX,
    QdrantCollectionManager,
    QdrantVectorBackend,
    QdrantVectorConfig,
)
from ragplan.core.engine import VectorSearchEngine
from ragplan.core.errors import ErrorCode, RAGPlanError
from ragplan.core.models import PlannerMode, SearchRequest, SearchResponse, VectorStageManifest
from ragplan.ingestion.embedder import SentenceTransformerEmbedder
from ragplan.ingestion.model_manifest import (
    ModelArtifactManifest,
    load_default_model_artifact_manifest,
    load_model_artifact_manifest,
)
from ragplan.planner.catalog import load_default_plan_catalog, load_plan_catalog


class _SafeArgumentParser(argparse.ArgumentParser):
    """Avoid including a raw query or path in command-line validation failures."""

    def error(self, message: str) -> None:
        del message
        raise RAGPlanError(ErrorCode.INVALID_REQUEST, "invalid command arguments")


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected a positive integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    """Build the explicit vector-mode search interface."""

    parser = _SafeArgumentParser(
        description="Search one verified vector-staged corpus through VectorSearchEngine."
    )
    parser.add_argument("--query", required=True, help="Query text; never emitted in logs/trace.")
    parser.add_argument("--stage-manifest", type=Path, required=True)
    parser.add_argument(
        "--model-snapshot",
        type=Path,
        required=True,
        help="Checksum-verifiable local snapshot prepared by prepare_model.py.",
    )
    parser.add_argument(
        "--model-manifest",
        type=Path,
        help="Optional artifact manifest; defaults to the packaged immutable manifest.",
    )
    parser.add_argument("--plan-catalog", type=Path, help="Optional immutable plan catalog.")
    parser.add_argument(
        "--qdrant-url",
        default="http://127.0.0.1:6333",
        help="Qdrant HTTP URL (default: local Docker service).",
    )
    parser.add_argument(
        "--collection-prefix",
        default=DEFAULT_COLLECTION_PREFIX,
        help="Must match the prefix used to create the staged collection.",
    )
    parser.add_argument("--top-k", type=_positive_int, default=10)
    parser.add_argument("--latency-budget-ms", type=_positive_int, default=200)
    parser.add_argument("--embedding-batch-size", type=_positive_int, default=32)
    return parser


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def load_stage_manifest(path: Path) -> VectorStageManifest:
    """Load exact, immutable vector staging evidence; an active pointer is not accepted."""

    try:
        serialized = path.read_text(encoding="utf-8")
        decoded = json.loads(serialized, object_pairs_hook=_reject_duplicate_keys)
        return VectorStageManifest.model_validate(decoded)
    except (OSError, UnicodeError, json.JSONDecodeError, ValidationError, ValueError) as exc:
        raise RAGPlanError(
            ErrorCode.CORPUS_INCONSISTENT,
            "vector stage manifest is invalid",
            retryable=False,
        ) from exc


def _load_model_manifest(path: Path | None) -> ModelArtifactManifest:
    if path is None:
        return load_default_model_artifact_manifest()
    return load_model_artifact_manifest(path)


async def _execute(args: argparse.Namespace, *, request_id: str) -> SearchResponse:
    stage = load_stage_manifest(args.stage_manifest)
    model_manifest = _load_model_manifest(args.model_manifest)
    if stage.embedding_artifact_manifest_sha256 != model_manifest.sha256:
        raise RAGPlanError(
            ErrorCode.MODEL_INCOMPATIBLE,
            "vector stage and embedding artifact manifests do not match",
            retryable=False,
        )
    embedder = SentenceTransformerEmbedder.from_local_snapshot(
        snapshot_path=args.model_snapshot,
        manifest=model_manifest,
        batch_size=args.embedding_batch_size,
    )
    request = SearchRequest(
        query=args.query,
        top_k=args.top_k,
        latency_budget_ms=args.latency_budget_ms,
        planner=PlannerMode.VECTOR,
    )
    plan_catalog = (
        load_default_plan_catalog()
        if args.plan_catalog is None
        else load_plan_catalog(args.plan_catalog)
    )

    client = AsyncQdrantClient(url=args.qdrant_url)
    manager = QdrantCollectionManager(
        client,
        QdrantVectorConfig(collection_prefix=args.collection_prefix),
    )
    backend = QdrantVectorBackend(manager)
    try:
        verified_stage = await manager.verify_stage(stage)
        engine = VectorSearchEngine(
            embedder=embedder,
            vector_backend=backend,
            plan_catalog=plan_catalog,
            vector_stage=verified_stage,
        )
        return await engine.search(request, request_id=request_id)
    finally:
        await backend.close()


def _json_line(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _emit_error(error: RAGPlanError, request_id: str) -> None:
    print(
        _json_line(error.response(request_id).model_dump(mode="json")),
        file=sys.stderr,
    )


def run(argv: Sequence[str] | None = None) -> int:
    """Run one search and emit no raw-query-bearing diagnostics."""

    request_id = f"search-{uuid4()}"
    try:
        args = build_parser().parse_args(argv)
        response = asyncio.run(_execute(args, request_id=request_id))
    except RAGPlanError as exc:
        _emit_error(exc, request_id)
        return 1
    except (OSError, TypeError, ValueError) as exc:
        del exc
        _emit_error(
            RAGPlanError(
                ErrorCode.INVALID_REQUEST,
                "search input is invalid",
                retryable=False,
            ),
            request_id,
        )
        return 1

    # SearchResponse/trace serializers exclude QueryAnalysis embeddings and no raw query is a field.
    print(_json_line(response.model_dump(mode="json")))
    return 0


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
