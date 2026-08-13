#!/usr/bin/env python3
"""Deterministically ingest a strict JSON corpus into a vector-staged Qdrant collection."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from qdrant_client import AsyncQdrantClient

from ragplan.backends.vector.qdrant import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_COLLECTION_PREFIX,
    QdrantCollectionManager,
    QdrantVectorConfig,
    QdrantVectorWriter,
)
from ragplan.core.errors import ErrorCode, RAGPlanError
from ragplan.core.ids import canonical_document_id
from ragplan.core.models import Chunk, VectorStageManifest
from ragplan.ingestion.chunker import ChunkerConfig, chunk_document
from ragplan.ingestion.embedder import Embedder, SentenceTransformerEmbedder
from ragplan.ingestion.model_manifest import (
    ModelArtifactManifest,
    load_default_model_artifact_manifest,
    load_model_artifact_manifest,
)
from ragplan.ingestion.normalize import normalize_text


class _SafeArgumentParser(argparse.ArgumentParser):
    """Avoid echoing arbitrary command-line values in parser errors."""

    def error(self, message: str) -> None:
        del message
        raise RAGPlanError(ErrorCode.INVALID_REQUEST, "invalid command arguments")


class VectorStageWriter(Protocol):
    """Writer capability required by the offline ingestion orchestration."""

    async def stage_chunks(
        self,
        chunks: Sequence[Chunk],
        embeddings: Sequence[Sequence[float]],
        corpus_version: str,
        *,
        embedding_artifact_manifest_sha256: str,
    ) -> VectorStageManifest: ...


@dataclass(frozen=True, slots=True)
class CorpusDocument:
    source_document_id: str
    text: str


@dataclass(frozen=True, slots=True)
class CorpusFile:
    schema_version: str
    source_dataset: str
    documents: tuple[CorpusDocument, ...]


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected a positive integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    """Build the Stage 3 vector-ingestion command interface."""

    parser = _SafeArgumentParser(
        description="Chunk, embed, and stage a strict JSON corpus in Qdrant."
    )
    parser.add_argument("--input", type=Path, required=True, help="Strict v1 corpus JSON file.")
    parser.add_argument("--corpus-version", required=True, help="Immutable corpus version label.")
    parser.add_argument(
        "--model-snapshot",
        type=Path,
        required=True,
        help="Checksum-verifiable local snapshot prepared by prepare_model.py.",
    )
    parser.add_argument(
        "--stage-manifest",
        type=Path,
        required=True,
        help="Destination for the atomically written vector_staged manifest.",
    )
    parser.add_argument(
        "--chunks-output",
        type=Path,
        help="Optional canonical Chunk JSONL for the Stage 4 graph writer.",
    )
    parser.add_argument(
        "--model-manifest",
        type=Path,
        help="Optional artifact manifest; defaults to the packaged immutable manifest.",
    )
    parser.add_argument(
        "--qdrant-url",
        default="http://127.0.0.1:6333",
        help="Qdrant HTTP URL (default: local Docker service).",
    )
    parser.add_argument(
        "--collection-prefix",
        default=DEFAULT_COLLECTION_PREFIX,
        help="Prefix for deterministic corpus-version-specific Qdrant collections.",
    )
    parser.add_argument("--embedding-batch-size", type=_positive_int, default=32)
    parser.add_argument("--qdrant-batch-size", type=_positive_int, default=DEFAULT_BATCH_SIZE)
    return parser


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _strict_nonempty_string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RAGPlanError(
            ErrorCode.INVALID_REQUEST,
            f"corpus field {field} must be a non-empty string",
            retryable=False,
        )
    return value.strip()


def load_corpus_file(path: Path) -> CorpusFile:
    """Load the exact v1 corpus schema, rejecting extras, coercions, and duplicate keys."""

    try:
        serialized = path.read_text(encoding="utf-8")
        decoded = json.loads(serialized, object_pairs_hook=_reject_duplicate_keys)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise RAGPlanError(
            ErrorCode.INVALID_REQUEST,
            "corpus file is not valid strict JSON",
            retryable=False,
        ) from exc

    expected_root_fields = {"schema_version", "source_dataset", "documents"}
    if not isinstance(decoded, dict) or set(decoded) != expected_root_fields:
        raise RAGPlanError(
            ErrorCode.INVALID_REQUEST,
            "corpus file fields do not match schema v1",
            retryable=False,
        )
    if decoded["schema_version"] != "v1":
        raise RAGPlanError(
            ErrorCode.INVALID_REQUEST,
            "unsupported corpus file schema version",
            retryable=False,
        )
    source_dataset = _strict_nonempty_string(decoded["source_dataset"], field="source_dataset")
    raw_documents = decoded["documents"]
    if not isinstance(raw_documents, list) or not raw_documents:
        raise RAGPlanError(
            ErrorCode.INVALID_REQUEST,
            "corpus documents must be a non-empty JSON array",
            retryable=False,
        )

    documents: list[CorpusDocument] = []
    canonical_document_ids: set[str] = set()
    for raw_document in raw_documents:
        if not isinstance(raw_document, dict) or set(raw_document) != {
            "source_document_id",
            "text",
        }:
            raise RAGPlanError(
                ErrorCode.INVALID_REQUEST,
                "corpus document fields do not match schema v1",
                retryable=False,
            )
        source_document_id = _strict_nonempty_string(
            raw_document["source_document_id"], field="source_document_id"
        )
        text = raw_document["text"]
        if not isinstance(text, str) or not normalize_text(text):
            raise RAGPlanError(
                ErrorCode.INVALID_REQUEST,
                "corpus document text must contain normalized content",
                retryable=False,
            )
        document_id = canonical_document_id(source_dataset, source_document_id)
        if document_id in canonical_document_ids:
            raise RAGPlanError(
                ErrorCode.INVALID_REQUEST,
                "corpus contains duplicate canonical document IDs",
                retryable=False,
            )
        canonical_document_ids.add(document_id)
        documents.append(CorpusDocument(source_document_id=source_document_id, text=text))

    return CorpusFile(
        schema_version="v1",
        source_dataset=source_dataset,
        documents=tuple(documents),
    )


async def ingest_corpus(
    *,
    corpus: CorpusFile,
    corpus_version: str,
    embedder: Embedder,
    writer: VectorStageWriter,
    artifact_manifest_sha256: str,
    chunks_output: Path | None = None,
) -> VectorStageManifest:
    """Use the canonical chunker, one batched embed call, and the verified stage writer."""

    chunks = tuple(
        chunk
        for document in corpus.documents
        for chunk in chunk_document(
            source_dataset=corpus.source_dataset,
            source_document_id=document.source_document_id,
            corpus_version=corpus_version,
            text=document.text,
            tokenizer=embedder.tokenizer,
            config=ChunkerConfig(window_size=220, overlap=40),
        )
    )
    if not chunks:
        raise RAGPlanError(
            ErrorCode.CORPUS_INCONSISTENT,
            "corpus produced no non-empty chunks",
            retryable=False,
        )
    embeddings = await embedder.embed_documents(tuple(chunk.text for chunk in chunks))
    manifest = await writer.stage_chunks(
        chunks,
        embeddings,
        corpus_version,
        embedding_artifact_manifest_sha256=artifact_manifest_sha256,
    )
    if chunks_output is not None:
        write_chunks_jsonl(chunks_output, chunks)
    return manifest


def write_chunks_jsonl(path: Path, chunks: Sequence[Chunk]) -> None:
    """Atomically persist the exact chunks accepted by the vector stage."""

    payload = "".join(
        f"{chunk.model_dump_json()}\n" for chunk in sorted(chunks, key=lambda item: item.id)
    ).encode("utf-8")
    temporary_path: Path | None = None
    descriptor: int | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
        )
        temporary_path = Path(temporary_name)
        with os.fdopen(descriptor, "wb") as output:
            descriptor = None
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    except OSError as exc:
        raise RAGPlanError(
            ErrorCode.INVALID_REQUEST,
            "canonical chunk JSONL could not be written",
            retryable=False,
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def write_stage_manifest(path: Path, manifest: VectorStageManifest) -> None:
    """Durably replace one explicit vector-stage pointer without creating an active pointer."""

    temporary_path: Path | None = None
    descriptor: int | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
        )
        temporary_path = Path(temporary_name)
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            descriptor = None
            output.write(manifest.model_dump_json(indent=2))
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    except OSError as exc:
        raise RAGPlanError(
            ErrorCode.INVALID_REQUEST,
            "vector stage manifest could not be written",
            retryable=False,
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _load_manifest(path: Path | None) -> ModelArtifactManifest:
    if path is None:
        return load_default_model_artifact_manifest()
    return load_model_artifact_manifest(path)


async def _execute(args: argparse.Namespace) -> VectorStageManifest:
    corpus = load_corpus_file(args.input)
    manifest = _load_manifest(args.model_manifest)
    embedder = SentenceTransformerEmbedder.from_local_snapshot(
        snapshot_path=args.model_snapshot,
        manifest=manifest,
        batch_size=args.embedding_batch_size,
    )
    client = AsyncQdrantClient(url=args.qdrant_url)
    writer = QdrantVectorWriter(
        QdrantCollectionManager(
            client,
            QdrantVectorConfig(
                collection_prefix=args.collection_prefix,
                batch_size=args.qdrant_batch_size,
            ),
        )
    )
    try:
        stage = await ingest_corpus(
            corpus=corpus,
            corpus_version=args.corpus_version,
            embedder=embedder,
            writer=writer,
            artifact_manifest_sha256=manifest.sha256,
            chunks_output=args.chunks_output,
        )
        write_stage_manifest(args.stage_manifest, stage)
        return stage
    finally:
        await writer.close()


def _json_line(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _emit_error(error: RAGPlanError, request_id: str) -> None:
    print(
        _json_line(error.response(request_id).model_dump(mode="json")),
        file=sys.stderr,
    )


def run(argv: Sequence[str] | None = None) -> int:
    """Run ingestion and emit exactly one JSON result or one stable JSON error."""

    request_id = f"ingest-{uuid4()}"
    try:
        args = build_parser().parse_args(argv)
        stage = asyncio.run(_execute(args))
    except RAGPlanError as exc:
        _emit_error(exc, request_id)
        return 1
    except (OSError, TypeError, ValueError) as exc:
        del exc
        _emit_error(
            RAGPlanError(
                ErrorCode.INVALID_REQUEST,
                "ingestion input is invalid",
                retryable=False,
            ),
            request_id,
        )
        return 1

    print(_json_line(stage.model_dump(mode="json")))
    return 0


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
