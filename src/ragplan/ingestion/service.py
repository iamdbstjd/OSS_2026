"""Packaged vector-ingestion and pinned-model provisioning services for the CLI."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, Self

from pydantic import Field, model_validator
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
from ragplan.core.models import (
    Chunk,
    ChunkerVersion,
    FrozenModel,
    NonEmptyString,
    VectorStageManifest,
)
from ragplan.ingestion.chunker import ChunkerConfig, chunk_document
from ragplan.ingestion.embedder import Embedder, SentenceTransformerEmbedder
from ragplan.ingestion.manifest import write_contract_json
from ragplan.ingestion.model_manifest import (
    ModelArtifactManifest,
    load_default_model_artifact_manifest,
    verify_model_artifacts,
)
from ragplan.ingestion.normalize import normalize_text


class SnapshotDownloader(Protocol):
    def __call__(
        self,
        *,
        repo_id: str,
        revision: str,
        cache_dir: str,
        local_files_only: bool,
        allow_patterns: list[str],
    ) -> str: ...


class CorpusDocument(FrozenModel):
    source_document_id: NonEmptyString
    text: NonEmptyString

    @model_validator(mode="after")
    def _normalized_content(self) -> Self:
        if not normalize_text(self.text):
            raise ValueError("corpus document text must contain normalized content")
        return self


class CorpusFile(FrozenModel):
    schema_version: Literal["v1"] = "v1"
    source_dataset: NonEmptyString
    documents: tuple[CorpusDocument, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_documents(self) -> Self:
        identifiers = tuple(
            canonical_document_id(self.source_dataset, item.source_document_id)
            for item in self.documents
        )
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("corpus contains duplicate canonical document IDs")
        return self


class VectorStageWriter(Protocol):
    async def stage_chunks(
        self,
        chunks: Sequence[Chunk],
        embeddings: Sequence[Sequence[float]],
        corpus_version: str,
        *,
        embedding_artifact_manifest_sha256: str,
        chunker_version: ChunkerVersion,
    ) -> VectorStageManifest: ...


@dataclass(frozen=True, slots=True)
class VectorIngestResult:
    stage: VectorStageManifest
    model_snapshot: Path
    stage_manifest_path: Path


def load_corpus_file(path: Path) -> CorpusFile:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
        canonical = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        return CorpusFile.model_validate_json(canonical)
    except RAGPlanError:
        raise
    except Exception as exc:
        raise RAGPlanError(
            ErrorCode.INVALID_REQUEST,
            "corpus file is not valid strict schema-v1 JSON",
            retryable=False,
        ) from exc


def prepare_pinned_model(
    cache_dir: Path,
    *,
    manifest: ModelArtifactManifest | None = None,
    downloader: SnapshotDownloader | None = None,
) -> Path:
    selected_manifest = manifest if manifest is not None else load_default_model_artifact_manifest()
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_root = cache_dir.resolve(strict=True)
    except OSError as exc:
        raise RAGPlanError(
            ErrorCode.INVALID_REQUEST,
            "model cache directory could not be prepared",
            retryable=False,
        ) from exc
    selected_downloader = downloader
    if selected_downloader is None:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise RAGPlanError(
                ErrorCode.DEPENDENCY_UNAVAILABLE,
                "Hugging Face model downloader is unavailable",
            ) from exc
        selected_downloader = snapshot_download
    try:
        downloaded = selected_downloader(
            repo_id=selected_manifest.model_id,
            revision=selected_manifest.revision,
            cache_dir=str(cache_root),
            local_files_only=False,
            allow_patterns=sorted(selected_manifest.artifacts),
        )
        snapshot = Path(downloaded).resolve(strict=True)
    except RAGPlanError:
        raise
    except Exception as exc:
        raise RAGPlanError(
            ErrorCode.DEPENDENCY_UNAVAILABLE,
            "approved embedding-model revision could not be downloaded",
        ) from exc
    if not snapshot.is_relative_to(cache_root):
        raise RAGPlanError(
            ErrorCode.MODEL_INCOMPATIBLE,
            "downloaded embedding-model snapshot escaped the dedicated cache",
            retryable=False,
        )
    verify_model_artifacts(snapshot, selected_manifest)
    return snapshot


async def ingest_vector_corpus(
    *,
    input_path: Path,
    corpus_version: str,
    stage_manifest_path: Path,
    model_snapshot: Path | None = None,
    model_cache: Path = Path("models/minilm"),
    chunks_output: Path | None = None,
    qdrant_url: str = "http://127.0.0.1:6333",
    collection_prefix: str = DEFAULT_COLLECTION_PREFIX,
    embedding_batch_size: int = 32,
    qdrant_batch_size: int = DEFAULT_BATCH_SIZE,
    chunker_version: ChunkerVersion = ChunkerVersion.TOKEN_DECODE_V1,
) -> VectorIngestResult:
    if not corpus_version.strip():
        raise RAGPlanError(
            ErrorCode.INVALID_REQUEST,
            "corpus version must be non-empty",
            retryable=False,
        )
    manifest = load_default_model_artifact_manifest()
    snapshot = (
        model_snapshot
        if model_snapshot is not None
        else prepare_pinned_model(model_cache, manifest=manifest)
    )
    corpus = load_corpus_file(input_path)
    embedder = SentenceTransformerEmbedder.from_local_snapshot(
        snapshot_path=snapshot,
        manifest=manifest,
        batch_size=embedding_batch_size,
    )
    client = AsyncQdrantClient(url=qdrant_url)
    writer = QdrantVectorWriter(
        QdrantCollectionManager(
            client,
            QdrantVectorConfig(
                collection_prefix=collection_prefix,
                batch_size=qdrant_batch_size,
            ),
        )
    )
    try:
        stage, chunks = await stage_corpus(
            corpus,
            corpus_version=corpus_version,
            embedder=embedder,
            writer=writer,
            embedding_artifact_manifest_sha256=manifest.sha256,
            chunker_version=chunker_version,
        )
    finally:
        await writer.close()
    write_contract_json(stage_manifest_path, stage)
    if chunks_output is not None:
        _write_chunks_jsonl(chunks_output, chunks)
    return VectorIngestResult(
        stage=stage,
        model_snapshot=snapshot,
        stage_manifest_path=stage_manifest_path,
    )


async def stage_corpus(
    corpus: CorpusFile,
    *,
    corpus_version: str,
    embedder: Embedder,
    writer: VectorStageWriter,
    embedding_artifact_manifest_sha256: str,
    chunker_version: ChunkerVersion = ChunkerVersion.TOKEN_DECODE_V1,
) -> tuple[VectorStageManifest, tuple[Chunk, ...]]:
    chunks = _chunk_corpus(
        corpus,
        corpus_version=corpus_version,
        embedder=embedder,
        chunker_version=chunker_version,
    )
    embeddings = await embedder.embed_documents(tuple(chunk.text for chunk in chunks))
    stage = await writer.stage_chunks(
        chunks,
        embeddings,
        corpus_version,
        embedding_artifact_manifest_sha256=embedding_artifact_manifest_sha256,
        chunker_version=chunker_version,
    )
    return stage, chunks


def _chunk_corpus(
    corpus: CorpusFile,
    *,
    corpus_version: str,
    embedder: Embedder,
    chunker_version: ChunkerVersion,
) -> tuple[Chunk, ...]:
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
            chunker_version=chunker_version,
        )
    )
    if not chunks:
        raise RAGPlanError(
            ErrorCode.CORPUS_INCONSISTENT,
            "corpus produced no non-empty chunks",
            retryable=False,
        )
    return chunks


def _write_chunks_jsonl(path: Path, chunks: Sequence[Chunk]) -> None:
    payload = "".join(
        f"{item.model_dump_json()}\n" for item in sorted(chunks, key=lambda chunk: chunk.id)
    ).encode("utf-8")
    _atomic_write(path, payload)


def _atomic_write(path: Path, payload: bytes) -> None:
    descriptor: int | None = None
    temporary: Path | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
        temporary = Path(name)
        with os.fdopen(descriptor, "wb") as output:
            descriptor = None
            os.fchmod(output.fileno(), 0o644)
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        temporary = None
    except OSError as exc:
        raise RAGPlanError(
            ErrorCode.DEPENDENCY_UNAVAILABLE,
            "ingestion artifact could not be written",
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> Mapping[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


__all__ = [
    "CorpusDocument",
    "CorpusFile",
    "SnapshotDownloader",
    "VectorIngestResult",
    "ingest_vector_corpus",
    "load_corpus_file",
    "prepare_pinned_model",
    "stage_corpus",
]
