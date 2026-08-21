"""Exact frozen-benchmark vector staging without re-chunking canonical evidence."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from pathlib import Path

from ragplan.backends.base import canonical_id_checksum
from ragplan.backends.vector.qdrant import QdrantVectorWriter
from ragplan.benchmark.artifacts import load_chunk_index, load_chunks
from ragplan.benchmark.records import BenchmarkProtocolConfig
from ragplan.core.errors import ErrorCode, RAGPlanError
from ragplan.core.models import Chunk, VectorStageManifest
from ragplan.ingestion.embedder import Embedder, EmbeddingVector

ProgressReporter = Callable[[int, int], None]


def load_verified_frozen_chunks(
    chunks_path: Path,
    chunk_index_path: Path,
    *,
    protocol: BenchmarkProtocolConfig,
) -> tuple[Chunk, ...]:
    """Load the Stage 2 chunks and prove every frozen index field before embedding."""

    try:
        chunks = load_chunks(chunks_path)
        index = load_chunk_index(chunk_index_path)
    except (OSError, UnicodeError, ValueError) as exc:
        raise RAGPlanError(
            ErrorCode.CORPUS_INCONSISTENT,
            "frozen benchmark chunks or index are invalid",
            retryable=False,
        ) from exc
    if len(chunks) != protocol.corpus_chunk_count or len(index) != len(chunks):
        raise RAGPlanError(
            ErrorCode.CORPUS_INCONSISTENT,
            "frozen benchmark chunk count does not match the protocol",
            retryable=False,
        )
    chunks_by_id = {chunk.canonical_chunk_id: chunk for chunk in chunks}
    if len(chunks_by_id) != len(chunks):
        raise RAGPlanError(
            ErrorCode.CORPUS_INCONSISTENT,
            "frozen benchmark chunks contain duplicate canonical IDs",
            retryable=False,
        )
    index_ids = tuple(item.canonical_chunk_id for item in index)
    if len(set(index_ids)) != len(index_ids) or set(index_ids) != set(chunks_by_id):
        raise RAGPlanError(
            ErrorCode.CORPUS_INCONSISTENT,
            "frozen benchmark chunk index and payload IDs differ",
            retryable=False,
        )
    if canonical_id_checksum(index_ids) != protocol.corpus_chunk_ids_sha256:
        raise RAGPlanError(
            ErrorCode.CORPUS_INCONSISTENT,
            "frozen benchmark canonical ID checksum differs from the protocol",
            retryable=False,
        )
    for item in index:
        chunk = chunks_by_id[item.canonical_chunk_id]
        if (
            chunk.corpus_version != protocol.corpus_version
            or chunk.document_id != item.document_id
            or chunk.position != item.position
            or chunk.token_count != item.token_count
            or hashlib.sha256(chunk.text.encode("utf-8")).hexdigest() != item.text_sha256
        ):
            raise RAGPlanError(
                ErrorCode.CORPUS_INCONSISTENT,
                "frozen benchmark chunk payload differs from its immutable index",
                retryable=False,
            )
    return tuple(sorted(chunks, key=lambda chunk: chunk.canonical_chunk_id))


async def stage_frozen_chunks(
    *,
    chunks: Sequence[Chunk],
    embedder: Embedder,
    writer: QdrantVectorWriter,
    protocol: BenchmarkProtocolConfig,
    embedding_artifact_manifest_sha256: str,
    embedding_call_size: int = 512,
    progress: ProgressReporter | None = None,
) -> VectorStageManifest:
    """Embed bounded batches, then atomically verify the complete Qdrant stage."""

    if embedding_call_size < 1:
        raise ValueError("embedding_call_size must be positive")
    if len(chunks) != protocol.corpus_chunk_count:
        raise ValueError("frozen chunk count differs from the benchmark protocol")
    embeddings: list[EmbeddingVector] = []
    for offset in range(0, len(chunks), embedding_call_size):
        batch = chunks[offset : offset + embedding_call_size]
        embeddings.extend(await embedder.embed_documents(tuple(chunk.text for chunk in batch)))
        if progress is not None:
            progress(min(offset + len(batch), len(chunks)), len(chunks))
    return await writer.stage_chunks(
        chunks,
        embeddings,
        protocol.corpus_version,
        embedding_artifact_manifest_sha256=embedding_artifact_manifest_sha256,
    )


__all__ = ["load_verified_frozen_chunks", "stage_frozen_chunks"]
