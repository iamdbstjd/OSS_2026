"""Runtime and ingestion contracts for vector storage."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from ragplan.backends.base import BackendHealth, BackendWriteResult
from ragplan.core.deadline import Deadline
from ragplan.core.models import Chunk, RetrievalHit


@runtime_checkable
class VectorBackend(Protocol):
    """Online vector retrieval interface.

    The query embedding is created once by analysis and every call receives the
    same absolute request deadline rather than a fresh relative timeout.
    """

    async def search(
        self,
        embedding: Sequence[float],
        top_k: int,
        corpus_version: str,
        deadline: Deadline,
    ) -> Sequence[RetrievalHit]: ...

    async def health(self) -> BackendHealth: ...

    async def close(self) -> None: ...


@runtime_checkable
class VectorIngestionWriter(Protocol):
    """Offline vector writer, intentionally separate from online search."""

    async def write_chunks(
        self,
        chunks: Sequence[Chunk],
        embeddings: Sequence[Sequence[float]],
        corpus_version: str,
        *,
        embedding_artifact_manifest_sha256: str,
    ) -> BackendWriteResult: ...

    async def close(self) -> None: ...
