"""Qdrant collection, ingestion, and online vector retrieval adapters."""

from __future__ import annotations

import asyncio
import hashlib
import math
import re
import struct
from collections import deque
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Final, cast
from uuid import UUID

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import ResponseHandlingException

from ragplan.backends.base import (
    BackendHealth,
    BackendHealthStatus,
    BackendWriteResult,
    canonical_id_checksum,
)
from ragplan.core.deadline import Deadline
from ragplan.core.errors import ErrorCode, RAGPlanError, TimeoutOrigin
from ragplan.core.ids import qdrant_point_id
from ragplan.core.models import Chunk, RetrievalHit, VectorStageManifest

VECTOR_SIZE: Final = 384
DEFAULT_BATCH_SIZE: Final = 64
DEFAULT_COLLECTION_PREFIX: Final = "ragplan_chunks"
_COLLECTION_PREFIX_PATTERN: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")
_REQUIRED_PAYLOAD_INDEXES: Final = {
    "corpus_version": models.PayloadSchemaType.KEYWORD,
    "canonical_chunk_id": models.PayloadSchemaType.KEYWORD,
    "document_id": models.PayloadSchemaType.KEYWORD,
    "position": models.PayloadSchemaType.INTEGER,
    "embedding_artifact_manifest_sha256": models.PayloadSchemaType.KEYWORD,
    "embedding_checksum": models.PayloadSchemaType.KEYWORD,
}
_SHA256_PATTERN: Final = re.compile(r"^[0-9a-f]{64}$")
_DEPENDENCY_MESSAGE: Final = "vector storage dependency is unavailable"
_COLLECTION_MESSAGE: Final = "vector collection is unavailable"
_VECTOR_SCHEMA_MESSAGE: Final = "vector collection schema is incompatible with the model"
_PAYLOAD_SCHEMA_MESSAGE: Final = "vector payload schema is incompatible with the corpus"
_CORPUS_DATA_MESSAGE: Final = "vector storage contains inconsistent corpus data"
_DEADLINE_MESSAGE: Final = "vector retrieval deadline exceeded"


@dataclass(frozen=True, slots=True)
class QdrantVectorConfig:
    """Immutable Stage 3 storage contract for one family of versioned collections."""

    collection_prefix: str = DEFAULT_COLLECTION_PREFIX
    vector_size: int = VECTOR_SIZE
    batch_size: int = DEFAULT_BATCH_SIZE
    require_payload_indexes: bool = True
    hnsw_ef: int | None = None

    def __post_init__(self) -> None:
        if _COLLECTION_PREFIX_PATTERN.fullmatch(self.collection_prefix) is None:
            raise ValueError(
                "collection_prefix must start with an alphanumeric character and contain only "
                "letters, digits, underscores, or hyphens"
            )
        if isinstance(self.vector_size, bool) or self.vector_size != VECTOR_SIZE:
            raise ValueError(f"vector_size must be exactly {VECTOR_SIZE}")
        if isinstance(self.batch_size, bool) or self.batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if self.hnsw_ef is not None and (isinstance(self.hnsw_ef, bool) or self.hnsw_ef < 1):
            raise ValueError("hnsw_ef must be a positive integer when provided")


@dataclass(frozen=True, slots=True)
class _StoredVectorProvenance:
    artifact_sha256: str
    declared_embedding_sha256: str
    actual_embedding_sha256: str
    vector: tuple[float, ...]


def collection_name_for_version(collection_prefix: str, corpus_version: str) -> str:
    """Return a deterministic collection name without exposing arbitrary version text."""

    if _COLLECTION_PREFIX_PATTERN.fullmatch(collection_prefix) is None:
        raise ValueError("invalid Qdrant collection prefix")
    _validate_corpus_version(corpus_version)
    version_digest = hashlib.sha256(corpus_version.encode("utf-8")).hexdigest()
    return f"{collection_prefix}_{version_digest}"


def embedding_set_checksum(vector_checksums: Mapping[str, str]) -> str:
    """Hash the immutable canonical-ID to embedding-checksum mapping."""

    serialized: list[bytes] = []
    for canonical_id, vector_checksum in sorted(vector_checksums.items()):
        if not canonical_id or _SHA256_PATTERN.fullmatch(vector_checksum) is None:
            raise ValueError(
                "embedding provenance entries must be canonical IDs and SHA-256 values"
            )
        encoded_id = canonical_id.encode("utf-8")
        serialized.append(len(encoded_id).to_bytes(8, "big"))
        serialized.append(encoded_id)
        serialized.append(bytes.fromhex(vector_checksum))
    return hashlib.sha256(b"".join(serialized)).hexdigest()


class QdrantCollectionManager:
    """Create and validate immutable, corpus-version-specific Qdrant collections."""

    def __init__(
        self,
        client: AsyncQdrantClient,
        config: QdrantVectorConfig | None = None,
        *,
        client_factory: Callable[[], AsyncQdrantClient] | None = None,
    ) -> None:
        self._client = client
        self._config = config if config is not None else QdrantVectorConfig()
        self._client_factory = client_factory
        self._validated_versions: set[str] = set()

    @property
    def client(self) -> AsyncQdrantClient:
        return self._client

    async def recycle_client(self) -> None:
        """Discard a possibly-poisoned HTTP connection pool and rebuild the client.

        Cancelling a request while httpx is establishing its connection leaks one
        pool slot permanently (httpx 0.28), so any application-deadline
        cancellation that interrupted an in-flight backend call must be followed
        by a pool rebuild to keep later requests from blocking forever.
        """

        if self._client_factory is None:
            return
        replacement = self._client_factory()
        previous, self._client = self._client, replacement
        try:
            await previous.close()
        except Exception:  # noqa: BLE001 - the old pool is being discarded anyway
            pass

    def invalidate_validation(self, corpus_version: str | None = None) -> None:
        """Forget cached collection validation (e.g., after a 404 from the store)."""

        if corpus_version is None:
            self._validated_versions.clear()
        else:
            self._validated_versions.discard(corpus_version)

    @property
    def config(self) -> QdrantVectorConfig:
        return self._config

    def collection_name(self, corpus_version: str) -> str:
        return collection_name_for_version(self._config.collection_prefix, corpus_version)

    async def ensure_collection(self, corpus_version: str) -> str:
        """Create a missing collection, while refusing to mutate an incompatible one."""

        collection_name = self.collection_name(corpus_version)
        try:
            exists = await self._client.collection_exists(collection_name)
            if exists:
                info = await self._client.get_collection(collection_name)
                self._validate_collection_info(info, corpus_version=corpus_version)
                return collection_name

            created = await self._client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=self._config.vector_size,
                    distance=models.Distance.COSINE,
                ),
                metadata={
                    "ragplan_schema": "vector-v2",
                    "corpus_version_sha256": hashlib.sha256(
                        corpus_version.encode("utf-8")
                    ).hexdigest(),
                },
            )
            if not created:
                raise RAGPlanError(ErrorCode.DEPENDENCY_UNAVAILABLE, _DEPENDENCY_MESSAGE)

            if self._config.require_payload_indexes:
                for field_name, field_schema in _REQUIRED_PAYLOAD_INDEXES.items():
                    result = await self._client.create_payload_index(
                        collection_name=collection_name,
                        field_name=field_name,
                        field_schema=field_schema,
                        wait=True,
                    )
                    if result.status is not models.UpdateStatus.COMPLETED:
                        raise RAGPlanError(
                            ErrorCode.DEPENDENCY_UNAVAILABLE,
                            _DEPENDENCY_MESSAGE,
                        )

            info = await self._client.get_collection(collection_name)
            self._validate_collection_info(info, corpus_version=corpus_version)
            self._validated_versions.add(corpus_version)
            return collection_name
        except RAGPlanError:
            raise
        except Exception as exc:
            if _is_qdrant_client_timeout(exc):
                raise RAGPlanError(
                    ErrorCode.DEADLINE_EXCEEDED,
                    _DEADLINE_MESSAGE,
                    timeout_origin=TimeoutOrigin.BACKEND_CLIENT,
                ) from exc
            raise RAGPlanError(
                ErrorCode.DEPENDENCY_UNAVAILABLE,
                _DEPENDENCY_MESSAGE,
            ) from exc

    async def require_collection(
        self,
        corpus_version: str,
        *,
        use_cache: bool = True,
    ) -> str:
        """Validate a collection without creating or repairing online state.

        Successful validation is cached per corpus version so the serving path
        performs zero metadata round trips; ``use_cache=False`` forces a fresh
        read (boot-time verification and post-invalidation re-checks).
        """

        collection_name = self.collection_name(corpus_version)
        if use_cache and corpus_version in self._validated_versions:
            return collection_name
        try:
            exists = await self._client.collection_exists(collection_name)
            if not exists:
                raise RAGPlanError(ErrorCode.DEPENDENCY_UNAVAILABLE, _COLLECTION_MESSAGE)
            info = await self._client.get_collection(collection_name)
            self._validate_collection_info(info, corpus_version=corpus_version)
            self._validated_versions.add(corpus_version)
            return collection_name
        except RAGPlanError:
            raise
        except Exception as exc:
            if _is_qdrant_client_timeout(exc):
                raise RAGPlanError(
                    ErrorCode.DEADLINE_EXCEEDED,
                    _DEADLINE_MESSAGE,
                    timeout_origin=TimeoutOrigin.BACKEND_CLIENT,
                ) from exc
            raise RAGPlanError(
                ErrorCode.DEPENDENCY_UNAVAILABLE,
                _DEPENDENCY_MESSAGE,
            ) from exc

    async def verify_stage(self, manifest: VectorStageManifest) -> VectorStageManifest:
        """Re-read Qdrant and verify that a vector staging manifest is exact."""

        expected_collection = self.collection_name(manifest.corpus_version)
        if manifest.collection_name != expected_collection:
            raise RAGPlanError(ErrorCode.CORPUS_INCONSISTENT, _CORPUS_DATA_MESSAGE)
        await self.require_collection(manifest.corpus_version)
        try:
            observed_count = await _count_version(
                self._client,
                expected_collection,
                manifest.corpus_version,
            )
            total_count = await _count_all(self._client, expected_collection)
            observed_provenance = await _read_vector_provenance(
                self._client,
                expected_collection,
                manifest.corpus_version,
            )
        except RAGPlanError:
            raise
        except Exception as exc:
            if _is_qdrant_client_timeout(exc):
                raise RAGPlanError(
                    ErrorCode.DEADLINE_EXCEEDED,
                    _DEADLINE_MESSAGE,
                    timeout_origin=TimeoutOrigin.BACKEND_CLIENT,
                ) from exc
            raise RAGPlanError(
                ErrorCode.DEPENDENCY_UNAVAILABLE,
                _DEPENDENCY_MESSAGE,
            ) from exc
        if (
            observed_count != manifest.chunk_count
            or total_count != observed_count
            or len(observed_provenance) != observed_count
            or canonical_id_checksum(tuple(observed_provenance)) != manifest.canonical_id_checksum
            or any(
                provenance.artifact_sha256 != manifest.embedding_artifact_manifest_sha256
                for provenance in observed_provenance.values()
            )
            or embedding_set_checksum(
                {
                    canonical_id: provenance.actual_embedding_sha256
                    for canonical_id, provenance in observed_provenance.items()
                }
            )
            != manifest.embedding_set_checksum
        ):
            raise RAGPlanError(ErrorCode.CORPUS_INCONSISTENT, _CORPUS_DATA_MESSAGE)
        return manifest

    async def discard_version(self, corpus_version: str) -> None:
        """Delete one explicit immutable collection; never called implicitly."""

        collection_name = self.collection_name(corpus_version)
        self.invalidate_validation(corpus_version)
        try:
            if await self._client.collection_exists(collection_name):
                deleted = await self._client.delete_collection(collection_name)
                if not deleted:
                    raise RAGPlanError(
                        ErrorCode.DEPENDENCY_UNAVAILABLE,
                        _DEPENDENCY_MESSAGE,
                    )
        except RAGPlanError:
            raise
        except Exception as exc:
            raise RAGPlanError(
                ErrorCode.DEPENDENCY_UNAVAILABLE,
                _DEPENDENCY_MESSAGE,
            ) from exc

    def _validate_collection_info(
        self,
        info: models.CollectionInfo,
        *,
        corpus_version: str,
    ) -> None:
        vectors = info.config.params.vectors
        if not isinstance(vectors, models.VectorParams):
            raise RAGPlanError(ErrorCode.MODEL_INCOMPATIBLE, _VECTOR_SCHEMA_MESSAGE)
        if (
            vectors.size != self._config.vector_size
            or vectors.distance is not models.Distance.COSINE
        ):
            raise RAGPlanError(ErrorCode.MODEL_INCOMPATIBLE, _VECTOR_SCHEMA_MESSAGE)

        metadata = info.config.metadata
        expected_version_digest = hashlib.sha256(corpus_version.encode("utf-8")).hexdigest()
        if not isinstance(metadata, Mapping) or (
            metadata.get("ragplan_schema") != "vector-v2"
            or metadata.get("corpus_version_sha256") != expected_version_digest
        ):
            raise RAGPlanError(ErrorCode.CORPUS_INCONSISTENT, _PAYLOAD_SCHEMA_MESSAGE)

        if not self._config.require_payload_indexes:
            return
        for field_name, expected_type in _REQUIRED_PAYLOAD_INDEXES.items():
            actual = info.payload_schema.get(field_name)
            if actual is None or actual.data_type is not expected_type:
                raise RAGPlanError(ErrorCode.CORPUS_INCONSISTENT, _PAYLOAD_SCHEMA_MESSAGE)


class QdrantVectorWriter:
    """Idempotent batch writer with exact post-write count and ID reconciliation."""

    def __init__(self, collection_manager: QdrantCollectionManager) -> None:
        self._collections = collection_manager
        self._closed = False

    async def write_chunks(
        self,
        chunks: Sequence[Chunk],
        embeddings: Sequence[Sequence[float]],
        corpus_version: str,
        *,
        embedding_artifact_manifest_sha256: str,
    ) -> BackendWriteResult:
        _validate_corpus_version(corpus_version)
        _validate_sha256(
            embedding_artifact_manifest_sha256,
            field_name="embedding_artifact_manifest_sha256",
        )
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")

        canonical_ids = [chunk.canonical_chunk_id for chunk in chunks]
        if len(set(canonical_ids)) != len(canonical_ids):
            raise ValueError("chunks must not contain duplicate canonical IDs")
        for chunk in chunks:
            if chunk.corpus_version != corpus_version:
                raise ValueError("every chunk must match the requested corpus_version")

        vectors = [_validate_unit_embedding(vector) for vector in embeddings]
        expected_vectors = {
            chunk.canonical_chunk_id: tuple(vector)
            for chunk, vector in zip(chunks, vectors, strict=True)
        }
        collection_name = await self._collections.ensure_collection(corpus_version)
        expected_ids = set(canonical_ids)
        points = [
            models.PointStruct(
                id=qdrant_point_id(chunk.canonical_chunk_id),
                vector=vector,
                payload={
                    "corpus_version": corpus_version,
                    "canonical_chunk_id": chunk.canonical_chunk_id,
                    "document_id": chunk.document_id,
                    "position": chunk.position,
                    "text": chunk.text,
                    "embedding_artifact_manifest_sha256": (embedding_artifact_manifest_sha256),
                    # Qdrant normalizes cosine vectors on write. This provisional
                    # value is replaced with the exact stored-vector checksum
                    # after the write has completed.
                    "embedding_checksum": _embedding_vector_checksum(vector),
                },
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]

        try:
            existing_count = await self._count_version(collection_name, corpus_version)
            total_count = await _count_all(self._collections.client, collection_name)
            existing_provenance = await self._read_vector_provenance(
                collection_name,
                corpus_version,
                # A non-empty version is immutable. An unsealed checksum is
                # evidence of a partial or externally modified write and must
                # never be repaired during a same-version retry.
                require_sealed=existing_count > 0,
            )
            existing_ids = set(existing_provenance)
            if total_count != existing_count or len(existing_provenance) != existing_count:
                raise RAGPlanError(ErrorCode.CORPUS_INCONSISTENT, _CORPUS_DATA_MESSAGE)
            if existing_count:
                if existing_ids != expected_ids:
                    raise RAGPlanError(ErrorCode.CORPUS_INCONSISTENT, _CORPUS_DATA_MESSAGE)
                _require_expected_vectors(
                    existing_provenance,
                    expected_vectors,
                    embedding_artifact_manifest_sha256,
                )
                return BackendWriteResult(
                    corpus_version=corpus_version,
                    written_count=existing_count,
                    canonical_id_checksum=canonical_id_checksum(tuple(existing_ids)),
                )

            for offset in range(0, len(points), self._collections.config.batch_size):
                result = await self._collections.client.upsert(
                    collection_name=collection_name,
                    points=points[offset : offset + self._collections.config.batch_size],
                    wait=True,
                )
                if result.status is not models.UpdateStatus.COMPLETED:
                    raise RAGPlanError(
                        ErrorCode.DEPENDENCY_UNAVAILABLE,
                        _DEPENDENCY_MESSAGE,
                    )

            observed_count = await self._count_version(collection_name, corpus_version)
            total_count = await _count_all(self._collections.client, collection_name)
            observed_provenance = await self._read_vector_provenance(
                collection_name,
                corpus_version,
                require_sealed=False,
            )
            observed_ids = set(observed_provenance)
            _require_expected_vectors(
                observed_provenance,
                expected_vectors,
                embedding_artifact_manifest_sha256,
            )
            observed_provenance = await self._seal_vector_provenance(
                collection_name,
                corpus_version,
                observed_provenance,
            )
        except RAGPlanError:
            raise
        except Exception as exc:
            raise RAGPlanError(
                ErrorCode.DEPENDENCY_UNAVAILABLE,
                _DEPENDENCY_MESSAGE,
            ) from exc

        if (
            observed_count != len(expected_ids)
            or total_count != observed_count
            or observed_ids != expected_ids
            or set(observed_provenance) != expected_ids
        ):
            raise RAGPlanError(ErrorCode.CORPUS_INCONSISTENT, _CORPUS_DATA_MESSAGE)
        return BackendWriteResult(
            corpus_version=corpus_version,
            written_count=observed_count,
            canonical_id_checksum=canonical_id_checksum(tuple(observed_ids)),
        )

    async def stage_chunks(
        self,
        chunks: Sequence[Chunk],
        embeddings: Sequence[Sequence[float]],
        corpus_version: str,
        *,
        embedding_artifact_manifest_sha256: str,
    ) -> VectorStageManifest:
        """Write chunks and return verified evidence suitable for engine startup."""

        result = await self.write_chunks(
            chunks,
            embeddings,
            corpus_version,
            embedding_artifact_manifest_sha256=embedding_artifact_manifest_sha256,
        )
        provenance = await self._read_vector_provenance(
            self._collections.collection_name(result.corpus_version),
            result.corpus_version,
        )
        manifest = VectorStageManifest(
            corpus_version=result.corpus_version,
            collection_name=self._collections.collection_name(result.corpus_version),
            chunk_count=result.written_count,
            canonical_id_checksum=result.canonical_id_checksum,
            embedding_set_checksum=embedding_set_checksum(
                {
                    canonical_id: stored.actual_embedding_sha256
                    for canonical_id, stored in provenance.items()
                }
            ),
            embedding_artifact_manifest_sha256=embedding_artifact_manifest_sha256,
        )
        return await self._collections.verify_stage(manifest)

    async def _count_version(self, collection_name: str, corpus_version: str) -> int:
        return await _count_version(
            self._collections.client,
            collection_name,
            corpus_version,
        )

    async def _read_vector_provenance(
        self,
        collection_name: str,
        corpus_version: str,
        *,
        require_sealed: bool = True,
    ) -> dict[str, _StoredVectorProvenance]:
        return await _read_vector_provenance(
            self._collections.client,
            collection_name,
            corpus_version,
            require_sealed=require_sealed,
        )

    async def _seal_vector_provenance(
        self,
        collection_name: str,
        corpus_version: str,
        provenance: Mapping[str, _StoredVectorProvenance],
    ) -> dict[str, _StoredVectorProvenance]:
        operations = [
            models.SetPayloadOperation(
                set_payload=models.SetPayload(
                    payload={"embedding_checksum": stored.actual_embedding_sha256},
                    points=[qdrant_point_id(canonical_id)],
                )
            )
            for canonical_id, stored in provenance.items()
            if stored.declared_embedding_sha256 != stored.actual_embedding_sha256
        ]
        for offset in range(0, len(operations), self._collections.config.batch_size):
            batch = operations[offset : offset + self._collections.config.batch_size]
            results = await self._collections.client.batch_update_points(
                collection_name=collection_name,
                update_operations=batch,
                wait=True,
            )
            if len(results) != len(batch) or any(
                result.status is not models.UpdateStatus.COMPLETED for result in results
            ):
                raise RAGPlanError(ErrorCode.DEPENDENCY_UNAVAILABLE, _DEPENDENCY_MESSAGE)
        return await self._read_vector_provenance(collection_name, corpus_version)

    async def close(self) -> None:
        """Release writer state only.

        The underlying ``AsyncQdrantClient`` is owned by the collection manager
        and shared with the search backend, so the writer must never close it;
        doing so poisoned every later search in the same process.
        """

        self._closed = True


class QdrantVectorBackend:
    """Deadline-aware async Qdrant search backend for the Stage 3 vector path."""

    def __init__(self, collection_manager: QdrantCollectionManager) -> None:
        self._collections = collection_manager
        self._closed = False
        self._inflight: deque[asyncio.Task[object]] = deque()

    async def search(
        self,
        embedding: Sequence[float],
        top_k: int,
        corpus_version: str,
        deadline: Deadline,
    ) -> Sequence[RetrievalHit]:
        vector = _validate_embedding(embedding)
        _validate_top_k(top_k)
        _validate_corpus_version(corpus_version)

        try:
            collection_name = await _within_deadline(
                deadline,
                lambda: cast(
                    "Awaitable[str]",
                    self._collections.require_collection(corpus_version),
                ),
                inflight=self._inflight,
            )
            response = await _within_deadline(
                deadline,
                lambda: self._query(
                    collection_name=collection_name,
                    vector=vector,
                    top_k=top_k,
                    corpus_version=corpus_version,
                ),
                inflight=self._inflight,
            )
        except RAGPlanError as exc:
            if exc.code is ErrorCode.DEPENDENCY_UNAVAILABLE and not exc.retryable:
                # A cached validation can go stale only when the store lost the
                # collection; force a fresh metadata read on the next request.
                self._collections.invalidate_validation(corpus_version)
            raise

        try:
            return tuple(
                _retrieval_hit(point, corpus_version=corpus_version, rank=rank)
                for rank, point in enumerate(response.points, start=1)
            )
        except RAGPlanError:
            raise
        except Exception as exc:
            raise RAGPlanError(ErrorCode.CORPUS_INCONSISTENT, _CORPUS_DATA_MESSAGE) from exc

    async def _query(
        self,
        *,
        collection_name: str,
        vector: list[float],
        top_k: int,
        corpus_version: str,
    ) -> models.QueryResponse:
        try:
            return await self._collections.client.query_points(
                collection_name=collection_name,
                query=vector,
                query_filter=_corpus_filter(corpus_version),
                limit=top_k,
                with_payload=True,
                with_vectors=False,
                search_params=(
                    models.SearchParams(hnsw_ef=self._collections.config.hnsw_ef)
                    if self._collections.config.hnsw_ef is not None
                    else None
                ),
            )
        except RAGPlanError:
            raise
        except Exception as exc:
            if _is_qdrant_client_timeout(exc):
                raise RAGPlanError(
                    ErrorCode.DEADLINE_EXCEEDED,
                    _DEADLINE_MESSAGE,
                    timeout_origin=TimeoutOrigin.BACKEND_CLIENT,
                ) from exc
            raise RAGPlanError(
                ErrorCode.DEPENDENCY_UNAVAILABLE,
                _DEPENDENCY_MESSAGE,
            ) from exc

    async def health(self) -> BackendHealth:
        try:
            await self._collections.client.get_collections()
        except Exception:
            return BackendHealth(
                BackendHealthStatus.UNAVAILABLE,
                "vector storage health check failed",
            )
        return BackendHealth(BackendHealthStatus.HEALTHY)

    async def close(self) -> None:
        if not self._closed:
            self._closed = True
            await self._collections.client.close()


_MAX_ABANDONED_BACKEND_TASKS: Final = 64


def _consume_finished_task(task: asyncio.Task[object]) -> None:
    """Retrieve the outcome of an abandoned backend task.

    Detached tasks that outlive their request finish later on their own; their
    result is intentionally discarded, but the exception must still be
    retrieved to keep asyncio from logging "exception was never retrieved".
    """

    if not task.cancelled() and task.exception() is not None:
        pass  # expected: client-side timeouts of requests we already gave up on


async def _within_deadline[T](
    deadline: Deadline,
    operation_factory: Callable[[], Awaitable[T]],
    *,
    inflight: deque[asyncio.Task[object]] | None = None,
) -> T:
    """Run one backend operation bounded by the application deadline.

    On deadline exhaustion the operation task is deliberately NOT cancelled:
    cancelling qdrant-client while httpx is establishing its connection
    permanently leaks one connection-pool slot (httpx 0.28), and enough leaked
    slots wedge every later request until the process restarts — the Stage 9
    r2 incident where all vector traffic stopped mid-run. Instead the detached
    task unwinds on qdrant-client's own client timeout (a clean error path),
    its outcome is consumed by a done callback, and this call raises
    ``DEADLINE_EXCEEDED`` immediately so the caller stays fail-closed.
    """

    remaining_seconds = deadline.remaining_seconds(reserve_finalization=True)
    if remaining_seconds <= 0:
        raise RAGPlanError(
            ErrorCode.DEADLINE_EXCEEDED,
            _DEADLINE_MESSAGE,
            timeout_origin=TimeoutOrigin.APPLICATION_DEADLINE,
        )
    if inflight is not None:
        while len(inflight) >= _MAX_ABANDONED_BACKEND_TASKS:
            stale = inflight.popleft()
            stale.cancel()

    task: asyncio.Task[T] = asyncio.create_task(operation_factory())  # type: ignore[arg-type]
    try:
        done, _pending = await asyncio.wait({task}, timeout=remaining_seconds)
    except BaseException:
        # The caller itself was cancelled (request teardown): keep nothing dangling.
        task.cancel()
        raise
    if not done:
        if inflight is not None:
            inflight.append(cast("asyncio.Task[object]", task))
            task.add_done_callback(_consume_finished_task)
        raise RAGPlanError(
            ErrorCode.DEADLINE_EXCEEDED,
            _DEADLINE_MESSAGE,
            timeout_origin=TimeoutOrigin.APPLICATION_DEADLINE,
        )
    return task.result()


def _is_qdrant_client_timeout(error: Exception) -> bool:
    source = error.source if isinstance(error, ResponseHandlingException) else error
    if isinstance(source, TimeoutError):
        return True
    return any(
        candidate.__module__.split(".", 1)[0] == "httpx"
        and candidate.__name__ == "TimeoutException"
        for candidate in type(source).__mro__
    )


def _retrieval_hit(
    point: models.ScoredPoint,
    *,
    corpus_version: str,
    rank: int,
) -> RetrievalHit:
    payload = _payload_mapping(point.payload)
    stored_version = _required_string(payload, "corpus_version")
    canonical_chunk_id = _required_string(payload, "canonical_chunk_id")
    document_id = _required_string(payload, "document_id")
    text = _required_string(payload, "text")
    position = payload.get("position")
    if isinstance(position, bool) or not isinstance(position, int) or position < 0:
        raise RAGPlanError(ErrorCode.CORPUS_INCONSISTENT, _CORPUS_DATA_MESSAGE)
    if stored_version != corpus_version or not math.isfinite(point.score):
        raise RAGPlanError(ErrorCode.CORPUS_INCONSISTENT, _CORPUS_DATA_MESSAGE)
    try:
        expected_point_id = qdrant_point_id(canonical_chunk_id)
        actual_point_id = UUID(str(point.id))
    except (TypeError, ValueError) as exc:
        raise RAGPlanError(ErrorCode.CORPUS_INCONSISTENT, _CORPUS_DATA_MESSAGE) from exc
    if actual_point_id != expected_point_id:
        raise RAGPlanError(ErrorCode.CORPUS_INCONSISTENT, _CORPUS_DATA_MESSAGE)
    return RetrievalHit(
        canonical_chunk_id=canonical_chunk_id,
        text=text,
        score=point.score,
        source="vector",
        document_id=document_id,
        metadata={"corpus_version": stored_version, "position": position},
        rank=rank,
    )


def _payload_mapping(payload: Mapping[str, object] | None) -> Mapping[str, object]:
    if payload is None or not isinstance(payload, Mapping):
        raise RAGPlanError(ErrorCode.CORPUS_INCONSISTENT, _CORPUS_DATA_MESSAGE)
    return payload


def _required_string(payload: Mapping[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise RAGPlanError(ErrorCode.CORPUS_INCONSISTENT, _CORPUS_DATA_MESSAGE)
    return value


def _corpus_filter(corpus_version: str) -> models.Filter:
    return models.Filter(
        must=[
            models.FieldCondition(
                key="corpus_version",
                match=models.MatchValue(value=corpus_version),
            )
        ]
    )


async def _count_version(
    client: AsyncQdrantClient,
    collection_name: str,
    corpus_version: str,
) -> int:
    result = await client.count(
        collection_name=collection_name,
        count_filter=_corpus_filter(corpus_version),
        exact=True,
    )
    return result.count


async def _count_all(client: AsyncQdrantClient, collection_name: str) -> int:
    result = await client.count(collection_name=collection_name, exact=True)
    return result.count


async def _read_vector_provenance(
    client: AsyncQdrantClient,
    collection_name: str,
    corpus_version: str,
    *,
    require_sealed: bool = True,
) -> dict[str, _StoredVectorProvenance]:
    provenance: dict[str, _StoredVectorProvenance] = {}
    offset: int | str | UUID | None = None
    seen_offsets: set[int | str | UUID] = set()
    while True:
        records, next_offset = await client.scroll(
            collection_name=collection_name,
            scroll_filter=_corpus_filter(corpus_version),
            limit=256,
            offset=offset,
            with_payload=[
                "corpus_version",
                "canonical_chunk_id",
                "embedding_artifact_manifest_sha256",
                "embedding_checksum",
            ],
            with_vectors=True,
        )
        for record in records:
            payload = _payload_mapping(record.payload)
            stored_version = _required_string(payload, "corpus_version")
            canonical_id = _required_string(payload, "canonical_chunk_id")
            artifact_sha = _required_string(payload, "embedding_artifact_manifest_sha256")
            declared_vector_sha = _required_string(payload, "embedding_checksum")
            try:
                _validate_sha256(
                    artifact_sha,
                    field_name="embedding_artifact_manifest_sha256",
                )
                _validate_sha256(declared_vector_sha, field_name="embedding_checksum")
            except ValueError as exc:
                raise RAGPlanError(
                    ErrorCode.CORPUS_INCONSISTENT,
                    _CORPUS_DATA_MESSAGE,
                ) from exc
            if stored_version != corpus_version:
                raise RAGPlanError(ErrorCode.CORPUS_INCONSISTENT, _CORPUS_DATA_MESSAGE)
            try:
                expected_point_id = qdrant_point_id(canonical_id)
                actual_point_id = UUID(str(record.id))
            except (TypeError, ValueError) as exc:
                raise RAGPlanError(
                    ErrorCode.CORPUS_INCONSISTENT,
                    _CORPUS_DATA_MESSAGE,
                ) from exc
            if actual_point_id != expected_point_id or canonical_id in provenance:
                raise RAGPlanError(ErrorCode.CORPUS_INCONSISTENT, _CORPUS_DATA_MESSAGE)
            vector = _stored_vector(record.vector)
            actual_vector_sha = _embedding_vector_checksum(vector)
            if require_sealed and declared_vector_sha != actual_vector_sha:
                raise RAGPlanError(ErrorCode.CORPUS_INCONSISTENT, _CORPUS_DATA_MESSAGE)
            provenance[canonical_id] = _StoredVectorProvenance(
                artifact_sha256=artifact_sha,
                declared_embedding_sha256=declared_vector_sha,
                actual_embedding_sha256=actual_vector_sha,
                vector=vector,
            )

        if next_offset is None:
            return provenance
        comparable_offset = _comparable_offset(next_offset)
        if comparable_offset in seen_offsets:
            raise RAGPlanError(ErrorCode.CORPUS_INCONSISTENT, _CORPUS_DATA_MESSAGE)
        seen_offsets.add(comparable_offset)
        offset = comparable_offset


def _validate_embedding(embedding: Sequence[float]) -> list[float]:
    if len(embedding) != VECTOR_SIZE:
        raise ValueError(f"embedding must contain exactly {VECTOR_SIZE} values")
    vector: list[float] = []
    for value in embedding:
        if isinstance(value, bool):
            raise ValueError("embedding values must be finite numbers")
        try:
            converted = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("embedding values must be finite numbers") from exc
        if not math.isfinite(converted):
            raise ValueError("embedding values must be finite numbers")
        vector.append(converted)
    return vector


def _embedding_vector_checksum(embedding: Sequence[float]) -> str:
    digest = hashlib.sha256()
    for value in embedding:
        digest.update(struct.pack(">d", float(value)))
    return digest.hexdigest()


def _stored_vector(value: object) -> tuple[float, ...]:
    if isinstance(value, (str, bytes, Mapping)) or not isinstance(value, Sequence):
        raise RAGPlanError(ErrorCode.CORPUS_INCONSISTENT, _CORPUS_DATA_MESSAGE)
    try:
        return tuple(_validate_unit_embedding(cast(Sequence[float], value)))
    except ValueError as exc:
        raise RAGPlanError(ErrorCode.CORPUS_INCONSISTENT, _CORPUS_DATA_MESSAGE) from exc


def _validate_unit_embedding(embedding: Sequence[float]) -> list[float]:
    vector = _validate_embedding(embedding)
    norm = math.sqrt(sum(value * value for value in vector))
    if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1e-5):
        raise ValueError("embedding must be L2-normalized")
    return vector


def _require_expected_vectors(
    observed: Mapping[str, _StoredVectorProvenance],
    expected: Mapping[str, tuple[float, ...]],
    artifact_sha256: str,
) -> None:
    if set(observed) != set(expected):
        raise RAGPlanError(ErrorCode.CORPUS_INCONSISTENT, _CORPUS_DATA_MESSAGE)
    for canonical_id, stored in observed.items():
        expected_vector = expected[canonical_id]
        if stored.artifact_sha256 != artifact_sha256 or any(
            not math.isclose(actual, intended, rel_tol=1e-6, abs_tol=1e-6)
            for actual, intended in zip(stored.vector, expected_vector, strict=True)
        ):
            raise RAGPlanError(ErrorCode.CORPUS_INCONSISTENT, _CORPUS_DATA_MESSAGE)


def _validate_sha256(value: str, *, field_name: str) -> None:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a lowercase SHA-256 hex digest")


def _validate_top_k(top_k: int) -> None:
    if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 50:
        raise ValueError("top_k must be an integer between 1 and 50")


def _validate_corpus_version(corpus_version: str) -> None:
    if not isinstance(corpus_version, str) or not corpus_version.strip():
        raise ValueError("corpus_version must be a non-empty string")
    if corpus_version != corpus_version.strip():
        raise ValueError("corpus_version must not have surrounding whitespace")


def _comparable_offset(value: object) -> int | str | UUID:
    if isinstance(value, bool) or not isinstance(value, (int, str, UUID)):
        raise RAGPlanError(ErrorCode.CORPUS_INCONSISTENT, _CORPUS_DATA_MESSAGE)
    return value


__all__ = [
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_COLLECTION_PREFIX",
    "VECTOR_SIZE",
    "QdrantCollectionManager",
    "QdrantVectorBackend",
    "QdrantVectorConfig",
    "QdrantVectorWriter",
    "canonical_id_checksum",
    "collection_name_for_version",
    "embedding_set_checksum",
]
