"""Unit tests for the Qdrant Stage 3 vector storage slice."""

from __future__ import annotations

import asyncio
import hashlib
from types import SimpleNamespace
import math
from collections.abc import Sequence
from uuid import UUID

import httpx
import pytest
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import ResponseHandlingException

from ragplan.backends.vector.qdrant import (
    VECTOR_SIZE,
    QdrantCollectionManager,
    QdrantVectorBackend,
    QdrantVectorConfig,
    QdrantVectorWriter,
    canonical_id_checksum,
    collection_name_for_version,
)
from ragplan.core.deadline import Deadline, ManualClock
from ragplan.core.errors import ErrorCode, RAGPlanError, TimeoutOrigin
from ragplan.core.ids import canonical_chunk_id, canonical_document_id, qdrant_point_id
from ragplan.core.models import Chunk

pytestmark = pytest.mark.unit


def _vector(axis: int = 0) -> list[float]:
    result = [0.0] * VECTOR_SIZE
    result[axis] = 1.0
    return result


def _chunk(position: int, text: str, *, corpus_version: str = "corpus-v1") -> Chunk:
    document_id = canonical_document_id("unit", "document-1")
    return Chunk(
        id=canonical_chunk_id(document_id, position, text),
        document_id=document_id,
        corpus_version=corpus_version,
        position=position,
        text=text,
        token_count=len(text.split()),
    )


def _local_manager(
    client: AsyncQdrantClient,
    *,
    prefix: str = "unit_chunks",
    batch_size: int = 64,
) -> QdrantCollectionManager:
    return QdrantCollectionManager(
        client,
        QdrantVectorConfig(
            collection_prefix=prefix,
            batch_size=batch_size,
            require_payload_indexes=False,
        ),
    )


def test_collection_name_and_checksum_are_deterministic_and_order_independent() -> None:
    expected_digest = hashlib.sha256(b"corpus-v1").hexdigest()

    assert collection_name_for_version("chunks", "corpus-v1") == f"chunks_{expected_digest}"
    assert canonical_id_checksum(("id-b", "id-a")) == canonical_id_checksum(("id-a", "id-b"))
    assert canonical_id_checksum(()) == hashlib.sha256(b"").hexdigest()


@pytest.mark.parametrize(
    "config",
    [
        {"collection_prefix": "bad prefix"},
        {"vector_size": 383},
        {"batch_size": 0},
        {"batch_size": True},
    ],
)
def test_config_rejects_non_stage3_schema(config: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        QdrantVectorConfig(**config)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_writer_is_idempotent_and_uses_uuid_payload_contract() -> None:
    client = AsyncQdrantClient(location=":memory:")
    manager = _local_manager(client, batch_size=1)
    writer = QdrantVectorWriter(manager)
    chunks = (_chunk(0, "alpha evidence"), _chunk(1, "beta evidence"))
    embeddings: Sequence[Sequence[float]] = (_vector(0), _vector(1))
    try:
        first = await writer.write_chunks(
            chunks,
            embeddings,
            "corpus-v1",
            embedding_artifact_manifest_sha256="b" * 64,
        )
        second = await writer.write_chunks(
            chunks,
            embeddings,
            "corpus-v1",
            embedding_artifact_manifest_sha256="b" * 64,
        )
        stage = await writer.stage_chunks(
            chunks,
            embeddings,
            "corpus-v1",
            embedding_artifact_manifest_sha256="b" * 64,
        )

        collection_name = manager.collection_name("corpus-v1")
        count = await client.count(collection_name=collection_name, exact=True)
        records, next_offset = await client.scroll(
            collection_name=collection_name,
            limit=10,
            with_payload=True,
            with_vectors=False,
        )

        assert first == second
        assert stage.collection_name == collection_name
        assert stage.chunk_count == 2
        assert stage.canonical_id_checksum == first.canonical_id_checksum
        assert await manager.verify_stage(stage) is stage
        assert first.written_count == 2
        assert first.canonical_id_checksum == canonical_id_checksum(
            tuple(chunk.canonical_chunk_id for chunk in chunks)
        )
        assert count.count == 2
        assert next_offset is None
        assert {UUID(str(record.id)) for record in records} == {
            qdrant_point_id(chunk.canonical_chunk_id) for chunk in chunks
        }
        assert {record.payload["canonical_chunk_id"] for record in records if record.payload} == {
            chunk.canonical_chunk_id for chunk in chunks
        }
        assert all(
            record.payload and record.payload["corpus_version"] == "corpus-v1" for record in records
        )
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_stage_verification_rejects_count_or_checksum_drift() -> None:
    client = AsyncQdrantClient(location=":memory:")
    manager = _local_manager(client, prefix="manifest_chunks")
    writer = QdrantVectorWriter(manager)
    chunk = _chunk(0, "manifest evidence")
    try:
        stage = await writer.stage_chunks(
            (chunk,),
            (_vector(),),
            "corpus-v1",
            embedding_artifact_manifest_sha256="b" * 64,
        )

        with pytest.raises(RAGPlanError) as captured:
            await manager.verify_stage(stage.model_copy(update={"chunk_count": 2}))

        assert captured.value.code is ErrorCode.CORPUS_INCONSISTENT
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_explicit_version_discard_is_idempotent() -> None:
    client = AsyncQdrantClient(location=":memory:")
    manager = _local_manager(client, prefix="discard_chunks")
    writer = QdrantVectorWriter(manager)
    try:
        await writer.stage_chunks(
            (_chunk(0, "discardable evidence"),),
            (_vector(),),
            "corpus-v1",
            embedding_artifact_manifest_sha256="b" * 64,
        )
        assert await client.collection_exists(manager.collection_name("corpus-v1"))

        await manager.discard_version("corpus-v1")
        await manager.discard_version("corpus-v1")

        assert not await client.collection_exists(manager.collection_name("corpus-v1"))
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_writer_rejects_invalid_dimensions_and_version_mutation() -> None:
    client = AsyncQdrantClient(location=":memory:")
    writer = QdrantVectorWriter(_local_manager(client))
    chunk = _chunk(0, "alpha evidence")
    try:
        with pytest.raises(ValueError, match="384"):
            await writer.write_chunks(
                (chunk,),
                ([0.0] * 383,),
                "corpus-v1",
                embedding_artifact_manifest_sha256="a" * 64,
            )
        with pytest.raises(ValueError, match="L2-normalized"):
            await writer.write_chunks(
                (chunk,),
                ([1.0] * VECTOR_SIZE,),
                "corpus-v1",
                embedding_artifact_manifest_sha256="a" * 64,
            )
        with pytest.raises(ValueError, match="corpus_version"):
            await writer.write_chunks(
                (chunk,),
                (_vector(),),
                "corpus-v2",
                embedding_artifact_manifest_sha256="a" * 64,
            )
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_same_version_changed_chunk_set_is_rejected_before_mutation() -> None:
    client = AsyncQdrantClient(location=":memory:")
    manager = _local_manager(client, prefix="immutable_chunks")
    writer = QdrantVectorWriter(manager)
    original = _chunk(0, "original evidence")
    replacement = _chunk(1, "replacement evidence")
    try:
        await writer.write_chunks(
            (original,),
            (_vector(),),
            "corpus-v1",
            embedding_artifact_manifest_sha256="a" * 64,
        )

        with pytest.raises(RAGPlanError) as captured:
            await writer.write_chunks(
                (replacement,),
                (_vector(1),),
                "corpus-v1",
                embedding_artifact_manifest_sha256="a" * 64,
            )

        records, _ = await client.scroll(
            collection_name=manager.collection_name("corpus-v1"),
            limit=10,
            with_payload=True,
            with_vectors=False,
        )
        assert captured.value.code is ErrorCode.CORPUS_INCONSISTENT
        assert len(records) == 1
        assert records[0].payload
        assert records[0].payload["canonical_chunk_id"] == original.canonical_chunk_id
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_same_version_changed_embedding_provenance_is_rejected_before_mutation() -> None:
    client = AsyncQdrantClient(location=":memory:")
    manager = _local_manager(client, prefix="immutable_embedding_chunks")
    writer = QdrantVectorWriter(manager)
    chunk = _chunk(0, "immutable embedding evidence")
    try:
        original_stage = await writer.stage_chunks(
            (chunk,),
            (_vector(0),),
            "corpus-v1",
            embedding_artifact_manifest_sha256="a" * 64,
        )

        with pytest.raises(RAGPlanError) as captured:
            await writer.stage_chunks(
                (chunk,),
                (_vector(1),),
                "corpus-v1",
                embedding_artifact_manifest_sha256="b" * 64,
            )

        records, _ = await client.scroll(
            collection_name=manager.collection_name("corpus-v1"),
            limit=10,
            with_payload=True,
            with_vectors=True,
        )
        assert captured.value.code is ErrorCode.CORPUS_INCONSISTENT
        assert len(records) == 1
        assert records[0].payload
        assert records[0].payload["embedding_artifact_manifest_sha256"] == "a" * 64
        assert records[0].vector == _vector(0)
        assert await manager.verify_stage(original_stage) is original_stage

        for field in ("embedding_artifact_manifest_sha256", "embedding_set_checksum"):
            with pytest.raises(RAGPlanError) as drift:
                await manager.verify_stage(original_stage.model_copy(update={field: "c" * 64}))
            assert drift.value.code is ErrorCode.CORPUS_INCONSISTENT
    finally:
        await client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tampered_vector",
    [
        _vector(1),
        [math.sqrt(1.0 - 1e-14), 1e-7, *([0.0] * (VECTOR_SIZE - 2))],
    ],
)
async def test_stage_verification_rejects_vector_tampering_with_unchanged_payload(
    tampered_vector: Sequence[float],
) -> None:
    client = AsyncQdrantClient(location=":memory:")
    manager = _local_manager(client, prefix="tampered_vector_chunks")
    writer = QdrantVectorWriter(manager)
    chunk = _chunk(0, "tamper-evident vector")
    try:
        stage = await writer.stage_chunks(
            (chunk,),
            (_vector(0),),
            "corpus-v1",
            embedding_artifact_manifest_sha256="a" * 64,
        )
        records, _ = await client.scroll(
            collection_name=stage.collection_name,
            limit=1,
            with_payload=True,
            with_vectors=False,
        )
        assert records[0].payload
        await client.upsert(
            collection_name=stage.collection_name,
            points=[
                models.PointStruct(
                    id=qdrant_point_id(chunk.canonical_chunk_id),
                    vector=tampered_vector,
                    payload=records[0].payload,
                )
            ],
            wait=True,
        )

        with pytest.raises(RAGPlanError) as verification:
            await manager.verify_stage(stage)
        with pytest.raises(RAGPlanError) as restaging:
            await writer.stage_chunks(
                (chunk,),
                (_vector(0),),
                "corpus-v1",
                embedding_artifact_manifest_sha256="a" * 64,
            )

        assert verification.value.code is ErrorCode.CORPUS_INCONSISTENT
        assert restaging.value.code is ErrorCode.CORPUS_INCONSISTENT
        after_records, _ = await client.scroll(
            collection_name=stage.collection_name,
            limit=1,
            with_payload=True,
            with_vectors=False,
        )
        assert after_records[0].payload
        assert (
            after_records[0].payload["embedding_checksum"]
            == records[0].payload["embedding_checksum"]
        )
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_existing_vector_schema_mismatch_fails_fast() -> None:
    client = AsyncQdrantClient(location=":memory:")
    manager = _local_manager(client, prefix="schema_chunks")
    collection_name = manager.collection_name("corpus-v1")
    try:
        await client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=32, distance=models.Distance.DOT),
        )

        with pytest.raises(RAGPlanError) as captured:
            await manager.ensure_collection("corpus-v1")

        assert captured.value.code is ErrorCode.MODEL_INCOMPATIBLE
        assert "model" in captured.value.message
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_existing_collection_without_v2_version_metadata_fails_fast() -> None:
    client = AsyncQdrantClient(location=":memory:")
    manager = _local_manager(client, prefix="metadata_chunks")
    collection_name = manager.collection_name("corpus-v1")
    try:
        await client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=VECTOR_SIZE, distance=models.Distance.COSINE),
            metadata={"ragplan_schema": "vector-v1"},
        )

        with pytest.raises(RAGPlanError) as captured:
            await manager.ensure_collection("corpus-v1")

        assert captured.value.code is ErrorCode.CORPUS_INCONSISTENT
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_search_preserves_qdrant_order_and_maps_payload() -> None:
    client = AsyncQdrantClient(location=":memory:")
    manager = _local_manager(client, prefix="search_chunks")
    writer = QdrantVectorWriter(manager)
    backend = QdrantVectorBackend(manager)
    chunks = (_chunk(0, "relevant alpha"), _chunk(1, "unrelated beta"))
    try:
        await writer.write_chunks(
            chunks,
            (_vector(0), _vector(1)),
            "corpus-v1",
            embedding_artifact_manifest_sha256="a" * 64,
        )

        hits = await backend.search(
            _vector(0),
            top_k=2,
            corpus_version="corpus-v1",
            deadline=Deadline.start(5000),
        )

        assert [hit.canonical_chunk_id for hit in hits] == [
            chunks[0].canonical_chunk_id,
            chunks[1].canonical_chunk_id,
        ]
        assert [hit.rank for hit in hits] == [1, 2]
        assert hits[0].document_id == chunks[0].document_id
        assert hits[0].source == "vector"
        assert hits[0].metadata == {"corpus_version": "corpus-v1", "position": 0}
        assert hits[0].score > hits[1].score
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_empty_collection_is_a_successful_zero_hit_search() -> None:
    client = AsyncQdrantClient(location=":memory:")
    manager = _local_manager(client, prefix="empty_chunks")
    backend = QdrantVectorBackend(manager)
    try:
        await manager.ensure_collection("corpus-v1")

        hits = await backend.search(
            _vector(),
            top_k=10,
            corpus_version="corpus-v1",
            deadline=Deadline.start(5000),
        )

        assert hits == ()
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_nonexistent_collection_is_a_stable_dependency_failure() -> None:
    client = AsyncQdrantClient(location=":memory:")
    backend = QdrantVectorBackend(_local_manager(client, prefix="missing_chunks"))
    try:
        with pytest.raises(RAGPlanError) as captured:
            await backend.search(
                _vector(),
                top_k=10,
                corpus_version="missing-v1",
                deadline=Deadline.start(5000),
            )

        assert captured.value.code is ErrorCode.DEPENDENCY_UNAVAILABLE
        assert captured.value.message == "vector collection is unavailable"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_corpus_versions_are_isolated_by_collection_and_filter() -> None:
    client = AsyncQdrantClient(location=":memory:")
    manager = _local_manager(client, prefix="version_chunks")
    writer = QdrantVectorWriter(manager)
    backend = QdrantVectorBackend(manager)
    first = _chunk(0, "first version", corpus_version="corpus-v1")
    second = _chunk(0, "second version", corpus_version="corpus-v2")
    try:
        await writer.write_chunks(
            (first,),
            (_vector(),),
            "corpus-v1",
            embedding_artifact_manifest_sha256="a" * 64,
        )
        await writer.write_chunks(
            (second,),
            (_vector(),),
            "corpus-v2",
            embedding_artifact_manifest_sha256="a" * 64,
        )

        first_hits = await backend.search(_vector(), 1, "corpus-v1", Deadline.start(5000))
        second_hits = await backend.search(_vector(), 1, "corpus-v2", Deadline.start(5000))

        assert [hit.text for hit in first_hits] == ["first version"]
        assert [hit.text for hit in second_hits] == ["second version"]
        assert manager.collection_name("corpus-v1") != manager.collection_name("corpus-v2")
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_invalid_top_k_and_expired_deadline_make_no_search_request() -> None:
    client = AsyncQdrantClient(location=":memory:")
    manager = _local_manager(client, prefix="deadline_chunks")
    backend = QdrantVectorBackend(manager)
    clock = ManualClock()
    deadline = Deadline.start(100, clock=clock)
    clock.advance_ms(95)
    try:
        with pytest.raises(ValueError, match="between 1 and 50"):
            await backend.search(_vector(), 0, "corpus-v1", Deadline.start(5000))
        with pytest.raises(RAGPlanError) as captured:
            await backend.search(_vector(), 10, "corpus-v1", deadline)

        assert captured.value.code is ErrorCode.DEADLINE_EXCEEDED
        assert await client.collection_exists(manager.collection_name("corpus-v1")) is False
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_malformed_qdrant_payload_is_reported_as_corpus_inconsistency() -> None:
    client = AsyncQdrantClient(location=":memory:")
    manager = _local_manager(client, prefix="malformed_chunks")
    backend = QdrantVectorBackend(manager)
    chunk = _chunk(0, "malformed evidence")
    try:
        collection_name = await manager.ensure_collection("corpus-v1")
        await client.upsert(
            collection_name=collection_name,
            points=[
                models.PointStruct(
                    id=qdrant_point_id(chunk.canonical_chunk_id),
                    vector=_vector(),
                    payload={
                        "corpus_version": "corpus-v1",
                        "canonical_chunk_id": chunk.canonical_chunk_id,
                        "document_id": chunk.document_id,
                        "position": 0,
                    },
                )
            ],
            wait=True,
        )

        with pytest.raises(RAGPlanError) as captured:
            await backend.search(_vector(), 1, "corpus-v1", Deadline.start(5000))

        assert captured.value.code is ErrorCode.CORPUS_INCONSISTENT
        assert captured.value.retryable is False
        assert "inconsistent corpus data" in captured.value.message
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_health_and_close_contract() -> None:
    client = AsyncQdrantClient(location=":memory:")
    backend = QdrantVectorBackend(_local_manager(client, prefix="health_chunks"))

    health = await backend.health()
    await backend.close()
    await backend.close()

    assert health.available is True


@pytest.mark.asyncio
async def test_qdrant_transport_timeout_is_typed_as_backend_client_timeout() -> None:
    class TimeoutClient:
        async def query_points(self, **kwargs: object) -> None:
            del kwargs
            raise ResponseHandlingException(httpx.ReadTimeout("transport timeout"))

    class OnlineManager:
        client = TimeoutClient()
        config = QdrantVectorConfig()

        async def require_collection(self, corpus_version: str) -> str:
            del corpus_version
            return "collection"

        async def recycle_client(self) -> None:
            return None

    backend = QdrantVectorBackend(OnlineManager())  # type: ignore[arg-type]

    with pytest.raises(RAGPlanError) as captured:
        await backend.search(_vector(), 1, "corpus-v1", Deadline.start(200))

    assert captured.value.code is ErrorCode.DEADLINE_EXCEEDED
    assert captured.value.timeout_origin is TimeoutOrigin.BACKEND_CLIENT


@pytest.mark.asyncio
async def test_deadline_expiration_does_not_cancel_the_inflight_backend_task() -> None:
    """Detached execution: deadline misses must never cancel qdrant-client.

    Cancelling httpx during connection setup permanently leaks a connection
    pool slot, which is how Stage 9 r2 wedged all vector traffic; the backend
    must instead detach the task and fail closed immediately.
    """

    outcomes: list[str] = []

    class HangingClient:
        async def query_points(self, **kwargs: object) -> object:
            try:
                await asyncio.sleep(0.3)
                outcomes.append("completed-cleanly")
                return models.QueryResponse(points=[], scored_points=[])
            except asyncio.CancelledError:
                outcomes.append("cancelled")
                raise

    class HangingManager:
        config = QdrantVectorConfig()
        client = HangingClient()

        async def require_collection(self, corpus_version: str) -> str:
            del corpus_version
            return "collection"

    backend = QdrantVectorBackend(HangingManager())  # type: ignore[arg-type]

    with pytest.raises(RAGPlanError) as captured:
        await backend.search(_vector(), 1, "corpus-v1", Deadline.start(40))

    assert captured.value.code is ErrorCode.DEADLINE_EXCEEDED
    assert captured.value.timeout_origin is TimeoutOrigin.APPLICATION_DEADLINE
    await asyncio.sleep(0.4)
    assert outcomes == ["completed-cleanly"]


@pytest.mark.asyncio
async def test_abandoned_tasks_are_tracked_and_pruned() -> None:
    """Abandoned inflight tasks stay bounded so pools cannot pile up."""

    class SlowClient:
        def __init__(self) -> None:
            self.release: asyncio.Event | None = None

        async def query_points(self, **kwargs: object) -> object:
            await asyncio.sleep(5)
            raise AssertionError("unreachable")

    class SlowManager:
        config = QdrantVectorConfig()
        client = SlowClient()

        async def require_collection(self, corpus_version: str) -> str:
            del corpus_version
            return "collection"

    manager = SlowManager()
    backend = QdrantVectorBackend(manager)  # type: ignore[arg-type]

    for _ in range(3):
        with pytest.raises(RAGPlanError):
            await backend.search(_vector(), 1, "corpus-v1", Deadline.start(30))

    assert len(backend._inflight) == 3

    for task in list(backend._inflight):
        task.cancel()
    await asyncio.gather(*list(backend._inflight), return_exceptions=True)

    assert all(task.done() for task in backend._inflight)


@pytest.mark.asyncio
async def test_require_collection_caches_validation_round_trips() -> None:
    """Warm serving path performs zero exists/get metadata round trips."""

    calls: list[str] = []

    class CountingClient:
        async def collection_exists(self, collection_name: str) -> bool:
            calls.append(f"exists:{collection_name}")
            return True

        async def get_collection(self, collection_name: str) -> object:
            calls.append(f"get:{collection_name}")
            return SimpleNamespace(
                config=SimpleNamespace(
                    params=SimpleNamespace(
                        vectors=models.VectorParams(size=VECTOR_SIZE, distance=models.Distance.COSINE),
                    ),
                    metadata={
                        "ragplan_schema": "vector-v2",
                        "corpus_version_sha256": hashlib.sha256(b"corpus-v1").hexdigest(),
                    },
                ),
                payload_schema={},
            )

    manager = QdrantCollectionManager(
        CountingClient(),  # type: ignore[arg-type]
        QdrantVectorConfig(collection_prefix="cache_chunks", require_payload_indexes=False),
    )

    first = await manager.require_collection("corpus-v1")
    second = await manager.require_collection("corpus-v1")

    assert first == second == collection_name_for_version("cache_chunks", "corpus-v1")
    assert len(calls) == 2  # one exists + one get, then cache serves the rest

    manager.invalidate_validation()
    await manager.require_collection("corpus-v1")
    assert len(calls) == 4


@pytest.mark.asyncio
async def test_hnsw_ef_is_forwarded_as_search_params() -> None:
    """Configured HNSW ef reaches query_points as SearchParams."""

    seen_params: list[object] = []

    class ParamsClient:
        async def query_points(self, **kwargs: object) -> object:
            seen_params.append(kwargs.get("search_params"))
            return models.QueryResponse(points=[], scored_points=[])

    class ParamsManager:
        config = QdrantVectorConfig(hnsw_ef=64)
        client = ParamsClient()

        async def require_collection(self, corpus_version: str) -> str:
            del corpus_version
            return "collection"

        async def recycle_client(self) -> None:
            return None

    backend = QdrantVectorBackend(ParamsManager())  # type: ignore[arg-type]
    await backend.search(_vector(), 1, "corpus-v1", Deadline.start(200))

    assert isinstance(seen_params[0], models.SearchParams)
    assert seen_params[0].hnsw_ef == 64
