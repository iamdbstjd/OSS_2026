"""Live dual-store reconciliation and atomic corpus activation."""

from __future__ import annotations

import hashlib
from typing import Protocol

from ragplan.core.errors import ErrorCode, RAGPlanError
from ragplan.core.models import (
    ActivationStatus,
    FrozenModel,
    GraphStageManifest,
    IngestionManifest,
    IngestionStoreStatus,
    NonEmptyString,
    Sha256Hex,
    VectorStageManifest,
)
from ragplan.ingestion.manifest import ActiveCorpusPointer, ManifestRepository


class IngestionSource(FrozenModel):
    source_dataset: NonEmptyString
    source_version: NonEmptyString
    source_sha256: Sha256Hex
    chunker_version: NonEmptyString


class VectorStageVerifier(Protocol):
    async def verify_stage(self, manifest: VectorStageManifest) -> VectorStageManifest: ...


class GraphStageVerifier(Protocol):
    async def verify_stage(self, manifest: GraphStageManifest) -> GraphStageManifest: ...


def reconcile_stage_manifests(
    *,
    ingestion_run_id: str,
    source: IngestionSource,
    vector: VectorStageManifest,
    graph: GraphStageManifest,
) -> IngestionManifest:
    """Build an active candidate only when both immutable stage records agree."""

    if vector.corpus_version != graph.corpus_version:
        raise RAGPlanError(
            ErrorCode.CORPUS_INCONSISTENT,
            "vector and graph stages reference different corpus versions",
            retryable=False,
        )
    if vector.chunker_version.value != source.chunker_version:
        raise RAGPlanError(
            ErrorCode.CORPUS_INCONSISTENT,
            "vector stage and activation source reference different chunker versions",
            retryable=False,
        )
    if (
        vector.chunk_count != graph.chunk_count
        or vector.canonical_id_checksum != graph.canonical_id_checksum
    ):
        raise RAGPlanError(
            ErrorCode.CORPUS_INCONSISTENT,
            "vector and graph canonical chunk IDs did not reconcile",
            retryable=False,
        )
    return IngestionManifest(
        ingestion_run_id=ingestion_run_id,
        corpus_version=vector.corpus_version,
        source_dataset=source.source_dataset,
        source_version=source.source_version,
        source_sha256=source.source_sha256,
        chunker_version=source.chunker_version,
        embedding_model_revision=vector.embedding_model_revision,
        extractor_version=graph.extractor_version,
        document_count=graph.document_count,
        chunk_count=graph.chunk_count,
        qdrant_count=vector.chunk_count,
        qdrant_id_checksum=vector.canonical_id_checksum,
        qdrant_status=IngestionStoreStatus.SUCCEEDED,
        neo4j_count=graph.chunk_count,
        neo4j_id_checksum=graph.canonical_id_checksum,
        neo4j_status=IngestionStoreStatus.SUCCEEDED,
        activation_status=ActivationStatus.ACTIVE,
    )


class ActivationCoordinator:
    """Re-read both stores before changing the sole local serving pointer."""

    def __init__(
        self,
        *,
        vector_verifier: VectorStageVerifier,
        graph_verifier: GraphStageVerifier,
        repository: ManifestRepository,
    ) -> None:
        self._vector_verifier = vector_verifier
        self._graph_verifier = graph_verifier
        self._repository = repository

    async def activate(
        self,
        *,
        ingestion_run_id: str,
        source: IngestionSource,
        vector: VectorStageManifest,
        graph: GraphStageManifest,
    ) -> tuple[ActiveCorpusPointer, IngestionManifest]:
        verified_vector: VectorStageManifest | None = None
        verified_graph: GraphStageManifest | None = None
        verification_error: Exception | None = None
        try:
            verified_vector = await self._vector_verifier.verify_stage(vector)
        except Exception as exc:
            verification_error = exc
        try:
            verified_graph = await self._graph_verifier.verify_stage(graph)
        except Exception as exc:
            if verification_error is None:
                verification_error = exc
        if verification_error is not None:
            self._repository.record(
                _failed_manifest(
                    ingestion_run_id=ingestion_run_id,
                    source=source,
                    vector=vector,
                    graph=graph,
                    vector_verified=verified_vector is not None,
                    graph_verified=verified_graph is not None,
                )
            )
            raise verification_error
        assert verified_vector is not None
        assert verified_graph is not None
        try:
            manifest = reconcile_stage_manifests(
                ingestion_run_id=ingestion_run_id,
                source=source,
                vector=verified_vector,
                graph=verified_graph,
            )
        except Exception:
            self._repository.record(
                _failed_manifest(
                    ingestion_run_id=ingestion_run_id,
                    source=source,
                    vector=verified_vector,
                    graph=verified_graph,
                    vector_verified=True,
                    graph_verified=True,
                )
            )
            raise
        pointer = self._repository.activate(manifest)
        return pointer, manifest


def _failed_manifest(
    *,
    ingestion_run_id: str,
    source: IngestionSource,
    vector: VectorStageManifest,
    graph: GraphStageManifest,
    vector_verified: bool,
    graph_verified: bool,
) -> IngestionManifest:
    empty_checksum = hashlib.sha256(b"").hexdigest()
    return IngestionManifest(
        ingestion_run_id=ingestion_run_id,
        corpus_version=vector.corpus_version,
        source_dataset=source.source_dataset,
        source_version=source.source_version,
        source_sha256=source.source_sha256,
        chunker_version=source.chunker_version,
        embedding_model_revision=vector.embedding_model_revision,
        extractor_version=graph.extractor_version,
        document_count=graph.document_count,
        chunk_count=max(vector.chunk_count, graph.chunk_count),
        qdrant_count=vector.chunk_count if vector_verified else 0,
        qdrant_id_checksum=vector.canonical_id_checksum if vector_verified else empty_checksum,
        qdrant_status=(
            IngestionStoreStatus.SUCCEEDED if vector_verified else IngestionStoreStatus.FAILED
        ),
        neo4j_count=graph.chunk_count if graph_verified else 0,
        neo4j_id_checksum=graph.canonical_id_checksum if graph_verified else empty_checksum,
        neo4j_status=(
            IngestionStoreStatus.SUCCEEDED if graph_verified else IngestionStoreStatus.FAILED
        ),
        activation_status=ActivationStatus.FAILED,
    )
