from __future__ import annotations

import stat
from pathlib import Path

import pytest

from ragplan.core.errors import ErrorCode, RAGPlanError
from ragplan.core.models import (
    ActivationStatus,
    GraphStageManifest,
    IngestionStoreStatus,
    VectorStageManifest,
)
from ragplan.ingestion.manifest import ActiveCorpusResolver, ManifestRepository
from ragplan.ingestion.reconcile import ActivationCoordinator, IngestionSource

pytestmark = pytest.mark.integration


def _vector(version: str, checksum: str = "a" * 64) -> VectorStageManifest:
    return VectorStageManifest(
        corpus_version=version,
        collection_name=f"ragplan_{version}",
        chunk_count=2,
        canonical_id_checksum=checksum,
        embedding_set_checksum="b" * 64,
        embedding_artifact_manifest_sha256="c" * 64,
    )


def _graph(version: str, checksum: str = "a" * 64) -> GraphStageManifest:
    return GraphStageManifest(
        corpus_version=version,
        database="neo4j",
        document_count=1,
        chunk_count=2,
        entity_count=2,
        mention_count=2,
        relation_count=1,
        canonical_id_checksum=checksum,
        graph_content_checksum="d" * 64,
        extractor_version="graph-extractor-v1-fixture",
    )


class _VectorVerifier:
    async def verify_stage(self, manifest: VectorStageManifest) -> VectorStageManifest:
        return manifest


class _GraphVerifier:
    def __init__(self, *, fail: bool = False) -> None:
        self._fail = fail

    async def verify_stage(self, manifest: GraphStageManifest) -> GraphStageManifest:
        if self._fail:
            raise RAGPlanError(ErrorCode.DEPENDENCY_UNAVAILABLE, "graph unavailable")
        return manifest


def _source() -> IngestionSource:
    return IngestionSource(
        source_dataset="fixture",
        source_version="v1",
        source_sha256="e" * 64,
        chunker_version="token-window-220-overlap-40-v1",
    )


def _coordinator(root: Path, *, graph_fails: bool = False) -> ActivationCoordinator:
    return ActivationCoordinator(
        vector_verifier=_VectorVerifier(),
        graph_verifier=_GraphVerifier(fail=graph_fails),
        repository=ManifestRepository(root),
    )


@pytest.mark.asyncio
async def test_partial_dual_write_never_changes_active_pointer(tmp_path: Path) -> None:
    repository = ManifestRepository(tmp_path)
    coordinator = _coordinator(tmp_path)
    await coordinator.activate(
        ingestion_run_id="run-v1",
        source=_source(),
        vector=_vector("corpus-v1"),
        graph=_graph("corpus-v1"),
    )

    failing = _coordinator(tmp_path, graph_fails=True)
    with pytest.raises(RAGPlanError) as caught:
        await failing.activate(
            ingestion_run_id="run-v2",
            source=_source(),
            vector=_vector("corpus-v2"),
            graph=_graph("corpus-v2"),
        )

    assert caught.value.code is ErrorCode.DEPENDENCY_UNAVAILABLE
    assert ActiveCorpusResolver(repository).resolve() == "corpus-v1"
    assert stat.S_IMODE(repository.active_pointer_path.stat().st_mode) == 0o644
    failed = repository.load_run("run-v2")
    assert failed.activation_status is ActivationStatus.FAILED
    assert failed.qdrant_status is IngestionStoreStatus.SUCCEEDED
    assert failed.neo4j_status is IngestionStoreStatus.FAILED


@pytest.mark.asyncio
async def test_new_activation_and_old_version_rollback_are_atomic(tmp_path: Path) -> None:
    repository = ManifestRepository(tmp_path)
    coordinator = _coordinator(tmp_path)
    await coordinator.activate(
        ingestion_run_id="run-v1",
        source=_source(),
        vector=_vector("corpus-v1"),
        graph=_graph("corpus-v1"),
    )
    await coordinator.activate(
        ingestion_run_id="run-v2",
        source=_source(),
        vector=_vector("corpus-v2"),
        graph=_graph("corpus-v2"),
    )
    assert ActiveCorpusResolver(repository).resolve() == "corpus-v2"

    repository.rollback("run-v1")

    assert ActiveCorpusResolver(repository).resolve() == "corpus-v1"


@pytest.mark.asyncio
async def test_id_checksum_mismatch_rejects_activation(tmp_path: Path) -> None:
    with pytest.raises(RAGPlanError) as caught:
        await _coordinator(tmp_path).activate(
            ingestion_run_id="run-bad",
            source=_source(),
            vector=_vector("corpus-v1"),
            graph=_graph("corpus-v1", checksum="f" * 64),
        )

    assert caught.value.code is ErrorCode.CORPUS_INCONSISTENT
    assert not ManifestRepository(tmp_path).active_pointer_path.exists()
    failed = ManifestRepository(tmp_path).load_run("run-bad")
    assert failed.activation_status is ActivationStatus.FAILED
    assert failed.qdrant_status is IngestionStoreStatus.SUCCEEDED
    assert failed.neo4j_status is IngestionStoreStatus.SUCCEEDED
