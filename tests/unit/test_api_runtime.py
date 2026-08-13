"""Stage 3 API bootstrap remains explicit, verified, and injectable."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from ragplan.api import runtime
from ragplan.api.server import create_app
from ragplan.core.errors import ErrorCode, RAGPlanError
from ragplan.core.models import (
    ActivationStatus,
    GraphStageManifest,
    IngestionManifest,
    IngestionStoreStatus,
    VectorStageManifest,
)
from ragplan.ingestion.extractor_version import build_extractor_version
from ragplan.ingestion.manifest import ManifestRepository
from ragplan.ingestion.model_manifest import load_default_model_artifact_manifest

pytestmark = pytest.mark.unit


def _stage() -> VectorStageManifest:
    corpus_version = "api-runtime-v1"
    return VectorStageManifest(
        corpus_version=corpus_version,
        collection_name="api_runtime_" + hashlib.sha256(corpus_version.encode()).hexdigest(),
        chunk_count=0,
        canonical_id_checksum=hashlib.sha256(b"").hexdigest(),
        embedding_set_checksum=hashlib.sha256(b"").hexdigest(),
        embedding_artifact_manifest_sha256=load_default_model_artifact_manifest().sha256,
    )


def _environment(tmp_path: Path) -> dict[str, str]:
    stage_path = tmp_path / "stage.json"
    stage_path.write_text(_stage().model_dump_json(), encoding="utf-8")
    return {
        "RAGPLAN_STAGE3_MODEL_SNAPSHOT": str(tmp_path / "model"),
        "RAGPLAN_STAGE3_VECTOR_STAGE_MANIFEST": str(stage_path),
        "RAGPLAN_STAGE3_QDRANT_URL": "http://qdrant:6333",
        "RAGPLAN_STAGE3_QDRANT_COLLECTION_PREFIX": "api_runtime",
    }


def test_runtime_configuration_is_all_or_nothing(tmp_path: Path) -> None:
    assert runtime.Stage3RuntimeConfig.from_environment({}) is None

    incomplete = _environment(tmp_path)
    del incomplete["RAGPLAN_STAGE3_QDRANT_URL"]
    with pytest.raises(RAGPlanError) as captured:
        runtime.Stage3RuntimeConfig.from_environment(incomplete)
    assert captured.value.code is ErrorCode.INVALID_REQUEST


def _active_environment(tmp_path: Path, *, stage_version: str = "active-v1") -> dict[str, str]:
    checksum = "a" * 64
    stage = _stage().model_copy(
        update={
            "corpus_version": stage_version,
            "collection_name": "api_runtime_" + hashlib.sha256(stage_version.encode()).hexdigest(),
            "chunk_count": 1,
            "canonical_id_checksum": checksum,
        }
    )
    stage_path = tmp_path / "active-vector-stage.json"
    stage_path.write_text(stage.model_dump_json(), encoding="utf-8")
    repository = ManifestRepository(tmp_path / "ingestion")
    repository.activate(
        IngestionManifest(
            ingestion_run_id="active-run-v1",
            corpus_version="active-v1",
            source_dataset="fixture",
            source_version="v1",
            source_sha256="b" * 64,
            chunker_version="v1",
            embedding_model_revision=stage.embedding_model_revision,
            extractor_version="graph-v1",
            document_count=1,
            chunk_count=1,
            qdrant_count=1,
            qdrant_id_checksum=checksum,
            qdrant_status=IngestionStoreStatus.SUCCEEDED,
            neo4j_count=1,
            neo4j_id_checksum=checksum,
            neo4j_status=IngestionStoreStatus.SUCCEEDED,
            activation_status=ActivationStatus.ACTIVE,
        )
    )
    return {
        "RAGPLAN_STAGE4_MODEL_SNAPSHOT": str(tmp_path / "model"),
        "RAGPLAN_STAGE4_VECTOR_STAGE_MANIFEST": str(stage_path),
        "RAGPLAN_STAGE4_MANIFEST_ROOT": str(tmp_path / "ingestion"),
        "RAGPLAN_STAGE4_QDRANT_URL": "http://qdrant:6333",
        "RAGPLAN_STAGE4_QDRANT_COLLECTION_PREFIX": "api_runtime",
    }


@pytest.mark.asyncio
async def test_stage4_runtime_serves_only_matching_active_pointer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed: list[runtime.Stage3RuntimeConfig] = []
    expected_engine = object()

    async def fake_build(config: runtime.Stage3RuntimeConfig, **kwargs: object) -> object:
        observed.append(config)
        return expected_engine

    monkeypatch.setattr(runtime, "build_search_engine", fake_build)

    engine = await runtime.build_search_engine_from_environment(_active_environment(tmp_path))

    assert engine is expected_engine
    assert observed[0].vector_stage_manifest.name == "active-vector-stage.json"


@pytest.mark.asyncio
async def test_stage4_runtime_rejects_unactivated_new_vector_stage(tmp_path: Path) -> None:
    environment = _active_environment(tmp_path, stage_version="unactivated-v2")

    with pytest.raises(RAGPlanError) as captured:
        await runtime.build_search_engine_from_environment(environment)

    assert captured.value.code is ErrorCode.CORPUS_INCONSISTENT


@pytest.mark.asyncio
async def test_runtime_bootstrap_verifies_packaged_model_and_vector_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = runtime.Stage3RuntimeConfig.from_environment(_environment(tmp_path))
    assert config is not None
    events: list[str] = []
    verified_artifacts: list[tuple[Path, str]] = []

    class FakeEmbedderFactory:
        @classmethod
        def from_local_snapshot(cls, *, snapshot_path: Path, manifest: object) -> object:
            events.append("embedder")
            assert snapshot_path == config.model_snapshot
            assert manifest == load_default_model_artifact_manifest()
            return object()

    class FakeClient:
        def __init__(self, *, url: str, timeout: float) -> None:
            assert timeout == runtime.BACKEND_CLIENT_TIMEOUT_SECONDS
            events.append(f"client:{url}")

    class FakeManager:
        def __init__(self, client: object, config: object) -> None:
            events.append("manager")

        async def verify_stage(self, stage: VectorStageManifest) -> VectorStageManifest:
            events.append("verify_stage")
            return stage

    class FakeBackend:
        def __init__(self, manager: object) -> None:
            events.append("backend")

        async def close(self) -> None:
            events.append("close")

    class FakeEngine:
        def __init__(self, **kwargs: object) -> None:
            events.append("engine")

    def fake_verify(snapshot: Path, manifest: object) -> None:
        events.append("verify_artifacts")
        verified_artifacts.append((snapshot, getattr(manifest, "sha256")))

    monkeypatch.setattr(runtime, "verify_model_artifacts", fake_verify)
    monkeypatch.setattr(runtime, "SentenceTransformerEmbedder", FakeEmbedderFactory)
    monkeypatch.setattr(runtime, "AsyncQdrantClient", FakeClient)
    monkeypatch.setattr(runtime, "QdrantCollectionManager", FakeManager)
    monkeypatch.setattr(runtime, "QdrantVectorBackend", FakeBackend)
    monkeypatch.setattr(runtime, "VectorSearchEngine", FakeEngine)

    engine = await runtime.build_search_engine(config)

    assert isinstance(engine, FakeEngine)
    assert events == [
        "verify_artifacts",
        "embedder",
        "client:http://qdrant:6333",
        "manager",
        "backend",
        "verify_stage",
        "engine",
    ]
    assert verified_artifacts == [
        (config.model_snapshot, _stage().embedding_artifact_manifest_sha256)
    ]


@pytest.mark.asyncio
async def test_runtime_closes_qdrant_backend_when_stage_verification_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = runtime.Stage3RuntimeConfig.from_environment(_environment(tmp_path))
    assert config is not None
    events: list[str] = []

    class FakeEmbedderFactory:
        @classmethod
        def from_local_snapshot(cls, **kwargs: object) -> object:
            return object()

    class FakeClient:
        def __init__(self, **kwargs: object) -> None: ...

    class FakeManager:
        def __init__(self, client: object, config: object) -> None: ...

        async def verify_stage(self, stage: VectorStageManifest) -> VectorStageManifest:
            raise RAGPlanError(ErrorCode.CORPUS_INCONSISTENT, "inconsistent")

    class FakeBackend:
        def __init__(self, manager: object) -> None: ...

        async def close(self) -> None:
            events.append("close")

    monkeypatch.setattr(runtime, "verify_model_artifacts", lambda *_: None)
    monkeypatch.setattr(runtime, "SentenceTransformerEmbedder", FakeEmbedderFactory)
    monkeypatch.setattr(runtime, "AsyncQdrantClient", FakeClient)
    monkeypatch.setattr(runtime, "QdrantCollectionManager", FakeManager)
    monkeypatch.setattr(runtime, "QdrantVectorBackend", FakeBackend)

    with pytest.raises(RAGPlanError) as captured:
        await runtime.build_search_engine(config)
    assert captured.value.code is ErrorCode.CORPUS_INCONSISTENT
    assert events == ["close"]


@pytest.mark.asyncio
async def test_api_lifespan_owns_and_closes_bootstrapped_engine() -> None:
    events: list[str] = []

    class FakeEngine:
        async def close(self) -> None:
            events.append("close")

    async def factory() -> FakeEngine:
        events.append("build")
        return FakeEngine()

    app = create_app(runtime_factory=factory)
    async with app.router.lifespan_context(app):
        assert app.state.search_engine is not None
        assert events == ["build"]
    assert app.state.search_engine is None
    assert events == ["build", "close"]


def _stage5_environment(tmp_path: Path) -> dict[str, str]:
    checksum = "c" * 64
    extractor_version = build_extractor_version(Path("uv.lock"))
    repository = ManifestRepository(tmp_path / "stage5-ingestion")
    repository.activate(
        IngestionManifest(
            ingestion_run_id="stage5-active-run-v1",
            corpus_version="stage5-active-v1",
            source_dataset="fixture",
            source_version="v1",
            source_sha256="a" * 64,
            chunker_version="v1",
            embedding_model_revision="embedding-v1",
            extractor_version=extractor_version,
            document_count=1,
            chunk_count=1,
            qdrant_count=1,
            qdrant_id_checksum=checksum,
            qdrant_status=IngestionStoreStatus.SUCCEEDED,
            neo4j_count=1,
            neo4j_id_checksum=checksum,
            neo4j_status=IngestionStoreStatus.SUCCEEDED,
            activation_status=ActivationStatus.ACTIVE,
        )
    )
    graph_stage = GraphStageManifest(
        corpus_version="stage5-active-v1",
        database="neo4j",
        document_count=1,
        chunk_count=1,
        entity_count=2,
        mention_count=2,
        relation_count=1,
        canonical_id_checksum=checksum,
        graph_content_checksum="d" * 64,
        extractor_version=extractor_version,
    )
    graph_path = tmp_path / "graph-stage.json"
    graph_path.write_text(graph_stage.model_dump_json(), encoding="utf-8")
    return {
        "RAGPLAN_STAGE5_GRAPH_STAGE_MANIFEST": str(graph_path),
        "RAGPLAN_STAGE5_MANIFEST_ROOT": str(tmp_path / "stage5-ingestion"),
        "RAGPLAN_STAGE5_EXTRACTOR_LOCKFILE": str(Path("uv.lock").resolve()),
        "RAGPLAN_STAGE5_NEO4J_URI": "bolt://neo4j:7687",
        "RAGPLAN_STAGE5_NEO4J_USER": "neo4j",
        "RAGPLAN_STAGE5_NEO4J_DATABASE": "neo4j",
        "RAGPLAN_GRAPH__PASSWORD": "test-only",
    }


def test_stage5_runtime_configuration_is_all_or_nothing(tmp_path: Path) -> None:
    assert runtime.Stage5RuntimeConfig.from_environment({}) is None
    incomplete = _stage5_environment(tmp_path)
    del incomplete["RAGPLAN_GRAPH__PASSWORD"]

    with pytest.raises(RAGPlanError) as caught:
        runtime.Stage5RuntimeConfig.from_environment(incomplete)

    assert caught.value.code is ErrorCode.INVALID_REQUEST


def _stage6_environment(tmp_path: Path) -> dict[str, str]:
    return {
        "RAGPLAN_STAGE6_MODEL_SNAPSHOT": str(tmp_path / "model"),
        "RAGPLAN_STAGE6_VECTOR_STAGE_MANIFEST": str(tmp_path / "vector.json"),
        "RAGPLAN_STAGE6_GRAPH_STAGE_MANIFEST": str(tmp_path / "graph.json"),
        "RAGPLAN_STAGE6_MANIFEST_ROOT": str(tmp_path / "ingestion"),
        "RAGPLAN_STAGE6_EXTRACTOR_LOCKFILE": str(Path("uv.lock").resolve()),
        "RAGPLAN_STAGE6_QDRANT_URL": "http://qdrant:6333",
        "RAGPLAN_STAGE6_QDRANT_COLLECTION_PREFIX": "ragplan_chunks",
        "RAGPLAN_STAGE6_NEO4J_URI": "bolt://neo4j:7687",
        "RAGPLAN_STAGE6_NEO4J_USER": "neo4j",
        "RAGPLAN_STAGE6_NEO4J_DATABASE": "neo4j",
        "RAGPLAN_GRAPH__PASSWORD": "test-only",
    }


def _stage6_artifacts(tmp_path: Path) -> runtime.Stage6RuntimeConfig:
    environment = _stage6_environment(tmp_path)
    extractor_version = build_extractor_version(Path("uv.lock"))
    checksum = "7" * 64
    vector_stage = _stage().model_copy(
        update={
            "corpus_version": "stage6-active-v1",
            "collection_name": "ragplan_chunks_" + "8" * 64,
            "chunk_count": 1,
            "canonical_id_checksum": checksum,
        }
    )
    Path(environment["RAGPLAN_STAGE6_VECTOR_STAGE_MANIFEST"]).write_text(
        vector_stage.model_dump_json(),
        encoding="utf-8",
    )
    graph_stage = GraphStageManifest(
        corpus_version="stage6-active-v1",
        database="neo4j",
        document_count=1,
        chunk_count=1,
        entity_count=2,
        mention_count=2,
        relation_count=1,
        canonical_id_checksum=checksum,
        graph_content_checksum="9" * 64,
        extractor_version=extractor_version,
    )
    Path(environment["RAGPLAN_STAGE6_GRAPH_STAGE_MANIFEST"]).write_text(
        graph_stage.model_dump_json(),
        encoding="utf-8",
    )
    repository = ManifestRepository(Path(environment["RAGPLAN_STAGE6_MANIFEST_ROOT"]))
    repository.activate(
        IngestionManifest(
            ingestion_run_id="stage6-run-v1",
            corpus_version="stage6-active-v1",
            source_dataset="fixture",
            source_version="v1",
            source_sha256="a" * 64,
            chunker_version="v1",
            embedding_model_revision=vector_stage.embedding_model_revision,
            extractor_version=extractor_version,
            document_count=1,
            chunk_count=1,
            qdrant_count=1,
            qdrant_id_checksum=checksum,
            qdrant_status=IngestionStoreStatus.SUCCEEDED,
            neo4j_count=1,
            neo4j_id_checksum=checksum,
            neo4j_status=IngestionStoreStatus.SUCCEEDED,
            activation_status=ActivationStatus.ACTIVE,
        )
    )
    config = runtime.Stage6RuntimeConfig.from_environment(environment)
    assert config is not None
    return config


def test_stage6_runtime_configuration_is_all_or_nothing_and_secret_safe(
    tmp_path: Path,
) -> None:
    assert runtime.Stage6RuntimeConfig.from_environment({}) is None
    environment = _stage6_environment(tmp_path)
    config = runtime.Stage6RuntimeConfig.from_environment(environment)
    assert config is not None
    assert config.neo4j_password == "test-only"
    assert "test-only" not in repr(config)

    del environment["RAGPLAN_STAGE6_GRAPH_STAGE_MANIFEST"]
    with pytest.raises(RAGPlanError) as caught:
        runtime.Stage6RuntimeConfig.from_environment(environment)
    assert caught.value.code is ErrorCode.INVALID_REQUEST


@pytest.mark.asyncio
async def test_runtime_rejects_multiple_stage_profiles_with_stage6(tmp_path: Path) -> None:
    environment = {**_environment(tmp_path), **_stage6_environment(tmp_path)}
    with pytest.raises(RAGPlanError) as caught:
        await runtime.build_search_engine_from_environment(environment)
    assert caught.value.code is ErrorCode.INVALID_REQUEST


@pytest.mark.asyncio
async def test_stage6_runtime_verifies_both_live_stores_before_shared_engine(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _stage6_artifacts(tmp_path)
    events: list[object] = []

    class FakeEmbedderFactory:
        @classmethod
        def from_local_snapshot(cls, **kwargs: object) -> object:
            events.append("embedder")
            return object()

    class FakeExtractor:
        extractor_version = build_extractor_version(Path("uv.lock"))

        @classmethod
        def load_pinned(cls, **kwargs: object) -> FakeExtractor:
            events.append("extractor")
            return cls()

    class FakeClient:
        def __init__(self, **kwargs: object) -> None:
            events.append("qdrant-client")

    class FakeManager:
        def __init__(self, client: object, config: object) -> None:
            events.append("qdrant-manager")

        async def verify_stage(self, stage: VectorStageManifest) -> VectorStageManifest:
            events.append(("verify-vector", stage.corpus_version))
            return stage

    class FakeVectorBackend:
        def __init__(self, manager: object) -> None:
            events.append("vector-backend")

        async def close(self) -> None:
            events.append("close-vector")

    class FakeGraphBackend:
        @classmethod
        def connect(cls, config: object) -> FakeGraphBackend:
            events.append("graph-backend")
            return cls()

        async def require_active_corpus(self, **kwargs: object) -> None:
            events.append(("verify-graph", kwargs["corpus_version"]))

        async def close(self) -> None:
            events.append("close-graph")

    class FakeAnalyzer:
        def __init__(self, extractor: object, **kwargs: object) -> None:
            events.append("analyzer")

    class FakeEngine:
        def __init__(self, **kwargs: object) -> None:
            events.append(("engine", kwargs["active_manifest"].corpus_version))

    monkeypatch.setattr(runtime, "verify_model_artifacts", lambda *_: events.append("model"))
    monkeypatch.setattr(runtime, "SentenceTransformerEmbedder", FakeEmbedderFactory)
    monkeypatch.setattr(runtime, "EntityExtractor", FakeExtractor)
    monkeypatch.setattr(runtime, "AsyncQdrantClient", FakeClient)
    monkeypatch.setattr(runtime, "QdrantCollectionManager", FakeManager)
    monkeypatch.setattr(runtime, "QdrantVectorBackend", FakeVectorBackend)
    monkeypatch.setattr(runtime, "Neo4jGraphBackend", FakeGraphBackend)
    monkeypatch.setattr(runtime, "GraphQueryAnalyzer", FakeAnalyzer)
    monkeypatch.setattr(runtime, "BaselineSearchEngine", FakeEngine)

    engine = await runtime.build_baseline_search_engine(config)

    assert isinstance(engine, FakeEngine)
    assert ("verify-vector", "stage6-active-v1") in events
    assert ("verify-graph", "stage6-active-v1") in events
    assert events[-1] == ("engine", "stage6-active-v1")


@pytest.mark.asyncio
async def test_stage5_runtime_verifies_active_graph_before_building_engine(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = runtime.Stage5RuntimeConfig.from_environment(_stage5_environment(tmp_path))
    assert config is not None
    events: list[object] = []

    class FakeExtractor:
        extractor_version = build_extractor_version(Path("uv.lock"))

        @classmethod
        def load_pinned(cls, **kwargs: object) -> FakeExtractor:
            events.append(("extractor", kwargs["lockfile"]))
            return cls()

    class FakeBackend:
        @classmethod
        def connect(cls, backend_config: object) -> FakeBackend:
            events.append("connect")
            return cls()

        async def require_active_corpus(self, **kwargs: object) -> None:
            events.append(("verify", kwargs["corpus_version"], kwargs["chunk_count"]))

        async def close(self) -> None:
            events.append("close")

    class FakeEngine:
        def __init__(self, **kwargs: object) -> None:
            events.append(("engine", kwargs["active_manifest"].corpus_version))

    monkeypatch.setattr(runtime, "EntityExtractor", FakeExtractor)
    monkeypatch.setattr(runtime, "Neo4jGraphBackend", FakeBackend)
    monkeypatch.setattr(runtime, "GraphSearchEngine", FakeEngine)

    engine = await runtime.build_graph_search_engine(config)

    assert isinstance(engine, FakeEngine)
    assert events == [
        ("extractor", config.extractor_lockfile),
        "connect",
        ("verify", "stage5-active-v1", 1),
        ("engine", "stage5-active-v1"),
    ]
