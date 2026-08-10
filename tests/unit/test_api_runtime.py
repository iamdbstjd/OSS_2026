"""Stage 3 API bootstrap remains explicit, verified, and injectable."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from ragplan.api import runtime
from ragplan.api.server import create_app
from ragplan.core.errors import ErrorCode, RAGPlanError
from ragplan.core.models import VectorStageManifest
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
        def __init__(self, *, url: str) -> None:
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
