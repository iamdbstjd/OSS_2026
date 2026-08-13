"""Fail-closed staged and active-corpus API runtime bootstrap."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import ValidationError
from qdrant_client import AsyncQdrantClient

from ragplan.backends.graph.neo4j import Neo4jGraphBackend, Neo4jGraphConfig
from ragplan.backends.vector.qdrant import (
    DEFAULT_COLLECTION_PREFIX,
    QdrantCollectionManager,
    QdrantVectorBackend,
    QdrantVectorConfig,
)
from ragplan.core.deadline import PerfCounterClock
from ragplan.core.engine import (
    BaselineSearchEngine,
    GraphSearchEngine,
    SearchEngine,
    VectorSearchEngine,
)
from ragplan.core.errors import ErrorCode, RAGPlanError
from ragplan.core.models import GraphStageManifest, VectorStageManifest
from ragplan.ingestion.embedder import SentenceTransformerEmbedder
from ragplan.ingestion.entities import EntityExtractor
from ragplan.ingestion.manifest import ManifestRepository, load_contract_json
from ragplan.ingestion.model_manifest import (
    load_default_model_artifact_manifest,
    verify_model_artifacts,
)
from ragplan.planner.catalog import PlanCatalog, load_default_plan_catalog
from ragplan.retrieval.graph import GraphQueryAnalyzer

_CONFIGURATION_KEYS = (
    "RAGPLAN_STAGE3_MODEL_SNAPSHOT",
    "RAGPLAN_STAGE3_VECTOR_STAGE_MANIFEST",
    "RAGPLAN_STAGE3_QDRANT_URL",
    "RAGPLAN_STAGE3_QDRANT_COLLECTION_PREFIX",
)
_STAGE4_CONFIGURATION_KEYS = (
    "RAGPLAN_STAGE4_MODEL_SNAPSHOT",
    "RAGPLAN_STAGE4_VECTOR_STAGE_MANIFEST",
    "RAGPLAN_STAGE4_MANIFEST_ROOT",
    "RAGPLAN_STAGE4_QDRANT_URL",
    "RAGPLAN_STAGE4_QDRANT_COLLECTION_PREFIX",
)
_STAGE5_CONFIGURATION_KEYS = (
    "RAGPLAN_STAGE5_GRAPH_STAGE_MANIFEST",
    "RAGPLAN_STAGE5_MANIFEST_ROOT",
    "RAGPLAN_STAGE5_EXTRACTOR_LOCKFILE",
    "RAGPLAN_STAGE5_NEO4J_URI",
    "RAGPLAN_STAGE5_NEO4J_USER",
    "RAGPLAN_STAGE5_NEO4J_DATABASE",
)
_STAGE6_CONFIGURATION_KEYS = (
    "RAGPLAN_STAGE6_MODEL_SNAPSHOT",
    "RAGPLAN_STAGE6_VECTOR_STAGE_MANIFEST",
    "RAGPLAN_STAGE6_GRAPH_STAGE_MANIFEST",
    "RAGPLAN_STAGE6_MANIFEST_ROOT",
    "RAGPLAN_STAGE6_EXTRACTOR_LOCKFILE",
    "RAGPLAN_STAGE6_QDRANT_URL",
    "RAGPLAN_STAGE6_QDRANT_COLLECTION_PREFIX",
    "RAGPLAN_STAGE6_NEO4J_URI",
    "RAGPLAN_STAGE6_NEO4J_USER",
    "RAGPLAN_STAGE6_NEO4J_DATABASE",
)
BACKEND_CLIENT_TIMEOUT_SECONDS = 30


def _configuration_error(message: str) -> RAGPlanError:
    return RAGPlanError(ErrorCode.INVALID_REQUEST, message, retryable=False)


@dataclass(frozen=True, slots=True)
class Stage3RuntimeConfig:
    """Explicit environment configuration for one verified vector-staged corpus."""

    model_snapshot: Path
    vector_stage_manifest: Path
    qdrant_url: str
    collection_prefix: str

    @classmethod
    def from_environment(
        cls, environment: Mapping[str, str] | None = None
    ) -> Stage3RuntimeConfig | None:
        values = {
            key: value.strip()
            for key in _CONFIGURATION_KEYS
            if (value := (os.environ if environment is None else environment).get(key, "")).strip()
        }
        if not values:
            return None
        if len(values) != len(_CONFIGURATION_KEYS):
            raise _configuration_error("Stage 3 runtime configuration is incomplete")
        return cls(
            model_snapshot=Path(values["RAGPLAN_STAGE3_MODEL_SNAPSHOT"]),
            vector_stage_manifest=Path(values["RAGPLAN_STAGE3_VECTOR_STAGE_MANIFEST"]),
            qdrant_url=values["RAGPLAN_STAGE3_QDRANT_URL"],
            collection_prefix=values["RAGPLAN_STAGE3_QDRANT_COLLECTION_PREFIX"],
        )


@dataclass(frozen=True, slots=True)
class Stage4RuntimeConfig:
    """Runtime configuration that serves only the atomically activated corpus."""

    model_snapshot: Path
    vector_stage_manifest: Path
    manifest_root: Path
    qdrant_url: str
    collection_prefix: str

    @classmethod
    def from_environment(
        cls, environment: Mapping[str, str] | None = None
    ) -> Stage4RuntimeConfig | None:
        source = os.environ if environment is None else environment
        values = {
            key: value.strip()
            for key in _STAGE4_CONFIGURATION_KEYS
            if (value := source.get(key, "")).strip()
        }
        if not values:
            return None
        if len(values) != len(_STAGE4_CONFIGURATION_KEYS):
            raise _configuration_error("Stage 4 active runtime configuration is incomplete")
        return cls(
            model_snapshot=Path(values["RAGPLAN_STAGE4_MODEL_SNAPSHOT"]),
            vector_stage_manifest=Path(values["RAGPLAN_STAGE4_VECTOR_STAGE_MANIFEST"]),
            manifest_root=Path(values["RAGPLAN_STAGE4_MANIFEST_ROOT"]),
            qdrant_url=values["RAGPLAN_STAGE4_QDRANT_URL"],
            collection_prefix=values["RAGPLAN_STAGE4_QDRANT_COLLECTION_PREFIX"],
        )


@dataclass(frozen=True, slots=True)
class Stage5RuntimeConfig:
    """Explicit graph-only runtime over one atomically activated corpus."""

    graph_stage_manifest: Path
    manifest_root: Path
    extractor_lockfile: Path
    neo4j_uri: str
    neo4j_user: str
    neo4j_password: str = field(repr=False)
    neo4j_database: str = "neo4j"

    @classmethod
    def from_environment(
        cls, environment: Mapping[str, str] | None = None
    ) -> Stage5RuntimeConfig | None:
        source = os.environ if environment is None else environment
        values = {
            key: value.strip()
            for key in _STAGE5_CONFIGURATION_KEYS
            if (value := source.get(key, "")).strip()
        }
        if not values:
            return None
        password = source.get("RAGPLAN_GRAPH__PASSWORD", "").strip()
        if len(values) != len(_STAGE5_CONFIGURATION_KEYS) or not password:
            raise _configuration_error("Stage 5 graph runtime configuration is incomplete")
        return cls(
            graph_stage_manifest=Path(values["RAGPLAN_STAGE5_GRAPH_STAGE_MANIFEST"]),
            manifest_root=Path(values["RAGPLAN_STAGE5_MANIFEST_ROOT"]),
            extractor_lockfile=Path(values["RAGPLAN_STAGE5_EXTRACTOR_LOCKFILE"]),
            neo4j_uri=values["RAGPLAN_STAGE5_NEO4J_URI"],
            neo4j_user=values["RAGPLAN_STAGE5_NEO4J_USER"],
            neo4j_password=password,
            neo4j_database=values["RAGPLAN_STAGE5_NEO4J_DATABASE"],
        )


@dataclass(frozen=True, slots=True)
class Stage6RuntimeConfig:
    """One dual-store runtime pinned to a reconciled active corpus."""

    model_snapshot: Path
    vector_stage_manifest: Path
    graph_stage_manifest: Path
    manifest_root: Path
    extractor_lockfile: Path
    qdrant_url: str
    collection_prefix: str
    neo4j_uri: str
    neo4j_user: str
    neo4j_password: str = field(repr=False)
    neo4j_database: str = "neo4j"

    @classmethod
    def from_environment(
        cls, environment: Mapping[str, str] | None = None
    ) -> Stage6RuntimeConfig | None:
        source = os.environ if environment is None else environment
        values = {
            key: value.strip()
            for key in _STAGE6_CONFIGURATION_KEYS
            if (value := source.get(key, "")).strip()
        }
        if not values:
            return None
        password = source.get("RAGPLAN_GRAPH__PASSWORD", "").strip()
        if len(values) != len(_STAGE6_CONFIGURATION_KEYS) or not password:
            raise _configuration_error("Stage 6 dual-store runtime configuration is incomplete")
        return cls(
            model_snapshot=Path(values["RAGPLAN_STAGE6_MODEL_SNAPSHOT"]),
            vector_stage_manifest=Path(values["RAGPLAN_STAGE6_VECTOR_STAGE_MANIFEST"]),
            graph_stage_manifest=Path(values["RAGPLAN_STAGE6_GRAPH_STAGE_MANIFEST"]),
            manifest_root=Path(values["RAGPLAN_STAGE6_MANIFEST_ROOT"]),
            extractor_lockfile=Path(values["RAGPLAN_STAGE6_EXTRACTOR_LOCKFILE"]),
            qdrant_url=values["RAGPLAN_STAGE6_QDRANT_URL"],
            collection_prefix=values["RAGPLAN_STAGE6_QDRANT_COLLECTION_PREFIX"],
            neo4j_uri=values["RAGPLAN_STAGE6_NEO4J_URI"],
            neo4j_user=values["RAGPLAN_STAGE6_NEO4J_USER"],
            neo4j_password=password,
            neo4j_database=values["RAGPLAN_STAGE6_NEO4J_DATABASE"],
        )


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    decoded: dict[str, object] = {}
    for key, value in pairs:
        if key in decoded:
            raise ValueError("duplicate JSON key")
        decoded[key] = value
    return decoded


def load_vector_stage_manifest(path: Path) -> VectorStageManifest:
    """Load only validated ``vector_staged`` evidence, never an active pointer."""

    try:
        serialized = path.read_text(encoding="utf-8")
        decoded = json.loads(serialized, object_pairs_hook=_reject_duplicate_keys)
        return VectorStageManifest.model_validate(decoded)
    except (OSError, UnicodeError, json.JSONDecodeError, ValidationError, ValueError) as exc:
        raise RAGPlanError(
            ErrorCode.CORPUS_INCONSISTENT,
            "vector stage manifest is invalid",
            retryable=False,
        ) from exc


async def build_search_engine(
    config: Stage3RuntimeConfig,
    *,
    plan_catalog: PlanCatalog | None = None,
) -> SearchEngine:
    """Build one engine after verifying model, staging, and Qdrant provenance."""

    stage = load_vector_stage_manifest(config.vector_stage_manifest)
    manifest = load_default_model_artifact_manifest()
    if stage.embedding_artifact_manifest_sha256 != manifest.sha256:
        raise RAGPlanError(
            ErrorCode.MODEL_INCOMPATIBLE,
            "vector stage and packaged embedding artifact manifests do not match",
            retryable=False,
        )
    verify_model_artifacts(config.model_snapshot, manifest)
    embedder = SentenceTransformerEmbedder.from_local_snapshot(
        snapshot_path=config.model_snapshot,
        manifest=manifest,
    )

    try:
        manager = QdrantCollectionManager(
            AsyncQdrantClient(
                url=config.qdrant_url,
                timeout=BACKEND_CLIENT_TIMEOUT_SECONDS,
            ),
            QdrantVectorConfig(collection_prefix=config.collection_prefix),
        )
    except ValueError as exc:
        raise _configuration_error("Stage 3 Qdrant configuration is invalid") from exc
    backend = QdrantVectorBackend(manager)
    try:
        verified_stage = await manager.verify_stage(stage)
        return VectorSearchEngine(
            embedder=embedder,
            vector_backend=backend,
            plan_catalog=plan_catalog if plan_catalog is not None else load_default_plan_catalog(),
            vector_stage=verified_stage,
        )
    except BaseException:
        await backend.close()
        raise


async def build_graph_search_engine(
    config: Stage5RuntimeConfig,
    *,
    plan_catalog: PlanCatalog | None = None,
) -> SearchEngine:
    """Build graph-only serving after active-pointer and live Neo4j verification."""

    repository = ManifestRepository(config.manifest_root)
    _, active = repository.load_active()
    graph_stage = load_contract_json(config.graph_stage_manifest, GraphStageManifest)
    if (
        graph_stage.corpus_version != active.corpus_version
        or graph_stage.database != config.neo4j_database
        or graph_stage.chunk_count != active.neo4j_count
        or graph_stage.canonical_id_checksum != active.neo4j_id_checksum
        or graph_stage.extractor_version != active.extractor_version
    ):
        raise RAGPlanError(
            ErrorCode.CORPUS_INCONSISTENT,
            "active corpus and graph stage evidence do not match",
            retryable=False,
        )
    extractor = EntityExtractor.load_pinned(lockfile=config.extractor_lockfile)
    if extractor.extractor_version != active.extractor_version:
        raise RAGPlanError(
            ErrorCode.MODEL_INCOMPATIBLE,
            "runtime graph extractor does not match the active corpus",
            retryable=False,
        )
    try:
        backend = Neo4jGraphBackend.connect(
            Neo4jGraphConfig(
                uri=config.neo4j_uri,
                user=config.neo4j_user,
                password=config.neo4j_password,
                database=config.neo4j_database,
            )
        )
    except ValueError as exc:
        raise _configuration_error("Stage 5 Neo4j configuration is invalid") from exc
    try:
        await backend.require_active_corpus(
            corpus_version=active.corpus_version,
            chunk_count=active.neo4j_count,
            canonical_id_checksum=active.neo4j_id_checksum,
            extractor_version=active.extractor_version,
        )
        clock = PerfCounterClock()
        return GraphSearchEngine(
            analyzer=GraphQueryAnalyzer(extractor, clock=clock),
            graph_backend=backend,
            plan_catalog=plan_catalog if plan_catalog is not None else load_default_plan_catalog(),
            active_manifest=active,
            clock=clock,
        )
    except BaseException:
        await backend.close()
        raise


async def build_baseline_search_engine(
    config: Stage6RuntimeConfig,
    *,
    plan_catalog: PlanCatalog | None = None,
) -> SearchEngine:
    """Build Stage 6 only after both stores prove the same active corpus."""

    repository = ManifestRepository(config.manifest_root)
    _, active = repository.load_active()
    vector_stage = load_vector_stage_manifest(config.vector_stage_manifest)
    graph_stage = load_contract_json(config.graph_stage_manifest, GraphStageManifest)

    artifact_manifest = load_default_model_artifact_manifest()
    if vector_stage.embedding_artifact_manifest_sha256 != artifact_manifest.sha256:
        raise RAGPlanError(
            ErrorCode.MODEL_INCOMPATIBLE,
            "vector stage and packaged embedding artifact manifests do not match",
            retryable=False,
        )
    verify_model_artifacts(config.model_snapshot, artifact_manifest)
    embedder = SentenceTransformerEmbedder.from_local_snapshot(
        snapshot_path=config.model_snapshot,
        manifest=artifact_manifest,
    )
    extractor = EntityExtractor.load_pinned(lockfile=config.extractor_lockfile)

    vector_matches = (
        vector_stage.corpus_version == active.corpus_version
        and vector_stage.chunk_count == active.qdrant_count
        and vector_stage.canonical_id_checksum == active.qdrant_id_checksum
        and vector_stage.embedding_model_revision == active.embedding_model_revision
    )
    graph_matches = (
        graph_stage.corpus_version == active.corpus_version
        and graph_stage.database == config.neo4j_database
        and graph_stage.document_count == active.document_count
        and graph_stage.chunk_count == active.neo4j_count
        and graph_stage.canonical_id_checksum == active.neo4j_id_checksum
        and graph_stage.extractor_version == active.extractor_version
        and extractor.extractor_version == active.extractor_version
    )
    if not vector_matches or not graph_matches:
        raise RAGPlanError(
            ErrorCode.CORPUS_INCONSISTENT,
            "active corpus and Stage 6 store evidence do not match",
            retryable=False,
        )

    qdrant_client: AsyncQdrantClient | None = None
    try:
        qdrant_client = AsyncQdrantClient(
            url=config.qdrant_url,
            timeout=BACKEND_CLIENT_TIMEOUT_SECONDS,
        )
        manager = QdrantCollectionManager(
            qdrant_client,
            QdrantVectorConfig(collection_prefix=config.collection_prefix),
        )
    except ValueError as exc:
        if qdrant_client is not None:
            await qdrant_client.close()
        raise _configuration_error("Stage 6 Qdrant configuration is invalid") from exc
    vector_backend = QdrantVectorBackend(manager)
    try:
        graph_backend = Neo4jGraphBackend.connect(
            Neo4jGraphConfig(
                uri=config.neo4j_uri,
                user=config.neo4j_user,
                password=config.neo4j_password,
                database=config.neo4j_database,
            )
        )
    except ValueError as exc:
        await vector_backend.close()
        raise _configuration_error("Stage 6 Neo4j configuration is invalid") from exc
    except BaseException:
        await vector_backend.close()
        raise
    try:
        verified_vector_stage = await manager.verify_stage(vector_stage)
        await graph_backend.require_active_corpus(
            corpus_version=active.corpus_version,
            chunk_count=active.neo4j_count,
            canonical_id_checksum=active.neo4j_id_checksum,
            extractor_version=active.extractor_version,
        )
        clock = PerfCounterClock()
        return BaselineSearchEngine(
            embedder=embedder,
            vector_backend=vector_backend,
            analyzer=GraphQueryAnalyzer(extractor, clock=clock),
            graph_backend=graph_backend,
            plan_catalog=plan_catalog if plan_catalog is not None else load_default_plan_catalog(),
            active_manifest=active,
            vector_stage=verified_vector_stage,
            graph_stage=graph_stage,
            clock=clock,
        )
    except BaseException:
        await vector_backend.close()
        await graph_backend.close()
        raise


async def build_search_engine_from_environment(
    environment: Mapping[str, str] | None = None,
    *,
    plan_catalog: PlanCatalog | None = None,
) -> SearchEngine | None:
    """Return an engine only for one complete, verified runtime profile."""

    stage3_config = Stage3RuntimeConfig.from_environment(environment)
    stage4_config = Stage4RuntimeConfig.from_environment(environment)
    stage5_config = Stage5RuntimeConfig.from_environment(environment)
    stage6_config = Stage6RuntimeConfig.from_environment(environment)
    configured_modes = sum(
        item is not None for item in (stage3_config, stage4_config, stage5_config, stage6_config)
    )
    if configured_modes > 1:
        raise _configuration_error(
            "Stage 3, Stage 4, Stage 5, and Stage 6 runtime modes cannot be configured together"
        )
    if stage6_config is not None:
        return await build_baseline_search_engine(stage6_config, plan_catalog=plan_catalog)
    if stage5_config is not None:
        return await build_graph_search_engine(stage5_config, plan_catalog=plan_catalog)
    if stage4_config is not None:
        active_repository = ManifestRepository(stage4_config.manifest_root)
        _, active_manifest = active_repository.load_active()
        vector_stage = load_vector_stage_manifest(stage4_config.vector_stage_manifest)
        if (
            vector_stage.corpus_version != active_manifest.corpus_version
            or vector_stage.chunk_count != active_manifest.qdrant_count
            or vector_stage.canonical_id_checksum != active_manifest.qdrant_id_checksum
            or vector_stage.embedding_model_revision != active_manifest.embedding_model_revision
        ):
            raise RAGPlanError(
                ErrorCode.CORPUS_INCONSISTENT,
                "active corpus and vector stage evidence do not match",
                retryable=False,
            )
        return await build_search_engine(
            Stage3RuntimeConfig(
                model_snapshot=stage4_config.model_snapshot,
                vector_stage_manifest=stage4_config.vector_stage_manifest,
                qdrant_url=stage4_config.qdrant_url,
                collection_prefix=stage4_config.collection_prefix,
            ),
            plan_catalog=plan_catalog,
        )
    if stage3_config is None:
        return None
    return await build_search_engine(stage3_config, plan_catalog=plan_catalog)


__all__ = [
    "DEFAULT_COLLECTION_PREFIX",
    "Stage3RuntimeConfig",
    "Stage4RuntimeConfig",
    "Stage5RuntimeConfig",
    "Stage6RuntimeConfig",
    "build_baseline_search_engine",
    "build_graph_search_engine",
    "build_search_engine",
    "build_search_engine_from_environment",
    "load_vector_stage_manifest",
]
