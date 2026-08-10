"""Fail-closed Stage 3 API runtime bootstrap."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError
from qdrant_client import AsyncQdrantClient

from ragplan.backends.vector.qdrant import (
    DEFAULT_COLLECTION_PREFIX,
    QdrantCollectionManager,
    QdrantVectorBackend,
    QdrantVectorConfig,
)
from ragplan.core.engine import SearchEngine, VectorSearchEngine
from ragplan.core.errors import ErrorCode, RAGPlanError
from ragplan.core.models import VectorStageManifest
from ragplan.ingestion.embedder import SentenceTransformerEmbedder
from ragplan.ingestion.model_manifest import (
    load_default_model_artifact_manifest,
    verify_model_artifacts,
)
from ragplan.planner.catalog import PlanCatalog, load_default_plan_catalog

_CONFIGURATION_KEYS = (
    "RAGPLAN_STAGE3_MODEL_SNAPSHOT",
    "RAGPLAN_STAGE3_VECTOR_STAGE_MANIFEST",
    "RAGPLAN_STAGE3_QDRANT_URL",
    "RAGPLAN_STAGE3_QDRANT_COLLECTION_PREFIX",
)


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
            AsyncQdrantClient(url=config.qdrant_url),
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


async def build_search_engine_from_environment(
    environment: Mapping[str, str] | None = None,
    *,
    plan_catalog: PlanCatalog | None = None,
) -> SearchEngine | None:
    """Return an engine only for fully configured Stage 3 deployments."""

    config = Stage3RuntimeConfig.from_environment(environment)
    if config is None:
        return None
    return await build_search_engine(config, plan_catalog=plan_catalog)


__all__ = [
    "DEFAULT_COLLECTION_PREFIX",
    "Stage3RuntimeConfig",
    "build_search_engine",
    "build_search_engine_from_environment",
    "load_vector_stage_manifest",
]
