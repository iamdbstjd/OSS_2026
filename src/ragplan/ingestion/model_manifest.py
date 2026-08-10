"""Fail-closed integrity manifest for the pinned Stage 3 embedding model."""

from __future__ import annotations

import hashlib
import importlib.resources
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Final

from ragplan.core.errors import ErrorCode, RAGPlanError

MODEL_ID: Final = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION: Final = "b8903db39f65d93ae28d49a37c4f3fa90c5f94e0"
EMBEDDING_DIMENSION: Final = 384
DEFAULT_MODEL_MANIFEST_FILENAME: Final = "all_minilm_l6_v2.json"
DEFAULT_MODEL_MANIFEST_PATH: Final = (
    Path(__file__).resolve().parents[3]
    / "configs"
    / "models"
    / "all_minilm_l6_v2.b8903db.manifest.json"
)

_SHA256_PATTERN: Final = re.compile(r"^[0-9a-f]{64}$")
_REQUIRED_CONFIGURATION_ARTIFACTS: Final = frozenset(
    {
        "1_Pooling/config.json",
        "config.json",
        "config_sentence_transformers.json",
        "modules.json",
        "sentence_bert_config.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.txt",
    }
)
_SUPPORTED_WEIGHT_ARTIFACTS: Final = frozenset({"model.safetensors", "pytorch_model.bin"})
_MANIFEST_KEYS: Final = frozenset(
    {
        "schema_version",
        "model_id",
        "revision",
        "embedding_dimension",
        "normalize_embeddings",
        "artifacts",
    }
)


def _model_incompatible(message: str) -> RAGPlanError:
    return RAGPlanError(ErrorCode.MODEL_INCOMPATIBLE, message, retryable=False)


@dataclass(frozen=True, slots=True)
class ModelArtifactManifest:
    """Exact files authorized for the immutable embedding-model snapshot."""

    schema_version: str
    model_id: str
    revision: str
    embedding_dimension: int
    normalize_embeddings: bool
    artifacts: Mapping[str, str]

    def __post_init__(self) -> None:
        if self.schema_version != "v1":
            raise _model_incompatible("unsupported embedding-model manifest schema")
        if self.model_id != MODEL_ID or self.revision != MODEL_REVISION:
            raise _model_incompatible("embedding model identity or revision is not approved")
        if self.embedding_dimension != EMBEDDING_DIMENSION:
            raise _model_incompatible("embedding dimension does not match the Stage 3 contract")
        if self.normalize_embeddings is not True:
            raise _model_incompatible("embedding normalization must be enabled")

        if not isinstance(self.artifacts, Mapping):
            raise _model_incompatible("embedding-model artifact table is invalid")

        validated: dict[str, str] = {}
        for artifact, checksum in self.artifacts.items():
            if not isinstance(artifact, str) or not isinstance(checksum, str):
                raise _model_incompatible("model artifact entries must be strings")
            artifact_path = PurePosixPath(artifact)
            if (
                not artifact
                or artifact_path.is_absolute()
                or ".." in artifact_path.parts
                or artifact_path.as_posix() != artifact
            ):
                raise _model_incompatible("model artifact path is invalid")
            if _SHA256_PATTERN.fullmatch(checksum) is None:
                raise _model_incompatible("model artifact checksum must be lowercase SHA-256")
            validated[artifact] = checksum

        if not _REQUIRED_CONFIGURATION_ARTIFACTS.issubset(validated):
            raise _model_incompatible("embedding-model manifest is missing required artifacts")
        if not _SUPPORTED_WEIGHT_ARTIFACTS.intersection(validated):
            raise _model_incompatible("embedding-model manifest is missing model weights")
        object.__setattr__(self, "artifacts", MappingProxyType(validated))

    def canonical_json_bytes(self) -> bytes:
        """Serialize the validated manifest deterministically for provenance."""

        payload = {
            "artifacts": dict(sorted(self.artifacts.items())),
            "embedding_dimension": self.embedding_dimension,
            "model_id": self.model_id,
            "normalize_embeddings": self.normalize_embeddings,
            "revision": self.revision,
            "schema_version": self.schema_version,
        }
        return json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")

    @property
    def sha256(self) -> str:
        """Return a stable fingerprint for the full model manifest."""

        return hashlib.sha256(self.canonical_json_bytes()).hexdigest()


def load_model_artifact_manifest(path: Path) -> ModelArtifactManifest:
    """Load and strictly validate a JSON model-artifact manifest."""

    try:
        serialized = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise _model_incompatible("embedding-model manifest could not be loaded") from exc
    return _parse_model_artifact_manifest(serialized)


def load_default_model_artifact_manifest() -> ModelArtifactManifest:
    """Load the packaged, checksum-pinned MiniLM manifest."""

    if DEFAULT_MODEL_MANIFEST_PATH.is_file():
        return load_model_artifact_manifest(DEFAULT_MODEL_MANIFEST_PATH)
    try:
        manifest_resource = importlib.resources.files("ragplan.resources.models").joinpath(
            DEFAULT_MODEL_MANIFEST_FILENAME
        )
        serialized = manifest_resource.read_text(encoding="utf-8")
    except (ImportError, ModuleNotFoundError, OSError, UnicodeError) as exc:
        raise _model_incompatible("packaged embedding-model manifest is unavailable") from exc
    return _parse_model_artifact_manifest(serialized)


def _parse_model_artifact_manifest(serialized: str) -> ModelArtifactManifest:
    try:
        decoded: object = json.loads(serialized)
    except json.JSONDecodeError as exc:
        raise _model_incompatible("embedding-model manifest could not be loaded") from exc
    if not isinstance(decoded, dict) or set(decoded) != _MANIFEST_KEYS:
        raise _model_incompatible("embedding-model manifest fields are invalid")

    artifacts = decoded.get("artifacts")
    if not isinstance(artifacts, dict):
        raise _model_incompatible("embedding-model artifact table is invalid")
    if not all(isinstance(key, str) and isinstance(value, str) for key, value in artifacts.items()):
        raise _model_incompatible("embedding-model artifact table is invalid")

    schema_version = decoded.get("schema_version")
    model_id = decoded.get("model_id")
    revision = decoded.get("revision")
    embedding_dimension = decoded.get("embedding_dimension")
    normalize_embeddings = decoded.get("normalize_embeddings")
    if not isinstance(schema_version, str):
        raise _model_incompatible("embedding-model manifest schema is invalid")
    if not isinstance(model_id, str) or not isinstance(revision, str):
        raise _model_incompatible("embedding-model manifest identity is invalid")
    if isinstance(embedding_dimension, bool) or not isinstance(embedding_dimension, int):
        raise _model_incompatible("embedding-model manifest dimension is invalid")
    if not isinstance(normalize_embeddings, bool):
        raise _model_incompatible("embedding-model manifest normalization flag is invalid")

    return ModelArtifactManifest(
        schema_version=schema_version,
        model_id=model_id,
        revision=revision,
        embedding_dimension=embedding_dimension,
        normalize_embeddings=normalize_embeddings,
        artifacts=artifacts,
    )


def verify_model_artifacts(snapshot_path: Path, manifest: ModelArtifactManifest) -> None:
    """Verify every declared artifact and reject unverified candidate weight files."""

    if not snapshot_path.is_dir():
        raise _model_incompatible("local embedding-model snapshot is unavailable")

    observed_artifacts = {
        artifact.relative_to(snapshot_path).as_posix()
        for artifact in snapshot_path.rglob("*")
        if artifact.is_file() and ".cache" not in artifact.relative_to(snapshot_path).parts
    }
    unexpected_artifacts = observed_artifacts - set(manifest.artifacts)
    if unexpected_artifacts:
        raise _model_incompatible("embedding-model snapshot contains unverified artifacts")

    for relative_path, expected_checksum in manifest.artifacts.items():
        artifact_path = snapshot_path.joinpath(*PurePosixPath(relative_path).parts)
        if not artifact_path.is_file():
            raise _model_incompatible("local embedding-model snapshot is incomplete")
        digest = hashlib.sha256()
        try:
            with artifact_path.open("rb") as artifact_file:
                for block in iter(lambda: artifact_file.read(1024 * 1024), b""):
                    digest.update(block)
        except OSError as exc:
            raise _model_incompatible("embedding-model artifact could not be read") from exc
        if digest.hexdigest() != expected_checksum:
            raise _model_incompatible("embedding-model artifact checksum mismatch")
