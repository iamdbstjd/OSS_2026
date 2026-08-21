"""Checksum-first, trusted-type-only Stage 11 cost-model artifact lifecycle."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from enum import StrEnum
from importlib.metadata import PackageNotFoundError, version
from importlib.resources import files
from pathlib import Path
from typing import Annotated, Any, Literal, cast

import skops.io as sio  # type: ignore[import-untyped]
from pydantic import Field, model_validator
from sklearn.ensemble import HistGradientBoostingRegressor  # type: ignore[import-untyped]

from ragplan.benchmark.artifacts import write_json_model
from ragplan.benchmark.contracts import canonical_json_bytes
from ragplan.benchmark.records import EnvironmentManifest
from ragplan.core.errors import ErrorCode, RAGPlanError
from ragplan.core.models import (
    FrozenJsonMapping,
    FrozenModel,
    ModelManifest,
    NonEmptyString,
    Sha256Hex,
)

DEFAULT_TRUSTED_TYPES_PATH = (
    Path(__file__).resolve().parents[3] / "configs" / "skops_trusted_types_v1.json"
)
ESTIMATOR_TYPE = (
    "sklearn.ensemble._hist_gradient_boosting.gradient_boosting.HistGradientBoostingRegressor"
)


class ModelKind(StrEnum):
    QUALITY = "quality"
    LATENCY_P95 = "latency_p95"


class ArtifactStatus(StrEnum):
    SERVING_ELIGIBLE = "serving_eligible"
    RESEARCH_ONLY = "research_only"


class ArtifactContext(FrozenModel):
    feature_schema_version: NonEmptyString
    plan_catalog_hash: Sha256Hex
    corpus_version: NonEmptyString
    qrels_version: NonEmptyString
    embedding_model_revision: NonEmptyString
    extractor_version: NonEmptyString
    qdrant_version: NonEmptyString
    neo4j_version: NonEmptyString
    qdrant_client_version: NonEmptyString
    training_config_hash: Sha256Hex
    train_validation_split_hash: Sha256Hex
    runtime_fingerprint: NonEmptyString
    runtime_semantics_version: NonEmptyString
    hardware_fingerprint: Sha256Hex
    dependency_versions: FrozenJsonMapping
    training_matrix_sha256: Sha256Hex


class CostModelArtifactManifest(FrozenModel):
    schema_version: Literal["cost_model_manifest_v1"] = "cost_model_manifest_v1"
    kind: ModelKind
    status: ArtifactStatus
    estimator_type: NonEmptyString = ESTIMATOR_TYPE
    model: ModelManifest
    feature_names: tuple[NonEmptyString, ...] = Field(min_length=1)
    raw_embeddings_used: Literal[False] = False
    training_matrix_sha256: Sha256Hex
    training_row_count: Annotated[int, Field(ge=1)]
    validation_row_count: Annotated[int, Field(ge=1)]
    hardware_fingerprint: Sha256Hex
    dependency_versions: FrozenJsonMapping
    trusted_types: tuple[str, ...] = ()
    gate_failures: tuple[NonEmptyString, ...] = ()

    @model_validator(mode="after")
    def _status_matches_failures(self) -> CostModelArtifactManifest:
        if (self.status is ArtifactStatus.SERVING_ELIGIBLE) is not (not self.gate_failures):
            raise ValueError("serving eligibility must exactly reflect model gate failures")
        if self.model.artifact_version != "cost_model_artifact_v1":
            raise ValueError("cost model artifact version is unsupported")
        if self.estimator_type != ESTIMATOR_TYPE:
            raise ValueError("cost model estimator type is unsupported")
        return self

    @property
    def sha256(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.model_dump(mode="json"))).hexdigest()


class CompatibilityContext(FrozenModel):
    feature_schema_version: NonEmptyString
    plan_catalog_hash: Sha256Hex
    corpus_version: NonEmptyString
    qrels_version: NonEmptyString
    embedding_model_revision: NonEmptyString
    extractor_version: NonEmptyString
    qdrant_version: NonEmptyString
    neo4j_version: NonEmptyString
    qdrant_client_version: NonEmptyString
    runtime_fingerprint: NonEmptyString
    runtime_semantics_version: NonEmptyString
    hardware_fingerprint: Sha256Hex
    dependency_versions: FrozenJsonMapping


class CompatibilityResult(FrozenModel):
    load_allowed: bool
    default_activation_allowed: bool
    critical_mismatches: tuple[NonEmptyString, ...] = ()
    warnings: tuple[NonEmptyString, ...] = ()

    @model_validator(mode="after")
    def _consistent(self) -> CompatibilityResult:
        if self.load_allowed is not (not self.critical_mismatches):
            raise ValueError("load compatibility must reflect critical mismatches")
        if self.default_activation_allowed and (not self.load_allowed or self.warnings):
            raise ValueError("default activation requires exact compatibility")
        return self


class CostModelBundleManifest(FrozenModel):
    schema_version: Literal["cost_model_bundle_v1"] = "cost_model_bundle_v1"
    status: ArtifactStatus
    quality_manifest_sha256: Sha256Hex
    latency_manifest_sha256: Sha256Hex
    model_report_sha256: Sha256Hex
    training_matrix_sha256: Sha256Hex
    gate_failures: tuple[NonEmptyString, ...] = ()

    @model_validator(mode="after")
    def _status_matches(self) -> CostModelBundleManifest:
        if (self.status is ArtifactStatus.SERVING_ELIGIBLE) is not (not self.gate_failures):
            raise ValueError("bundle status must reflect gate failures")
        return self


def load_trusted_types(path: Path | None = None) -> tuple[str, ...]:
    try:
        if path is not None:
            payload = json.loads(path.read_text(encoding="utf-8"))
        elif DEFAULT_TRUSTED_TYPES_PATH.is_file():
            payload = json.loads(DEFAULT_TRUSTED_TYPES_PATH.read_text(encoding="utf-8"))
        else:
            payload = json.loads(
                files("ragplan.resources")
                .joinpath("skops_trusted_types_v1.json")
                .read_text(encoding="utf-8")
            )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("skops trusted-type allowlist is unavailable or invalid") from exc
    if not isinstance(payload, dict) or set(payload) != {"schema_version", "trusted_types"}:
        raise ValueError("skops trusted-type allowlist has invalid fields")
    if payload["schema_version"] != "skops_trusted_types_v1":
        raise ValueError("skops trusted-type allowlist version is unsupported")
    trusted = payload["trusted_types"]
    if (
        not isinstance(trusted, list)
        or not all(isinstance(item, str) and item for item in trusted)
        or len(trusted) != len(set(trusted))
        or trusted != sorted(trusted)
    ):
        raise ValueError("skops trusted types must be unique sorted strings")
    return tuple(trusted)


def installed_dependency_versions() -> dict[str, str]:
    packages = ("numpy", "scikit-learn", "skops", "qdrant-client", "neo4j")
    result: dict[str, str] = {}
    for package in packages:
        try:
            result[package] = version(package)
        except PackageNotFoundError as exc:
            raise RuntimeError(f"required model dependency is not installed: {package}") from exc
    return result


def hardware_fingerprint(environment: EnvironmentManifest) -> str:
    payload = {
        "os_name": environment.os_name,
        "os_release": environment.os_release,
        "machine": environment.machine,
        "cpu_model": environment.cpu_model,
        "logical_cpu_count": environment.logical_cpu_count,
        "cpu_governor": environment.cpu_governor,
        "container_resource_limits": environment.container_resource_limits,
    }
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def save_cost_model(
    estimator: HistGradientBoostingRegressor,
    artifact_path: Path,
    *,
    kind: ModelKind,
    context: ArtifactContext,
    feature_names: tuple[str, ...],
    validation_metrics: Mapping[str, float],
    status: ArtifactStatus,
    gate_failures: tuple[str, ...],
    training_row_count: int,
    validation_row_count: int,
    trusted_types: tuple[str, ...] = (),
) -> CostModelArtifactManifest:
    if artifact_path.suffix != ".skops":
        raise ValueError("cost model artifact path must end in .skops")
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=artifact_path.parent,
        prefix=f".{artifact_path.name}.",
        suffix=".skops",
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    temporary.unlink()
    try:
        sio.dump(estimator, temporary)
        with temporary.open("rb") as source:
            os.fsync(source.fileno())
        os.chmod(temporary, 0o644)
        artifact_sha256 = _file_sha256(temporary)
        os.replace(temporary, artifact_path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    model = ModelManifest(
        model_name=f"ragplan_{kind.value}",
        model_version=f"{kind.value}_v1_{artifact_sha256[:16]}",
        artifact_version="cost_model_artifact_v1",
        artifact_sha256=artifact_sha256,
        feature_schema_version=context.feature_schema_version,
        plan_catalog_hash=context.plan_catalog_hash,
        corpus_version=context.corpus_version,
        qrels_version=context.qrels_version,
        embedding_model_revision=context.embedding_model_revision,
        extractor_version=context.extractor_version,
        qdrant_version=context.qdrant_version,
        neo4j_version=context.neo4j_version,
        qdrant_client_version=context.qdrant_client_version,
        training_config_hash=context.training_config_hash,
        train_validation_split_hash=context.train_validation_split_hash,
        runtime_fingerprint=context.runtime_fingerprint,
        runtime_semantics_version=context.runtime_semantics_version,
        validation_metrics=dict(validation_metrics),
    )
    manifest = CostModelArtifactManifest(
        kind=kind,
        status=status,
        model=model,
        feature_names=feature_names,
        training_matrix_sha256=context.training_matrix_sha256,
        training_row_count=training_row_count,
        validation_row_count=validation_row_count,
        hardware_fingerprint=context.hardware_fingerprint,
        dependency_versions=context.dependency_versions,
        trusted_types=trusted_types,
        gate_failures=gate_failures,
    )
    write_json_model(manifest_path_for(artifact_path), manifest)
    return manifest


def load_cost_model(
    artifact_path: Path,
    *,
    compatibility: CompatibilityContext,
    trusted_types_path: Path | None = None,
    require_serving_eligible: bool = True,
) -> tuple[HistGradientBoostingRegressor, CostModelArtifactManifest, CompatibilityResult]:
    if artifact_path.suffix != ".skops":
        raise RAGPlanError(
            ErrorCode.MODEL_INCOMPATIBLE,
            "only .skops cost model artifacts are accepted",
            retryable=False,
        )
    try:
        manifest = CostModelArtifactManifest.model_validate_json(
            manifest_path_for(artifact_path).read_text(encoding="utf-8")
        )
    except Exception as exc:
        raise RAGPlanError(
            ErrorCode.MODEL_INCOMPATIBLE,
            "cost model manifest is invalid",
            retryable=False,
        ) from exc
    if _file_sha256(artifact_path) != manifest.model.artifact_sha256:
        raise RAGPlanError(
            ErrorCode.MODEL_INCOMPATIBLE,
            "cost model artifact checksum mismatch",
            retryable=False,
        )
    allowed = load_trusted_types(trusted_types_path)
    if tuple(allowed) != manifest.trusted_types:
        raise RAGPlanError(
            ErrorCode.MODEL_INCOMPATIBLE,
            "cost model trusted-type policy differs from its manifest",
            retryable=False,
        )
    try:
        observed_untrusted = tuple(sorted(sio.get_untrusted_types(file=artifact_path)))
    except Exception as exc:
        raise RAGPlanError(
            ErrorCode.MODEL_INCOMPATIBLE,
            "cost model artifact cannot be inspected safely",
            retryable=False,
        ) from exc
    if any(item not in allowed for item in observed_untrusted):
        raise RAGPlanError(
            ErrorCode.MODEL_INCOMPATIBLE,
            "cost model artifact contains an untrusted type",
            retryable=False,
        )
    result = check_compatibility(manifest, compatibility)
    if not result.load_allowed:
        raise RAGPlanError(
            ErrorCode.MODEL_INCOMPATIBLE,
            "cost model artifact is incompatible with this runtime",
            retryable=False,
        )
    if require_serving_eligible and (
        manifest.status is not ArtifactStatus.SERVING_ELIGIBLE
        or not result.default_activation_allowed
    ):
        raise RAGPlanError(
            ErrorCode.MODE_UNAVAILABLE,
            "cost model is research-only or requires operator review",
            retryable=False,
        )
    try:
        loaded = sio.load(artifact_path, trusted=list(allowed))
    except Exception as exc:
        raise RAGPlanError(
            ErrorCode.MODEL_INCOMPATIBLE,
            "cost model artifact could not be loaded",
            retryable=False,
        ) from exc
    if type(loaded) is not HistGradientBoostingRegressor:
        raise RAGPlanError(
            ErrorCode.MODEL_INCOMPATIBLE,
            "cost model estimator type is not allowed",
            retryable=False,
        )
    return cast(HistGradientBoostingRegressor, loaded), manifest, result


def check_compatibility(
    manifest: CostModelArtifactManifest,
    context: CompatibilityContext,
) -> CompatibilityResult:
    observed = manifest.model
    critical_fields = {
        "feature_schema_version": (
            observed.feature_schema_version,
            context.feature_schema_version,
        ),
        "plan_catalog_hash": (observed.plan_catalog_hash, context.plan_catalog_hash),
        "corpus_version": (observed.corpus_version, context.corpus_version),
        "qrels_version": (observed.qrels_version, context.qrels_version),
        "embedding_model_revision": (
            observed.embedding_model_revision,
            context.embedding_model_revision,
        ),
        "extractor_version": (observed.extractor_version, context.extractor_version),
        "qdrant_version": (observed.qdrant_version, context.qdrant_version),
        "neo4j_version": (observed.neo4j_version, context.neo4j_version),
        "qdrant_client_version": (
            observed.qdrant_client_version,
            context.qdrant_client_version,
        ),
        "runtime_fingerprint": (observed.runtime_fingerprint, context.runtime_fingerprint),
        "runtime_semantics_version": (
            observed.runtime_semantics_version,
            context.runtime_semantics_version,
        ),
        "dependency_versions": (
            dict(manifest.dependency_versions),
            dict(context.dependency_versions),
        ),
    }
    mismatches = tuple(name for name, (left, right) in critical_fields.items() if left != right)
    warnings = (
        ("hardware_fingerprint_mismatch",)
        if manifest.hardware_fingerprint != context.hardware_fingerprint
        else ()
    )
    load_allowed = not mismatches
    return CompatibilityResult(
        load_allowed=load_allowed,
        default_activation_allowed=load_allowed and not warnings,
        critical_mismatches=mismatches,
        warnings=warnings,
    )


def manifest_path_for(artifact_path: Path) -> Path:
    return artifact_path.with_suffix(".manifest.json")


def estimator_type_name(estimator: Any) -> str:
    return f"{type(estimator).__module__}.{type(estimator).__qualname__}"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


__all__ = [
    "ArtifactContext",
    "ArtifactStatus",
    "CompatibilityContext",
    "CompatibilityResult",
    "CostModelArtifactManifest",
    "CostModelBundleManifest",
    "ModelKind",
    "check_compatibility",
    "estimator_type_name",
    "hardware_fingerprint",
    "installed_dependency_versions",
    "load_cost_model",
    "load_trusted_types",
    "manifest_path_for",
    "save_cost_model",
]
