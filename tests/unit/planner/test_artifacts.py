from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import skops.io as sio  # type: ignore[import-untyped]
from sklearn.ensemble import HistGradientBoostingRegressor  # type: ignore[import-untyped]

from ragplan.benchmark.artifacts import write_json_model
from ragplan.core.errors import ErrorCode, RAGPlanError
from ragplan.planner.artifacts import (
    ArtifactContext,
    ArtifactStatus,
    CompatibilityContext,
    ModelKind,
    check_compatibility,
    load_cost_model,
    manifest_path_for,
    save_cost_model,
)

pytestmark = pytest.mark.unit


def _context() -> ArtifactContext:
    return ArtifactContext(
        feature_schema_version="cost_model_features_v1:qf_v1",
        plan_catalog_hash="1" * 64,
        corpus_version="corpus-v1",
        qrels_version="qrels-v1",
        embedding_model_revision="embedding-v1",
        extractor_version="extractor-v1",
        qdrant_version="1.18.2",
        neo4j_version="5.26.28",
        qdrant_client_version="1.18.0",
        training_config_hash="2" * 64,
        train_validation_split_hash="3" * 64,
        runtime_fingerprint="runtime-v1",
        runtime_semantics_version="v1",
        hardware_fingerprint="4" * 64,
        dependency_versions={"numpy": "2.5.1", "scikit-learn": "1.9.0"},
        training_matrix_sha256="5" * 64,
    )


def _compatibility() -> CompatibilityContext:
    context = _context()
    return CompatibilityContext(
        feature_schema_version=context.feature_schema_version,
        plan_catalog_hash=context.plan_catalog_hash,
        corpus_version=context.corpus_version,
        qrels_version=context.qrels_version,
        embedding_model_revision=context.embedding_model_revision,
        extractor_version=context.extractor_version,
        qdrant_version=context.qdrant_version,
        neo4j_version=context.neo4j_version,
        qdrant_client_version=context.qdrant_client_version,
        runtime_fingerprint=context.runtime_fingerprint,
        runtime_semantics_version=context.runtime_semantics_version,
        hardware_fingerprint=context.hardware_fingerprint,
        dependency_versions=context.dependency_versions,
    )


def _estimator() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(random_state=1).fit(
        np.asarray([[0.0], [1.0], [2.0], [3.0]]),
        np.asarray([0.0, 1.0, 2.0, 3.0]),
    )


def test_skops_round_trip_checksum_and_research_only_guard(tmp_path: Path) -> None:
    path = tmp_path / "quality.skops"
    manifest = save_cost_model(
        _estimator(),
        path,
        kind=ModelKind.QUALITY,
        context=_context(),
        feature_names=("feature",),
        validation_metrics={"mae": 0.2},
        status=ArtifactStatus.RESEARCH_ONLY,
        gate_failures=("quality_mae_gt_0.10",),
        training_row_count=4,
        validation_row_count=2,
    )
    loaded, observed, result = load_cost_model(
        path,
        compatibility=_compatibility(),
        require_serving_eligible=False,
    )
    assert type(loaded) is HistGradientBoostingRegressor
    assert observed == manifest
    assert result.load_allowed is True

    with pytest.raises(RAGPlanError) as blocked:
        load_cost_model(path, compatibility=_compatibility())
    assert blocked.value.code is ErrorCode.MODE_UNAVAILABLE

    path.write_bytes(path.read_bytes() + b"corruption")
    with pytest.raises(RAGPlanError, match="checksum"):
        load_cost_model(
            path,
            compatibility=_compatibility(),
            require_serving_eligible=False,
        )


@pytest.mark.parametrize(
    "field,value",
    (
        ("feature_schema_version", "wrong"),
        ("plan_catalog_hash", "f" * 64),
        ("corpus_version", "wrong"),
        ("qrels_version", "wrong"),
        ("embedding_model_revision", "wrong"),
        ("extractor_version", "wrong"),
        ("qdrant_version", "wrong"),
        ("neo4j_version", "wrong"),
        ("qdrant_client_version", "wrong"),
        ("runtime_fingerprint", "wrong"),
        ("runtime_semantics_version", "wrong"),
        ("dependency_versions", {"numpy": "0"}),
    ),
)
def test_every_critical_compatibility_mismatch_is_rejected(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    path = tmp_path / "quality.skops"
    manifest = save_cost_model(
        _estimator(),
        path,
        kind=ModelKind.QUALITY,
        context=_context(),
        feature_names=("feature",),
        validation_metrics={"mae": 0.01},
        status=ArtifactStatus.SERVING_ELIGIBLE,
        gate_failures=(),
        training_row_count=4,
        validation_row_count=2,
    )
    context = _compatibility().model_copy(update={field: value})
    result = check_compatibility(manifest, context)
    assert result.load_allowed is False
    assert field in result.critical_mismatches
    with pytest.raises(RAGPlanError) as caught:
        load_cost_model(path, compatibility=context, require_serving_eligible=False)
    assert caught.value.code is ErrorCode.MODEL_INCOMPATIBLE


def test_hardware_mismatch_requires_operator_review(tmp_path: Path) -> None:
    path = tmp_path / "quality.skops"
    manifest = save_cost_model(
        _estimator(),
        path,
        kind=ModelKind.QUALITY,
        context=_context(),
        feature_names=("feature",),
        validation_metrics={"mae": 0.01},
        status=ArtifactStatus.SERVING_ELIGIBLE,
        gate_failures=(),
        training_row_count=4,
        validation_row_count=2,
    )
    different = _compatibility().model_copy(update={"hardware_fingerprint": "9" * 64})
    result = check_compatibility(manifest, different)
    assert result.load_allowed is True
    assert result.default_activation_allowed is False
    assert result.warnings == ("hardware_fingerprint_mismatch",)


def test_untrusted_type_and_non_skops_suffix_are_rejected(tmp_path: Path) -> None:
    class Untrusted:
        value = 1

    path = tmp_path / "quality.skops"
    manifest = save_cost_model(
        _estimator(),
        path,
        kind=ModelKind.QUALITY,
        context=_context(),
        feature_names=("feature",),
        validation_metrics={"mae": 0.01},
        status=ArtifactStatus.SERVING_ELIGIBLE,
        gate_failures=(),
        training_row_count=4,
        validation_row_count=2,
    )
    sio.dump(Untrusted(), path)
    artifact_hash = __import__("hashlib").sha256(path.read_bytes()).hexdigest()
    replaced_model = manifest.model.model_copy(update={"artifact_sha256": artifact_hash})
    write_json_model(
        manifest_path_for(path),
        manifest.model_copy(update={"model": replaced_model}),
    )
    with pytest.raises(RAGPlanError, match="untrusted"):
        load_cost_model(path, compatibility=_compatibility())

    pickle_path = tmp_path / "quality.pkl"
    pickle_path.write_bytes(b"not a model")
    with pytest.raises(RAGPlanError, match="only .skops"):
        load_cost_model(pickle_path, compatibility=_compatibility())
