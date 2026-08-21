"""Deterministic Stage 11 Recall@10 regression model."""

from __future__ import annotations

import hashlib
from typing import Literal, Self

import numpy as np
from numpy.typing import NDArray
from pydantic import model_validator
from sklearn.ensemble import HistGradientBoostingRegressor  # type: ignore[import-untyped]

from ragplan.benchmark.contracts import canonical_json_bytes
from ragplan.core.models import FrozenModel
from ragplan.planner.training import SampleSet


class QualityTrainingConfig(FrozenModel):
    schema_version: Literal["quality_hgb_v1"] = "quality_hgb_v1"
    loss: Literal["squared_error"] = "squared_error"
    learning_rate: float = 0.05
    max_iter: Literal[300] = 300
    max_leaf_nodes: Literal[31] = 31
    min_samples_leaf: Literal[20] = 20
    l2_regularization: float = 1.0
    early_stopping: Literal[False] = False
    random_seed: Literal[20260809] = 20260809

    @model_validator(mode="after")
    def _frozen_hyperparameters(self) -> Self:
        if self.learning_rate != 0.05 or self.l2_regularization != 1.0:
            raise ValueError("quality model hyperparameters are immutable in v1")
        return self

    @property
    def sha256(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.model_dump(mode="json"))).hexdigest()


def train_quality_model(
    samples: SampleSet,
    *,
    config: QualityTrainingConfig | None = None,
) -> HistGradientBoostingRegressor:
    selected = config if config is not None else QualityTrainingConfig()
    estimator = HistGradientBoostingRegressor(
        loss=selected.loss,
        learning_rate=selected.learning_rate,
        max_iter=selected.max_iter,
        max_leaf_nodes=selected.max_leaf_nodes,
        min_samples_leaf=selected.min_samples_leaf,
        l2_regularization=selected.l2_regularization,
        early_stopping=selected.early_stopping,
        random_state=selected.random_seed,
    )
    estimator.fit(samples.features, samples.targets)
    return estimator


def predict_quality(
    estimator: HistGradientBoostingRegressor,
    features: NDArray[np.float64],
) -> NDArray[np.float64]:
    prediction = np.asarray(estimator.predict(features), dtype=np.float64)
    if prediction.shape != (features.shape[0],) or not np.isfinite(prediction).all():
        raise ValueError("quality model returned invalid predictions")
    return np.clip(prediction, 0.0, 1.0)


__all__ = ["QualityTrainingConfig", "predict_quality", "train_quality_model"]
