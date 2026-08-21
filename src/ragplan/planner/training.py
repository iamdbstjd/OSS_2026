"""Leakage-safe Stage 11 dataset construction and deterministic feature encoding."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Self

import numpy as np
from numpy.typing import NDArray
from pydantic import Field, model_validator

from ragplan.benchmark.contracts import QueryTag, SourceDataset, SplitName, canonical_sha256
from ragplan.benchmark.profile_records import (
    P0_PROFILE_PLAN_IDS,
    ProfilePlanFeatures,
    TrainingMatrixRow,
)
from ragplan.core.models import FrozenModel, QueryFeatures

COST_FEATURE_SCHEMA_VERSION: Final = "cost_model_features_v1:qf_v1"
SOURCE_CATEGORIES: Final = tuple(item.value for item in SourceDataset)
TAG_CATEGORIES: Final = tuple(item.value for item in QueryTag)
PLAN_CATEGORIES: Final = P0_PROFILE_PLAN_IDS

_QUERY_NUMERIC: Final = (
    "token_count",
    "entity_count",
    "entity_density",
    "relation_signal",
    "multi_hop_signal",
    "comparison_signal",
    "aggregation_signal",
    "global_signal",
    "final_top_k",
)
_PLAN_NUMERIC: Final = (
    "vector_enabled",
    "graph_enabled",
    "vector_top_k",
    "graph_top_k",
    "graph_depth",
    "vector_weight",
    "graph_weight",
    "rerank_enabled",
    "rerank_top_k",
)
FEATURE_NAMES: Final = (
    *(f"query_{name}" for name in _QUERY_NUMERIC),
    *(f"plan_{name}" for name in _PLAN_NUMERIC),
    *(f"source_{value}" for value in SOURCE_CATEGORIES),
    *(f"tag_{value}" for value in TAG_CATEGORIES),
    *(f"plan_id_{value}" for value in PLAN_CATEGORIES),
)


@dataclass(frozen=True, slots=True)
class SampleSet:
    features: NDArray[np.float64]
    targets: NDArray[np.float64]
    rows: tuple[TrainingMatrixRow, ...]

    def __post_init__(self) -> None:
        if self.features.ndim != 2 or self.features.shape[1] != len(FEATURE_NAMES):
            raise ValueError("sample feature matrix does not match the frozen schema")
        if self.targets.ndim != 1 or self.features.shape[0] != self.targets.shape[0]:
            raise ValueError("sample features and targets have different row counts")
        if self.features.shape[0] != len(self.rows) or not self.rows:
            raise ValueError("sample row metadata must be non-empty and aligned")
        if not np.isfinite(self.features).all() or not np.isfinite(self.targets).all():
            raise ValueError("sample data contains NaN or infinite values")


@dataclass(frozen=True, slots=True)
class TrainingDatasets:
    all_rows: tuple[TrainingMatrixRow, ...]
    quality_train: SampleSet
    quality_validation: SampleSet
    latency_train: SampleSet
    latency_validation: SampleSet
    train_validation_split_hash: str


class OfflineCostModelInput(FrozenModel):
    """Explicit benchmark-only inputs for the frozen Stage 11 feature schema.

    ``source_dataset`` and ``query_tags`` are not available to the public online
    analyzer.  Keeping them in a separately named contract prevents an offline
    research artifact from being mistaken for a serving-compatible model.
    """

    query_features: QueryFeatures
    source_dataset: SourceDataset
    query_tags: tuple[QueryTag, ...] = Field(min_length=1)
    plan_features: ProfilePlanFeatures

    @model_validator(mode="after")
    def _check_identity(self) -> Self:
        if len(set(self.query_tags)) != len(self.query_tags):
            raise ValueError("offline cost-model query tags must be unique")
        if self.plan_features.plan_id not in PLAN_CATEGORIES:
            raise ValueError("offline cost-model plan is outside the P0 catalog")
        return self

    @classmethod
    def from_matrix_row(cls, row: TrainingMatrixRow) -> OfflineCostModelInput:
        return cls(
            query_features=row.query_features,
            source_dataset=row.source_dataset,
            query_tags=row.query_tags,
            plan_features=row.plan_features,
        )

    @property
    def sha256(self) -> str:
        payload = self.model_dump(mode="json")
        payload["query_tags"] = sorted(payload["query_tags"])
        return canonical_sha256(payload)


def load_training_matrix(path: Path) -> tuple[TrainingMatrixRow, ...]:
    rows: list[TrainingMatrixRow] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                raise ValueError(f"training matrix has an empty row at line {line_number}")
            try:
                rows.append(TrainingMatrixRow.model_validate_json(line))
            except (ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"training matrix row {line_number} is invalid") from exc
    if not rows:
        raise ValueError("training matrix is empty")
    return tuple(rows)


def build_training_datasets(rows: Sequence[TrainingMatrixRow]) -> TrainingDatasets:
    ordered = tuple(
        sorted(
            rows,
            key=lambda row: (
                row.query_id,
                row.latency_budget_ms,
                int(row.plan_id[1:]),
            ),
        )
    )
    _validate_matrix(ordered)
    quality_train_rows = tuple(
        row for row in ordered if row.split is SplitName.TRAIN and row.quality_label_valid
    )
    quality_validation_rows = tuple(
        row for row in ordered if row.split is SplitName.VALIDATION and row.quality_label_valid
    )
    latency_train_rows = tuple(
        row for row in ordered if row.split is SplitName.TRAIN and row.execution_latency_label_valid
    )
    latency_validation_rows = tuple(
        row
        for row in ordered
        if row.split is SplitName.VALIDATION and row.execution_latency_label_valid
    )
    split_rows = [
        {"query_id": query_id, "split": split.value}
        for query_id, split in sorted(_query_splits(ordered).items())
    ]
    return TrainingDatasets(
        all_rows=ordered,
        quality_train=_sample_set(quality_train_rows, target="quality"),
        quality_validation=_sample_set(quality_validation_rows, target="quality"),
        latency_train=_sample_set(latency_train_rows, target="latency"),
        latency_validation=_sample_set(latency_validation_rows, target="latency"),
        train_validation_split_hash=canonical_sha256(split_rows),
    )


def encode_rows(rows: Sequence[TrainingMatrixRow]) -> NDArray[np.float64]:
    return encode_offline_inputs(tuple(OfflineCostModelInput.from_matrix_row(row) for row in rows))


def encode_offline_inputs(
    inputs: Sequence[OfflineCostModelInput],
) -> NDArray[np.float64]:
    """Encode only explicitly labelled offline inputs in the frozen order."""

    encoded = np.asarray([_encode_offline_input(item) for item in inputs], dtype=np.float64)
    if encoded.ndim != 2 or encoded.shape != (len(inputs), len(FEATURE_NAMES)):
        raise ValueError("encoded feature matrix has an unexpected shape")
    if not np.isfinite(encoded).all():
        raise ValueError("encoded feature matrix contains NaN or infinite values")
    return encoded


def _sample_set(
    rows: tuple[TrainingMatrixRow, ...],
    *,
    target: str,
) -> SampleSet:
    if target == "quality":
        targets = np.asarray([row.recall_at_10 for row in rows], dtype=np.float64)
    elif target == "latency":
        targets = np.asarray(
            [_required_latency(row) for row in rows],
            dtype=np.float64,
        )
    else:
        raise ValueError("unknown training target")
    return SampleSet(features=encode_rows(rows), targets=targets, rows=rows)


def _required_latency(row: TrainingMatrixRow) -> float:
    value = row.p95_execution_latency_ms
    if value is None or not row.execution_latency_label_valid:
        raise ValueError("latency dataset contains an invalid execution label")
    return value


def _encode_row(row: TrainingMatrixRow) -> tuple[float, ...]:
    return _encode_offline_input(OfflineCostModelInput.from_matrix_row(row))


def _encode_offline_input(item: OfflineCostModelInput) -> tuple[float, ...]:
    query = item.query_features
    plan = item.plan_features
    query_values = tuple(float(getattr(query, name)) for name in _QUERY_NUMERIC)
    plan_values = tuple(float(getattr(plan, name)) for name in _PLAN_NUMERIC)
    source_values = tuple(
        float(item.source_dataset.value == category) for category in SOURCE_CATEGORIES
    )
    tags = {tag.value for tag in item.query_tags}
    tag_values = tuple(float(category in tags) for category in TAG_CATEGORIES)
    plan_id_values = tuple(float(plan.plan_id == category) for category in PLAN_CATEGORIES)
    return (*query_values, *plan_values, *source_values, *tag_values, *plan_id_values)


def _validate_matrix(rows: tuple[TrainingMatrixRow, ...]) -> None:
    if not rows:
        raise ValueError("training matrix must not be empty")
    identities = {
        (
            row.profile_protocol_sha256,
            row.environment_manifest_sha256,
            row.benchmark_manifest_sha256,
            row.split_hash,
            row.qrels_sha256,
            row.corpus_version,
            row.corpus_chunk_ids_sha256,
            row.plan_catalog_sha256,
            row.query_feature_config_sha256,
            row.runtime_semantics_version,
        )
        for row in rows
    }
    if len(identities) != 1:
        raise ValueError("training matrix contains mixed version/hash identities")
    keys = [(row.query_id, row.plan_id, row.latency_budget_ms) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("training matrix contains duplicate query-plan-budget rows")
    if any(row.split is SplitName.TEST for row in rows):
        raise ValueError("training matrix must not contain held-out test queries")
    query_features: dict[str, object] = {}
    plan_features: dict[str, object] = {}
    for row in rows:
        existing_query = query_features.setdefault(row.query_id, row.query_features)
        if existing_query != row.query_features:
            raise ValueError("query features change across one query group")
        existing_plan = plan_features.setdefault(row.plan_id, row.plan_features)
        if existing_plan != row.plan_features:
            raise ValueError("plan features change across one plan ID")
    if not set(row.plan_id for row in rows) <= set(PLAN_CATEGORIES):
        raise ValueError("training matrix contains a plan outside the P0 catalog")
    _query_splits(rows)
    numeric_values = (
        value
        for row in rows
        for value in (
            row.recall_at_10,
            row.fallback_rate,
            row.budget_violation_rate,
            *(_encode_row(row)),
        )
    )
    if any(not math.isfinite(float(value)) for value in numeric_values):
        raise ValueError("training matrix contains NaN or infinite values")


def _query_splits(rows: Sequence[TrainingMatrixRow]) -> dict[str, SplitName]:
    result: dict[str, SplitName] = {}
    for row in rows:
        existing = result.setdefault(row.query_id, row.split)
        if existing is not row.split:
            raise ValueError("one query appears in multiple dataset splits")
    if SplitName.TRAIN not in result.values() or SplitName.VALIDATION not in result.values():
        raise ValueError("training matrix requires both train and validation queries")
    return result


__all__ = [
    "COST_FEATURE_SCHEMA_VERSION",
    "FEATURE_NAMES",
    "OfflineCostModelInput",
    "PLAN_CATEGORIES",
    "SampleSet",
    "TrainingDatasets",
    "build_training_datasets",
    "encode_offline_inputs",
    "encode_rows",
    "load_training_matrix",
]
