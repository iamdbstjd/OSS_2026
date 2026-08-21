"""Stage 10 Oracle@Budget labels and deterministic plan distributions."""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from ragplan.benchmark.artifacts import write_json, write_json_model
from ragplan.benchmark.contracts import SplitName, canonical_json_bytes
from ragplan.benchmark.profile_records import ProfileRunManifest, TrainingMatrixRow
from ragplan.core.models import FrozenModel, NonEmptyString, PlanId, Sha256Hex


class OracleSelection(FrozenModel):
    query_id: NonEmptyString
    split: SplitName
    latency_budget_ms: Annotated[int, Field(ge=25, le=5000)]
    oracle_plan_id: PlanId | None = None
    oracle_recall_at_10: float | None = Field(default=None, ge=0.0, le=1.0)
    oracle_p95_execution_latency_ms: float | None = Field(default=None, ge=0.0)
    feasible_plan_count: Annotated[int, Field(ge=0)]
    no_feasible_reason: str | None = None

    @model_validator(mode="after")
    def _selection_shape(self) -> Self:
        selected = self.oracle_plan_id is not None
        if selected is not (
            self.oracle_recall_at_10 is not None
            and self.oracle_p95_execution_latency_ms is not None
            and self.feasible_plan_count > 0
        ):
            raise ValueError("Oracle selection metrics require a selected feasible plan")
        if selected is (self.no_feasible_reason is not None):
            raise ValueError("only an empty Oracle selection requires a reason")
        return self


class OracleDistribution(FrozenModel):
    latency_budget_ms: Annotated[int, Field(ge=25, le=5000)]
    plan_id: PlanId | None = None
    query_count: Annotated[int, Field(ge=1)]


class OracleReport(FrozenModel):
    schema_version: Literal["oracle_at_budget_v1"] = "oracle_at_budget_v1"
    run_id: NonEmptyString
    source_training_matrix_sha256: Sha256Hex
    profile_protocol_sha256: Sha256Hex
    environment_manifest_sha256: Sha256Hex
    benchmark_manifest_sha256: Sha256Hex
    split_hash: Sha256Hex
    qrels_sha256: Sha256Hex
    corpus_version: NonEmptyString
    plan_catalog_sha256: Sha256Hex
    query_feature_config_sha256: Sha256Hex
    runtime_semantics_version: Literal["v1"] = "v1"
    tie_break: Literal["higher_recall_at_10,lower_p95_latency,lower_graph_depth,lower_plan_id"] = (
        "higher_recall_at_10,lower_p95_latency,lower_graph_depth,lower_plan_id"
    )
    selections: tuple[OracleSelection, ...] = Field(min_length=1)
    distribution: tuple[OracleDistribution, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_labels(self) -> Self:
        keys = tuple((item.query_id, item.latency_budget_ms) for item in self.selections)
        if len(keys) != len(set(keys)):
            raise ValueError("Oracle report contains duplicate query-budget labels")
        if keys != tuple(sorted(keys)):
            raise ValueError("Oracle selections must use deterministic query-budget order")
        return self


def choose_oracle_plan(rows: Sequence[TrainingMatrixRow]) -> TrainingMatrixRow | None:
    """Select the measured Oracle using the normative four-level tie-break."""

    candidates = tuple(rows)
    if not candidates:
        return None
    query_budget = {(item.query_id, item.latency_budget_ms) for item in candidates}
    if len(query_budget) != 1:
        raise ValueError("Oracle candidates must belong to one query and budget")
    budget = candidates[0].latency_budget_ms
    feasible = tuple(
        item
        for item in candidates
        if item.usable_for_model_training
        and item.p95_execution_latency_ms is not None
        and item.p95_execution_latency_ms <= budget
    )
    if not feasible:
        return None
    return min(
        feasible,
        key=lambda item: (
            -item.recall_at_10,
            item.p95_execution_latency_ms,
            item.plan_features.graph_depth,
            int(item.plan_id[1:]),
        ),
    )


def build_oracle_report(
    matrix: Sequence[TrainingMatrixRow],
    *,
    manifest: ProfileRunManifest,
) -> OracleReport:
    rows = tuple(matrix)
    _validate_matrix_identity(rows, manifest=manifest)
    grouped: dict[tuple[str, int], list[TrainingMatrixRow]] = {}
    for row in rows:
        grouped.setdefault((row.query_id, row.latency_budget_ms), []).append(row)
    expected = manifest.query_count * len(manifest.latency_budgets_ms)
    if len(grouped) != expected:
        raise ValueError("Oracle input is missing query-budget groups")
    selections: list[OracleSelection] = []
    for key in sorted(grouped):
        candidates = tuple(grouped[key])
        if {item.plan_id for item in candidates} != set(manifest.plan_ids):
            raise ValueError("Oracle input is missing one or more plan rows")
        winner = choose_oracle_plan(candidates)
        selections.append(
            OracleSelection(
                query_id=key[0],
                split=candidates[0].split,
                latency_budget_ms=key[1],
                oracle_plan_id=winner.plan_id if winner is not None else None,
                oracle_recall_at_10=winner.recall_at_10 if winner is not None else None,
                oracle_p95_execution_latency_ms=(
                    winner.p95_execution_latency_ms if winner is not None else None
                ),
                feasible_plan_count=sum(
                    item.usable_for_model_training
                    and item.p95_execution_latency_ms is not None
                    and item.p95_execution_latency_ms <= key[1]
                    for item in candidates
                ),
                no_feasible_reason=(
                    None if winner is not None else "no_complete_measured_p95_within_budget"
                ),
            )
        )
    counts = Counter((item.latency_budget_ms, item.oracle_plan_id) for item in selections)
    distribution = tuple(
        OracleDistribution(latency_budget_ms=budget, plan_id=plan_id, query_count=count)
        for (budget, plan_id), count in sorted(
            counts.items(),
            key=lambda item: (
                item[0][0],
                999 if item[0][1] is None else int(item[0][1][1:]),
            ),
        )
    )
    matrix_hash = hashlib.sha256(
        b"\n".join(
            canonical_json_bytes(item.model_dump(mode="json"))
            for item in sorted(
                rows,
                key=lambda item: (item.query_id, int(item.plan_id[1:]), item.latency_budget_ms),
            )
        )
        + b"\n"
    ).hexdigest()
    return OracleReport(
        run_id=manifest.run_id,
        source_training_matrix_sha256=matrix_hash,
        profile_protocol_sha256=manifest.profile_protocol_sha256,
        environment_manifest_sha256=manifest.environment_manifest_sha256,
        benchmark_manifest_sha256=manifest.benchmark_manifest_sha256,
        split_hash=manifest.split_hash,
        qrels_sha256=manifest.qrels_sha256,
        corpus_version=manifest.corpus_version,
        plan_catalog_sha256=manifest.plan_catalog_sha256,
        query_feature_config_sha256=manifest.query_feature_config_sha256,
        runtime_semantics_version=manifest.runtime_semantics_version,
        selections=tuple(selections),
        distribution=distribution,
    )


def write_oracle_artifact(run_dir: Path, report: OracleReport) -> None:
    directory = run_dir
    write_json_model(directory / "oracle_at_budget.json", report)
    checksum_path = directory / "checksums.json"
    existing: dict[str, object] = {}
    if checksum_path.is_file():
        import json

        decoded = json.loads(checksum_path.read_text(encoding="utf-8"))
        if not isinstance(decoded, dict):
            raise ValueError("profile checksums artifact must be an object")
        existing.update(decoded)
    existing["oracle_at_budget.json"] = hashlib.sha256(
        (directory / "oracle_at_budget.json").read_bytes()
    ).hexdigest()
    existing["source_training_matrix_sha256"] = report.source_training_matrix_sha256
    write_json(checksum_path, existing)


def _validate_matrix_identity(
    rows: tuple[TrainingMatrixRow, ...],
    *,
    manifest: ProfileRunManifest,
) -> None:
    expected_count = (
        manifest.query_count * len(manifest.plan_ids) * len(manifest.latency_budgets_ms)
    )
    if len(rows) != expected_count:
        raise ValueError("Oracle input training matrix has the wrong row count")
    versions = {
        (
            item.run_id,
            item.profile_protocol_sha256,
            item.environment_manifest_sha256,
            item.benchmark_manifest_sha256,
            item.split_hash,
            item.qrels_sha256,
            item.corpus_version,
            item.corpus_chunk_ids_sha256,
            item.plan_catalog_sha256,
            item.query_feature_config_sha256,
            item.runtime_semantics_version,
        )
        for item in rows
    }
    expected = {
        (
            manifest.run_id,
            manifest.profile_protocol_sha256,
            manifest.environment_manifest_sha256,
            manifest.benchmark_manifest_sha256,
            manifest.split_hash,
            manifest.qrels_sha256,
            manifest.corpus_version,
            manifest.corpus_chunk_ids_sha256,
            manifest.plan_catalog_sha256,
            manifest.query_feature_config_sha256,
            manifest.runtime_semantics_version,
        )
    }
    if versions != expected:
        raise ValueError("Oracle input version/hash bundle differs from the profile manifest")
    if any(item.split is SplitName.TEST for item in rows):
        raise ValueError("Oracle input cannot contain the held-out test split")


__all__ = [
    "OracleDistribution",
    "OracleReport",
    "OracleSelection",
    "build_oracle_report",
    "choose_oracle_plan",
    "write_oracle_artifact",
]
