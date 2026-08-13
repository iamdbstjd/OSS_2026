"""Reproducible Stage 4 human-review sampling, metrics, and graph-tier gate."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from enum import StrEnum
from importlib.resources import files
from pathlib import Path
from typing import Annotated, Final, Literal, Self

from pydantic import Field, model_validator

from ragplan.core.models import EntityType, FrozenModel, NonEmptyString, Sha256Hex

AUDIT_VERSION: Final[Literal["graph_extraction_v1"]] = "graph_extraction_v1"
AUDIT_SAMPLE_SIZE = 100
DOUBLE_REVIEW_SIZE = 20
ENTITY_F1_GATE: Final = 0.75
RELATION_PRECISION_GATE: Final = 0.70
DEFAULT_GRAPH_TIER_POLICY_PATH: Final = (
    Path(__file__).resolve().parents[3] / "configs" / "graph_tier_policy.json"
)


class AuditStatus(StrEnum):
    PENDING_HUMAN_REVIEW = "pending_human_review"
    COMPLETE = "complete"


class AuditEntityLabel(FrozenModel):
    start_char: Annotated[int, Field(ge=0)]
    end_char: Annotated[int, Field(ge=1)]
    entity_type: EntityType
    text: NonEmptyString

    @model_validator(mode="after")
    def _valid_span(self) -> Self:
        if self.end_char <= self.start_char:
            raise ValueError("entity label span must be non-empty")
        return self


class AuditRelationLabel(FrozenModel):
    source_start_char: Annotated[int, Field(ge=0)]
    source_end_char: Annotated[int, Field(ge=1)]
    target_start_char: Annotated[int, Field(ge=0)]
    target_end_char: Annotated[int, Field(ge=1)]
    predicate: NonEmptyString

    @model_validator(mode="after")
    def _valid_spans(self) -> Self:
        if self.source_end_char <= self.source_start_char:
            raise ValueError("relation source span must be non-empty")
        if self.target_end_char <= self.target_start_char:
            raise ValueError("relation target span must be non-empty")
        return self


class AuditSentence(FrozenModel):
    sentence_id: NonEmptyString
    source_chunk_id: NonEmptyString
    start_char: Annotated[int, Field(ge=0)]
    end_char: Annotated[int, Field(ge=1)]
    text: NonEmptyString
    selection_sha256: Sha256Hex
    predicted_entities: tuple[AuditEntityLabel, ...] = ()
    predicted_relations: tuple[AuditRelationLabel, ...] = ()

    @model_validator(mode="after")
    def _validate_sentence(self) -> Self:
        if self.end_char <= self.start_char:
            raise ValueError("audit sentence source span must be non-empty")
        expected = audit_selection_sha256(
            self.source_chunk_id,
            self.start_char,
            self.end_char,
            self.text,
        )
        if self.selection_sha256 != expected:
            raise ValueError("audit sentence selection hash is invalid")
        if self.sentence_id != f"{AUDIT_VERSION}:{expected}":
            raise ValueError("audit sentence ID is invalid")
        return self


class GraphAuditManifest(FrozenModel):
    schema_version: Literal["v1"] = "v1"
    audit_version: Literal["graph_extraction_v1"] = AUDIT_VERSION
    corpus_version: NonEmptyString
    benchmark_manifest_sha256: Sha256Hex
    split_hash: Sha256Hex
    extractor_version: NonEmptyString
    sentences: tuple[AuditSentence, ...] = Field(
        min_length=AUDIT_SAMPLE_SIZE,
        max_length=AUDIT_SAMPLE_SIZE,
    )
    second_reviewer_sentence_ids: tuple[NonEmptyString, ...] = Field(
        min_length=DOUBLE_REVIEW_SIZE,
        max_length=DOUBLE_REVIEW_SIZE,
    )
    sample_checksum: Sha256Hex

    @model_validator(mode="after")
    def _validate_sample(self) -> Self:
        ids = tuple(item.sentence_id for item in self.sentences)
        if len(set(ids)) != AUDIT_SAMPLE_SIZE:
            raise ValueError("audit sample sentence IDs must be unique")
        hashes = tuple(item.selection_sha256 for item in self.sentences)
        if hashes != tuple(sorted(hashes)):
            raise ValueError("audit sample must be ordered by SHA-256")
        expected_second = tuple(item.sentence_id for item in self.sentences[:DOUBLE_REVIEW_SIZE])
        if self.second_reviewer_sentence_ids != expected_second:
            raise ValueError("the first 20 hash-ordered samples must receive a second review")
        if self.sample_checksum != audit_sample_checksum(self.sentences):
            raise ValueError("audit sample checksum is invalid")
        return self


class AuditReview(FrozenModel):
    audit_version: Literal["graph_extraction_v1"] = AUDIT_VERSION
    sentence_id: NonEmptyString
    reviewer_id: NonEmptyString
    reviewer_role: Literal["primary", "secondary", "adjudicator"]
    entities: tuple[AuditEntityLabel, ...]
    relations: tuple[AuditRelationLabel, ...]
    completed: Literal[True] = True


class AuditEvaluation(FrozenModel):
    audit_version: Literal["graph_extraction_v1"] = AUDIT_VERSION
    status: AuditStatus
    reviewed_sentence_count: Annotated[int, Field(ge=0, le=AUDIT_SAMPLE_SIZE)]
    double_reviewed_sentence_count: Annotated[int, Field(ge=0, le=DOUBLE_REVIEW_SIZE)]
    adjudicated_sentence_count: Annotated[int, Field(ge=0, le=DOUBLE_REVIEW_SIZE)]
    entity_precision: Annotated[float, Field(ge=0.0, le=1.0)] | None = None
    entity_recall: Annotated[float, Field(ge=0.0, le=1.0)] | None = None
    entity_f1: Annotated[float, Field(ge=0.0, le=1.0)] | None = None
    relation_precision: Annotated[float, Field(ge=0.0, le=1.0)] | None = None
    relation_recall: Annotated[float, Field(ge=0.0, le=1.0)] | None = None
    entity_reviewer_agreement_f1: Annotated[float, Field(ge=0.0, le=1.0)] | None = None
    relation_reviewer_agreement_f1: Annotated[float, Field(ge=0.0, le=1.0)] | None = None
    graph_tier_enabled: bool
    gate_reason: NonEmptyString


class RuleGraphTierPolicy(FrozenModel):
    """Machine-readable fail-closed handoff consumed by the later rule planner."""

    schema_version: Literal["v1"] = "v1"
    audit_version: Literal["graph_extraction_v1"] = AUDIT_VERSION
    audit_sample_checksum: Sha256Hex
    audit_status: AuditStatus
    entity_f1_gate: Annotated[float, Field(ge=0.0, le=1.0)] = ENTITY_F1_GATE
    relation_precision_gate: Annotated[float, Field(ge=0.0, le=1.0)] = RELATION_PRECISION_GATE
    observed_entity_f1: Annotated[float, Field(ge=0.0, le=1.0)] | None = None
    observed_relation_precision: Annotated[float, Field(ge=0.0, le=1.0)] | None = None
    graph_tier_enabled: bool
    reason: NonEmptyString

    @model_validator(mode="after")
    def _fail_closed(self) -> Self:
        if (
            self.entity_f1_gate != ENTITY_F1_GATE
            or self.relation_precision_gate != RELATION_PRECISION_GATE
        ):
            raise ValueError("graph tier policy thresholds are immutable")
        metrics_pass = (
            self.observed_entity_f1 is not None
            and self.observed_entity_f1 >= ENTITY_F1_GATE
            and self.observed_relation_precision is not None
            and self.observed_relation_precision >= RELATION_PRECISION_GATE
        )
        allowed = self.audit_status is AuditStatus.COMPLETE and metrics_pass
        if self.graph_tier_enabled is not allowed:
            raise ValueError("graph tier policy must exactly reflect completed audit gates")
        return self


def audit_selection_sha256(
    source_chunk_id: str,
    start_char: int,
    end_char: int,
    text: str,
) -> str:
    payload = f"{source_chunk_id}\0{start_char}\0{end_char}\0{text}".encode()
    return hashlib.sha256(payload).hexdigest()


def build_audit_sentence(
    *,
    source_chunk_id: str,
    start_char: int,
    end_char: int,
    text: str,
    predicted_entities: Sequence[AuditEntityLabel] = (),
    predicted_relations: Sequence[AuditRelationLabel] = (),
) -> AuditSentence:
    selection_hash = audit_selection_sha256(source_chunk_id, start_char, end_char, text)
    return AuditSentence(
        sentence_id=f"{AUDIT_VERSION}:{selection_hash}",
        source_chunk_id=source_chunk_id,
        start_char=start_char,
        end_char=end_char,
        text=text,
        selection_sha256=selection_hash,
        predicted_entities=tuple(predicted_entities),
        predicted_relations=tuple(predicted_relations),
    )


def select_audit_sample(candidates: Iterable[AuditSentence]) -> tuple[AuditSentence, ...]:
    """Freeze exactly 100 unique train-passage sentences by ascending SHA-256."""

    unique = {item.sentence_id: item for item in candidates}
    ordered = tuple(sorted(unique.values(), key=lambda item: item.selection_sha256))
    if len(ordered) < AUDIT_SAMPLE_SIZE:
        raise ValueError("at least 100 unique train sentences are required for the audit")
    return ordered[:AUDIT_SAMPLE_SIZE]


def audit_sample_checksum(sentences: Sequence[AuditSentence]) -> str:
    payload = [item.model_dump(mode="json") for item in sentences]
    serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def evaluate_graph_audit(
    manifest: GraphAuditManifest,
    reviews: Sequence[AuditReview],
) -> AuditEvaluation:
    """Compute exact-span/directed-edge metrics, or retain a fail-closed pending gate."""

    sentence_ids = {item.sentence_id for item in manifest.sentences}
    primary = _reviews_by_role(reviews, "primary", sentence_ids)
    secondary = _reviews_by_role(reviews, "secondary", sentence_ids)
    adjudicated = _reviews_by_role(reviews, "adjudicator", sentence_ids)
    second_ids = set(manifest.second_reviewer_sentence_ids)
    complete = (
        set(primary) == sentence_ids
        and set(secondary) == second_ids
        and set(adjudicated) == second_ids
    )
    if not complete:
        return AuditEvaluation(
            status=AuditStatus.PENDING_HUMAN_REVIEW,
            reviewed_sentence_count=len(primary),
            double_reviewed_sentence_count=len(second_ids & set(secondary)),
            adjudicated_sentence_count=len(second_ids & set(adjudicated)),
            graph_tier_enabled=False,
            gate_reason="human review and adjudication are incomplete",
        )
    if any(primary[item].reviewer_id == secondary[item].reviewer_id for item in second_ids):
        raise ValueError("double-reviewed audit sentences require two distinct reviewers")

    predictions = {item.sentence_id: item for item in manifest.sentences}
    gold = {**primary, **adjudicated}
    predicted_entity_sets = {
        sentence_id: _entity_set(sentence.predicted_entities)
        for sentence_id, sentence in predictions.items()
    }
    predicted_relation_sets = {
        sentence_id: _relation_set(sentence.predicted_relations)
        for sentence_id, sentence in predictions.items()
    }
    gold_entity_sets = {
        sentence_id: _entity_set(review.entities) for sentence_id, review in gold.items()
    }
    gold_relation_sets = {
        sentence_id: _relation_set(review.relations) for sentence_id, review in gold.items()
    }
    entity_precision, entity_recall, entity_f1 = _micro_metrics(
        predicted_entity_sets,
        gold_entity_sets,
    )
    relation_precision, relation_recall, _ = _micro_metrics(
        predicted_relation_sets,
        gold_relation_sets,
    )
    entity_agreement = _agreement(primary, secondary, second_ids, labels="entities")
    relation_agreement = _agreement(primary, secondary, second_ids, labels="relations")
    enabled = entity_f1 >= ENTITY_F1_GATE and relation_precision >= RELATION_PRECISION_GATE
    reason = (
        "entity F1 and relation precision meet the Stage 4 audit gates"
        if enabled
        else "entity F1 or relation precision is below the Stage 4 audit gate"
    )
    return AuditEvaluation(
        status=AuditStatus.COMPLETE,
        reviewed_sentence_count=AUDIT_SAMPLE_SIZE,
        double_reviewed_sentence_count=DOUBLE_REVIEW_SIZE,
        adjudicated_sentence_count=DOUBLE_REVIEW_SIZE,
        entity_precision=entity_precision,
        entity_recall=entity_recall,
        entity_f1=entity_f1,
        relation_precision=relation_precision,
        relation_recall=relation_recall,
        entity_reviewer_agreement_f1=entity_agreement,
        relation_reviewer_agreement_f1=relation_agreement,
        graph_tier_enabled=enabled,
        gate_reason=reason,
    )


def graph_tier_policy(
    manifest: GraphAuditManifest,
    evaluation: AuditEvaluation,
) -> RuleGraphTierPolicy:
    return RuleGraphTierPolicy(
        audit_sample_checksum=manifest.sample_checksum,
        audit_status=evaluation.status,
        observed_entity_f1=evaluation.entity_f1,
        observed_relation_precision=evaluation.relation_precision,
        graph_tier_enabled=evaluation.graph_tier_enabled,
        reason=evaluation.gate_reason,
    )


def load_graph_tier_policy(path: Path | None = None) -> RuleGraphTierPolicy:
    """Load the fail-closed handoff from an explicit or packaged policy file."""

    try:
        if path is not None:
            serialized = path.read_text(encoding="utf-8")
        elif DEFAULT_GRAPH_TIER_POLICY_PATH.is_file():
            serialized = DEFAULT_GRAPH_TIER_POLICY_PATH.read_text(encoding="utf-8")
        else:
            serialized = (
                files("ragplan.resources")
                .joinpath("graph_tier_policy.json")
                .read_text(encoding="utf-8")
            )
        return RuleGraphTierPolicy.model_validate_json(serialized)
    except Exception as exc:
        raise ValueError("graph tier policy is missing or invalid") from exc


def _reviews_by_role(
    reviews: Sequence[AuditReview],
    role: Literal["primary", "secondary", "adjudicator"],
    sentence_ids: set[str],
) -> dict[str, AuditReview]:
    selected: dict[str, AuditReview] = {}
    for review in reviews:
        if review.reviewer_role != role:
            continue
        if review.sentence_id not in sentence_ids or review.sentence_id in selected:
            raise ValueError("audit reviews contain an unknown or duplicate sentence/role")
        selected[review.sentence_id] = review
    return selected


def _entity_set(labels: Sequence[AuditEntityLabel]) -> set[tuple[object, ...]]:
    return {(item.start_char, item.end_char, item.entity_type.value) for item in labels}


def _relation_set(labels: Sequence[AuditRelationLabel]) -> set[tuple[object, ...]]:
    return {
        (
            item.source_start_char,
            item.source_end_char,
            item.target_start_char,
            item.target_end_char,
            item.predicate.casefold(),
        )
        for item in labels
    }


def _micro_metrics(
    predicted: Mapping[str, set[tuple[object, ...]]],
    gold: Mapping[str, set[tuple[object, ...]]],
) -> tuple[float, float, float]:
    true_positive = sum(len(predicted[key] & gold[key]) for key in gold)
    predicted_count = sum(len(predicted[key]) for key in gold)
    gold_count = sum(len(gold[key]) for key in gold)
    precision = true_positive / predicted_count if predicted_count else float(gold_count == 0)
    recall = true_positive / gold_count if gold_count else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def _agreement(
    primary: Mapping[str, AuditReview],
    secondary: Mapping[str, AuditReview],
    sentence_ids: set[str],
    *,
    labels: Literal["entities", "relations"],
) -> float:
    if labels == "entities":
        first = {key: _entity_set(primary[key].entities) for key in sentence_ids}
        second = {key: _entity_set(secondary[key].entities) for key in sentence_ids}
    else:
        first = {key: _relation_set(primary[key].relations) for key in sentence_ids}
        second = {key: _relation_set(secondary[key].relations) for key in sentence_ids}
    _, _, f1 = _micro_metrics(first, second)
    return f1
