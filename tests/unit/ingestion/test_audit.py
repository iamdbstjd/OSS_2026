from __future__ import annotations

import pytest

from ragplan.core.models import EntityType
from ragplan.ingestion.audit import (
    AUDIT_SAMPLE_SIZE,
    DOUBLE_REVIEW_SIZE,
    AuditEntityLabel,
    AuditRelationLabel,
    AuditReview,
    AuditStatus,
    GraphAuditManifest,
    audit_sample_checksum,
    build_audit_sentence,
    evaluate_graph_audit,
    select_audit_sample,
)

pytestmark = pytest.mark.unit


def _labels() -> tuple[AuditEntityLabel, ...]:
    return (
        AuditEntityLabel(start_char=0, end_char=5, entity_type=EntityType.ORG, text="Apple"),
        AuditEntityLabel(start_char=15, end_char=20, entity_type=EntityType.ORG, text="Beats"),
    )


def _relations(predicate: str = "acquire") -> tuple[AuditRelationLabel, ...]:
    return (
        AuditRelationLabel(
            source_start_char=0,
            source_end_char=5,
            target_start_char=15,
            target_end_char=20,
            predicate=predicate,
        ),
    )


def _manifest() -> GraphAuditManifest:
    candidates = [
        build_audit_sentence(
            source_chunk_id=f"v1:chunk:fixture:{index}:abc",
            start_char=0,
            end_char=21,
            text=f"Apple acquired Beats {index}",
            predicted_entities=_labels(),
            predicted_relations=_relations(),
        )
        for index in reversed(range(AUDIT_SAMPLE_SIZE + 5))
    ]
    sample = select_audit_sample(candidates)
    return GraphAuditManifest(
        corpus_version="fixture-v1",
        benchmark_manifest_sha256="a" * 64,
        split_hash="b" * 64,
        extractor_version="graph-extractor-v1-fixture",
        sentences=sample,
        second_reviewer_sentence_ids=tuple(
            item.sentence_id for item in sample[:DOUBLE_REVIEW_SIZE]
        ),
        sample_checksum=audit_sample_checksum(sample),
    )


def _completed_reviews(manifest: GraphAuditManifest) -> tuple[AuditReview, ...]:
    reviews = [
        AuditReview(
            sentence_id=item.sentence_id,
            reviewer_id="reviewer-primary",
            reviewer_role="primary",
            entities=_labels(),
            relations=_relations(),
        )
        for item in manifest.sentences
    ]
    for sentence_id in manifest.second_reviewer_sentence_ids:
        reviews.extend(
            (
                AuditReview(
                    sentence_id=sentence_id,
                    reviewer_id="reviewer-secondary",
                    reviewer_role="secondary",
                    entities=_labels(),
                    relations=_relations(),
                ),
                AuditReview(
                    sentence_id=sentence_id,
                    reviewer_id="reviewer-adjudicator",
                    reviewer_role="adjudicator",
                    entities=_labels(),
                    relations=_relations(),
                ),
            )
        )
    return tuple(reviews)


def test_audit_selection_is_reproducible_and_hash_ordered() -> None:
    first = _manifest()
    second = _manifest()

    assert first == second
    assert len(first.sentences) == AUDIT_SAMPLE_SIZE
    assert tuple(item.selection_sha256 for item in first.sentences) == tuple(
        sorted(item.selection_sha256 for item in first.sentences)
    )


def test_incomplete_human_review_fails_closed() -> None:
    evaluation = evaluate_graph_audit(_manifest(), ())

    assert evaluation.status is AuditStatus.PENDING_HUMAN_REVIEW
    assert evaluation.graph_tier_enabled is False
    assert evaluation.entity_f1 is None


def test_complete_perfect_audit_enables_graph_tier_and_records_agreement() -> None:
    manifest = _manifest()
    evaluation = evaluate_graph_audit(manifest, _completed_reviews(manifest))

    assert evaluation.status is AuditStatus.COMPLETE
    assert evaluation.entity_f1 == 1.0
    assert evaluation.relation_precision == 1.0
    assert evaluation.entity_reviewer_agreement_f1 == 1.0
    assert evaluation.relation_reviewer_agreement_f1 == 1.0
    assert evaluation.graph_tier_enabled is True


def test_completed_but_bad_relation_precision_disables_graph_tier() -> None:
    manifest = _manifest()
    reviews = tuple(
        review.model_copy(update={"relations": _relations("unrelated")})
        for review in _completed_reviews(manifest)
    )

    evaluation = evaluate_graph_audit(manifest, reviews)

    assert evaluation.status is AuditStatus.COMPLETE
    assert evaluation.relation_precision == 0.0
    assert evaluation.graph_tier_enabled is False
