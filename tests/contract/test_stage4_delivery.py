from __future__ import annotations

import json
from pathlib import Path

import pytest

from ragplan.ingestion.audit import (
    AUDIT_SAMPLE_SIZE,
    DOUBLE_REVIEW_SIZE,
    AuditEvaluation,
    AuditStatus,
    GraphAuditManifest,
    RuleGraphTierPolicy,
    load_graph_tier_policy,
)
from ragplan.ingestion.extractor_version import build_extractor_version

pytestmark = [pytest.mark.contract, pytest.mark.unit]

ROOT = Path(__file__).resolve().parents[2]
AUDIT_ROOT = ROOT / "benchmark/audits/graph_extraction_v1"


def test_frozen_graph_audit_is_bound_to_stage2_and_fails_closed() -> None:
    manifest = GraphAuditManifest.model_validate_json(
        (AUDIT_ROOT / "manifest_v1.json").read_text(encoding="utf-8")
    )
    evaluation = AuditEvaluation.model_validate_json(
        (AUDIT_ROOT / "evaluation_v1.json").read_text(encoding="utf-8")
    )
    policy = RuleGraphTierPolicy.model_validate_json(
        (ROOT / "configs/graph_tier_policy.json").read_text(encoding="utf-8")
    )
    split = json.loads((ROOT / "benchmark/configs/splits_v1.json").read_text(encoding="utf-8"))
    reference = json.loads(
        (ROOT / "benchmark/manifests/graph_extraction_audit_v1.json").read_text(encoding="utf-8")
    )

    assert len(manifest.sentences) == AUDIT_SAMPLE_SIZE
    assert len(manifest.second_reviewer_sentence_ids) == DOUBLE_REVIEW_SIZE
    assert manifest.benchmark_manifest_sha256 == split["benchmark_manifest_sha256"]
    assert manifest.split_hash == split["split_hash"]
    assert reference["audit_sample_checksum"] == manifest.sample_checksum
    assert evaluation.status is AuditStatus.PENDING_HUMAN_REVIEW
    assert policy.audit_sample_checksum == manifest.sample_checksum
    assert policy.graph_tier_enabled is False
    assert load_graph_tier_policy().model_dump() == policy.model_dump()
    assert manifest.extractor_version == build_extractor_version(ROOT / "uv.lock")


def test_review_sheet_does_not_fabricate_human_annotations() -> None:
    records = [
        json.loads(line)
        for line in (AUDIT_ROOT / "reviews_v1.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    primary = [item for item in records if item["reviewer_role"] == "primary"]
    secondary = [item for item in records if item["reviewer_role"] == "secondary"]
    adjudicators = [item for item in records if item["reviewer_role"] == "adjudicator"]

    assert len(primary) == AUDIT_SAMPLE_SIZE
    assert len(secondary) == DOUBLE_REVIEW_SIZE
    assert len(adjudicators) == DOUBLE_REVIEW_SIZE
    assert all(item["completed"] is False for item in records)
    assert all(item["reviewer_id"] is None for item in records)
    assert all(item["entities"] is None and item["relations"] is None for item in records)
