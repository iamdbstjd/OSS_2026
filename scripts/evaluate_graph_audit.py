#!/usr/bin/env python3
"""Validate human graph annotations and update the fail-closed planner policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ragplan.ingestion.audit import (
    AuditReview,
    GraphAuditManifest,
    evaluate_graph_audit,
    graph_tier_policy,
)
from ragplan.ingestion.manifest import load_contract_json, write_contract_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path(__file__).resolve().parents[1])
    return parser


def _load_completed_reviews(path: Path) -> tuple[AuditReview, ...]:
    reviews: list[AuditReview] = []
    with path.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                raise ValueError("review JSONL contains a blank record")
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError("review JSONL rows must be objects")
            if payload.get("completed") is True:
                reviews.append(AuditReview.model_validate_json(json.dumps(payload)))
    return tuple(reviews)


def main() -> None:
    args = build_parser().parse_args()
    root = args.repository_root.resolve()
    audit_root = root / "benchmark/audits/graph_extraction_v1"
    manifest = load_contract_json(audit_root / "manifest_v1.json", GraphAuditManifest)
    reviews = _load_completed_reviews(audit_root / "reviews_v1.jsonl")
    evaluation = evaluate_graph_audit(manifest, reviews)
    policy = graph_tier_policy(manifest, evaluation)
    write_contract_json(audit_root / "evaluation_v1.json", evaluation)
    write_contract_json(root / "configs/graph_tier_policy.json", policy)
    reference_path = root / "benchmark/manifests/graph_extraction_audit_v1.json"
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    reference.update(
        {
            "status": evaluation.status.value,
            "rule_graph_tier_enabled": policy.graph_tier_enabled,
        }
    )
    reference_path.write_text(
        json.dumps(reference, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(evaluation.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
