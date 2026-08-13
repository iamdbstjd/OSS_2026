#!/usr/bin/env python3
"""Freeze Stage 4 train-sentence predictions and a fail-closed human review sheet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from ragplan.core.models import Chunk
from ragplan.ingestion.audit import (
    DOUBLE_REVIEW_SIZE,
    AuditEntityLabel,
    AuditRelationLabel,
    GraphAuditManifest,
    audit_sample_checksum,
    build_audit_sentence,
    evaluate_graph_audit,
    graph_tier_policy,
    select_audit_sample,
)
from ragplan.ingestion.entities import ChunkExtraction, EntityExtractor
from ragplan.ingestion.manifest import write_contract_json
from ragplan.ingestion.relations import extract_relations


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--uv-lock", type=Path)
    return parser


def _load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> tuple[dict[str, object], ...]:
    records: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as source:
        for line in source:
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"{path} must contain JSON objects")
            records.append(payload)
    return tuple(records)


def _train_document_ids(root: Path) -> set[str]:
    split_payload = _load_json(root / "benchmark/configs/splits_v1.json")
    if not isinstance(split_payload, dict) or not isinstance(
        split_payload.get("assignments"), list
    ):
        raise ValueError("split manifest is invalid")
    train_query_ids = {
        item["query_id"]
        for item in split_payload["assignments"]
        if isinstance(item, dict)
        and item.get("split") == "train"
        and isinstance(item.get("query_id"), str)
    }
    documents: set[str] = set()
    for record in _load_jsonl(root / "benchmark/manifests/corpus_index_v1.jsonl"):
        document_id = record.get("document_id")
        query_ids = record.get("query_ids")
        if (
            isinstance(document_id, str)
            and isinstance(query_ids, list)
            and any(item in train_query_ids for item in query_ids)
        ):
            documents.add(document_id)
    return documents


def _train_chunks(root: Path) -> tuple[Chunk, ...]:
    train_document_ids = _train_document_ids(root)
    chunks = tuple(
        Chunk.model_validate_json(json.dumps(record))
        for record in _load_jsonl(root / "benchmark/datasets/normalized/chunks_v1.jsonl")
        if record.get("document_id") in train_document_ids
    )
    if not chunks:
        raise ValueError("no normalized train chunks were found")
    return chunks


def _sentence_candidates(
    extractions: tuple[ChunkExtraction, ...],
) -> tuple[object, ...]:
    candidates = []
    for extraction in extractions:
        spans = sorted(
            {
                (token.sentence_start_char, token.sentence_end_char)
                for token in extraction.tokens
                if token.sentence_end_char > token.sentence_start_char
            }
        )
        relations = extract_relations(extraction)
        for start_char, end_char in spans:
            sentence_text = extraction.chunk.text[start_char:end_char]
            mentions = tuple(
                mention
                for mention in extraction.mentions
                if mention.sentence_start_char == start_char
                and mention.sentence_end_char == end_char
            )
            mention_by_entity = {
                mention.entity_id: mention
                for mention in sorted(mentions, key=lambda item: (item.start_char, item.end_char))
            }
            entity_labels = tuple(
                AuditEntityLabel(
                    start_char=mention.start_char - start_char,
                    end_char=mention.end_char - start_char,
                    entity_type=mention.entity_type,
                    text=mention.raw_text,
                )
                for mention in mentions
            )
            relation_labels = []
            for relation in relations:
                source = mention_by_entity.get(relation.source_entity_id)
                target = mention_by_entity.get(relation.target_entity_id)
                if source is None or target is None:
                    continue
                relation_labels.append(
                    AuditRelationLabel(
                        source_start_char=source.start_char - start_char,
                        source_end_char=source.end_char - start_char,
                        target_start_char=target.start_char - start_char,
                        target_end_char=target.end_char - start_char,
                        predicate=relation.predicate,
                    )
                )
            candidates.append(
                build_audit_sentence(
                    source_chunk_id=extraction.chunk.id,
                    start_char=start_char,
                    end_char=end_char,
                    text=sentence_text,
                    predicted_entities=entity_labels,
                    predicted_relations=relation_labels,
                )
            )
    return tuple(candidates)


def _write_jsonl(path: Path, records: tuple[object, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = "".join(
        f"{json.dumps(record, ensure_ascii=False, separators=(',', ':'), sort_keys=True)}\n"
        for record in records
    )
    path.write_text(serialized, encoding="utf-8")


def _write_review_template_if_safe(path: Path, records: tuple[object, ...]) -> None:
    if path.exists():
        existing = _load_jsonl(path)
        if any(item.get("completed") is True or item.get("reviewer_id") for item in existing):
            return
    _write_jsonl(path, records)


def _review_template(manifest: GraphAuditManifest) -> tuple[object, ...]:
    records: list[object] = []
    second_ids = set(manifest.second_reviewer_sentence_ids)
    for sentence in manifest.sentences:
        roles = ["primary"]
        if sentence.sentence_id in second_ids:
            roles.extend(("secondary", "adjudicator"))
        for role in roles:
            records.append(
                {
                    "audit_version": manifest.audit_version,
                    "sentence_id": sentence.sentence_id,
                    "reviewer_id": None,
                    "reviewer_role": role,
                    "entities": None,
                    "relations": None,
                    "completed": False,
                }
            )
    return tuple(records)


def main() -> None:
    args = build_parser().parse_args()
    root = args.repository_root.resolve()
    lockfile = (args.uv_lock or root / "uv.lock").resolve()
    benchmark_payload = yaml.safe_load(
        (root / "benchmark/manifests/adaptive_rag_bench_v1.yaml").read_text(encoding="utf-8")
    )
    split_payload = _load_json(root / "benchmark/configs/splits_v1.json")
    if not isinstance(benchmark_payload, dict) or not isinstance(split_payload, dict):
        raise ValueError("benchmark and split manifests must be objects")
    extractor = EntityExtractor.load_pinned(lockfile=lockfile, benchmark_mode=True)
    extractions = extractor.extract_many(_train_chunks(root))
    sample = select_audit_sample(_sentence_candidates(extractions))
    manifest = GraphAuditManifest(
        corpus_version=str(benchmark_payload["corpus_version"]),
        benchmark_manifest_sha256=str(benchmark_payload["manifest_sha256"]),
        split_hash=str(split_payload["split_hash"]),
        extractor_version=extractor.extractor_version,
        sentences=sample,
        second_reviewer_sentence_ids=tuple(
            item.sentence_id for item in sample[:DOUBLE_REVIEW_SIZE]
        ),
        sample_checksum=audit_sample_checksum(sample),
    )
    evaluation = evaluate_graph_audit(manifest, ())
    policy = graph_tier_policy(manifest, evaluation)
    audit_root = root / "benchmark/audits/graph_extraction_v1"
    write_contract_json(audit_root / "manifest_v1.json", manifest)
    write_contract_json(audit_root / "evaluation_v1.json", evaluation)
    write_contract_json(root / "configs/graph_tier_policy.json", policy)
    _write_jsonl(
        audit_root / "sample_v1.jsonl",
        tuple(item.model_dump(mode="json") for item in sample),
    )
    _write_review_template_if_safe(
        audit_root / "reviews_v1.jsonl",
        _review_template(manifest),
    )
    reference = {
        "schema_version": "v1",
        "audit_version": manifest.audit_version,
        "audit_manifest": "benchmark/audits/graph_extraction_v1/manifest_v1.json",
        "audit_sample_checksum": manifest.sample_checksum,
        "benchmark_manifest_sha256": manifest.benchmark_manifest_sha256,
        "split_hash": manifest.split_hash,
        "status": evaluation.status.value,
        "rule_graph_tier_enabled": policy.graph_tier_enabled,
    }
    (root / "benchmark/manifests/graph_extraction_audit_v1.json").write_text(
        json.dumps(reference, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(reference, sort_keys=True))


if __name__ == "__main__":
    main()
