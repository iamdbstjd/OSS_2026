from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest
import yaml

from ragplan.benchmark.artifacts import (
    load_benchmark_manifest,
    load_chunk_index,
    load_corpus_index,
    load_immutable_test_manifest,
    load_qrels,
    load_split_manifest,
)
from ragplan.benchmark.contracts import (
    SOURCE_SPLIT_QUOTAS,
    SOURCE_TOTALS,
    SPLIT_TOTALS,
    QueryTag,
    SplitName,
)
from ragplan.benchmark.selection import assign_splits
from ragplan.benchmark.synthetic import SyntheticGraphManifest
from ragplan.benchmark.validation import validate_stage2_artifacts

ROOT = Path(__file__).parents[2]
BENCHMARK = ROOT / "benchmark"


def test_frozen_primary_artifacts_satisfy_every_stage2_exit_invariant() -> None:
    manifest = load_benchmark_manifest(BENCHMARK / "manifests/adaptive_rag_bench_v1.yaml")
    splits = load_split_manifest(BENCHMARK / "configs/splits_v1.json")
    test_ids = load_immutable_test_manifest(BENCHMARK / "manifests/test_ids_v1.json")
    qrels = load_qrels(BENCHMARK / "qrels/qrels_v1.jsonl")
    documents = load_corpus_index(BENCHMARK / "manifests/corpus_index_v1.jsonl")
    chunks = load_chunk_index(BENCHMARK / "manifests/chunk_index_v1.jsonl")

    summary = validate_stage2_artifacts(
        manifest=manifest,
        splits=splits,
        qrels=qrels,
        corpus_index=documents,
        immutable_test=test_ids,
        chunk_ids=frozenset(chunk.canonical_chunk_id for chunk in chunks),
    )
    assert summary.query_count == 600
    assert Counter(query.source_dataset for query in manifest.queries) == Counter(SOURCE_TOTALS)
    assert Counter(assignment.split for assignment in splits.assignments) == Counter(SPLIT_TOTALS)
    assert all(summary.test_tag_counts[tag] >= 15 for tag in QueryTag)
    assert len({chunk.canonical_chunk_id for chunk in chunks}) == len(chunks)
    assert all(qrel.relevance_grade in {1, 2} for qrel in qrels)


def test_selection_hash_formula_and_source_split_quotas_are_frozen() -> None:
    manifest = load_benchmark_manifest(BENCHMARK / "manifests/adaptive_rag_bench_v1.yaml")
    splits = load_split_manifest(BENCHMARK / "configs/splits_v1.json")
    split_by_id = {assignment.query_id: assignment.split for assignment in splits.assignments}
    for query in manifest.queries:
        value = f"{query.source_dataset.value}:{query.source_query_id}:20260809"
        assert query.selection_hash == hashlib.sha256(value.encode()).hexdigest()
    actual = Counter(
        (query.source_dataset, split_by_id[query.query_id]) for query in manifest.queries
    )
    for source, quotas in SOURCE_SPLIT_QUOTAS.items():
        for split, expected in quotas.items():
            assert actual[(source, split)] == expected


def test_license_audit_is_complete_and_raw_artifacts_are_not_redistributed() -> None:
    audit = yaml.safe_load((BENCHMARK / "manifests/licenses.yaml").read_text(encoding="utf-8"))
    assert audit["raw_artifacts_redistributed"] is False
    assert len(audit["sources"]) == 3
    licenses = {source["license"] for source in audit["sources"].values()}
    assert licenses == {"CC BY-SA 3.0", "CC BY-SA 4.0", "CC BY 4.0"}
    for source in audit["sources"].values():
        assert source["official_page"].startswith("https://")
        assert source["required_attribution"]
        assert all(len(artifact["sha256"]) == 64 for artifact in source["artifacts"])


def test_synthetic_fixture_is_separate_from_primary_manifest() -> None:
    synthetic = SyntheticGraphManifest.model_validate_json(
        (BENCHMARK / "manifests/synthetic_graph_v1.json").read_text(encoding="utf-8")
    )
    primary = load_benchmark_manifest(BENCHMARK / "manifests/adaptive_rag_bench_v1.yaml")
    assert synthetic.primary_metrics_eligible is False
    assert len(synthetic.queries) == 100
    synthetic_ids = {query.query_id for query in synthetic.queries}
    primary_ids = {query.query_id for query in primary.queries}
    assert not (synthetic_ids & primary_ids)


def test_test_ids_are_explicitly_immutable() -> None:
    splits = load_split_manifest(BENCHMARK / "configs/splits_v1.json")
    frozen = load_immutable_test_manifest(BENCHMARK / "manifests/test_ids_v1.json")
    expected = tuple(
        assignment.query_id
        for assignment in splits.assignments
        if assignment.split is SplitName.TEST
    )
    assert frozen.query_ids == expected
    assert frozen.query_ids_sha256 == (
        "2fa161f8517900f723426c4f896f6b2c8b6308decac23c31cb15883347833fcc"
    )


def test_frozen_logical_hashes_and_split_generation_are_reproducible() -> None:
    manifest = load_benchmark_manifest(BENCHMARK / "manifests/adaptive_rag_bench_v1.yaml")
    splits = load_split_manifest(BENCHMARK / "configs/splits_v1.json")
    assert manifest.manifest_sha256 == (
        "c5d959ca1da8c5b5e977af948c96012efc3a0754599e34edcefff30755e1ef76"
    )
    assert splits.split_hash == ("b01deb380f039447d626bb7b9d74ca344d3a55dc22cf9fea725d90049a5e1a31")
    assert assign_splits(manifest.queries) == splits.assignments


def test_zero_relevant_and_orphan_qrels_fail_closed() -> None:
    manifest = load_benchmark_manifest(BENCHMARK / "manifests/adaptive_rag_bench_v1.yaml")
    splits = load_split_manifest(BENCHMARK / "configs/splits_v1.json")
    test_ids = load_immutable_test_manifest(BENCHMARK / "manifests/test_ids_v1.json")
    qrels = load_qrels(BENCHMARK / "qrels/qrels_v1.jsonl")
    documents = load_corpus_index(BENCHMARK / "manifests/corpus_index_v1.jsonl")
    chunks = load_chunk_index(BENCHMARK / "manifests/chunk_index_v1.jsonl")
    chunk_ids = frozenset(chunk.canonical_chunk_id for chunk in chunks)

    missing_query_id = manifest.queries[0].query_id
    without_one_query = tuple(qrel for qrel in qrels if qrel.query_id != missing_query_id)
    with pytest.raises(ValueError, match="every benchmark query"):
        validate_stage2_artifacts(
            manifest=manifest,
            splits=splits,
            qrels=without_one_query,
            corpus_index=documents,
            immutable_test=test_ids,
            chunk_ids=chunk_ids,
        )

    orphan = qrels[0].model_copy(
        update={"query_id": "adaptive_rag_bench_v1:nq:unknown-upstream-record"}
    )
    with pytest.raises(ValueError, match="orphan query"):
        validate_stage2_artifacts(
            manifest=manifest,
            splits=splits,
            qrels=(orphan, *qrels[1:]),
            corpus_index=documents,
            immutable_test=test_ids,
            chunk_ids=chunk_ids,
        )


def test_public_artifact_set_is_complete_and_byte_verified() -> None:
    artifact_set = json.loads(
        (BENCHMARK / "manifests/artifact_set_v1.json").read_text(encoding="utf-8")
    )
    assert artifact_set["ready"] is True
    assert len(artifact_set["files"]) == 15
    assert not any(path.startswith("datasets/") for path in artifact_set["files"])
    for relative_path, expected in artifact_set["files"].items():
        path = BENCHMARK / relative_path
        assert path.stat().st_size == expected["size_bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected["sha256"]


def test_raw_and_normalized_dataset_text_stays_out_of_git_and_docker() -> None:
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    dockerignore = (ROOT / ".dockerignore").read_text(encoding="utf-8")
    assert "benchmark/datasets/*" in gitignore
    assert "benchmark/datasets" in dockerignore


def test_exact_duplicate_aliases_survive_canonicalization_for_audit() -> None:
    duplicate_report = json.loads(
        (BENCHMARK / "manifests/duplicate_report_v1.json").read_text(encoding="utf-8")
    )
    policy = json.loads((BENCHMARK / "manifests/corpus_policy_v1.json").read_text(encoding="utf-8"))
    aliases = duplicate_report["exact_duplicate_aliases_before_canonicalization"]
    assert len(aliases) == 8
    assert all(len(document_ids) >= 2 for document_ids in aliases.values())
    assert duplicate_report["exact_duplicate_aliases_after_canonicalization"] == {}
    assert policy["exact_duplicate_groups_before_canonicalization"] == 8
    assert policy["exact_duplicate_groups_after_canonicalization"] == 0
