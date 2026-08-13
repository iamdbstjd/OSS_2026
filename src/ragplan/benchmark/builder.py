"""End-to-end builder for the frozen ``adaptive_rag_bench_v1`` artifacts."""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ragplan.benchmark.artifacts import (
    write_json,
    write_json_model,
    write_jsonl_models,
    write_jsonl_values,
    write_yaml_model,
)
from ragplan.benchmark.contracts import (
    CORPUS_VERSION,
    SELECTION_RULE,
    SOURCE_TOTALS,
    BenchmarkManifest,
    CorpusChunkIndex,
    ImmutableTestManifest,
    SourceDataset,
    SplitManifest,
    SplitName,
    benchmark_manifest_sha256,
    canonical_sha256,
    split_manifest_sha256,
)
from ragplan.benchmark.dedupe import deduplicate_exact_documents_with_report
from ragplan.benchmark.download import extract_musique, file_sha256, verify_raw_datasets
from ragplan.benchmark.qrels import QrelsBuildResult, build_corpus_and_qrels
from ragplan.benchmark.selection import (
    PinnedNer,
    SelectionReport,
    assign_splits,
    build_query_contract,
    document_namespace,
    select_smallest_hashes,
)
from ragplan.benchmark.sources import (
    NormalizedQuery,
    iter_hotpot,
    iter_musique,
    iter_nq,
    load_musique_singlehop_question_hashes,
    question_overlap_hash,
)
from ragplan.benchmark.synthetic import build_synthetic_graph_fixture
from ragplan.benchmark.validation import ValidationSummary, validate_stage2_artifacts
from ragplan.core.ids import canonical_document_id
from ragplan.ingestion.embedder import SentenceTransformerEmbedder
from ragplan.ingestion.model_manifest import load_default_model_artifact_manifest


@dataclass(frozen=True, slots=True)
class Stage2BuildResult:
    benchmark_manifest_sha256: str
    split_hash: str
    query_count: int
    document_count: int
    chunk_count: int
    qrel_row_count: int
    test_ids_sha256: str
    output_root: str


def build_stage2(
    *,
    repository_root: Path,
    raw_dir: Path,
    model_snapshot: Path,
) -> Stage2BuildResult:
    """Verify pinned inputs, select 600 records, materialize and validate artifacts."""

    repository_root = repository_root.resolve()
    raw_paths = verify_raw_datasets(raw_dir)
    musique_root = raw_dir / "musique"
    musique_train = musique_root / "data/musique_ans_v1.0_train.jsonl"
    musique_singlehop = musique_root / "data/dev_test_singlehop_questions_v1.0.json"
    if not musique_train.is_file() or not musique_singlehop.is_file():
        extract_musique(raw_paths[-1], destination=musique_root)

    singlehop_hashes = load_musique_singlehop_question_hashes(musique_singlehop)
    selected, reports = _select_queries(
        nq_path=raw_paths[0],
        hotpot_paths=raw_paths[1:3],
        musique_path=musique_train,
        excluded_nq_hashes=singlehop_hashes,
    )
    _validate_selection_overlap(selected, singlehop_hashes=singlehop_hashes)
    deduplication = deduplicate_exact_documents_with_report(selected)
    selected = deduplication.queries

    ner = PinnedNer.load()
    query_contracts = tuple(build_query_contract(query, ner=ner) for query in selected)
    query_contracts = tuple(sorted(query_contracts, key=lambda item: item.query_id))
    manifest = BenchmarkManifest(
        selection_rule=SELECTION_RULE,
        queries=query_contracts,
        manifest_sha256=benchmark_manifest_sha256(query_contracts),
    )
    assignments = assign_splits(query_contracts)
    splits = SplitManifest(
        benchmark_manifest_sha256=manifest.manifest_sha256,
        assignments=assignments,
        split_hash=split_manifest_sha256(assignments),
    )
    test_ids = tuple(
        assignment.query_id for assignment in assignments if assignment.split is SplitName.TEST
    )
    immutable_test = ImmutableTestManifest(
        split_hash=splits.split_hash,
        query_ids=test_ids,
        query_ids_sha256=canonical_sha256(list(test_ids)),
    )

    model_manifest = load_default_model_artifact_manifest()
    embedder = SentenceTransformerEmbedder.from_local_snapshot(
        snapshot_path=model_snapshot,
        manifest=model_manifest,
        device="cpu",
    )
    normalized_by_id = {query.query_id: query for query in selected}
    ordered_normalized = tuple(normalized_by_id[query.query_id] for query in query_contracts)
    qrels_result = build_corpus_and_qrels(
        ordered_normalized,
        {query.query_id: query for query in query_contracts},
        tokenizer=embedder.tokenizer,
    )
    summary = validate_stage2_artifacts(
        manifest=manifest,
        splits=splits,
        qrels=qrels_result.qrels,
        corpus_index=qrels_result.document_index,
        immutable_test=immutable_test,
        chunk_ids=frozenset(chunk.id for chunk in qrels_result.chunks),
    )
    _write_artifacts(
        repository_root=repository_root,
        selected=ordered_normalized,
        reports=reports,
        exact_duplicate_aliases=deduplication.aliases,
        manifest=manifest,
        splits=splits,
        immutable_test=immutable_test,
        qrels_result=qrels_result,
        summary=summary,
        model_manifest_sha256=model_manifest.sha256,
    )
    return Stage2BuildResult(
        benchmark_manifest_sha256=manifest.manifest_sha256,
        split_hash=splits.split_hash,
        query_count=len(query_contracts),
        document_count=len(qrels_result.document_index),
        chunk_count=len(qrels_result.chunks),
        qrel_row_count=len(qrels_result.qrels),
        test_ids_sha256=immutable_test.query_ids_sha256,
        output_root=str(repository_root / "benchmark"),
    )


def _select_queries(
    *,
    nq_path: Path,
    hotpot_paths: Sequence[Path],
    musique_path: Path,
    excluded_nq_hashes: frozenset[str],
) -> tuple[tuple[NormalizedQuery, ...], tuple[SelectionReport, ...]]:
    nq, nq_report = select_smallest_hashes(
        iter_nq(nq_path, excluded_question_hashes=excluded_nq_hashes),
        source_dataset=SourceDataset.NQ,
        limit=SOURCE_TOTALS[SourceDataset.NQ],
    )
    bridge, bridge_report = select_smallest_hashes(
        iter_hotpot(hotpot_paths),
        source_dataset=SourceDataset.HOTPOT_BRIDGE,
        limit=SOURCE_TOTALS[SourceDataset.HOTPOT_BRIDGE],
    )
    comparison, comparison_report = select_smallest_hashes(
        iter_hotpot(hotpot_paths),
        source_dataset=SourceDataset.HOTPOT_COMPARISON,
        limit=SOURCE_TOTALS[SourceDataset.HOTPOT_COMPARISON],
    )
    musique, musique_report = select_smallest_hashes(
        iter_musique(musique_path),
        source_dataset=SourceDataset.MUSIQUE,
        limit=SOURCE_TOTALS[SourceDataset.MUSIQUE],
    )
    return (
        nq + bridge + comparison + musique,
        (nq_report, bridge_report, comparison_report, musique_report),
    )


def _validate_selection_overlap(
    selected: Sequence[NormalizedQuery], *, singlehop_hashes: frozenset[str]
) -> None:
    if len(selected) != 600:
        raise ValueError("frozen selection must contain exactly 600 records")
    counts = Counter(query.source_dataset for query in selected)
    if counts != Counter(SOURCE_TOTALS):
        raise ValueError("frozen selection source counts do not match the contract")
    question_owners: dict[str, SourceDataset] = {}
    for query in selected:
        digest = question_overlap_hash(query.question)
        if query.source_dataset is SourceDataset.NQ and digest in singlehop_hashes:
            raise ValueError("NQ selection overlaps published MuSiQue single-hop sources")
        previous = question_owners.get(digest)
        if previous is not None and previous is not query.source_dataset:
            raise ValueError("cross-source normalized question overlap detected")
        question_owners[digest] = query.source_dataset


def _write_artifacts(
    *,
    repository_root: Path,
    selected: Sequence[NormalizedQuery],
    reports: Sequence[SelectionReport],
    exact_duplicate_aliases: Mapping[str, tuple[str, ...]],
    manifest: BenchmarkManifest,
    splits: SplitManifest,
    immutable_test: ImmutableTestManifest,
    qrels_result: QrelsBuildResult,
    summary: ValidationSummary,
    model_manifest_sha256: str,
) -> None:
    benchmark_root = repository_root / "benchmark"
    manifests = benchmark_root / "manifests"
    configs = benchmark_root / "configs"
    qrels_dir = benchmark_root / "qrels"
    normalized_dir = benchmark_root / "datasets/normalized"

    write_yaml_model(manifests / "adaptive_rag_bench_v1.yaml", manifest)
    write_json_model(configs / "splits_v1.json", splits)
    write_json_model(manifests / "test_ids_v1.json", immutable_test)
    write_jsonl_models(qrels_dir / "qrels_v1.jsonl", qrels_result.qrels)
    write_jsonl_models(manifests / "corpus_index_v1.jsonl", qrels_result.document_index)
    chunk_index = tuple(
        CorpusChunkIndex(
            canonical_chunk_id=chunk.id,
            document_id=chunk.document_id,
            position=chunk.position,
            token_count=chunk.token_count,
            text_sha256=hashlib.sha256(chunk.text.encode("utf-8")).hexdigest(),
        )
        for chunk in qrels_result.chunks
    )
    write_jsonl_models(manifests / "chunk_index_v1.jsonl", chunk_index)
    write_jsonl_models(normalized_dir / "chunks_v1.jsonl", qrels_result.chunks)
    write_jsonl_values(normalized_dir / "corpus_v1.jsonl", _corpus_rows(selected))

    report_values = [
        {
            **asdict(report),
            "source_dataset": report.source_dataset.value,
            "exclusions": dict(report.exclusions),
        }
        for report in reports
    ]
    write_json(
        manifests / "selection_report_v1.json",
        {
            "benchmark_manifest_sha256": manifest.manifest_sha256,
            "source_id_policy": {
                "nq": "SHA256(normalized question) because DPR NQ records expose no source ID",
                "hotpot_bridge": "upstream HotpotQA id",
                "hotpot_comparison": "upstream HotpotQA id",
                "musique": "upstream MuSiQue id",
            },
            "duplicate_policy": (
                "retain the first occurrence in each checksum-pinned upstream train artifact "
                "for duplicate source IDs or normalized questions"
            ),
            "nq_musique_singlehop_normalized_question_hash_overlap": 0,
            "reports": report_values,
        },
    )
    write_json(
        manifests / "duplicate_report_v1.json",
        {
            "exact_duplicate_aliases_before_canonicalization": exact_duplicate_aliases,
            "exact_duplicate_aliases_after_canonicalization": (
                qrels_result.exact_duplicate_aliases
            ),
            "near_duplicate_pairs": qrels_result.near_duplicate_pairs,
        },
    )
    write_json(
        manifests / "corpus_policy_v1.json",
        {
            "exact_duplicate_policy": (
                "merge normalized byte-identical text within each source namespace; "
                "choose the lexicographically smallest source_document_id"
            ),
            "near_duplicate_policy": (
                "retain and audit same-title token-Jaccard pairs in [0.90, 1.00); "
                "do not merge because doing so can remove upstream gold evidence"
            ),
            "near_duplicate_threshold": 0.9,
            "exact_duplicate_groups_after_canonicalization": len(
                qrels_result.exact_duplicate_aliases
            ),
            "exact_duplicate_groups_before_canonicalization": len(exact_duplicate_aliases),
            "near_duplicate_pairs_after_canonicalization": len(qrels_result.near_duplicate_pairs),
        },
    )
    write_json(
        manifests / "build_report_v1.json",
        {
            **asdict(summary),
            "benchmark_manifest_sha256": manifest.manifest_sha256,
            "split_hash": splits.split_hash,
            "qrels_logical_sha256": canonical_sha256(
                [qrel.model_dump(mode="json") for qrel in qrels_result.qrels]
            ),
            "split_counts": {key.value: value for key, value in summary.split_counts.items()},
            "test_tag_counts": {key.value: value for key, value in summary.test_tag_counts.items()},
            "qrel_grade_counts": {
                str(grade): count
                for grade, count in sorted(
                    Counter(qrel.relevance_grade for qrel in qrels_result.qrels).items()
                )
            },
            "corpus_version": CORPUS_VERSION,
            "embedding_artifact_manifest_sha256": model_manifest_sha256,
            "normalized_corpus_sha256": file_sha256(normalized_dir / "corpus_v1.jsonl"),
            "normalized_chunks_sha256": file_sha256(normalized_dir / "chunks_v1.jsonl"),
            "chunker": {"window_size": 220, "overlap": 40},
        },
    )
    write_json(
        manifests / "corpus_manifest_v1.json",
        {
            "corpus_version": CORPUS_VERSION,
            "document_count": len(qrels_result.document_index),
            "chunk_count": len(chunk_index),
            "document_index_sha256": canonical_sha256(
                [item.model_dump(mode="json") for item in qrels_result.document_index]
            ),
            "chunk_index_sha256": canonical_sha256(
                [item.model_dump(mode="json") for item in chunk_index]
            ),
            "chunker": {"window_size": 220, "overlap": 40},
            "embedding_artifact_manifest_sha256": model_manifest_sha256,
        },
    )

    synthetic = build_synthetic_graph_fixture()
    write_json_model(manifests / "synthetic_graph_v1.json", synthetic)
    write_jsonl_values(
        qrels_dir / "synthetic_graph_qrels_v1.jsonl",
        (
            {
                "query_id": query.query_id,
                "canonical_chunk_id": chunk_id,
                "relevance_grade": 2,
                "hop_count": query.hop_count,
                "primary_metrics_eligible": False,
            }
            for query in synthetic.queries
            for chunk_id in query.relevant_chunk_ids
        ),
    )
    _write_artifact_set(
        benchmark_root=benchmark_root,
        benchmark_manifest_sha256=manifest.manifest_sha256,
        split_hash=splits.split_hash,
        test_ids_sha256=immutable_test.query_ids_sha256,
        qrels=qrels_result,
    )


def _corpus_rows(queries: Sequence[NormalizedQuery]) -> tuple[Mapping[str, Any], ...]:
    documents: dict[str, Mapping[str, Any]] = {}
    for query in queries:
        namespace = document_namespace(query.source_dataset)
        for paragraph in query.paragraphs:
            document_id = canonical_document_id(namespace, paragraph.source_document_id)
            row: Mapping[str, Any] = {
                "document_id": document_id,
                "source_dataset": namespace,
                "source_document_id": paragraph.source_document_id,
                "title": paragraph.title,
                "text": paragraph.text,
                "text_sha256": hashlib.sha256(paragraph.text.encode("utf-8")).hexdigest(),
            }
            previous = documents.setdefault(document_id, row)
            if previous != row:
                raise ValueError(f"canonical corpus document changed: {document_id}")
    return tuple(documents[key] for key in sorted(documents))


def _write_artifact_set(
    *,
    benchmark_root: Path,
    benchmark_manifest_sha256: str,
    split_hash: str,
    test_ids_sha256: str,
    qrels: QrelsBuildResult,
) -> None:
    """Publish one readiness record after every public Stage 2 artifact is complete."""

    relative_paths = (
        "NOTICE.md",
        "configs/splits_v1.json",
        "manifests/adaptive_rag_bench_v1.yaml",
        "manifests/build_report_v1.json",
        "manifests/chunk_index_v1.jsonl",
        "manifests/corpus_index_v1.jsonl",
        "manifests/corpus_manifest_v1.json",
        "manifests/corpus_policy_v1.json",
        "manifests/duplicate_report_v1.json",
        "manifests/licenses.yaml",
        "manifests/selection_report_v1.json",
        "manifests/synthetic_graph_v1.json",
        "manifests/test_ids_v1.json",
        "qrels/qrels_v1.jsonl",
        "qrels/synthetic_graph_qrels_v1.jsonl",
    )
    files: dict[str, Mapping[str, int | str]] = {}
    for relative_path in relative_paths:
        path = benchmark_root / relative_path
        if not path.is_file():
            raise ValueError(f"required Stage 2 artifact is missing: {relative_path}")
        files[relative_path] = {
            "size_bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
    write_json(
        benchmark_root / "manifests/artifact_set_v1.json",
        {
            "schema_version": "v1",
            "ready": True,
            "benchmark_manifest_sha256": benchmark_manifest_sha256,
            "split_hash": split_hash,
            "test_ids_sha256": test_ids_sha256,
            "qrels_logical_sha256": canonical_sha256(
                [qrel.model_dump(mode="json") for qrel in qrels.qrels]
            ),
            "files": files,
        },
    )
