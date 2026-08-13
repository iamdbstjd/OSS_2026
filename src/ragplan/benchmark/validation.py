"""Fail-closed validation for frozen Stage 2 benchmark artifacts."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from ragplan.benchmark.contracts import (
    REQUIRED_TEST_TAG_MINIMUM,
    SOURCE_SPLIT_QUOTAS,
    BenchmarkManifest,
    CorpusDocumentIndex,
    ImmutableTestManifest,
    Qrel,
    QueryTag,
    SourceDataset,
    SplitManifest,
    SplitName,
    canonical_sha256,
)
from ragplan.benchmark.qrels import aggregate_relevance
from ragplan.benchmark.sources import question_overlap_hash


@dataclass(frozen=True, slots=True)
class ValidationSummary:
    query_count: int
    split_counts: Mapping[SplitName, int]
    qrel_row_count: int
    relevant_pair_count: int
    corpus_document_count: int
    test_tag_counts: Mapping[QueryTag, int]
    test_ids_sha256: str


def validate_stage2_artifacts(
    *,
    manifest: BenchmarkManifest,
    splits: SplitManifest,
    qrels: Sequence[Qrel],
    corpus_index: Sequence[CorpusDocumentIndex],
    immutable_test: ImmutableTestManifest,
    chunk_ids: frozenset[str] | None = None,
) -> ValidationSummary:
    """Validate every Stage 2 exit invariant without trusting producer metadata."""

    queries_by_id = {query.query_id: query for query in manifest.queries}
    if splits.benchmark_manifest_sha256 != manifest.manifest_sha256:
        raise ValueError("split manifest references a different benchmark manifest")
    split_by_id = {assignment.query_id: assignment.split for assignment in splits.assignments}
    if set(split_by_id) != set(queries_by_id):
        raise ValueError("split assignments must cover exactly the frozen query IDs")

    _validate_source_quotas(manifest, split_by_id)
    _validate_group_leakage(manifest, split_by_id)
    _validate_question_duplicates(manifest, split_by_id)
    tag_counts = _validate_test_tags(manifest, split_by_id)
    _validate_corpus_index(manifest, corpus_index, split_by_id=split_by_id)
    relevant_pair_count = _validate_qrels(
        manifest,
        qrels,
        chunk_ids=chunk_ids,
    )

    expected_test_ids = tuple(
        sorted(query_id for query_id, split in split_by_id.items() if split is SplitName.TEST)
    )
    if immutable_test.split_hash != splits.split_hash:
        raise ValueError("immutable test manifest references a different split")
    if immutable_test.query_ids != expected_test_ids:
        raise ValueError("immutable test IDs do not equal the frozen test split")

    return ValidationSummary(
        query_count=len(queries_by_id),
        split_counts=dict(Counter(split_by_id.values())),
        qrel_row_count=len(qrels),
        relevant_pair_count=relevant_pair_count,
        corpus_document_count=len(corpus_index),
        test_tag_counts=dict(tag_counts),
        test_ids_sha256=canonical_sha256(list(expected_test_ids)),
    )


def _validate_source_quotas(
    manifest: BenchmarkManifest,
    split_by_id: Mapping[str, SplitName],
) -> None:
    actual: Counter[tuple[SourceDataset, SplitName]] = Counter(
        (query.source_dataset, split_by_id[query.query_id]) for query in manifest.queries
    )
    for source, quotas in SOURCE_SPLIT_QUOTAS.items():
        for split, expected in quotas.items():
            if actual[(source, split)] != expected:
                raise ValueError(
                    f"source quota mismatch for {source.value}/{split.value}: "
                    f"{actual[(source, split)]} != {expected}"
                )


def _validate_group_leakage(
    manifest: BenchmarkManifest,
    split_by_id: Mapping[str, SplitName],
) -> None:
    splits_by_group: dict[str, set[SplitName]] = defaultdict(set)
    for query in manifest.queries:
        for group_key in query.group_keys:
            splits_by_group[group_key].add(split_by_id[query.query_id])
    leaking = sorted(key for key, values in splits_by_group.items() if len(values) > 1)
    if leaking:
        raise ValueError(f"document/entity group leakage detected: {leaking[0]}")


def _validate_question_duplicates(
    manifest: BenchmarkManifest,
    split_by_id: Mapping[str, SplitName],
) -> None:
    owners: dict[str, tuple[str, SplitName]] = {}
    for query in manifest.queries:
        digest = question_overlap_hash(query.question)
        existing = owners.get(digest)
        current = (query.query_id, split_by_id[query.query_id])
        if existing is not None and existing[1] is not current[1]:
            raise ValueError(
                f"normalized duplicate query crosses splits: {existing[0]} / {query.query_id}"
            )
        owners[digest] = current


def _validate_test_tags(
    manifest: BenchmarkManifest,
    split_by_id: Mapping[str, SplitName],
) -> Counter[QueryTag]:
    counts: Counter[QueryTag] = Counter(
        tag
        for query in manifest.queries
        if split_by_id[query.query_id] is SplitName.TEST
        for tag in query.query_tags
    )
    missing = {
        tag.value: counts[tag] for tag in QueryTag if counts[tag] < REQUIRED_TEST_TAG_MINIMUM
    }
    if missing:
        raise ValueError(f"test taxonomy minimum is not satisfied: {missing}")
    return counts


def _validate_corpus_index(
    manifest: BenchmarkManifest,
    corpus_index: Sequence[CorpusDocumentIndex],
    *,
    split_by_id: Mapping[str, SplitName],
) -> None:
    indexed = {document.document_id: document for document in corpus_index}
    if len(indexed) != len(corpus_index):
        raise ValueError("corpus index contains duplicate canonical document IDs")
    query_ids = {query.query_id for query in manifest.queries}
    for document in corpus_index:
        if not set(document.query_ids) <= query_ids:
            raise ValueError("corpus index references an unknown query")
        if not set(document.supporting_query_ids) <= set(document.query_ids):
            raise ValueError("supporting query IDs must be a subset of document query IDs")
        document_splits = {split_by_id[query_id] for query_id in document.query_ids}
        if len(document_splits) > 1:
            raise ValueError(f"corpus document crosses splits: {document.document_id}")
    required_documents = {
        document_id for query in manifest.queries for document_id in query.supporting_document_ids
    }
    if not required_documents <= set(indexed):
        raise ValueError("a supporting document is absent from the frozen corpus")


def _validate_qrels(
    manifest: BenchmarkManifest,
    qrels: Sequence[Qrel],
    *,
    chunk_ids: frozenset[str] | None,
) -> int:
    queries_by_id = {query.query_id: query for query in manifest.queries}
    unique_rows: set[tuple[str, str, str]] = set()
    for qrel in qrels:
        query = queries_by_id.get(qrel.query_id)
        if query is None:
            raise ValueError("qrels contain an orphan query ID")
        if chunk_ids is not None and qrel.canonical_chunk_id not in chunk_ids:
            raise ValueError("qrels contain an orphan canonical chunk ID")
        if qrel.source_dataset is not query.source_dataset:
            raise ValueError("qrel source does not match its benchmark query")
        if qrel.query_tags != query.query_tags:
            raise ValueError("qrel tags do not match their benchmark query")
        row_key = (qrel.query_id, qrel.canonical_chunk_id, qrel.supporting_fact_id)
        if row_key in unique_rows:
            raise ValueError("qrels contain a duplicate query/chunk/fact row")
        unique_rows.add(row_key)

    relevance = aggregate_relevance(qrels)
    if set(relevance) != set(queries_by_id):
        raise ValueError("every benchmark query must have at least one qrel")
    if any(not any(grade >= 1 for grade in values.values()) for values in relevance.values()):
        raise ValueError("every benchmark query must have at least one relevant chunk")
    return sum(len(values) for values in relevance.values())
