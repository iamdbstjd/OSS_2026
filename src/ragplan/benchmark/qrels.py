"""Production-chunker corpus materialization and deterministic graded qrels."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from ragplan.benchmark.contracts import (
    CORPUS_VERSION,
    BenchmarkQuery,
    CorpusDocumentIndex,
    Qrel,
    SourceDataset,
)
from ragplan.benchmark.selection import document_namespace
from ragplan.benchmark.sources import NormalizedQuery, SourceParagraph
from ragplan.core.ids import canonical_document_id
from ragplan.core.models import Chunk
from ragplan.ingestion.chunker import ChunkerConfig, Tokenizer, chunk_document
from ragplan.ingestion.normalize import normalize_text


@dataclass(slots=True)
class _DocumentState:
    namespace: str
    source_document_id: str
    title: str
    text: str
    source_dataset: SourceDataset
    query_ids: set[str] = field(default_factory=set)
    supporting_query_ids: set[str] = field(default_factory=set)


@dataclass(frozen=True, slots=True)
class QrelsBuildResult:
    document_index: tuple[CorpusDocumentIndex, ...]
    chunks: tuple[Chunk, ...]
    qrels: tuple[Qrel, ...]
    exact_duplicate_aliases: Mapping[str, tuple[str, ...]]
    near_duplicate_pairs: tuple[tuple[str, str, float], ...]


def build_corpus_and_qrels(
    normalized_queries: Sequence[NormalizedQuery],
    query_contracts: Mapping[str, BenchmarkQuery],
    *,
    tokenizer: Tokenizer,
) -> QrelsBuildResult:
    """Build one frozen corpus and project upstream support onto Stage 3 chunks."""

    documents = _collect_documents(normalized_queries)
    chunks_by_document: dict[str, tuple[Chunk, ...]] = {}
    all_chunks: list[Chunk] = []
    for document_id, state in sorted(documents.items()):
        chunks = chunk_document(
            source_dataset=state.namespace,
            source_document_id=state.source_document_id,
            corpus_version=CORPUS_VERSION,
            text=state.text,
            tokenizer=tokenizer,
            config=ChunkerConfig(window_size=220, overlap=40),
        )
        if not chunks:
            raise ValueError(f"document produced no chunks: {document_id}")
        if chunks[0].document_id != document_id:
            raise ValueError("production chunker document ID does not match corpus index")
        chunks_by_document[document_id] = chunks
        all_chunks.extend(chunks)

    qrels: list[Qrel] = []
    for query in normalized_queries:
        contract = query_contracts[query.query_id]
        namespace = document_namespace(query.source_dataset)
        for paragraph in query.supporting_paragraphs:
            document_id = canonical_document_id(namespace, paragraph.source_document_id)
            chunks = chunks_by_document[document_id]
            qrels.extend(
                _paragraph_qrels(
                    query=contract,
                    paragraph=paragraph,
                    chunks=chunks,
                    tokenizer=tokenizer,
                )
            )

    document_index = tuple(
        CorpusDocumentIndex(
            document_id=document_id,
            source_dataset=state.source_dataset,
            source_document_id=state.source_document_id,
            title=state.title,
            text_sha256=hashlib.sha256(state.text.encode("utf-8")).hexdigest(),
            query_ids=tuple(sorted(state.query_ids)),
            supporting_query_ids=tuple(sorted(state.supporting_query_ids)),
        )
        for document_id, state in sorted(documents.items())
    )
    exact_aliases, near_duplicates = _duplicate_report(documents)
    return QrelsBuildResult(
        document_index=document_index,
        chunks=tuple(sorted(all_chunks, key=lambda item: item.id)),
        qrels=tuple(
            sorted(
                qrels,
                key=lambda item: (
                    item.query_id,
                    item.canonical_chunk_id,
                    item.supporting_fact_id,
                ),
            )
        ),
        exact_duplicate_aliases=exact_aliases,
        near_duplicate_pairs=near_duplicates,
    )


def aggregate_relevance(qrels: Sequence[Qrel]) -> dict[str, dict[str, int]]:
    """Collapse multiple fact rows to maximum grade per query/chunk for metrics."""

    aggregated: dict[str, dict[str, int]] = defaultdict(dict)
    for qrel in qrels:
        previous = aggregated[qrel.query_id].get(qrel.canonical_chunk_id, 0)
        aggregated[qrel.query_id][qrel.canonical_chunk_id] = max(previous, qrel.relevance_grade)
    return dict(aggregated)


def _collect_documents(
    normalized_queries: Sequence[NormalizedQuery],
) -> dict[str, _DocumentState]:
    documents: dict[str, _DocumentState] = {}
    for query in normalized_queries:
        namespace = document_namespace(query.source_dataset)
        for paragraph in query.paragraphs:
            document_id = canonical_document_id(namespace, paragraph.source_document_id)
            state = documents.get(document_id)
            if state is None:
                state = _DocumentState(
                    namespace=namespace,
                    source_document_id=paragraph.source_document_id,
                    title=paragraph.title,
                    text=paragraph.text,
                    source_dataset=query.source_dataset,
                )
                documents[document_id] = state
            elif state.text != paragraph.text or state.title != paragraph.title:
                raise ValueError(f"canonical document content changed: {document_id}")
            state.query_ids.add(query.query_id)
            if paragraph.is_supporting:
                state.supporting_query_ids.add(query.query_id)
    return documents


def _paragraph_qrels(
    *,
    query: BenchmarkQuery,
    paragraph: SourceParagraph,
    chunks: Sequence[Chunk],
    tokenizer: Tokenizer,
) -> list[Qrel]:
    rows: list[Qrel] = []
    for fact in paragraph.supporting_facts:
        sentence_key = _tokenizer_match_key(fact.sentence, tokenizer=tokenizer)
        grade_two_ids = {
            chunk.id
            for chunk in chunks
            if sentence_key is not None and sentence_key in _match_key(chunk.text)
        }
        # A long sentence can straddle the deterministic 220-token window.
        # In that case the supporting paragraph remains grade 1, but no chunk
        # may be promoted to grade 2 because the full gold sentence is absent.
        for chunk in chunks:
            rows.append(
                Qrel(
                    query_id=query.query_id,
                    canonical_chunk_id=chunk.id,
                    relevance_grade=2 if chunk.id in grade_two_ids else 1,
                    query_tags=query.query_tags,
                    source_dataset=query.source_dataset,
                    supporting_fact_id=fact.fact_id,
                )
            )
    return rows


def _tokenizer_match_key(sentence: str | None, *, tokenizer: Tokenizer) -> str | None:
    if sentence is None:
        return None
    encoded = tokenizer.encode(normalize_text(sentence))
    if encoded.token_count == 0:
        return None
    return _match_key(encoded.decode(0, encoded.token_count))


def _match_key(value: str) -> str:
    normalized = normalize_text(value).casefold()
    normalized = normalized.replace(" ,", ",").replace(" .", ".")
    normalized = normalized.replace(" ?", "?").replace(" !", "!")
    return " ".join(normalized.split())


def _duplicate_report(
    documents: Mapping[str, _DocumentState],
) -> tuple[dict[str, tuple[str, ...]], tuple[tuple[str, str, float], ...]]:
    ids_by_hash: dict[str, list[str]] = defaultdict(list)
    ids_by_title: dict[str, list[str]] = defaultdict(list)
    for document_id, state in documents.items():
        digest = hashlib.sha256(state.text.encode("utf-8")).hexdigest()
        ids_by_hash[digest].append(document_id)
        ids_by_title[normalize_text(state.title).casefold()].append(document_id)
    exact = {
        digest: tuple(sorted(ids_)) for digest, ids_ in sorted(ids_by_hash.items()) if len(ids_) > 1
    }
    near: list[tuple[str, str, float]] = []
    for ids_ in ids_by_title.values():
        if len(ids_) < 2:
            continue
        for left_index, left_id in enumerate(sorted(ids_)):
            left_tokens = set(documents[left_id].text.casefold().split())
            for right_id in sorted(ids_)[left_index + 1 :]:
                right_tokens = set(documents[right_id].text.casefold().split())
                union = left_tokens | right_tokens
                similarity = len(left_tokens & right_tokens) / len(union) if union else 1.0
                if 0.9 <= similarity < 1.0:
                    near.append((left_id, right_id, round(similarity, 6)))
    return exact, tuple(sorted(near))
