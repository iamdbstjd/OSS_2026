"""Streaming adapters from pinned upstream formats to one normalized benchmark shape."""

from __future__ import annotations

import gzip
import hashlib
import json
import re
import unicodedata
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast
from urllib.parse import quote

from ragplan.benchmark.contracts import BENCHMARK_ID, SPLIT_SEED, QueryId, SourceDataset
from ragplan.ingestion.normalize import normalize_text

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[\"'([{A-Z0-9])")


@dataclass(frozen=True, slots=True)
class SupportingFact:
    fact_id: str
    sentence: str | None


@dataclass(frozen=True, slots=True)
class SourceParagraph:
    source_document_id: str
    title: str
    text: str
    supporting_facts: tuple[SupportingFact, ...] = ()

    @property
    def is_supporting(self) -> bool:
        return bool(self.supporting_facts)


@dataclass(frozen=True, slots=True)
class NormalizedQuery:
    source_dataset: SourceDataset
    source_query_id: str
    question: str
    answers: tuple[str, ...]
    paragraphs: tuple[SourceParagraph, ...]

    @property
    def query_id(self) -> QueryId:
        encoded = quote(self.source_query_id, safe="-._~")
        return f"{BENCHMARK_ID}:{self.source_dataset.value}:{encoded}"

    @property
    def selection_hash(self) -> str:
        value = f"{self.source_dataset.value}:{self.source_query_id}:{SPLIT_SEED}"
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    @property
    def supporting_paragraphs(self) -> tuple[SourceParagraph, ...]:
        return tuple(item for item in self.paragraphs if item.is_supporting)


@dataclass(frozen=True, slots=True)
class CandidateOutcome:
    query: NormalizedQuery | None
    exclusion_reason: str | None = None


def iter_nq(path: Path, *, excluded_question_hashes: frozenset[str]) -> Iterator[CandidateOutcome]:
    """Stream the DPR NQ train array without materializing its multi-gigabyte JSON."""

    try:
        import ijson  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover - dependency-group guard
        raise RuntimeError("install the benchmark dependency group to parse DPR NQ") from exc

    with gzip.open(path, "rb") as source:
        for raw in ijson.items(source, "item"):
            yield _normalize_nq(_as_mapping(raw), excluded_question_hashes=excluded_question_hashes)


def iter_hotpot(paths: Sequence[Path]) -> Iterator[CandidateOutcome]:
    """Stream checksum-pinned HotpotQA parquet shards from the official dataset mirror."""

    try:
        import pyarrow.parquet as parquet  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover - dependency-group guard
        raise RuntimeError("install the benchmark dependency group to parse HotpotQA") from exc

    for path in paths:
        source = parquet.ParquetFile(path)
        for batch in source.iter_batches(batch_size=512):
            for raw in batch.to_pylist():
                yield _normalize_hotpot(_as_mapping(raw))


def iter_musique(path: Path) -> Iterator[CandidateOutcome]:
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid MuSiQue JSON at line {line_number}") from exc
            yield _normalize_musique(_as_mapping(raw))


def load_musique_singlehop_question_hashes(path: Path) -> frozenset[str]:
    """Freeze the overlap guard against MuSiQue's published dev/test single-hop sources."""

    decoded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(decoded, dict):
        raise ValueError("MuSiQue single-hop metadata must be an object")
    hashes: set[str] = set()
    for records in decoded.values():
        if not isinstance(records, list):
            raise ValueError("MuSiQue single-hop metadata values must be lists")
        for raw in records:
            record = _as_mapping(raw)
            question = _required_string(record, "question")
            hashes.add(question_overlap_hash(question))
    return frozenset(hashes)


def question_overlap_hash(question: str) -> str:
    normalized = []
    for character in unicodedata.normalize("NFKC", question).casefold():
        category = unicodedata.category(character)
        if character.isspace() or category.startswith("P"):
            normalized.append(" ")
        else:
            normalized.append(character)
    collapsed = " ".join("".join(normalized).split())
    return hashlib.sha256(collapsed.encode("utf-8")).hexdigest()


def split_sentences(text: str) -> tuple[str, ...]:
    normalized = normalize_text(text)
    if not normalized:
        return ()
    return tuple(item.strip() for item in _SENTENCE_BOUNDARY.split(normalized) if item.strip())


def _normalize_nq(
    raw: Mapping[str, Any], *, excluded_question_hashes: frozenset[str]
) -> CandidateOutcome:
    question = normalize_text(_required_string(raw, "question"))
    overlap_hash = question_overlap_hash(question)
    if overlap_hash in excluded_question_hashes:
        return CandidateOutcome(query=None, exclusion_reason="musique_singlehop_question_overlap")
    source_query_id = overlap_hash
    answers = _unique_strings(_required_sequence(raw, "answers"))
    if not answers:
        return CandidateOutcome(query=None, exclusion_reason="missing_answer")

    positives = _required_sequence(raw, "positive_ctxs")
    hard_negatives = _required_sequence(raw, "hard_negative_ctxs")[:9]
    paragraphs: list[SourceParagraph] = []
    seen_documents: set[str] = set()
    mapped_sentence_count = 0
    for raw_context in positives:
        context = _as_mapping(raw_context)
        paragraph, mapped = _nq_paragraph(context, answers=answers, supporting=True)
        if paragraph.source_document_id not in seen_documents:
            seen_documents.add(paragraph.source_document_id)
            paragraphs.append(paragraph)
            mapped_sentence_count += mapped
    for raw_context in hard_negatives:
        context = _as_mapping(raw_context)
        paragraph, _ = _nq_paragraph(context, answers=answers, supporting=False)
        if paragraph.source_document_id not in seen_documents:
            seen_documents.add(paragraph.source_document_id)
            paragraphs.append(paragraph)
    if not paragraphs or not any(item.is_supporting for item in paragraphs):
        return CandidateOutcome(query=None, exclusion_reason="missing_positive_context")
    if mapped_sentence_count == 0:
        return CandidateOutcome(query=None, exclusion_reason="answer_not_found_in_positive_context")
    return CandidateOutcome(
        query=NormalizedQuery(
            source_dataset=SourceDataset.NQ,
            source_query_id=source_query_id,
            question=question,
            answers=answers,
            paragraphs=tuple(paragraphs),
        )
    )


def _nq_paragraph(
    raw: Mapping[str, Any], *, answers: tuple[str, ...], supporting: bool
) -> tuple[SourceParagraph, int]:
    title = normalize_text(_required_string(raw, "title"))
    text = normalize_text(_required_string(raw, "text"))
    source_document_id = str(raw.get("passage_id") or _title_and_text_id(title, text))
    facts: list[SupportingFact] = []
    if supporting:
        answer_keys = tuple(_match_key(answer) for answer in answers)
        for sentence_index, sentence in enumerate(split_sentences(text)):
            sentence_key = _match_key(sentence)
            if any(answer_key and answer_key in sentence_key for answer_key in answer_keys):
                facts.append(
                    SupportingFact(
                        fact_id=f"{source_document_id}:answer_sentence:{sentence_index}",
                        sentence=sentence,
                    )
                )
        if not facts:
            facts.append(SupportingFact(fact_id=f"{source_document_id}:positive", sentence=None))
    return (
        SourceParagraph(
            source_document_id=source_document_id,
            title=title,
            text=text,
            supporting_facts=tuple(facts),
        ),
        sum(item.sentence is not None for item in facts),
    )


def _normalize_hotpot(raw: Mapping[str, Any]) -> CandidateOutcome:
    kind = _required_string(raw, "type")
    if kind == "bridge":
        source_dataset = SourceDataset.HOTPOT_BRIDGE
    elif kind == "comparison":
        source_dataset = SourceDataset.HOTPOT_COMPARISON
    else:
        return CandidateOutcome(query=None, exclusion_reason="unsupported_hotpot_type")

    source_query_id = _required_string(raw, "id")
    question = normalize_text(_required_string(raw, "question"))
    answers = (normalize_text(_required_string(raw, "answer")),)
    context = _as_mapping(raw.get("context"))
    titles = _required_sequence(context, "title")
    sentence_groups = _required_sequence(context, "sentences")
    if len(titles) != len(sentence_groups):
        raise ValueError(f"Hotpot context length mismatch for {source_query_id}")

    facts_raw = _as_mapping(raw.get("supporting_facts"))
    fact_titles = _required_sequence(facts_raw, "title")
    fact_indices = _required_sequence(facts_raw, "sent_id")
    if len(fact_titles) != len(fact_indices):
        raise ValueError(f"Hotpot supporting-fact length mismatch for {source_query_id}")
    facts_by_title: dict[str, list[int]] = defaultdict(list)
    for raw_title, raw_index in zip(fact_titles, fact_indices, strict=True):
        if isinstance(raw_index, bool) or not isinstance(raw_index, int) or raw_index < 0:
            raise ValueError(f"invalid Hotpot sentence index for {source_query_id}")
        facts_by_title[str(raw_title)].append(raw_index)

    paragraphs: list[SourceParagraph] = []
    mapped_facts = 0
    for raw_title, raw_sentences in zip(titles, sentence_groups, strict=True):
        title = normalize_text(str(raw_title))
        # Supporting-fact indices are positional.  De-duplicating sentences here
        # would silently move every later index and corrupt the gold evidence.
        sentences = _normalized_strings(_as_sequence(raw_sentences))
        text = normalize_text(" ".join(sentences))
        document_source_id = _title_and_text_id(title, text)
        facts: list[SupportingFact] = []
        for sentence_index in facts_by_title.get(str(raw_title), []):
            if sentence_index >= len(sentences):
                return CandidateOutcome(query=None, exclusion_reason="invalid_supporting_sentence")
            if not sentences[sentence_index]:
                return CandidateOutcome(query=None, exclusion_reason="empty_supporting_sentence")
            facts.append(
                SupportingFact(
                    fact_id=f"{document_source_id}:sentence:{sentence_index}",
                    sentence=sentences[sentence_index],
                )
            )
            mapped_facts += 1
        paragraphs.append(
            SourceParagraph(
                source_document_id=document_source_id,
                title=title,
                text=text,
                supporting_facts=tuple(facts),
            )
        )
    if mapped_facts == 0:
        return CandidateOutcome(query=None, exclusion_reason="missing_supporting_fact")
    return CandidateOutcome(
        query=NormalizedQuery(
            source_dataset=source_dataset,
            source_query_id=source_query_id,
            question=question,
            answers=answers,
            paragraphs=tuple(paragraphs),
        )
    )


def _normalize_musique(raw: Mapping[str, Any]) -> CandidateOutcome:
    decomposition = _required_sequence(raw, "question_decomposition")
    if len(decomposition) != 3:
        return CandidateOutcome(query=None, exclusion_reason="not_exactly_three_hop")
    if raw.get("answerable") is not True:
        return CandidateOutcome(query=None, exclusion_reason="not_answerable")
    source_query_id = _required_string(raw, "id")
    paragraphs_raw = _required_sequence(raw, "paragraphs")
    support_by_index: dict[int, list[SupportingFact]] = defaultdict(list)
    for step_index, raw_step in enumerate(decomposition):
        step = _as_mapping(raw_step)
        paragraph_index = step.get("paragraph_support_idx")
        if isinstance(paragraph_index, bool) or not isinstance(paragraph_index, int):
            return CandidateOutcome(query=None, exclusion_reason="invalid_decomposition_support")
        answer = _required_string(step, "answer")
        support_by_index[paragraph_index].append(
            SupportingFact(
                fact_id=f"{source_query_id}:decomposition:{step_index}",
                sentence=answer,
            )
        )

    paragraphs: list[SourceParagraph] = []
    seen_indices: set[int] = set()
    for raw_paragraph in paragraphs_raw:
        paragraph = _as_mapping(raw_paragraph)
        index = paragraph.get("idx")
        if isinstance(index, bool) or not isinstance(index, int):
            raise ValueError(f"invalid MuSiQue paragraph index for {source_query_id}")
        seen_indices.add(index)
        title = normalize_text(_required_string(paragraph, "title"))
        text = normalize_text(_required_string(paragraph, "paragraph_text"))
        facts: list[SupportingFact] = []
        for fact in support_by_index.get(index, []):
            supporting_sentence = _find_answer_sentence(text, answer=fact.sentence or "")
            facts.append(SupportingFact(fact_id=fact.fact_id, sentence=supporting_sentence))
        paragraphs.append(
            SourceParagraph(
                source_document_id=_title_and_text_id(title, text),
                title=title,
                text=text,
                supporting_facts=tuple(facts),
            )
        )
    if set(support_by_index) - seen_indices:
        return CandidateOutcome(query=None, exclusion_reason="missing_decomposition_paragraph")
    if not all(
        any(fact.sentence for fact in item.supporting_facts)
        for item in paragraphs
        if item.is_supporting
    ):
        return CandidateOutcome(
            query=None,
            exclusion_reason="decomposition_answer_not_in_paragraph",
        )

    answer_values: list[object] = [raw.get("answer")]
    aliases = raw.get("answer_aliases", [])
    answer_values.extend(_as_sequence(aliases))
    answers = _unique_strings(answer_values)
    if not answers:
        return CandidateOutcome(query=None, exclusion_reason="missing_answer")
    return CandidateOutcome(
        query=NormalizedQuery(
            source_dataset=SourceDataset.MUSIQUE,
            source_query_id=source_query_id,
            question=normalize_text(_required_string(raw, "question")),
            answers=answers,
            paragraphs=tuple(paragraphs),
        )
    )


def _find_answer_sentence(text: str, *, answer: str) -> str | None:
    answer_key = _match_key(answer)
    for sentence in split_sentences(text):
        if answer_key and answer_key in _match_key(sentence):
            return sentence
    return None


def _title_and_text_id(title: str, text: str) -> str:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return f"{title}::{digest}"


def _match_key(value: str) -> str:
    return " ".join(normalize_text(value).casefold().split())


def _unique_strings(values: Sequence[object]) -> tuple[str, ...]:
    unique: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str):
            continue
        normalized = normalize_text(value)
        if normalized and normalized not in seen:
            seen.add(normalized)
            unique.append(normalized)
    return tuple(unique)


def _normalized_strings(values: Sequence[object]) -> tuple[str, ...]:
    normalized: list[str] = []
    for value in values:
        if not isinstance(value, str):
            raise ValueError("upstream sentence must be a string")
        sentence = normalize_text(value)
        # Empty upstream slots must remain in place because Hotpot supporting
        # facts address this list by integer index.
        normalized.append(sentence)
    return tuple(normalized)


def _as_mapping(value: object) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("upstream record must be an object")
    return cast(Mapping[str, Any], value)


def _as_sequence(value: object) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("upstream field must be an array")
    return cast(Sequence[object], value)


def _required_sequence(raw: Mapping[str, Any], key: str) -> Sequence[object]:
    return _as_sequence(raw.get(key))


def _required_string(raw: Mapping[str, Any], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"upstream field {key!r} must be a non-empty string")
    return value
