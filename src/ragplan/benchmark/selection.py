"""Deterministic query selection, taxonomy, grouping, and quota-exact splits."""

from __future__ import annotations

import hashlib
import heapq
import importlib
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, cast

from ragplan.benchmark.contracts import (
    REQUIRED_TEST_TAG_MINIMUM,
    SOURCE_SPLIT_QUOTAS,
    SOURCE_TOTALS,
    SPLIT_SEED,
    BenchmarkQuery,
    QueryTag,
    SourceDataset,
    SplitAssignment,
    SplitName,
    TagEvidence,
)
from ragplan.benchmark.sources import CandidateOutcome, NormalizedQuery, question_overlap_hash
from ragplan.core.ids import ALLOWED_ENTITY_TYPES, canonical_document_id, normalize_entity_name

PINNED_NER_PACKAGE = "en_core_web_sm"
PINNED_NER_VERSION = "3.8.0"
_DOCUMENT_NAMESPACES: Mapping[SourceDataset, str] = {
    SourceDataset.NQ: "dpr_nq",
    SourceDataset.HOTPOT_BRIDGE: "hotpotqa",
    SourceDataset.HOTPOT_COMPARISON: "hotpotqa",
    SourceDataset.MUSIQUE: "musique",
}


class _EntityLike(Protocol):
    text: str
    label_: str


class _DocumentLike(Protocol):
    ents: Iterable[_EntityLike]


class _NlpLike(Protocol):
    def __call__(self, text: str) -> _DocumentLike: ...


@dataclass(frozen=True, slots=True)
class EntityMention:
    label: str
    normalized_name: str


@dataclass(frozen=True, slots=True)
class SelectionReport:
    source_dataset: SourceDataset
    seen_records: int
    eligible_records: int
    selected_records: int
    exclusions: Mapping[str, int]
    duplicate_source_ids: int
    duplicate_questions: int


class PinnedNer:
    """Exact spaCy package/version boundary used by ADR-006 taxonomy and grouping."""

    def __init__(self, nlp: _NlpLike) -> None:
        self._nlp = nlp

    @classmethod
    def load(cls) -> PinnedNer:
        module = importlib.import_module(PINNED_NER_PACKAGE)
        version = getattr(module, "__version__", None)
        if version != PINNED_NER_VERSION:
            raise RuntimeError(
                f"{PINNED_NER_PACKAGE} {PINNED_NER_VERSION} is required; found {version!r}"
            )
        loader = getattr(module, "load", None)
        if not callable(loader):
            raise RuntimeError("pinned spaCy model has no load function")
        nlp = loader(disable=["parser", "tagger", "lemmatizer", "attribute_ruler"])
        return cls(cast(_NlpLike, nlp))

    def mentions(self, text: str) -> tuple[EntityMention, ...]:
        mentions: set[EntityMention] = set()
        for entity in self._nlp(text).ents:
            label = entity.label_.upper()
            if label not in ALLOWED_ENTITY_TYPES:
                continue
            try:
                normalized_name = normalize_entity_name(entity.text)
            except ValueError:
                continue
            mentions.add(EntityMention(label=label, normalized_name=normalized_name))
        return tuple(sorted(mentions, key=lambda item: (item.label, item.normalized_name)))


def select_smallest_hashes(
    outcomes: Iterable[CandidateOutcome],
    *,
    source_dataset: SourceDataset,
    limit: int,
) -> tuple[tuple[NormalizedQuery, ...], SelectionReport]:
    """Keep the exact smallest selection hashes with bounded memory."""

    if limit < 1:
        raise ValueError("selection limit must be positive")
    heap: list[tuple[int, int, NormalizedQuery]] = []
    seen_source_ids: set[str] = set()
    seen_questions: set[str] = set()
    exclusions: Counter[str] = Counter()
    seen_records = 0
    eligible_records = 0
    duplicate_source_ids = 0
    duplicate_questions = 0
    counter = 0
    for outcome in outcomes:
        seen_records += 1
        if outcome.query is None:
            exclusions[outcome.exclusion_reason or "unspecified"] += 1
            continue
        query = outcome.query
        if query.source_dataset is not source_dataset:
            continue
        if query.source_query_id in seen_source_ids:
            duplicate_source_ids += 1
            continue
        seen_source_ids.add(query.source_query_id)
        question_hash = question_overlap_hash(query.question)
        if question_hash in seen_questions:
            duplicate_questions += 1
            continue
        seen_questions.add(question_hash)
        eligible_records += 1
        counter += 1
        entry = (-int(query.selection_hash, 16), counter, query)
        if len(heap) < limit:
            heapq.heappush(heap, entry)
        elif entry[0] > heap[0][0]:
            heapq.heapreplace(heap, entry)
    selected = tuple(sorted((item[2] for item in heap), key=lambda item: item.selection_hash))
    if len(selected) != limit:
        raise ValueError(
            f"{source_dataset.value} has only {len(selected)} eligible records; expected {limit}"
        )
    return (
        selected,
        SelectionReport(
            source_dataset=source_dataset,
            seen_records=seen_records,
            eligible_records=eligible_records,
            selected_records=len(selected),
            exclusions=dict(sorted(exclusions.items())),
            duplicate_source_ids=duplicate_source_ids,
            duplicate_questions=duplicate_questions,
        ),
    )


def build_query_contract(query: NormalizedQuery, *, ner: PinnedNer) -> BenchmarkQuery:
    mentions = ner.mentions(query.question)
    tags, evidence = _classify(query, mentions=mentions)
    namespace = _DOCUMENT_NAMESPACES[query.source_dataset]
    corpus_document_ids = tuple(
        sorted(
            {
                canonical_document_id(namespace, paragraph.source_document_id)
                for paragraph in query.paragraphs
            }
        )
    )
    supporting_document_ids = tuple(
        sorted(
            {
                canonical_document_id(namespace, paragraph.source_document_id)
                for paragraph in query.supporting_paragraphs
            }
        )
    )
    # Every supporting *and distractor* passage is part of the frozen corpus.
    # Grouping only positive passages would leak an identical distractor into
    # another split and make retrieval quality optimistically biased.
    group_keys = {f"document:{document_id}" for document_id in corpus_document_ids}
    group_keys.update(f"entity:{mention.label}:{mention.normalized_name}" for mention in mentions)
    group_keys.add(f"template:{_query_template_hash(query.question, mentions=mentions)}")
    return BenchmarkQuery(
        query_id=query.query_id,
        source_dataset=query.source_dataset,
        source_query_id=query.source_query_id,
        question=query.question,
        answers=query.answers,
        query_tags=tags,
        tag_evidence=evidence,
        selection_hash=query.selection_hash,
        supporting_document_ids=supporting_document_ids,
        group_keys=tuple(sorted(group_keys)),
    )


def assign_splits(queries: Sequence[BenchmarkQuery]) -> tuple[SplitAssignment, ...]:
    """Assign complete document/entity components while satisfying every source quota."""

    if len(queries) != sum(SOURCE_TOTALS.values()):
        raise ValueError("split generation requires exactly 600 queries")
    components = _connected_components(queries)
    for attempt in range(10_000):
        assignments = _attempt_assignment(components, attempt=attempt)
        if assignments is None:
            continue
        test_tag_counts = _test_tag_counts(queries, assignments)
        if all(test_tag_counts[tag] >= REQUIRED_TEST_TAG_MINIMUM for tag in QueryTag):
            return tuple(sorted(assignments, key=lambda item: item.query_id))
    raise ValueError("group constraints, quotas, and test tag minimums cannot be satisfied")


def document_namespace(source_dataset: SourceDataset) -> str:
    return _DOCUMENT_NAMESPACES[source_dataset]


def query_template_hash(question: str, *, ner: PinnedNer) -> str:
    """Return the frozen entity/number-masked template signature used for leakage checks."""

    return _query_template_hash(question, mentions=ner.mentions(question))


def _classify(
    query: NormalizedQuery, *, mentions: tuple[EntityMention, ...]
) -> tuple[tuple[QueryTag, ...], tuple[TagEvidence, ...]]:
    evidence: list[TagEvidence] = []
    if query.source_dataset is SourceDataset.NQ:
        if mentions:
            names = ", ".join(f"{item.label}:{item.normalized_name}" for item in mentions)
            evidence.append(
                TagEvidence(
                    tag=QueryTag.ENTITY,
                    rule="NQ query with at least one allowed pinned-NER entity",
                    evidence=names,
                )
            )
        else:
            evidence.append(
                TagEvidence(
                    tag=QueryTag.SEMANTIC,
                    rule="NQ query with zero allowed pinned-NER entities",
                    evidence="allowed_entity_count=0",
                )
            )
    elif query.source_dataset is SourceDataset.HOTPOT_BRIDGE:
        evidence.append(
            TagEvidence(
                tag=QueryTag.RELATIONSHIP,
                rule="HotpotQA record has type=bridge",
                evidence="type=bridge",
            )
        )
    elif query.source_dataset is SourceDataset.HOTPOT_COMPARISON:
        evidence.append(
            TagEvidence(
                tag=QueryTag.COMPARISON,
                rule="HotpotQA record has type=comparison",
                evidence="type=comparison",
            )
        )
    else:
        evidence.append(
            TagEvidence(
                tag=QueryTag.THREE_HOP,
                rule="MuSiQue decomposition contains exactly three steps",
                evidence="decomposition_step_count=3",
            )
        )

    if query.source_dataset in {
        SourceDataset.HOTPOT_BRIDGE,
        SourceDataset.HOTPOT_COMPARISON,
    }:
        supporting_titles = {item.title for item in query.supporting_paragraphs}
        if len(supporting_titles) >= 2:
            evidence.append(
                TagEvidence(
                    tag=QueryTag.TWO_HOP,
                    rule="HotpotQA query has at least two supporting titles",
                    evidence=f"supporting_title_count={len(supporting_titles)}",
                )
            )
    evidence.sort(key=lambda item: item.tag.value)
    return tuple(item.tag for item in evidence), tuple(evidence)


def _connected_components(
    queries: Sequence[BenchmarkQuery],
) -> tuple[tuple[BenchmarkQuery, ...], ...]:
    parent = list(range(len(queries)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    owner_by_key: dict[str, int] = {}
    for index, query in enumerate(queries):
        for key in query.group_keys:
            owner = owner_by_key.setdefault(key, index)
            union(index, owner)
    grouped: dict[int, list[BenchmarkQuery]] = defaultdict(list)
    for index, query in enumerate(queries):
        grouped[find(index)].append(query)
    components = [
        tuple(sorted(group, key=lambda item: item.query_id)) for group in grouped.values()
    ]
    return tuple(sorted(components, key=lambda group: (-len(group), group[0].query_id)))


def _attempt_assignment(
    components: Sequence[tuple[BenchmarkQuery, ...]], *, attempt: int
) -> tuple[SplitAssignment, ...] | None:
    remaining = {
        source: {split: quota for split, quota in quotas.items()}
        for source, quotas in SOURCE_SPLIT_QUOTAS.items()
    }
    ordered = sorted(
        components,
        key=lambda group: (
            -len(group),
            _stable_int(f"{SPLIT_SEED}:{attempt}:{'|'.join(item.query_id for item in group)}"),
        ),
    )
    assignments: list[SplitAssignment] = []
    for component in ordered:
        needed = Counter(item.source_dataset for item in component)
        feasible = [
            split
            for split in SplitName
            if all(remaining[source][split] >= count for source, count in needed.items())
        ]
        if not feasible:
            return None

        def rank(split: SplitName) -> tuple[float, int]:
            pressure = sum(
                remaining[source][split] / SOURCE_SPLIT_QUOTAS[source][split] for source in needed
            )
            tie = _stable_int(
                f"{SPLIT_SEED}:{attempt}:{split.value}:"
                f"{'|'.join(item.query_id for item in component)}"
            )
            return (-pressure, tie)

        chosen = min(feasible, key=rank)
        for source, count in needed.items():
            remaining[source][chosen] -= count
        assignments.extend(
            SplitAssignment(query_id=item.query_id, split=chosen) for item in component
        )
    if any(value for source in remaining.values() for value in source.values()):
        return None
    return tuple(assignments)


def _test_tag_counts(
    queries: Sequence[BenchmarkQuery], assignments: Sequence[SplitAssignment]
) -> Counter[QueryTag]:
    split_by_id = {item.query_id: item.split for item in assignments}
    return Counter(
        tag
        for query in queries
        if split_by_id[query.query_id] is SplitName.TEST
        for tag in query.query_tags
    )


def _stable_int(value: str) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest(), 16)


def _query_template_hash(question: str, *, mentions: Sequence[EntityMention]) -> str:
    template = " ".join(question.casefold().split())
    for mention in sorted(mentions, key=lambda item: -len(item.normalized_name)):
        escaped = re.escape(mention.normalized_name)
        template = re.sub(
            rf"(?<!\w){escaped}(?!\w)",
            f" entity_{mention.label.casefold()} ",
            template,
        )
    template = re.sub(r"\d+(?:[.,:/-]\d+)*", " number ", template)
    return question_overlap_hash(template)
