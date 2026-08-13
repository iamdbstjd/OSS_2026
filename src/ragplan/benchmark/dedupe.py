"""Evidence-preserving canonicalization of exact duplicate corpus documents."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from ragplan.benchmark.selection import document_namespace
from ragplan.benchmark.sources import (
    NormalizedQuery,
    SourceParagraph,
    SupportingFact,
)
from ragplan.core.ids import canonical_document_id


@dataclass(frozen=True, slots=True)
class ExactDeduplicationResult:
    queries: tuple[NormalizedQuery, ...]
    aliases: Mapping[str, tuple[str, ...]]


def deduplicate_exact_documents(
    queries: Sequence[NormalizedQuery],
) -> tuple[NormalizedQuery, ...]:
    """Map byte-identical normalized documents to one deterministic source ID.

    Near duplicates are intentionally not merged: changing their text can remove
    the sentence that established upstream relevance.  They are reported by the
    corpus builder for audit instead.
    """

    return deduplicate_exact_documents_with_report(queries).queries


def deduplicate_exact_documents_with_report(
    queries: Sequence[NormalizedQuery],
) -> ExactDeduplicationResult:
    """Return canonicalized queries plus all pre-canonicalization alias IDs."""

    grouped: dict[tuple[str, str], list[SourceParagraph]] = defaultdict(list)
    for query in queries:
        namespace = document_namespace(query.source_dataset)
        for paragraph in query.paragraphs:
            digest = hashlib.sha256(paragraph.text.encode("utf-8")).hexdigest()
            grouped[(namespace, digest)].append(paragraph)

    representatives: dict[tuple[str, str], SourceParagraph] = {}
    for key, paragraphs in grouped.items():
        representatives[key] = min(
            paragraphs,
            key=lambda item: (item.source_document_id, item.title),
        )

    normalized_queries: list[NormalizedQuery] = []
    for query in queries:
        namespace = document_namespace(query.source_dataset)
        merged: dict[str, SourceParagraph] = {}
        facts_by_document: dict[str, set[SupportingFact]] = defaultdict(set)
        for paragraph in query.paragraphs:
            digest = hashlib.sha256(paragraph.text.encode("utf-8")).hexdigest()
            representative = representatives[(namespace, digest)]
            document_id = representative.source_document_id
            merged[document_id] = SourceParagraph(
                source_document_id=document_id,
                title=representative.title,
                text=representative.text,
            )
            facts_by_document[document_id].update(paragraph.supporting_facts)
        remapped = tuple(
            SourceParagraph(
                source_document_id=document_id,
                title=paragraph.title,
                text=paragraph.text,
                supporting_facts=tuple(
                    sorted(
                        facts_by_document[document_id],
                        key=lambda item: (item.fact_id, item.sentence or ""),
                    )
                ),
            )
            for document_id, paragraph in sorted(merged.items())
        )
        normalized_queries.append(
            NormalizedQuery(
                source_dataset=query.source_dataset,
                source_query_id=query.source_query_id,
                question=query.question,
                answers=query.answers,
                paragraphs=remapped,
            )
        )
    aliases = {
        f"{namespace}:{digest}": tuple(
            sorted(
                {
                    canonical_document_id(namespace, paragraph.source_document_id)
                    for paragraph in paragraphs
                }
            )
        )
        for (namespace, digest), paragraphs in sorted(grouped.items())
        if len({paragraph.source_document_id for paragraph in paragraphs}) > 1
    }
    return ExactDeduplicationResult(queries=tuple(normalized_queries), aliases=aliases)
