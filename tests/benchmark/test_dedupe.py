from __future__ import annotations

from ragplan.benchmark.contracts import SourceDataset
from ragplan.benchmark.dedupe import (
    deduplicate_exact_documents,
    deduplicate_exact_documents_with_report,
)
from ragplan.benchmark.sources import NormalizedQuery, SourceParagraph, SupportingFact


def test_exact_documents_converge_without_losing_supporting_facts() -> None:
    query = NormalizedQuery(
        source_dataset=SourceDataset.NQ,
        source_query_id="source",
        question="Question?",
        answers=("answer",),
        paragraphs=(
            SourceParagraph(
                source_document_id="z",
                title="Title",
                text="Same normalized text.",
                supporting_facts=(SupportingFact("fact-z", "Same normalized text."),),
            ),
            SourceParagraph(
                source_document_id="a",
                title="Title",
                text="Same normalized text.",
                supporting_facts=(SupportingFact("fact-a", "Same normalized text."),),
            ),
        ),
    )
    result = deduplicate_exact_documents((query,))
    assert len(result[0].paragraphs) == 1
    assert result[0].paragraphs[0].source_document_id == "a"
    assert {fact.fact_id for fact in result[0].paragraphs[0].supporting_facts} == {
        "fact-a",
        "fact-z",
    }


def test_near_duplicate_text_is_not_merged() -> None:
    query = NormalizedQuery(
        source_dataset=SourceDataset.NQ,
        source_query_id="source",
        question="Question?",
        answers=("answer",),
        paragraphs=(
            SourceParagraph("a", "Title", "Alpha beta gamma."),
            SourceParagraph("b", "Title", "Alpha beta gamma delta."),
        ),
    )
    assert len(deduplicate_exact_documents((query,))[0].paragraphs) == 2


def test_exact_aliases_are_preserved_for_attribution_audit() -> None:
    query = NormalizedQuery(
        source_dataset=SourceDataset.NQ,
        source_query_id="source",
        question="Question?",
        answers=("answer",),
        paragraphs=(
            SourceParagraph("z", "Title Z", "Same text."),
            SourceParagraph("a", "Title A", "Same text."),
        ),
    )
    result = deduplicate_exact_documents_with_report((query,))
    aliases = next(iter(result.aliases.values()))
    assert aliases == (
        "v1:document:dpr_nq:a",
        "v1:document:dpr_nq:z",
    )
