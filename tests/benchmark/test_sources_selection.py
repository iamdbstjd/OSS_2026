from __future__ import annotations

import json
from pathlib import Path

from ragplan.benchmark.contracts import SourceDataset
from ragplan.benchmark.selection import (
    PinnedNer,
    build_query_contract,
    query_template_hash,
    select_smallest_hashes,
)
from ragplan.benchmark.sources import (
    CandidateOutcome,
    NormalizedQuery,
    SourceParagraph,
    SupportingFact,
    iter_musique,
)
from ragplan.core.ids import canonical_document_id


class _Entity:
    def __init__(self, text: str, label: str) -> None:
        self.text = text
        self.label_ = label


class _Document:
    def __init__(self, entities: list[_Entity]) -> None:
        self.ents = entities


class _Nlp:
    def __call__(self, text: str) -> _Document:
        if "Ada Lovelace" in text:
            return _Document([_Entity("Ada Lovelace", "PERSON")])
        return _Document([])


def _query(source_id: str, question: str) -> NormalizedQuery:
    return NormalizedQuery(
        source_dataset=SourceDataset.NQ,
        source_query_id=source_id,
        question=question,
        answers=("answer",),
        paragraphs=(
            SourceParagraph(
                source_document_id=f"positive-{source_id}",
                title="Positive",
                text="The answer is present.",
                supporting_facts=(SupportingFact("fact", "The answer is present."),),
            ),
            SourceParagraph(
                source_document_id="shared-distractor",
                title="Distractor",
                text="Distractor evidence.",
            ),
        ),
    )


def test_hash_selection_is_input_order_independent_for_unique_records() -> None:
    candidates = tuple(
        CandidateOutcome(query=_query(str(index), f"Unique question {index}?"))
        for index in range(10)
    )
    forward, _ = select_smallest_hashes(
        candidates,
        source_dataset=SourceDataset.NQ,
        limit=4,
    )
    reverse, _ = select_smallest_hashes(
        reversed(candidates),
        source_dataset=SourceDataset.NQ,
        limit=4,
    )
    assert tuple(query.source_query_id for query in forward) == tuple(
        query.source_query_id for query in reverse
    )
    candidate_hashes = sorted(
        candidate.query.selection_hash for candidate in candidates if candidate.query is not None
    )
    assert tuple(query.selection_hash for query in forward) == tuple(candidate_hashes[:4])


def test_every_corpus_document_and_template_becomes_a_group_key() -> None:
    query = _query("source", "What did Ada Lovelace publish in 1843?")
    ner = PinnedNer(_Nlp())
    contract = build_query_contract(query, ner=ner)
    expected_documents = {
        f"document:{canonical_document_id('dpr_nq', paragraph.source_document_id)}"
        for paragraph in query.paragraphs
    }
    assert expected_documents <= set(contract.group_keys)
    assert "entity:PERSON:ada lovelace" in contract.group_keys
    assert sum(key.startswith("template:") for key in contract.group_keys) == 1
    assert query_template_hash(
        "What did Ada Lovelace publish in 1843?", ner=ner
    ) == query_template_hash("What did Ada Lovelace publish in 2020?", ner=ner)


def test_musique_requires_exactly_three_decomposition_steps(tmp_path: Path) -> None:
    path = tmp_path / "musique.jsonl"
    base = {
        "id": "query-id",
        "answerable": True,
        "question": "Which entity is reached?",
        "answer": "Gamma",
        "answer_aliases": [],
        "paragraphs": [
            {
                "idx": 0,
                "title": "Alpha",
                "paragraph_text": "The first answer is Alpha.",
            },
            {
                "idx": 1,
                "title": "Beta",
                "paragraph_text": "The second answer is Beta.",
            },
            {
                "idx": 2,
                "title": "Gamma",
                "paragraph_text": "The final answer is Gamma.",
            },
        ],
    }
    records = []
    for count in (2, 3, 4):
        record = dict(base)
        record["id"] = f"query-{count}"
        record["question_decomposition"] = [
            {"paragraph_support_idx": index % 3, "answer": ("Alpha", "Beta", "Gamma")[index % 3]}
            for index in range(count)
        ]
        records.append(record)
    path.write_text("".join(json.dumps(item) + "\n" for item in records), encoding="utf-8")
    outcomes = tuple(iter_musique(path))
    assert outcomes[0].exclusion_reason == "not_exactly_three_hop"
    assert outcomes[1].query is not None
    assert outcomes[1].query.source_dataset is SourceDataset.MUSIQUE
    assert outcomes[2].exclusion_reason == "not_exactly_three_hop"
