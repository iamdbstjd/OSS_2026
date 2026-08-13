from __future__ import annotations

from pathlib import Path

import pytest

from ragplan.core.ids import canonical_chunk_id, canonical_document_id
from ragplan.core.models import Chunk, RelationExtractionRule
from ragplan.ingestion.entities import EntityExtractor
from ragplan.ingestion.relations import extract_relations, normalize_predicate

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def extractor() -> EntityExtractor:
    return EntityExtractor.load_pinned(lockfile=Path("uv.lock"))


def _extract(text: str, extractor: EntityExtractor) -> tuple[object, ...]:
    document_id = canonical_document_id("fixture", text)
    chunk = Chunk(
        id=canonical_chunk_id(document_id, 0, text),
        document_id=document_id,
        corpus_version="fixture-v1",
        position=0,
        text=text,
        token_count=len(text.split()),
    )
    extraction = extractor.extract(chunk)
    return extraction, extract_relations(extraction)


@pytest.mark.parametrize(
    ("text", "predicate", "rule", "source_name", "target_name"),
    [
        (
            "Apple acquired Beats Electronics.",
            "acquire",
            RelationExtractionRule.DIRECT_SVO,
            "Apple",
            "Beats Electronics",
        ),
        (
            "Beats Electronics was acquired by Apple.",
            "acquire",
            RelationExtractionRule.PASSIVE,
            "Apple",
            "Beats Electronics",
        ),
        (
            "Paris is the capital of France.",
            "be capital of",
            RelationExtractionRule.COPULAR,
            "Paris",
            "France",
        ),
        (
            "Steve Jobs, founder of Apple, led Pixar.",
            "founder of",
            RelationExtractionRule.APPOSITIONAL,
            "Steve Jobs",
            "Apple",
        ),
        (
            "Ada Lovelace worked with Charles Babbage.",
            "work with",
            RelationExtractionRule.PREPOSITIONAL,
            "Ada Lovelace",
            "Charles Babbage",
        ),
    ],
)
def test_dependency_relation_fixtures(
    text: str,
    predicate: str,
    rule: RelationExtractionRule,
    source_name: str,
    target_name: str,
    extractor: EntityExtractor,
) -> None:
    extraction, relations = _extract(text, extractor)
    mentions = {item.raw_text: item.entity_id for item in extraction.mentions}
    relation = next(item for item in relations if item.predicate == predicate)

    assert relation.source_entity_id == mentions[source_name]
    assert relation.target_entity_id == mentions[target_name]
    assert relation.extraction_rule is rule
    assert relation.confidence >= 0.70
    assert relation.source_chunk_id == extraction.chunk.id
    assert relation.extractor_version == extraction.extractor_version


def test_unsupported_sentence_does_not_create_false_relation(
    extractor: EntityExtractor,
) -> None:
    _, relations = _extract("A quiet blue room near the old bridge.", extractor)

    assert relations == ()


def test_predicate_normalization_is_stable() -> None:
    assert normalize_predicate("  Founded,   By! ") == "founded by"
