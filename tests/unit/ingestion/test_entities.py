from __future__ import annotations

from pathlib import Path

import pytest

from ragplan.core.errors import ErrorCode, RAGPlanError
from ragplan.core.ids import (
    canonical_chunk_id,
    canonical_document_id,
    entity_id,
    entity_mention_id,
)
from ragplan.core.models import Chunk, EntityMention, EntityType
from ragplan.ingestion.entities import EntityExtractor
from ragplan.ingestion.extractor_version import build_extractor_version
from ragplan.ingestion.resolver import resolve_entities

pytestmark = pytest.mark.unit


def _chunk(text: str, *, index: int = 0) -> Chunk:
    document_id = canonical_document_id("fixture", f"document-{index}")
    return Chunk(
        id=canonical_chunk_id(document_id, 0, text),
        document_id=document_id,
        corpus_version="fixture-v1",
        position=0,
        text=text,
        token_count=max(1, len(text.split())),
    )


@pytest.fixture(scope="module")
def extractor() -> EntityExtractor:
    return EntityExtractor.load_pinned(lockfile=Path("uv.lock"))


def test_pinned_entity_extraction_preserves_raw_and_sentence_spans(
    extractor: EntityExtractor,
) -> None:
    chunk = _chunk("Apple acquired Beats Electronics. Paris is in France.")

    extraction = extractor.extract(chunk)

    assert [(item.raw_text, item.entity_type) for item in extraction.mentions] == [
        ("Apple", EntityType.ORG),
        ("Beats Electronics", EntityType.ORG),
        ("Paris", EntityType.GPE),
        ("France", EntityType.GPE),
    ]
    assert all(item.source_chunk_id == chunk.id for item in extraction.mentions)
    assert all(
        chunk.text[item.start_char : item.end_char] == item.raw_text for item in extraction.mentions
    )
    assert all(
        item.sentence_start_char <= item.start_char < item.end_char <= item.sentence_end_char
        for item in extraction.mentions
    )


def test_no_entity_and_unsupported_labels_are_safe(extractor: EntityExtractor) -> None:
    extraction = extractor.extract(_chunk("The quick brown fox jumps over a small log.", index=1))

    assert extraction.mentions == ()


def test_exact_resolver_normalizes_aliases_but_separates_types() -> None:
    chunk = _chunk("Apple and APPLE", index=2)

    def mention(raw: str, entity_type: EntityType, start: int, end: int) -> EntityMention:
        normalized = "apple"
        entity_uuid = str(entity_id(entity_type.value, normalized))
        return EntityMention(
            id=str(entity_mention_id(chunk.id, entity_uuid, start, end)),
            entity_id=entity_uuid,
            entity_type=entity_type,
            raw_text=raw,
            normalized_name=normalized,
            source_chunk_id=chunk.id,
            start_char=start,
            end_char=end,
            sentence_start_char=0,
            sentence_end_char=len(chunk.text),
            token_start=0 if start == 0 else 2,
            token_end=1 if start == 0 else 3,
            root_token=0 if start == 0 else 2,
        )

    entities = resolve_entities(
        (
            mention("Apple", EntityType.ORG, 0, 5),
            mention("APPLE", EntityType.ORG, 10, 15),
            mention("Apple", EntityType.PRODUCT, 0, 5),
        )
    )

    assert len(entities) == 2
    assert {item.entity_type for item in entities} == {EntityType.ORG, EntityType.PRODUCT}
    organization = next(item for item in entities if item.entity_type is EntityType.ORG)
    assert organization.aliases == ("Apple", "APPLE")


def test_benchmark_extractor_refuses_missing_lockfile() -> None:
    with pytest.raises(RAGPlanError) as caught:
        build_extractor_version(None, benchmark_mode=True)

    assert caught.value.code is ErrorCode.MODEL_INCOMPATIBLE
