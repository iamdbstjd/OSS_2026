import uuid

import pytest

from ragplan.core.ids import (
    CONTENT_HASH_PREFIX_LENGTH,
    canonical_chunk_id,
    canonical_document_id,
    content_hash_prefix,
    entity_id,
    normalize_entity_name,
    qdrant_point_id,
)

pytestmark = pytest.mark.unit


def test_document_components_are_versioned_and_unambiguous() -> None:
    assert canonical_document_id("a:b", "c") != canonical_document_id("a", "b:c")
    assert canonical_document_id("a/b", "c") == "v1:document:a%2Fb:c"
    assert canonical_document_id(" dataset ", "doc") == canonical_document_id("dataset", "doc")


def test_chunk_id_changes_with_content_and_keeps_document_as_one_component() -> None:
    document_id = canonical_document_id("dataset", "document:1")
    first = canonical_chunk_id(document_id, 0, "first")
    assert first != canonical_chunk_id(document_id, 0, "second")
    assert first.startswith("v1:chunk:v1%3Adocument%3Adataset%3Adocument%253A1:0:")
    assert len(content_hash_prefix("first")) == CONTENT_HASH_PREFIX_LENGTH
    with pytest.raises(ValueError, match="canonical"):
        canonical_chunk_id("plain-document-id", 0, "text")


def test_uuid_ids_are_deterministic_and_version_namespace_scoped() -> None:
    assert entity_id("PERSON", "alice") == entity_id("PERSON", "alice")
    assert entity_id("PERSON", "alice") != entity_id("ORG", "alice")
    point_id = qdrant_point_id("v1:chunk:doc:0:deadbeef")
    assert isinstance(point_id, uuid.UUID)
    assert point_id == qdrant_point_id("v1:chunk:doc:0:deadbeef")
    with pytest.raises(ValueError):
        canonical_chunk_id("doc", -1, "text")
    with pytest.raises(ValueError, match="canonical"):
        qdrant_point_id("plain-chunk-id")


def test_entity_normalization_and_uuid_are_canonical() -> None:
    assert normalize_entity_name("  ‘Ａｌｉｃｅ   Smith!’  ") == "alice smith"
    assert entity_id("PERSON", "Alice") == entity_id("person", "alice")
    with pytest.raises(ValueError, match="unsupported"):
        entity_id("DATE", "2026")


def test_entity_normalization_is_idempotent_with_spaced_surrounding_punctuation() -> None:
    normalized = normalize_entity_name(" ( P. O. W. ) ")

    assert normalized == "p. o. w"
    assert normalize_entity_name(normalized) == normalized
