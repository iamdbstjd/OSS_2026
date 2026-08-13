"""Canonical, unambiguous identifiers for corpus records."""

from __future__ import annotations

import hashlib
import unicodedata
import uuid
from typing import Final
from urllib.parse import quote

ID_VERSION: Final = "v1"
CONTENT_HASH_PREFIX_LENGTH: Final = 16
ENTITY_NAMESPACE: Final = uuid.UUID("7c630d62-2a54-5e68-b828-b3d2370ed232")
QDRANT_NAMESPACE: Final = uuid.UUID("5947f0b1-6553-5713-83e0-f334f8aa2139")
MENTION_NAMESPACE: Final = uuid.UUID("e907af56-b387-56a4-9aba-09a70b80a312")
RELATION_NAMESPACE: Final = uuid.UUID("e7cd41ad-dbd1-5de3-bb72-f0a28e9889d4")
ALLOWED_ENTITY_TYPES: Final = frozenset(
    {"PERSON", "ORG", "GPE", "LOC", "FAC", "PRODUCT", "EVENT", "WORK_OF_ART"}
)


def _component(value: str) -> str:
    """Encode one ID component; delimiters therefore cannot be ambiguous."""
    if not isinstance(value, str):
        raise TypeError("ID components must be strings")
    normalized = unicodedata.normalize("NFKC", value).strip()
    if not normalized:
        raise ValueError("ID components must be non-empty strings")
    return quote(normalized, safe="-._~")


def content_hash_prefix(content: str) -> str:
    """Return the fixed SHA-256 prefix for already-normalized UTF-8 chunk text."""
    if not isinstance(content, str):
        raise TypeError("content must be a string")
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:CONTENT_HASH_PREFIX_LENGTH]


def canonical_document_id(source_dataset: str, source_document_id: str) -> str:
    """Create a versioned document identifier from two encoded components."""
    return f"{ID_VERSION}:document:{_component(source_dataset)}:{_component(source_document_id)}"


def canonical_chunk_id(document_id: str, chunk_index: int, content: str) -> str:
    """Create a chunk ID whose document boundary cannot be confused with its index."""
    if isinstance(chunk_index, bool) or not isinstance(chunk_index, int) or chunk_index < 0:
        raise ValueError("chunk_index must be a non-negative integer")
    if not document_id.startswith(f"{ID_VERSION}:document:"):
        raise ValueError("document_id must be a canonical versioned document ID")
    return (
        f"{ID_VERSION}:chunk:{_component(document_id)}:{chunk_index}:{content_hash_prefix(content)}"
    )


def normalize_entity_name(name: str) -> str:
    """Apply the version-1 ADR-008 exact entity normalization pipeline."""

    if not isinstance(name, str):
        raise TypeError("entity name must be a string")
    normalized = " ".join(unicodedata.normalize("NFKC", name).strip().split()).casefold()
    previous = None
    while normalized != previous:
        previous = normalized
        normalized = normalized.strip()
        start = 0
        end = len(normalized)
        while start < end and unicodedata.category(normalized[start]).startswith("P"):
            start += 1
        while end > start and unicodedata.category(normalized[end - 1]).startswith("P"):
            end -= 1
        normalized = normalized[start:end]
    if not normalized:
        raise ValueError("normalized entity name must not be empty")
    return normalized


def entity_id(entity_type: str, normalized_name: str) -> uuid.UUID:
    """Return the deterministic UUIDv5 entity ID."""

    normalized_type = unicodedata.normalize("NFKC", entity_type).strip().upper()
    if normalized_type not in ALLOWED_ENTITY_TYPES:
        raise ValueError(f"unsupported entity type: {entity_type!r}")
    canonical_name = normalize_entity_name(normalized_name)
    canonical_key = f"{_component(normalized_type)}:{_component(canonical_name)}"
    return uuid.uuid5(ENTITY_NAMESPACE, canonical_key)


def qdrant_point_id(canonical_chunk_id: str) -> uuid.UUID:
    """Return the deterministic UUIDv5 Qdrant point ID for a chunk."""
    if not isinstance(canonical_chunk_id, str):
        raise TypeError("canonical_chunk_id must be a string")
    if not canonical_chunk_id.startswith(f"{ID_VERSION}:chunk:"):
        raise ValueError("canonical_chunk_id must be a canonical versioned chunk ID")
    return uuid.uuid5(QDRANT_NAMESPACE, canonical_chunk_id)


def entity_mention_id(
    canonical_chunk_id: str,
    entity_uuid: str,
    start_char: int,
    end_char: int,
) -> uuid.UUID:
    """Return a stable UUIDv5 for one entity span in one canonical chunk."""

    if not canonical_chunk_id.startswith(f"{ID_VERSION}:chunk:"):
        raise ValueError("canonical_chunk_id must be a canonical versioned chunk ID")
    try:
        canonical_entity_uuid = str(uuid.UUID(entity_uuid))
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("entity_uuid must be a UUID string") from exc
    if (
        isinstance(start_char, bool)
        or isinstance(end_char, bool)
        or not isinstance(start_char, int)
        or not isinstance(end_char, int)
        or start_char < 0
        or end_char <= start_char
    ):
        raise ValueError("mention character offsets must form a non-empty span")
    key = f"{canonical_chunk_id}:{canonical_entity_uuid}:{start_char}:{end_char}"
    return uuid.uuid5(MENTION_NAMESPACE, key)


def relation_id(
    source_entity_uuid: str,
    target_entity_uuid: str,
    predicate: str,
    canonical_chunk_id: str,
    extraction_rule: str,
) -> uuid.UUID:
    """Return a stable UUIDv5 for one directed, provenance-bound relation."""

    try:
        source = str(uuid.UUID(source_entity_uuid))
        target = str(uuid.UUID(target_entity_uuid))
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("relation endpoints must be UUID strings") from exc
    if source == target:
        raise ValueError("relation endpoints must be distinct")
    if not canonical_chunk_id.startswith(f"{ID_VERSION}:chunk:"):
        raise ValueError("canonical_chunk_id must be a canonical versioned chunk ID")
    normalized_predicate = " ".join(unicodedata.normalize("NFKC", predicate).casefold().split())
    normalized_rule = " ".join(unicodedata.normalize("NFKC", extraction_rule).casefold().split())
    if not normalized_predicate or not normalized_rule:
        raise ValueError("predicate and extraction_rule must be non-empty")
    key = f"{source}:{target}:{normalized_predicate}:{canonical_chunk_id}:{normalized_rule}"
    return uuid.uuid5(RELATION_NAMESPACE, key)
