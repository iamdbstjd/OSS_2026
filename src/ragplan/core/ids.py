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
    start = 0
    end = len(normalized)
    while start < end and unicodedata.category(normalized[start]).startswith("P"):
        start += 1
    while end > start and unicodedata.category(normalized[end - 1]).startswith("P"):
        end -= 1
    normalized = normalized[start:end].strip()
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
