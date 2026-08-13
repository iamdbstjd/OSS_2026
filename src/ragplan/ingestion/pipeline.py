"""Deterministic Stage 4 graph extraction orchestration."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass

from ragplan.core.models import Chunk, Entity, EntityMention, Relation
from ragplan.ingestion.entities import ChunkExtraction, EntityExtractor
from ragplan.ingestion.relations import extract_relations
from ragplan.ingestion.resolver import resolve_entities


@dataclass(frozen=True, slots=True)
class GraphExtractionResult:
    chunks: tuple[Chunk, ...]
    entities: tuple[Entity, ...]
    mentions: tuple[EntityMention, ...]
    relations: tuple[Relation, ...]
    extractor_version: str


def extract_graph(
    chunks: Sequence[Chunk],
    extractor: EntityExtractor,
) -> GraphExtractionResult:
    """Run NER, exact resolution, and relation rules over the exact vector chunks."""

    immutable_chunks = tuple(chunks)
    chunk_ids = tuple(chunk.canonical_chunk_id for chunk in immutable_chunks)
    if len(set(chunk_ids)) != len(chunk_ids):
        raise ValueError("graph extraction chunks must have unique canonical IDs")
    corpus_versions = {chunk.corpus_version for chunk in immutable_chunks}
    if len(corpus_versions) > 1:
        raise ValueError("one graph extraction run must contain one corpus version")
    extractions: tuple[ChunkExtraction, ...] = extractor.extract_many(immutable_chunks)
    mentions = tuple(
        sorted(
            (mention for extraction in extractions for mention in extraction.mentions),
            key=lambda item: item.id,
        )
    )
    entities = resolve_entities(mentions)
    relations = tuple(
        sorted(
            (relation for extraction in extractions for relation in extract_relations(extraction)),
            key=lambda item: (
                item.source_chunk_id,
                item.source_entity_id,
                item.target_entity_id,
                item.predicate,
                item.extraction_rule,
            ),
        )
    )
    return GraphExtractionResult(
        chunks=immutable_chunks,
        entities=entities,
        mentions=mentions,
        relations=relations,
        extractor_version=extractor.extractor_version,
    )


def graph_content_checksum(
    chunks: Sequence[Chunk],
    entities: Sequence[Entity],
    mentions: Sequence[EntityMention],
    relations: Sequence[Relation],
    *,
    extractor_version: str,
) -> str:
    """Hash all graph inputs so a corpus version cannot be silently reinterpreted."""

    payload = {
        "schema_version": "graph-content-v1",
        "extractor_version": extractor_version,
        "chunks": [item.model_dump(mode="json") for item in sorted(chunks, key=lambda x: x.id)],
        "entities": [item.model_dump(mode="json") for item in sorted(entities, key=lambda x: x.id)],
        "mentions": [item.model_dump(mode="json") for item in sorted(mentions, key=lambda x: x.id)],
        "relations": [
            item.model_dump(mode="json")
            for item in sorted(
                relations,
                key=lambda x: (
                    x.source_chunk_id,
                    x.source_entity_id,
                    x.target_entity_id,
                    x.predicate,
                    x.extraction_rule,
                ),
            )
        ],
    }
    serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
