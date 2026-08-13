"""Pinned spaCy entity extraction with immutable source provenance."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from ragplan.core.errors import ErrorCode, RAGPlanError
from ragplan.core.ids import (
    ALLOWED_ENTITY_TYPES,
    entity_id,
    entity_mention_id,
    normalize_entity_name,
)
from ragplan.core.models import Chunk, EntityMention, EntityType
from ragplan.ingestion.extractor_version import (
    SPACY_MODEL_NAME,
    SPACY_MODEL_VERSION,
    build_extractor_version,
)


@dataclass(frozen=True, slots=True)
class ParsedToken:
    index: int
    text: str
    lemma: str
    pos: str
    dependency: str
    head_index: int
    sentence_start_char: int
    sentence_end_char: int


@dataclass(frozen=True, slots=True)
class ChunkExtraction:
    chunk: Chunk
    tokens: tuple[ParsedToken, ...]
    mentions: tuple[EntityMention, ...]
    extractor_version: str


Pipeline = Callable[[str], Any]
_MODEL_ERROR: Final = "pinned spaCy graph extractor could not be loaded"


class EntityExtractor:
    """Filter NER spans to ADR-008 types and preserve exact offsets."""

    def __init__(self, pipeline: Pipeline, extractor_version: str) -> None:
        if not extractor_version.strip():
            raise ValueError("extractor_version must not be blank")
        self._pipeline = pipeline
        self.extractor_version = extractor_version

    @classmethod
    def load_pinned(
        cls,
        *,
        lockfile: Path | None,
        benchmark_mode: bool = True,
    ) -> EntityExtractor:
        version_value = build_extractor_version(lockfile, benchmark_mode=benchmark_mode)
        try:
            import spacy

            pipeline = spacy.load(
                SPACY_MODEL_NAME,
                disable=(),
                exclude=(),
                config={"nlp": {"batch_size": 64}},
            )
        except Exception as exc:
            raise RAGPlanError(ErrorCode.MODEL_INCOMPATIBLE, _MODEL_ERROR) from exc
        model_version = pipeline.meta.get("version")
        if model_version != SPACY_MODEL_VERSION or not {"ner", "parser"} <= set(
            pipeline.pipe_names
        ):
            raise RAGPlanError(ErrorCode.MODEL_INCOMPATIBLE, _MODEL_ERROR)
        return cls(pipeline, version_value)

    def extract(self, chunk: Chunk) -> ChunkExtraction:
        """Extract one chunk; no-entity and unsupported-label chunks are valid empties."""

        document = self._pipeline(chunk.text)
        return self._from_document(chunk, document)

    def extract_many(self, chunks: tuple[Chunk, ...]) -> tuple[ChunkExtraction, ...]:
        pipe = getattr(self._pipeline, "pipe", None)
        if not callable(pipe):
            return tuple(self.extract(chunk) for chunk in chunks)
        documents = pipe((chunk.text for chunk in chunks), batch_size=64)
        return tuple(
            self._from_document(chunk, document)
            for chunk, document in zip(chunks, documents, strict=True)
        )

    def _from_document(self, chunk: Chunk, document: Any) -> ChunkExtraction:
        tokens = tuple(
            ParsedToken(
                index=int(token.i),
                text=str(token.text),
                lemma=str(token.lemma_).casefold(),
                pos=str(token.pos_),
                dependency=str(token.dep_).casefold(),
                head_index=int(token.head.i),
                sentence_start_char=int(token.sent.start_char),
                sentence_end_char=int(token.sent.end_char),
            )
            for token in document
        )
        mentions: list[EntityMention] = []
        for span in document.ents:
            label = str(span.label_)
            if label not in ALLOWED_ENTITY_TYPES:
                continue
            raw_text = str(span.text)
            try:
                normalized_name = normalize_entity_name(raw_text)
            except ValueError:
                continue
            entity_uuid = str(entity_id(label, normalized_name))
            start_char = int(span.start_char)
            end_char = int(span.end_char)
            sentence = span.sent
            mentions.append(
                EntityMention(
                    id=str(
                        entity_mention_id(
                            chunk.canonical_chunk_id,
                            entity_uuid,
                            start_char,
                            end_char,
                        )
                    ),
                    entity_id=entity_uuid,
                    entity_type=EntityType(label),
                    raw_text=raw_text,
                    normalized_name=normalized_name,
                    source_chunk_id=chunk.canonical_chunk_id,
                    start_char=start_char,
                    end_char=end_char,
                    sentence_start_char=int(sentence.start_char),
                    sentence_end_char=int(sentence.end_char),
                    token_start=int(span.start),
                    token_end=int(span.end),
                    root_token=int(span.root.i),
                )
            )
        return ChunkExtraction(
            chunk=chunk,
            tokens=tokens,
            mentions=tuple(sorted(mentions, key=lambda item: (item.start_char, item.end_char))),
            extractor_version=self.extractor_version,
        )
