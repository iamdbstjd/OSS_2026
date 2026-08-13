"""Deterministic dependency-rule relation extraction."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from typing import Final

from ragplan.core.models import EntityMention, Relation, RelationExtractionRule
from ragplan.ingestion.entities import ChunkExtraction, ParsedToken

_ACTIVE_SUBJECTS: Final = frozenset({"nsubj", "csubj"})
_PASSIVE_SUBJECTS: Final = frozenset({"nsubjpass", "nsubj:pass", "csubjpass"})
_DIRECT_OBJECTS: Final = frozenset({"dobj", "obj", "iobj", "dative", "oprd"})
_PUNCTUATION: Final = re.compile(r"[^\w\- ]+", flags=re.UNICODE)


def normalize_predicate(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    normalized = _PUNCTUATION.sub(" ", normalized)
    return " ".join(normalized.split())


def extract_relations(extraction: ChunkExtraction) -> tuple[Relation, ...]:
    """Apply active/passive/copular/appositional dependency patterns."""

    tokens = extraction.tokens
    children = _children_by_head(tokens)
    relations: list[Relation] = []
    for predicate in tokens:
        if predicate.pos not in {"VERB", "AUX"}:
            continue
        child_tokens = tuple(tokens[index] for index in children.get(predicate.index, ()))
        passive_subjects = _mentions_for_dependencies(
            extraction.mentions,
            child_tokens,
            _PASSIVE_SUBJECTS,
        )
        if passive_subjects:
            agents = _passive_agents(extraction.mentions, tokens, children, predicate.index)
            relations.extend(
                _relations_for_pairs(
                    agents,
                    passive_subjects,
                    predicate.lemma,
                    0.90,
                    RelationExtractionRule.PASSIVE,
                    extraction,
                )
            )
            continue

        subjects = _mentions_for_dependencies(
            extraction.mentions,
            child_tokens,
            _ACTIVE_SUBJECTS,
        )
        if not subjects:
            continue
        objects = _mentions_for_dependencies(
            extraction.mentions,
            child_tokens,
            _DIRECT_OBJECTS,
        )
        relations.extend(
            _relations_for_pairs(
                subjects,
                objects,
                predicate.lemma,
                0.90,
                RelationExtractionRule.DIRECT_SVO,
                extraction,
            )
        )
        for prep_index in children.get(predicate.index, ()):
            prep = tokens[prep_index]
            if prep.dependency not in {"prep", "agent"}:
                continue
            prep_objects = _mentions_in_subtree(
                extraction.mentions,
                children,
                prep_index,
            )
            relations.extend(
                _relations_for_pairs(
                    subjects,
                    prep_objects,
                    f"{predicate.lemma} {prep.lemma}",
                    0.80,
                    RelationExtractionRule.PREPOSITIONAL,
                    extraction,
                )
            )
        if predicate.lemma == "be":
            relations.extend(_copular_relations(extraction, predicate, children, subjects))

    relations.extend(_appositional_relations(extraction, children))
    unique = {
        (
            relation.source_entity_id,
            relation.target_entity_id,
            relation.predicate,
            relation.source_chunk_id,
            relation.extraction_rule,
        ): relation
        for relation in relations
        if relation.source_entity_id != relation.target_entity_id
    }
    return tuple(unique[key] for key in sorted(unique))


def _children_by_head(tokens: tuple[ParsedToken, ...]) -> dict[int, tuple[int, ...]]:
    mutable: dict[int, list[int]] = {}
    for token in tokens:
        if token.head_index == token.index:
            continue
        mutable.setdefault(token.head_index, []).append(token.index)
    return {key: tuple(sorted(value)) for key, value in mutable.items()}


def _mention_at_token(
    mentions: Iterable[EntityMention],
    token_index: int,
) -> EntityMention | None:
    candidates = [
        mention for mention in mentions if mention.token_start <= token_index < mention.token_end
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda item: (item.token_end - item.token_start, item.start_char))


def _mentions_for_dependencies(
    mentions: tuple[EntityMention, ...],
    tokens: tuple[ParsedToken, ...],
    dependencies: frozenset[str],
) -> tuple[EntityMention, ...]:
    found = {
        mention.id: mention
        for token in tokens
        if token.dependency in dependencies
        if (mention := _mention_at_token(mentions, token.index)) is not None
    }
    return tuple(found[key] for key in sorted(found))


def _descendants(children: dict[int, tuple[int, ...]], root: int) -> tuple[int, ...]:
    pending = list(children.get(root, ()))
    observed: set[int] = set()
    while pending:
        item = pending.pop()
        if item in observed:
            continue
        observed.add(item)
        pending.extend(children.get(item, ()))
    return tuple(sorted(observed))


def _mentions_in_subtree(
    mentions: tuple[EntityMention, ...],
    children: dict[int, tuple[int, ...]],
    root: int,
) -> tuple[EntityMention, ...]:
    indexes = (root, *_descendants(children, root))
    found = {
        mention.id: mention
        for index in indexes
        if (mention := _mention_at_token(mentions, index)) is not None
    }
    return tuple(found[key] for key in sorted(found))


def _passive_agents(
    mentions: tuple[EntityMention, ...],
    tokens: tuple[ParsedToken, ...],
    children: dict[int, tuple[int, ...]],
    predicate_index: int,
) -> tuple[EntityMention, ...]:
    found: dict[str, EntityMention] = {}
    for child_index in children.get(predicate_index, ()):
        child = tokens[child_index]
        if child.dependency == "agent" or (child.dependency == "prep" and child.lemma == "by"):
            for mention in _mentions_in_subtree(mentions, children, child_index):
                found[mention.id] = mention
    return tuple(found[key] for key in sorted(found))


def _relations_for_pairs(
    sources: tuple[EntityMention, ...],
    targets: tuple[EntityMention, ...],
    predicate: str,
    confidence: float,
    rule: RelationExtractionRule,
    extraction: ChunkExtraction,
) -> list[Relation]:
    normalized_predicate = normalize_predicate(predicate)
    if not normalized_predicate:
        return []
    return [
        Relation(
            source_entity_id=source.entity_id,
            target_entity_id=target.entity_id,
            predicate=normalized_predicate,
            confidence=confidence,
            source_chunk_id=extraction.chunk.canonical_chunk_id,
            extractor_version=extraction.extractor_version,
            extraction_rule=rule,
        )
        for source in sources
        for target in targets
        if source.entity_id != target.entity_id
        and source.sentence_start_char == target.sentence_start_char
    ]


def _copular_relations(
    extraction: ChunkExtraction,
    predicate: ParsedToken,
    children: dict[int, tuple[int, ...]],
    subjects: tuple[EntityMention, ...],
) -> list[Relation]:
    relations: list[Relation] = []
    for complement_index in children.get(predicate.index, ()):
        complement = extraction.tokens[complement_index]
        if complement.dependency not in {"attr", "acomp", "oprd"}:
            continue
        direct_target = _mention_at_token(extraction.mentions, complement.index)
        if direct_target is not None:
            relations.extend(
                _relations_for_pairs(
                    subjects,
                    (direct_target,),
                    "be",
                    0.85,
                    RelationExtractionRule.COPULAR,
                    extraction,
                )
            )
        for prep_index in children.get(complement.index, ()):
            prep = extraction.tokens[prep_index]
            if prep.dependency != "prep":
                continue
            targets = _mentions_in_subtree(extraction.mentions, children, prep_index)
            relations.extend(
                _relations_for_pairs(
                    subjects,
                    targets,
                    f"be {complement.lemma} {prep.lemma}",
                    0.85,
                    RelationExtractionRule.COPULAR,
                    extraction,
                )
            )
    return relations


def _appositional_relations(
    extraction: ChunkExtraction,
    children: dict[int, tuple[int, ...]],
) -> list[Relation]:
    relations: list[Relation] = []
    for token in extraction.tokens:
        if token.dependency != "appos":
            continue
        source = _mention_at_token(extraction.mentions, token.head_index)
        if source is None:
            continue
        direct_target = _mention_at_token(extraction.mentions, token.index)
        if direct_target is not None:
            relations.extend(
                _relations_for_pairs(
                    (source,),
                    (direct_target,),
                    "appos",
                    0.75,
                    RelationExtractionRule.APPOSITIONAL,
                    extraction,
                )
            )
        for prep_index in children.get(token.index, ()):
            prep = extraction.tokens[prep_index]
            if prep.dependency != "prep":
                continue
            targets = _mentions_in_subtree(extraction.mentions, children, prep_index)
            relations.extend(
                _relations_for_pairs(
                    (source,),
                    targets,
                    f"{token.lemma} {prep.lemma}",
                    0.80,
                    RelationExtractionRule.APPOSITIONAL,
                    extraction,
                )
            )
    return relations
