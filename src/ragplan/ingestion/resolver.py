"""P0 exact-normalized entity resolution."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

from ragplan.core.models import Entity, EntityMention


def resolve_entities(mentions: Iterable[EntityMention]) -> tuple[Entity, ...]:
    """Collapse only exact ``(entity_type, normalized_name)`` aliases."""

    grouped: dict[str, list[EntityMention]] = defaultdict(list)
    for mention in mentions:
        grouped[mention.entity_id].append(mention)
    entities: list[Entity] = []
    for entity_uuid, group in grouped.items():
        first = group[0]
        if any(
            mention.entity_type is not first.entity_type
            or mention.normalized_name != first.normalized_name
            for mention in group
        ):
            raise ValueError("one entity UUID cannot contain conflicting normalized identities")
        aliases = tuple(
            sorted(
                {mention.raw_text for mention in group},
                key=lambda value: (
                    value.casefold(),
                    sum(character.isupper() for character in value),
                    value,
                ),
            )
        )
        entities.append(
            Entity(
                id=entity_uuid,
                name=aliases[0],
                entity_type=first.entity_type,
                normalized_name=first.normalized_name,
                aliases=aliases,
            )
        )
    return tuple(sorted(entities, key=lambda item: item.id))
