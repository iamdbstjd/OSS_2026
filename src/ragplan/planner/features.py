"""Deterministic, versioned query feature extraction for the rule planner."""

from __future__ import annotations

import hashlib
import re
from importlib.resources import files
from pathlib import Path
from typing import Final, Literal, Self

from pydantic import Field, model_validator

from ragplan.core.models import FrozenModel, QueryFeatures
from ragplan.planner.catalog import canonical_json

FEATURE_SCHEMA_VERSION: Final[Literal["qf_v1"]] = "qf_v1"
DEFAULT_FEATURE_CONFIG_PATH: Final = (
    Path(__file__).resolve().parents[3] / "configs" / "query_features_v1.json"
)


class QueryFeatureConfig(FrozenModel):
    """Frozen keyword/regex contract; changes require a schema version bump."""

    schema_version: Literal["qf_v1"] = FEATURE_SCHEMA_VERSION
    supported_language: Literal["en"] = "en"
    embedding_feature_enabled: Literal[False] = False
    relation_patterns: tuple[str, ...] = Field(min_length=1)
    multi_hop_patterns: tuple[str, ...] = Field(min_length=1)
    comparison_patterns: tuple[str, ...] = Field(min_length=1)
    aggregation_patterns: tuple[str, ...] = Field(min_length=1)
    global_patterns: tuple[str, ...] = Field(min_length=1)
    per_pattern_increment: float = Field(gt=0.0, le=1.0)
    two_entity_relation_bonus: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _valid_patterns(self) -> Self:
        groups = (
            self.relation_patterns,
            self.multi_hop_patterns,
            self.comparison_patterns,
            self.aggregation_patterns,
            self.global_patterns,
        )
        if any(len(group) != len(set(group)) for group in groups):
            raise ValueError("feature pattern groups cannot contain duplicates")
        try:
            for pattern in (pattern for group in groups for pattern in group):
                re.compile(pattern, flags=re.IGNORECASE)
        except re.error as exc:
            raise ValueError("feature config contains an invalid regular expression") from exc
        return self

    @property
    def sha256(self) -> str:
        payload = canonical_json(self.model_dump(mode="json"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_query_feature_config(path: Path) -> QueryFeatureConfig:
    try:
        return QueryFeatureConfig.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError("query feature config is missing or invalid") from exc


def load_default_query_feature_config() -> QueryFeatureConfig:
    if DEFAULT_FEATURE_CONFIG_PATH.is_file():
        return load_query_feature_config(DEFAULT_FEATURE_CONFIG_PATH)
    payload = (
        files("ragplan.resources").joinpath("query_features_v1.json").read_text(encoding="utf-8")
    )
    try:
        return QueryFeatureConfig.model_validate_json(payload)
    except Exception as exc:
        raise ValueError("packaged query feature config is invalid") from exc


def extract_query_features(
    normalized_query: str,
    *,
    token_count: int,
    entity_count: int,
    final_top_k: int,
    config: QueryFeatureConfig,
) -> QueryFeatures:
    """Map one normalized English query to the exact ``qf_v1`` numeric schema."""

    def signal(patterns: tuple[str, ...]) -> float:
        matches = sum(
            1
            for pattern in patterns
            if re.search(pattern, normalized_query, flags=re.IGNORECASE) is not None
        )
        return min(1.0, matches * config.per_pattern_increment)

    relation_signal = signal(config.relation_patterns)
    if entity_count >= 2:
        relation_signal = min(1.0, relation_signal + config.two_entity_relation_bonus)
    return QueryFeatures(
        token_count=token_count,
        entity_count=entity_count,
        entity_density=min(1.0, entity_count / max(1, token_count)),
        relation_signal=relation_signal,
        multi_hop_signal=signal(config.multi_hop_patterns),
        comparison_signal=signal(config.comparison_patterns),
        aggregation_signal=signal(config.aggregation_patterns),
        global_signal=signal(config.global_patterns),
        final_top_k=final_top_k,
    )
