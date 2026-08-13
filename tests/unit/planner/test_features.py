from __future__ import annotations

import json

import pytest

from ragplan.planner.features import (
    FEATURE_SCHEMA_VERSION,
    extract_query_features,
    load_default_query_feature_config,
)

pytestmark = pytest.mark.unit


def test_qf_v1_has_exact_stable_schema() -> None:
    features = extract_query_features(
        "Compare the founder of the company that acquired Acme with Ada Lovelace",
        token_count=12,
        entity_count=3,
        final_top_k=10,
        config=load_default_query_feature_config(),
    )

    assert FEATURE_SCHEMA_VERSION == "qf_v1"
    assert tuple(features.model_dump()) == (
        "token_count",
        "entity_count",
        "entity_density",
        "relation_signal",
        "multi_hop_signal",
        "comparison_signal",
        "aggregation_signal",
        "global_signal",
        "final_top_k",
    )
    assert features.entity_density == 0.25
    assert features.comparison_signal == 0.5
    assert features.multi_hop_signal == 0.5
    assert features.relation_signal == 1.0


def test_feature_config_identity_and_output_are_deterministic() -> None:
    config = load_default_query_feature_config()
    first = extract_query_features(
        "How many companies were founded by Ada?",
        token_count=8,
        entity_count=1,
        final_top_k=5,
        config=config,
    )
    second = extract_query_features(
        "How many companies were founded by Ada?",
        token_count=8,
        entity_count=1,
        final_top_k=5,
        config=config,
    )

    assert first == second
    assert first.aggregation_signal == 0.5
    assert first.relation_signal == 1.0
    assert len(config.sha256) == 64
    assert config.supported_language == "en"
    assert config.embedding_feature_enabled is False
    assert json.loads(config.model_dump_json())["schema_version"] == "qf_v1"


def test_prd_relation_and_chained_attribute_phrases_are_detected() -> None:
    config = load_default_query_feature_config()
    works_at = extract_query_features(
        "Who works at Acme?",
        token_count=4,
        entity_count=2,
        final_top_k=10,
        config=config,
    )
    chained = extract_query_features(
        "Where did the founder of Acme study?",
        token_count=7,
        entity_count=1,
        final_top_k=10,
        config=config,
    )

    assert works_at.relation_signal >= 0.5
    assert chained.relation_signal >= 0.5
    assert chained.multi_hop_signal >= 0.5
