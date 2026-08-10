from __future__ import annotations

import pytest

from ragplan.core.errors import ErrorResponse
from ragplan.core.models import (
    BranchResult,
    Chunk,
    Entity,
    FrozenModel,
    GraphPath,
    IngestionManifest,
    ModelManifest,
    PlanDefinition,
    PlanEstimate,
    PlannerDecision,
    QueryAnalysis,
    QueryFeatures,
    Relation,
    RetrievalHit,
    SearchRequest,
    SearchResponse,
    SearchTrace,
)

pytestmark = [pytest.mark.unit, pytest.mark.contract]


def test_every_stage1_domain_contract_is_frozen_and_closed() -> None:
    models = (
        Chunk,
        Entity,
        Relation,
        RetrievalHit,
        GraphPath,
        QueryFeatures,
        QueryAnalysis,
        PlanDefinition,
        PlanEstimate,
        PlannerDecision,
        BranchResult,
        SearchRequest,
        SearchResponse,
        SearchTrace,
        IngestionManifest,
        ModelManifest,
    )

    assert all(issubclass(model, FrozenModel) for model in models)
    assert all(model.model_config["frozen"] is True for model in models)
    assert all(model.model_config["extra"] == "forbid" for model in models)


def test_public_request_schema_freezes_bounds_and_planner_enum() -> None:
    schema = SearchRequest.model_json_schema(mode="validation")
    properties = schema["properties"]
    planner_schema = properties["planner"]
    planner_reference = planner_schema["$ref"].split("/")[-1]

    assert properties["query"]["minLength"] == 1
    assert properties["query"]["maxLength"] == 4096
    assert properties["top_k"]["minimum"] == 1
    assert properties["top_k"]["maximum"] == 50
    assert properties["latency_budget_ms"]["minimum"] == 25
    assert properties["latency_budget_ms"]["maximum"] == 5000
    assert schema["$defs"][planner_reference]["enum"] == [
        "vector",
        "graph",
        "fixed_hybrid",
        "rule",
        "cost_aware",
    ]


def test_serialization_schemas_cannot_emit_query_embeddings_or_raw_query() -> None:
    feature_fields = QueryFeatures.model_json_schema(mode="serialization")["properties"]
    analysis_fields = QueryAnalysis.model_json_schema(mode="serialization")["properties"]
    trace_fields = SearchTrace.model_json_schema(mode="serialization")["properties"]

    assert "query_embedding" not in feature_fields
    assert "query_embedding" not in analysis_fields
    assert "query" not in trace_fields
    assert "normalized_query" not in trace_fields


def test_static_plan_and_public_error_shapes_are_stable() -> None:
    plan_fields = set(PlanDefinition.model_fields)
    error_fields = set(ErrorResponse.model_fields)

    assert {"timeout_ms", "expected_quality", "expected_latency_ms"}.isdisjoint(plan_fields)
    assert error_fields == {"code", "message", "request_id", "retryable"}
