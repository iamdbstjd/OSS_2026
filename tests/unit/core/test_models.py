from __future__ import annotations

from collections.abc import Mapping

import pytest
from pydantic import ValidationError

from ragplan.core.errors import ErrorCode
from ragplan.core.models import (
    ActivationStatus,
    BranchKind,
    BranchResult,
    BranchStatus,
    Chunk,
    GraphPath,
    GraphSeedMatch,
    IngestionManifest,
    IngestionStoreStatus,
    PlanDefinition,
    PlannerDecision,
    PlannerMode,
    QueryAnalysis,
    QueryFeatures,
    Relation,
    RetrievalHit,
    SearchRequest,
    SearchResponse,
    SearchStatus,
    SearchTrace,
)

pytestmark = pytest.mark.unit


def features() -> QueryFeatures:
    return QueryFeatures(
        token_count=2,
        entity_count=0,
        entity_density=0.0,
        relation_signal=0.0,
        multi_hop_signal=0.0,
        comparison_signal=0.0,
        aggregation_signal=0.0,
        global_signal=0.0,
        final_top_k=10,
    )


def vector_plan() -> PlanDefinition:
    return PlanDefinition(
        id="P0",
        name="VECTOR_FAST",
        vector_enabled=True,
        graph_enabled=False,
        vector_top_k=10,
        graph_top_k=0,
        graph_depth=0,
        vector_weight=1.0,
        graph_weight=0.0,
        rerank_enabled=False,
        enabled_in_p0=True,
    )


def planner_decision() -> PlannerDecision:
    plan = vector_plan()
    return PlannerDecision(
        mode=PlannerMode.RULE,
        selected_plan_id=plan.id,
        selected_plan=plan,
        executed_vector_top_k=10,
        remaining_budget_ms=199.0,
        feature_version="v1",
        config_version="v1",
    )


def test_request_trims_query_and_enforces_cost_aware_top_k() -> None:
    assert SearchRequest(query="  a query  ").query == "a query"
    assert len(SearchRequest(query="x" * 4096).query) == 4096
    with pytest.raises(ValidationError):
        SearchRequest(query=" \t ")
    with pytest.raises(ValidationError):
        SearchRequest(query="x" * 4097)
    with pytest.raises(ValidationError):
        SearchRequest(query="query", planner=PlannerMode.COST_AWARE, top_k=9)
    with pytest.raises(ValidationError):
        SearchRequest.model_validate_json('{"query":"query","planner":"adaptive"}')


@pytest.mark.parametrize(
    "payload",
    [
        {"query": "query", "top_k": 0},
        {"query": "query", "top_k": 51},
        {"query": "query", "latency_budget_ms": 24},
        {"query": "query", "latency_budget_ms": 5001},
        {"query": "query", "top_k": True},
        {"query": "query", "unknown": "field"},
    ],
)
def test_request_boundaries_and_unknown_fields(payload: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        SearchRequest(**payload)  # type: ignore[arg-type]


def test_plan_definition_enforces_static_branch_invariants() -> None:
    plan = vector_plan()

    with pytest.raises(ValidationError):
        plan.model_copy(update={"graph_enabled": True})
    with pytest.raises(ValidationError):
        PlanDefinition(**{**plan.model_dump(), "graph_depth": 1})
    with pytest.raises(ValidationError):
        PlanDefinition(**{**plan.model_dump(), "rerank_enabled": True, "rerank_top_k": 11})
    assert {"timeout_ms", "expected_quality", "expected_latency_ms"}.isdisjoint(
        PlanDefinition.model_fields
    )


@pytest.mark.parametrize(
    "updates",
    [
        {"vector_enabled": False, "vector_top_k": 0, "vector_weight": 0.0},
        {"vector_enabled": False},
        {"vector_top_k": 0},
        {"graph_top_k": 1},
        {
            "graph_enabled": True,
            "graph_top_k": 1,
            "graph_depth": 0,
            "vector_weight": 0.5,
            "graph_weight": 0.5,
        },
        {"vector_weight": 0.9},
        {"rerank_top_k": 1},
        {"rerank_enabled": True, "rerank_top_k": 0},
        {"rerank_enabled": True, "rerank_top_k": 11},
    ],
)
def test_invalid_plan_invariant_table(updates: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        vector_plan().model_copy(update=updates)


def test_models_are_frozen_and_closed() -> None:
    request = SearchRequest(query="query")
    with pytest.raises(ValidationError):
        request.top_k = 3  # type: ignore[misc]
    with pytest.raises(ValidationError):
        SearchRequest(query="query", unexpected=True)


def test_mapping_fields_are_defensively_copied_and_deeply_frozen() -> None:
    source_metadata = {"source": {"labels": ["original"]}}
    chunk = Chunk(
        id="v1:chunk:fixture:0:abc",
        document_id="v1:document:fixture:1",
        corpus_version="corpus-v1",
        position=0,
        text="evidence",
        token_count=1,
        metadata=source_metadata,
    )
    source_metadata["source"] = {"labels": ["mutated"]}

    assert chunk.model_dump()["metadata"] == {"source": {"labels": ["original"]}}
    with pytest.raises(TypeError):
        chunk.metadata["source"] = "mutated"  # type: ignore[index]
    nested = chunk.metadata["source"]
    assert isinstance(nested, Mapping)
    with pytest.raises(TypeError):
        nested["labels"] = ()  # type: ignore[index]
    assert chunk.model_dump_json().endswith('"metadata":{"source":{"labels":["original"]}}}')
    shallow_copy = chunk.model_copy()
    deep_copy = chunk.model_copy(deep=True)
    assert shallow_copy.model_dump() == chunk.model_dump()
    assert deep_copy.model_dump() == chunk.model_dump()
    assert shallow_copy.metadata is not chunk.metadata
    assert deep_copy.metadata is not chunk.metadata
    chunk_with_default = Chunk(
        id="v1:chunk:fixture:1:def",
        document_id="v1:document:fixture:1",
        corpus_version="corpus-v1",
        position=1,
        text="more evidence",
        token_count=2,
    )
    with pytest.raises(TypeError):
        chunk_with_default.metadata["source"] = "mutated"  # type: ignore[index]


def test_embedding_is_excluded_from_analysis_and_trace_serialization() -> None:
    analysis = QueryAnalysis(
        normalized_query="query",
        language_supported=True,
        token_count=1,
        query_embedding=(0.1, 0.2),
        features=features(),
        analyzer_version="v1",
        analysis_latency_ms=1.0,
    )
    assert "query_embedding" not in analysis.model_dump()
    assert analysis.model_copy().query_embedding == (0.1, 0.2)

    trace = SearchTrace(
        request_id="request-1",
        query_hash="a" * 64,
        query_length=5,
        language_supported=True,
        features=features(),
        planner_decision=planner_decision(),
        branch_results=(
            BranchResult(
                branch=BranchKind.VECTOR,
                status=BranchStatus.SUCCEEDED,
                latency_ms=1.0,
            ),
        ),
        analyzer_latency_ms=0.5,
        planner_latency_ms=0.5,
        embedding_latency_ms=0.25,
        vector_latency_ms=1.0,
        total_latency_ms=2.0,
        latency_budget_ms=200,
        finalization_reserve_ms=10.0,
        budget_feasible=True,
        budget_violated=False,
        fallback=False,
        result_count=0,
        corpus_version="v1",
        config_version="v1",
    )
    serialized = trace.model_dump_json()
    assert "query_embedding" not in serialized
    assert "normalized_query" not in serialized
    assert 'query"' not in serialized
    assert '"embedding_latency_ms":0.25' in serialized
    assert '"vector_latency_ms":1.0' in serialized
    with pytest.raises(ValidationError):
        SearchTrace(**{**trace.model_dump(), "query": "sensitive input"})


def test_trace_branch_serialization_excludes_hit_text_and_metadata() -> None:
    hit = RetrievalHit(
        canonical_chunk_id="v1:chunk:fixture:0:abc",
        text="CONFIDENTIAL FULL DOCUMENT TEXT",
        score=1.0,
        source="vector",
        metadata={"credential": "secret"},
    )
    branch = BranchResult(
        branch=BranchKind.VECTOR,
        status=BranchStatus.SUCCEEDED,
        latency_ms=1.0,
        hits=(hit,),
    )

    assert branch.hit_count == 1
    trace = SearchTrace(
        request_id="request-private",
        query_hash="c" * 64,
        query_length=5,
        language_supported=True,
        features=features(),
        planner_decision=planner_decision(),
        branch_results=(branch,),
        analyzer_latency_ms=0.5,
        planner_latency_ms=0.5,
        vector_latency_ms=1.0,
        total_latency_ms=2.0,
        latency_budget_ms=200,
        finalization_reserve_ms=10.0,
        budget_feasible=True,
        budget_violated=False,
        fallback=False,
        result_count=1,
        corpus_version="v1",
        config_version="v1",
    )
    serialized = trace.model_dump_json()
    assert '"hit_count":1' in serialized
    assert "CONFIDENTIAL" not in serialized
    assert "credential" not in serialized
    assert "secret" not in serialized
    copied_trace = trace.model_copy(deep=True)
    assert copied_trace.branch_results[0].hits[0].metadata == {"credential": "secret"}


def test_complete_response_cannot_hide_a_failed_or_running_branch() -> None:
    trace = SearchTrace(
        request_id="request-outcome",
        query_hash="d" * 64,
        query_length=5,
        language_supported=True,
        features=features(),
        planner_decision=planner_decision(),
        branch_results=(
            BranchResult(
                branch=BranchKind.VECTOR,
                status=BranchStatus.SUCCEEDED,
                latency_ms=1.0,
            ),
        ),
        analyzer_latency_ms=0.5,
        planner_latency_ms=0.5,
        vector_latency_ms=1.0,
        total_latency_ms=2.0,
        latency_budget_ms=200,
        finalization_reserve_ms=10.0,
        budget_feasible=True,
        budget_violated=False,
        fallback=False,
        result_count=0,
        corpus_version="v1",
        config_version="v1",
    )
    response = SearchResponse(
        status=SearchStatus.COMPLETE,
        results=(),
        planner_decision=planner_decision(),
        trace=trace,
        request_id="request-outcome",
    )
    assert response.status is SearchStatus.COMPLETE

    failed_trace = trace.model_copy(
        update={
            "branch_results": (
                BranchResult(
                    branch=BranchKind.VECTOR,
                    status=BranchStatus.FAILED,
                    latency_ms=1.0,
                    error_code=ErrorCode.DEPENDENCY_UNAVAILABLE,
                ),
            )
        }
    )
    with pytest.raises(ValidationError, match="every scheduled branch"):
        response.model_copy(update={"trace": failed_trace})


def test_trace_has_no_hidden_deadline_grace() -> None:
    common = {
        "request_id": "request-1",
        "query_hash": "a" * 64,
        "query_length": 5,
        "language_supported": True,
        "features": features(),
        "planner_decision": planner_decision(),
        "branch_results": (
            BranchResult(
                branch=BranchKind.VECTOR,
                status=BranchStatus.SUCCEEDED,
                latency_ms=0.0,
            ),
        ),
        "analyzer_latency_ms": 1.0,
        "planner_latency_ms": 1.0,
        "vector_latency_ms": 0.0,
        "latency_budget_ms": 100,
        "finalization_reserve_ms": 5.0,
        "budget_feasible": True,
        "fallback": False,
        "result_count": 0,
        "corpus_version": "v1",
        "config_version": "v1",
    }
    SearchTrace(**common, total_latency_ms=100.0, budget_violated=False)
    SearchTrace(**common, total_latency_ms=100.0001, budget_violated=True)
    with pytest.raises(ValidationError, match="hidden grace"):
        SearchTrace(**common, total_latency_ms=100.0001, budget_violated=False)


def test_graph_path_rejects_cycles_and_disconnected_relations() -> None:
    relation = Relation(
        source_entity_id="entity-a",
        target_entity_id="entity-b",
        predicate="founded",
        confidence=0.8,
        source_chunk_id="chunk-1",
        extractor_version="v1",
    )
    assert GraphPath(entity_ids=("entity-b", "entity-a"), relations=(relation,)).hop_count == 1
    with pytest.raises(ValidationError, match="repeat"):
        GraphPath(entity_ids=("entity-a", "entity-b", "entity-a"), relations=(relation, relation))


def test_exact_graph_seed_scores_cannot_be_probabilistic() -> None:
    matched = {
        "mention_sha256": "a" * 64,
        "requested_entity_id": "entity-a",
        "matched_entity_id": "entity-a",
    }
    assert GraphSeedMatch(**matched, lookup_score=1.0).lookup_score == 1.0
    assert (
        GraphSeedMatch(
            mention_sha256="a" * 64,
            requested_entity_id="entity-a",
            lookup_score=0.0,
        ).matched_entity_id
        is None
    )
    with pytest.raises(ValidationError, match="exact seed lookup"):
        GraphSeedMatch(**matched, lookup_score=0.5)
    with pytest.raises(ValidationError, match="exact seed lookup"):
        GraphSeedMatch(
            mention_sha256="a" * 64,
            requested_entity_id="entity-a",
            lookup_score=0.5,
        )


def test_only_reconciled_ingestion_can_be_active() -> None:
    values = {
        "ingestion_run_id": "run-1",
        "corpus_version": "corpus-v1",
        "source_dataset": "fixture",
        "source_version": "v1",
        "source_sha256": "a" * 64,
        "chunker_version": "v1",
        "embedding_model_revision": "revision",
        "extractor_version": "v1",
        "document_count": 1,
        "chunk_count": 2,
        "qdrant_count": 2,
        "qdrant_id_checksum": "b" * 64,
        "qdrant_status": IngestionStoreStatus.SUCCEEDED,
        "neo4j_count": 2,
        "neo4j_id_checksum": "b" * 64,
        "neo4j_status": IngestionStoreStatus.SUCCEEDED,
        "activation_status": ActivationStatus.ACTIVE,
    }
    assert IngestionManifest(**values).activation_status is ActivationStatus.ACTIVE
    with pytest.raises(ValidationError, match="reconciled"):
        IngestionManifest(**{**values, "neo4j_count": 1})
    with pytest.raises(ValidationError, match="reconciled"):
        IngestionManifest(**{**values, "qdrant_count": 0, "neo4j_count": 0})
    with pytest.raises(ValidationError, match="reconciled"):
        IngestionManifest(
            **{
                **values,
                "document_count": 0,
                "chunk_count": 0,
                "qdrant_count": 0,
                "neo4j_count": 0,
            }
        )
    (Chunk,)
