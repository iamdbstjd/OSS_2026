from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from ragplan.backends.graph.neo4j import (
    _LOOKUP_SEEDS,
    _READ_ADJACENT_RELATIONS,
    _RECOVER_CHUNKS,
)
from ragplan.core.models import GraphPath, GraphTrace, PlannerDecision
from ragplan.retrieval.graph import (
    MAX_PATHS_PER_SEED,
    MAX_RECOVERED_CHUNKS,
    MAX_SEED_ENTITIES,
    MAX_VISITED_ENTITIES,
)

pytestmark = [pytest.mark.contract, pytest.mark.unit]
ROOT = Path(__file__).resolve().parents[2]


def test_stage5_hard_bounds_and_public_trace_schema_are_frozen() -> None:
    assert (MAX_SEED_ENTITIES, MAX_PATHS_PER_SEED) == (5, 50)
    assert (MAX_VISITED_ENTITIES, MAX_RECOVERED_CHUNKS) == (500, 100)

    graph_trace_fields = GraphTrace.model_json_schema(mode="serialization")["properties"]
    path_fields = GraphPath.model_json_schema(mode="serialization")["properties"]
    decision_fields = PlannerDecision.model_json_schema(mode="serialization")["properties"]
    assert {
        "seed_matches",
        "requested_depth",
        "actual_depth",
        "visited_entity_count",
        "path_count",
        "recovered_chunk_count",
        "limit_hits",
        "seed_lookup_latency_ms",
        "traversal_latency_ms",
        "recovery_latency_ms",
        "ranking_latency_ms",
    } <= set(graph_trace_fields)
    assert "hop_count" in path_fields
    assert "executed_graph_top_k" in decision_fields


def test_runtime_cypher_has_no_variable_length_or_non_relation_traversal() -> None:
    source = (ROOT / "src/ragplan/backends/graph/neo4j.py").read_text(encoding="utf-8")
    traversal = source.split('_READ_ADJACENT_RELATIONS = """', 1)[1].split('"""', 1)[0]
    recovery = source.split('_RECOVER_CHUNKS = """', 1)[1].split('"""', 1)[0]

    assert "[*" not in source
    assert "[relation:RELATES_TO]" in traversal
    assert "MENTIONS" not in traversal
    assert "HAS_CHUNK" not in traversal
    assert "LIMIT $edge_limit" in traversal
    assert "MENTIONS" in recovery
    assert "corpus_version: $corpus_version" in recovery
    assert "LIMIT $candidate_limit" in recovery


def test_stage5_cli_container_and_real_integration_gate_are_delivered() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    runtime_docs = (ROOT / "docs/runtime.md").read_text(encoding="utf-8")

    assert (ROOT / "scripts/search_graph.py").is_file()
    assert "--group graph-extraction" in dockerfile
    assert "tests/integration/test_graph_retrieval.py" in workflow
    assert "scripts/search_graph.py" in runtime_docs


def test_explain_profile_artifact_is_bound_to_current_runtime_queries() -> None:
    artifact = json.loads(
        (ROOT / "benchmark/results/stage5_graph_query_plan_v1.json").read_text(encoding="utf-8")
    )
    queries = {item["id"]: item for item in artifact["queries"]}
    source_queries = {
        "seed_lookup": _LOOKUP_SEEDS,
        "adjacent_relates_to": _READ_ADJACENT_RELATIONS,
        "chunk_recovery": _RECOVER_CHUNKS,
    }

    assert set(queries) == set(source_queries)
    for query_id, statement in source_queries.items():
        assert (
            queries[query_id]["statement_sha256"]
            == hashlib.sha256(statement.encode("utf-8")).hexdigest()
        )
        assert queries[query_id]["review"]["parameterized"] is True
        assert queries[query_id]["profile"]["returned_records"] >= 1
    assert "NodeUniqueIndexSeek" in queries["adjacent_relates_to"]["explain_operators"]
    assert artifact["review_outcome"]["all_nodes_scan_present"] is False
    assert artifact["review_outcome"]["unbounded_variable_length_pattern_present"] is False
