from __future__ import annotations

from typing import Any

import pytest
from neo4j.exceptions import ConnectionAcquisitionTimeoutError

from ragplan.backends.graph.neo4j import Neo4jGraphBackend, Neo4jGraphConfig
from ragplan.core.deadline import Deadline, ManualClock
from ragplan.core.errors import ErrorCode, RAGPlanError, TimeoutOrigin
from ragplan.core.models import BranchStatus, QueryAnalysis, QueryFeatures
from ragplan.planner.catalog import load_default_plan_catalog
from ragplan.retrieval.graph import execute_graph_branch

pytestmark = pytest.mark.unit


class _RuntimeDriver:
    def __init__(self, *, matched: bool = True, timeout: bool = False) -> None:
        self.matched = matched
        self.timeout = timeout
        self.calls: list[tuple[str, dict[str, object], float | None]] = []
        self.closed = False

    async def execute_query(
        self,
        query: Any,
        *,
        parameters_: dict[str, object],
        database_: str,
    ) -> tuple[list[dict[str, object]], None, list[str]]:
        del database_
        statement = str(query)
        self.calls.append((statement, parameters_, query.timeout))
        if self.timeout:
            raise TimeoutError
        if "UNWIND $seeds AS seed" in statement:
            rows = parameters_["seeds"]
            assert isinstance(rows, list)
            return (
                [
                    {
                        "position": row["position"],
                        "mention_sha256": row["mention_sha256"],
                        "requested_entity_id": row["entity_id"],
                        "matched_entity_id": row["entity_id"] if self.matched else None,
                    }
                    for row in rows
                ],
                None,
                [],
            )
        if "[relation:RELATES_TO]" in statement:
            frontier = parameters_["frontier_ids"]
            assert isinstance(frontier, list)
            edges = (
                _edge_record("entity-a", "entity-b", "chunk-ab", 0.9),
                _edge_record("entity-c", "entity-b", "chunk-bc", 0.8),
                _edge_record("entity-c", "entity-d", "chunk-cd", 1.0),
                _edge_record("entity-d", "entity-a", "chunk-cycle", 0.7),
            )
            return (
                [
                    edge
                    for edge in edges
                    if edge["source_entity_id"] in frontier or edge["target_entity_id"] in frontier
                ],
                None,
                [],
            )
        if "-[:MENTIONS" in statement and "RETURN chunk.id" in statement:
            reached = parameters_["entity_ids"]
            assert isinstance(reached, list)
            return (
                [
                    {
                        "canonical_chunk_id": f"v1:chunk:{entity_id}",
                        "document_id": f"v1:document:{entity_id}",
                        "text": f"evidence for {entity_id}",
                        "entity_ids": [entity_id],
                    }
                    for entity_id in reached
                ],
                None,
                [],
            )
        raise AssertionError(f"unexpected Cypher: {statement}")

    async def verify_connectivity(self) -> None:
        return None

    async def close(self) -> None:
        self.closed = True


def _edge_record(
    source: str,
    target: str,
    chunk: str,
    confidence: float,
) -> dict[str, object]:
    return {
        "frontier_entity_id": source,
        "neighbor_entity_id": target,
        "source_entity_id": source,
        "target_entity_id": target,
        "predicate": f"rel-{source}-{target}",
        "confidence": confidence,
        "source_chunk_id": f"v1:chunk:{chunk}",
        "extractor_version": "fixture-v1",
        "extraction_rule": "direct_svo",
    }


def _analysis(*, with_seed: bool = True) -> QueryAnalysis:
    return QueryAnalysis(
        normalized_query="entity a relationship",
        language_supported=True,
        token_count=3,
        query_embedding=(),
        seed_entity_mentions=("entity a",) if with_seed else (),
        seed_entity_ids=("entity-a",) if with_seed else (),
        features=QueryFeatures(
            token_count=3,
            entity_count=1 if with_seed else 0,
            entity_density=1 / 3 if with_seed else 0.0,
            relation_signal=0.0,
            multi_hop_signal=0.0,
            comparison_signal=0.0,
            aggregation_signal=0.0,
            global_signal=0.0,
            final_top_k=10,
        ),
        analyzer_version="fixture-v1",
        analysis_latency_ms=0.0,
    )


@pytest.mark.asyncio
async def test_runtime_backend_is_exact_bounded_and_preserves_edge_direction() -> None:
    clock = ManualClock()
    deadline = Deadline.start(200, clock=clock)
    driver = _RuntimeDriver()
    backend = Neo4jGraphBackend(
        driver,
        Neo4jGraphConfig(password="test-only", transaction_timeout_seconds=30.0),
    )

    result = await backend.search(
        _analysis(),
        load_default_plan_catalog().plan_for_id("P3"),
        "active-v1",
        deadline,
    )

    assert result.trace.requested_depth == 3
    assert result.trace.actual_depth == 3
    assert result.trace.visited_entity_count == 4
    assert result.trace.path_count <= 50
    assert result.trace.recovered_chunk_count == 3
    assert {hit.canonical_chunk_id for hit in result.hits} == {
        "v1:chunk:entity-b",
        "v1:chunk:entity-c",
        "v1:chunk:entity-d",
    }
    reverse_path = next(
        path
        for hit in result.hits
        for path in hit.paths
        if path.entity_ids == ("entity-a", "entity-b", "entity-c")
    )
    assert reverse_path.relations[1].source_entity_id == "entity-c"
    assert reverse_path.relations[1].target_entity_id == "entity-b"
    lookup_call = next(call for call in driver.calls if "UNWIND $seeds" in call[0])
    assert lookup_call[1]["seeds"] == [
        {
            "position": 0,
            "normalized_alias": "entity a",
            "entity_id": "entity-a",
            "mention_sha256": lookup_call[1]["seeds"][0]["mention_sha256"],  # type: ignore[index]
        }
    ]
    traversal_statements = [
        statement for statement, *_ in driver.calls if "RELATES_TO" in statement
    ]
    assert traversal_statements
    assert all("[*" not in statement for statement in traversal_statements)
    assert all("MENTIONS" not in statement for statement in traversal_statements)
    assert all(timeout is not None and timeout <= 0.19 for _, _, timeout in driver.calls)


@pytest.mark.asyncio
async def test_no_seed_and_unmatched_seed_are_valid_empty_results() -> None:
    plan = load_default_plan_catalog().plan_for_id("P2")
    clock = ManualClock()

    no_seed_driver = _RuntimeDriver()
    no_seed_backend = Neo4jGraphBackend(no_seed_driver, Neo4jGraphConfig(password="test-only"))
    no_seed = await no_seed_backend.search(
        _analysis(with_seed=False),
        plan,
        "active-v1",
        Deadline.start(200, clock=clock),
    )
    assert no_seed.hits == ()
    assert no_seed.trace.seed_matches == ()
    assert no_seed_driver.calls == []

    unmatched_driver = _RuntimeDriver(matched=False)
    unmatched_backend = Neo4jGraphBackend(
        unmatched_driver,
        Neo4jGraphConfig(password="test-only"),
    )
    unmatched = await unmatched_backend.search(
        _analysis(),
        plan,
        "active-v1",
        Deadline.start(200, clock=clock),
    )
    assert unmatched.hits == ()
    assert unmatched.trace.seed_matches[0].lookup_score == 0.0
    assert unmatched.trace.visited_entity_count == 0
    assert len(unmatched_driver.calls) == 1


@pytest.mark.asyncio
async def test_graph_timeout_returns_typed_deadline_failure() -> None:
    direct_backend = Neo4jGraphBackend(
        _RuntimeDriver(timeout=True),
        Neo4jGraphConfig(password="test-only"),
    )

    with pytest.raises(RAGPlanError) as caught:
        await direct_backend.search(
            _analysis(),
            load_default_plan_catalog().plan_for_id("P2"),
            "active-v1",
            Deadline.start(25, clock=ManualClock()),
        )

    assert caught.value.code is ErrorCode.DEADLINE_EXCEEDED
    assert caught.value.timeout_origin is TimeoutOrigin.APPLICATION_DEADLINE

    branch_backend = Neo4jGraphBackend(
        _RuntimeDriver(timeout=True),
        Neo4jGraphConfig(password="test-only"),
    )
    branch = await execute_graph_branch(
        backend=branch_backend,
        query_analysis=_analysis(),
        plan=load_default_plan_catalog().plan_for_id("P2"),
        corpus_version="active-v1",
        deadline=Deadline.start(25, clock=ManualClock()),
    )
    assert branch.result.status is BranchStatus.TIMED_OUT
    assert branch.result.error_code is None
    assert branch.trace is None


@pytest.mark.asyncio
async def test_graph_driver_timeout_is_typed_as_backend_client_timeout() -> None:
    class NativeTimeoutDriver(_RuntimeDriver):
        async def execute_query(self, *args: object, **kwargs: object) -> object:
            del args, kwargs
            raise ConnectionAcquisitionTimeoutError("pool timeout")

    backend = Neo4jGraphBackend(
        NativeTimeoutDriver(),
        Neo4jGraphConfig(password="test-only"),
    )

    with pytest.raises(RAGPlanError) as captured:
        await backend.search(
            _analysis(),
            load_default_plan_catalog().plan_for_id("P2"),
            "active-v1",
            Deadline.start(200),
        )

    assert captured.value.code is ErrorCode.DEADLINE_EXCEEDED
    assert captured.value.timeout_origin is TimeoutOrigin.BACKEND_CLIENT
