from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from ragplan.api.server import create_app
from ragplan.core.deadline import Deadline, PerfCounterClock
from ragplan.core.engine import GraphSearchEngine
from ragplan.core.models import (
    ActivationStatus,
    GraphPath,
    GraphSeedMatch,
    GraphTrace,
    IngestionManifest,
    IngestionStoreStatus,
    PlanDefinition,
    QueryAnalysis,
    Relation,
    RelationExtractionRule,
    RetrievalHit,
)
from ragplan.ingestion.entities import EntityExtractor
from ragplan.planner.catalog import load_default_plan_catalog
from ragplan.retrieval.graph import GraphBackendExecution, GraphQueryAnalyzer

pytestmark = pytest.mark.unit


class _GraphBackend:
    def __init__(self) -> None:
        self.closed = False

    async def search(
        self,
        query_analysis: QueryAnalysis,
        plan: PlanDefinition,
        corpus_version: str,
        deadline: Deadline,
    ) -> GraphBackendExecution:
        del plan, corpus_version, deadline
        seed_id = query_analysis.seed_entity_ids[0]
        relation = Relation(
            source_entity_id=seed_id,
            target_entity_id="target-entity",
            predicate="relate",
            confidence=0.9,
            source_chunk_id="v1:chunk:graph-evidence",
            extractor_version="fixture-v1",
            extraction_rule=RelationExtractionRule.DIRECT_SVO,
        )
        path = GraphPath(
            entity_ids=(seed_id, "target-entity"),
            relations=(relation,),
            score=0.98,
        )
        return GraphBackendExecution(
            hits=(
                RetrievalHit(
                    canonical_chunk_id="v1:chunk:graph-evidence",
                    document_id="v1:document:graph-evidence",
                    text="bounded graph evidence",
                    score=0.98,
                    source="graph",
                    rank=1,
                    paths=(path,),
                ),
            ),
            trace=GraphTrace(
                seed_matches=(
                    GraphSeedMatch(
                        mention_sha256="a" * 64,
                        requested_entity_id=seed_id,
                        matched_entity_id=seed_id,
                        lookup_score=1.0,
                    ),
                ),
                requested_depth=1,
                actual_depth=1,
                visited_entity_count=2,
                path_count=1,
                recovered_chunk_count=1,
                seed_lookup_latency_ms=0.0,
                traversal_latency_ms=0.0,
                recovery_latency_ms=0.0,
                ranking_latency_ms=0.0,
            ),
        )

    async def close(self) -> None:
        self.closed = True


def _active() -> IngestionManifest:
    return IngestionManifest(
        ingestion_run_id="api-stage5-run-v1",
        corpus_version="api-stage5-v1",
        source_dataset="fixture",
        source_version="v1",
        source_sha256="a" * 64,
        chunker_version="v1",
        embedding_model_revision="embedding-v1",
        extractor_version="fixture-v1",
        document_count=1,
        chunk_count=1,
        qdrant_count=1,
        qdrant_id_checksum="b" * 64,
        qdrant_status=IngestionStoreStatus.SUCCEEDED,
        neo4j_count=1,
        neo4j_id_checksum="b" * 64,
        neo4j_status=IngestionStoreStatus.SUCCEEDED,
        activation_status=ActivationStatus.ACTIVE,
    )


@pytest.mark.asyncio
async def test_injected_graph_engine_serves_paths_and_bounded_trace() -> None:
    clock = PerfCounterClock()
    backend = _GraphBackend()
    engine = GraphSearchEngine(
        analyzer=GraphQueryAnalyzer(
            EntityExtractor.load_pinned(lockfile=Path("uv.lock")),
            clock=clock,
        ),
        graph_backend=backend,
        plan_catalog=load_default_plan_catalog(),
        active_manifest=_active(),
        clock=clock,
    )
    app = create_app(search_engine=engine)

    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
            response = await http.post(
                "/v1/search",
                json={"query": "How is Apple related?", "planner": "graph", "top_k": 1},
                headers={"x-request-id": "stage5-http-request"},
            )

        assert response.status_code == 200
        body = response.json()
        assert body["results"][0]["source"] == "graph"
        assert body["results"][0]["paths"][0]["hop_count"] == 1
        assert body["trace"]["graph_trace"]["requested_depth"] == 1
        assert body["trace"]["graph_trace"]["visited_entity_count"] == 2
        assert body["trace"]["graph_latency_ms"] >= 0

    assert backend.closed is True
