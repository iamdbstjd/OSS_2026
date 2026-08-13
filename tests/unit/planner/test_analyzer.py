from __future__ import annotations

from collections.abc import Sequence

import pytest

from ragplan.core.deadline import Deadline, ManualClock
from ragplan.core.models import QueryAnalysis, QueryFeatures
from ragplan.planner.analyzer import QueryAnalyzer
from ragplan.planner.features import load_default_query_feature_config

pytestmark = pytest.mark.unit


class _EntityAnalyzer:
    calls = 0

    def __init__(self, clock: ManualClock) -> None:
        self.clock = clock

    def analyze(self, query: str, *, final_top_k: int) -> QueryAnalysis:
        self.calls += 1
        self.clock.advance_ms(1)
        return QueryAnalysis(
            normalized_query=query.strip(),
            language_supported=True,
            token_count=8,
            query_embedding=(),
            seed_entity_mentions=("Ada", "Acme"),
            seed_entity_ids=("person:ada", "org:acme"),
            features=QueryFeatures(
                token_count=8,
                entity_count=2,
                entity_density=0.25,
                relation_signal=0.0,
                multi_hop_signal=0.0,
                comparison_signal=0.0,
                aggregation_signal=0.0,
                global_signal=0.0,
                final_top_k=final_top_k,
            ),
            analyzer_version="fake-entity-v1",
            analysis_latency_ms=1.0,
        )


class _Embedder:
    calls = 0

    def __init__(self, clock: ManualClock) -> None:
        self.clock = clock

    async def embed_query(self, query: str) -> Sequence[float]:
        self.calls += 1
        self.clock.advance_ms(2)
        return (1.0, 0.0, 0.0)


@pytest.mark.asyncio
async def test_analyzer_runs_entity_extraction_and_embedding_exactly_once() -> None:
    clock = ManualClock()
    entity = _EntityAnalyzer(clock)
    embedder = _Embedder(clock)
    analyzer = QueryAnalyzer(
        entity_analyzer=entity,
        embedder=embedder,
        feature_config=load_default_query_feature_config(),
        clock=clock,
    )

    execution = await analyzer.analyze(
        "Who founded Acme and who acquired it?",
        final_top_k=10,
        deadline=Deadline.start(200, clock=clock),
    )

    assert entity.calls == 1
    assert embedder.calls == 1
    assert execution.analysis.query_embedding == (1.0, 0.0, 0.0)
    assert execution.analysis.features.entity_count == 2
    assert execution.analysis.features.relation_signal >= 0.5
    assert execution.embedding_latency_ms == 2.0
    serialized = execution.analysis.model_dump_json()
    assert "query_embedding" not in serialized
    assert "Who founded" not in execution.analysis.features.model_dump_json()
