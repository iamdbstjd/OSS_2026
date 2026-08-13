from __future__ import annotations

from collections import Counter

from ragplan.benchmark.synthetic import build_synthetic_graph_fixture


def test_synthetic_graph_fixture_is_deterministic_bounded_and_non_primary() -> None:
    first = build_synthetic_graph_fixture()
    second = build_synthetic_graph_fixture()
    assert first == second
    assert first.manifest_sha256 == second.manifest_sha256
    assert first.primary_metrics_eligible is False
    assert Counter(query.hop_count for query in first.queries) == {1: 34, 2: 33, 3: 33}
    assert any(edge.relation == "CYCLES_TO" for edge in first.edges)
    assert any(edge.source_entity == "entity_shared_hub" for edge in first.edges)
    assert any(edge.source_entity == "entity_disconnected_a" for edge in first.edges)
