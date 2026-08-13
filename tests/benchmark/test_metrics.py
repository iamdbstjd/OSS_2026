from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from ragplan.benchmark.metrics import mrr_at_10, ndcg_at_10, recall_at_5, recall_at_10

FIXTURE = Path(__file__).parents[1] / "fixtures/benchmark/metric_cases.json"


@pytest.mark.benchmark
@pytest.mark.parametrize("case", json.loads(FIXTURE.read_text(encoding="utf-8")))
def test_metrics_match_frozen_reference_cases(case: dict[str, Any]) -> None:
    ranked = case["ranked"]
    relevance = case["relevance"]
    expected = case["expected"]

    assert recall_at_5(ranked, relevance) == pytest.approx(expected["recall_at_5"])
    assert recall_at_10(ranked, relevance) == pytest.approx(expected["recall_at_10"])
    assert mrr_at_10(ranked, relevance) == pytest.approx(expected["mrr_at_10"])
    assert ndcg_at_10(ranked, relevance) == pytest.approx(expected["ndcg_at_10"])


def test_metrics_reject_zero_relevant_and_invalid_grades() -> None:
    with pytest.raises(ValueError, match="no relevant"):
        recall_at_5(("a",), {"a": 0})
    with pytest.raises(ValueError, match=r"\[0, 2\]"):
        ndcg_at_10(("a",), {"a": 3})


def test_duplicate_retrieval_ids_do_not_consume_multiple_ranks() -> None:
    assert mrr_at_10(("x", "x", "a"), {"a": 1}) == 0.5
