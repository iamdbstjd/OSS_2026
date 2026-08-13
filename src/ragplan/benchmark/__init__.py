"""Frozen benchmark-data, qrels, split, and metric foundations."""

from ragplan.benchmark.builder import Stage2BuildResult, build_stage2
from ragplan.benchmark.metrics import mrr_at_k, ndcg_at_k, recall_at_k

__all__ = [
    "Stage2BuildResult",
    "build_stage2",
    "mrr_at_k",
    "ndcg_at_k",
    "recall_at_k",
]
