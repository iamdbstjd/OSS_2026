"""Dependency-free, deterministic Stage 2 ranking metrics."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence


def recall_at_k(
    ranked_chunk_ids: Sequence[str],
    relevance: Mapping[str, int],
    *,
    k: int,
) -> float:
    """Return set recall at ``k`` using grade >= 1 as relevant."""

    _validate_inputs(ranked_chunk_ids, relevance, k=k)
    relevant = {chunk_id for chunk_id, grade in relevance.items() if grade >= 1}
    if not relevant:
        raise ValueError("recall is undefined when a query has no relevant chunks")
    retrieved = set(_unique_prefix(ranked_chunk_ids, k=k))
    return len(relevant & retrieved) / len(relevant)


def mrr_at_k(
    ranked_chunk_ids: Sequence[str],
    relevance: Mapping[str, int],
    *,
    k: int,
) -> float:
    """Return reciprocal rank of the first grade >= 1 result within ``k``."""

    _validate_inputs(ranked_chunk_ids, relevance, k=k)
    if not any(grade >= 1 for grade in relevance.values()):
        raise ValueError("MRR is undefined when a query has no relevant chunks")
    for rank, chunk_id in enumerate(_unique_prefix(ranked_chunk_ids, k=k), start=1):
        if relevance.get(chunk_id, 0) >= 1:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(
    ranked_chunk_ids: Sequence[str],
    relevance: Mapping[str, int],
    *,
    k: int,
) -> float:
    """Return graded nDCG at ``k`` using gains ``2**grade - 1``."""

    _validate_inputs(ranked_chunk_ids, relevance, k=k)
    ideal_grades = sorted(relevance.values(), reverse=True)[:k]
    ideal = _dcg(ideal_grades)
    if ideal <= 0.0:
        raise ValueError("nDCG is undefined when a query has no relevant chunks")
    observed = [relevance.get(chunk_id, 0) for chunk_id in _unique_prefix(ranked_chunk_ids, k=k)]
    return _dcg(observed) / ideal


def recall_at_5(ranked_chunk_ids: Sequence[str], relevance: Mapping[str, int]) -> float:
    return recall_at_k(ranked_chunk_ids, relevance, k=5)


def recall_at_10(ranked_chunk_ids: Sequence[str], relevance: Mapping[str, int]) -> float:
    return recall_at_k(ranked_chunk_ids, relevance, k=10)


def mrr_at_10(ranked_chunk_ids: Sequence[str], relevance: Mapping[str, int]) -> float:
    return mrr_at_k(ranked_chunk_ids, relevance, k=10)


def ndcg_at_10(ranked_chunk_ids: Sequence[str], relevance: Mapping[str, int]) -> float:
    return ndcg_at_k(ranked_chunk_ids, relevance, k=10)


def _validate_inputs(
    ranked_chunk_ids: Sequence[str], relevance: Mapping[str, int], *, k: int
) -> None:
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer")
    if isinstance(ranked_chunk_ids, (str, bytes)):
        raise TypeError("ranked_chunk_ids must be a sequence of IDs")
    if not all(isinstance(chunk_id, str) and chunk_id for chunk_id in ranked_chunk_ids):
        raise ValueError("ranked chunk IDs must be non-empty strings")
    for chunk_id, grade in relevance.items():
        if not isinstance(chunk_id, str) or not chunk_id:
            raise ValueError("relevance keys must be non-empty strings")
        if isinstance(grade, bool) or not isinstance(grade, int) or not 0 <= grade <= 2:
            raise ValueError("relevance grades must be integers in [0, 2]")


def _unique_prefix(ranked_chunk_ids: Sequence[str], *, k: int) -> tuple[str, ...]:
    unique: list[str] = []
    seen: set[str] = set()
    for chunk_id in ranked_chunk_ids:
        if chunk_id in seen:
            continue
        seen.add(chunk_id)
        unique.append(chunk_id)
        if len(unique) == k:
            break
    return tuple(unique)


def _dcg(grades: Sequence[int]) -> float:
    return float(sum((2**grade - 1) / math.log2(rank + 1) for rank, grade in enumerate(grades, 1)))
