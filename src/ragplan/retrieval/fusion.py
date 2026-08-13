"""Deterministic cross-store fusion and canonical-hit provenance."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

from ragplan.core.errors import ErrorCode, RAGPlanError
from ragplan.core.models import (
    BranchKind,
    FusionTrace,
    GraphPath,
    RetrievalContribution,
    RetrievalHit,
)

FUSION_VERSION: Final = "weighted_rrf_v1"
RRF_K: Final = 60
MAX_GRAPH_PATHS_PER_HIT: Final = 50
_CONSISTENCY_MESSAGE: Final = "retrieval branches disagree on canonical corpus evidence"
_BRANCH_NATIVE_METADATA_KEYS: Final = frozenset(
    {
        "normalized_seed_overlap",
        "inverse_hop_count",
        "mean_relation_confidence",
        "seed_overlap_contribution",
        "hop_contribution",
        "confidence_contribution",
        "matched_seed_count",
        "contributing_seed_count",
    }
)


@dataclass(frozen=True, slots=True)
class FusionResult:
    """Final ranked hits and a redacted aggregate fusion trace."""

    hits: tuple[RetrievalHit, ...]
    trace: FusionTrace


@dataclass(frozen=True, slots=True)
class _Candidate:
    vector_hit: RetrievalHit | None
    graph_hit: RetrievalHit | None


def weighted_rrf_v1(
    *,
    vector_hits: Sequence[RetrievalHit] | None,
    graph_hits: Sequence[RetrievalHit] | None,
    vector_weight: float,
    graph_weight: float,
    top_k: int,
) -> FusionResult:
    """Fuse branch rankings by canonical chunk ID using ADR-015 exactly.

    ``None`` means a branch was unavailable. An empty sequence means the branch
    completed successfully and found no candidates. A sole available branch
    retains its input order, including when its configured weight is zero.
    """

    _validate_weights(vector_weight, graph_weight)
    if not 1 <= top_k <= 50:
        raise ValueError("fusion top_k must be between one and fifty")

    vector_by_id = _index_branch(vector_hits, BranchKind.VECTOR)
    graph_by_id = _index_branch(graph_hits, BranchKind.GRAPH)
    all_ids = set(vector_by_id) | set(graph_by_id)
    candidates: dict[str, _Candidate] = {}
    for canonical_id in all_ids:
        vector_hit = vector_by_id.get(canonical_id)
        graph_hit = graph_by_id.get(canonical_id)
        if vector_hit is not None and graph_hit is not None:
            _require_consistent_evidence(vector_hit, graph_hit)
        candidates[canonical_id] = _Candidate(vector_hit, graph_hit)

    fused = tuple(
        _fused_hit(
            candidate,
            vector_rank=_rank_of(vector_hits, canonical_id),
            graph_rank=_rank_of(graph_hits, canonical_id),
            vector_weight=vector_weight,
            graph_weight=graph_weight,
        )
        for canonical_id, candidate in candidates.items()
    )

    if vector_hits is None and graph_hits is not None:
        ordered_ids = tuple(hit.canonical_chunk_id for hit in graph_hits)
        ordered = tuple(candidates[canonical_id] for canonical_id in ordered_ids)
        ranked = tuple(
            _fused_hit(
                candidate,
                vector_rank=None,
                graph_rank=rank,
                vector_weight=vector_weight,
                graph_weight=graph_weight,
            ).model_copy(update={"rank": rank})
            for rank, candidate in enumerate(ordered[:top_k], 1)
        )
    elif graph_hits is None and vector_hits is not None:
        ordered_ids = tuple(hit.canonical_chunk_id for hit in vector_hits)
        ordered = tuple(candidates[canonical_id] for canonical_id in ordered_ids)
        ranked = tuple(
            _fused_hit(
                candidate,
                vector_rank=rank,
                graph_rank=None,
                vector_weight=vector_weight,
                graph_weight=graph_weight,
            ).model_copy(update={"rank": rank})
            for rank, candidate in enumerate(ordered[:top_k], 1)
        )
    else:
        ordered_hits = sorted(fused, key=lambda hit: (-hit.score, hit.canonical_chunk_id))
        ranked = tuple(
            hit.model_copy(update={"rank": rank})
            for rank, hit in enumerate(ordered_hits[:top_k], 1)
        )

    missing = tuple(
        branch
        for branch, value in (
            (BranchKind.VECTOR, vector_hits),
            (BranchKind.GRAPH, graph_hits),
        )
        if value is None
    )
    duplicate_count = len(set(vector_by_id) & set(graph_by_id))
    return FusionResult(
        hits=ranked,
        trace=FusionTrace(
            vector_input_count=len(vector_by_id),
            graph_input_count=len(graph_by_id),
            output_count=len(ranked),
            duplicate_count=duplicate_count,
            missing_branches=missing,
        ),
    )


def annotate_single_source(
    hits: Sequence[RetrievalHit],
    *,
    source: BranchKind,
    top_k: int,
) -> tuple[RetrievalHit, ...]:
    """Add the same auditable provenance contract without changing branch scores."""

    indexed = _index_branch(hits, source)
    annotated: list[RetrievalHit] = []
    for rank, hit in enumerate(hits[:top_k], 1):
        contribution = _contribution(hit, source=source, rank=rank, weight=1.0)
        annotated.append(
            hit.model_copy(
                update={
                    "rank": rank,
                    "sources": (source,),
                    "source_contributions": (contribution,),
                }
            )
        )
    assert len(indexed) >= len(annotated)
    return tuple(annotated)


def _validate_weights(vector_weight: float, graph_weight: float) -> None:
    weights = (vector_weight, graph_weight)
    if any(not math.isfinite(weight) or not 0.0 <= weight <= 1.0 for weight in weights):
        raise ValueError("fusion weights must be finite values between zero and one")
    if not math.isclose(sum(weights), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("fusion weights must sum to one")


def _index_branch(
    hits: Sequence[RetrievalHit] | None,
    source: BranchKind,
) -> Mapping[str, RetrievalHit]:
    if hits is None:
        return {}
    indexed: dict[str, RetrievalHit] = {}
    for rank, hit in enumerate(hits, 1):
        if hit.source != source.value:
            raise RAGPlanError(ErrorCode.CORPUS_INCONSISTENT, _CONSISTENCY_MESSAGE)
        if hit.rank is not None and hit.rank != rank:
            raise RAGPlanError(ErrorCode.CORPUS_INCONSISTENT, _CONSISTENCY_MESSAGE)
        if hit.canonical_chunk_id in indexed:
            raise RAGPlanError(ErrorCode.CORPUS_INCONSISTENT, _CONSISTENCY_MESSAGE)
        indexed[hit.canonical_chunk_id] = hit
    return indexed


def _rank_of(hits: Sequence[RetrievalHit] | None, canonical_id: str) -> int | None:
    if hits is None:
        return None
    return next(
        (rank for rank, hit in enumerate(hits, 1) if hit.canonical_chunk_id == canonical_id),
        None,
    )


def _require_consistent_evidence(vector_hit: RetrievalHit, graph_hit: RetrievalHit) -> None:
    if vector_hit.text != graph_hit.text:
        raise RAGPlanError(ErrorCode.CORPUS_INCONSISTENT, _CONSISTENCY_MESSAGE)
    if (
        vector_hit.document_id is not None
        and graph_hit.document_id is not None
        and vector_hit.document_id != graph_hit.document_id
    ):
        raise RAGPlanError(ErrorCode.CORPUS_INCONSISTENT, _CONSISTENCY_MESSAGE)
    shared_metadata_keys = {
        key
        for key in set(vector_hit.metadata) & set(graph_hit.metadata)
        if key not in _BRANCH_NATIVE_METADATA_KEYS and not key.startswith(("native_", "graph_"))
    }
    if any(vector_hit.metadata[key] != graph_hit.metadata[key] for key in shared_metadata_keys):
        raise RAGPlanError(ErrorCode.CORPUS_INCONSISTENT, _CONSISTENCY_MESSAGE)


def _fused_hit(
    candidate: _Candidate,
    *,
    vector_rank: int | None,
    graph_rank: int | None,
    vector_weight: float,
    graph_weight: float,
) -> RetrievalHit:
    primary = candidate.vector_hit or candidate.graph_hit
    assert primary is not None
    contributions: list[RetrievalContribution] = []
    if candidate.vector_hit is not None:
        assert vector_rank is not None
        contributions.append(
            _contribution(
                candidate.vector_hit,
                source=BranchKind.VECTOR,
                rank=vector_rank,
                weight=vector_weight,
            )
        )
    if candidate.graph_hit is not None:
        assert graph_rank is not None
        contributions.append(
            _contribution(
                candidate.graph_hit,
                source=BranchKind.GRAPH,
                rank=graph_rank,
                weight=graph_weight,
            )
        )
    graph_paths = candidate.graph_hit.paths if candidate.graph_hit is not None else ()
    score = sum(item.rrf_contribution for item in contributions)
    document_id = primary.document_id
    if document_id is None and candidate.graph_hit is not None:
        document_id = candidate.graph_hit.document_id
    entity_ids = tuple(
        sorted(
            set(candidate.vector_hit.entity_ids if candidate.vector_hit is not None else ())
            | set(candidate.graph_hit.entity_ids if candidate.graph_hit is not None else ())
        )
    )
    return RetrievalHit(
        canonical_chunk_id=primary.canonical_chunk_id,
        text=primary.text,
        score=score,
        source="fusion",
        document_id=document_id,
        entity_ids=entity_ids,
        metadata=primary.model_dump(mode="json")["metadata"],
        paths=_deduplicate_paths(graph_paths),
        sources=tuple(item.source for item in contributions),
        source_contributions=tuple(contributions),
    )


def _contribution(
    hit: RetrievalHit,
    *,
    source: BranchKind,
    rank: int,
    weight: float,
) -> RetrievalContribution:
    return RetrievalContribution(
        source=source,
        original_rank=rank,
        original_score=hit.score,
        weight=weight,
        rrf_contribution=weight / (RRF_K + rank),
        metadata=hit.model_dump(mode="json")["metadata"],
    )


def _deduplicate_paths(paths: Sequence[GraphPath]) -> tuple[GraphPath, ...]:
    unique: dict[tuple[object, ...], GraphPath] = {}
    for path in paths:
        key = _path_key(path)
        existing = unique.get(key)
        if existing is None or path.score > existing.score:
            unique[key] = path
    return tuple(sorted(unique.values(), key=lambda item: (-item.score, _path_key(item))))[
        :MAX_GRAPH_PATHS_PER_HIT
    ]


def _path_key(path: GraphPath) -> tuple[object, ...]:
    return (
        path.entity_ids,
        tuple(
            (
                relation.source_entity_id,
                relation.target_entity_id,
                relation.predicate,
                relation.confidence,
                relation.source_chunk_id,
                relation.extractor_version,
                relation.extraction_rule.value,
            )
            for relation in path.relations
        ),
    )


__all__ = [
    "FUSION_VERSION",
    "MAX_GRAPH_PATHS_PER_HIT",
    "RRF_K",
    "FusionResult",
    "annotate_single_source",
    "weighted_rrf_v1",
]
