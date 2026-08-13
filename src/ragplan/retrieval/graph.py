"""Bounded, deterministic Stage 5 graph analysis, traversal, and ranking."""

from __future__ import annotations

import hashlib
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Final, Protocol

from ragplan.core.deadline import NANOSECONDS_PER_MILLISECOND, Deadline, MonotonicClock
from ragplan.core.errors import ErrorCode, RAGPlanError, TimeoutOrigin
from ragplan.core.ids import canonical_chunk_id, canonical_document_id
from ragplan.core.models import (
    BranchKind,
    BranchResult,
    BranchStatus,
    CancellationReason,
    Chunk,
    FailureOrigin,
    GraphPath,
    GraphTrace,
    PlanDefinition,
    QueryAnalysis,
    QueryFeatures,
    Relation,
    RetrievalHit,
)
from ragplan.ingestion.entities import EntityExtractor
from ragplan.ingestion.normalize import normalize_text

MAX_SEED_ENTITIES: Final = 5
MAX_PATHS_PER_SEED: Final = 50
MAX_VISITED_ENTITIES: Final = 500
MAX_RECOVERED_CHUNKS: Final = 100
MAX_EDGE_ROWS_PER_DEPTH: Final = 1_001
GRAPH_SCORE_VERSION: Final = "graph_score_v0"
GRAPH_ANALYZER_VERSION: Final = "stage5-query-v1"


@dataclass(frozen=True, slots=True)
class GraphTraversalEdge:
    """One stored directed relation made discoverable from either endpoint."""

    relation: Relation

    def other_endpoint(self, entity_id: str) -> str | None:
        if self.relation.source_entity_id == entity_id:
            return self.relation.target_entity_id
        if self.relation.target_entity_id == entity_id:
            return self.relation.source_entity_id
        return None


@dataclass(frozen=True, slots=True)
class GraphTraversalPath:
    """Internal path whose first entity is always its originating query seed."""

    seed_entity_id: str
    entity_ids: tuple[str, ...]
    relations: tuple[Relation, ...]

    def __post_init__(self) -> None:
        if not self.seed_entity_id or self.entity_ids[0] != self.seed_entity_id:
            raise ValueError("a traversal path must begin at its seed")
        if len(self.entity_ids) != len(self.relations) + 1:
            raise ValueError("a traversal path must contain one entity per relation endpoint")
        if len(self.relations) not in {1, 2, 3}:
            raise ValueError("a traversal path must contain one to three relations")
        if len(set(self.entity_ids)) != len(self.entity_ids):
            raise ValueError("a traversal path cannot repeat an entity")

    @property
    def hop_count(self) -> int:
        return len(self.relations)


@dataclass(frozen=True, slots=True)
class EdgeBatch:
    """One deterministically capped adjacency read."""

    edges: tuple[GraphTraversalEdge, ...]
    truncated: bool = False


@dataclass(frozen=True, slots=True)
class GraphTraversalOutcome:
    paths: tuple[GraphTraversalPath, ...]
    visited_entity_count: int
    actual_depth: int
    path_limit_hit: bool
    visited_limit_hit: bool


@dataclass(frozen=True, slots=True)
class RecoveredGraphChunk:
    canonical_chunk_id: str
    document_id: str
    text: str
    entity_ids: tuple[str, ...]
    paths: tuple[GraphTraversalPath, ...]


@dataclass(frozen=True, slots=True)
class GraphBackendExecution:
    """Graph hits plus the bounded phase evidence produced by the backend."""

    hits: tuple[RetrievalHit, ...]
    trace: GraphTrace


@dataclass(frozen=True, slots=True)
class GraphExecution:
    """One complete graph branch measured at the shared engine boundary."""

    hits: tuple[RetrievalHit, ...]
    trace: GraphTrace
    latency_ms: float


@dataclass(frozen=True, slots=True)
class GraphBranchExecution:
    """Typed branch state consumed by graph-only now and the later scheduler."""

    result: BranchResult
    trace: GraphTrace | None = None


EdgeLoader = Callable[[tuple[str, ...]], Awaitable[EdgeBatch]]


class GraphExecutionBackend(Protocol):
    async def search(
        self,
        query_analysis: QueryAnalysis,
        plan: PlanDefinition,
        corpus_version: str,
        deadline: Deadline,
    ) -> GraphBackendExecution: ...

    async def close(self) -> None: ...


class GraphQueryAnalyzer:
    """Reuse the pinned ingestion NER pipeline and expose at most five exact seeds."""

    def __init__(self, extractor: EntityExtractor, *, clock: MonotonicClock) -> None:
        self._extractor = extractor
        self._clock = clock

    @property
    def extractor_version(self) -> str:
        return self._extractor.extractor_version

    def analyze(self, query: str, *, final_top_k: int) -> QueryAnalysis:
        started_ns = self._clock.now_ns()
        normalized_query = normalize_text(query)
        query_digest = hashlib.sha256(normalized_query.encode("utf-8")).hexdigest()
        document_id = canonical_document_id("runtime-query", query_digest)
        query_chunk = Chunk(
            id=canonical_chunk_id(document_id, 0, normalized_query),
            document_id=document_id,
            corpus_version="runtime-query-v1",
            position=0,
            text=normalized_query,
            token_count=1,
        )
        extraction = self._extractor.extract(query_chunk)

        unique_mentions: list[tuple[str, str]] = []
        observed: set[tuple[str, str]] = set()
        for mention in extraction.mentions:
            candidate = (mention.normalized_name, mention.entity_id)
            if candidate in observed:
                continue
            observed.add(candidate)
            unique_mentions.append(candidate)
        selected = unique_mentions[:MAX_SEED_ENTITIES]
        token_count = len(extraction.tokens)
        entity_count = len(selected)
        features = QueryFeatures(
            token_count=token_count,
            entity_count=entity_count,
            entity_density=min(1.0, entity_count / max(1, token_count)),
            relation_signal=1.0 if entity_count >= 2 else 0.0,
            multi_hop_signal=0.0,
            comparison_signal=0.0,
            aggregation_signal=0.0,
            global_signal=0.0,
            final_top_k=final_top_k,
        )
        return QueryAnalysis(
            normalized_query=normalized_query,
            language_supported=is_p0_english(normalized_query),
            token_count=token_count,
            query_embedding=(),
            seed_entity_mentions=tuple(alias for alias, _ in selected),
            seed_entity_ids=tuple(entity_id for _, entity_id in selected),
            seed_limit_hit=len(unique_mentions) > MAX_SEED_ENTITIES,
            features=features,
            analyzer_version=f"{GRAPH_ANALYZER_VERSION}:{self.extractor_version}",
            analysis_latency_ms=_elapsed_ms(started_ns, self._clock.now_ns()),
        )


async def traverse_bounded(
    seed_entity_ids: Sequence[str],
    *,
    requested_depth: int,
    load_edges: EdgeLoader,
) -> GraphTraversalOutcome:
    """Discover 1–3 hop simple paths with fixed path/entity memory bounds."""

    seeds = tuple(dict.fromkeys(seed_entity_ids))
    if not 1 <= requested_depth <= 3:
        raise ValueError("requested graph depth must be between one and three")
    if len(seeds) > MAX_SEED_ENTITIES:
        raise ValueError("no more than five seed entities may be traversed")
    if not seeds:
        return GraphTraversalOutcome((), 0, 0, False, False)

    visited = set(seeds)
    frontier: dict[str, tuple[_OpenPath, ...]] = {seed: (_OpenPath((seed,), ()),) for seed in seeds}
    retained: dict[str, tuple[GraphTraversalPath, ...]] = {seed: () for seed in seeds}
    actual_depth = 0
    path_limit_hit = False
    visited_limit_hit = False

    for depth in range(1, requested_depth + 1):
        frontier_ids = tuple(
            sorted({path.entity_ids[-1] for paths in frontier.values() for path in paths})
        )
        if not frontier_ids:
            break
        batch = await load_edges(frontier_ids)
        path_limit_hit = path_limit_hit or batch.truncated
        adjacency = _adjacency(batch.edges, frontier_ids)
        next_frontier: dict[str, tuple[_OpenPath, ...]] = {}
        generated_any = False

        for seed in seeds:
            candidates: dict[tuple[object, ...], GraphTraversalPath] = {}
            for open_path in frontier.get(seed, ()):
                endpoint = open_path.entity_ids[-1]
                for edge in adjacency.get(endpoint, ()):
                    neighbor = edge.other_endpoint(endpoint)
                    if neighbor is None or neighbor in open_path.entity_ids:
                        continue
                    if neighbor not in visited:
                        if len(visited) >= MAX_VISITED_ENTITIES:
                            visited_limit_hit = True
                            continue
                        visited.add(neighbor)
                    path = GraphTraversalPath(
                        seed_entity_id=seed,
                        entity_ids=(*open_path.entity_ids, neighbor),
                        relations=(*open_path.relations, edge.relation),
                    )
                    candidates[_traversal_path_key(path)] = path

            ordered_candidates = tuple(sorted(candidates.values(), key=_path_priority))
            if len(ordered_candidates) > MAX_PATHS_PER_SEED:
                path_limit_hit = True
            selected = ordered_candidates[:MAX_PATHS_PER_SEED]
            if selected:
                generated_any = True
                next_frontier[seed] = tuple(
                    _OpenPath(path.entity_ids, path.relations) for path in selected
                )
                combined = {
                    _traversal_path_key(path): path for path in (*retained[seed], *selected)
                }
                if len(combined) > MAX_PATHS_PER_SEED:
                    path_limit_hit = True
                retained[seed] = tuple(
                    sorted(combined.values(), key=_path_priority)[:MAX_PATHS_PER_SEED]
                )

        if not generated_any:
            break
        actual_depth = depth
        frontier = next_frontier

    paths = tuple(path for seed in seeds for path in sorted(retained[seed], key=_path_priority))
    return GraphTraversalOutcome(
        paths=paths,
        visited_entity_count=len(visited),
        actual_depth=actual_depth,
        path_limit_hit=path_limit_hit,
        visited_limit_hit=visited_limit_hit,
    )


def rank_graph_chunks(
    chunks: Sequence[RecoveredGraphChunk],
    *,
    matched_seed_ids: Sequence[str],
    top_k: int,
) -> tuple[RetrievalHit, ...]:
    """Apply the exact ADR-014 V0 score and canonical-ID tie-break."""

    seeds = tuple(dict.fromkeys(matched_seed_ids))
    if not 1 <= top_k <= 50:
        raise ValueError("graph top-k must be between one and fifty")
    if not seeds:
        return ()

    unranked: list[RetrievalHit] = []
    for chunk in chunks:
        unique_paths = {
            _traversal_path_key(path): path for path in chunk.paths if path.seed_entity_id in seeds
        }
        if not unique_paths:
            continue
        paths = tuple(unique_paths.values())
        contributing_seeds = {path.seed_entity_id for path in paths}
        normalized_seed_overlap = len(contributing_seeds) / len(seeds)
        scored_paths = tuple(
            _scored_path(path, normalized_seed_overlap=normalized_seed_overlap) for path in paths
        )
        ordered_paths = tuple(
            sorted(
                scored_paths,
                key=lambda path: (-path.score, path.entity_ids, _graph_path_relation_key(path)),
            )
        )
        best_path = ordered_paths[0]
        mean_confidence = sum(item.confidence for item in best_path.relations) / len(
            best_path.relations
        )
        inverse_hop = 1.0 / best_path.hop_count
        unranked.append(
            RetrievalHit(
                canonical_chunk_id=chunk.canonical_chunk_id,
                document_id=chunk.document_id,
                text=chunk.text,
                score=best_path.score,
                source="graph",
                entity_ids=tuple(sorted(set(chunk.entity_ids))),
                paths=ordered_paths,
                metadata={
                    "graph_score_version": GRAPH_SCORE_VERSION,
                    "normalized_seed_overlap": normalized_seed_overlap,
                    "inverse_hop_count": inverse_hop,
                    "mean_relation_confidence": mean_confidence,
                    "seed_overlap_contribution": 0.45 * normalized_seed_overlap,
                    "hop_contribution": 0.35 * inverse_hop,
                    "confidence_contribution": 0.20 * mean_confidence,
                    "matched_seed_count": len(seeds),
                    "contributing_seed_count": len(contributing_seeds),
                },
            )
        )

    ordered_hits = sorted(
        unranked,
        key=lambda hit: (-hit.score, hit.canonical_chunk_id),
    )[:top_k]
    return tuple(hit.model_copy(update={"rank": rank}) for rank, hit in enumerate(ordered_hits, 1))


async def execute_graph_search(
    *,
    backend: GraphExecutionBackend,
    query_analysis: QueryAnalysis,
    plan: PlanDefinition,
    corpus_version: str,
    deadline: Deadline,
) -> GraphExecution:
    """Execute a graph branch and map its typed terminal state to graph-only semantics."""

    execution = await execute_graph_branch(
        backend=backend,
        query_analysis=query_analysis,
        plan=plan,
        corpus_version=corpus_version,
        deadline=deadline,
    )
    if execution.result.status is BranchStatus.TIMED_OUT:
        raise RAGPlanError(
            ErrorCode.DEADLINE_EXCEEDED,
            "graph retrieval deadline exceeded",
            timeout_origin=execution.result.timeout_origin,
        )
    if execution.result.status is BranchStatus.FAILED:
        raise RAGPlanError(
            execution.result.error_code or ErrorCode.RETRIEVAL_FAILED,
            "graph retrieval failed",
        )
    if execution.result.status is not BranchStatus.SUCCEEDED or execution.trace is None:
        raise RAGPlanError(ErrorCode.RETRIEVAL_FAILED, "graph retrieval failed")
    return GraphExecution(
        hits=execution.result.hits,
        trace=execution.trace,
        latency_ms=execution.result.latency_ms or 0.0,
    )


async def execute_graph_branch(
    *,
    backend: GraphExecutionBackend,
    query_analysis: QueryAnalysis,
    plan: PlanDefinition,
    corpus_version: str,
    deadline: Deadline,
) -> GraphBranchExecution:
    """Always finish with a typed graph branch state, including timeout/error."""

    started_ns = deadline.clock.now_ns()
    if deadline.remaining_seconds(reserve_finalization=True) <= 0:
        return GraphBranchExecution(
            BranchResult(
                branch=BranchKind.GRAPH,
                status=BranchStatus.TIMED_OUT,
                latency_ms=0.0,
                cancellation_reason=CancellationReason.APPLICATION_DEADLINE,
                failure_origin=FailureOrigin.APPLICATION,
                timeout_origin=TimeoutOrigin.APPLICATION_DEADLINE,
            )
        )
    try:
        result = await backend.search(query_analysis, plan, corpus_version, deadline)
    except RAGPlanError as exc:
        finished_ns = deadline.clock.now_ns()
        latency_ms = _elapsed_ms(started_ns, finished_ns)
        if exc.code is ErrorCode.DEADLINE_EXCEEDED:
            origin = exc.timeout_origin or TimeoutOrigin.APPLICATION_DEADLINE
            return GraphBranchExecution(
                BranchResult(
                    branch=BranchKind.GRAPH,
                    status=BranchStatus.TIMED_OUT,
                    latency_ms=latency_ms,
                    cancellation_reason=(
                        CancellationReason.APPLICATION_DEADLINE
                        if origin is TimeoutOrigin.APPLICATION_DEADLINE
                        else None
                    ),
                    failure_origin=(
                        FailureOrigin.APPLICATION
                        if origin is TimeoutOrigin.APPLICATION_DEADLINE
                        else FailureOrigin.BACKEND_NATIVE
                    ),
                    timeout_origin=origin,
                )
            )
        return GraphBranchExecution(
            BranchResult(
                branch=BranchKind.GRAPH,
                status=BranchStatus.FAILED,
                latency_ms=latency_ms,
                error_code=exc.code,
            )
        )
    except Exception:
        finished_ns = deadline.clock.now_ns()
        return GraphBranchExecution(
            BranchResult(
                branch=BranchKind.GRAPH,
                status=BranchStatus.FAILED,
                latency_ms=_elapsed_ms(started_ns, finished_ns),
                error_code=ErrorCode.RETRIEVAL_FAILED,
            )
        )
    finished_ns = deadline.clock.now_ns()
    latency_ms = _elapsed_ms(started_ns, finished_ns)
    if finished_ns >= deadline.branch_cutoff_ns:
        return GraphBranchExecution(
            BranchResult(
                branch=BranchKind.GRAPH,
                status=BranchStatus.TIMED_OUT,
                latency_ms=latency_ms,
                cancellation_reason=CancellationReason.APPLICATION_DEADLINE,
                failure_origin=FailureOrigin.APPLICATION,
                timeout_origin=TimeoutOrigin.APPLICATION_DEADLINE,
            )
        )
    return GraphBranchExecution(
        BranchResult(
            branch=BranchKind.GRAPH,
            status=BranchStatus.SUCCEEDED,
            latency_ms=latency_ms,
            hits=result.hits,
        ),
        trace=result.trace,
    )


def is_p0_english(query: str) -> bool:
    """Conservatively mark non-ASCII-letter queries unsupported in the P0 analyzer."""

    return not any(character.isalpha() and not character.isascii() for character in query)


@dataclass(frozen=True, slots=True)
class _OpenPath:
    entity_ids: tuple[str, ...]
    relations: tuple[Relation, ...]


def _adjacency(
    edges: Sequence[GraphTraversalEdge],
    frontier_ids: Sequence[str],
) -> dict[str, tuple[GraphTraversalEdge, ...]]:
    frontier = set(frontier_ids)
    mutable: dict[str, list[GraphTraversalEdge]] = {}
    for edge in sorted(edges, key=_edge_priority):
        source = edge.relation.source_entity_id
        target = edge.relation.target_entity_id
        if source in frontier:
            mutable.setdefault(source, []).append(edge)
        if target in frontier:
            mutable.setdefault(target, []).append(edge)
    return {key: tuple(value) for key, value in mutable.items()}


def _scored_path(path: GraphTraversalPath, *, normalized_seed_overlap: float) -> GraphPath:
    mean_confidence = sum(item.confidence for item in path.relations) / len(path.relations)
    score = 0.45 * normalized_seed_overlap + 0.35 * (1.0 / path.hop_count) + 0.20 * mean_confidence
    return GraphPath(entity_ids=path.entity_ids, relations=path.relations, score=score)


def _edge_priority(edge: GraphTraversalEdge) -> tuple[object, ...]:
    relation = edge.relation
    return (-relation.confidence, *_relation_key(relation))


def _path_priority(path: GraphTraversalPath) -> tuple[object, ...]:
    mean_confidence = sum(item.confidence for item in path.relations) / len(path.relations)
    return (
        -mean_confidence,
        path.hop_count,
        path.entity_ids,
        tuple(_relation_key(relation) for relation in path.relations),
    )


def _traversal_path_key(path: GraphTraversalPath) -> tuple[object, ...]:
    return (
        path.seed_entity_id,
        path.entity_ids,
        tuple(_relation_key(relation) for relation in path.relations),
    )


def _graph_path_relation_key(path: GraphPath) -> tuple[tuple[str, ...], ...]:
    return tuple(_relation_key(relation) for relation in path.relations)


def _relation_key(relation: Relation) -> tuple[str, ...]:
    return (
        relation.source_entity_id,
        relation.target_entity_id,
        relation.predicate,
        relation.source_chunk_id,
        relation.extraction_rule.value,
    )


def _elapsed_ms(start_ns: int, end_ns: int) -> float:
    return max(0, end_ns - start_ns) / NANOSECONDS_PER_MILLISECOND


__all__ = [
    "GRAPH_ANALYZER_VERSION",
    "GRAPH_SCORE_VERSION",
    "MAX_EDGE_ROWS_PER_DEPTH",
    "MAX_PATHS_PER_SEED",
    "MAX_RECOVERED_CHUNKS",
    "MAX_SEED_ENTITIES",
    "MAX_VISITED_ENTITIES",
    "EdgeBatch",
    "GraphBackendExecution",
    "GraphBranchExecution",
    "GraphExecution",
    "GraphExecutionBackend",
    "GraphQueryAnalyzer",
    "GraphTraversalEdge",
    "GraphTraversalOutcome",
    "GraphTraversalPath",
    "RecoveredGraphChunk",
    "execute_graph_search",
    "execute_graph_branch",
    "is_p0_english",
    "rank_graph_chunks",
    "traverse_bounded",
]
