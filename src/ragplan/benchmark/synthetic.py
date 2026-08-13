"""Deterministic graph-only fixture kept separate from primary benchmark metrics."""

from __future__ import annotations

from collections import Counter
from typing import Annotated, Self

from pydantic import Field, StringConstraints, model_validator

from ragplan.benchmark.contracts import BENCHMARK_SCHEMA_VERSION, canonical_sha256
from ragplan.core.models import FrozenModel, NonEmptyString, Sha256Hex

SYNTHETIC_GRAPH_ID = "adaptive_rag_synthetic_graph_v1"
SyntheticQueryId = Annotated[str, StringConstraints(pattern=r"^synthetic_graph_v1:q[0-9]{3}$")]


class SyntheticEdge(FrozenModel):
    source_entity: NonEmptyString
    relation: NonEmptyString
    target_entity: NonEmptyString


class SyntheticGraphQuery(FrozenModel):
    query_id: SyntheticQueryId
    question: NonEmptyString
    hop_count: int = Field(ge=1, le=3)
    start_entity: NonEmptyString
    answer_entity: NonEmptyString
    gold_path: tuple[SyntheticEdge, ...] = Field(min_length=1, max_length=3)
    relevant_chunk_ids: tuple[NonEmptyString, ...] = Field(min_length=1, max_length=3)

    @model_validator(mode="after")
    def _validate_path(self) -> Self:
        if len(self.gold_path) != self.hop_count:
            raise ValueError("gold path length must equal hop_count")
        if len(self.relevant_chunk_ids) != self.hop_count:
            raise ValueError("one relevant evidence chunk is required per hop")
        if self.gold_path[0].source_entity != self.start_entity:
            raise ValueError("gold path must start at start_entity")
        if self.gold_path[-1].target_entity != self.answer_entity:
            raise ValueError("gold path must end at answer_entity")
        for left, right in zip(self.gold_path, self.gold_path[1:], strict=False):
            if left.target_entity != right.source_entity:
                raise ValueError("gold path must be connected")
        return self


class SyntheticGraphManifest(FrozenModel):
    schema_version: Annotated[str, StringConstraints(pattern=r"^v1$")] = BENCHMARK_SCHEMA_VERSION
    fixture_id: Annotated[str, StringConstraints(pattern=r"^adaptive_rag_synthetic_graph_v1$")] = (
        SYNTHETIC_GRAPH_ID
    )
    primary_metrics_eligible: bool = False
    edges: tuple[SyntheticEdge, ...] = Field(min_length=1)
    queries: tuple[SyntheticGraphQuery, ...] = Field(min_length=100, max_length=100)
    manifest_sha256: Sha256Hex

    @model_validator(mode="after")
    def _validate_manifest(self) -> Self:
        if self.primary_metrics_eligible:
            raise ValueError("synthetic graph fixture must be excluded from primary metrics")
        if len({query.query_id for query in self.queries}) != 100:
            raise ValueError("synthetic query IDs must be unique")
        counts = Counter(query.hop_count for query in self.queries)
        if counts != Counter({1: 34, 2: 33, 3: 33}):
            raise ValueError("synthetic hop distribution must be 34/33/33")
        if self.manifest_sha256 != _synthetic_sha256(self.edges, self.queries):
            raise ValueError("synthetic manifest hash mismatch")
        return self


def build_synthetic_graph_fixture() -> SyntheticGraphManifest:
    edges: set[tuple[str, str, str]] = set()
    queries: list[SyntheticGraphQuery] = []
    for index in range(100):
        hop_count = (index % 3) + 1
        nodes = tuple(f"entity_{index:03d}_{step}" for step in range(hop_count + 1))
        path: list[SyntheticEdge] = []
        chunk_ids: list[str] = []
        for step in range(hop_count):
            edge = SyntheticEdge(
                source_entity=nodes[step],
                relation=f"REL_STEP_{step + 1}",
                target_entity=nodes[step + 1],
            )
            path.append(edge)
            edges.add((edge.source_entity, edge.relation, edge.target_entity))
            chunk_ids.append(f"synthetic:chunk:q{index:03d}:hop{step + 1}")

        # Deliberate hub and cycles exercise traversal bounds and visited-set handling.
        edges.add((nodes[0], "LINKS_TO_HUB", "entity_shared_hub"))
        edges.add(("entity_shared_hub", "LINKS_FROM_HUB", nodes[0]))
        if hop_count > 1:
            edges.add((nodes[-1], "CYCLES_TO", nodes[0]))
        queries.append(
            SyntheticGraphQuery(
                query_id=f"synthetic_graph_v1:q{index:03d}",
                question=(
                    f"Starting at {nodes[0]}, which entity is reached after "
                    f"{hop_count} gold relation step(s)?"
                ),
                hop_count=hop_count,
                start_entity=nodes[0],
                answer_entity=nodes[-1],
                gold_path=tuple(path),
                relevant_chunk_ids=tuple(chunk_ids),
            )
        )
    # A disconnected component is intentionally unreachable from every query seed.
    edges.add(("entity_disconnected_a", "DISCONNECTED", "entity_disconnected_b"))
    edge_models = tuple(
        SyntheticEdge(source_entity=source, relation=relation, target_entity=target)
        for source, relation, target in sorted(edges)
    )
    query_models = tuple(queries)
    return SyntheticGraphManifest(
        edges=edge_models,
        queries=query_models,
        manifest_sha256=_synthetic_sha256(edge_models, query_models),
    )


def _synthetic_sha256(
    edges: tuple[SyntheticEdge, ...], queries: tuple[SyntheticGraphQuery, ...]
) -> str:
    return canonical_sha256(
        {
            "edges": [edge.model_dump(mode="json") for edge in edges],
            "queries": [query.model_dump(mode="json") for query in queries],
        }
    )
