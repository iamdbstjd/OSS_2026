"""Runtime and ingestion contracts for graph storage."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from ragplan.backends.base import BackendHealth, BackendWriteResult
from ragplan.core.deadline import Deadline
from ragplan.core.models import (
    Chunk,
    Entity,
    EntityMention,
    GraphStageManifest,
    PlanDefinition,
    QueryAnalysis,
    Relation,
)
from ragplan.retrieval.graph import GraphBackendExecution


@runtime_checkable
class GraphBackend(Protocol):
    """Online graph evidence retrieval over one immutable corpus version."""

    async def search(
        self,
        query_analysis: QueryAnalysis,
        plan: PlanDefinition,
        corpus_version: str,
        deadline: Deadline,
    ) -> GraphBackendExecution: ...

    async def health(self) -> BackendHealth: ...

    async def close(self) -> None: ...


@runtime_checkable
class GraphIngestionWriter(Protocol):
    """Offline graph writer, intentionally separate from online search."""

    async def write_graph(
        self,
        chunks: Sequence[Chunk],
        entities: Sequence[Entity],
        mentions: Sequence[EntityMention],
        relations: Sequence[Relation],
        corpus_version: str,
        *,
        extractor_version: str,
    ) -> BackendWriteResult: ...

    async def stage_graph(
        self,
        chunks: Sequence[Chunk],
        entities: Sequence[Entity],
        mentions: Sequence[EntityMention],
        relations: Sequence[Relation],
        corpus_version: str,
        *,
        extractor_version: str,
    ) -> GraphStageManifest: ...

    async def close(self) -> None: ...
