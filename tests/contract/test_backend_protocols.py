from __future__ import annotations

import inspect
from collections.abc import Sequence

import pytest

from ragplan.backends.base import BackendHealth, BackendHealthStatus, BackendWriteResult
from ragplan.backends.graph.base import GraphBackend, GraphIngestionWriter
from ragplan.backends.vector.base import VectorBackend, VectorIngestionWriter
from ragplan.core.deadline import Deadline
from ragplan.core.models import Chunk, Entity, PlanDefinition, QueryAnalysis, Relation, RetrievalHit

pytestmark = [pytest.mark.unit, pytest.mark.contract]


class _VectorBackend:
    async def search(
        self,
        embedding: Sequence[float],
        top_k: int,
        corpus_version: str,
        deadline: Deadline,
    ) -> Sequence[RetrievalHit]:
        return ()

    async def health(self) -> BackendHealth:
        return BackendHealth(BackendHealthStatus.HEALTHY)

    async def close(self) -> None:
        return None


def test_runtime_vector_protocol_is_structural() -> None:
    assert isinstance(_VectorBackend(), VectorBackend)
    assert not isinstance(_VectorBackend(), VectorIngestionWriter)


def test_backend_search_contracts_share_absolute_deadline() -> None:
    vector_parameters = inspect.signature(VectorBackend.search).parameters
    graph_parameters = inspect.signature(GraphBackend.search).parameters

    assert list(vector_parameters) == [
        "self",
        "embedding",
        "top_k",
        "corpus_version",
        "deadline",
    ]
    assert list(graph_parameters) == [
        "self",
        "query_analysis",
        "plan",
        "corpus_version",
        "deadline",
    ]
    assert "timeout_ms" not in vector_parameters
    assert "timeout_ms" not in graph_parameters


def test_ingestion_writers_are_separate_protocols() -> None:
    assert "search" not in VectorIngestionWriter.__dict__
    assert "search" not in GraphIngestionWriter.__dict__
    assert "write_chunks" not in VectorBackend.__dict__
    assert "write_graph" not in GraphBackend.__dict__


def test_backend_write_result_validates_reconciliation_fields() -> None:
    result = BackendWriteResult(
        corpus_version="corpus-v1",
        written_count=3,
        canonical_id_checksum="a" * 64,
    )

    assert result.written_count == 3
    with pytest.raises(ValueError, match="SHA-256"):
        BackendWriteResult("corpus-v1", 3, "not-a-digest")


def _type_import_smoke(
    chunk: Chunk,
    entity: Entity,
    relation: Relation,
    analysis: QueryAnalysis,
    plan: PlanDefinition,
) -> tuple[object, ...]:
    """Keep protocol-domain imports under static type checking."""

    return chunk, entity, relation, analysis, plan
