"""Version-isolated, idempotent Neo4j graph ingestion."""

from __future__ import annotations

import asyncio
import hashlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

from neo4j import AsyncGraphDatabase, Query
from neo4j.exceptions import ConnectionAcquisitionTimeoutError, Neo4jError

from ragplan.backends.base import (
    BackendHealth,
    BackendHealthStatus,
    BackendWriteResult,
    canonical_id_checksum,
)
from ragplan.core.deadline import NANOSECONDS_PER_MILLISECOND, Deadline
from ragplan.core.errors import ErrorCode, RAGPlanError, TimeoutOrigin
from ragplan.core.ids import relation_id
from ragplan.core.models import (
    Chunk,
    Entity,
    EntityMention,
    GraphLimit,
    GraphSeedMatch,
    GraphStageManifest,
    GraphTrace,
    PlanDefinition,
    QueryAnalysis,
    Relation,
    RelationExtractionRule,
)
from ragplan.ingestion.pipeline import graph_content_checksum
from ragplan.retrieval.graph import (
    MAX_EDGE_ROWS_PER_DEPTH,
    MAX_RECOVERED_CHUNKS,
    EdgeBatch,
    GraphBackendExecution,
    GraphTraversalEdge,
    GraphTraversalPath,
    RecoveredGraphChunk,
    rank_graph_chunks,
    traverse_bounded,
)

DEFAULT_DATABASE: Final = "neo4j"
DEFAULT_BATCH_SIZE: Final = 250
DEFAULT_TRANSACTION_TIMEOUT_SECONDS: Final = 30.0
_NAME_PATTERN: Final = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,62}$")
_DEPENDENCY_MESSAGE: Final = "graph storage dependency is unavailable"
_CORPUS_MESSAGE: Final = "graph storage contains inconsistent corpus data"

_SCHEMA_QUERIES: Final = (
    "CREATE CONSTRAINT ragplan_corpus_version IF NOT EXISTS "
    "FOR (node:CorpusVersion) REQUIRE node.version IS UNIQUE",
    "CREATE CONSTRAINT ragplan_document_version_id IF NOT EXISTS "
    "FOR (node:Document) REQUIRE (node.corpus_version, node.id) IS UNIQUE",
    "CREATE CONSTRAINT ragplan_chunk_version_id IF NOT EXISTS "
    "FOR (node:Chunk) REQUIRE (node.corpus_version, node.id) IS UNIQUE",
    "CREATE CONSTRAINT ragplan_entity_id IF NOT EXISTS FOR (node:Entity) REQUIRE node.id IS UNIQUE",
    "CREATE INDEX ragplan_document_corpus IF NOT EXISTS "
    "FOR (node:Document) ON (node.corpus_version)",
    "CREATE INDEX ragplan_chunk_corpus IF NOT EXISTS FOR (node:Chunk) ON (node.corpus_version)",
    "CREATE INDEX ragplan_entity_identity IF NOT EXISTS "
    "FOR (node:Entity) ON (node.normalized_name, node.entity_type)",
)

_UPSERT_MARKER = """
MERGE (version:CorpusVersion {version: $corpus_version})
ON CREATE SET
  version.status = 'writing',
  version.graph_content_checksum = $graph_content_checksum,
  version.extractor_version = $extractor_version
RETURN version.status AS status,
       version.graph_content_checksum AS graph_content_checksum,
       version.extractor_version AS extractor_version
"""
_WRITE_DOCUMENTS = """
UNWIND $rows AS row
MATCH (version:CorpusVersion {version: $corpus_version})
MERGE (document:Document {corpus_version: $corpus_version, id: row.id})
SET document.storage_id = row.storage_id
MERGE (version)-[:CONTAINS_DOCUMENT]->(document)
"""
_WRITE_CHUNKS = """
UNWIND $rows AS row
MATCH (document:Document {corpus_version: $corpus_version, id: row.document_id})
MERGE (chunk:Chunk {corpus_version: $corpus_version, id: row.id})
SET chunk.storage_id = row.storage_id,
    chunk.document_id = row.document_id,
    chunk.position = row.position,
    chunk.text = row.text,
    chunk.token_count = row.token_count
MERGE (document)-[:HAS_CHUNK {corpus_version: $corpus_version}]->(chunk)
"""
_WRITE_ENTITIES = """
UNWIND $rows AS row
MERGE (entity:Entity {id: row.id})
ON CREATE SET entity.name = row.name,
              entity.entity_type = row.entity_type,
              entity.normalized_name = row.normalized_name,
              entity.aliases = row.aliases
ON MATCH SET entity.aliases = reduce(
  aliases = coalesce(entity.aliases, []),
  alias IN row.aliases |
  CASE WHEN alias IN aliases THEN aliases ELSE aliases + alias END
)
"""
_WRITE_MENTIONS = """
UNWIND $rows AS row
MATCH (chunk:Chunk {corpus_version: $corpus_version, id: row.source_chunk_id})
MATCH (entity:Entity {id: row.entity_id})
MERGE (chunk)-[mention:MENTIONS {id: row.id, corpus_version: $corpus_version}]->(entity)
SET mention.raw_text = row.raw_text,
    mention.normalized_name = row.normalized_name,
    mention.entity_type = row.entity_type,
    mention.start_char = row.start_char,
    mention.end_char = row.end_char,
    mention.sentence_start_char = row.sentence_start_char,
    mention.sentence_end_char = row.sentence_end_char
"""
_WRITE_RELATIONS = """
UNWIND $rows AS row
MATCH (source:Entity {id: row.source_entity_id})
MATCH (target:Entity {id: row.target_entity_id})
MERGE (source)-[relation:RELATES_TO {id: row.id, corpus_version: $corpus_version}]->(target)
SET relation.predicate = row.predicate,
    relation.confidence = row.confidence,
    relation.source_chunk_id = row.source_chunk_id,
    relation.extractor_version = row.extractor_version,
    relation.extraction_rule = row.extraction_rule
"""
_SEAL_MARKER = """
MATCH (version:CorpusVersion {version: $corpus_version})
SET version.status = 'graph_staged',
    version.document_count = $document_count,
    version.chunk_count = $chunk_count,
    version.entity_count = $entity_count,
    version.mention_count = $mention_count,
    version.relation_count = $relation_count,
    version.canonical_id_checksum = $canonical_id_checksum
RETURN version.status AS status
"""
_FAIL_MARKER = """
MATCH (version:CorpusVersion {version: $corpus_version})
WHERE version.status <> 'graph_staged'
SET version.status = 'failed'
RETURN version.status AS status
"""
_READ_MARKER = """
MATCH (version:CorpusVersion {version: $corpus_version})
RETURN version.status AS status,
       version.graph_content_checksum AS graph_content_checksum,
       version.extractor_version AS extractor_version,
       version.document_count AS document_count,
       version.chunk_count AS chunk_count,
       version.entity_count AS entity_count,
       version.mention_count AS mention_count,
       version.relation_count AS relation_count,
       version.canonical_id_checksum AS canonical_id_checksum
"""
_READ_CHUNK_IDS = """
MATCH (chunk:Chunk {corpus_version: $corpus_version})
RETURN chunk.id AS canonical_chunk_id
ORDER BY canonical_chunk_id
"""
_COUNT_DOCUMENTS = """
MATCH (document:Document {corpus_version: $corpus_version})
RETURN count(document) AS count
"""
_COUNT_ENTITIES = """
MATCH (:Chunk {corpus_version: $corpus_version})
      -[mention:MENTIONS {corpus_version: $corpus_version}]->(entity:Entity)
RETURN count(DISTINCT entity) AS count
"""
_COUNT_MENTIONS = """
MATCH (:Chunk {corpus_version: $corpus_version})
      -[mention:MENTIONS {corpus_version: $corpus_version}]->(:Entity)
RETURN count(mention) AS count
"""
_COUNT_RELATIONS = """
MATCH (:Entity)-[relation:RELATES_TO {corpus_version: $corpus_version}]->(:Entity)
RETURN count(relation) AS count
"""
_DISCARD_RELATIONS = """
MATCH ()-[relationship]->()
WHERE relationship.corpus_version = $corpus_version
DELETE relationship
"""
_DISCARD_NODES = """
MATCH (node)
WHERE node.corpus_version = $corpus_version OR
      (node:CorpusVersion AND node.version = $corpus_version)
DETACH DELETE node
"""
_DISCARD_ORPHAN_ENTITIES = """
MATCH (entity:Entity)
WHERE NOT (entity)--()
DELETE entity
"""

_LOOKUP_SEEDS = """
UNWIND $seeds AS seed
OPTIONAL MATCH (entity:Entity)
WHERE entity.id = seed.entity_id
  AND entity.normalized_name = seed.normalized_alias
  AND EXISTS {
    MATCH (:Chunk {corpus_version: $corpus_version})
          -[:MENTIONS {corpus_version: $corpus_version}]->(entity)
  }
RETURN seed.position AS position,
       seed.mention_sha256 AS mention_sha256,
       seed.entity_id AS requested_entity_id,
       entity.id AS matched_entity_id
ORDER BY position
"""
_READ_ADJACENT_RELATIONS = """
UNWIND $frontier_ids AS frontier_id
MATCH (frontier:Entity {id: frontier_id})-[relation:RELATES_TO]-(neighbor:Entity)
WHERE relation.corpus_version = $corpus_version
RETURN frontier.id AS frontier_entity_id,
       neighbor.id AS neighbor_entity_id,
       startNode(relation).id AS source_entity_id,
       endNode(relation).id AS target_entity_id,
       relation.predicate AS predicate,
       relation.confidence AS confidence,
       relation.source_chunk_id AS source_chunk_id,
       relation.extractor_version AS extractor_version,
       relation.extraction_rule AS extraction_rule
ORDER BY relation.confidence DESC,
         source_entity_id,
         target_entity_id,
         predicate,
         source_chunk_id,
         extraction_rule
LIMIT $edge_limit
"""
_RECOVER_CHUNKS = """
UNWIND $entity_ids AS entity_id
MATCH (chunk:Chunk {corpus_version: $corpus_version})
      -[:MENTIONS {corpus_version: $corpus_version}]->(entity:Entity {id: entity_id})
WITH chunk, collect(DISTINCT entity.id) AS entity_ids
RETURN chunk.id AS canonical_chunk_id,
       chunk.document_id AS document_id,
       chunk.text AS text,
       entity_ids
ORDER BY canonical_chunk_id
LIMIT $candidate_limit
"""


@dataclass(frozen=True, slots=True)
class Neo4jGraphConfig:
    uri: str = "bolt://127.0.0.1:7687"
    user: str = "neo4j"
    password: str = field(default="", repr=False)
    database: str = DEFAULT_DATABASE
    batch_size: int = DEFAULT_BATCH_SIZE
    transaction_timeout_seconds: float = DEFAULT_TRANSACTION_TIMEOUT_SECONDS

    def __post_init__(self) -> None:
        if not self.uri.strip() or not self.user.strip() or not self.password:
            raise ValueError("Neo4j URI, user, and password are required")
        if _NAME_PATTERN.fullmatch(self.database) is None:
            raise ValueError("invalid Neo4j database name")
        if isinstance(self.batch_size, bool) or self.batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if self.transaction_timeout_seconds <= 0:
            raise ValueError("transaction_timeout_seconds must be positive")


def neo4j_storage_id(corpus_version: str, canonical_id: str) -> str:
    """Create an opaque version-specific storage key without changing canonical IDs."""

    if not corpus_version.strip() or not canonical_id.strip():
        raise ValueError("storage ID components must be non-empty")
    digest = hashlib.sha256(f"{corpus_version}\0{canonical_id}".encode()).hexdigest()
    return f"ragplan-v1-{digest}"


class Neo4jGraphWriter:
    """Write a complete immutable graph stage using parameterized Cypher only."""

    def __init__(self, driver: Any, config: Neo4jGraphConfig) -> None:
        self._driver = driver
        self._config = config
        self._closed = False

    @classmethod
    def connect(cls, config: Neo4jGraphConfig) -> Neo4jGraphWriter:
        driver = AsyncGraphDatabase.driver(
            config.uri,
            auth=(config.user, config.password),
            connection_timeout=config.transaction_timeout_seconds,
            max_transaction_retry_time=0.0,
        )
        return cls(driver, config)

    async def ensure_schema(self) -> None:
        for statement in _SCHEMA_QUERIES:
            await self._execute(statement, {})

    async def write_graph(
        self,
        chunks: Sequence[Chunk],
        entities: Sequence[Entity],
        mentions: Sequence[EntityMention],
        relations: Sequence[Relation],
        corpus_version: str,
        *,
        extractor_version: str,
    ) -> BackendWriteResult:
        manifest = await self.stage_graph(
            chunks,
            entities,
            mentions,
            relations,
            corpus_version,
            extractor_version=extractor_version,
        )
        return BackendWriteResult(
            corpus_version=manifest.corpus_version,
            written_count=manifest.chunk_count,
            canonical_id_checksum=manifest.canonical_id_checksum,
        )

    async def stage_graph(
        self,
        chunks: Sequence[Chunk],
        entities: Sequence[Entity],
        mentions: Sequence[EntityMention],
        relations: Sequence[Relation],
        corpus_version: str,
        *,
        extractor_version: str,
    ) -> GraphStageManifest:
        inputs = _validate_graph_inputs(
            chunks,
            entities,
            mentions,
            relations,
            corpus_version,
            extractor_version,
        )
        content_checksum = graph_content_checksum(
            inputs.chunks,
            inputs.entities,
            inputs.mentions,
            inputs.relations,
            extractor_version=extractor_version,
        )
        await self.ensure_schema()
        try:
            marker_records = await self._execute(
                _UPSERT_MARKER,
                {
                    "corpus_version": corpus_version,
                    "graph_content_checksum": content_checksum,
                    "extractor_version": extractor_version,
                },
            )
            marker = _one_record(marker_records, "corpus version marker")
            if (
                marker.get("graph_content_checksum") != content_checksum
                or marker.get("extractor_version") != extractor_version
            ):
                raise RAGPlanError(ErrorCode.CORPUS_INCONSISTENT, _CORPUS_MESSAGE)
            expected = _manifest_for_inputs(
                inputs,
                database=self._config.database,
                content_checksum=content_checksum,
                extractor_version=extractor_version,
            )
            if marker.get("status") == "graph_staged":
                observed = await self.verify_stage(expected)
                return observed

            await self._write_batches(_WRITE_DOCUMENTS, inputs.documents, corpus_version)
            await self._write_batches(_WRITE_CHUNKS, inputs.chunk_rows, corpus_version)
            await self._write_batches(_WRITE_ENTITIES, inputs.entity_rows, corpus_version)
            await self._write_batches(_WRITE_MENTIONS, inputs.mention_rows, corpus_version)
            await self._write_batches(_WRITE_RELATIONS, inputs.relation_rows, corpus_version)
            await self._execute(
                _SEAL_MARKER,
                {
                    "corpus_version": corpus_version,
                    "document_count": expected.document_count,
                    "chunk_count": expected.chunk_count,
                    "entity_count": expected.entity_count,
                    "mention_count": expected.mention_count,
                    "relation_count": expected.relation_count,
                    "canonical_id_checksum": expected.canonical_id_checksum,
                },
            )
            return await self.verify_stage(expected)
        except RAGPlanError:
            await self._mark_failed(corpus_version)
            raise
        except Exception as exc:
            await self._mark_failed(corpus_version)
            raise RAGPlanError(ErrorCode.DEPENDENCY_UNAVAILABLE, _DEPENDENCY_MESSAGE) from exc

    async def verify_stage(self, manifest: GraphStageManifest) -> GraphStageManifest:
        if manifest.database != self._config.database:
            raise RAGPlanError(ErrorCode.CORPUS_INCONSISTENT, _CORPUS_MESSAGE)
        try:
            marker = _one_record(
                await self._execute(
                    _READ_MARKER,
                    {"corpus_version": manifest.corpus_version},
                ),
                "corpus version marker",
            )
            chunk_records = await self._execute(
                _READ_CHUNK_IDS,
                {"corpus_version": manifest.corpus_version},
            )
            chunk_ids = tuple(
                _required_string(item, "canonical_chunk_id") for item in chunk_records
            )
            observed = {
                "status": marker.get("status"),
                "graph_content_checksum": marker.get("graph_content_checksum"),
                "extractor_version": marker.get("extractor_version"),
                "document_count": await self._count(_COUNT_DOCUMENTS, manifest.corpus_version),
                "chunk_count": len(chunk_ids),
                "entity_count": await self._count(_COUNT_ENTITIES, manifest.corpus_version),
                "mention_count": await self._count(_COUNT_MENTIONS, manifest.corpus_version),
                "relation_count": await self._count(_COUNT_RELATIONS, manifest.corpus_version),
                "canonical_id_checksum": canonical_id_checksum(chunk_ids),
            }
        except RAGPlanError:
            raise
        except Exception as exc:
            raise RAGPlanError(ErrorCode.DEPENDENCY_UNAVAILABLE, _DEPENDENCY_MESSAGE) from exc
        expected = {
            "status": manifest.status,
            "graph_content_checksum": manifest.graph_content_checksum,
            "extractor_version": manifest.extractor_version,
            "document_count": manifest.document_count,
            "chunk_count": manifest.chunk_count,
            "entity_count": manifest.entity_count,
            "mention_count": manifest.mention_count,
            "relation_count": manifest.relation_count,
            "canonical_id_checksum": manifest.canonical_id_checksum,
        }
        if observed != expected:
            raise RAGPlanError(ErrorCode.CORPUS_INCONSISTENT, _CORPUS_MESSAGE)
        return manifest

    async def discard_version(self, corpus_version: str) -> None:
        """Delete one explicit inactive/failed graph version; never called implicitly."""

        if not corpus_version.strip():
            raise ValueError("corpus_version must not be blank")
        try:
            await self._execute(_DISCARD_RELATIONS, {"corpus_version": corpus_version})
            await self._execute(_DISCARD_NODES, {"corpus_version": corpus_version})
            await self._execute(_DISCARD_ORPHAN_ENTITIES, {})
        except Exception as exc:
            raise RAGPlanError(ErrorCode.DEPENDENCY_UNAVAILABLE, _DEPENDENCY_MESSAGE) from exc

    async def health(self) -> BackendHealth:
        try:
            await self._driver.verify_connectivity()
        except Exception:
            return BackendHealth(
                BackendHealthStatus.UNAVAILABLE,
                "graph storage health check failed",
            )
        return BackendHealth(BackendHealthStatus.HEALTHY)

    async def close(self) -> None:
        if not self._closed:
            self._closed = True
            await self._driver.close()

    async def _write_batches(
        self,
        query: str,
        rows: Sequence[Mapping[str, object]],
        corpus_version: str,
    ) -> None:
        for offset in range(0, len(rows), self._config.batch_size):
            await self._execute(
                query,
                {
                    "corpus_version": corpus_version,
                    "rows": list(rows[offset : offset + self._config.batch_size]),
                },
            )

    async def _count(self, query: str, corpus_version: str) -> int:
        record = _one_record(
            await self._execute(query, {"corpus_version": corpus_version}),
            "graph count",
        )
        value = record.get("count")
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise RAGPlanError(ErrorCode.CORPUS_INCONSISTENT, _CORPUS_MESSAGE)
        return value

    async def _mark_failed(self, corpus_version: str) -> None:
        try:
            await self._execute(_FAIL_MARKER, {"corpus_version": corpus_version})
        except Exception:
            return

    async def _execute(
        self,
        statement: str,
        parameters: Mapping[str, object],
    ) -> list[Mapping[str, object]]:
        query = Query(statement, timeout=self._config.transaction_timeout_seconds)
        result = await self._driver.execute_query(
            query,
            parameters_=dict(parameters),
            database_=self._config.database,
        )
        records = result[0]
        return [dict(record) for record in records]


class Neo4jGraphBackend:
    """Read one active graph corpus with exact seeds and bounded Python traversal."""

    def __init__(self, driver: Any, config: Neo4jGraphConfig) -> None:
        self._driver = driver
        self._config = config
        self._closed = False

    @classmethod
    def connect(cls, config: Neo4jGraphConfig) -> Neo4jGraphBackend:
        driver = AsyncGraphDatabase.driver(
            config.uri,
            auth=(config.user, config.password),
            connection_timeout=config.transaction_timeout_seconds,
            max_transaction_retry_time=0.0,
        )
        return cls(driver, config)

    async def require_active_corpus(
        self,
        *,
        corpus_version: str,
        chunk_count: int,
        canonical_id_checksum: str,
        extractor_version: str,
    ) -> None:
        """Validate the immutable graph marker against active-manifest evidence."""

        try:
            marker = _one_record(
                await self._execute(
                    _READ_MARKER,
                    {"corpus_version": corpus_version},
                    deadline=None,
                ),
                "corpus version marker",
            )
        except RAGPlanError:
            raise
        except Exception as exc:
            raise RAGPlanError(ErrorCode.DEPENDENCY_UNAVAILABLE, _DEPENDENCY_MESSAGE) from exc
        if (
            marker.get("status") != "graph_staged"
            or marker.get("chunk_count") != chunk_count
            or marker.get("canonical_id_checksum") != canonical_id_checksum
            or marker.get("extractor_version") != extractor_version
        ):
            raise RAGPlanError(ErrorCode.CORPUS_INCONSISTENT, _CORPUS_MESSAGE)

    async def search(
        self,
        query_analysis: QueryAnalysis,
        plan: PlanDefinition,
        corpus_version: str,
        deadline: Deadline,
    ) -> GraphBackendExecution:
        if not plan.graph_enabled or not 1 <= plan.graph_depth <= 3:
            raise RAGPlanError(
                ErrorCode.PLAN_INVARIANT_VIOLATION,
                "graph retrieval requires an enabled bounded graph plan",
                retryable=False,
            )
        if len(query_analysis.seed_entity_ids) > 5:
            raise RAGPlanError(
                ErrorCode.PLAN_INVARIANT_VIOLATION,
                "graph retrieval cannot execute more than five seeds",
                retryable=False,
            )
        if not query_analysis.language_supported:
            raise RAGPlanError(
                ErrorCode.MODE_UNAVAILABLE,
                "graph mode is unavailable for unsupported query language",
                retryable=False,
            )

        seed_started_ns = deadline.clock.now_ns()
        seed_matches = await self._lookup_seeds(query_analysis, corpus_version, deadline)
        seed_finished_ns = deadline.clock.now_ns()
        matched_seed_ids = tuple(
            match.matched_entity_id for match in seed_matches if match.matched_entity_id is not None
        )

        traversal_started_ns = seed_finished_ns

        async def load_edges(frontier_ids: tuple[str, ...]) -> EdgeBatch:
            records = await self._execute(
                _READ_ADJACENT_RELATIONS,
                {
                    "corpus_version": corpus_version,
                    "frontier_ids": list(frontier_ids),
                    "edge_limit": MAX_EDGE_ROWS_PER_DEPTH,
                },
                deadline=deadline,
            )
            truncated = len(records) >= MAX_EDGE_ROWS_PER_DEPTH
            usable = records[: MAX_EDGE_ROWS_PER_DEPTH - 1]
            unique: dict[tuple[str, ...], GraphTraversalEdge] = {}
            for record in usable:
                edge = _edge_from_record(record)
                key = (
                    edge.relation.source_entity_id,
                    edge.relation.target_entity_id,
                    edge.relation.predicate,
                    edge.relation.source_chunk_id,
                    edge.relation.extraction_rule.value,
                )
                unique[key] = edge
            return EdgeBatch(tuple(unique[key] for key in sorted(unique)), truncated)

        traversal = await traverse_bounded(
            matched_seed_ids,
            requested_depth=plan.graph_depth,
            load_edges=load_edges,
        )
        traversal_finished_ns = deadline.clock.now_ns()

        recovery_started_ns = traversal_finished_ns
        reached_ids = tuple(
            sorted({entity_id for path in traversal.paths for entity_id in path.entity_ids[1:]})
        )
        candidate_records: list[Mapping[str, object]] = []
        chunk_limit_hit = False
        if reached_ids:
            candidate_records = await self._execute(
                _RECOVER_CHUNKS,
                {
                    "corpus_version": corpus_version,
                    "entity_ids": list(reached_ids),
                    "candidate_limit": MAX_RECOVERED_CHUNKS + 1,
                },
                deadline=deadline,
            )
            chunk_limit_hit = len(candidate_records) > MAX_RECOVERED_CHUNKS
            candidate_records = candidate_records[:MAX_RECOVERED_CHUNKS]
        recovered = _recovered_chunks(candidate_records, traversal.paths)
        recovery_finished_ns = deadline.clock.now_ns()

        ranking_started_ns = recovery_finished_ns
        top_k = min(50, max(plan.graph_top_k, query_analysis.features.final_top_k))
        hits = rank_graph_chunks(recovered, matched_seed_ids=matched_seed_ids, top_k=top_k)
        ranking_finished_ns = deadline.clock.now_ns()
        if ranking_finished_ns >= deadline.branch_cutoff_ns:
            raise RAGPlanError(ErrorCode.DEADLINE_EXCEEDED, "graph retrieval deadline exceeded")

        limit_hits: list[GraphLimit] = []
        if query_analysis.seed_limit_hit:
            limit_hits.append(GraphLimit.SEEDS)
        if traversal.path_limit_hit:
            limit_hits.append(GraphLimit.PATHS)
        if traversal.visited_limit_hit:
            limit_hits.append(GraphLimit.VISITED_ENTITIES)
        if chunk_limit_hit:
            limit_hits.append(GraphLimit.RECOVERED_CHUNKS)
        trace = GraphTrace(
            seed_matches=seed_matches,
            requested_depth=plan.graph_depth,
            actual_depth=traversal.actual_depth,
            visited_entity_count=traversal.visited_entity_count,
            path_count=len(traversal.paths),
            recovered_chunk_count=len(recovered),
            limit_hits=tuple(limit_hits),
            seed_lookup_latency_ms=_latency_ms(seed_started_ns, seed_finished_ns),
            traversal_latency_ms=_latency_ms(traversal_started_ns, traversal_finished_ns),
            recovery_latency_ms=_latency_ms(recovery_started_ns, recovery_finished_ns),
            ranking_latency_ms=_latency_ms(ranking_started_ns, ranking_finished_ns),
        )
        return GraphBackendExecution(hits=hits, trace=trace)

    async def health(self) -> BackendHealth:
        try:
            await self._driver.verify_connectivity()
        except Exception:
            return BackendHealth(
                BackendHealthStatus.UNAVAILABLE,
                "graph storage health check failed",
            )
        return BackendHealth(BackendHealthStatus.HEALTHY)

    async def close(self) -> None:
        if not self._closed:
            self._closed = True
            await self._driver.close()

    async def _lookup_seeds(
        self,
        analysis: QueryAnalysis,
        corpus_version: str,
        deadline: Deadline,
    ) -> tuple[GraphSeedMatch, ...]:
        if not analysis.seed_entity_ids:
            return ()
        seed_rows = [
            {
                "position": position,
                "normalized_alias": alias,
                "entity_id": entity_id,
                "mention_sha256": hashlib.sha256(alias.encode("utf-8")).hexdigest(),
            }
            for position, (alias, entity_id) in enumerate(
                zip(
                    analysis.seed_entity_mentions,
                    analysis.seed_entity_ids,
                    strict=True,
                )
            )
        ]
        records = await self._execute(
            _LOOKUP_SEEDS,
            {"corpus_version": corpus_version, "seeds": seed_rows},
            deadline=deadline,
        )
        if len(records) != len(seed_rows):
            raise RAGPlanError(ErrorCode.CORPUS_INCONSISTENT, _CORPUS_MESSAGE)
        matches: list[GraphSeedMatch] = []
        for record in records:
            requested_id = _required_string(record, "requested_entity_id")
            mention_sha256 = _required_string(record, "mention_sha256")
            matched_value = record.get("matched_entity_id")
            if matched_value is not None and not isinstance(matched_value, str):
                raise RAGPlanError(ErrorCode.CORPUS_INCONSISTENT, _CORPUS_MESSAGE)
            matches.append(
                GraphSeedMatch(
                    mention_sha256=mention_sha256,
                    requested_entity_id=requested_id,
                    matched_entity_id=matched_value,
                    lookup_score=1.0 if matched_value is not None else 0.0,
                )
            )
        return tuple(matches)

    async def _execute(
        self,
        statement: str,
        parameters: Mapping[str, object],
        *,
        deadline: Deadline | None,
    ) -> list[Mapping[str, object]]:
        configured_timeout = self._config.transaction_timeout_seconds
        if deadline is None:
            query_timeout = configured_timeout
            timeout_seconds = configured_timeout
        else:
            remaining = deadline.remaining_seconds(reserve_finalization=True)
            if remaining <= 0:
                raise RAGPlanError(
                    ErrorCode.DEADLINE_EXCEEDED,
                    "graph retrieval deadline exceeded",
                    timeout_origin=TimeoutOrigin.APPLICATION_DEADLINE,
                )
            query_timeout = min(configured_timeout, remaining)
            timeout_seconds = remaining
        query = Query(statement, timeout=query_timeout)
        try:
            async with asyncio.timeout(timeout_seconds):
                result = await self._driver.execute_query(
                    query,
                    parameters_=dict(parameters),
                    database_=self._config.database,
                )
        except ConnectionAcquisitionTimeoutError as exc:
            raise RAGPlanError(
                ErrorCode.DEADLINE_EXCEEDED,
                "graph backend client timeout",
                timeout_origin=TimeoutOrigin.BACKEND_CLIENT,
            ) from exc
        except Neo4jError as exc:
            if "timeout" in str(exc.code).casefold():
                raise RAGPlanError(
                    ErrorCode.DEADLINE_EXCEEDED,
                    "graph backend client timeout",
                    timeout_origin=TimeoutOrigin.BACKEND_CLIENT,
                ) from exc
            raise RAGPlanError(ErrorCode.DEPENDENCY_UNAVAILABLE, _DEPENDENCY_MESSAGE) from exc
        except TimeoutError as exc:
            raise RAGPlanError(
                ErrorCode.DEADLINE_EXCEEDED,
                "graph retrieval deadline exceeded",
                timeout_origin=TimeoutOrigin.APPLICATION_DEADLINE,
            ) from exc
        except RAGPlanError:
            raise
        except Exception as exc:
            raise RAGPlanError(ErrorCode.DEPENDENCY_UNAVAILABLE, _DEPENDENCY_MESSAGE) from exc
        return [dict(record) for record in result[0]]


@dataclass(frozen=True, slots=True)
class _GraphInputs:
    corpus_version: str
    chunks: tuple[Chunk, ...]
    entities: tuple[Entity, ...]
    mentions: tuple[EntityMention, ...]
    relations: tuple[Relation, ...]
    documents: tuple[Mapping[str, object], ...]
    chunk_rows: tuple[Mapping[str, object], ...]
    entity_rows: tuple[Mapping[str, object], ...]
    mention_rows: tuple[Mapping[str, object], ...]
    relation_rows: tuple[Mapping[str, object], ...]


def _edge_from_record(record: Mapping[str, object]) -> GraphTraversalEdge:
    try:
        confidence_value = record["confidence"]
        if isinstance(confidence_value, bool) or not isinstance(confidence_value, (int, float)):
            raise TypeError("relation confidence must be numeric")
        relation = Relation(
            source_entity_id=_required_string(record, "source_entity_id"),
            target_entity_id=_required_string(record, "target_entity_id"),
            predicate=_required_string(record, "predicate"),
            confidence=float(confidence_value),
            source_chunk_id=_required_string(record, "source_chunk_id"),
            extractor_version=_required_string(record, "extractor_version"),
            extraction_rule=RelationExtractionRule(_required_string(record, "extraction_rule")),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise RAGPlanError(ErrorCode.CORPUS_INCONSISTENT, _CORPUS_MESSAGE) from exc
    return GraphTraversalEdge(relation)


def _recovered_chunks(
    records: Sequence[Mapping[str, object]],
    paths: Sequence[GraphTraversalPath],
) -> tuple[RecoveredGraphChunk, ...]:
    recovered: list[RecoveredGraphChunk] = []
    for record in records:
        canonical_chunk = _required_string(record, "canonical_chunk_id")
        document_id = _required_string(record, "document_id")
        text = _required_string(record, "text")
        entity_values = record.get("entity_ids")
        if not isinstance(entity_values, Sequence) or isinstance(entity_values, (str, bytes)):
            raise RAGPlanError(ErrorCode.CORPUS_INCONSISTENT, _CORPUS_MESSAGE)
        entity_ids = tuple(sorted({_required_sequence_string(value) for value in entity_values}))
        relevant_paths = tuple(path for path in paths if set(path.entity_ids[1:]) & set(entity_ids))
        if not relevant_paths:
            continue
        recovered.append(
            RecoveredGraphChunk(
                canonical_chunk_id=canonical_chunk,
                document_id=document_id,
                text=text,
                entity_ids=entity_ids,
                paths=relevant_paths,
            )
        )
    return tuple(recovered)


def _required_sequence_string(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise RAGPlanError(ErrorCode.CORPUS_INCONSISTENT, _CORPUS_MESSAGE)
    return value


def _latency_ms(start_ns: int, end_ns: int) -> float:
    return max(0, end_ns - start_ns) / NANOSECONDS_PER_MILLISECOND


def _validate_graph_inputs(
    chunks: Sequence[Chunk],
    entities: Sequence[Entity],
    mentions: Sequence[EntityMention],
    relations: Sequence[Relation],
    corpus_version: str,
    extractor_version: str,
) -> _GraphInputs:
    if not corpus_version.strip() or not extractor_version.strip():
        raise ValueError("corpus_version and extractor_version must not be blank")
    chunk_tuple = tuple(chunks)
    entity_tuple = tuple(entities)
    mention_tuple = tuple(mentions)
    relation_tuple = tuple(relations)
    if not chunk_tuple:
        raise ValueError("a graph stage requires at least one canonical chunk")
    chunk_ids = {chunk.canonical_chunk_id for chunk in chunk_tuple}
    entity_ids = {entity.id for entity in entity_tuple}
    if len(chunk_ids) != len(chunk_tuple) or len(entity_ids) != len(entity_tuple):
        raise ValueError("graph inputs contain duplicate canonical identities")
    if any(chunk.corpus_version != corpus_version for chunk in chunk_tuple):
        raise ValueError("every chunk must match the requested corpus_version")
    if any(
        mention.source_chunk_id not in chunk_ids or mention.entity_id not in entity_ids
        for mention in mention_tuple
    ):
        raise ValueError("every mention must reference an input chunk and entity")
    mention_ids = {mention.id for mention in mention_tuple}
    if len(mention_ids) != len(mention_tuple):
        raise ValueError("graph inputs contain duplicate mention identities")
    if any(
        relation.source_chunk_id not in chunk_ids
        or relation.source_entity_id not in entity_ids
        or relation.target_entity_id not in entity_ids
        or relation.extractor_version != extractor_version
        for relation in relation_tuple
    ):
        raise ValueError("every relation must reference graph inputs and extractor version")
    mentioned_by_chunk: dict[str, set[str]] = {}
    for mention in mention_tuple:
        mentioned_by_chunk.setdefault(mention.source_chunk_id, set()).add(mention.entity_id)
    if any(
        {
            relation.source_entity_id,
            relation.target_entity_id,
        }
        - mentioned_by_chunk.get(relation.source_chunk_id, set())
        for relation in relation_tuple
    ):
        raise ValueError("relation endpoints must be mentioned in their source chunk")
    relation_ids = {
        relation_id(
            relation.source_entity_id,
            relation.target_entity_id,
            relation.predicate,
            relation.source_chunk_id,
            relation.extraction_rule.value,
        )
        for relation in relation_tuple
    }
    if len(relation_ids) != len(relation_tuple):
        raise ValueError("graph inputs contain duplicate relation identities")
    referenced_entities = {mention.entity_id for mention in mention_tuple} | {
        endpoint
        for relation in relation_tuple
        for endpoint in (relation.source_entity_id, relation.target_entity_id)
    }
    if referenced_entities != entity_ids:
        raise ValueError("every staged entity must have mention or relation provenance")
    document_ids = sorted({chunk.document_id for chunk in chunk_tuple})
    documents = tuple(
        {
            "id": document_id,
            "storage_id": neo4j_storage_id(corpus_version, document_id),
        }
        for document_id in document_ids
    )
    chunk_rows = tuple(
        {
            "id": chunk.id,
            "storage_id": neo4j_storage_id(corpus_version, chunk.id),
            "document_id": chunk.document_id,
            "position": chunk.position,
            "text": chunk.text,
            "token_count": chunk.token_count,
        }
        for chunk in sorted(chunk_tuple, key=lambda item: item.id)
    )
    entity_rows = tuple(
        {
            "id": entity.id,
            "name": entity.name,
            "entity_type": entity.entity_type.value,
            "normalized_name": entity.normalized_name,
            "aliases": list(entity.aliases),
        }
        for entity in sorted(entity_tuple, key=lambda item: item.id)
    )
    mention_rows = tuple(
        mention.model_dump(mode="json")
        for mention in sorted(mention_tuple, key=lambda item: item.id)
    )
    relation_rows = tuple(
        {
            "id": str(
                relation_id(
                    relation.source_entity_id,
                    relation.target_entity_id,
                    relation.predicate,
                    relation.source_chunk_id,
                    relation.extraction_rule.value,
                )
            ),
            **relation.model_dump(mode="json"),
        }
        for relation in sorted(
            relation_tuple,
            key=lambda item: (
                item.source_chunk_id,
                item.source_entity_id,
                item.target_entity_id,
                item.predicate,
                item.extraction_rule,
            ),
        )
    )
    return _GraphInputs(
        corpus_version=corpus_version,
        chunks=chunk_tuple,
        entities=entity_tuple,
        mentions=mention_tuple,
        relations=relation_tuple,
        documents=documents,
        chunk_rows=chunk_rows,
        entity_rows=entity_rows,
        mention_rows=mention_rows,
        relation_rows=relation_rows,
    )


def _manifest_for_inputs(
    inputs: _GraphInputs,
    *,
    database: str,
    content_checksum: str,
    extractor_version: str,
) -> GraphStageManifest:
    return GraphStageManifest(
        corpus_version=inputs.corpus_version,
        database=database,
        document_count=len(inputs.documents),
        chunk_count=len(inputs.chunks),
        entity_count=len(inputs.entities),
        mention_count=len(inputs.mentions),
        relation_count=len(inputs.relations),
        canonical_id_checksum=canonical_id_checksum(tuple(item.id for item in inputs.chunks)),
        graph_content_checksum=content_checksum,
        extractor_version=extractor_version,
    )


def _one_record(
    records: Sequence[Mapping[str, object]],
    label: str,
) -> Mapping[str, object]:
    if len(records) != 1:
        raise RAGPlanError(
            ErrorCode.CORPUS_INCONSISTENT,
            f"{label} is missing or duplicated",
        )
    return records[0]


def _required_string(record: Mapping[str, object], field_name: str) -> str:
    value = record.get(field_name)
    if not isinstance(value, str) or not value:
        raise RAGPlanError(ErrorCode.CORPUS_INCONSISTENT, _CORPUS_MESSAGE)
    return value
