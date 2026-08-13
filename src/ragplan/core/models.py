"""Frozen, validated domain contracts shared by RAGPlan components."""

from __future__ import annotations

import math
from collections.abc import Mapping
from copy import deepcopy
from enum import StrEnum
from types import MappingProxyType
from typing import Annotated, Any, Literal, Self, cast

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    PlainSerializer,
    StringConstraints,
    computed_field,
    field_validator,
    model_validator,
)
from pydantic.types import JsonValue

from ragplan.core.config import (
    DEFAULT_LATENCY_BUDGET_MS,
    DEFAULT_TOP_K,
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL_ID,
    EMBEDDING_MODEL_REVISION,
    MAX_LATENCY_BUDGET_MS,
    MAX_QUERY_CODEPOINTS,
    MAX_TOP_K,
    MIN_LATENCY_BUDGET_MS,
    MIN_QUERY_CODEPOINTS,
    MIN_TOP_K,
)
from ragplan.core.errors import ErrorCode, TimeoutOrigin
from ragplan.core.ids import entity_id, normalize_entity_name

NonEmptyString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Sha256Hex = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
PlanId = Annotated[str, StringConstraints(pattern=r"^P(?:0|[1-9][0-9]*)$")]
QueryString = Annotated[
    str,
    StringConstraints(min_length=MIN_QUERY_CODEPOINTS, max_length=MAX_QUERY_CODEPOINTS),
]
NonNegativeFloat = Annotated[float, Field(ge=0.0)]
UnitFloat = Annotated[float, Field(ge=0.0, le=1.0)]


def _freeze_json_value(value: JsonValue) -> JsonValue:
    if isinstance(value, Mapping):
        frozen = MappingProxyType(
            {str(key): _freeze_json_value(item) for key, item in value.items()}
        )
        return cast(JsonValue, frozen)
    if isinstance(value, (list, tuple)):
        return cast(JsonValue, tuple(_freeze_json_value(item) for item in value))
    return value


def _freeze_json_mapping(value: Mapping[str, JsonValue]) -> Mapping[str, JsonValue]:
    return MappingProxyType({str(key): _freeze_json_value(item) for key, item in value.items()})


def _thaw_json_value(value: object) -> JsonValue:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unsupported frozen JSON value: {type(value).__name__}")


def _serialize_json_mapping(value: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    return {str(key): _thaw_json_value(item) for key, item in value.items()}


def _freeze_float_mapping(value: Mapping[str, float]) -> Mapping[str, float]:
    return MappingProxyType(dict(value))


FrozenJsonMapping = Annotated[
    Mapping[str, JsonValue],
    AfterValidator(_freeze_json_mapping),
    PlainSerializer(_serialize_json_mapping, return_type=dict[str, JsonValue]),
]
FrozenFloatMapping = Annotated[
    Mapping[str, float],
    AfterValidator(_freeze_float_mapping),
    PlainSerializer(dict, return_type=dict[str, float]),
]


class FrozenModel(BaseModel):
    """Stage 1 base: immutable, strict, finite, and closed to extra fields."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
        allow_inf_nan=False,
        validate_default=True,
    )

    def model_copy(self, *, update: Mapping[str, Any] | None = None, deep: bool = False) -> Self:
        """Copy through validation so updates cannot bypass cross-field invariants."""

        values: dict[str, Any] = {}
        for field_name in type(self).model_fields:
            value = getattr(self, field_name)
            if deep:
                values[field_name] = _deep_copy_contract_value(value)
            elif isinstance(value, Mapping):
                values[field_name] = _thaw_json_value(value)
            else:
                values[field_name] = value
        if update:
            values.update(update)
        return type(self).model_validate(values)


def _deep_copy_contract_value(value: object) -> object:
    if isinstance(value, FrozenModel):
        return value.model_copy(deep=True)
    if isinstance(value, Mapping):
        return _thaw_json_value(value)
    if isinstance(value, tuple):
        return tuple(_deep_copy_contract_value(item) for item in value)
    return deepcopy(value)


class PlannerMode(StrEnum):
    VECTOR = "vector"
    GRAPH = "graph"
    FIXED_HYBRID = "fixed_hybrid"
    RULE = "rule"
    COST_AWARE = "cost_aware"


class SearchStatus(StrEnum):
    COMPLETE = "complete"
    PARTIAL = "partial"


class BranchKind(StrEnum):
    VECTOR = "vector"
    GRAPH = "graph"


class BranchStatus(StrEnum):
    NOT_SCHEDULED = "not_scheduled"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    TIMED_OUT = "timed_out"
    FAILED = "failed"
    CANCELLED = "cancelled"


class RequestState(StrEnum):
    RECEIVED = "received"
    ANALYZING = "analyzing"
    PLANNING = "planning"
    EXECUTING = "executing"
    FUSING = "fusing"
    RERANKING = "reranking"
    COMPLETE = "complete"
    PARTIAL = "partial"
    FAILED = "failed"


class CancellationReason(StrEnum):
    APPLICATION_DEADLINE = "application_deadline"
    CLIENT_DISCONNECT = "client_disconnect"
    PARENT_CANCELLED = "parent_cancelled"
    ENGINE_SHUTDOWN = "engine_shutdown"


class FailureOrigin(StrEnum):
    APPLICATION = "application"
    BACKEND_NATIVE = "backend_native"
    CIRCUIT_OPEN = "circuit_open"
    CLIENT = "client"


class CircuitState(StrEnum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class KillSwitch(StrEnum):
    FORCE_VECTOR_ONLY = "force_vector_only"
    DISABLE_COST_AWARE = "disable_cost_aware"


class EntityType(StrEnum):
    PERSON = "PERSON"
    ORG = "ORG"
    GPE = "GPE"
    LOC = "LOC"
    FAC = "FAC"
    PRODUCT = "PRODUCT"
    EVENT = "EVENT"
    WORK_OF_ART = "WORK_OF_ART"


class RelationExtractionRule(StrEnum):
    DIRECT_SVO = "direct_svo"
    PASSIVE = "passive"
    PREPOSITIONAL = "prepositional"
    COPULAR = "copular"
    APPOSITIONAL = "appositional"


class IngestionStoreStatus(StrEnum):
    PENDING = "pending"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class ActivationStatus(StrEnum):
    INACTIVE = "inactive"
    ACTIVE = "active"
    FAILED = "failed"


class VectorStageManifest(FrozenModel):
    """Verified Qdrant-only staging evidence; never a dual-store active pointer."""

    stage_schema_version: Literal["v2"] = "v2"
    status: Literal["vector_staged"] = "vector_staged"
    corpus_version: NonEmptyString
    collection_name: NonEmptyString
    chunk_count: Annotated[int, Field(ge=0)]
    canonical_id_checksum: Sha256Hex
    embedding_set_checksum: Sha256Hex
    embedding_model_id: Literal["sentence-transformers/all-MiniLM-L6-v2"] = EMBEDDING_MODEL_ID
    embedding_model_revision: Literal["b8903db39f65d93ae28d49a37c4f3fa90c5f94e0"] = (
        EMBEDDING_MODEL_REVISION
    )
    embedding_artifact_manifest_sha256: Sha256Hex
    embedding_dimension: Literal[384] = EMBEDDING_DIMENSION
    distance: Literal["cosine"] = "cosine"


class Chunk(FrozenModel):
    """One canonical, versioned evidence chunk shared by both stores."""

    id: NonEmptyString
    document_id: NonEmptyString
    corpus_version: NonEmptyString
    position: Annotated[int, Field(ge=0)]
    text: NonEmptyString
    token_count: Annotated[int, Field(ge=1)]
    entities: tuple[str, ...] = ()
    metadata: FrozenJsonMapping = Field(default_factory=dict)

    @property
    def canonical_chunk_id(self) -> str:
        return self.id


class Entity(FrozenModel):
    """An exact-normalized entity with its source aliases."""

    id: NonEmptyString
    name: NonEmptyString
    entity_type: EntityType
    normalized_name: NonEmptyString
    aliases: tuple[str, ...] = ()

    @field_validator("normalized_name")
    @classmethod
    def _require_canonical_name(cls, value: str) -> str:
        if value != normalize_entity_name(value):
            raise ValueError("normalized_name must use the ADR-008 normalization pipeline")
        return value

    @model_validator(mode="after")
    def _require_canonical_id(self) -> Entity:
        expected = str(entity_id(self.entity_type.value, self.normalized_name))
        if self.id != expected:
            raise ValueError("entity ID must be UUIDv5 of type and normalized name")
        return self


class EntityMention(FrozenModel):
    """One NER span with exact source and sentence provenance."""

    id: NonEmptyString
    entity_id: NonEmptyString
    entity_type: EntityType
    raw_text: NonEmptyString
    normalized_name: NonEmptyString
    source_chunk_id: NonEmptyString
    start_char: Annotated[int, Field(ge=0)]
    end_char: Annotated[int, Field(ge=1)]
    sentence_start_char: Annotated[int, Field(ge=0)]
    sentence_end_char: Annotated[int, Field(ge=1)]
    token_start: Annotated[int, Field(ge=0)]
    token_end: Annotated[int, Field(ge=1)]
    root_token: Annotated[int, Field(ge=0)]

    @field_validator("normalized_name")
    @classmethod
    def _require_normalized_name(cls, value: str) -> str:
        if value != normalize_entity_name(value):
            raise ValueError("normalized_name must use the ADR-008 normalization pipeline")
        return value

    @model_validator(mode="after")
    def _check_span(self) -> EntityMention:
        if self.end_char <= self.start_char:
            raise ValueError("mention character span must be non-empty")
        if self.token_end <= self.token_start or not (
            self.token_start <= self.root_token < self.token_end
        ):
            raise ValueError("mention token span must contain its root token")
        if not (
            self.sentence_start_char <= self.start_char and self.end_char <= self.sentence_end_char
        ):
            raise ValueError("mention span must be contained by its sentence span")
        expected_entity_id = str(entity_id(self.entity_type.value, self.normalized_name))
        if self.entity_id != expected_entity_id:
            raise ValueError("mention entity ID must match its type and normalized name")
        from ragplan.core.ids import entity_mention_id

        expected_mention_id = str(
            entity_mention_id(
                self.source_chunk_id,
                self.entity_id,
                self.start_char,
                self.end_char,
            )
        )
        if self.id != expected_mention_id:
            raise ValueError("mention ID must match its source span")
        return self


class Relation(FrozenModel):
    """A directed, provenance-preserving RELATES_TO edge."""

    source_entity_id: NonEmptyString
    target_entity_id: NonEmptyString
    relation_type: Literal["RELATES_TO"] = "RELATES_TO"
    predicate: NonEmptyString
    confidence: Annotated[float, Field(ge=0.70, le=1.0)]
    source_chunk_id: NonEmptyString
    extractor_version: NonEmptyString
    extraction_rule: RelationExtractionRule = RelationExtractionRule.DIRECT_SVO

    @model_validator(mode="after")
    def _reject_self_relation(self) -> Relation:
        if self.source_entity_id == self.target_entity_id:
            raise ValueError("a relation must connect two distinct entities")
        return self


class GraphStageManifest(FrozenModel):
    """Verified Neo4j-only staging evidence; it is never an active pointer."""

    stage_schema_version: Literal["v1"] = "v1"
    status: Literal["graph_staged"] = "graph_staged"
    corpus_version: NonEmptyString
    database: NonEmptyString
    document_count: Annotated[int, Field(ge=0)]
    chunk_count: Annotated[int, Field(ge=0)]
    entity_count: Annotated[int, Field(ge=0)]
    mention_count: Annotated[int, Field(ge=0)]
    relation_count: Annotated[int, Field(ge=0)]
    canonical_id_checksum: Sha256Hex
    graph_content_checksum: Sha256Hex
    extractor_version: NonEmptyString


class GraphPath(FrozenModel):
    """A 1–3 hop traversal whose relations preserve their stored directions."""

    entity_ids: tuple[NonEmptyString, ...] = Field(min_length=2, max_length=4)
    relations: tuple[Relation, ...] = Field(min_length=1, max_length=3)
    score: NonNegativeFloat = 0.0

    @model_validator(mode="after")
    def _check_path_shape(self) -> GraphPath:
        if len(self.entity_ids) != len(self.relations) + 1:
            raise ValueError("a graph path needs exactly one more entity than relations")
        if len(set(self.entity_ids)) != len(self.entity_ids):
            raise ValueError("a graph path cannot repeat an entity")
        for left, right, relation in zip(
            self.entity_ids[:-1], self.entity_ids[1:], self.relations, strict=True
        ):
            if {relation.source_entity_id, relation.target_entity_id} != {left, right}:
                raise ValueError("each relation must connect its adjacent path entities")
        return self

    @computed_field  # type: ignore[prop-decorator]
    @property
    def hop_count(self) -> int:
        return len(self.relations)


class GraphLimit(StrEnum):
    """Hard Stage 5 limits that may truncate graph work deterministically."""

    SEEDS = "seeds"
    PATHS = "paths"
    VISITED_ENTITIES = "visited_entities"
    RECOVERED_CHUNKS = "recovered_chunks"


class GraphSeedMatch(FrozenModel):
    """Privacy-safe evidence for one exact normalized query-seed lookup."""

    mention_sha256: Sha256Hex
    requested_entity_id: NonEmptyString
    matched_entity_id: NonEmptyString | None = None
    lookup_score: UnitFloat

    @model_validator(mode="after")
    def _check_match_score(self) -> GraphSeedMatch:
        matched = self.matched_entity_id is not None
        expected_score = 1.0 if matched else 0.0
        if not math.isclose(self.lookup_score, expected_score, rel_tol=0.0, abs_tol=0.0):
            raise ValueError("exact seed lookup score must be 1 for a match and 0 otherwise")
        if matched and self.matched_entity_id != self.requested_entity_id:
            raise ValueError("exact seed lookup cannot substitute a different entity")
        return self


class GraphTrace(FrozenModel):
    """Bounded graph phases without raw query text or normalized entity aliases."""

    score_version: Literal["graph_score_v0"] = "graph_score_v0"
    seed_matches: tuple[GraphSeedMatch, ...] = Field(max_length=5)
    requested_depth: Annotated[int, Field(ge=1, le=3)]
    actual_depth: Annotated[int, Field(ge=0, le=3)]
    visited_entity_count: Annotated[int, Field(ge=0, le=500)]
    path_count: Annotated[int, Field(ge=0, le=250)]
    recovered_chunk_count: Annotated[int, Field(ge=0, le=100)]
    limit_hits: tuple[GraphLimit, ...] = ()
    seed_lookup_latency_ms: NonNegativeFloat
    traversal_latency_ms: NonNegativeFloat
    recovery_latency_ms: NonNegativeFloat
    ranking_latency_ms: NonNegativeFloat

    @model_validator(mode="after")
    def _check_graph_trace(self) -> GraphTrace:
        if self.actual_depth > self.requested_depth:
            raise ValueError("actual graph depth cannot exceed requested depth")
        if len(set(self.limit_hits)) != len(self.limit_hits):
            raise ValueError("graph limit hits must be unique")
        if not self.seed_matches and (
            self.actual_depth
            or self.visited_entity_count
            or self.path_count
            or self.recovered_chunk_count
        ):
            raise ValueError("graph work requires at least one seed lookup")
        matched_count = sum(item.matched_entity_id is not None for item in self.seed_matches)
        if matched_count == 0 and (
            self.actual_depth or self.visited_entity_count or self.path_count
        ):
            raise ValueError("traversal work requires at least one matched seed")
        if matched_count > self.visited_entity_count:
            raise ValueError("visited entities must include every matched seed")
        if (self.actual_depth == 0) is not (self.path_count == 0):
            raise ValueError("actual graph depth and retained paths must be jointly empty")
        if self.recovered_chunk_count and self.path_count == 0:
            raise ValueError("chunk recovery requires at least one retained path")
        return self


class RetrievalContribution(FrozenModel):
    """One branch's auditable contribution to a final retrieval hit."""

    source: BranchKind
    original_rank: Annotated[int, Field(ge=1)]
    original_score: float
    weight: UnitFloat
    rrf_contribution: NonNegativeFloat
    metadata: FrozenJsonMapping = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_weighted_rrf(self) -> RetrievalContribution:
        expected = self.weight / (60 + self.original_rank)
        if not math.isclose(
            self.rrf_contribution,
            expected,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError("source contribution must use weighted_rrf_v1")
        return self


class RetrievalHit(FrozenModel):
    """Rankable evidence returned through vector, graph, or fused retrieval."""

    canonical_chunk_id: NonEmptyString
    text: NonEmptyString
    score: float
    source: NonEmptyString
    document_id: str | None = None
    entity_ids: tuple[str, ...] = ()
    metadata: FrozenJsonMapping = Field(default_factory=dict)
    rank: Annotated[int, Field(ge=1)] | None = None
    paths: tuple[GraphPath, ...] = ()
    sources: tuple[BranchKind, ...] = ()
    source_contributions: tuple[RetrievalContribution, ...] = ()

    @model_validator(mode="after")
    def _check_provenance(self) -> RetrievalHit:
        contribution_sources = tuple(item.source for item in self.source_contributions)
        if len(set(contribution_sources)) != len(contribution_sources):
            raise ValueError("retrieval source contributions must be unique")
        canonical_source_order = tuple(
            branch
            for branch in (BranchKind.VECTOR, BranchKind.GRAPH)
            if branch in contribution_sources
        )
        if contribution_sources != canonical_source_order:
            raise ValueError("retrieval source contributions must use canonical branch order")
        if self.sources != contribution_sources:
            raise ValueError("retrieval sources must match contribution order")
        if self.source == "fusion" and not self.source_contributions:
            raise ValueError("a fused retrieval hit requires source contributions")
        if self.source == "fusion" and not math.isclose(
            self.score,
            sum(item.rrf_contribution for item in self.source_contributions),
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError("a fused retrieval score must equal its source contributions")
        if self.source == "fusion" and self.paths and BranchKind.GRAPH not in contribution_sources:
            raise ValueError("fused graph paths require a graph source contribution")
        return self

    @property
    def id(self) -> str:
        return self.canonical_chunk_id


class QueryFeatures(FrozenModel):
    """Numeric feature schema v1; deliberately contains no embedding."""

    token_count: Annotated[int, Field(ge=0)]
    entity_count: Annotated[int, Field(ge=0)]
    entity_density: UnitFloat
    relation_signal: UnitFloat
    multi_hop_signal: UnitFloat
    comparison_signal: UnitFloat
    aggregation_signal: UnitFloat
    global_signal: UnitFloat
    final_top_k: Annotated[int, Field(ge=MIN_TOP_K, le=MAX_TOP_K)]


class QueryAnalysis(FrozenModel):
    """Single-pass query analysis; its embedding is never serialized."""

    normalized_query: NonEmptyString
    language_supported: bool
    token_count: Annotated[int, Field(ge=0)]
    query_embedding: tuple[float, ...] = Field(exclude=True, repr=False)
    seed_entity_mentions: tuple[str, ...] = ()
    seed_entity_ids: tuple[str, ...] = ()
    seed_limit_hit: bool = False
    features: QueryFeatures
    analyzer_version: NonEmptyString
    analysis_latency_ms: NonNegativeFloat

    @model_validator(mode="after")
    def _check_seed_contract(self) -> QueryAnalysis:
        if len(self.seed_entity_mentions) != len(self.seed_entity_ids):
            raise ValueError("seed mentions and IDs must have the same length")
        if len(self.seed_entity_ids) > 5:
            raise ValueError("query analysis cannot expose more than five graph seeds")
        if len(set(zip(self.seed_entity_mentions, self.seed_entity_ids, strict=True))) != len(
            self.seed_entity_ids
        ):
            raise ValueError("query analysis graph seeds must be unique")
        return self


class PlanDefinition(FrozenModel):
    """Immutable static plan parameters; predictions and deadlines live elsewhere."""

    id: PlanId
    name: NonEmptyString
    vector_enabled: bool
    graph_enabled: bool
    vector_top_k: Annotated[int, Field(ge=0)]
    graph_top_k: Annotated[int, Field(ge=0)]
    graph_depth: Annotated[int, Field(ge=0, le=3)]
    vector_weight: UnitFloat
    graph_weight: UnitFloat
    rerank_enabled: bool
    rerank_top_k: Annotated[int, Field(ge=0)] = 0
    enabled_in_p0: bool

    @model_validator(mode="after")
    def _check_invariants(self) -> PlanDefinition:
        if not self.vector_enabled and not self.graph_enabled:
            raise ValueError("at least one retrieval branch must be enabled")
        if not self.vector_enabled and (self.vector_top_k != 0 or self.vector_weight != 0):
            raise ValueError("disabled vector branch must have zero top-k and weight")
        if self.vector_enabled and self.vector_top_k < 1:
            raise ValueError("enabled vector branch requires top-k >= 1")
        if not self.graph_enabled and (
            self.graph_top_k != 0 or self.graph_weight != 0 or self.graph_depth != 0
        ):
            raise ValueError("disabled graph branch must have zero top-k, weight, and depth")
        if self.graph_enabled and (self.graph_top_k < 1 or not 1 <= self.graph_depth <= 3):
            raise ValueError("enabled graph branch requires top-k >= 1 and depth 1..3")
        if not math.isclose(
            self.vector_weight + self.graph_weight,
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("branch weights must sum to 1")
        candidate_count = self.vector_top_k + self.graph_top_k
        if self.rerank_enabled and (self.rerank_top_k < 1 or self.rerank_top_k > candidate_count):
            raise ValueError("reranking requires candidates and a bounded rerank_top_k")
        if not self.rerank_enabled and self.rerank_top_k != 0:
            raise ValueError("disabled reranking must have rerank_top_k 0")
        return self


class PlanEstimate(FrozenModel):
    plan_id: PlanId
    predicted_quality: UnitFloat | None = None
    predicted_p95_latency_ms: NonNegativeFloat | None = None
    feasible: bool
    infeasible_reason: str | None = None
    model_version: str | None = None
    inputs_hash: Sha256Hex | None = None


class PlannerDecision(FrozenModel):
    mode: PlannerMode
    effective_mode: PlannerMode | None = None
    selected_plan_id: PlanId
    selected_plan: PlanDefinition | None = None
    executed_vector_top_k: Annotated[int, Field(ge=MIN_TOP_K, le=MAX_TOP_K)] | None = None
    executed_graph_top_k: Annotated[int, Field(ge=MIN_TOP_K, le=MAX_TOP_K)] | None = None
    matched_rules: tuple[str, ...] = ()
    remaining_budget_ms: NonNegativeFloat
    candidate_estimates: tuple[PlanEstimate, ...] = ()
    budget_feasible: bool = True
    selection_reason: str | None = None
    fallback_reason: str | None = None
    feature_version: NonEmptyString
    config_version: NonEmptyString
    model_version: str | None = None

    @model_validator(mode="after")
    def _check_selected_plan(self) -> PlannerDecision:
        if self.selected_plan is not None and self.selected_plan.id != self.selected_plan_id:
            raise ValueError("selected_plan_id must match selected_plan.id")
        if self.effective_mode is None and self.fallback_reason is not None:
            raise ValueError("a planner fallback requires effective_mode")
        return self


class BranchResult(FrozenModel):
    branch: BranchKind
    status: BranchStatus
    latency_ms: NonNegativeFloat | None = None
    hits: tuple[RetrievalHit, ...] = Field(default=(), exclude=True, repr=False)
    error_code: ErrorCode | None = None
    hit_count: Annotated[int, Field(ge=0)] = 0
    started_at_ms: NonNegativeFloat | None = None
    ended_at_ms: NonNegativeFloat | None = None
    remaining_budget_at_start_ms: NonNegativeFloat | None = None
    remaining_budget_at_end_ms: NonNegativeFloat | None = None
    cancellation_reason: CancellationReason | None = None
    failure_origin: FailureOrigin | None = None
    timeout_origin: TimeoutOrigin | None = None
    circuit_state_before: CircuitState | None = None
    circuit_state_after: CircuitState | None = None

    @model_validator(mode="before")
    @classmethod
    def _derive_internal_hit_count(cls, values: object) -> object:
        if isinstance(values, dict) and "hit_count" not in values:
            hits = values.get("hits", ())
            if isinstance(hits, (tuple, list)):
                return {**values, "hit_count": len(hits)}
        return values

    @model_validator(mode="after")
    def _check_terminal_shape(self) -> BranchResult:
        terminal = {
            BranchStatus.SUCCEEDED,
            BranchStatus.TIMED_OUT,
            BranchStatus.FAILED,
            BranchStatus.CANCELLED,
        }
        if self.status not in terminal and self.latency_ms is not None:
            raise ValueError("non-terminal branch cannot have latency")
        if self.status in terminal and self.latency_ms is None:
            raise ValueError("terminal branch requires latency")
        if self.status is not BranchStatus.SUCCEEDED and self.hits:
            raise ValueError("only a succeeded branch can contain hits")
        if self.hits and self.hit_count != len(self.hits):
            raise ValueError("branch hit count must match retained internal hits")
        if self.status is not BranchStatus.SUCCEEDED and self.hit_count != 0:
            raise ValueError("only a succeeded branch can report hits")
        if self.status is BranchStatus.FAILED and self.error_code is None:
            raise ValueError("a failed branch requires an error code")
        if self.status is not BranchStatus.FAILED and self.error_code is not None:
            raise ValueError("only a failed branch can contain an error code")
        timing_values = (
            self.started_at_ms,
            self.ended_at_ms,
            self.remaining_budget_at_start_ms,
            self.remaining_budget_at_end_ms,
        )
        if any(value is not None for value in timing_values) and any(
            value is None for value in timing_values
        ):
            raise ValueError("branch boundary timing fields must be jointly present")
        if self.started_at_ms is not None and self.ended_at_ms is not None:
            if self.ended_at_ms < self.started_at_ms:
                raise ValueError("branch end cannot precede branch start")
            measured = self.ended_at_ms - self.started_at_ms
            if self.latency_ms is None or not math.isclose(
                self.latency_ms,
                measured,
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                raise ValueError("branch latency must match scheduler boundaries")
        if self.cancellation_reason is not None and self.status not in {
            BranchStatus.TIMED_OUT,
            BranchStatus.CANCELLED,
        }:
            raise ValueError("only timed-out or cancelled branches have a cancellation reason")
        if self.timeout_origin is not None and self.status is not BranchStatus.TIMED_OUT:
            raise ValueError("timeout origin requires a timed-out branch")
        if self.failure_origin is not None and self.status is BranchStatus.SUCCEEDED:
            raise ValueError("a succeeded branch cannot have a failure origin")
        return self


class RequestStateEvent(FrozenModel):
    state: RequestState
    elapsed_ms: NonNegativeFloat
    remaining_budget_ms: NonNegativeFloat
    branch_remaining_budget_ms: NonNegativeFloat


class SchedulerTrace(FrozenModel):
    """Frozen Stage 7 execution semantics shared by serving and profiling."""

    runtime_semantics_version: Literal["v1"] = "v1"
    state_events: tuple[RequestStateEvent, ...] = Field(min_length=2)
    expected_terminal_state: Literal[RequestState.COMPLETE] = RequestState.COMPLETE
    actual_terminal_state: Literal[RequestState.COMPLETE, RequestState.PARTIAL]
    backend_task_count: Annotated[int, Field(ge=1, le=2)]
    branch_start_skew_ms: NonNegativeFloat
    deadline_overshoot_ms: NonNegativeFloat
    admission_limit: Annotated[int, Field(ge=1)]
    kill_switches: tuple[KillSwitch, ...] = ()
    vector_circuit_state: CircuitState | None = None
    graph_circuit_state: CircuitState | None = None
    fallback_reason: str | None = None

    @model_validator(mode="after")
    def _check_scheduler_trace(self) -> SchedulerTrace:
        states = tuple(event.state for event in self.state_events)
        required_prefix = (
            RequestState.RECEIVED,
            RequestState.ANALYZING,
            RequestState.PLANNING,
            RequestState.EXECUTING,
            RequestState.FUSING,
        )
        valid_states = (*required_prefix, self.actual_terminal_state)
        valid_rerank_states = (
            *required_prefix,
            RequestState.RERANKING,
            self.actual_terminal_state,
        )
        if states not in {valid_states, valid_rerank_states}:
            raise ValueError("scheduler trace must follow the frozen request state machine")
        if any(
            later.elapsed_ms < earlier.elapsed_ms
            for earlier, later in zip(self.state_events, self.state_events[1:], strict=False)
        ):
            raise ValueError("scheduler state event times must be monotonic")
        if len(set(self.kill_switches)) != len(self.kill_switches):
            raise ValueError("kill switch trace values must be unique")
        canonical_switches = tuple(
            switch
            for switch in (KillSwitch.FORCE_VECTOR_ONLY, KillSwitch.DISABLE_COST_AWARE)
            if switch in self.kill_switches
        )
        if self.kill_switches != canonical_switches:
            raise ValueError("kill switch trace values must use canonical order")
        if (self.actual_terminal_state is RequestState.PARTIAL) is not (
            self.fallback_reason is not None
        ):
            raise ValueError("partial scheduler completion requires one fallback reason")
        return self


class FusionTrace(FrozenModel):
    """Privacy-safe deterministic fusion summary."""

    fusion_version: Literal["weighted_rrf_v1"] = "weighted_rrf_v1"
    rrf_k: Literal[60] = 60
    vector_input_count: Annotated[int, Field(ge=0)]
    graph_input_count: Annotated[int, Field(ge=0)]
    output_count: Annotated[int, Field(ge=0)]
    duplicate_count: Annotated[int, Field(ge=0)]
    missing_branches: tuple[BranchKind, ...] = ()

    @model_validator(mode="after")
    def _check_fusion_counts(self) -> FusionTrace:
        if len(set(self.missing_branches)) != len(self.missing_branches):
            raise ValueError("missing fusion branches must be unique")
        canonical_missing_order = tuple(
            branch
            for branch in (BranchKind.VECTOR, BranchKind.GRAPH)
            if branch in self.missing_branches
        )
        if self.missing_branches != canonical_missing_order:
            raise ValueError("missing fusion branches must use canonical branch order")
        if self.duplicate_count > min(self.vector_input_count, self.graph_input_count):
            raise ValueError("fusion duplicate count exceeds branch intersection")
        if self.output_count > (
            self.vector_input_count + self.graph_input_count - self.duplicate_count
        ):
            raise ValueError("fusion output count exceeds the deduplicated candidate count")
        return self


class SearchRequest(FrozenModel):
    query: QueryString
    top_k: Annotated[int, Field(ge=MIN_TOP_K, le=MAX_TOP_K)] = DEFAULT_TOP_K
    latency_budget_ms: Annotated[int, Field(ge=MIN_LATENCY_BUDGET_MS, le=MAX_LATENCY_BUDGET_MS)] = (
        DEFAULT_LATENCY_BUDGET_MS
    )
    planner: PlannerMode = PlannerMode.RULE
    plan_id: PlanId | None = None

    @field_validator("planner", mode="before")
    @classmethod
    def _parse_public_planner_mode(cls, value: object) -> object:
        if isinstance(value, str):
            try:
                return PlannerMode(value)
            except ValueError:
                return value
        return value

    @model_validator(mode="before")
    @classmethod
    def _trim_query(cls, values: object) -> object:
        if isinstance(values, dict) and isinstance(values.get("query"), str):
            return {**values, "query": values["query"].strip()}
        return values

    @model_validator(mode="after")
    def _check_planner_constraints(self) -> SearchRequest:
        if self.planner is PlannerMode.COST_AWARE and self.top_k != 10:
            raise ValueError("cost_aware planner supports only top_k=10")
        if self.plan_id is not None and self.planner is not PlannerMode.FIXED_HYBRID:
            raise ValueError("plan_id is supported only by fixed_hybrid")
        if self.planner is PlannerMode.FIXED_HYBRID and self.plan_id not in {
            None,
            "P4",
            "P5",
            "P6",
            "P8",
        }:
            raise ValueError("fixed_hybrid supports only P4, P5, P6, and P8")
        return self


class SearchTrace(FrozenModel):
    """Redacted execution trace. Raw query, embedding, and result text are absent."""

    trace_schema_version: Literal["v1"] = "v1"
    request_id: NonEmptyString
    query_hash: Sha256Hex
    query_length: Annotated[int, Field(ge=MIN_QUERY_CODEPOINTS, le=MAX_QUERY_CODEPOINTS)]
    language_supported: bool
    features: QueryFeatures
    planner_decision: PlannerDecision
    branch_results: tuple[BranchResult, ...] = Field(min_length=1, max_length=2)
    analyzer_latency_ms: NonNegativeFloat
    planner_latency_ms: NonNegativeFloat
    embedding_latency_ms: NonNegativeFloat = 0.0
    vector_latency_ms: NonNegativeFloat | None = None
    graph_latency_ms: NonNegativeFloat | None = None
    graph_trace: GraphTrace | None = None
    fusion_latency_ms: NonNegativeFloat = 0.0
    fusion_trace: FusionTrace | None = None
    scheduler_trace: SchedulerTrace | None = None
    rerank_latency_ms: NonNegativeFloat = 0.0
    total_latency_ms: NonNegativeFloat
    latency_budget_ms: Annotated[int, Field(ge=MIN_LATENCY_BUDGET_MS, le=MAX_LATENCY_BUDGET_MS)]
    finalization_reserve_ms: NonNegativeFloat
    budget_feasible: bool
    budget_violated: bool
    fallback: bool
    result_count: Annotated[int, Field(ge=0)]
    corpus_version: NonEmptyString
    config_version: NonEmptyString
    model_version: str | None = None

    @model_validator(mode="after")
    def _check_budget_violation(self) -> SearchTrace:
        actual_violation = self.total_latency_ms > self.latency_budget_ms
        if self.budget_violated is not actual_violation:
            raise ValueError("budget_violated must reflect total latency without hidden grace")
        expected_reserve = min(20.0, max(5.0, self.latency_budget_ms * 0.05))
        if not math.isclose(
            self.finalization_reserve_ms,
            expected_reserve,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("finalization reserve must match the request budget")
        if self.result_count > self.features.final_top_k:
            raise ValueError("trace result count cannot exceed requested final_top_k")
        branch_kinds = tuple(branch.branch for branch in self.branch_results)
        if len(set(branch_kinds)) != len(branch_kinds):
            raise ValueError("a trace cannot contain duplicate retrieval branches")
        vector_branches = tuple(
            branch
            for branch in self.branch_results
            if branch.branch is BranchKind.VECTOR
            and branch.status not in {BranchStatus.NOT_SCHEDULED, BranchStatus.RUNNING}
        )
        if not vector_branches and self.vector_latency_ms is not None:
            raise ValueError("vector latency requires a terminal vector branch")
        if vector_branches:
            branch_latency = vector_branches[0].latency_ms
            if self.vector_latency_ms is None or not math.isclose(
                self.vector_latency_ms,
                cast(float, branch_latency),
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError("vector latency must match the vector branch latency")
        graph_branches = tuple(
            branch
            for branch in self.branch_results
            if branch.branch is BranchKind.GRAPH
            and branch.status not in {BranchStatus.NOT_SCHEDULED, BranchStatus.RUNNING}
        )
        if not graph_branches and (
            self.graph_latency_ms is not None or self.graph_trace is not None
        ):
            raise ValueError("graph timing requires a terminal graph branch")
        if graph_branches:
            branch_latency = graph_branches[0].latency_ms
            if self.graph_latency_ms is None or not math.isclose(
                self.graph_latency_ms,
                cast(float, branch_latency),
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError("graph latency must match the graph branch latency")
            if graph_branches[0].status is BranchStatus.SUCCEEDED and self.graph_trace is None:
                raise ValueError("a succeeded graph branch requires bounded graph trace evidence")
        if self.fusion_trace is None and self.fusion_latency_ms != 0.0:
            raise ValueError("fusion latency requires a fusion trace")
        if self.fusion_trace is not None:
            if self.planner_decision.effective_mode is not PlannerMode.FIXED_HYBRID:
                raise ValueError("fusion trace requires effective fixed_hybrid mode")
            if self.fusion_trace.output_count != self.result_count:
                raise ValueError("fusion output count must match trace result count")
            branches_by_kind = {branch.branch: branch for branch in self.branch_results}
            input_counts = {
                BranchKind.VECTOR: self.fusion_trace.vector_input_count,
                BranchKind.GRAPH: self.fusion_trace.graph_input_count,
            }
            for branch_kind, input_count in input_counts.items():
                branch = branches_by_kind.get(branch_kind)
                missing = branch_kind in self.fusion_trace.missing_branches
                if missing and input_count != 0:
                    raise ValueError("a missing fusion branch must have zero input candidates")
                if missing and branch is not None and branch.status is BranchStatus.SUCCEEDED:
                    raise ValueError("a succeeded fusion branch cannot be marked missing")
                if not missing and (
                    branch is None
                    or branch.status is not BranchStatus.SUCCEEDED
                    or branch.hit_count != input_count
                ):
                    raise ValueError("fusion inputs must match succeeded branch results")
        if self.scheduler_trace is not None:
            expected_terminal = RequestState.PARTIAL if self.fallback else RequestState.COMPLETE
            if self.scheduler_trace.actual_terminal_state is not expected_terminal:
                raise ValueError("scheduler terminal state must match response fallback semantics")
            if self.fallback and (
                self.planner_decision.fallback_reason != self.scheduler_trace.fallback_reason
            ):
                raise ValueError("planner and scheduler fallback reasons must match")
            expected_overshoot = max(0.0, self.total_latency_ms - self.latency_budget_ms)
            if not math.isclose(
                self.scheduler_trace.deadline_overshoot_ms,
                expected_overshoot,
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                raise ValueError("scheduler overshoot must match total request latency")
            if self.scheduler_trace.backend_task_count != len(self.branch_results):
                raise ValueError("scheduler task count must match scheduled branches")
            if self.scheduler_trace.state_events[-1].elapsed_ms > self.total_latency_ms:
                raise ValueError("scheduler terminal boundary cannot follow total latency")
            starts = tuple(
                branch.started_at_ms
                for branch in self.branch_results
                if branch.started_at_ms is not None
            )
            expected_skew = max(starts) - min(starts) if starts else 0.0
            if not math.isclose(
                self.scheduler_trace.branch_start_skew_ms,
                expected_skew,
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                raise ValueError("scheduler start skew must match branch boundaries")
            branches_by_kind = {branch.branch: branch for branch in self.branch_results}
            for branch_kind, circuit_state in (
                (BranchKind.VECTOR, self.scheduler_trace.vector_circuit_state),
                (BranchKind.GRAPH, self.scheduler_trace.graph_circuit_state),
            ):
                branch = branches_by_kind.get(branch_kind)
                expected_state = branch.circuit_state_after if branch is not None else None
                if circuit_state is not expected_state:
                    raise ValueError("scheduler circuit state must match its branch result")
        return self


class SearchResponse(FrozenModel):
    api_schema_version: Literal["v1"] = "v1"
    status: SearchStatus
    results: tuple[RetrievalHit, ...]
    planner_decision: PlannerDecision
    trace: SearchTrace
    fallback: bool = False
    request_id: NonEmptyString

    @model_validator(mode="after")
    def _check_response_trace(self) -> SearchResponse:
        if self.request_id != self.trace.request_id:
            raise ValueError("response and trace request_id must match")
        if self.planner_decision != self.trace.planner_decision:
            raise ValueError("response and trace planner decisions must match")
        if self.fallback != self.trace.fallback:
            raise ValueError("response and trace fallback must match")
        if len(self.results) != self.trace.result_count:
            raise ValueError("response result count must match trace")
        if (self.status is SearchStatus.PARTIAL) is not self.fallback:
            raise ValueError("partial response status must match fallback")
        statuses = {
            branch.status
            for branch in self.trace.branch_results
            if branch.status is not BranchStatus.NOT_SCHEDULED
        }
        failed_statuses = {
            BranchStatus.TIMED_OUT,
            BranchStatus.FAILED,
            BranchStatus.CANCELLED,
        }
        if BranchStatus.RUNNING in statuses:
            raise ValueError("a final response cannot contain a running branch")
        if self.status is SearchStatus.COMPLETE and statuses != {BranchStatus.SUCCEEDED}:
            raise ValueError("complete response requires every scheduled branch to succeed")
        if self.status is SearchStatus.PARTIAL and not (
            BranchStatus.SUCCEEDED in statuses and bool(statuses & failed_statuses)
        ):
            raise ValueError("partial response requires both a success and a failed branch")
        return self


class IngestionManifest(FrozenModel):
    ingestion_run_id: NonEmptyString
    corpus_version: NonEmptyString
    source_dataset: NonEmptyString
    source_version: NonEmptyString
    source_sha256: Sha256Hex
    chunker_version: NonEmptyString
    embedding_model_revision: NonEmptyString
    extractor_version: NonEmptyString
    document_count: Annotated[int, Field(ge=0)]
    chunk_count: Annotated[int, Field(ge=0)]
    qdrant_count: Annotated[int, Field(ge=0)]
    qdrant_id_checksum: Sha256Hex
    qdrant_status: IngestionStoreStatus
    neo4j_count: Annotated[int, Field(ge=0)]
    neo4j_id_checksum: Sha256Hex
    neo4j_status: IngestionStoreStatus
    activation_status: ActivationStatus

    @model_validator(mode="after")
    def _check_activation(self) -> IngestionManifest:
        stores_succeeded = (
            self.qdrant_status is IngestionStoreStatus.SUCCEEDED
            and self.neo4j_status is IngestionStoreStatus.SUCCEEDED
        )
        reconciled = (
            self.qdrant_count == self.neo4j_count
            and self.qdrant_count == self.chunk_count
            and self.qdrant_id_checksum == self.neo4j_id_checksum
        )
        if self.activation_status is ActivationStatus.ACTIVE and not (
            stores_succeeded and reconciled and self.document_count > 0 and self.chunk_count > 0
        ):
            raise ValueError("only reconciled successful stores can become active")
        return self


class ModelManifest(FrozenModel):
    model_name: NonEmptyString
    model_version: NonEmptyString
    artifact_version: NonEmptyString
    artifact_sha256: Sha256Hex
    feature_schema_version: NonEmptyString
    plan_catalog_hash: Sha256Hex
    corpus_version: NonEmptyString
    qrels_version: NonEmptyString
    embedding_model_revision: NonEmptyString
    extractor_version: NonEmptyString
    qdrant_version: NonEmptyString
    neo4j_version: NonEmptyString
    qdrant_client_version: NonEmptyString
    training_config_hash: Sha256Hex
    train_validation_split_hash: Sha256Hex
    runtime_fingerprint: NonEmptyString
    runtime_semantics_version: NonEmptyString
    validation_metrics: FrozenFloatMapping
