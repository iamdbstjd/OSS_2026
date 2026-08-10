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
from ragplan.core.errors import ErrorCode
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


class EntityType(StrEnum):
    PERSON = "PERSON"
    ORG = "ORG"
    GPE = "GPE"
    LOC = "LOC"
    FAC = "FAC"
    PRODUCT = "PRODUCT"
    EVENT = "EVENT"
    WORK_OF_ART = "WORK_OF_ART"


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


class Relation(FrozenModel):
    """A directed, provenance-preserving RELATES_TO edge."""

    source_entity_id: NonEmptyString
    target_entity_id: NonEmptyString
    relation_type: Literal["RELATES_TO"] = "RELATES_TO"
    predicate: NonEmptyString
    confidence: Annotated[float, Field(ge=0.70, le=1.0)]
    source_chunk_id: NonEmptyString
    extractor_version: NonEmptyString

    @model_validator(mode="after")
    def _reject_self_relation(self) -> Relation:
        if self.source_entity_id == self.target_entity_id:
            raise ValueError("a relation must connect two distinct entities")
        return self


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

    @property
    def hop_count(self) -> int:
        return len(self.relations)


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
    features: QueryFeatures
    analyzer_version: NonEmptyString
    analysis_latency_ms: NonNegativeFloat


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

    @computed_field
    def hit_count(self) -> int:
        return len(self.hits)

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
        if self.status is BranchStatus.FAILED and self.error_code is None:
            raise ValueError("a failed branch requires an error code")
        if self.status is not BranchStatus.FAILED and self.error_code is not None:
            raise ValueError("only a failed branch can contain an error code")
        return self


class SearchRequest(FrozenModel):
    query: QueryString
    top_k: Annotated[int, Field(ge=MIN_TOP_K, le=MAX_TOP_K)] = DEFAULT_TOP_K
    latency_budget_ms: Annotated[int, Field(ge=MIN_LATENCY_BUDGET_MS, le=MAX_LATENCY_BUDGET_MS)] = (
        DEFAULT_LATENCY_BUDGET_MS
    )
    planner: PlannerMode = PlannerMode.RULE

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
    def _check_cost_aware_top_k(self) -> SearchRequest:
        if self.planner is PlannerMode.COST_AWARE and self.top_k != 10:
            raise ValueError("cost_aware planner supports only top_k=10")
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
    fusion_latency_ms: NonNegativeFloat = 0.0
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
            stores_succeeded and reconciled
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
