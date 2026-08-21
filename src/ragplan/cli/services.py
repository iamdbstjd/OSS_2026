"""Reusable Stage 13 command services shared by Typer wrappers and tests."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Literal
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from pydantic import Field

from ragplan.api.readiness import ReadinessResponse, inspect_readiness
from ragplan.api.runtime import (
    Stage3RuntimeConfig,
    build_search_engine,
    build_search_engine_from_environment,
)
from ragplan.backends.vector.qdrant import DEFAULT_COLLECTION_PREFIX
from ragplan.core.deadline import Deadline
from ragplan.core.engine import SearchEngine
from ragplan.core.errors import ErrorCode, ErrorResponse, RAGPlanError
from ragplan.core.models import (
    FrozenModel,
    PlannerDecision,
    PlannerMode,
    QueryAnalysis,
    QueryFeatures,
    SearchRequest,
    SearchResponse,
)
from ragplan.ingestion.audit import load_graph_tier_policy
from ragplan.ingestion.model_manifest import load_default_model_artifact_manifest
from ragplan.ingestion.normalize import normalize_text
from ragplan.ingestion.service import VectorIngestResult, ingest_vector_corpus
from ragplan.planner.catalog import load_default_plan_catalog
from ragplan.planner.features import extract_query_features, load_default_query_feature_config
from ragplan.planner.rule import RulePlanner, load_default_rule_planner_config

_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?")


class PlannerDemoResult(FrozenModel):
    schema_version: Literal["planner_demo_v1"] = "planner_demo_v1"
    mode: Literal["planner_only_no_embedding"] = "planner_only_no_embedding"
    query_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    query_length: Annotated[int, Field(ge=1, le=4096)]
    language_supported: bool
    supplied_entity_count: Annotated[int, Field(ge=0)]
    features: QueryFeatures
    decision: PlannerDecision
    executes_retrieval: Literal[False] = False
    limitations: tuple[str, ...] = (
        "no embedding or backend call",
        "entity_count is caller-supplied",
        "graph routing remains fail-closed when the audit gate is disabled",
    )


class VerifyResult(FrozenModel):
    schema_version: Literal["verify_v1"] = "verify_v1"
    status: Literal["ok"] = "ok"
    level: Literal["code", "configured_runtime"]
    plan_catalog_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    query_feature_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    rule_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    embedding_artifact_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    graph_tier_enabled: bool
    cost_aware_status: Literal["research_only_disabled"] = "research_only_disabled"
    readiness: ReadinessResponse | None = None


class QuickstartVectorResult(FrozenModel):
    schema_version: Literal["quickstart_vector_v1"] = "quickstart_vector_v1"
    infrastructure: Literal["qdrant_plus_minilm"] = "qdrant_plus_minilm"
    model_snapshot: str
    vector_stage_manifest: str
    corpus_version: str
    search: SearchResponse


def explain_plan(
    *,
    query: str,
    latency_budget_ms: int = 100,
    top_k: int = 10,
    entity_count: int = 0,
) -> PlannerDemoResult:
    request = SearchRequest(
        query=query,
        latency_budget_ms=latency_budget_ms,
        top_k=top_k,
        planner=PlannerMode.RULE,
    )
    if isinstance(entity_count, bool) or entity_count < 0:
        raise RAGPlanError(
            ErrorCode.INVALID_REQUEST,
            "entity count must be a non-negative integer",
            retryable=False,
        )
    normalized = normalize_text(request.query)
    token_count = len(_TOKEN_PATTERN.findall(normalized))
    feature_config = load_default_query_feature_config()
    features = extract_query_features(
        normalized,
        token_count=token_count,
        entity_count=entity_count,
        final_top_k=request.top_k,
        config=feature_config,
    )
    language_supported = not any(
        character.isalpha() and not character.isascii() for character in normalized
    )
    analysis = QueryAnalysis(
        normalized_query=normalized,
        language_supported=language_supported,
        token_count=token_count,
        query_embedding=(),
        features=features,
        analyzer_version="planner-demo-v1",
        analysis_latency_ms=0.0,
    )
    catalog = load_default_plan_catalog()
    planner = RulePlanner(
        catalog=catalog,
        graph_policy=load_graph_tier_policy(),
        config=load_default_rule_planner_config(),
        feature_config_sha256=feature_config.sha256,
    )
    decision = planner.select(
        analysis,
        deadline=Deadline.start(request.latency_budget_ms),
        graph_runtime_available=False,
    )
    return PlannerDemoResult(
        query_hash=hashlib.sha256(request.query.encode("utf-8")).hexdigest(),
        query_length=len(request.query),
        language_supported=language_supported,
        supplied_entity_count=entity_count,
        features=features,
        decision=decision,
    )


async def search_configured_runtime(
    request: SearchRequest,
    *,
    request_id: str,
    environment: Mapping[str, str] | None = None,
) -> SearchResponse:
    engine = await build_search_engine_from_environment(environment)
    if engine is None:
        raise RAGPlanError(
            ErrorCode.NOT_READY,
            "no complete RAGPlan runtime profile is configured",
        )
    try:
        return await engine.search(request, request_id=request_id)
    finally:
        await engine.close()


def search_http_api(
    request: SearchRequest,
    *,
    request_id: str,
    api_url: str,
    timeout_seconds: float = 10.0,
) -> SearchResponse:
    endpoint = _search_endpoint(api_url)
    body = request.model_dump_json().encode("utf-8")
    http_request = Request(
        endpoint,
        data=body,
        method="POST",
        headers={
            "content-type": "application/json",
            "x-request-id": request_id,
        },
    )
    try:
        with urlopen(http_request, timeout=timeout_seconds) as response:  # noqa: S310
            payload = response.read()
    except HTTPError as exc:
        try:
            error = ErrorResponse.model_validate_json(exc.read())
        except Exception as decode_error:
            raise RAGPlanError(
                ErrorCode.DEPENDENCY_UNAVAILABLE,
                "RAGPlan API returned an invalid error response",
            ) from decode_error
        raise RAGPlanError(
            error.code,
            error.message,
            retryable=error.retryable,
        ) from exc
    except (OSError, URLError) as exc:
        raise RAGPlanError(
            ErrorCode.DEPENDENCY_UNAVAILABLE,
            "RAGPlan API is unavailable",
        ) from exc
    try:
        return SearchResponse.model_validate_json(payload)
    except ValueError as exc:
        raise RAGPlanError(
            ErrorCode.DEPENDENCY_UNAVAILABLE,
            "RAGPlan API returned an invalid search response",
        ) from exc


async def verify_installation(
    *,
    configured_runtime: bool,
    environment: Mapping[str, str] | None = None,
) -> VerifyResult:
    catalog = load_default_plan_catalog()
    features = load_default_query_feature_config()
    rule = load_default_rule_planner_config()
    graph_policy = load_graph_tier_policy()
    embedding_manifest = load_default_model_artifact_manifest()
    readiness: ReadinessResponse | None = None
    if configured_runtime:
        engine: SearchEngine | None = await build_search_engine_from_environment(environment)
        try:
            readiness, status_code = await inspect_readiness(
                engine,
                graph_tier_enabled=graph_policy.graph_tier_enabled,
            )
            if status_code != 200:
                raise RAGPlanError(
                    ErrorCode.NOT_READY,
                    f"configured runtime is not ready: {readiness.reason or 'unknown'}",
                )
        finally:
            if engine is not None:
                await engine.close()
    return VerifyResult(
        level="configured_runtime" if configured_runtime else "code",
        plan_catalog_sha256=catalog.sha256(),
        query_feature_config_sha256=features.sha256,
        rule_config_sha256=rule.sha256,
        embedding_artifact_manifest_sha256=embedding_manifest.sha256,
        graph_tier_enabled=graph_policy.graph_tier_enabled,
        readiness=readiness,
    )


async def quickstart_vector(
    *,
    query: str,
    input_path: Path,
    model_cache: Path,
    stage_manifest_path: Path,
    corpus_version: str,
    qdrant_url: str,
    collection_prefix: str = DEFAULT_COLLECTION_PREFIX,
) -> QuickstartVectorResult:
    ingested = await ingest_vector_corpus(
        input_path=input_path,
        corpus_version=corpus_version,
        stage_manifest_path=stage_manifest_path,
        model_cache=model_cache,
        qdrant_url=qdrant_url,
        collection_prefix=collection_prefix,
    )
    engine = await build_search_engine(
        Stage3RuntimeConfig(
            model_snapshot=ingested.model_snapshot,
            vector_stage_manifest=ingested.stage_manifest_path,
            qdrant_url=qdrant_url,
            collection_prefix=collection_prefix,
        )
    )
    try:
        response = await engine.search(
            SearchRequest(query=query, planner=PlannerMode.VECTOR),
            request_id="quickstart-vector-v1",
        )
    finally:
        await engine.close()
    return QuickstartVectorResult(
        model_snapshot=str(ingested.model_snapshot),
        vector_stage_manifest=str(ingested.stage_manifest_path),
        corpus_version=ingested.stage.corpus_version,
        search=response,
    )


def ingest_result_payload(result: VectorIngestResult) -> dict[str, object]:
    return {
        "schema_version": "cli_ingest_result_v1",
        "model_snapshot": str(result.model_snapshot),
        "stage_manifest": str(result.stage_manifest_path),
        "vector_stage": result.stage.model_dump(mode="json"),
    }


def _search_endpoint(api_url: str) -> str:
    parsed = urlparse(api_url)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise RAGPlanError(
            ErrorCode.INVALID_REQUEST,
            "API URL must be an HTTP(S) origin without credentials",
            retryable=False,
        )
    base = api_url.rstrip("/")
    return base if parsed.path.rstrip("/").endswith("/v1/search") else f"{base}/v1/search"


def json_line(payload: object) -> str:
    if isinstance(payload, FrozenModel):
        value: object = payload.model_dump(mode="json")
    elif hasattr(payload, "model_dump"):
        value = payload.model_dump(mode="json")
    else:
        value = payload
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


__all__ = [
    "PlannerDemoResult",
    "QuickstartVectorResult",
    "VerifyResult",
    "explain_plan",
    "ingest_result_payload",
    "json_line",
    "quickstart_vector",
    "search_configured_runtime",
    "search_http_api",
    "verify_installation",
]
