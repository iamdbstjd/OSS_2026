"""Bounded process-local JSON metrics without query-derived labels."""

from __future__ import annotations

from collections import Counter
from threading import Lock
from typing import Annotated, Final, Literal

from pydantic import BaseModel, ConfigDict, Field

from ragplan.core.errors import ErrorCode
from ragplan.core.models import BranchKind, BranchStatus, PlannerMode, SearchResponse, SearchStatus

LATENCY_BUCKETS_MS: Final = (25.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 5000.0)


class HistogramSnapshot(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    count: Annotated[int, Field(ge=0)]
    sum_ms: Annotated[float, Field(ge=0.0)]
    buckets_ms: dict[str, Annotated[int, Field(ge=0)]]


class MetricsSnapshot(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    schema_version: Literal["metrics_v1"] = "metrics_v1"
    request_count: Annotated[int, Field(ge=0)]
    complete_count: Annotated[int, Field(ge=0)]
    partial_count: Annotated[int, Field(ge=0)]
    error_count: Annotated[int, Field(ge=0)]
    timeout_count: Annotated[int, Field(ge=0)]
    fallback_count: Annotated[int, Field(ge=0)]
    budget_violation_count: Annotated[int, Field(ge=0)]
    model_fallback_count: Annotated[int, Field(ge=0)]
    planner_distribution: dict[str, Annotated[int, Field(ge=0)]]
    error_distribution: dict[str, Annotated[int, Field(ge=0)]]
    total_latency: HistogramSnapshot
    vector_latency: HistogramSnapshot
    graph_latency: HistogramSnapshot


class _Histogram:
    def __init__(self) -> None:
        self.count = 0
        self.sum_ms = 0.0
        self.buckets = {value: 0 for value in LATENCY_BUCKETS_MS}

    def observe(self, value_ms: float) -> None:
        value = max(0.0, float(value_ms))
        self.count += 1
        self.sum_ms += value
        for boundary in LATENCY_BUCKETS_MS:
            if value <= boundary:
                self.buckets[boundary] += 1

    def snapshot(self) -> HistogramSnapshot:
        return HistogramSnapshot(
            count=self.count,
            sum_ms=self.sum_ms,
            buckets_ms={_bucket_label(key): value for key, value in self.buckets.items()},
        )


class MetricsRegistry:
    """Small synchronized counters suitable for the single-process P0 service."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._request_count = 0
        self._complete_count = 0
        self._partial_count = 0
        self._error_count = 0
        self._timeout_count = 0
        self._fallback_count = 0
        self._budget_violation_count = 0
        self._model_fallback_count = 0
        self._planner: Counter[str] = Counter()
        self._errors: Counter[str] = Counter()
        self._total_latency = _Histogram()
        self._vector_latency = _Histogram()
        self._graph_latency = _Histogram()

    def record_request(self, planner: PlannerMode | None) -> None:
        with self._lock:
            self._request_count += 1
            self._planner[planner.value if planner is not None else "invalid"] += 1

    def record_success(self, response: SearchResponse) -> None:
        trace = response.trace
        with self._lock:
            if response.status is SearchStatus.COMPLETE:
                self._complete_count += 1
            else:
                self._partial_count += 1
            self._fallback_count += int(response.fallback)
            self._budget_violation_count += int(trace.budget_violated)
            self._model_fallback_count += int(
                response.planner_decision.mode is PlannerMode.COST_AWARE
                and response.planner_decision.effective_mode is PlannerMode.RULE
            )
            self._timeout_count += int(
                any(item.status is BranchStatus.TIMED_OUT for item in trace.branch_results)
            )
            self._total_latency.observe(trace.total_latency_ms)
            for branch in trace.branch_results:
                if branch.latency_ms is None:
                    continue
                target = (
                    self._vector_latency
                    if branch.branch is BranchKind.VECTOR
                    else self._graph_latency
                )
                target.observe(branch.latency_ms)

    def record_error(self, code: ErrorCode) -> None:
        with self._lock:
            self._error_count += 1
            self._errors[code.value] += 1
            self._timeout_count += int(code is ErrorCode.DEADLINE_EXCEEDED)

    def snapshot(self) -> MetricsSnapshot:
        with self._lock:
            return MetricsSnapshot(
                request_count=self._request_count,
                complete_count=self._complete_count,
                partial_count=self._partial_count,
                error_count=self._error_count,
                timeout_count=self._timeout_count,
                fallback_count=self._fallback_count,
                budget_violation_count=self._budget_violation_count,
                model_fallback_count=self._model_fallback_count,
                planner_distribution=dict(sorted(self._planner.items())),
                error_distribution=dict(sorted(self._errors.items())),
                total_latency=self._total_latency.snapshot(),
                vector_latency=self._vector_latency.snapshot(),
                graph_latency=self._graph_latency.snapshot(),
            )


def _bucket_label(value: float) -> str:
    return f"le_{int(value)}"


__all__ = ["LATENCY_BUCKETS_MS", "HistogramSnapshot", "MetricsRegistry", "MetricsSnapshot"]
