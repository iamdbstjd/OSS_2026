"""Validated request/branch state machines, kill switches, and circuit breakers."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

from ragplan.core.deadline import NANOSECONDS_PER_SECOND, Deadline, MonotonicClock, PerfCounterClock
from ragplan.core.errors import ErrorCode, RAGPlanError
from ragplan.core.models import (
    BranchStatus,
    CircuitState,
    KillSwitch,
    RequestState,
    RequestStateEvent,
)

CIRCUIT_FAILURE_THRESHOLD: Final = 5
CIRCUIT_OPEN_SECONDS: Final = 30.0

_REQUEST_TRANSITIONS: Final = {
    RequestState.RECEIVED: frozenset({RequestState.ANALYZING}),
    RequestState.ANALYZING: frozenset({RequestState.PLANNING, RequestState.FAILED}),
    RequestState.PLANNING: frozenset({RequestState.EXECUTING, RequestState.FAILED}),
    RequestState.EXECUTING: frozenset({RequestState.FUSING, RequestState.FAILED}),
    RequestState.FUSING: frozenset(
        {
            RequestState.RERANKING,
            RequestState.COMPLETE,
            RequestState.PARTIAL,
            RequestState.FAILED,
        }
    ),
    RequestState.RERANKING: frozenset(
        {RequestState.COMPLETE, RequestState.PARTIAL, RequestState.FAILED}
    ),
    RequestState.COMPLETE: frozenset(),
    RequestState.PARTIAL: frozenset(),
    RequestState.FAILED: frozenset(),
}

_BRANCH_TRANSITIONS: Final = {
    BranchStatus.NOT_SCHEDULED: frozenset({BranchStatus.RUNNING}),
    BranchStatus.RUNNING: frozenset(
        {
            BranchStatus.SUCCEEDED,
            BranchStatus.TIMED_OUT,
            BranchStatus.FAILED,
            BranchStatus.CANCELLED,
        }
    ),
    BranchStatus.SUCCEEDED: frozenset(),
    BranchStatus.TIMED_OUT: frozenset(),
    BranchStatus.FAILED: frozenset(),
    BranchStatus.CANCELLED: frozenset(),
}


class RequestStateMachine:
    """Record every legal request boundary against one absolute deadline."""

    def __init__(self, deadline: Deadline) -> None:
        self._deadline = deadline
        self._state = RequestState.RECEIVED
        self._events = [self._event(RequestState.RECEIVED)]

    @property
    def state(self) -> RequestState:
        return self._state

    @property
    def events(self) -> tuple[RequestStateEvent, ...]:
        return tuple(self._events)

    def transition(self, target: RequestState) -> None:
        if target not in _REQUEST_TRANSITIONS[self._state]:
            raise RAGPlanError(
                ErrorCode.INTERNAL_ERROR,
                "invalid request scheduler state transition",
                retryable=False,
            )
        self._state = target
        self._events.append(self._event(target))

    def fail_if_possible(self) -> None:
        if RequestState.FAILED in _REQUEST_TRANSITIONS[self._state]:
            self.transition(RequestState.FAILED)

    def _event(self, state: RequestState) -> RequestStateEvent:
        snapshot = self._deadline.snapshot()
        return RequestStateEvent(
            state=state,
            elapsed_ms=snapshot.elapsed_ms,
            remaining_budget_ms=snapshot.remaining_ms,
            branch_remaining_budget_ms=snapshot.branch_remaining_ms,
        )


class BranchStateMachine:
    """Reject impossible branch terminal-state rewrites."""

    def __init__(self) -> None:
        self._state = BranchStatus.NOT_SCHEDULED

    @property
    def state(self) -> BranchStatus:
        return self._state

    def transition(self, target: BranchStatus) -> None:
        if target not in _BRANCH_TRANSITIONS[self._state]:
            raise RAGPlanError(
                ErrorCode.INTERNAL_ERROR,
                "invalid branch scheduler state transition",
                retryable=False,
            )
        self._state = target


@dataclass(frozen=True, slots=True)
class KillSwitchSnapshot:
    """One immutable ingress-time view of process environment kill switches."""

    force_vector_only: bool = False
    disable_cost_aware: bool = False

    @classmethod
    def from_environment(
        cls,
        environment: Mapping[str, str] | None = None,
    ) -> KillSwitchSnapshot:
        source = os.environ if environment is None else environment
        return cls(
            force_vector_only=_environment_bool(source, "RAGPLAN_FORCE_VECTOR_ONLY"),
            disable_cost_aware=_environment_bool(source, "RAGPLAN_DISABLE_COST_AWARE"),
        )

    @property
    def active(self) -> tuple[KillSwitch, ...]:
        return tuple(
            switch
            for switch, enabled in (
                (KillSwitch.FORCE_VECTOR_ONLY, self.force_vector_only),
                (KillSwitch.DISABLE_COST_AWARE, self.disable_cost_aware),
            )
            if enabled
        )


def _environment_bool(source: Mapping[str, str], key: str) -> bool:
    value = source.get(key, "").strip().casefold()
    if value in {"", "0", "false", "no", "off"}:
        return False
    if value in {"1", "true", "yes", "on"}:
        return True
    raise RAGPlanError(
        ErrorCode.INVALID_REQUEST,
        "runtime kill switch configuration is invalid",
        retryable=False,
    )


@dataclass(frozen=True, slots=True)
class CircuitPermit:
    state_before: CircuitState
    half_open_probe: bool


@dataclass(frozen=True, slots=True)
class CircuitSnapshot:
    state: CircuitState
    consecutive_failures: int
    half_open_probe_in_flight: bool
    open_remaining_ms: float


class CircuitOpenError(Exception):
    """Internal signal used to avoid executing a backend while its circuit is open."""


class CircuitBreaker:
    """Five-failure/30-second breaker with exactly one half-open probe."""

    def __init__(
        self,
        *,
        clock: MonotonicClock | None = None,
        failure_threshold: int = CIRCUIT_FAILURE_THRESHOLD,
        open_seconds: float = CIRCUIT_OPEN_SECONDS,
    ) -> None:
        if failure_threshold < 1 or open_seconds <= 0:
            raise ValueError("circuit limits must be positive")
        self._clock = clock if clock is not None else PerfCounterClock()
        self._threshold = failure_threshold
        self._open_ns = int(open_seconds * NANOSECONDS_PER_SECOND)
        self._state = CircuitState.CLOSED
        self._failures = 0
        self._open_until_ns = 0
        self._probe_in_flight = False
        self._lock = asyncio.Lock()

    async def acquire(self) -> CircuitPermit:
        async with self._lock:
            self._refresh()
            state_before = self._state
            if self._state is CircuitState.OPEN:
                raise CircuitOpenError
            if self._state is CircuitState.HALF_OPEN:
                if self._probe_in_flight:
                    raise CircuitOpenError
                self._probe_in_flight = True
                return CircuitPermit(state_before, True)
            return CircuitPermit(state_before, False)

    async def record_success(self, permit: CircuitPermit) -> CircuitState:
        async with self._lock:
            if permit.half_open_probe:
                self._probe_in_flight = False
            self._state = CircuitState.CLOSED
            self._failures = 0
            self._open_until_ns = 0
            return self._state

    async def record_failure(self, permit: CircuitPermit) -> CircuitState:
        async with self._lock:
            if permit.half_open_probe:
                self._probe_in_flight = False
                self._open()
                return self._state
            self._failures += 1
            if self._failures >= self._threshold:
                self._open()
            return self._state

    async def record_ignored(self, permit: CircuitPermit) -> CircuitState:
        """Release a half-open probe without counting an application cancellation."""

        async with self._lock:
            if permit.half_open_probe:
                self._probe_in_flight = False
                self._state = CircuitState.OPEN
                self._open_until_ns = self._clock.now_ns()
            return self._state

    async def snapshot(self) -> CircuitSnapshot:
        async with self._lock:
            self._refresh()
            remaining_ns = max(0, self._open_until_ns - self._clock.now_ns())
            return CircuitSnapshot(
                state=self._state,
                consecutive_failures=self._failures,
                half_open_probe_in_flight=self._probe_in_flight,
                open_remaining_ms=remaining_ns / 1_000_000,
            )

    def _refresh(self) -> None:
        if self._state is CircuitState.OPEN and self._clock.now_ns() >= self._open_until_ns:
            self._state = CircuitState.HALF_OPEN

    def _open(self) -> None:
        self._state = CircuitState.OPEN
        self._open_until_ns = self._clock.now_ns() + self._open_ns


__all__ = [
    "CIRCUIT_FAILURE_THRESHOLD",
    "CIRCUIT_OPEN_SECONDS",
    "BranchStateMachine",
    "CircuitBreaker",
    "CircuitOpenError",
    "CircuitPermit",
    "CircuitSnapshot",
    "KillSwitchSnapshot",
    "RequestStateMachine",
]
