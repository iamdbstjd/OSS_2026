"""Monotonic absolute-deadline primitives shared by every online stage."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from decimal import Decimal
from typing import Protocol, runtime_checkable

from ragplan.core.config import (
    DEFAULT_LATENCY_BUDGET_MS,
    MAX_LATENCY_BUDGET_MS,
    MIN_LATENCY_BUDGET_MS,
)

NANOSECONDS_PER_MILLISECOND = 1_000_000
NANOSECONDS_PER_SECOND = 1_000_000_000


@runtime_checkable
class MonotonicClock(Protocol):
    """A clock suitable for elapsed-time and timeout calculations."""

    def now_ns(self) -> int:
        """Return an arbitrary, monotonically non-decreasing nanosecond value."""


@dataclass(frozen=True, slots=True)
class PerfCounterClock:
    """Production monotonic clock backed by :func:`time.perf_counter_ns`."""

    def now_ns(self) -> int:
        return time.perf_counter_ns()


@dataclass(slots=True)
class ManualClock:
    """Deterministic injectable clock for tests and simulations."""

    _now_ns: int = 0

    def __post_init__(self) -> None:
        if self._now_ns < 0:
            raise ValueError("clock value must be non-negative")

    def now_ns(self) -> int:
        return self._now_ns

    def advance_ns(self, nanoseconds: int) -> None:
        if nanoseconds < 0:
            raise ValueError("a monotonic clock cannot move backwards")
        self._now_ns += nanoseconds

    def advance_ms(self, milliseconds: int | float) -> None:
        value = float(milliseconds)
        if not math.isfinite(value) or value < 0:
            raise ValueError("milliseconds must be finite and non-negative")
        nanoseconds = int(Decimal(str(milliseconds)) * NANOSECONDS_PER_MILLISECOND)
        self.advance_ns(nanoseconds)


def finalization_reserve_ms(latency_budget_ms: int) -> float:
    """Return the ADR-010 DTO/finalization reserve for a request budget."""

    _validate_budget(latency_budget_ms)
    return min(20.0, max(5.0, latency_budget_ms * 0.05))


@dataclass(frozen=True, slots=True)
class DeadlineSnapshot:
    """One internally consistent observation of a deadline."""

    now_ns: int
    elapsed_ns: int
    remaining_ns: int
    branch_remaining_ns: int
    expired: bool
    budget_violated: bool

    @property
    def elapsed_ms(self) -> float:
        return self.elapsed_ns / NANOSECONDS_PER_MILLISECOND

    @property
    def remaining_ms(self) -> float:
        return self.remaining_ns / NANOSECONDS_PER_MILLISECOND

    @property
    def branch_remaining_ms(self) -> float:
        return self.branch_remaining_ns / NANOSECONDS_PER_MILLISECOND


@dataclass(frozen=True, slots=True)
class Deadline:
    """An absolute soft deadline measured exclusively with one monotonic clock."""

    clock: MonotonicClock
    start_ns: int
    absolute_ns: int
    branch_cutoff_ns: int
    budget_ms: int
    finalization_reserve_ns: int

    def __post_init__(self) -> None:
        _validate_budget(self.budget_ms)
        if self.start_ns < 0:
            raise ValueError("deadline start must be non-negative")
        expected_reserve_ns = int(
            Decimal(str(finalization_reserve_ms(self.budget_ms))) * NANOSECONDS_PER_MILLISECOND
        )
        expected_absolute_ns = self.start_ns + self.budget_ms * NANOSECONDS_PER_MILLISECOND
        if self.finalization_reserve_ns != expected_reserve_ns:
            raise ValueError("finalization reserve does not match the request budget")
        if self.absolute_ns != expected_absolute_ns:
            raise ValueError("absolute deadline does not match start plus request budget")
        if self.branch_cutoff_ns != self.absolute_ns - self.finalization_reserve_ns:
            raise ValueError("branch cutoff must preserve the finalization reserve")

    @classmethod
    def start(
        cls,
        latency_budget_ms: int = DEFAULT_LATENCY_BUDGET_MS,
        *,
        clock: MonotonicClock | None = None,
    ) -> Deadline:
        _validate_budget(latency_budget_ms)
        selected_clock = clock if clock is not None else PerfCounterClock()
        start_ns = selected_clock.now_ns()
        reserve_ns = int(
            Decimal(str(finalization_reserve_ms(latency_budget_ms))) * NANOSECONDS_PER_MILLISECOND
        )
        absolute_ns = start_ns + latency_budget_ms * NANOSECONDS_PER_MILLISECOND
        return cls(
            clock=selected_clock,
            start_ns=start_ns,
            absolute_ns=absolute_ns,
            branch_cutoff_ns=absolute_ns - reserve_ns,
            budget_ms=latency_budget_ms,
            finalization_reserve_ns=reserve_ns,
        )

    @property
    def finalization_reserve_ms(self) -> float:
        return self.finalization_reserve_ns / NANOSECONDS_PER_MILLISECOND

    def snapshot(self) -> DeadlineSnapshot:
        now_ns = self.clock.now_ns()
        elapsed_ns = max(0, now_ns - self.start_ns)
        remaining_ns = max(0, self.absolute_ns - now_ns)
        branch_remaining_ns = max(0, self.branch_cutoff_ns - now_ns)
        return DeadlineSnapshot(
            now_ns=now_ns,
            elapsed_ns=elapsed_ns,
            remaining_ns=remaining_ns,
            branch_remaining_ns=branch_remaining_ns,
            expired=now_ns >= self.absolute_ns,
            budget_violated=now_ns > self.absolute_ns,
        )

    def remaining_seconds(self, *, reserve_finalization: bool = False) -> float:
        snapshot = self.snapshot()
        remaining_ns = (
            snapshot.branch_remaining_ns if reserve_finalization else snapshot.remaining_ns
        )
        return remaining_ns / NANOSECONDS_PER_SECOND


def _validate_budget(latency_budget_ms: int) -> None:
    if isinstance(latency_budget_ms, bool) or not isinstance(latency_budget_ms, int):
        raise TypeError("latency budget must be an integer number of milliseconds")
    if not MIN_LATENCY_BUDGET_MS <= latency_budget_ms <= MAX_LATENCY_BUDGET_MS:
        raise ValueError(
            f"latency budget must be between {MIN_LATENCY_BUDGET_MS} and {MAX_LATENCY_BUDGET_MS} ms"
        )
