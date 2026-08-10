from __future__ import annotations

import pytest

from ragplan.core.deadline import Deadline, ManualClock, finalization_reserve_ms

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("budget_ms", "expected_reserve_ms"),
    [(25, 5.0), (100, 5.0), (200, 10.0), (400, 20.0), (5000, 20.0)],
)
def test_finalization_reserve_formula(budget_ms: int, expected_reserve_ms: float) -> None:
    assert finalization_reserve_ms(budget_ms) == expected_reserve_ms


def test_deadline_uses_one_absolute_monotonic_timeline() -> None:
    clock = ManualClock(1_000_000_000)
    deadline = Deadline.start(200, clock=clock)

    assert deadline.start_ns == 1_000_000_000
    assert deadline.absolute_ns == 1_200_000_000
    assert deadline.branch_cutoff_ns == 1_190_000_000

    clock.advance_ms(35.5)
    snapshot = deadline.snapshot()

    assert snapshot.elapsed_ms == 35.5
    assert snapshot.remaining_ms == 164.5
    assert snapshot.branch_remaining_ms == 154.5
    assert deadline.remaining_seconds(reserve_finalization=True) == 0.1545


def test_deadline_has_no_hidden_grace_at_budget_boundary() -> None:
    clock = ManualClock()
    deadline = Deadline.start(100, clock=clock)

    clock.advance_ms(100)
    at_boundary = deadline.snapshot()
    assert at_boundary.expired is True
    assert at_boundary.budget_violated is False
    assert at_boundary.remaining_ns == 0

    clock.advance_ns(1)
    after_boundary = deadline.snapshot()
    assert after_boundary.expired is True
    assert after_boundary.budget_violated is True


def test_branch_time_ends_before_request_deadline() -> None:
    clock = ManualClock()
    deadline = Deadline.start(50, clock=clock)

    clock.advance_ms(45)
    assert deadline.snapshot().branch_remaining_ns == 0
    assert deadline.snapshot().remaining_ms == 5.0


@pytest.mark.parametrize("budget", [24, 5001, True, 25.0])
def test_invalid_budget_is_rejected(budget: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        Deadline.start(budget)  # type: ignore[arg-type]


def test_manual_clock_cannot_move_backwards() -> None:
    clock = ManualClock()

    with pytest.raises(ValueError, match="backwards"):
        clock.advance_ns(-1)


def test_deadline_constructor_cannot_bypass_absolute_invariants() -> None:
    clock = ManualClock()

    with pytest.raises(ValueError, match="absolute deadline"):
        Deadline(
            clock=clock,
            start_ns=0,
            absolute_ns=1,
            branch_cutoff_ns=0,
            budget_ms=100,
            finalization_reserve_ns=5_000_000,
        )
