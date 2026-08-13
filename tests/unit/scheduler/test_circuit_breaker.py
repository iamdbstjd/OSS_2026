from __future__ import annotations

import pytest

from ragplan.core.deadline import ManualClock
from ragplan.core.models import CircuitState
from ragplan.scheduler.states import CircuitBreaker, CircuitOpenError

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_circuit_closed_open_half_open_and_recovery() -> None:
    clock = ManualClock()
    circuit = CircuitBreaker(clock=clock)

    for failure in range(5):
        permit = await circuit.acquire()
        state = await circuit.record_failure(permit)
        assert state is (CircuitState.OPEN if failure == 4 else CircuitState.CLOSED)

    snapshot = await circuit.snapshot()
    assert snapshot.state is CircuitState.OPEN
    assert snapshot.consecutive_failures == 5
    with pytest.raises(CircuitOpenError):
        await circuit.acquire()

    clock.advance_ms(30_000)
    probe = await circuit.acquire()
    assert probe.half_open_probe is True
    with pytest.raises(CircuitOpenError):
        await circuit.acquire()

    assert await circuit.record_success(probe) is CircuitState.CLOSED
    recovered = await circuit.snapshot()
    assert recovered.consecutive_failures == 0
    assert recovered.state is CircuitState.CLOSED


@pytest.mark.asyncio
async def test_ignored_application_deadlines_do_not_increment_failures() -> None:
    circuit = CircuitBreaker(clock=ManualClock())
    permit = await circuit.acquire()
    assert await circuit.record_ignored(permit) is CircuitState.CLOSED
    assert (await circuit.snapshot()).consecutive_failures == 0
