from __future__ import annotations

import pytest

from ragplan.core.errors import ErrorCode, RAGPlanError
from ragplan.scheduler.executor import AdmissionController

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_admission_rejects_immediately_at_bound_and_recovers() -> None:
    admission = AdmissionController(limit=2)
    first = admission.slot()
    second = admission.slot()
    await first.__aenter__()
    await second.__aenter__()
    assert admission.in_flight == 2

    with pytest.raises(RAGPlanError) as error:
        async with admission.slot():
            pytest.fail("over-capacity request must not enter")
    assert error.value.code is ErrorCode.OVERLOADED
    assert error.value.retryable is True

    await second.__aexit__(None, None, None)
    async with admission.slot():
        assert admission.in_flight == 2
    await first.__aexit__(None, None, None)
    assert admission.in_flight == 0


@pytest.mark.asyncio
async def test_default_admission_accepts_exactly_thirty_two_without_queueing() -> None:
    admission = AdmissionController()
    slots = [admission.slot() for _ in range(32)]
    try:
        for slot in slots:
            await slot.__aenter__()
        assert admission.in_flight == 32
        with pytest.raises(RAGPlanError) as captured:
            await admission.slot().__aenter__()
        assert captured.value.code is ErrorCode.OVERLOADED
        assert admission.in_flight == 32
    finally:
        for slot in reversed(slots):
            await slot.__aexit__(None, None, None)
    assert admission.in_flight == 0
