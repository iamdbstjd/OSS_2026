from __future__ import annotations

import asyncio
import time

import pytest

from ragplan.core.deadline import Deadline, ManualClock
from ragplan.core.errors import ErrorCode, RAGPlanError, TimeoutOrigin
from ragplan.core.models import (
    BranchKind,
    BranchStatus,
    CancellationReason,
    FailureOrigin,
    RetrievalHit,
)
from ragplan.scheduler.cancellation import run_until_disconnect
from ragplan.scheduler.executor import BranchPayload, BranchWork, SchedulerExecutor
from ragplan.scheduler.states import CircuitBreaker

pytestmark = pytest.mark.unit


def _hit(branch: BranchKind) -> RetrievalHit:
    return RetrievalHit(
        canonical_chunk_id=f"{branch.value}-hit",
        text=f"{branch.value} evidence",
        score=1.0,
        source=branch.value,
        rank=1,
    )


def _work(
    branch: BranchKind,
    operation: object,
    *,
    circuit: CircuitBreaker | None = None,
) -> BranchWork:
    return BranchWork(
        branch=branch,
        operation=operation,  # type: ignore[arg-type]
        circuit=circuit if circuit is not None else CircuitBreaker(),
    )


@pytest.mark.asyncio
async def test_scheduler_rejects_zero_or_more_than_two_backend_tasks() -> None:
    async def operation() -> BranchPayload:
        raise AssertionError("invalid task sets must not run")

    scheduler = SchedulerExecutor()
    with pytest.raises(ValueError, match="one or two"):
        await scheduler.execute((), deadline=Deadline.start(200))
    with pytest.raises(ValueError, match="one or two"):
        await scheduler.execute(
            (
                _work(BranchKind.VECTOR, operation),
                _work(BranchKind.GRAPH, operation),
                _work(BranchKind.VECTOR, operation),
            ),
            deadline=Deadline.start(200),
        )


@pytest.mark.asyncio
async def test_two_branches_start_together_and_wall_clock_is_parallel() -> None:
    starts: dict[BranchKind, float] = {}

    async def vector() -> BranchPayload:
        starts[BranchKind.VECTOR] = time.perf_counter()
        await asyncio.sleep(0.03)
        return BranchPayload((_hit(BranchKind.VECTOR),))

    async def graph() -> BranchPayload:
        starts[BranchKind.GRAPH] = time.perf_counter()
        await asyncio.sleep(0.08)
        return BranchPayload((_hit(BranchKind.GRAPH),))

    started = time.perf_counter()
    execution = await SchedulerExecutor().execute(
        (_work(BranchKind.VECTOR, vector), _work(BranchKind.GRAPH, graph)),
        deadline=Deadline.start(500),
    )
    elapsed = time.perf_counter() - started

    branch_latency_sum_ms = sum(result.latency_ms or 0.0 for result in execution.branch_results)
    assert elapsed < 0.15
    assert elapsed * 1000 < branch_latency_sum_ms - 10
    assert abs(starts[BranchKind.VECTOR] - starts[BranchKind.GRAPH]) < 0.02
    assert {result.status for result in execution.branch_results} == {BranchStatus.SUCCEEDED}
    assert execution.branch_start_skew_ms < 20


@pytest.mark.asyncio
async def test_application_timeout_cancels_and_awaits_only_pending_branch() -> None:
    vector_cleaned = asyncio.Event()

    async def vector() -> BranchPayload:
        try:
            await asyncio.Event().wait()
        finally:
            vector_cleaned.set()
        raise AssertionError("unreachable")

    async def graph() -> BranchPayload:
        await asyncio.sleep(0.002)
        return BranchPayload((_hit(BranchKind.GRAPH),))

    execution = await SchedulerExecutor().execute(
        (_work(BranchKind.VECTOR, vector), _work(BranchKind.GRAPH, graph)),
        deadline=Deadline.start(25),
    )
    vector_result = execution.result_for(BranchKind.VECTOR)
    graph_result = execution.result_for(BranchKind.GRAPH)
    assert vector_result is not None and vector_result.status is BranchStatus.TIMED_OUT
    assert vector_result.cancellation_reason is CancellationReason.APPLICATION_DEADLINE
    assert vector_result.timeout_origin is TimeoutOrigin.APPLICATION_DEADLINE
    assert graph_result is not None and graph_result.status is BranchStatus.SUCCEEDED
    assert vector_cleaned.is_set()


@pytest.mark.asyncio
async def test_backend_error_does_not_discard_sibling_success_and_has_zero_retry() -> None:
    calls = 0

    async def vector() -> BranchPayload:
        nonlocal calls
        calls += 1
        raise RAGPlanError(ErrorCode.DEPENDENCY_UNAVAILABLE, "backend unavailable")

    async def graph() -> BranchPayload:
        return BranchPayload((_hit(BranchKind.GRAPH),))

    execution = await SchedulerExecutor().execute(
        (_work(BranchKind.VECTOR, vector), _work(BranchKind.GRAPH, graph)),
        deadline=Deadline.start(200),
    )
    vector_result = execution.result_for(BranchKind.VECTOR)
    graph_result = execution.result_for(BranchKind.GRAPH)
    assert calls == 1
    assert vector_result is not None and vector_result.status is BranchStatus.FAILED
    assert vector_result.failure_origin is FailureOrigin.BACKEND_NATIVE
    assert graph_result is not None and graph_result.status is BranchStatus.SUCCEEDED


@pytest.mark.asyncio
async def test_backend_client_timeout_is_distinct_and_counts_circuit_failure() -> None:
    circuit = CircuitBreaker(clock=ManualClock())

    async def timed_out() -> BranchPayload:
        raise RAGPlanError(
            ErrorCode.DEADLINE_EXCEEDED,
            "backend client timeout",
            timeout_origin=TimeoutOrigin.BACKEND_CLIENT,
        )

    execution = await SchedulerExecutor().execute(
        (_work(BranchKind.VECTOR, timed_out, circuit=circuit),),
        deadline=Deadline.start(200),
    )
    result = execution.branch_results[0]
    assert result.status is BranchStatus.TIMED_OUT
    assert result.timeout_origin is TimeoutOrigin.BACKEND_CLIENT
    assert result.cancellation_reason is None
    assert (await circuit.snapshot()).consecutive_failures == 1


@pytest.mark.asyncio
async def test_parent_cancellation_awaits_all_children_without_orphans() -> None:
    started = {BranchKind.VECTOR: asyncio.Event(), BranchKind.GRAPH: asyncio.Event()}
    cleaned = {BranchKind.VECTOR: asyncio.Event(), BranchKind.GRAPH: asyncio.Event()}

    def operation(branch: BranchKind) -> object:
        async def wait_forever() -> BranchPayload:
            started[branch].set()
            try:
                await asyncio.Event().wait()
            finally:
                cleaned[branch].set()
            raise AssertionError("unreachable")

        return wait_forever

    scheduler_task = asyncio.create_task(
        SchedulerExecutor().execute(
            (
                _work(BranchKind.VECTOR, operation(BranchKind.VECTOR)),
                _work(BranchKind.GRAPH, operation(BranchKind.GRAPH)),
            ),
            deadline=Deadline.start(5000),
        )
    )
    await asyncio.gather(*(event.wait() for event in started.values()))
    scheduler_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await scheduler_task

    assert all(event.is_set() for event in cleaned.values())
    assert not [
        task
        for task in asyncio.all_tasks()
        if task is not asyncio.current_task()
        and not task.done()
        and task.get_name().startswith("ragplan-")
    ]


@pytest.mark.asyncio
async def test_client_disconnect_cancels_and_awaits_request_operation() -> None:
    started = asyncio.Event()
    cleaned = asyncio.Event()
    disconnected = asyncio.Event()

    async def operation() -> str:
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaned.set()
        raise AssertionError("unreachable")

    task = asyncio.create_task(
        run_until_disconnect(
            operation(),
            wait_for_disconnect=disconnected.wait,
        )
    )
    await started.wait()
    disconnected.set()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert cleaned.is_set()
    assert not [
        pending
        for pending in asyncio.all_tasks()
        if pending is not asyncio.current_task()
        and not pending.done()
        and pending.get_name().startswith("ragplan-")
    ]
