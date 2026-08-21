"""Parallel deadline scheduler with admission and deterministic branch outcomes."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Final

from ragplan.core.deadline import NANOSECONDS_PER_MILLISECOND, Deadline, DeadlineSnapshot
from ragplan.core.errors import ErrorCode, RAGPlanError, TimeoutOrigin
from ragplan.core.models import (
    BranchKind,
    BranchResult,
    BranchStatus,
    CancellationReason,
    CircuitState,
    FailureOrigin,
    GraphTrace,
    RetrievalHit,
)
from ragplan.scheduler.cancellation import cancel_and_await
from ragplan.scheduler.states import BranchStateMachine, CircuitBreaker, CircuitOpenError

DEFAULT_IN_FLIGHT_LIMIT: Final = 32
MAX_BACKEND_TASKS_PER_REQUEST: Final = 2


class AdmissionController:
    """Reject over-capacity requests immediately instead of queueing for a slot."""

    def __init__(self, limit: int = DEFAULT_IN_FLIGHT_LIMIT) -> None:
        if isinstance(limit, bool) or limit < 1:
            raise ValueError("admission limit must be a positive integer")
        self._limit = limit
        self._in_flight = 0
        self._lock = asyncio.Lock()

    @property
    def limit(self) -> int:
        return self._limit

    @property
    def in_flight(self) -> int:
        return self._in_flight

    @asynccontextmanager
    async def slot(self) -> AsyncIterator[None]:
        async with self._lock:
            if self._in_flight >= self._limit:
                raise RAGPlanError(ErrorCode.OVERLOADED, "request admission limit reached")
            self._in_flight += 1
        try:
            yield
        finally:
            async with self._lock:
                self._in_flight -= 1


@dataclass(frozen=True, slots=True)
class BranchPayload:
    hits: tuple[RetrievalHit, ...]
    graph_trace: GraphTrace | None = None


@dataclass(frozen=True, slots=True)
class BranchWork:
    branch: BranchKind
    operation: Callable[[], Awaitable[BranchPayload]]
    circuit: CircuitBreaker


@dataclass(frozen=True, slots=True)
class _BranchOutcome:
    result: BranchResult
    graph_trace: GraphTrace | None = None


@dataclass(slots=True)
class _BranchContext:
    cancellation_reason: CancellationReason | None = None


@dataclass(frozen=True, slots=True)
class SchedulerExecution:
    branch_results: tuple[BranchResult, ...]
    graph_trace: GraphTrace | None
    branch_start_skew_ms: float
    vector_circuit_state: CircuitState | None
    graph_circuit_state: CircuitState | None

    def result_for(self, branch: BranchKind) -> BranchResult | None:
        return next((result for result in self.branch_results if result.branch is branch), None)


class SchedulerExecutor:
    """Execute at most two active branches on one loop and one absolute deadline."""

    async def execute(
        self,
        works: Sequence[BranchWork],
        *,
        deadline: Deadline,
    ) -> SchedulerExecution:
        materialized = tuple(works)
        if not 1 <= len(materialized) <= MAX_BACKEND_TASKS_PER_REQUEST:
            raise ValueError("scheduler requires one or two backend tasks")
        kinds = tuple(work.branch for work in materialized)
        if len(set(kinds)) != len(kinds):
            raise ValueError("scheduler branches must be unique")

        barrier = asyncio.Event()
        contexts = {work.branch: _BranchContext() for work in materialized}
        tasks = {
            asyncio.create_task(
                self._run_branch(
                    work,
                    context=contexts[work.branch],
                    barrier=barrier,
                    deadline=deadline,
                ),
                name=f"ragplan-{work.branch.value}-branch",
            ): work.branch
            for work in materialized
        }
        barrier.set()
        pending = set(tasks)
        outcomes: dict[BranchKind, _BranchOutcome] = {}
        try:
            while pending:
                timeout_seconds = deadline.remaining_seconds(reserve_finalization=True)
                if timeout_seconds <= 0:
                    await self._expire_pending(
                        pending,
                        tasks,
                        contexts,
                        outcomes,
                        deadline=deadline,
                    )
                    break
                completed, still_pending = await asyncio.wait(
                    pending,
                    timeout=timeout_seconds,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if not completed:
                    await self._expire_pending(
                        pending,
                        tasks,
                        contexts,
                        outcomes,
                        deadline=deadline,
                    )
                    break
                for task in completed:
                    outcomes[tasks[task]] = task.result()
                pending = set(still_pending)
        except asyncio.CancelledError as exc:
            cancellation_reason = _parent_cancellation_reason(exc)
            for task in pending:
                contexts[tasks[task]].cancellation_reason = cancellation_reason
            await cancel_and_await(pending)
            raise
        except BaseException:
            for task in pending:
                contexts[tasks[task]].cancellation_reason = CancellationReason.PARENT_CANCELLED
            await cancel_and_await(pending)
            raise

        ordered_outcomes = tuple(outcomes[branch] for branch in kinds)
        results = tuple(outcome.result for outcome in ordered_outcomes)
        starts = tuple(
            result.started_at_ms for result in results if result.started_at_ms is not None
        )
        start_skew_ms = max(starts) - min(starts) if starts else 0.0
        graph_trace = next(
            (
                outcome.graph_trace
                for outcome in ordered_outcomes
                if outcome.result.branch is BranchKind.GRAPH
            ),
            None,
        )
        circuit_states = {result.branch: result.circuit_state_after for result in results}
        return SchedulerExecution(
            branch_results=results,
            graph_trace=graph_trace,
            branch_start_skew_ms=start_skew_ms,
            vector_circuit_state=circuit_states.get(BranchKind.VECTOR),
            graph_circuit_state=circuit_states.get(BranchKind.GRAPH),
        )

    async def _expire_pending(
        self,
        pending: set[asyncio.Task[_BranchOutcome]],
        tasks: dict[asyncio.Task[_BranchOutcome], BranchKind],
        contexts: dict[BranchKind, _BranchContext],
        outcomes: dict[BranchKind, _BranchOutcome],
        *,
        deadline: Deadline,
    ) -> None:
        for task in pending:
            contexts[tasks[task]].cancellation_reason = CancellationReason.APPLICATION_DEADLINE
        pending_tasks = tuple(pending)
        terminal = await cancel_and_await(pending_tasks)
        for task, outcome in zip(pending_tasks, terminal, strict=True):
            if isinstance(outcome, _BranchOutcome):
                outcomes[tasks[task]] = outcome
                continue
            if isinstance(outcome, asyncio.CancelledError):
                # The absolute deadline may already be exhausted before a newly
                # created task gets its first event-loop turn. In that case the
                # branch coroutine cannot translate cancellation itself, so the
                # scheduler must emit the same terminal timeout evidence here.
                boundary = deadline.snapshot()
                outcomes[tasks[task]] = _BranchOutcome(
                    self._result(
                        branch=tasks[task],
                        status=BranchStatus.TIMED_OUT,
                        started=boundary,
                        finished=boundary,
                        cancellation_reason=CancellationReason.APPLICATION_DEADLINE,
                        failure_origin=FailureOrigin.APPLICATION,
                        timeout_origin=TimeoutOrigin.APPLICATION_DEADLINE,
                    )
                )
                continue
            if isinstance(outcome, BaseException):
                raise outcome
            raise RAGPlanError(ErrorCode.INTERNAL_ERROR, "invalid scheduler cancellation result")

    async def _run_branch(
        self,
        work: BranchWork,
        *,
        context: _BranchContext,
        barrier: asyncio.Event,
        deadline: Deadline,
    ) -> _BranchOutcome:
        await barrier.wait()
        machine = BranchStateMachine()
        machine.transition(BranchStatus.RUNNING)
        started = deadline.snapshot()
        permit = None
        state_before: CircuitState | None = None
        state_after: CircuitState | None = None
        try:
            permit = await work.circuit.acquire()
            state_before = permit.state_before
            payload = await work.operation()
            finished = deadline.snapshot()
            if finished.now_ns >= deadline.branch_cutoff_ns:
                machine.transition(BranchStatus.TIMED_OUT)
                state_after = await work.circuit.record_ignored(permit)
                return _BranchOutcome(
                    self._result(
                        branch=work.branch,
                        status=machine.state,
                        started=started,
                        finished=finished,
                        cancellation_reason=CancellationReason.APPLICATION_DEADLINE,
                        failure_origin=FailureOrigin.APPLICATION,
                        timeout_origin=TimeoutOrigin.APPLICATION_DEADLINE,
                        circuit_state_before=state_before,
                        circuit_state_after=state_after,
                    )
                )
            machine.transition(BranchStatus.SUCCEEDED)
            state_after = await work.circuit.record_success(permit)
            return _BranchOutcome(
                self._result(
                    branch=work.branch,
                    status=machine.state,
                    started=started,
                    finished=finished,
                    hits=payload.hits,
                    circuit_state_before=state_before,
                    circuit_state_after=state_after,
                ),
                graph_trace=payload.graph_trace,
            )
        except CircuitOpenError:
            machine.transition(BranchStatus.FAILED)
            snapshot = await work.circuit.snapshot()
            finished = deadline.snapshot()
            return _BranchOutcome(
                self._result(
                    branch=work.branch,
                    status=machine.state,
                    started=started,
                    finished=finished,
                    error_code=ErrorCode.MODE_UNAVAILABLE,
                    failure_origin=FailureOrigin.CIRCUIT_OPEN,
                    circuit_state_before=snapshot.state,
                    circuit_state_after=snapshot.state,
                )
            )
        except RAGPlanError as exc:
            finished = deadline.snapshot()
            if exc.code is ErrorCode.DEADLINE_EXCEEDED:
                machine.transition(BranchStatus.TIMED_OUT)
                origin = exc.timeout_origin or TimeoutOrigin.APPLICATION_DEADLINE
                failure_origin = (
                    FailureOrigin.BACKEND_NATIVE
                    if origin is TimeoutOrigin.BACKEND_CLIENT
                    else FailureOrigin.APPLICATION
                )
                if permit is not None:
                    state_after = (
                        await work.circuit.record_failure(permit)
                        if origin is TimeoutOrigin.BACKEND_CLIENT
                        else await work.circuit.record_ignored(permit)
                    )
                return _BranchOutcome(
                    self._result(
                        branch=work.branch,
                        status=machine.state,
                        started=started,
                        finished=finished,
                        cancellation_reason=(
                            CancellationReason.APPLICATION_DEADLINE
                            if origin is TimeoutOrigin.APPLICATION_DEADLINE
                            else None
                        ),
                        failure_origin=failure_origin,
                        timeout_origin=origin,
                        circuit_state_before=state_before,
                        circuit_state_after=state_after,
                    )
                )
            machine.transition(BranchStatus.FAILED)
            counted = exc.code in {
                ErrorCode.DEPENDENCY_UNAVAILABLE,
                ErrorCode.RETRIEVAL_FAILED,
            }
            if permit is not None:
                state_after = (
                    await work.circuit.record_failure(permit)
                    if counted
                    else await work.circuit.record_ignored(permit)
                )
            return _BranchOutcome(
                self._result(
                    branch=work.branch,
                    status=machine.state,
                    started=started,
                    finished=finished,
                    error_code=exc.code,
                    failure_origin=(
                        FailureOrigin.BACKEND_NATIVE if counted else FailureOrigin.APPLICATION
                    ),
                    circuit_state_before=state_before,
                    circuit_state_after=state_after,
                )
            )
        except asyncio.CancelledError:
            finished = deadline.snapshot()
            reason = context.cancellation_reason or CancellationReason.PARENT_CANCELLED
            status = (
                BranchStatus.TIMED_OUT
                if reason is CancellationReason.APPLICATION_DEADLINE
                else BranchStatus.CANCELLED
            )
            machine.transition(status)
            if permit is not None:
                state_after = await work.circuit.record_ignored(permit)
            return _BranchOutcome(
                self._result(
                    branch=work.branch,
                    status=machine.state,
                    started=started,
                    finished=finished,
                    cancellation_reason=reason,
                    failure_origin=(
                        FailureOrigin.APPLICATION
                        if status is BranchStatus.TIMED_OUT
                        else FailureOrigin.CLIENT
                    ),
                    timeout_origin=(
                        TimeoutOrigin.APPLICATION_DEADLINE
                        if status is BranchStatus.TIMED_OUT
                        else None
                    ),
                    circuit_state_before=state_before,
                    circuit_state_after=state_after,
                )
            )
        except Exception:
            finished = deadline.snapshot()
            machine.transition(BranchStatus.FAILED)
            if permit is not None:
                state_after = await work.circuit.record_failure(permit)
            return _BranchOutcome(
                self._result(
                    branch=work.branch,
                    status=machine.state,
                    started=started,
                    finished=finished,
                    error_code=ErrorCode.DEPENDENCY_UNAVAILABLE,
                    failure_origin=FailureOrigin.BACKEND_NATIVE,
                    circuit_state_before=state_before,
                    circuit_state_after=state_after,
                )
            )

    @staticmethod
    def _result(
        *,
        branch: BranchKind,
        status: BranchStatus,
        started: DeadlineSnapshot,
        finished: DeadlineSnapshot,
        hits: tuple[RetrievalHit, ...] = (),
        error_code: ErrorCode | None = None,
        cancellation_reason: CancellationReason | None = None,
        failure_origin: FailureOrigin | None = None,
        timeout_origin: TimeoutOrigin | None = None,
        circuit_state_before: CircuitState | None = None,
        circuit_state_after: CircuitState | None = None,
    ) -> BranchResult:
        start_ms = started.elapsed_ns / NANOSECONDS_PER_MILLISECOND
        end_ms = finished.elapsed_ns / NANOSECONDS_PER_MILLISECOND
        return BranchResult(
            branch=branch,
            status=status,
            latency_ms=max(0.0, end_ms - start_ms),
            hits=hits,
            error_code=error_code,
            started_at_ms=start_ms,
            ended_at_ms=end_ms,
            remaining_budget_at_start_ms=started.remaining_ms,
            remaining_budget_at_end_ms=finished.remaining_ms,
            cancellation_reason=cancellation_reason,
            failure_origin=failure_origin,
            timeout_origin=timeout_origin,
            circuit_state_before=circuit_state_before,
            circuit_state_after=circuit_state_after,
        )


def _parent_cancellation_reason(error: asyncio.CancelledError) -> CancellationReason:
    message = error.args[0] if error.args else None
    if message == CancellationReason.CLIENT_DISCONNECT.value:
        return CancellationReason.CLIENT_DISCONNECT
    if message == CancellationReason.ENGINE_SHUTDOWN.value:
        return CancellationReason.ENGINE_SHUTDOWN
    return CancellationReason.PARENT_CANCELLED


__all__ = [
    "DEFAULT_IN_FLIGHT_LIMIT",
    "MAX_BACKEND_TASKS_PER_REQUEST",
    "AdmissionController",
    "BranchPayload",
    "BranchWork",
    "SchedulerExecution",
    "SchedulerExecutor",
]
