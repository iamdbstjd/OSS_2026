from __future__ import annotations

import pytest

from ragplan.core.deadline import Deadline, ManualClock
from ragplan.core.errors import ErrorCode, RAGPlanError
from ragplan.core.models import BranchStatus, KillSwitch, RequestState
from ragplan.scheduler.states import BranchStateMachine, KillSwitchSnapshot, RequestStateMachine

pytestmark = pytest.mark.unit


def test_request_state_machine_records_complete_frozen_path() -> None:
    clock = ManualClock()
    machine = RequestStateMachine(Deadline.start(200, clock=clock))
    for state in (
        RequestState.ANALYZING,
        RequestState.PLANNING,
        RequestState.EXECUTING,
        RequestState.FUSING,
        RequestState.COMPLETE,
    ):
        clock.advance_ms(1)
        machine.transition(state)

    assert tuple(event.state for event in machine.events) == (
        RequestState.RECEIVED,
        RequestState.ANALYZING,
        RequestState.PLANNING,
        RequestState.EXECUTING,
        RequestState.FUSING,
        RequestState.COMPLETE,
    )
    assert [event.elapsed_ms for event in machine.events] == [0, 1, 2, 3, 4, 5]


def test_invalid_request_and_branch_transitions_are_internal_errors() -> None:
    request = RequestStateMachine(Deadline.start(200, clock=ManualClock()))
    with pytest.raises(RAGPlanError) as request_error:
        request.transition(RequestState.EXECUTING)
    assert request_error.value.code is ErrorCode.INTERNAL_ERROR

    branch = BranchStateMachine()
    branch.transition(BranchStatus.RUNNING)
    branch.transition(BranchStatus.SUCCEEDED)
    with pytest.raises(RAGPlanError) as branch_error:
        branch.transition(BranchStatus.FAILED)
    assert branch_error.value.code is ErrorCode.INTERNAL_ERROR


def test_kill_switch_snapshot_is_strict_and_immutable() -> None:
    snapshot = KillSwitchSnapshot.from_environment(
        {
            "RAGPLAN_FORCE_VECTOR_ONLY": "true",
            "RAGPLAN_DISABLE_COST_AWARE": "0",
        }
    )
    assert snapshot.active == (KillSwitch.FORCE_VECTOR_ONLY,)

    with pytest.raises(RAGPlanError) as error:
        KillSwitchSnapshot.from_environment({"RAGPLAN_FORCE_VECTOR_ONLY": "sometimes"})
    assert error.value.code is ErrorCode.INVALID_REQUEST
