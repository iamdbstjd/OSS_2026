from __future__ import annotations

import pytest

from ragplan.planner.runtime_guard import RuntimeGuardObservation, RuntimeModelGuard

pytestmark = pytest.mark.unit


def _observation(
    *,
    artifact: str = "artifact-v1",
    violated: bool = False,
    underpredicted: bool = False,
) -> RuntimeGuardObservation:
    return RuntimeGuardObservation(
        artifact_version=artifact,
        predicted_p95_latency_ms=100.0,
        actual_execution_latency_ms=101.0 if underpredicted else 90.0,
        budget_violated=violated,
    )


def test_guard_waits_for_twenty_samples_and_uses_strict_thresholds() -> None:
    guard = RuntimeModelGuard("artifact-v1")
    for index in range(19):
        snapshot = guard.observe(_observation(violated=index < 3))
    assert snapshot.disabled is False

    # Exactly 10% violations and 20% underprediction must remain enabled.
    guard = RuntimeModelGuard("artifact-v1")
    for index in range(20):
        snapshot = guard.observe(_observation(violated=index < 2, underpredicted=index < 4))
    assert snapshot.budget_violation_rate == 0.10
    assert snapshot.p95_underprediction_rate == 0.20
    assert snapshot.disabled is False


def test_guard_latches_budget_violation_and_does_not_auto_reenable() -> None:
    guard = RuntimeModelGuard("artifact-v1")
    for index in range(20):
        snapshot = guard.observe(_observation(violated=index < 3))

    assert snapshot.disabled is True
    assert snapshot.disable_reason == "budget_violation_rate_gt_0.10"
    assert snapshot.fallback_mode == "rule"
    frozen_count = snapshot.observation_count
    for _ in range(100):
        snapshot = guard.observe(_observation())
    assert snapshot.disabled is True
    assert snapshot.observation_count == frozen_count


def test_guard_latches_underprediction_and_requires_new_artifact() -> None:
    guard = RuntimeModelGuard("artifact-v1")
    for index in range(20):
        snapshot = guard.observe(_observation(underpredicted=index < 5))
    assert snapshot.disabled is True
    assert snapshot.disable_reason == "p95_underprediction_rate_gt_0.20"

    with pytest.raises(ValueError, match="cannot reactivate"):
        guard.replace_artifact("artifact-v1")
    reset = guard.replace_artifact("artifact-v2")
    assert reset.disabled is False
    assert reset.observation_count == 0
    assert reset.disable_reason is None


def test_guard_rejects_observations_for_another_artifact() -> None:
    guard = RuntimeModelGuard("artifact-v1")
    with pytest.raises(ValueError, match="different artifact"):
        guard.observe(_observation(artifact="artifact-v2"))
