"""Latched Stage 12 calibration guard for cost-aware policy observations."""

from __future__ import annotations

from collections import deque
from typing import Annotated, Final, Literal

from pydantic import Field

from ragplan.core.models import FrozenModel, NonEmptyString, NonNegativeFloat, UnitFloat

GUARD_WINDOW_SIZE: Final = 100
GUARD_MINIMUM_SAMPLES: Final = 20
BUDGET_VIOLATION_THRESHOLD: Final = 0.10
P95_UNDERPREDICTION_THRESHOLD: Final = 0.20


class RuntimeGuardObservation(FrozenModel):
    artifact_version: NonEmptyString
    predicted_p95_latency_ms: NonNegativeFloat
    actual_execution_latency_ms: NonNegativeFloat
    budget_violated: bool


class RuntimeGuardSnapshot(FrozenModel):
    schema_version: Literal["cost_model_guard_v1"] = "cost_model_guard_v1"
    artifact_version: NonEmptyString
    window_size: Literal[100] = GUARD_WINDOW_SIZE
    minimum_samples: Literal[20] = GUARD_MINIMUM_SAMPLES
    observation_count: Annotated[int, Field(ge=0, le=GUARD_WINDOW_SIZE)]
    budget_violation_rate: UnitFloat
    p95_underprediction_rate: UnitFloat
    disabled: bool
    disable_reason: str | None = None
    fallback_mode: Literal["rule"] | None = None


class RuntimeModelGuard:
    """Disable one artifact for the process lifetime after calibration drift."""

    def __init__(self, artifact_version: str) -> None:
        if not artifact_version.strip():
            raise ValueError("runtime guard artifact version must be non-empty")
        self._artifact_version = artifact_version
        self._observations: deque[RuntimeGuardObservation] = deque(maxlen=GUARD_WINDOW_SIZE)
        self._disabled = False
        self._disable_reason: str | None = None

    @property
    def artifact_version(self) -> str:
        return self._artifact_version

    @property
    def disabled(self) -> bool:
        return self._disabled

    def observe(self, observation: RuntimeGuardObservation) -> RuntimeGuardSnapshot:
        if observation.artifact_version != self._artifact_version:
            raise ValueError("runtime guard observation belongs to a different artifact")
        if not self._disabled:
            self._observations.append(observation)
            self._evaluate()
        return self.snapshot()

    def snapshot(self) -> RuntimeGuardSnapshot:
        count = len(self._observations)
        budget_rate = (
            sum(item.budget_violated for item in self._observations) / count if count else 0.0
        )
        underprediction_rate = (
            sum(
                item.actual_execution_latency_ms > item.predicted_p95_latency_ms
                for item in self._observations
            )
            / count
            if count
            else 0.0
        )
        return RuntimeGuardSnapshot(
            artifact_version=self._artifact_version,
            observation_count=count,
            budget_violation_rate=budget_rate,
            p95_underprediction_rate=underprediction_rate,
            disabled=self._disabled,
            disable_reason=self._disable_reason,
            fallback_mode="rule" if self._disabled else None,
        )

    def replace_artifact(self, artifact_version: str) -> RuntimeGuardSnapshot:
        """Explicit operator action; the only operation that can clear a latch."""

        selected = artifact_version.strip()
        if not selected:
            raise ValueError("replacement artifact version must be non-empty")
        if selected == self._artifact_version:
            raise ValueError("a disabled artifact cannot reactivate itself")
        self._artifact_version = selected
        self._observations.clear()
        self._disabled = False
        self._disable_reason = None
        return self.snapshot()

    def _evaluate(self) -> None:
        snapshot = self.snapshot()
        if snapshot.observation_count < GUARD_MINIMUM_SAMPLES:
            return
        reasons: list[str] = []
        if snapshot.budget_violation_rate > BUDGET_VIOLATION_THRESHOLD:
            reasons.append("budget_violation_rate_gt_0.10")
        if snapshot.p95_underprediction_rate > P95_UNDERPREDICTION_THRESHOLD:
            reasons.append("p95_underprediction_rate_gt_0.20")
        if reasons:
            self._disabled = True
            self._disable_reason = "+".join(reasons)


__all__ = [
    "BUDGET_VIOLATION_THRESHOLD",
    "GUARD_MINIMUM_SAMPLES",
    "GUARD_WINDOW_SIZE",
    "P95_UNDERPREDICTION_THRESHOLD",
    "RuntimeGuardObservation",
    "RuntimeGuardSnapshot",
    "RuntimeModelGuard",
]
