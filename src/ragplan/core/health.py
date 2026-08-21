"""Backend-neutral runtime capability and readiness snapshots."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from ragplan.backends.base import BackendHealth
from ragplan.core.models import PlannerMode


class RuntimeProfile(StrEnum):
    VECTOR_STAGED = "vector_staged"
    VECTOR_ACTIVE = "vector_active"
    GRAPH_ACTIVE = "graph_active"
    DUAL_STORE_ACTIVE = "dual_store_active"


@dataclass(frozen=True, slots=True)
class EngineReadinessSnapshot:
    """One privacy-safe observation of a configured engine and its dependencies."""

    profile: RuntimeProfile
    corpus_version: str
    active_corpus: bool
    supported_modes: tuple[PlannerMode, ...]
    vector: BackendHealth | None = None
    graph: BackendHealth | None = None

    def __post_init__(self) -> None:
        if not self.corpus_version.strip():
            raise ValueError("readiness corpus version must be non-empty")
        if not self.supported_modes or len(set(self.supported_modes)) != len(self.supported_modes):
            raise ValueError("readiness supported modes must be non-empty and unique")
        expected = {
            RuntimeProfile.VECTOR_STAGED: (True, False, False),
            RuntimeProfile.VECTOR_ACTIVE: (True, False, True),
            RuntimeProfile.GRAPH_ACTIVE: (False, True, True),
            RuntimeProfile.DUAL_STORE_ACTIVE: (True, True, True),
        }[self.profile]
        observed = (self.vector is not None, self.graph is not None, self.active_corpus)
        if observed != expected:
            raise ValueError("readiness dependencies do not match the runtime profile")


__all__ = ["EngineReadinessSnapshot", "RuntimeProfile"]
