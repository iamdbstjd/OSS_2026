"""Shared backend lifecycle result types."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


class BackendHealthStatus(StrEnum):
    """Storage dependency state without leaking implementation exceptions."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True, slots=True)
class BackendHealth:
    """A small typed health result shared by vector and graph adapters."""

    status: BackendHealthStatus
    detail: str | None = None

    @property
    def available(self) -> bool:
        return self.status is not BackendHealthStatus.UNAVAILABLE


@dataclass(frozen=True, slots=True)
class BackendWriteResult:
    """Versioned ingestion result used for later dual-store reconciliation."""

    corpus_version: str
    written_count: int
    canonical_id_checksum: str

    def __post_init__(self) -> None:
        if not self.corpus_version.strip():
            raise ValueError("corpus_version must not be blank")
        if self.written_count < 0:
            raise ValueError("written_count must be non-negative")
        if _SHA256_PATTERN.fullmatch(self.canonical_id_checksum) is None:
            raise ValueError("canonical_id_checksum must be a lowercase SHA-256 hex digest")
