"""Bounded asynchronous JSONL trace logging with fail-open I/O isolation."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Final, Protocol

from ragplan.core.errors import ErrorCode, RAGPlanError
from ragplan.core.models import SearchResponse

TRACE_MAX_BYTES: Final = 10 * 1024 * 1024
TRACE_FILE_COUNT: Final = 5
TRACE_BACKUP_COUNT: Final = TRACE_FILE_COUNT - 1
TRACE_QUEUE_CAPACITY: Final = 1024


@dataclass(frozen=True, slots=True)
class TraceLoggingConfig:
    path: Path
    max_bytes: int = TRACE_MAX_BYTES
    file_count: int = TRACE_FILE_COUNT
    queue_capacity: int = TRACE_QUEUE_CAPACITY
    mode: str = "redacted"

    def __post_init__(self) -> None:
        if self.mode != "redacted":
            raise RAGPlanError(
                ErrorCode.INVALID_REQUEST,
                "service trace logging supports redacted mode only",
                retryable=False,
            )
        if self.max_bytes != TRACE_MAX_BYTES or self.file_count != TRACE_FILE_COUNT:
            raise ValueError("trace rotation is fixed at 10 MiB across five files")
        if self.queue_capacity < 1:
            raise ValueError("trace queue capacity must be positive")

    @classmethod
    def from_environment(
        cls,
        environment: Mapping[str, str] | None = None,
    ) -> TraceLoggingConfig:
        source = os.environ if environment is None else environment
        mode = source.get("RAGPLAN_LOGGING__MODE", "redacted").strip().casefold()
        if mode != "redacted":
            raise RAGPlanError(
                ErrorCode.INVALID_REQUEST,
                "service trace logging supports redacted mode only",
                retryable=False,
            )
        default_path = Path(tempfile.gettempdir()) / "ragplan" / "ragplan-trace.jsonl"
        configured_path = source.get("RAGPLAN_LOGGING__PATH", "").strip()
        path = Path(configured_path) if configured_path else default_path
        return cls(path=path, mode=mode)


@dataclass(frozen=True, slots=True)
class TraceWriterStats:
    written: int
    failures: int
    dropped: int


class TraceWriter(Protocol):
    async def start(self) -> None: ...

    async def close(self) -> None: ...

    def record_search(self, response: SearchResponse) -> None: ...

    def record_error(
        self,
        *,
        request_id: str,
        error_code: ErrorCode,
        requested_planner: str | None,
    ) -> None: ...

    def stats(self) -> TraceWriterStats: ...


class _StrictRotatingFileHandler(RotatingFileHandler):
    def handleError(self, record: logging.LogRecord) -> None:  # noqa: N802
        del record
        raise RuntimeError("redacted trace file write failed")


class RedactedTraceWriter:
    """Never block or fail retrieval on queue, rotation, or filesystem errors."""

    def __init__(
        self,
        config: TraceLoggingConfig,
        *,
        on_failure: Callable[[], None] | None = None,
        on_drop: Callable[[], None] | None = None,
    ) -> None:
        self._config = config
        self._on_failure = on_failure
        self._on_drop = on_drop
        self._queue: asyncio.Queue[dict[str, object] | None] | None = None
        self._worker: asyncio.Task[None] | None = None
        self._handler: _StrictRotatingFileHandler | None = None
        self._written = 0
        self._failures = 0
        self._dropped = 0
        self._closed = False

    @property
    def config(self) -> TraceLoggingConfig:
        return self._config

    async def start(self) -> None:
        if self._worker is not None or self._closed:
            return
        try:
            self._config.path.parent.mkdir(parents=True, exist_ok=True)
            self._handler = _StrictRotatingFileHandler(
                self._config.path,
                maxBytes=self._config.max_bytes,
                backupCount=self._config.file_count - 1,
                encoding="utf-8",
                delay=True,
            )
            self._handler.setFormatter(logging.Formatter("%(message)s"))
            self._queue = asyncio.Queue(maxsize=self._config.queue_capacity)
            self._worker = asyncio.create_task(self._run(), name="ragplan-redacted-trace-writer")
        except Exception:
            self._mark_failure()

    def record_search(self, response: SearchResponse) -> None:
        trace = response.trace
        self._enqueue(
            {
                "schema_version": "service_trace_v1",
                "event": "search_complete",
                "timestamp_utc": _timestamp(),
                "request_id": response.request_id,
                "query_hash": trace.query_hash,
                "query_length": trace.query_length,
                "status": response.status.value,
                "requested_planner": response.planner_decision.mode.value,
                "effective_planner": (
                    response.planner_decision.effective_mode.value
                    if response.planner_decision.effective_mode is not None
                    else None
                ),
                "selected_plan_id": response.planner_decision.selected_plan_id,
                "branch_results": [
                    {
                        "branch": item.branch.value,
                        "status": item.status.value,
                        "latency_ms": item.latency_ms,
                        "error_code": item.error_code.value if item.error_code else None,
                        "hit_count": item.hit_count,
                    }
                    for item in trace.branch_results
                ],
                "total_latency_ms": trace.total_latency_ms,
                "latency_budget_ms": trace.latency_budget_ms,
                "budget_violated": trace.budget_violated,
                "fallback": response.fallback,
                "fallback_reason": response.planner_decision.fallback_reason,
                "result_count": len(response.results),
                "corpus_version": trace.corpus_version,
                "config_version": trace.config_version,
                "model_version": response.planner_decision.model_version,
            }
        )

    def record_error(
        self,
        *,
        request_id: str,
        error_code: ErrorCode,
        requested_planner: str | None,
    ) -> None:
        self._enqueue(
            {
                "schema_version": "service_trace_v1",
                "event": "search_error",
                "timestamp_utc": _timestamp(),
                "request_id": request_id,
                "error_code": error_code.value,
                "requested_planner": requested_planner,
            }
        )

    def stats(self) -> TraceWriterStats:
        return TraceWriterStats(
            written=self._written,
            failures=self._failures,
            dropped=self._dropped,
        )

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._queue is not None and self._worker is not None:
            await self._queue.put(None)
            await self._worker
        if self._handler is not None:
            try:
                self._handler.close()
            except Exception:
                self._mark_failure()
        self._handler = None
        self._worker = None
        self._queue = None

    def _enqueue(self, event: dict[str, object]) -> None:
        if self._queue is None or self._closed:
            return
        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull:
            self._dropped += 1
            if self._on_drop is not None:
                self._on_drop()

    async def _run(self) -> None:
        assert self._queue is not None
        while True:
            event = await self._queue.get()
            try:
                if event is None:
                    return
                serialized = json.dumps(
                    event,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                    allow_nan=False,
                )
                self._write(serialized)
                self._written += 1
            except Exception:
                self._mark_failure()
            finally:
                self._queue.task_done()

    def _write(self, serialized: str) -> None:
        if self._handler is None:
            raise RuntimeError("redacted trace handler is unavailable")
        record = logging.LogRecord(
            name="ragplan.trace",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg=serialized,
            args=(),
            exc_info=None,
        )
        self._handler.emit(record)

    def _mark_failure(self) -> None:
        self._failures += 1
        if self._on_failure is not None:
            self._on_failure()


def _timestamp() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


__all__ = [
    "TRACE_BACKUP_COUNT",
    "TRACE_FILE_COUNT",
    "TRACE_MAX_BYTES",
    "TRACE_QUEUE_CAPACITY",
    "RedactedTraceWriter",
    "TraceLoggingConfig",
    "TraceWriter",
    "TraceWriterStats",
]
