"""Stable public failure contracts and immutable HTTP mappings."""

from __future__ import annotations

from enum import StrEnum
from types import MappingProxyType
from typing import Final

from pydantic import BaseModel, ConfigDict


class ErrorCode(StrEnum):
    INVALID_REQUEST = "INVALID_REQUEST"
    INVALID_QUERY = "INVALID_QUERY"
    PLAN_INVARIANT_VIOLATION = "PLAN_INVARIANT_VIOLATION"
    NOT_READY = "NOT_READY"
    DEPENDENCY_UNAVAILABLE = "DEPENDENCY_UNAVAILABLE"
    MODE_UNAVAILABLE = "MODE_UNAVAILABLE"
    OVERLOADED = "OVERLOADED"
    DEADLINE_EXCEEDED = "DEADLINE_EXCEEDED"
    MODEL_INCOMPATIBLE = "MODEL_INCOMPATIBLE"
    CORPUS_INCONSISTENT = "CORPUS_INCONSISTENT"
    RETRIEVAL_FAILED = "RETRIEVAL_FAILED"
    INTERNAL_ERROR = "INTERNAL_ERROR"


HTTP_STATUS_BY_ERROR: Final = MappingProxyType(
    {
        ErrorCode.INVALID_REQUEST: 422,
        ErrorCode.INVALID_QUERY: 422,
        ErrorCode.PLAN_INVARIANT_VIOLATION: 422,
        ErrorCode.NOT_READY: 503,
        ErrorCode.DEPENDENCY_UNAVAILABLE: 503,
        ErrorCode.MODE_UNAVAILABLE: 503,
        ErrorCode.OVERLOADED: 503,
        ErrorCode.DEADLINE_EXCEEDED: 504,
        ErrorCode.MODEL_INCOMPATIBLE: 503,
        ErrorCode.CORPUS_INCONSISTENT: 503,
        ErrorCode.RETRIEVAL_FAILED: 503,
        ErrorCode.INTERNAL_ERROR: 500,
    }
)
RETRYABLE_BY_ERROR: Final = MappingProxyType(
    {
        code: code
        in {
            ErrorCode.NOT_READY,
            ErrorCode.DEPENDENCY_UNAVAILABLE,
            ErrorCode.MODE_UNAVAILABLE,
            ErrorCode.OVERLOADED,
            ErrorCode.DEADLINE_EXCEEDED,
            ErrorCode.RETRIEVAL_FAILED,
        }
        for code in ErrorCode
    }
)


class ErrorResponse(BaseModel):
    """Safe public error body; deliberately contains no exception detail."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    code: ErrorCode
    message: str
    request_id: str
    retryable: bool


class RAGPlanError(Exception):
    """Exception carrying an intentionally stable public error code."""

    def __init__(self, code: ErrorCode, message: str, *, retryable: bool | None = None) -> None:
        self.code = code
        self.message = message
        self.retryable = RETRYABLE_BY_ERROR[code] if retryable is None else retryable
        super().__init__(message)

    @property
    def http_status(self) -> int:
        return HTTP_STATUS_BY_ERROR[self.code]

    def response(self, request_id: str) -> ErrorResponse:
        return ErrorResponse(
            code=self.code,
            message=self.message,
            request_id=request_id,
            retryable=self.retryable,
        )
