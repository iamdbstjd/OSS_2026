from types import MappingProxyType

import pytest

from ragplan.core.errors import (
    HTTP_STATUS_BY_ERROR,
    RETRYABLE_BY_ERROR,
    ErrorCode,
    RAGPlanError,
)

pytestmark = pytest.mark.unit


def test_error_taxonomy_has_stable_http_and_retry_mappings() -> None:
    assert isinstance(HTTP_STATUS_BY_ERROR, MappingProxyType)
    assert isinstance(RETRYABLE_BY_ERROR, MappingProxyType)
    assert HTTP_STATUS_BY_ERROR[ErrorCode.DEADLINE_EXCEEDED] == 504
    assert RETRYABLE_BY_ERROR[ErrorCode.OVERLOADED] is True
    with pytest.raises(TypeError):
        HTTP_STATUS_BY_ERROR[ErrorCode.INVALID_QUERY] = 500  # type: ignore[index]

    assert {code.value: HTTP_STATUS_BY_ERROR[code] for code in ErrorCode} == {
        "INVALID_REQUEST": 422,
        "INVALID_QUERY": 422,
        "PLAN_INVARIANT_VIOLATION": 422,
        "NOT_READY": 503,
        "DEPENDENCY_UNAVAILABLE": 503,
        "MODE_UNAVAILABLE": 503,
        "OVERLOADED": 503,
        "DEADLINE_EXCEEDED": 504,
        "MODEL_INCOMPATIBLE": 503,
        "CORPUS_INCONSISTENT": 503,
        "RETRIEVAL_FAILED": 503,
        "INTERNAL_ERROR": 500,
    }
    assert {code.value: RETRYABLE_BY_ERROR[code] for code in ErrorCode} == {
        "INVALID_REQUEST": False,
        "INVALID_QUERY": False,
        "PLAN_INVARIANT_VIOLATION": False,
        "NOT_READY": True,
        "DEPENDENCY_UNAVAILABLE": True,
        "MODE_UNAVAILABLE": True,
        "OVERLOADED": True,
        "DEADLINE_EXCEEDED": True,
        "MODEL_INCOMPATIBLE": False,
        "CORPUS_INCONSISTENT": False,
        "RETRIEVAL_FAILED": True,
        "INTERNAL_ERROR": False,
    }


def test_public_error_response_is_stable_and_safe() -> None:
    error = RAGPlanError(ErrorCode.INVALID_QUERY, "query must not be empty")
    response = error.response("request-1")
    assert error.http_status == 422
    assert response.model_dump() == {
        "code": ErrorCode.INVALID_QUERY,
        "message": "query must not be empty",
        "request_id": "request-1",
        "retryable": False,
    }
