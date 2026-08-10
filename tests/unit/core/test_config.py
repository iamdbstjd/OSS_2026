import pytest

from ragplan.core.config import MAX_REQUEST_BODY_BYTES, validate_request_body_size
from ragplan.core.errors import ErrorCode, RAGPlanError

pytestmark = pytest.mark.unit


def test_request_body_size_boundary() -> None:
    accepted = b"x" * MAX_REQUEST_BODY_BYTES
    assert validate_request_body_size(accepted) is accepted
    with pytest.raises(RAGPlanError, match="exceeds") as exc_info:
        validate_request_body_size(accepted + b"x")
    assert exc_info.value.code is ErrorCode.INVALID_REQUEST
