"""Fixed Stage 1 request limits and validation helpers."""

from __future__ import annotations

from typing import Final

from ragplan.core.errors import ErrorCode, RAGPlanError

MAX_REQUEST_BODY_BYTES: Final = 32 * 1024
DEFAULT_LATENCY_BUDGET_MS: Final = 200
MIN_LATENCY_BUDGET_MS: Final = 25
MAX_LATENCY_BUDGET_MS: Final = 5_000
MIN_TOP_K: Final = 1
MAX_TOP_K: Final = 50
DEFAULT_TOP_K: Final = 10
MIN_QUERY_CODEPOINTS: Final = 1
MAX_QUERY_CODEPOINTS: Final = 4_096
EMBEDDING_MODEL_ID: Final = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_MODEL_REVISION: Final = "b8903db39f65d93ae28d49a37c4f3fa90c5f94e0"
EMBEDDING_DIMENSION: Final = 384


def validate_request_body_size(body: bytes) -> bytes:
    """Return *body* unless it exceeds the public request-body limit."""
    if len(body) > MAX_REQUEST_BODY_BYTES:
        msg = f"request body exceeds {MAX_REQUEST_BODY_BYTES} bytes"
        raise RAGPlanError(ErrorCode.INVALID_REQUEST, msg)
    return body
