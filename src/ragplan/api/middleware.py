"""Small ASGI guards that run before the engine deadline starts."""

from __future__ import annotations

from uuid import uuid4

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from ragplan.core.config import MAX_REQUEST_BODY_BYTES
from ragplan.core.errors import ErrorCode, RAGPlanError


class RequestBodyLimitMiddleware:
    """Buffer and bound HTTP request bytes before request DTO validation.

    Reading the body here deliberately precedes ``RAGPlanEngine.search()`` so HTTP
    transfer time remains outside ADR-010's engine processing deadline.
    """

    def __init__(self, app: ASGIApp, max_body_bytes: int = MAX_REQUEST_BODY_BYTES) -> None:
        if max_body_bytes < 1:
            raise ValueError("max_body_bytes must be positive")
        self.app = app
        self.max_body_bytes = max_body_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = request_id_from_scope(scope)
        content_length = _content_length(scope)
        if content_length is None:
            await _send_invalid_request(send, request_id, "invalid Content-Length header")
            return
        if content_length > self.max_body_bytes:
            await _send_too_large(send, request_id, self.max_body_bytes)
            return

        messages: list[Message] = []
        total_bytes = 0
        while True:
            message = await receive()
            messages.append(message)
            if message["type"] == "http.disconnect":
                break
            if message["type"] != "http.request":
                continue
            total_bytes += len(message.get("body", b""))
            if total_bytes > self.max_body_bytes:
                await _send_too_large(send, request_id, self.max_body_bytes)
                return
            if not message.get("more_body", False):
                break

        replay = iter(messages)

        async def replay_receive() -> Message:
            try:
                return next(replay)
            except StopIteration:
                # Once FastAPI has consumed the buffered request body, retain
                # the live ASGI receive channel so client disconnect can cancel
                # the shared retrieval scheduler and all backend children.
                return await receive()

        await self.app(scope, replay_receive, send)


def _content_length(scope: Scope) -> int | None:
    values = [value for name, value in scope["headers"] if name.lower() == b"content-length"]
    if not values:
        return 0
    if len(values) != 1:
        return None
    try:
        parsed = int(values[0])
    except ValueError:
        return None
    return parsed if parsed >= 0 else None


def request_id_from_scope(scope: Scope) -> str:
    """Return a bounded printable request ID or create a local one."""

    for name, value in scope["headers"]:
        if name.lower() == b"x-request-id":
            candidate = str(value.decode("latin-1").strip())
            if candidate and len(candidate) <= 128 and candidate.isprintable():
                return candidate
    return uuid4().hex


async def _send_too_large(send: Send, request_id: str, maximum: int) -> None:
    await _send_invalid_request(
        send,
        request_id,
        f"request body exceeds {maximum} bytes",
    )


async def _send_invalid_request(send: Send, request_id: str, message: str) -> None:
    error = RAGPlanError(ErrorCode.INVALID_REQUEST, message)
    body = error.response(request_id).model_dump_json().encode("utf-8")
    await send(
        {
            "type": "http.response.start",
            "status": error.http_status,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode("ascii")),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})
