"""Cancellation helpers that always await every child task."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterable


async def cancel_and_await[T](
    tasks: Iterable[asyncio.Task[T]],
    *,
    cancel_message: str | None = None,
) -> tuple[T | BaseException, ...]:
    """Cancel unfinished tasks and consume every terminal result without orphaning work."""

    materialized = tuple(tasks)
    for task in materialized:
        if not task.done():
            task.cancel(cancel_message)
    if not materialized:
        return ()
    return tuple(await asyncio.gather(*materialized, return_exceptions=True))


async def run_until_disconnect[T](
    operation: Awaitable[T],
    *,
    wait_for_disconnect: Callable[[], Awaitable[None]],
) -> T:
    """Cancel and await request work as soon as the ASGI client disconnects."""

    async def run_operation() -> T:
        return await operation

    operation_task: asyncio.Task[T] = asyncio.create_task(
        run_operation(),
        name="ragplan-request-operation",
    )

    async def watch_disconnect() -> None:
        await wait_for_disconnect()

    watcher: asyncio.Task[None] = asyncio.create_task(
        watch_disconnect(),
        name="ragplan-client-disconnect",
    )
    try:
        completed, _ = await asyncio.wait(
            {operation_task, watcher},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if operation_task in completed:
            watcher.cancel()
            await cancel_and_await((watcher,))
            return operation_task.result()
        await cancel_and_await(
            (operation_task,),
            cancel_message="client_disconnect",
        )
        raise asyncio.CancelledError("HTTP client disconnected")
    except BaseException:
        await cancel_and_await((operation_task, watcher))
        raise


__all__ = ["cancel_and_await", "run_until_disconnect"]
