"""Cancellation handling for remote runtime client operations."""

from __future__ import annotations

import asyncio
import logging

from collections.abc import Awaitable
from typing import TypeVar

from nexus.utils.event_loop import get_current_task_cancellation_count, get_current_task_name


_T = TypeVar('_T')


class RemoteChildOperationCancelledError(RuntimeError):
    """Report a cancelled remote operation whose caller is still running."""


async def await_remote_operation(
    operation: Awaitable[_T],
    *,
    logger: logging.Logger,
    phase: str,
    base_url: str,
    request_id: str | None = None,
) -> _T:
    """Await a remote operation without promoting child cancellation to caller cancellation."""
    try:
        return await operation
    except asyncio.CancelledError as error:
        cancellation_count = get_current_task_cancellation_count()
        logger.error(
            '[nexus-cancellation] phase=%s base_url=%s request_id=%s task_name=%s task_cancelling=%s',
            phase,
            base_url,
            request_id,
            get_current_task_name(),
            cancellation_count,
            exc_info=True,
        )
        if cancellation_count == 0:
            raise RemoteChildOperationCancelledError(
                'Remote child operation was cancelled without cancelling its caller: '
                f'phase={phase} request_id={request_id}'
            ) from error
        raise
