"""Replay guard for Socket.IO RPC retries.

When the host disconnects after the runtime has already started processing an
RPC, the client may reconnect and resend the same request. This helper keeps
the in-flight/completed result keyed by ``request_id`` so retries do not
re-execute side effects.
"""

from __future__ import annotations

import asyncio
import logging
import time

from dataclasses import dataclass
from typing import TypeVar


logger = logging.getLogger(__name__)

ResultT = TypeVar('ResultT')


@dataclass
class _ReplayEntry:
    created_at: float
    future: asyncio.Future[object]


def _mark_future_retrieved(future: asyncio.Future[object]) -> None:
    """Consume a replay future's outcome so asyncio never logs a spurious error.

    The owner sets the result/exception on its replay future but never awaits it
    itself -- the future exists ONLY to replay the outcome to a duplicate/resent
    request. When the handler raises and no duplicate ever arrives (the common
    case), the exception-bearing future is later dropped by ``_prune_locked`` and
    garbage-collected; asyncio's ``Future.__del__`` then logs "Future exception
    was never retrieved" for every such failure. Retrieving the exception here (as
    a done-callback) marks it handled without changing behavior: a real duplicate
    still gets the exception re-raised through ``asyncio.shield(future)`` on the
    duplicate branch, and calling ``exception()`` more than once is safe.
    """
    if future.cancelled():
        return
    # Returns the stored exception (retrieving it) or None for a normal result.
    future.exception()


class RPCReplayGuard:
    """Deduplicate retried RPC requests by request id."""

    def __init__(self, ttl_seconds: float = 3600.0, max_entries: int = 2048) -> None:
        self._ttl_seconds = ttl_seconds
        self._max_entries = max_entries
        self._entries: dict[str, _ReplayEntry] = {}
        self._discard_tasks: set[asyncio.Task[None]] = set()
        self._lock = asyncio.Lock()

    async def run(self, scope: str, request_id: str | None, handler) -> ResultT:
        """Run *handler* once and replay the cached result for duplicates."""
        if not request_id:
            return await handler()

        cache_key = f'{scope}:{request_id}'
        owner = False
        future: asyncio.Future[object]

        async with self._lock:
            self._prune_locked()
            entry = self._entries.get(cache_key)
            if entry is None:
                future = asyncio.get_running_loop().create_future()
                # The owner never awaits this future (it only replays to duplicates),
                # so retrieve its outcome on completion to avoid a bogus
                # "Future exception was never retrieved" log when the handler fails
                # and no duplicate resend ever consumes the exception.
                future.add_done_callback(_mark_future_retrieved)
                self._entries[cache_key] = _ReplayEntry(created_at=time.monotonic(), future=future)
                owner = True
            else:
                future = entry.future

        if not owner:
            status = 'completed' if future.done() else 'in-flight'
            logger.warning('%s: duplicate request_id=%s detected, replaying %s result', scope, request_id, status)
            return await asyncio.shield(future)

        try:
            result = await handler()
        except asyncio.CancelledError:
            if not future.done():
                future.cancel()
            await self._discard_entry(cache_key, future)
            raise
        except Exception as exc:
            if not future.done():
                future.set_exception(exc)
            raise

        if not future.done():
            future.set_result(result)
        return result

    async def _discard_entry(self, cache_key: str, future: asyncio.Future[object]) -> None:
        async with self._lock:
            entry = self._entries.get(cache_key)
            if entry is not None and entry.future is future:
                self._entries.pop(cache_key, None)

    async def _discard_after(self, cache_key: str, future: asyncio.Future[object], delay: float) -> None:
        try:
            await asyncio.sleep(delay)
        except asyncio.CancelledError:
            return

        async with self._lock:
            entry = self._entries.get(cache_key)
            if entry is not None and entry.future is future:
                self._entries.pop(cache_key, None)

    async def discard(self, scope: str, request_id: str | None, delay: float = 0.0) -> bool:
        """Discard a completed replay result now or after a grace period.

        Delayed deletion is tied to the current replay future so it can never
        remove a newer entry that later reuses the same request id. Returns whether
        a completed entry was deleted or scheduled for deletion.
        """
        if not request_id:
            return False

        cache_key = f'{scope}:{request_id}'
        async with self._lock:
            entry = self._entries.get(cache_key)
            if entry is None or not entry.future.done():
                return False

            if delay <= 0:
                self._entries.pop(cache_key, None)
            else:
                task = asyncio.create_task(self._discard_after(cache_key, entry.future, delay))
                # The event loop keeps only weak references to tasks. Retain this fire-and-forget
                # task until completion, then remove it from the set to avoid accumulating tasks.
                self._discard_tasks.add(task)
                task.add_done_callback(self._discard_tasks.discard)
            return True

    def _prune_locked(self) -> None:
        now = time.monotonic()
        expired_keys = [
            key
            for key, entry in self._entries.items()
            if entry.future.done() and now - entry.created_at > self._ttl_seconds
        ]
        for key in expired_keys:
            self._entries.pop(key, None)

        if len(self._entries) <= self._max_entries:
            return

        for key in list(self._entries.keys()):
            if len(self._entries) <= self._max_entries:
                break
            entry = self._entries.get(key)
            if entry is not None and entry.future.done():
                self._entries.pop(key, None)
