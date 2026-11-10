"""异步预留句柄：单 pool / 多 pool。"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from .exceptions import AccountPoolError


if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import AsyncIterator

    from .acquired import AcquiredAccount
    from .manager import AccountPoolManager


class AsyncReservation:
    """Async batch reservation handle.

    Obtained via :meth:`AccountPoolManager.reservation()`.  Supports
    ``async with rsv.acquire()`` to obtain accounts from the reserved batch.

    The reservation is automatically cancelled when the outer ``async with``
    block exits.
    """

    def __init__(
        self,
        mgr: AccountPoolManager,
        pool: str,
        token: str,
        account_ids: list[int],
        *,
        ttl_seconds: int = 60,
        enable_heartbeat: bool = True,
    ) -> None:
        self._mgr = mgr
        self.pool = pool
        self.token = token
        self.account_ids = list(account_ids)
        self._ttl = ttl_seconds
        self._enable_heartbeat = enable_heartbeat

    @asynccontextmanager
    async def acquire(self, *, ttl_seconds: int | None = None) -> AsyncIterator[AcquiredAccount]:
        """Acquire a single account from this reservation batch.

        Yields:
            :class:`AcquiredAccount`
        """
        account = await self._mgr.acquire(
            self.pool,
            ttl_seconds=ttl_seconds or self._ttl,
            reservation_token=self.token,
            enable_heartbeat=self._enable_heartbeat,
        )
        try:
            yield account
        finally:
            await account.release()


class AsyncMultiReservation:
    """Async multi-pool atomic reservation handle.

    Obtained via :meth:`AccountPoolManager.multi_reservation()`.  All pools
    share a single ``reservation_token``.  ``acquire(pool)`` picks from the
    specified pool's reserved batch.
    """

    def __init__(
        self,
        mgr: AccountPoolManager,
        token: str,
        account_ids_by_pool: dict[str, list[int]],
        *,
        ttl_seconds: int = 60,
        enable_heartbeat: bool = True,
    ) -> None:
        self._mgr = mgr
        self.token = token
        self.account_ids_by_pool: dict[str, list[int]] = {k: list(v) for k, v in account_ids_by_pool.items()}
        self._ttl = ttl_seconds
        self._enable_heartbeat = enable_heartbeat

    @property
    def pools(self) -> list[str]:
        """List of pool names in this reservation."""
        return list(self.account_ids_by_pool.keys())

    @asynccontextmanager
    async def acquire(self, pool: str, *, ttl_seconds: int | None = None) -> AsyncIterator[AcquiredAccount]:
        """Acquire a single account from *pool* within this reservation batch.

        Raises:
            AccountPoolError: If *pool* is not part of this reservation.
        """
        if pool not in self.account_ids_by_pool:
            raise AccountPoolError(f'pool {pool!r} not in this reservation')

        account = await self._mgr.acquire(
            pool,
            ttl_seconds=ttl_seconds or self._ttl,
            reservation_token=self.token,
            enable_heartbeat=self._enable_heartbeat,
        )
        try:
            yield account
        finally:
            await account.release()
