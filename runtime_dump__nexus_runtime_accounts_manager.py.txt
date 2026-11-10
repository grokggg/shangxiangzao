"""``AccountPoolManager``：账号池的异步生命周期管理器。

包装 :class:`AsyncAccountPoolClient` 并管理 acquire/release 的账号状态与心跳协程。
"""

from __future__ import annotations

import asyncio
import logging
import random

from contextlib import asynccontextmanager, suppress
from typing import TYPE_CHECKING, Any

from .acquired import AcquiredAccount
from .client import AsyncAccountPoolClient
from .exceptions import AccountPoolError
from .reservation import AsyncMultiReservation, AsyncReservation


if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import AsyncIterator

    from .models import Account, PoolStats, ReserveResult


logger = logging.getLogger(__name__)

# Default heartbeat interval divisor: interval = ttl / _HEARTBEAT_DIVISOR
_HEARTBEAT_DIVISOR = 3.0
_HEARTBEAT_MIN_INTERVAL = 1.0
# Heartbeat retry policy on transient errors (network / 5xx / connection reset).
# 4xx client errors are treated as permanent (lease gone) and give up immediately.
_HEARTBEAT_MAX_RETRIES = 3
_HEARTBEAT_BASE_BACKOFF = 1.0


class AccountPoolManager:
    """Manages account pool lifecycle within an async execution context.

    Wraps :class:`AsyncAccountPoolClient` (aiohttp-based) and provides async
    APIs plus automatic asyncio heartbeat management for acquired leases.

    Usually not instantiated directly — use :meth:`ExecutionContext.get_account`
    instead.

    管理后台：https://nexus-pool.woa.com/ （建池、加号、打 tag）。
    建新池还是打 tag 见包文档 :mod:`nexus.runtime.accounts`。

    Args:
        endpoint: Account pool server URL.  Defaults to the
            ``NEXUS_ACCOUNT_POOL_ENDPOINT`` env var or
            ``http://nexus-pool.production.polaris:8080``.
        ttl_seconds: Default lease TTL for acquired accounts.
    """

    def __init__(
        self,
        endpoint: str | None = None,
        *,
        ttl_seconds: int = 60,
    ) -> None:
        self._client = AsyncAccountPoolClient(endpoint)
        self._default_ttl = int(ttl_seconds)
        # lease_id → AcquiredAccount
        self._accounts: dict[str, AcquiredAccount] = {}
        # lease_id → asyncio.Task
        self._heartbeat_tasks: dict[str, asyncio.Task[None]] = {}

    # ------------------------------------------------------------------
    # 流转 (circulation)
    # ------------------------------------------------------------------

    async def acquire(
        self,
        pool: str,
        *,
        ttl_seconds: int | None = None,
        reservation_token: str = '',
        enable_heartbeat: bool = True,
        tags: list[str] | None = None,
    ) -> AcquiredAccount:
        """Acquire an account from a pool.

        Args:
            pool: Pool name (e.g. ``'github'``).
            ttl_seconds: Lease TTL (defaults to manager's default).
            reservation_token: Token from a prior reservation. When non-empty,
                subsequent ``release`` automatically returns the account to
                ``reserved`` (kept in the batch); use ``cancel_reservation`` to
                fully release the whole batch back to ``idle``.
            enable_heartbeat: Whether to start asyncio heartbeat (default True).
            tags: 可选关键字参数 ``list[str] | None``。传参示例
                ``await mgr.acquire('gui-sandbox', tags=['chrome', 'vscode'])``。
                AND 匹配：账号必须同时具备列表里每一个非空 tag。不传 / ``None`` / ``[]``
                不按 tag 筛。账号上多出来的 tag 不妨碍匹配。

        Raises:
            NoAvailableAccountError: No available account in the pool.
            AcquireError: Other acquire failure (including missing/disabled pool).
        """
        ttl = int(ttl_seconds) if ttl_seconds is not None else self._default_ttl
        return await self._acquire_any(
            pool,
            ttl,
            reservation_token,
            enable_heartbeat=enable_heartbeat,
            tags=tags,
        )

    async def release(self, identifier: str) -> None:
        """Release an acquired account.

        *identifier* may be:
        - A ``lease_id`` returned by :attr:`AcquiredAccount.lease_id`.
        - A ``pool`` name — releases the latest account acquired from that pool.
        """
        account = self._accounts.pop(identifier, None)
        if account is None:
            account = self._pop_by_pool(identifier)
        if account is None or account._released or account._invalidated:
            return
        self._stop_heartbeat(account.lease_id)
        # 若 acquire 时携带了 reservation_token，release 时保留在 batch 中（回到 reserved）；
        # 需要彻底回 idle 请调用 cancel_reservation。
        token = account._reservation_token
        try:
            await self._client.release(account.lease_id, reservation_token=token)
            account._released = True
            logger.info('Released account %s (pool=%s, key=%s)', identifier, account.pool, account.account_key)
        except AccountPoolError as exc:
            logger.warning('Release failed for %s: %s', identifier, exc)

    async def release_all(self) -> None:
        """Release all acquired accounts and stop all heartbeat tasks."""
        for ident in list(self._accounts.keys()):
            await self.release(ident)

    async def heartbeat(self, lease_id: str, *, ttl_seconds: int = 60) -> str:
        """Renew a lease. Returns the new expiry time string from the server."""
        return await self._client.heartbeat(lease_id, ttl_seconds=ttl_seconds)

    # ------------------------------------------------------------------
    # 批量预留 (reservation) — direct calls
    # ------------------------------------------------------------------

    async def reserve(
        self,
        pool: str,
        count: int,
        *,
        ttl_seconds: int = 300,
        tags: list[str] | None = None,
    ) -> ReserveResult:
        """Reserve *count* accounts from a single pool (all-or-nothing).

        Args:
            pool: Pool name.
            count: How many accounts to reserve (must be > 0).
            ttl_seconds: Reservation TTL in seconds.
            tags: Same as :meth:`acquire`. Example::

                    await mgr.reserve('gui-sandbox', 3, tags=['chrome'])
        """
        return await self._client.reserve(pool, count, ttl_seconds=ttl_seconds, tags=tags)

    async def reserve_multi(self, items: list[dict[str, Any]], *, ttl_seconds: int = 300) -> ReserveResult:
        """Atomically reserve accounts across multiple pools.

        Each item in *items* is a dict with ``pool`` (str), ``count`` (int),
        and optional ``tags`` (``list[str]``). Example::

            await mgr.reserve_multi([
                {'pool': 'github', 'count': 2},
                {'pool': 'gui-sandbox', 'count': 1, 'tags': ['chrome']},
            ])
        """
        return await self._client.reserve_multi(items, ttl_seconds=ttl_seconds)

    async def cancel_reservation(self, reservation_token: str) -> None:
        """Cancel a reservation batch (all acquired/reserved → idle)."""
        await self._client.cancel_reservation(reservation_token)

    # ------------------------------------------------------------------
    # 批量预留 — async context managers
    # ------------------------------------------------------------------

    @asynccontextmanager
    async def reservation(
        self,
        pool: str,
        count: int,
        *,
        ttl_seconds: int = 300,
        lease_ttl_seconds: int = 60,
        enable_heartbeat: bool = True,
    ) -> AsyncIterator[AsyncReservation]:
        """Async context manager for single-pool batch reservation.

        Usage::

            async with mgr.reservation('github', count=3) as rsv:
                async with rsv.acquire() as act:
                    print(act.tokens)

        The reservation is automatically cancelled on exit.
        """
        result = await self.reserve(pool, count, ttl_seconds=ttl_seconds)
        rsv = AsyncReservation(
            self,
            pool,
            result.reservation_token,
            result.account_ids,
            ttl_seconds=lease_ttl_seconds,
            enable_heartbeat=enable_heartbeat,
        )
        try:
            yield rsv
        finally:
            try:
                await self.cancel_reservation(rsv.token)
            except AccountPoolError as exc:
                logger.warning('cancel_reservation failed on exit: %s', exc)

    @asynccontextmanager
    async def multi_reservation(
        self,
        items: list[dict[str, Any]],
        *,
        ttl_seconds: int = 300,
        lease_ttl_seconds: int = 60,
        enable_heartbeat: bool = True,
    ) -> AsyncIterator[AsyncMultiReservation]:
        """Async context manager for multi-pool atomic reservation."""
        result = await self.reserve_multi(items, ttl_seconds=ttl_seconds)
        rsv = AsyncMultiReservation(
            self,
            result.reservation_token,
            result.account_ids_by_pool,
            ttl_seconds=lease_ttl_seconds,
            enable_heartbeat=enable_heartbeat,
        )
        try:
            yield rsv
        finally:
            try:
                await self.cancel_reservation(rsv.token)
            except AccountPoolError as exc:
                logger.warning('cancel_reservation failed on exit: %s', exc)

    # ------------------------------------------------------------------
    # 管理 (admin)
    # ------------------------------------------------------------------

    async def add_account(
        self,
        pool: str,
        account_key: str,
        credentials: Any = None,
        tokens: Any = None,
        *,
        shared_cap: int = 0,
        note: str = '',
        tags: list[str] | None = None,
    ) -> int:
        """Add a new account to the pool, optionally with tags. Returns new account ID."""
        return await self._client.add_account(
            pool,
            account_key,
            credentials=credentials,
            tokens=tokens,
            shared_cap=shared_cap,
            note=note,
            tags=tags,
        )

    async def delete_account(self, account_id: int) -> None:
        """Delete an account by ID."""
        await self._client.delete_account(account_id)

    async def list_accounts(
        self,
        *,
        pool: str = '',
        status: str = '',
        offset: int = 0,
        limit: int = 20,
    ) -> list[Account]:
        """List accounts with optional filtering."""
        return await self._client.list_accounts(
            pool=pool,
            status=status,
            offset=offset,
            limit=limit,
        )

    async def invalidate(self, account_id: int) -> None:
        """Mark an account as invalid."""
        await self._client.invalidate(account_id)

    async def recover(self, account_id: int) -> None:
        """Recover an invalid account back to idle."""
        await self._client.recover(account_id)

    async def get_pool_stats(self, pool: str = '') -> list[PoolStats]:
        """Get pool statistics. Empty *pool* returns all pools."""
        return await self._client.get_pool_stats(pool)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def aclose(self) -> None:
        """Stop all heartbeats and close the underlying aiohttp session."""
        # 先快照 task，再逐个 cancel。_stop_heartbeat 会从 _heartbeat_tasks 里 pop，
        # 如果先 stop 再取 values() 就会拿到空 dict，gather 白等。
        tasks = list(self._heartbeat_tasks.values())
        for ident in list(self._heartbeat_tasks.keys()):
            self._stop_heartbeat(ident)
        # 等待任务真正结束（cancel 后被 CancelledError 吞掉，很快）
        pending = [t for t in tasks if not t.done()]
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        with suppress(Exception):
            await self._client.aclose()

    async def __aenter__(self) -> AccountPoolManager:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _acquire_any(
        self,
        pool: str,
        ttl: int,
        reservation_token: str,
        *,
        enable_heartbeat: bool = True,
        tags: list[str] | None = None,
    ) -> AcquiredAccount:
        """Acquire any available account from *pool*.

        Exception-safety: 一旦从服务端拿到 lease，后续任何异常（如
        ``_start_heartbeat`` 起协程失败等）都必须把服务端 lease 释放掉并从
        ``_accounts`` 清理，避免孤儿 lease 泄漏。
        """
        lease = await self._client.acquire(
            pool,
            ttl_seconds=ttl,
            reservation_token=reservation_token,
            tags=tags,
        )
        try:
            account = AcquiredAccount(
                lease_id=lease.lease_id,
                account_id=lease.account_id,
                pool=pool,
                account_key=lease.account_key,
                credentials_raw=lease.credentials_raw,
                tokens_raw=lease.tokens_raw,
                ttl_seconds=ttl,
                _manager=self,
                _reservation_token=reservation_token,
            )
            self._accounts[lease.lease_id] = account
            logger.info(
                'Acquired account %s/%s (id=%s, lease=%s)',
                pool,
                lease.account_key,
                lease.account_id,
                lease.lease_id[:12],
            )

            if enable_heartbeat:
                self._start_heartbeat(account, lease.lease_id)

            return account
        except BaseException:
            # 兜底释放：包括 CancelledError/KeyboardInterrupt，避免孤儿 lease。
            self._accounts.pop(lease.lease_id, None)
            with suppress(Exception):
                self._stop_heartbeat(lease.lease_id)
            try:
                await self._client.release(lease.lease_id, reservation_token=reservation_token)
            except Exception:
                logger.exception(
                    'emergency release failed for lease=%s (pool=%s)',
                    lease.lease_id[:12],
                    pool,
                )
            raise

    async def _release_account(self, account: AcquiredAccount) -> None:
        """Release a single account — called from AcquiredAccount.release()."""
        ident = self._find_identifier(account)
        if ident:
            await self.release(ident)

    async def _invalidate_account(self, account: AcquiredAccount, reason: str) -> None:
        """Invalidate a single account — called from AcquiredAccount.invalidate()."""
        ident = self._find_identifier(account)
        if ident:
            self._stop_heartbeat(ident)
        if account._invalidated:
            return
        try:
            await self.invalidate(account.account_id)
        finally:
            account._invalidated = True
            if reason:
                logger.warning(
                    'Account %s/%s (id=%s) invalidated: %s',
                    account.pool,
                    account.account_key,
                    account.account_id,
                    reason,
                )
            if ident:
                self._accounts.pop(ident, None)

    def _pop_by_pool(self, pool: str) -> AcquiredAccount | None:
        """Find and remove an account by pool name. Returns the last acquired match.

        ``_accounts`` 是插入有序的（Py3.7+ dict 保序），倒序遍历取到的
        第一个匹配就是最近一次 acquire 到的账号。
        """
        for key in reversed(list(self._accounts.keys())):
            acct = self._accounts[key]
            if acct.pool == pool and not acct._released and not acct._invalidated:
                del self._accounts[key]
                return acct
        return None

    def _find_identifier(self, account: AcquiredAccount) -> str | None:
        """Find the identifier key for an account in _accounts."""
        for ident, acct in self._accounts.items():
            if acct is account:
                return ident
        return None

    # ------------------------------------------------------------------
    # Heartbeat (asyncio-based)
    # ------------------------------------------------------------------

    def _start_heartbeat(self, account: AcquiredAccount, identifier: str) -> None:
        """Start an asyncio heartbeat task for the given account."""
        if identifier in self._heartbeat_tasks:
            return
        interval = max(account.ttl_seconds / _HEARTBEAT_DIVISOR, _HEARTBEAT_MIN_INTERVAL)
        task = asyncio.create_task(
            self._heartbeat_loop(account, identifier, interval),
            name=f'nexus-hb-{identifier}',
        )
        self._heartbeat_tasks[identifier] = task

    def _stop_heartbeat(self, identifier: str) -> None:
        """Stop the heartbeat task for *identifier*."""
        task = self._heartbeat_tasks.pop(identifier, None)
        if task is not None and not task.done():
            task.cancel()

    async def _heartbeat_loop(self, account: AcquiredAccount, identifier: str, interval: float) -> None:
        """Periodic heartbeat loop — runs as a background asyncio task.

        Retry policy: transient failures (network errors, 5xx, business errors
        without an HTTP code) trigger exponential backoff up to
        :data:`_HEARTBEAT_MAX_RETRIES` attempts. HTTP 4xx is treated as
        permanent (lease has been reclaimed) and gives up immediately.

        Backoff is capped at ``interval / 2`` to avoid sleeping past the TTL,
        with 50% jitter to prevent thundering herd across many leases.
        """
        try:
            await asyncio.sleep(interval)
            while not account._released and not account._invalidated:
                if not await self._heartbeat_once_with_retry(account, identifier, interval):
                    return
                await asyncio.sleep(interval)
        except asyncio.CancelledError:
            pass

    async def _heartbeat_once_with_retry(
        self,
        account: AcquiredAccount,
        identifier: str,
        interval: float,
    ) -> bool:
        """Send one heartbeat with retry on transient errors.

        Returns True if eventually succeeded, False if we should give up the loop.
        """
        for attempt in range(1, _HEARTBEAT_MAX_RETRIES + 1):
            try:
                await self.heartbeat(account.lease_id, ttl_seconds=account.ttl_seconds)
                return True
            except AccountPoolError as exc:
                # 4xx: lease is gone on the server side, no point retrying.
                code = getattr(exc, 'code', None)
                if code is not None and 400 <= code < 500:
                    logger.warning(
                        'Heartbeat gave up (lease gone) for %s (lease=%s, code=%s): %s',
                        identifier,
                        account.lease_id[:12],
                        code,
                        exc,
                    )
                    return False

                # Exhausted retries -> give up.
                if attempt >= _HEARTBEAT_MAX_RETRIES:
                    logger.warning(
                        'Heartbeat gave up after %d attempts for %s (lease=%s): %s',
                        attempt,
                        identifier,
                        account.lease_id[:12],
                        exc,
                    )
                    return False

                # Exponential backoff with jitter, capped at interval/2 so we
                # never sleep past the TTL boundary.
                base = _HEARTBEAT_BASE_BACKOFF * (2 ** (attempt - 1))
                backoff = min(base, interval / 2) * (0.5 + random.random() * 0.5)
                logger.info(
                    'Heartbeat retry %d/%d in %.1fs for %s (lease=%s): %s',
                    attempt,
                    _HEARTBEAT_MAX_RETRIES,
                    backoff,
                    identifier,
                    account.lease_id[:12],
                    exc,
                )
                # Break out promptly if the account has been released/invalidated
                # during our sleep (e.g. explicit release() by caller).
                try:
                    await asyncio.sleep(backoff)
                except asyncio.CancelledError:
                    raise
                if account._released or account._invalidated:
                    return False
        return False
