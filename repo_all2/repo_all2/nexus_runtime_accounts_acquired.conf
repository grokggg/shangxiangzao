"""``AcquiredAccount``：一次 acquire 返回的账号句柄。"""

from __future__ import annotations

import json

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:  # pragma: no cover
    from .manager import AccountPoolManager


@dataclass
class AcquiredAccount:
    """Represents an acquired account within an async execution context.

    Provides access to credentials, tokens, and lifecycle methods
    (invalidate, release).  Heartbeat is managed automatically by
    :class:`AccountPoolManager`.

    Attributes:
        lease_id: Unique lease identifier from the server.
        account_id: Server-side account ID.
        pool: Pool name (e.g. ``'github'``).
        account_key: Account key within the pool.
        credentials_raw: Raw JSON credentials string.
        tokens_raw: Raw JSON tokens string.
        ttl_seconds: Lease TTL in seconds.
    """

    lease_id: str
    account_id: int
    pool: str = ''
    account_key: str = ''
    credentials_raw: str = ''
    tokens_raw: str = ''
    ttl_seconds: int = 60

    # Internal
    _manager: AccountPoolManager | None = field(default=None, repr=False)
    _invalidated: bool = field(default=False, repr=False)
    _released: bool = field(default=False, repr=False)
    _credentials: dict[str, Any] = field(default_factory=dict, repr=False)
    _tokens: dict[str, Any] = field(default_factory=dict, repr=False)
    #: Reservation token this account was acquired under (empty if not from a reservation).
    #: When non-empty, ``release`` will automatically return the account to ``reserved``
    #: (kept inside the batch) instead of ``idle``. Call ``cancel_reservation`` on the
    #: token to fully return everything to ``idle``.
    _reservation_token: str = field(default='', repr=False)

    def __post_init__(self) -> None:
        if not self._credentials and self.credentials_raw:
            try:
                self._credentials = json.loads(self.credentials_raw) or {}
            except (json.JSONDecodeError, TypeError):
                self._credentials = {}
        if not self._tokens and self.tokens_raw:
            try:
                self._tokens = json.loads(self.tokens_raw) or {}
            except (json.JSONDecodeError, TypeError):
                self._tokens = {}

    @property
    def credentials(self) -> dict[str, Any]:
        """Parsed credentials dict (e.g. ``{'username': '...', 'password': '...'}``)."""
        return self._credentials

    @property
    def tokens(self) -> dict[str, Any]:
        """Parsed tokens dict (e.g. ``{'access_token': 'ghp_xxx'}``)."""
        return self._tokens

    @property
    def invalidated(self) -> bool:
        """Whether this account has been invalidated."""
        return self._invalidated

    async def invalidate(self, reason: str = '') -> None:
        """Mark this account as invalid on the server.

        Stops heartbeat and marks the account as ``invalid`` so it won't be
        re-acquired.  The ExecutionContext will skip releasing this account
        on exit.

        Args:
            reason: Human-readable reason for logging.
        """
        if self._invalidated or self._released:
            return
        if self._manager is not None:
            await self._manager._invalidate_account(self, reason)

    async def release(self) -> None:
        """Manually release this account back to idle.

        Stops heartbeat and releases the lease.  After calling this, the
        account should not be used.
        """
        if self._released or self._invalidated:
            return
        if self._manager is not None:
            await self._manager._release_account(self)
