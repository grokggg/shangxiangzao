"""Async account pool integration for nexus runtime.

管理后台（建池 / 加号 / 打 tag）：https://nexus-pool.woa.com/

建新池：凭证形态、环境、配额、TTL、启停要独立（如 ``github`` vs ``gui-sandbox``）。
打 tag：同一池内可组合能力（如 ``chrome``、``vscode``），一台号可被多种请求领走。
不要为每种 tag 组合新建池。

Exposes the async manager, client, and data models via a thin re-export layer::

    from nexus.runtime.accounts import AccountPoolManager

    mgr = AccountPoolManager()
    account = await mgr.acquire('github')
    gui = await mgr.acquire('gui-sandbox', tags=['chrome', 'vscode'])
    ...
    await mgr.release_all()
    await mgr.aclose()

Integrates with :class:`~nexus.core.context.ExecutionContext` via
``ctx.get_account()``.
"""

from __future__ import annotations

from .acquired import AcquiredAccount
from .client import AsyncAccountPoolClient
from .exceptions import (
    AccountPoolError,
    AcquireError,
    HeartbeatError,
    InvalidateError,
    NoAvailableAccountError,
    ReleaseError,
    ReservationError,
)
from .manager import AccountPoolManager
from .models import Account, Lease, PoolStats, ReserveResult
from .reservation import AsyncMultiReservation, AsyncReservation


__all__ = [
    # Manager & data holders
    'AccountPoolManager',
    'AcquiredAccount',
    'AsyncAccountPoolClient',
    'AsyncMultiReservation',
    'AsyncReservation',
    # SDK types
    'Account',
    'Lease',
    'PoolStats',
    'ReserveResult',
    # Exceptions
    'AccountPoolError',
    'AcquireError',
    'HeartbeatError',
    'InvalidateError',
    'NoAvailableAccountError',
    'ReleaseError',
    'ReservationError',
]
