"""数据模型。

服务器返回值均为 dict，为方便使用统一封装成 dataclass。
"""

from __future__ import annotations

import json

from dataclasses import dataclass, field
from typing import Any


def _parse_json_field(raw: str | None) -> dict[str, Any]:
    """服务端 credentials/tokens 是 JSON 字符串，帮用户解析成 dict。"""
    if not raw:
        return {}
    try:
        v = json.loads(raw)
        return v if isinstance(v, dict) else {}
    except (ValueError, TypeError):
        return {}


@dataclass
class Lease:
    """一次 acquire 返回的租约。"""

    lease_id: str
    account_id: int
    account_key: str = ''
    credentials_raw: str = ''
    tokens_raw: str = ''
    #: 请求时使用的 ttl（用于心跳线程续租）
    ttl_seconds: int = 60

    @property
    def credentials(self) -> dict[str, Any]:
        return _parse_json_field(self.credentials_raw)

    @property
    def tokens(self) -> dict[str, Any]:
        return _parse_json_field(self.tokens_raw)


@dataclass
class Account:
    """账号列表 / 详情。"""

    id: int
    pool: str
    account_key: str
    credentials_raw: str = ''
    tokens_raw: str = ''
    status: str = 'idle'
    shared_cap: int = 0
    shared_occupied: int = 0
    lease_id: str = ''
    reservation_id: str = ''
    expire_at: str = ''
    last_used_at: str = ''
    created_at: str = ''
    updated_at: str = ''
    note: str = ''

    @property
    def credentials(self) -> dict[str, Any]:
        return _parse_json_field(self.credentials_raw)

    @property
    def tokens(self) -> dict[str, Any]:
        return _parse_json_field(self.tokens_raw)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Account:
        return cls(
            id=int(d.get('id', 0) or 0),
            pool=d.get('pool', '') or '',
            account_key=d.get('account_key', '') or '',
            credentials_raw=d.get('credentials', '') or '',
            tokens_raw=d.get('tokens', '') or '',
            status=d.get('status', 'idle') or 'idle',
            shared_cap=int(d.get('shared_cap', 0) or 0),
            shared_occupied=int(d.get('shared_occupied', 0) or 0),
            lease_id=d.get('lease_id', '') or '',
            reservation_id=d.get('reservation_id', '') or '',
            expire_at=d.get('expire_at', '') or '',
            last_used_at=d.get('last_used_at', '') or '',
            created_at=d.get('created_at', '') or '',
            updated_at=d.get('updated_at', '') or '',
            note=d.get('note', '') or '',
        )


@dataclass
class PoolStats:
    """/v1/pool/stats 返回的单条统计。"""

    pool: str
    exclusive_total: int = 0
    exclusive_idle: int = 0
    exclusive_acquired: int = 0
    exclusive_reserved: int = 0
    exclusive_invalid: int = 0
    shared_total: int = 0
    shared_capacity: int = 0
    shared_occupied: int = 0
    shared_invalid: int = 0

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PoolStats:
        def _i(k: str) -> int:
            return int(d.get(k, 0) or 0)

        return cls(
            pool=d.get('pool', '') or '',
            exclusive_total=_i('exclusive_total'),
            exclusive_idle=_i('exclusive_idle'),
            exclusive_acquired=_i('exclusive_acquired'),
            exclusive_reserved=_i('exclusive_reserved'),
            exclusive_invalid=_i('exclusive_invalid'),
            shared_total=_i('shared_total'),
            shared_capacity=_i('shared_capacity'),
            shared_occupied=_i('shared_occupied'),
            shared_invalid=_i('shared_invalid'),
        )


@dataclass
class ReserveResult:
    """批量预留返回。

    - ``account_ids``：所有 pool 的账号 id 打平（兼容单 pool 用法）。
    - ``account_ids_by_pool``：按 pool 分组，多 pool 预留时使用。单 pool 时也会填。
    """

    reservation_token: str
    account_ids: list[int] = field(default_factory=list)
    account_ids_by_pool: dict[str, list[int]] = field(default_factory=dict)
