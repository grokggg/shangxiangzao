"""Nexus 账号池服务的异步 Python 客户端。

一个薄层封装：每个方法直接对应服务端一个 rpc，做好参数校验、类型转换和异常抛出，
不引入额外的业务逻辑（heartbeat/lifecycle 管理放在 :class:`AccountPoolManager`）。

底层直接复用 :mod:`nexus.utils.async_http` 的全局 ``http_client``：
    - session 由 nexus 统一管理（按 event loop 缓存 + atexit 清理），
      **本客户端不负责关闭它**；
    - 全局 client 默认无超时（``total=None``），因此本类每次请求都会**显式**
      传入 :class:`aiohttp.ClientTimeout`，保持与旧行为一致。
"""

from __future__ import annotations

import json
import logging
import os

from typing import Any
from urllib.parse import urlencode

import aiohttp

from nexus.utils.async_http import http_client

from .exceptions import (
    AccountPoolError,
    AcquireError,
    HeartbeatError,
    InvalidateError,
    NoAvailableAccountError,
    ReleaseError,
    ReservationError,
)
from .models import Account, Lease, PoolStats, ReserveResult


logger = logging.getLogger(__name__)


#: 单次 HTTP 请求默认超时（秒）。
DEFAULT_TIMEOUT = 15.0

#: 默认服务端地址（Polaris 生产域名）。可通过环境变量 ``NEXUS_ACCOUNT_POOL_ENDPOINT`` 覆盖。
DEFAULT_ENDPOINT = os.environ.get(
    'NEXUS_ACCOUNT_POOL_ENDPOINT',
    'http://nexus-pool.production.polaris:8080',
)


# 服务器采用 trpc-go restful 协议，路由定义见 proto 中的 ``trpc.api.http`` 注解。
# RESTful 路由表：method → (HTTP方法, 路径模板)
_ROUTES: dict[str, tuple[str, str]] = {
    'Acquire': ('POST', '/v1/accounts/acquire'),
    'Release': ('POST', '/v1/accounts/release'),
    'Heartbeat': ('POST', '/v1/accounts/heartbeat'),
    'Reserve': ('POST', '/v1/reservations'),
    'CancelReservation': ('DELETE', '/v1/reservations/{reservation_token}'),
    'AddAccount': ('POST', '/v1/accounts'),
    'DeleteAccount': ('DELETE', '/v1/accounts/{account_id}'),
    'ListAccounts': ('GET', '/v1/accounts'),
    'InvalidateAccount': ('POST', '/v1/accounts/{account_id}/invalidate'),
    'RecoverAccount': ('POST', '/v1/accounts/{account_id}/recover'),
    'GetPoolStats': ('GET', '/v1/pool/stats'),
}

# 需要从 body 提取到路径参数中的字段
_PATH_PARAM_KEYS = frozenset({'account_id', 'reservation_token'})


def _to_json_str(value: Any) -> str:
    """凭据/token 允许传 dict 或 JSON 字符串，统一转成 JSON 字符串。"""
    if value is None:
        return '{}'
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def _build_reserve_result(data: dict[str, Any]) -> ReserveResult:
    """把服务端 ReserveRsp 解析成 ReserveResult。"""
    grouped: dict[str, list[int]] = {}
    for item in data.get('account_ids_by_pool') or []:
        pool = item.get('pool', '') or ''
        ids = [int(x) for x in (item.get('account_ids') or [])]
        if pool:
            grouped[pool] = ids
    flat_ids = [int(x) for x in (data.get('account_ids') or [])]
    return ReserveResult(
        reservation_token=data.get('reservation_token', '') or '',
        account_ids=flat_ids,
        account_ids_by_pool=grouped,
    )


class AsyncAccountPoolClient:
    """账号池服务的异步客户端（基于 aiohttp）。

    Example::

        client = AsyncAccountPoolClient()  # 默认连 http://nexus-pool.production.polaris:8080
        lease = await client.acquire('github', ttl_seconds=60)
        # ...
        await client.release(lease.lease_id)
        # 无需 aclose：底层 http_client 由 nexus 全局管理。
    """

    def __init__(
        self,
        endpoint: str | None = None,
        *,
        timeout: float = DEFAULT_TIMEOUT,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        """
        Args:
            endpoint: 服务地址。为空则使用 ``DEFAULT_ENDPOINT``
                （可通过环境变量 ``NEXUS_ACCOUNT_POOL_ENDPOINT`` 覆盖）。
            timeout: 单次 HTTP 请求超时（秒）。因全局 ``http_client`` 默认不超时，
                本类每次请求都会显式传入 :class:`aiohttp.ClientTimeout(total=timeout)`。
            extra_headers: 附加请求头（如 X-Request-Id），每次请求都会带上。
        """
        self._endpoint = (endpoint or DEFAULT_ENDPOINT).rstrip('/')
        self._timeout = aiohttp.ClientTimeout(total=timeout)
        self._headers: dict[str, str] = {'Content-Type': 'application/json'}
        if extra_headers:
            self._headers.update(extra_headers)

    # ------------------------------------------------------------------ #
    #                              底层                                  #
    # ------------------------------------------------------------------ #

    async def _call(self, method: str, payload: dict[str, Any]) -> dict[str, Any]:
        """调用一个 rpc 方法。

        Raises:
            AccountPoolError: 网络/HTTP 层错误。业务错误由上层根据
                ``success`` / ``error`` 字段判断。
        """
        route = _ROUTES.get(method)
        if route is None:
            raise AccountPoolError(f'unknown method: {method}')

        http_method, path_template = route

        # 构建路径：从 payload 中提取路径参数
        path = path_template
        body = dict(payload)
        for key in _PATH_PARAM_KEYS:
            placeholder = '{' + key + '}'
            if placeholder in path and key in body:
                val = body.pop(key)
                path = path.replace(placeholder, str(val))

        url = f'{self._endpoint}{path}'

        # 按 method 组装参数：
        # - GET：剩余 body 里的字段拼成 query string；
        # - POST/DELETE：始终把剩余 body 作为 JSON body 发出（即便是 {}），
        #   因为服务端 trpc-go transcoder 要求请求体存在，否则会报
        #   `transcoding body {} caused by failed to unmarshal req body`。
        kwargs: dict[str, Any] = {'headers': self._headers, 'timeout': self._timeout}
        if http_method == 'GET':
            if body:
                qs = urlencode({k: v for k, v in body.items() if v is not None and v != ''})
                if qs:
                    url += '?' + qs
        else:
            kwargs['json'] = body

        try:
            resp = await http_client.request(http_method, url, **kwargs)
        except TimeoutError as exc:
            # aiohttp 超时抛的是 asyncio.TimeoutError（Py3.11 起是 builtins.TimeoutError 的别名），
            # 并不继承 aiohttp.ClientError。必须单独包装，否则会穿透到心跳循环，
            # 让 manager._heartbeat_once_with_retry 无法按预期重试。
            raise AccountPoolError(f'http request timeout: {exc}') from exc
        except aiohttp.ClientError as exc:
            raise AccountPoolError(f'http request failed: {exc}') from exc

        return self._parse(resp)

    @staticmethod
    def _parse(resp: Any) -> dict[str, Any]:
        """把 ``HTTPResponse`` 解析成 dict，并按状态码抛出 :class:`AccountPoolError`。"""
        text = resp.text
        if resp.status >= 500:
            raise AccountPoolError(
                f'server error: HTTP {resp.status} {text[:200]}',
                code=resp.status,
            )
        if resp.status >= 400:
            raise AccountPoolError(
                f'client error: HTTP {resp.status} {text[:200]}',
                code=resp.status,
            )
        try:
            data = resp.json()
        except ValueError as exc:
            raise AccountPoolError(f'invalid json response: {text[:200]}') from exc
        if not isinstance(data, dict):
            raise AccountPoolError(f'unexpected response type: {type(data).__name__}')
        return data

    # ------------------------------------------------------------------ #
    #                              流转                                  #
    # ------------------------------------------------------------------ #

    async def acquire(
        self,
        pool: str,
        *,
        ttl_seconds: int = 60,
        reservation_token: str = '',
        tags: list[str] | None = None,
    ) -> Lease:
        """从指定 pool 领一个账号。

        管理后台（建池 / 加号 / 打 tag）：https://nexus-pool.woa.com/
        建新池还是打 tag 见包文档 :mod:`nexus.runtime.accounts`。

        Args:
            pool: 池名，与账号池管理页上的池名称一致，例如 ``'github'``。
            ttl_seconds: 租约时长（秒）。
            reservation_token: 非空则从已预留批次里领；空字符串表示从 idle 池领。
            tags: 可选关键字参数，类型 ``list[str] | None``。
                传参示例::

                    await client.acquire('gui-sandbox')
                    await client.acquire('gui-sandbox', tags=['chrome', 'vscode'])

                匹配规则是 AND：账号必须同时具备列表里每一个非空 tag。
                不传、传 ``None`` 或 ``[]`` 时请求体不含 ``tags`` 字段，服务端不按 tag 过滤。
                账号上多出来的 tag 不妨碍匹配；列表里的空字符串会被丢掉。

        Raises:
            NoAvailableAccountError: 池里没有可用账号。
            AcquireError: 其他 acquire 失败（含池不存在 / 已停用）。
        """
        payload: dict[str, Any] = {'pool': pool, 'ttl_seconds': int(ttl_seconds)}
        if reservation_token:
            payload['reservation_token'] = reservation_token
        if tags:
            payload['tags'] = [t for t in tags if t]

        data = await self._call('Acquire', payload)
        if not data.get('success'):
            err = data.get('error', '') or 'acquire failed'
            if 'no available account' in err.lower():
                raise NoAvailableAccountError(err)
            raise AcquireError(err)

        return Lease(
            lease_id=data.get('lease_id', '') or '',
            account_id=int(data.get('account_id', 0) or 0),
            account_key=data.get('account_key', '') or '',
            credentials_raw=data.get('credentials', '') or '',
            tokens_raw=data.get('tokens', '') or '',
            ttl_seconds=int(ttl_seconds),
        )

    async def release(self, lease_id: str, *, reservation_token: str = '') -> None:
        """归还账号。

        Args:
            lease_id: acquire 返回的 lease_id。
            reservation_token: 非空则账号回 ``reserved``（保留预留、稍后可再 acquire）；
                为空则回 ``idle``。
        """
        if not lease_id:
            raise ReleaseError('lease_id is required')
        payload: dict[str, Any] = {'lease_id': lease_id}
        if reservation_token:
            payload['reservation_token'] = reservation_token
        data = await self._call('Release', payload)
        if not data.get('success'):
            raise ReleaseError(data.get('error', 'release failed'))

    async def heartbeat(self, lease_id: str, *, ttl_seconds: int = 60) -> str:
        """续租，返回服务端新的过期时间字符串。"""
        if not lease_id:
            raise HeartbeatError('lease_id is required')
        payload = {'lease_id': lease_id, 'ttl_seconds': int(ttl_seconds)}
        data = await self._call('Heartbeat', payload)
        if not data.get('success'):
            raise HeartbeatError(data.get('error', 'heartbeat failed'))
        return data.get('new_expire_at', '') or ''

    async def reserve(
        self,
        pool: str,
        count: int,
        *,
        ttl_seconds: int = 300,
        tags: list[str] | None = None,
    ) -> ReserveResult:
        """批量预留 count 个账号（单 pool），全有或全无。

        Args:
            pool: 池名。
            count: 预留数量，必须 > 0。
            ttl_seconds: 预留有效期（秒）。
            tags: 与 :meth:`acquire` 相同。传参示例::

                    await client.reserve('gui-sandbox', 3, tags=['chrome'])
        """
        if count <= 0:
            raise ReservationError('count must be > 0')
        payload: dict[str, Any] = {'pool': pool, 'count': int(count), 'ttl_seconds': int(ttl_seconds)}
        if tags:
            payload['tags'] = [t for t in tags if t]
        data = await self._call('Reserve', payload)
        if not data.get('success'):
            raise ReservationError(data.get('error', 'reserve failed'))
        return _build_reserve_result(data)

    async def reserve_multi(
        self,
        items: list[dict[str, Any]],
        *,
        ttl_seconds: int = 300,
    ) -> ReserveResult:
        """多 pool 原子预留：任一 pool 数量不足即整体失败。

        ``items`` 每项是 dict，字段：

        - ``pool`` (str, 必填)
        - ``count`` (int, 必填, > 0)
        - ``tags`` (``list[str]``, 可选)：与 :meth:`acquire` 相同，只筛该 pool。
          不传则该 pool 不按 tag 过滤。

        传参示例::

            await client.reserve_multi([
                {'pool': 'github', 'count': 2},
                {'pool': 'gui-sandbox', 'count': 1, 'tags': ['chrome', 'vscode']},
            ])
        """
        if not items:
            raise ReservationError('items must not be empty')
        norm: list[dict[str, Any]] = []
        for it in items:
            pool = it.get('pool', '')
            count = int(it.get('count', 0) or 0)
            if not pool:
                raise ReservationError('pool must not be empty')
            if count <= 0:
                raise ReservationError(f'count for pool {pool!r} must be > 0')
            item: dict[str, Any] = {'pool': pool, 'count': count}
            item_tags = it.get('tags')
            if item_tags:
                item['tags'] = [t for t in item_tags if t]
            norm.append(item)
        payload = {'items': norm, 'ttl_seconds': int(ttl_seconds)}
        data = await self._call('Reserve', payload)
        if not data.get('success'):
            raise ReservationError(data.get('error', 'reserve multi failed'))
        return _build_reserve_result(data)

    async def cancel_reservation(self, reservation_token: str) -> None:
        """取消整批预留（含 acquired/reserved 全部回 idle）。"""
        if not reservation_token:
            raise ReservationError('reservation_token is required')
        data = await self._call('CancelReservation', {'reservation_token': reservation_token})
        if not data.get('success'):
            raise ReservationError(data.get('error', 'cancel reservation failed'))

    # ------------------------------------------------------------------ #
    #                              管理                                  #
    # ------------------------------------------------------------------ #

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
        """新增账号，返回新账号 id。

        ``credentials`` / ``tokens`` 支持 dict 或 JSON 字符串。``tags`` 中的空字符串
        会被忽略；不传、传 ``None``、``[]`` 或仅包含空字符串时不发送 ``tags`` 字段。
        """
        payload = {
            'pool': pool,
            'account_key': account_key,
            'credentials': _to_json_str(credentials),
            'tokens': _to_json_str(tokens),
            'shared_cap': int(shared_cap),
            'note': note or '',
        }
        normalized_tags = [tag for tag in tags or [] if tag]
        if normalized_tags:
            payload['tags'] = normalized_tags
        data = await self._call('AddAccount', payload)
        if not data.get('success'):
            raise AccountPoolError(data.get('error', 'add account failed'))
        return int(data.get('account_id', 0) or 0)

    async def delete_account(self, account_id: int) -> None:
        data = await self._call('DeleteAccount', {'account_id': int(account_id)})
        if not data.get('success'):
            raise AccountPoolError(data.get('error', 'delete account failed'))

    async def list_accounts(
        self,
        *,
        pool: str = '',
        status: str = '',
        offset: int = 0,
        limit: int = 20,
    ) -> list[Account]:
        payload = {
            'pool': pool,
            'status': status,
            'offset': int(offset),
            'limit': int(limit),
        }
        data = await self._call('ListAccounts', payload)
        if not data.get('success'):
            raise AccountPoolError(data.get('error', 'list accounts failed'))
        return [Account.from_dict(d) for d in data.get('accounts') or []]

    async def invalidate(self, account_id: int) -> None:
        """把账号置为 invalid。"""
        data = await self._call('InvalidateAccount', {'account_id': int(account_id)})
        if not data.get('success'):
            raise InvalidateError(data.get('error', 'invalidate failed'))

    async def recover(self, account_id: int) -> None:
        """把 invalid 的账号恢复为 idle。"""
        data = await self._call('RecoverAccount', {'account_id': int(account_id)})
        if not data.get('success'):
            raise InvalidateError(data.get('error', 'recover failed'))

    async def get_pool_stats(self, pool: str = '') -> list[PoolStats]:
        """池状态统计。pool 为空则返回全部。"""
        data = await self._call('GetPoolStats', {'pool': pool})
        if not data.get('success'):
            raise AccountPoolError(data.get('error', 'get stats failed'))
        return [PoolStats.from_dict(d) for d in data.get('stats') or []]

    # ------------------------------------------------------------------ #
    #                            生命周期                                #
    # ------------------------------------------------------------------ #

    async def aclose(self) -> None:
        """No-op：底层 ``http_client`` 是全局共享的，由 :mod:`nexus.utils.async_http`
        统一管理生命周期，本类不负责关闭。保留此方法以保持向后兼容。"""
        return

    async def __aenter__(self) -> AsyncAccountPoolClient:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()
