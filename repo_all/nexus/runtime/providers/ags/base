"""AGS base runtime classes and helpers."""

from __future__ import annotations

import base64
import contextlib
import json
import logging

from typing import Any, Literal

import aiohttp

from nexus.runtime.client.base import Runtime
from nexus.runtime.providers.ags.handle import AGSInstanceHandle
from nexus.utils.async_http import HTTPResponse, http_client


logger = logging.getLogger(__name__)

_JSON_RESULT_START = 'NEXUS_JSON_RESULT_START'
_JSON_RESULT_END = 'NEXUS_JSON_RESULT_END'


def _collect_text_fragments(value: Any) -> list[str]:
    """Collect string fragments from nested execute responses."""

    if value is None:
        return []
    if isinstance(value, str):
        fragments = [value]
        for line in value.splitlines():
            stripped = line.strip()
            if not stripped.startswith('{') or not stripped.endswith('}'):
                continue
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            fragments.extend(_collect_text_fragments(parsed))
        return fragments
    if isinstance(value, dict):
        fragments: list[str] = []
        for item in value.values():
            fragments.extend(_collect_text_fragments(item))
        return fragments
    if isinstance(value, (list, tuple)):
        fragments = []
        for item in value:
            fragments.extend(_collect_text_fragments(item))
        return fragments
    return []


def _extract_tagged_json(value: Any) -> dict[str, Any] | None:
    """Extract a tagged JSON payload from a nested response."""

    for fragment in _collect_text_fragments(value):
        start = fragment.find(_JSON_RESULT_START)
        end = fragment.find(_JSON_RESULT_END, start + len(_JSON_RESULT_START))
        if start < 0 or end < 0:
            continue
        payload = fragment[start + len(_JSON_RESULT_START) : end].strip()
        if not payload:
            continue
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            logger.debug('Failed to decode tagged JSON payload: %s', payload[:500])
    return None


class _BaseAGSRuntime(Runtime):
    """Base class for AGS runtimes."""

    def __init__(
        self,
        handle: AGSInstanceHandle,
        *,
        request_timeout: float = 1800.0,
        resource_spec: dict[str, int] | None = None,
    ) -> None:
        super().__init__()
        self.handle = handle
        self.request_timeout = request_timeout
        self.resource_spec = resource_spec or {}
        self._timeout = aiohttp.ClientTimeout(total=request_timeout)

    async def close(self) -> None:
        """Close local resources held by this runtime."""

    async def _http_request(self, method: str, url: str, **kwargs: Any) -> HTTPResponse:
        headers = dict(self.handle.headers)
        extra_headers = kwargs.pop('headers', None)
        if extra_headers:
            headers.update({str(key): str(value) for key, value in extra_headers.items()})
        timeout = kwargs.pop('timeout', self._timeout)
        return await http_client.request(method, url, timeout=timeout, headers=headers, **kwargs)

    async def _http_get(self, url: str) -> tuple[Any, str, int]:
        response = await self._http_request('GET', url)
        try:
            body = response.json()
        except Exception:
            body = None
        return body, response.text, response.status_code

    async def _http_post_json(self, url: str, payload: dict[str, Any]) -> tuple[Any, str, int]:
        response = await self._http_request('POST', url, json=payload)
        try:
            body = response.json()
        except Exception:
            body = None
        return body, response.text, response.status_code

    async def _invoke(self, request: Any) -> Any:
        raise NotImplementedError(f'{self.__class__.__name__} does not support invoke')

    async def call_tool(self, import_path: str, params: dict[str, Any], context: Any) -> Any:
        raise NotImplementedError(f'{self.__class__.__name__} does not support call_tool')


class AGSRawRuntime(_BaseAGSRuntime):
    """Thin transport adapter that exposes raw AGS/native service endpoints."""

    def __init__(
        self,
        handle: AGSInstanceHandle,
        *,
        kind: Literal['browser', 'code', 'osworld', 'custom', 'waa'],
        default_port: int | None = None,
        request_timeout: float = 1800.0,
        resource_spec: dict[str, int] | None = None,
    ) -> None:
        super().__init__(handle, request_timeout=request_timeout, resource_spec=resource_spec)
        self.kind = kind
        self.default_port = default_port
        if default_port is not None:
            self.port_mappings[default_port] = self.handle.url(default_port)

    @property
    def base_url(self) -> str | None:
        if self.default_port is None:
            return None
        return self.handle.url(self.default_port)

    def url(self, path: str = '', *, port: int | None = None) -> str:
        resolved_port = self.default_port if port is None else int(port)
        if resolved_port is None:
            raise ValueError(f'{self.__class__.__name__} requires an explicit port for this request')
        return self.handle.url(resolved_port, path)

    async def request(self, method: str, path: str = '', *, port: int | None = None, **kwargs: Any) -> HTTPResponse:
        return await self._http_request(str(method).upper(), self.url(path, port=port), **kwargs)

    async def request_bytes(self, method: str, path: str = '', *, port: int | None = None, **kwargs: Any) -> bytes:
        response = await self.request(method, path, port=port, **kwargs)
        return response.content

    async def request_json(self, method: str, path: str = '', *, port: int | None = None, **kwargs: Any) -> Any:
        response = await self.request(method, path, port=port, **kwargs)
        return response.json()

    def get_metadata(self) -> dict[str, Any]:
        return {
            'kind': self.kind,
            'default_port': self.default_port,
            'base_url': self.base_url,
            'port_mappings': dict(self.port_mappings),
            'handle': {
                'instance_id': self.handle.instance_id,
                'domain': self.handle.domain,
                'tool_id': self.handle.tool_id,
                'tool_name': self.handle.tool_name,
                'tool_type': self.handle.tool_type,
                'metadata': self.handle.metadata,
            },
        }

    @staticmethod
    def _serialize_response(
        response: HTTPResponse,
        *,
        include_text: bool = True,
        include_content: bool = False,
        include_json: bool = True,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            'status_code': response.status_code,
            'ok': response.ok,
            'url': response.url,
            'headers': dict(response.headers),
        }
        if include_text:
            payload['text'] = response.text
        if include_content:
            payload['content_base64'] = base64.b64encode(response.content).decode('ascii')
        if include_json:
            with contextlib.suppress(Exception):
                payload['json'] = response.json()
        return payload

    async def call_executor(self, executor_name: str, arg: Any) -> Any:
        if executor_name == 'raw_get_metadata':
            return self.get_metadata()

        if executor_name == 'raw_request':
            if not isinstance(arg, dict):
                raise TypeError(f'raw_request expects a dict payload, got {type(arg).__name__}')

            response = await self.request(
                str(arg.get('method') or 'GET'),
                str(arg.get('path') or ''),
                port=int(arg['port']) if arg.get('port') is not None else None,
                params=arg.get('params'),
                json=arg.get('json'),
                data=arg.get('data'),
                headers=arg.get('headers'),
            )
            return self._serialize_response(
                response,
                include_text=bool(arg.get('include_text', True)),
                include_content=bool(arg.get('include_content', False)),
                include_json=bool(arg.get('include_json', True)),
            )

        raise NotImplementedError(f'Unsupported AGS raw executor: {executor_name!r}')
