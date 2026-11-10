"""Helpers for opening official MCP ClientSession instances."""

from __future__ import annotations

import asyncio
import builtins
import logging
import sys

from datetime import timedelta
from typing import Any

import aiohttp
import httpx

from mcp import ClientSession

from nexus.utils.otel.mcp import mcp_client_span, merge_trace_meta

from .models import MCPClientConfig, MCPToolInfo, MCPToolResult, MCPTransportKind
from .transport.factory import create_mcp_transport


logger = logging.getLogger(__name__)
BASE_EXCEPTION_GROUP = getattr(builtins, 'BaseExceptionGroup', None)


def _format_exception_line(exc: BaseException) -> str:
    detail = str(exc)
    if detail:
        return f'{type(exc).__name__}: {detail}'
    return type(exc).__name__


def _indent_block(text: str, prefix: str) -> str:
    return '\n'.join(f'{prefix}{line}' if line else prefix.rstrip() for line in text.splitlines())


def _format_exception(exc: BaseException) -> str:
    summary = _format_exception_line(exc)
    if BASE_EXCEPTION_GROUP is None or not isinstance(exc, BASE_EXCEPTION_GROUP):
        return summary

    children = getattr(exc, 'exceptions', ())
    child_lines = [_indent_block(_format_exception(child), f'[{index}] ') for index, child in enumerate(children)]
    if not child_lines:
        return summary
    return f'{summary}\n' + '\n'.join(child_lines)


def _prepare_client_config(config: MCPClientConfig) -> MCPClientConfig:
    if config.transport != MCPTransportKind.STDIO:
        return config

    env = dict(config.env)
    # Keep Nix libraries for the worker Python; clear them for image-installed servers.
    if config.command != sys.executable:
        env.setdefault('LD_LIBRARY_PATH', '')
    return config.model_copy(update={'env': env})


class ConnectedMCPSession:
    """A live official ClientSession with task-safe transport lifecycle."""

    def __init__(
        self,
        config: MCPClientConfig,
        *,
        server_name: str = 'mcp',
        nexus_managed: bool = False,
        aiohttp_session: aiohttp.ClientSession | None = None,
        httpx_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.config = _prepare_client_config(config)
        self.server_name = server_name
        self.nexus_managed = nexus_managed
        self._aiohttp_session = aiohttp_session
        self._httpx_client = httpx_client

        self._transport_task: asyncio.Task | None = None
        self._transport_ready: asyncio.Event | None = None
        self._transport_close: asyncio.Event | None = None
        self._transport_error: BaseException | None = None
        self._transport: tuple[Any, Any, Any] | None = None

        self._session: ClientSession | None = None
        self._session_task: asyncio.Task | None = None
        self._session_ready: asyncio.Event | None = None
        self._session_close: asyncio.Event | None = None
        self._session_error: BaseException | None = None

    @property
    def session(self) -> ClientSession:
        if self._session is None:
            raise RuntimeError('MCP session is not connected')
        return self._session

    async def connect(self) -> None:
        if self._session is not None:
            return

        await self._start_transport()
        if self._transport is None:
            raise RuntimeError('Failed to start MCP transport')

        try:
            await self._start_session(self._transport[:2])
        except BaseException:
            await self.close()
            raise

    async def close(self) -> None:
        await self._stop_session()
        await self._stop_transport()

    async def _start_transport(self) -> None:
        transport_ready = asyncio.Event()
        transport_close = asyncio.Event()
        self._transport_ready = transport_ready
        self._transport_close = transport_close
        self._transport_error = None
        self._transport = None

        transport_cm = create_mcp_transport(
            self.config,
            server_name=self.server_name,
            nexus_managed=self.nexus_managed,
            aiohttp_session=self._aiohttp_session,
            httpx_client=self._httpx_client,
        )

        async def run_transport() -> None:
            try:
                async with transport_cm as transport:
                    self._transport = transport
                    transport_ready.set()
                    await transport_close.wait()
            except BaseException as exc:
                self._transport_error = exc
                if not transport_ready.is_set():
                    transport_ready.set()
                raise
            finally:
                self._transport = None

        self._transport_task = asyncio.create_task(run_transport())
        await transport_ready.wait()
        if self._transport_error:
            transport_error = self._transport_error
            await self._stop_transport()
            raise RuntimeError(
                f'Failed to start MCP transport: {_format_exception(transport_error)}'
            ) from transport_error

    async def _start_session(self, transport: tuple[Any, Any]) -> None:
        session_ready = asyncio.Event()
        session_close = asyncio.Event()
        self._session_ready = session_ready
        self._session_close = session_close
        self._session_error = None

        async def run_session() -> None:
            try:
                async with ClientSession(*transport) as session:
                    await session.initialize()
                    self._session = session
                    session_ready.set()
                    await session_close.wait()
            except BaseException as exc:
                self._session_error = exc
                if not session_ready.is_set():
                    session_ready.set()
                raise
            finally:
                self._session = None

        self._session_task = asyncio.create_task(run_session())
        await session_ready.wait()
        if self._session_error:
            session_error = self._session_error
            await self._stop_session()
            raise RuntimeError(
                f'Failed to initialize MCP session: {_format_exception(session_error)}'
            ) from session_error

    async def _stop_session(self) -> None:
        if self._session_close:
            self._session_close.set()
        if self._session_task:
            try:
                await asyncio.wait_for(asyncio.shield(self._session_task), timeout=5.0)
            except BaseException as exc:
                logger.warning('Error closing MCP session: %s', _format_exception(exc))
        self._session = None
        self._session_task = None
        self._session_ready = None
        self._session_close = None
        self._session_error = None

    async def _stop_transport(self) -> None:
        if self._transport_close:
            self._transport_close.set()
        if self._transport_task:
            try:
                await asyncio.wait_for(asyncio.shield(self._transport_task), timeout=5.0)
            except BaseException as exc:
                logger.warning('Error closing MCP transport: %s', _format_exception(exc))
        self._transport_task = None
        self._transport_ready = None
        self._transport_close = None
        self._transport_error = None
        self._transport = None


async def connect_mcp_session(
    config: MCPClientConfig,
    *,
    server_name: str = 'mcp',
    nexus_managed: bool = False,
    aiohttp_session: aiohttp.ClientSession | None = None,
    httpx_client: httpx.AsyncClient | None = None,
) -> ConnectedMCPSession:
    """Open a configured MCP transport and initialize an official ClientSession."""

    connection = ConnectedMCPSession(
        config,
        server_name=server_name,
        nexus_managed=nexus_managed,
        aiohttp_session=aiohttp_session,
        httpx_client=httpx_client,
    )
    await connection.connect()
    return connection


async def list_mcp_tools_raw(session: ClientSession, *, timeout_s: float | None = None) -> Any:
    return await _await_with_timeout(session.list_tools(), timeout_s)


async def call_mcp_tool_raw(
    session: ClientSession,
    name: str,
    arguments: dict[str, Any] | None = None,
    *,
    timeout_s: float | None = None,
    meta: Any | None = None,
    server_name: str = 'mcp',
) -> Any:
    # read_timeout is set slightly smaller than the outer asyncio timeout so that
    # the MCP SDK's internal anyio cancel scope fires first, giving a cleaner error.
    read_timeout = timedelta(seconds=timeout_s * 0.9) if timeout_s is not None else None
    with mcp_client_span(server_name, name):
        merged_meta = merge_trace_meta(meta)
        if merged_meta is None:
            call = session.call_tool(name, arguments or {}, read_timeout_seconds=read_timeout)
        else:
            call = session.call_tool(name, arguments or {}, read_timeout_seconds=read_timeout, meta=merged_meta)
        return await _await_with_timeout(call, timeout_s)


async def list_mcp_tools(session: ClientSession, *, timeout_s: float | None = None) -> list[MCPToolInfo]:
    result = await list_mcp_tools_raw(session, timeout_s=timeout_s)
    return [
        MCPToolInfo(
            name=tool.name,
            description=getattr(tool, 'description', None),
            input_schema=getattr(tool, 'inputSchema', None) or getattr(tool, 'input_schema', None) or {},
        )
        for tool in result.tools
    ]


async def call_mcp_tool(
    session: ClientSession,
    name: str,
    arguments: dict[str, Any] | None = None,
    *,
    timeout_s: float | None = None,
    source_name: str = 'mcp',
) -> MCPToolResult:
    try:
        result = await call_mcp_tool_raw(
            session,
            name,
            arguments,
            timeout_s=timeout_s,
            server_name=source_name,
        )
    except Exception as exc:
        logger.warning('Error calling MCP tool %s on %s: %s', name, source_name, exc)
        return MCPToolResult(
            content=[{'type': 'text', 'text': f'MCP tool error: {type(exc).__name__}: {exc}'}],
            is_error=True,
        )

    return normalize_mcp_tool_result(result)


def normalize_mcp_tool_result(result: Any) -> MCPToolResult:
    content: list[dict[str, Any]] = []
    for item in result.content:
        if hasattr(item, 'text'):
            content.append({'type': 'text', 'text': item.text})
        elif hasattr(item, 'data'):
            mime_type = getattr(item, 'mimeType', 'image/png')
            content.append({
                'type': 'image_url',
                'image_url': {'url': f'data:{mime_type};base64,{item.data}'},
            })
        else:
            content.append({'type': 'unknown', 'raw': str(item)})

    return MCPToolResult(content=content, is_error=bool(getattr(result, 'isError', False)))


async def _await_with_timeout(coro: Any, timeout_s: float | None) -> Any:
    if timeout_s is None:
        return await coro
    return await asyncio.wait_for(coro, timeout=timeout_s)


__all__ = [
    'ConnectedMCPSession',
    'MCPToolResult',
    'call_mcp_tool',
    'call_mcp_tool_raw',
    'connect_mcp_session',
    'list_mcp_tools',
    'list_mcp_tools_raw',
    'normalize_mcp_tool_result',
]
