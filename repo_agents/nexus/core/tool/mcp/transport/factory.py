"""Transport factory for MCP clients."""

from __future__ import annotations

import contextlib
import os

from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager

import aiohttp
import httpx

from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from mcp.shared.message import SessionMessage

from nexus.utils.otel.auto import is_enabled, is_requested
from nexus.utils.otel.environment import is_managed_otel_env_key, managed_otel_env
from nexus.utils.otel.mcp import mcp_service_name

from .aiohttp_streamable_http import aiohttp_streamable_http_client
from ..models import MCPClientConfig, MCPTransportKind


@asynccontextmanager
async def create_mcp_transport(
    config: MCPClientConfig,
    *,
    server_name: str = 'mcp',
    nexus_managed: bool = False,
    aiohttp_session: aiohttp.ClientSession | None = None,
    httpx_client: httpx.AsyncClient | None = None,
) -> AsyncGenerator[
    tuple[
        MemoryObjectReceiveStream[SessionMessage | Exception],
        MemoryObjectSendStream[SessionMessage],
        Callable[[], str | None],
    ],
    None,
]:
    if config.transport == MCPTransportKind.STDIO:
        from mcp import StdioServerParameters
        from mcp.client.stdio import stdio_client

        if config.command is None:
            raise ValueError('stdio transport requires command')
        if is_requested():
            merged_env = {**os.environ, **config.env}
            merged_env = {key: value for key, value in merged_env.items() if not is_managed_otel_env_key(key)}
            if nexus_managed and is_enabled():
                merged_env.update(
                    managed_otel_env(
                        process_role='mcp',
                        service_name=mcp_service_name(server_name),
                        identifiers={'nexus.mcp.server': server_name},
                    )
                )
        else:
            merged_env = {**os.environ, **config.env} if config.env else None
        server_params = StdioServerParameters(
            command=config.command,
            args=config.args,
            env=merged_env,
            cwd=config.cwd,
        )
        async with contextlib.AsyncExitStack() as stack:
            if nexus_managed and is_enabled():
                from nexus.utils.otel.subprocess import preserve_managed_otel_environment

                with preserve_managed_otel_environment():
                    streams = await stack.enter_async_context(stdio_client(server_params))
            else:
                streams = await stack.enter_async_context(stdio_client(server_params))
            read_stream, write_stream = streams
            yield read_stream, write_stream, lambda: None
        return

    if config.transport == MCPTransportKind.STREAMABLE_HTTP:
        from mcp.client.streamable_http import streamable_http_client

        managed_client = httpx_client
        async with contextlib.AsyncExitStack() as stack:
            if managed_client is None and (config.headers or config.timeout_s is not None):
                timeout = config.timeout_s if config.timeout_s is not None else None
                managed_client = httpx.AsyncClient(headers=config.headers or None, timeout=timeout)
                await stack.enter_async_context(managed_client)
            async with streamable_http_client(config.url or '', http_client=managed_client) as streams:
                yield streams
        return

    if config.transport == MCPTransportKind.SSE:
        from mcp.client.sse import sse_client

        timeout = config.timeout_s if config.timeout_s is not None else 5
        async with sse_client(
            config.url or '',
            headers=config.headers or None,
            timeout=timeout,
            sse_read_timeout=config.timeout_s or 300,
        ) as streams:
            yield streams
        return

    async with aiohttp_streamable_http_client(
        config.url or '',
        session=aiohttp_session,
        headers=config.headers,
        timeout_s=config.timeout_s,
    ) as streams:
        yield streams


__all__ = ['create_mcp_transport']
