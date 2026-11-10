"""MCP executor with streamable HTTP proxy support."""

# /// script
# dependencies = []
#
# [project]
# name = "mcp"
# authors = [
#     { name = "tarschen" },
# ]
#
# [tool.nexus]
# executors = { mcp_start_proxy = "nexus.runtime.executors.mcp", mcp_stop_proxy = "nexus.runtime.executors.mcp" }
# ///

import asyncio
import contextlib
import logging
import uuid

from collections.abc import Awaitable, Callable, MutableMapping
from typing import Any

from mcp import ClientSession

from nexus.core import executor_registry
from nexus.core.tool.mcp import (
    ConnectedMCPSession,
    MCPClientConfig,
    MCPStartProxyRequest,
    MCPStartProxyResponse,
    MCPStopProxyRequest,
    MCPStopProxyResponse,
    MCPToolInfo,
    call_mcp_tool_raw,
    connect_mcp_session,
    list_mcp_tools,
    list_mcp_tools_raw,
)
from nexus.utils.otel.auto import is_enabled as is_otel_enabled
from nexus.utils.otel.mcp import custom_mcp_meta, mcp_server_span


logger = logging.getLogger(__name__)

MCP_PROXY_SESSIONS: dict[str, 'MCPProxySession'] = {}


def _is_in_worker() -> bool:
    """Return True if running inside a Worker invoke context.

    Checks the RuntimeControl contextvar (set by ``execute_invoked_func``).
    In a Worker the SIO bridge wraps an ``AsyncClient`` (``_is_server=False``),
    while in the Base it wraps an ``AsyncServer`` (``_is_server=True``).
    When RuntimeControl is not set (e.g. direct ``call_executor`` in Base),
    returns False.
    """
    from nexus.runtime.context import get_runtime_control

    ctrl = get_runtime_control()
    if ctrl is None or ctrl.sio is None:
        return False
    return not ctrl.sio._is_server


@executor_registry.register('mcp_start_proxy')
async def mcp_start_proxy(request: MCPStartProxyRequest) -> MCPStartProxyResponse:
    """Start an MCP backend client and expose it as a streamable HTTP proxy."""
    proxy_id = uuid.uuid4().hex[:12]

    if _is_in_worker():
        return await _start_proxy_worker_side(proxy_id, request)

    # Base path — unchanged
    proxy = MCPProxySession(
        proxy_id,
        request.client,
        server_name=request.server_name,
        nexus_managed=request.nexus_managed,
    )
    tools = await proxy.start()
    MCP_PROXY_SESSIONS[proxy_id] = proxy
    return MCPStartProxyResponse(proxy_id=proxy_id, endpoint_path=f'/mcp/proxy/{proxy_id}', tools=tools)


async def _start_proxy_worker_side(proxy_id: str, request: MCPStartProxyRequest) -> MCPStartProxyResponse:
    """Worker-side: create the proxy locally and register a reverse-proxy route on the Base."""
    from nexus.runtime.context import require_runtime_control
    from nexus.runtime.server.worker.http import get_or_create as get_worker_http

    # 1. Create MCPProxySession — identical to Base path
    proxy = MCPProxySession(
        proxy_id,
        request.client,
        server_name=request.server_name,
        nexus_managed=request.nexus_managed,
    )
    tools = await proxy.start()
    MCP_PROXY_SESSIONS[proxy_id] = proxy

    # 2. Ensure Worker HTTP server is running and mcp_proxy_asgi is mounted
    http_server = await get_worker_http()
    http_server.mount('/mcp/proxy', mcp_proxy_asgi)

    # 3. Register route with Base for reverse-proxy forwarding
    rt = require_runtime_control()
    result = await rt.sio.call(
        'worker:register_http_route',
        {
            'path_prefix': f'/mcp/proxy/{proxy_id}',
            'worker_url': http_server.base_url,
        },
    )
    if not result.get('success'):
        MCP_PROXY_SESSIONS.pop(proxy_id, None)
        await proxy.stop()
        raise RuntimeError(f'Base route registration failed: {result.get("error")}')

    logger.info('MCP proxy %s started in Worker at %s/mcp/proxy/%s', proxy_id, http_server.base_url, proxy_id)
    return MCPStartProxyResponse(proxy_id=proxy_id, endpoint_path=f'/mcp/proxy/{proxy_id}', tools=tools)


@executor_registry.register('mcp_stop_proxy')
async def mcp_stop_proxy(request: MCPStopProxyRequest) -> MCPStopProxyResponse:
    """Stop a running MCP proxy and its backend client."""
    proxy = MCP_PROXY_SESSIONS.pop(request.proxy_id, None)
    if proxy:
        await proxy.stop()

    # In Worker: unregister the reverse-proxy route on Base
    if _is_in_worker():
        with contextlib.suppress(Exception):
            from nexus.runtime.context import require_runtime_control

            rt = require_runtime_control()
            await rt.sio.call(
                'worker:unregister_http_route',
                {
                    'path_prefix': f'/mcp/proxy/{request.proxy_id}',
                },
            )

    return MCPStopProxyResponse(status='stopped')


class MCPProxySession:
    """Manage one backend MCP client and its HTTP proxy."""

    def __init__(
        self,
        proxy_id: str,
        client_config: MCPClientConfig,
        *,
        server_name: str = 'mcp',
        nexus_managed: bool = False,
    ):
        self.proxy_id = proxy_id
        self.client_config = client_config
        self.server_name = server_name
        self.nexus_managed = nexus_managed
        self.backend: ConnectedMCPSession | None = None
        self.session_manager: Any = None
        self._shutdown_event = asyncio.Event()
        self._manager_task: asyncio.Task | None = None

    @property
    def backend_session(self) -> ClientSession:
        if self.backend is None:
            raise RuntimeError(f'MCP proxy backend is not connected: {self.proxy_id}')
        return self.backend.session

    async def start(self) -> list[MCPToolInfo]:
        self.backend = await connect_mcp_session(
            self.client_config,
            server_name=self.server_name,
            nexus_managed=self.nexus_managed,
        )
        tools = await list_mcp_tools(self.backend_session, timeout_s=self.client_config.timeout_s)

        from mcp.server.lowlevel import Server as MCPServer
        from mcp.server.streamable_http_manager import StreamableHTTPSessionManager

        proxy_server = MCPServer(name=f'mcp-proxy-{self.proxy_id}')

        @proxy_server.list_tools()
        async def handle_list_tools():
            result = await list_mcp_tools_raw(self.backend_session, timeout_s=self.client_config.timeout_s)
            return result.tools

        @proxy_server.call_tool()
        async def handle_call_tool(name, arguments):
            return await self._call_tool(name, arguments or {}, proxy_server.request_context.meta)

        self.session_manager = StreamableHTTPSessionManager(
            app=proxy_server,
            stateless=True,
            json_response=True,
        )

        manager_ready = asyncio.Event()
        manager_error: list[BaseException] = []

        async def run_manager() -> None:
            try:
                async with self.session_manager.run():
                    manager_ready.set()
                    await self._shutdown_event.wait()
            except BaseException as exc:
                manager_error.append(exc)
                manager_ready.set()
                raise

        self._manager_task = asyncio.create_task(run_manager(), name=f'mcp-proxy-manager:{self.proxy_id}')
        await manager_ready.wait()
        if manager_error:
            if self.backend is not None:
                await self.backend.close()
                self.backend = None
            raise RuntimeError(f'Failed to start MCP proxy manager: {manager_error[0]}')

        logger.info('MCP proxy %s started with %s tools at /mcp/proxy/%s', self.proxy_id, len(tools), self.proxy_id)
        return tools

    async def _call_tool(self, name: str, arguments: dict[str, Any], meta: Any | None) -> Any:
        with mcp_server_span(meta, self.server_name, name):
            outbound_meta = custom_mcp_meta(meta) if is_otel_enabled() else None
            return await call_mcp_tool_raw(
                self.backend_session,
                name,
                arguments,
                timeout_s=self.client_config.timeout_s,
                meta=outbound_meta,
                server_name=self.server_name,
            )

    async def stop(self) -> None:
        self._shutdown_event.set()
        if self._manager_task:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(self._manager_task, timeout=5.0)
        if self.backend is not None:
            await self.backend.close()
            self.backend = None
        logger.info('MCP proxy %s stopped', self.proxy_id)


class MCPProxyASGI:
    """ASGI app that routes MCP streamable HTTP requests to the correct proxy session."""

    async def __call__(
        self,
        scope: MutableMapping[str, Any],
        receive: Callable[[], Awaitable[MutableMapping[str, Any]]],
        send: Callable[[MutableMapping[str, Any]], Awaitable[None]],
    ) -> None:
        if scope['type'] != 'http':
            return

        full_path: str = scope.get('path', '')
        root_path: str = scope.get('root_path', '')
        relative_path = full_path[len(root_path) :] if root_path and full_path.startswith(root_path) else full_path

        proxy_id = relative_path.strip('/').split('/')[0] if relative_path.strip('/') else ''
        proxy = MCP_PROXY_SESSIONS.get(proxy_id)

        if not proxy or not proxy.session_manager:
            from starlette.responses import Response

            response = Response(status_code=404, content='MCP proxy not found')
            await response(scope, receive, send)
            return

        await proxy.session_manager.handle_request(scope, receive, send)


mcp_proxy_asgi = MCPProxyASGI()


__all__ = [
    'MCP_PROXY_SESSIONS',
    'MCPProxyASGI',
    'MCPProxySession',
    'MCPStartProxyRequest',
    'MCPStartProxyResponse',
    'MCPStopProxyRequest',
    'MCPStopProxyResponse',
    'MCPToolInfo',
    'mcp_proxy_asgi',
    'mcp_start_proxy',
    'mcp_stop_proxy',
]
