"""ExecutionContext-scoped MCP tool manager."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

from mcp import ClientSession
from openai.types.chat import ChatCompletionContentPartParam

from nexus.core.registry import EntryNotFoundError

from .client import ConnectedMCPSession, call_mcp_tool, connect_mcp_session, list_mcp_tools
from .models import ConnectedMCPTool, MCPClientConfig, MCPServerSpec, MCPToolInfo, MCPTransportKind
from ..registry import tool_registry
from ..types import ToolResponse


if TYPE_CHECKING:
    from nexus.core.context import ExecutionContext


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class AttachedMCPServer:
    server_key: str
    spec: MCPServerSpec
    connection: ConnectedMCPSession
    tools: list[MCPToolInfo]
    runtime: Any | None = None
    runtime_proxy_id: str | None = None
    tool_names: list[str] = field(default_factory=list)

    @property
    def session(self) -> ClientSession:
        return self.connection.session

    async def close(self) -> None:
        close_error: BaseException | None = None
        try:
            await self.connection.close()
        except BaseException as exc:
            close_error = exc
        if self.runtime is not None and self.runtime_proxy_id is not None:
            try:
                await asyncio.wait_for(self.runtime.mcp_stop_proxy(self.runtime_proxy_id), timeout=60)
            except (Exception, asyncio.TimeoutError) as exc:
                logger.warning('Failed to stop MCP runtime proxy %s: %s', self.runtime_proxy_id, exc)
        if close_error is not None:
            raise close_error


class MCPToolManager:
    """Manage context-scoped MCP servers and tools."""

    def __init__(self, ctx: ExecutionContext):
        self._ctx = ctx
        self._servers: dict[str, AttachedMCPServer] = {}
        self._tools: dict[str, ConnectedMCPTool] = {}

    @property
    def tools(self) -> dict[str, ConnectedMCPTool]:
        return self._tools

    async def load(
        self,
        specs: list[MCPServerSpec | dict[str, Any]],
        *,
        clear_existing: bool = False,
    ) -> dict[str, ConnectedMCPTool]:
        normalized_specs = [MCPServerSpec.model_validate(spec) for spec in specs]

        if clear_existing:
            await self.close()

        newly_loaded: list[str] = []
        try:
            for spec in normalized_specs:
                server_key = self._server_key(spec)
                if server_key in self._servers:
                    continue

                attached = await self._attach_server(server_key, spec)
                self._register_attached_server(attached)
                self._servers[server_key] = attached
                newly_loaded.append(server_key)
        except Exception:
            for server_key in reversed(newly_loaded):
                attached = self._servers.pop(server_key, None)
                if attached is not None:
                    self._remove_server_tools(attached)
                    with contextlib.suppress(Exception):
                        await attached.close()
            raise

        return dict(self._tools)

    async def close(self) -> None:
        servers = list(self._servers.values())
        for attached in servers:
            self._remove_server_tools(attached)
        self._servers.clear()
        self._tools.clear()

        async def _close_one(attached: AttachedMCPServer) -> None:
            try:
                await attached.close()
            except BaseException as exc:
                logger.warning('Failed to close MCP server %s: %s', attached.spec.name, exc)

        await asyncio.gather(*[_close_one(s) for s in reversed(servers)])

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        schemas: list[dict[str, Any]] = []
        for tool in self._tools.values():
            schemas.append({
                'type': 'function',
                'function': {
                    'name': tool.full_name,
                    'description': tool.description or f'MCP tool: {tool.tool_name}',
                    'parameters': tool.input_schema or {'type': 'object', 'properties': {}},
                },
            })
        return schemas

    async def call_tool(self, full_name: str, arguments: dict[str, Any] | None = None) -> ToolResponse:
        tool = self._tools.get(full_name)
        if tool is None:
            raise KeyError(f"MCP tool '{full_name}' not found. Available: {sorted(self._tools.keys())}")

        attached = self._servers[tool.server_key]
        call_arguments = arguments or {}
        result = await call_mcp_tool(
            attached.session,
            tool.tool_name,
            call_arguments,
            timeout_s=attached.spec.client.timeout_s,
            source_name=attached.spec.name,
        )
        records = self._ctx.tool_state.setdefault('__mcp_call_records__', [])
        records.append({
            'server': tool.server_name,
            'tool_name': tool.tool_name,
            'full_name': full_name,
            'arguments': call_arguments,
            'response': result.model_dump(mode='json'),
            'response_content': result.content,
            'is_error': result.is_error,
        })
        return ToolResponse(
            output=cast(list[ChatCompletionContentPartParam | dict], result.content),
            meta={
                'mcp_server': tool.server_name,
                'mcp_tool': tool.tool_name,
                'is_error': result.is_error,
            },
        )

    async def _attach_server(self, server_key: str, spec: MCPServerSpec) -> AttachedMCPServer:
        max_retries = 3
        retry_delay = 2.0

        if spec.runtime_name:
            runtime = await self._ctx.get_runtime(spec.runtime_name)
            start_result = await runtime.mcp_start_proxy(
                spec.client,
                server_name=spec.name,
                nexus_managed=spec.nexus_managed,
            )
            proxy_url = f'{runtime.base_url}{start_result.endpoint_path}'
            last_error: BaseException | None = None
            for attempt in range(max_retries):
                try:
                    proxy_connection = await connect_mcp_session(
                        MCPClientConfig(transport=MCPTransportKind.AIOHTTP_STREAMABLE_HTTP, url=proxy_url),
                        server_name=spec.name,
                        nexus_managed=spec.nexus_managed,
                    )
                    return AttachedMCPServer(
                        server_key=server_key,
                        spec=spec,
                        connection=proxy_connection,
                        tools=start_result.tools,
                        runtime=runtime,
                        runtime_proxy_id=start_result.proxy_id,
                    )
                except Exception as exc:
                    last_error = exc
                    if attempt < max_retries - 1:
                        logger.warning(
                            'MCP session connect failed for %s (attempt %d/%d): %s. Retrying in %.1fs...',
                            spec.name,
                            attempt + 1,
                            max_retries,
                            exc,
                            retry_delay,
                        )
                        await asyncio.sleep(retry_delay)
                        retry_delay *= 2
                    else:
                        with contextlib.suppress(Exception, asyncio.TimeoutError):
                            await asyncio.wait_for(runtime.mcp_stop_proxy(start_result.proxy_id), timeout=60)
                        raise
            raise last_error  # type: ignore[misc]

        last_error = None
        for attempt in range(max_retries):
            try:
                connection = await connect_mcp_session(
                    spec.client,
                    server_name=spec.name,
                    nexus_managed=spec.nexus_managed,
                )
                try:
                    tools = await list_mcp_tools(connection.session, timeout_s=spec.client.timeout_s)
                except Exception:
                    with contextlib.suppress(Exception):
                        await connection.close()
                    raise
                return AttachedMCPServer(server_key=server_key, spec=spec, connection=connection, tools=tools)
            except Exception as exc:
                last_error = exc
                if attempt < max_retries - 1:
                    logger.warning(
                        'MCP session connect failed for %s (attempt %d/%d): %s. Retrying in %.1fs...',
                        spec.name,
                        attempt + 1,
                        max_retries,
                        exc,
                        retry_delay,
                    )
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    raise
        raise last_error  # type: ignore[misc]

    def _register_attached_server(self, attached: AttachedMCPServer) -> None:
        prefix = attached.spec.prefix or f'{attached.spec.name}_'

        for tool in attached.tools:
            full_name = f'{prefix}{tool.name}'
            self._ensure_tool_name_available(full_name)
            connected = ConnectedMCPTool(
                full_name=full_name,
                server_name=attached.spec.name,
                tool_name=tool.name,
                description=tool.description,
                input_schema=tool.input_schema,
                server_key=attached.server_key,
            )
            self._tools[full_name] = connected
            attached.tool_names.append(full_name)

    def _remove_server_tools(self, attached: AttachedMCPServer) -> None:
        for tool_name in attached.tool_names:
            self._tools.pop(tool_name, None)
        attached.tool_names.clear()

    def _ensure_tool_name_available(self, full_name: str) -> None:
        if full_name in self._tools:
            raise ValueError(f"Duplicate MCP tool name '{full_name}'")
        try:
            tool_registry.get_entry(full_name)
        except EntryNotFoundError:
            return
        raise ValueError(f"MCP tool name '{full_name}' conflicts with static tool registry")

    @staticmethod
    def _server_key(spec: MCPServerSpec) -> str:
        payload = json.dumps(spec.model_dump(mode='json', exclude_none=True), sort_keys=True, separators=(',', ':'))
        return hashlib.sha1(payload.encode('utf-8')).hexdigest()[:16]


__all__ = ['AttachedMCPServer', 'MCPToolManager']
