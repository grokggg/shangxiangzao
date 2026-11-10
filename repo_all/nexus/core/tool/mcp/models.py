"""Shared MCP models for core and runtime layers."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, model_validator


class MCPTransportKind(str, Enum):
    STDIO = 'stdio'
    SSE = 'sse'
    STREAMABLE_HTTP = 'streamable_http'
    AIOHTTP_STREAMABLE_HTTP = 'aiohttp_streamable_http'


class MCPClientConfig(BaseModel):
    transport: MCPTransportKind
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    cwd: str | None = None
    url: str | None = None
    headers: dict[str, str] = Field(default_factory=dict)
    timeout_s: float | None = None

    @model_validator(mode='after')
    def _validate_transport_fields(self) -> MCPClientConfig:
        if self.transport == MCPTransportKind.STDIO:
            if not self.command:
                raise ValueError('stdio transport requires command')
        else:
            if not self.url:
                raise ValueError(f'{self.transport} transport requires url')
        return self


class MCPServerSpec(BaseModel):
    name: str
    prefix: str | None = None
    runtime_name: str | None = None
    nexus_managed: bool = False
    client: MCPClientConfig

    @model_validator(mode='before')
    @classmethod
    def _normalize_legacy_shape(cls, data: Any) -> Any:
        if not isinstance(data, dict) or 'client' in data:
            return data

        client_data: dict[str, Any] = {
            'transport': data.get('transport'),
            'command': data.get('command'),
            'args': data.get('args', []),
            'env': data.get('env', {}),
            'cwd': data.get('cwd'),
            'url': data.get('url'),
            'headers': data.get('headers', {}),
            'timeout_s': data.get('timeout_s'),
        }

        if client_data['transport'] is None:
            if client_data.get('command'):
                client_data['transport'] = MCPTransportKind.STDIO
            elif client_data.get('url'):
                client_data['transport'] = MCPTransportKind.AIOHTTP_STREAMABLE_HTTP

        return {
            'name': data.get('name'),
            'prefix': data.get('prefix'),
            'runtime_name': data.get('runtime_name'),
            'nexus_managed': data.get('nexus_managed', False),
            'client': client_data,
        }


class MCPToolInfo(BaseModel):
    name: str
    description: str | None = None
    input_schema: dict[str, Any] = Field(default_factory=dict)


class MCPToolResult(BaseModel):
    content: list[dict[str, Any]] = Field(default_factory=list)
    is_error: bool = False


class ConnectedMCPTool(BaseModel):
    full_name: str
    server_name: str
    tool_name: str
    description: str | None = None
    input_schema: dict[str, Any] = Field(default_factory=dict)
    server_key: str


class MCPStartProxyRequest(BaseModel):
    client: MCPClientConfig
    server_name: str = 'mcp'
    nexus_managed: bool = False


class MCPStartProxyResponse(BaseModel):
    proxy_id: str
    endpoint_path: str
    tools: list[MCPToolInfo]


class MCPStopProxyRequest(BaseModel):
    proxy_id: str


class MCPStopProxyResponse(BaseModel):
    status: str


__all__ = [
    'ConnectedMCPTool',
    'MCPClientConfig',
    'MCPServerSpec',
    'MCPStartProxyRequest',
    'MCPStartProxyResponse',
    'MCPStopProxyRequest',
    'MCPStopProxyResponse',
    'MCPToolInfo',
    'MCPToolResult',
    'MCPTransportKind',
]
