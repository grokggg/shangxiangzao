"""Core tool package with lazy exports to avoid import cycles."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .registry import get_tool_meta, get_tool_schema, tool_registry
from .types import (
    InputT,
    ToolContext,
    ToolExposure,
    ToolFunc,
    ToolMeta,
    ToolMetaOverride,
    ToolParamOverride,
    ToolResponse,
    ToolSearchConfig,
    ToolType,
)


if TYPE_CHECKING:
    from .executor import ToolArgumentValidationError, ToolExecutor, ToolNameResolutionError
    from .mcp import (
        AttachedMCPServer,
        ConnectedMCPSession,
        ConnectedMCPTool,
        MCPClientConfig,
        MCPServerSpec,
        MCPStartProxyRequest,
        MCPStartProxyResponse,
        MCPStopProxyRequest,
        MCPStopProxyResponse,
        MCPToolInfo,
        MCPToolManager,
        MCPToolResult,
        MCPTransportKind,
        call_mcp_tool,
        call_mcp_tool_raw,
        connect_mcp_session,
        list_mcp_tools,
        list_mcp_tools_raw,
        normalize_mcp_tool_result,
    )


def __getattr__(name: str) -> Any:
    if name == 'ToolExecutor':
        from .executor import ToolExecutor

        return ToolExecutor

    if name == 'ToolArgumentValidationError':
        from .executor import ToolArgumentValidationError

        return ToolArgumentValidationError

    if name == 'ToolNameResolutionError':
        from .executor import ToolNameResolutionError

        return ToolNameResolutionError

    if name in {
        'AttachedMCPServer',
        'ConnectedMCPSession',
        'ConnectedMCPTool',
        'MCPClientConfig',
        'MCPServerSpec',
        'MCPStartProxyRequest',
        'MCPStartProxyResponse',
        'MCPStopProxyRequest',
        'MCPStopProxyResponse',
        'MCPToolInfo',
        'MCPToolManager',
        'MCPToolResult',
        'MCPTransportKind',
        'call_mcp_tool',
        'call_mcp_tool_raw',
        'connect_mcp_session',
        'list_mcp_tools',
        'list_mcp_tools_raw',
        'normalize_mcp_tool_result',
    }:
        from .mcp import __dict__ as mcp_exports

        return mcp_exports[name]

    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')


__all__ = [
    'AttachedMCPServer',
    'ConnectedMCPSession',
    'ConnectedMCPTool',
    'InputT',
    'MCPClientConfig',
    'MCPServerSpec',
    'MCPStartProxyRequest',
    'MCPStartProxyResponse',
    'MCPStopProxyRequest',
    'MCPStopProxyResponse',
    'MCPToolInfo',
    'MCPToolManager',
    'MCPToolResult',
    'MCPTransportKind',
    'ToolContext',
    'ToolExposure',
    'ToolArgumentValidationError',
    'ToolExecutor',
    'ToolFunc',
    'ToolMeta',
    'ToolMetaOverride',
    'ToolNameResolutionError',
    'ToolParamOverride',
    'ToolResponse',
    'ToolSearchConfig',
    'ToolType',
    'call_mcp_tool',
    'call_mcp_tool_raw',
    'connect_mcp_session',
    'get_tool_meta',
    'get_tool_schema',
    'list_mcp_tools',
    'list_mcp_tools_raw',
    'normalize_mcp_tool_result',
    'tool_registry',
]
