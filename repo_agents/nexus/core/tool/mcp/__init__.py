"""MCP integration for the core tool layer."""

from .client import (
    ConnectedMCPSession,
    call_mcp_tool,
    call_mcp_tool_raw,
    connect_mcp_session,
    list_mcp_tools,
    list_mcp_tools_raw,
    normalize_mcp_tool_result,
)
from .manager import AttachedMCPServer, MCPToolManager
from .models import (
    ConnectedMCPTool,
    MCPClientConfig,
    MCPServerSpec,
    MCPStartProxyRequest,
    MCPStartProxyResponse,
    MCPStopProxyRequest,
    MCPStopProxyResponse,
    MCPToolInfo,
    MCPToolResult,
    MCPTransportKind,
)


__all__ = [
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
]
