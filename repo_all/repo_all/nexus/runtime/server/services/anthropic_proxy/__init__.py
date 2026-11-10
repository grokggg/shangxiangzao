"""Anthropic API Proxy Service.

This module provides an Anthropic Messages API compatible proxy server
that can be used to run Anthropic clients (like Claude Code) with
OpenAI-compatible backends.

Usage:
    from nexus.runtime.server.services.anthropic_proxy import AnthropicProxy

    proxy = AnthropicProxy(
        runtime_socketio_url='http://localhost:8023',
        listen_port=8082
    )
    await proxy.start()

    # Run Claude Code with ANTHROPIC_BASE_URL=http://localhost:8082

    await proxy.stop()
"""

from typing import TYPE_CHECKING

from .converter import anthropic_to_openai, openai_to_anthropic
from .models import (
    ANTHROPIC_SERVER_TOOL_TYPE_PREFIXES,
    AnthropicServerTool,
    ContentBlock,
    DocumentContent,
    DocumentSource,
    ErrorDetail,
    ErrorResponse,
    ImageContent,
    ImageSource,
    Message,
    MessagesRequest,
    MessagesResponse,
    RedactedThinkingContent,
    SystemContent,
    TextContent,
    ThinkingConfig,
    Tool,
    ToolChoice,
    ToolDefinition,
    ToolInputSchema,
    ToolResultContent,
    ToolUseContent,
    Usage,
    get_thinking_config,
    is_anthropic_server_tool_type,
)
from .streaming import simulate_streaming


if TYPE_CHECKING:
    from .server import AnthropicProxy, LLMBridge, run_proxy


def __getattr__(name: str):
    if name in {'AnthropicProxy', 'LLMBridge', 'run_proxy'}:
        from .server import AnthropicProxy, LLMBridge, run_proxy

        values = {
            'AnthropicProxy': AnthropicProxy,
            'LLMBridge': LLMBridge,
            'run_proxy': run_proxy,
        }
        return values[name]
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')


__all__ = [
    # Server
    'AnthropicProxy',
    'LLMBridge',
    'run_proxy',
    # Streaming
    'simulate_streaming',
    # Converter
    'anthropic_to_openai',
    'openai_to_anthropic',
    # Models
    'ANTHROPIC_SERVER_TOOL_TYPE_PREFIXES',
    'AnthropicServerTool',
    'ContentBlock',
    'DocumentContent',
    'DocumentSource',
    'ErrorDetail',
    'ErrorResponse',
    'ImageContent',
    'ImageSource',
    'Message',
    'MessagesRequest',
    'MessagesResponse',
    'RedactedThinkingContent',
    'SystemContent',
    'TextContent',
    'ThinkingConfig',
    'Tool',
    'ToolChoice',
    'ToolDefinition',
    'ToolInputSchema',
    'ToolResultContent',
    'ToolUseContent',
    'Usage',
    'get_thinking_config',
    'is_anthropic_server_tool_type',
]
