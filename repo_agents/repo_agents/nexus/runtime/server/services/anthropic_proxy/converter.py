"""Convert between Anthropic Messages API and OpenAI Chat Completion API formats.

This module provides bidirectional conversion between Anthropic's Messages API
and OpenAI's Chat Completion API, enabling Anthropic clients (like Claude Code)
to use OpenAI-compatible backends.

Reference: https://github.com/1rgs/claude-code-proxy
"""

from __future__ import annotations

import json
import logging
import os
import uuid

from typing import Any, Literal

from .models import (
    KNOWN_THINKING_FIELDS,
    AnthropicServerTool,
    ContentBlock,
    DocumentContent,
    DocumentSource,
    ImageContent,
    Message,
    MessagesRequest,
    MessagesResponse,
    RedactedThinkingContent,
    SystemContent,
    TextContent,
    ThinkingConfig,
    ThinkingContent,
    ToolReferenceContent,
    ToolResultContent,
    ToolUseContent,
    Usage,
    get_thinking_config,
    is_anthropic_server_tool_type,
)


logger = logging.getLogger(__name__)

# Dedicated logger for before/after conversion dumps. Users can bump its level
# independently of the proxy's general logger via ``set_conversion_log_level``,
# or set ``NEXUS_CONVERSION_LOG_LEVEL`` to override the module-wide default.
conversion_logger = logging.getLogger('nexus.anthropic_proxy.converter')

_CLIENT_META_HEADERS = {
    'x-claude-code-agent-id': 'agent_id',
    'x-claude-code-session-id': 'session_id',
}


def client_request_meta(headers: Any) -> dict[str, str]:
    """Project supported Claude/HyCode identity headers into Gateway metadata."""
    items = headers.items() if hasattr(headers, 'items') else ()
    lowered = {str(key).lower(): value for key, value in items}
    return {meta_key: str(lowered[header]) for header, meta_key in _CLIENT_META_HEADERS.items() if lowered.get(header)}


def _initial_conversion_log_level() -> int:
    """Resolve the module-wide default level, honouring the env var override."""
    raw = os.getenv('NEXUS_CONVERSION_LOG_LEVEL', '').strip()
    if not raw:
        return logging.DEBUG
    name = raw.upper()
    mapping = getattr(logging, '_nameToLevel', None)
    if isinstance(mapping, dict) and name in mapping:
        return mapping[name]
    try:
        return int(raw)
    except ValueError:
        return logging.DEBUG


_conversion_log_level: int = _initial_conversion_log_level()
conversion_logger.setLevel(_conversion_log_level)


def set_conversion_log_level(level: str | int) -> None:
    """Set the level used for anthropic<->openai conversion dumps."""
    global _conversion_log_level
    if isinstance(level, str):
        name = level.upper()
        # ``getLevelNamesMapping`` is 3.11+; ``_nameToLevel`` is stable across
        # the supported range and avoids the deprecated ``getLevelName`` path
        # that Pylance complains about.
        mapping = getattr(logging, '_nameToLevel', None)
        if not isinstance(mapping, dict) or name not in mapping:
            raise ValueError(f'Invalid log level: {level!r}')
        _conversion_log_level = mapping[name]
    else:
        _conversion_log_level = int(level)
    conversion_logger.setLevel(_conversion_log_level)


def _dump(direction: str, before: Any, after: Any) -> None:
    """Emit a multi-line before/after log entry for one conversion step."""
    from nexus.utils.common import pformat

    if not conversion_logger.isEnabledFor(_conversion_log_level):
        return
    message = f'\n=== conversion: {direction} ===\nbefore:\n{pformat(before)}\nafter:\n{pformat(after)}'
    conversion_logger.log(_conversion_log_level, '%s', message)


# ============== Anthropic -> OpenAI Conversion ==============


# Map common document media types to a sensible filename extension so backends
# that key off the filename (litellm, OpenAI file inputs) recognise the format.
_DOCUMENT_MEDIA_EXTENSIONS: dict[str, str] = {
    'application/pdf': 'pdf',
    'text/plain': 'txt',
    'text/markdown': 'md',
    'text/csv': 'csv',
    'application/json': 'json',
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document': 'docx',
}


def _document_media_extension(media_type: str | None) -> str:
    if not media_type:
        return 'bin'
    if media_type in _DOCUMENT_MEDIA_EXTENSIONS:
        return _DOCUMENT_MEDIA_EXTENSIONS[media_type]
    tail = media_type.rsplit('/', 1)[-1]
    return tail or 'bin'


def convert_document_to_openai(
    source: dict[str, Any] | DocumentSource | None,
    title: str | None = None,
) -> dict[str, Any]:
    """Convert an Anthropic ``document`` source into an OpenAI content part.

    Claude Code's Read tool turns a PDF into a ``document`` block with a base64
    source (``{"type": "base64", "media_type": "application/pdf", ...}``). The
    OpenAI-compatible representation is a ``file`` content part carrying a data
    URI. Non-base64 sources (``text``/``url``/unknown) degrade to a ``text`` part
    so the request still reaches the backend instead of being rejected during
    request-body validation.
    """
    if isinstance(source, DocumentSource):
        src_type = source.type
        media_type = source.media_type
        data = source.data
        text = source.text
        url = source.url
    elif isinstance(source, dict):
        src_type = source.get('type')
        media_type = source.get('media_type')
        data = source.get('data')
        text = source.get('text')
        url = source.get('url')
    else:
        return {'type': 'text', 'text': '[Document]'}

    if src_type == 'base64' and data:
        resolved_media_type = media_type or 'application/pdf'
        filename = title or f'document.{_document_media_extension(resolved_media_type)}'
        file_data = f'data:{resolved_media_type};base64,{data}'
        return {'type': 'file', 'file': {'filename': filename, 'file_data': file_data}}
    if src_type == 'text' and text:
        return {'type': 'text', 'text': text}
    if src_type == 'url' and url:
        return {'type': 'text', 'text': f'[Document: {url}]'}

    descriptor = title or media_type or src_type or 'document'
    return {'type': 'text', 'text': f'[Document: {descriptor}]'}


def convert_content_to_openai(content: str | list[ContentBlock]) -> str | list[dict[str, Any]]:
    """Convert Anthropic content to OpenAI format.

    Args:
        content: Anthropic content (string or list of content blocks)

    Returns:
        OpenAI format content (string or list of content parts)
    """
    if isinstance(content, str):
        return content

    result: list[dict[str, Any]] = []
    for block in content:
        if isinstance(block, dict):
            block_type = block.get('type')
            if block_type == 'text':
                result.append({'type': 'text', 'text': block.get('text', '')})
            elif block_type == 'image':
                source = block.get('source', {})
                media_type = source.get('media_type', 'image/png')
                data = source.get('data', '')
                result.append({
                    'type': 'image_url',
                    'image_url': {'url': f'data:{media_type};base64,{data}'},
                })
            elif block_type == 'document':
                result.append(convert_document_to_openai(block.get('source'), block.get('title')))
            elif block_type == 'tool_result':
                # Tool results need special handling - convert text and images
                tool_content = block.get('content', '')
                if isinstance(tool_content, list):
                    for item in tool_content:
                        if item.get('type') == 'text':
                            result.append({'type': 'text', 'text': item.get('text', '')})
                        elif item.get('type') == 'image':
                            source = item.get('source', {})
                            media_type = source.get('media_type', 'image/png')
                            data = source.get('data', '')
                            result.append({
                                'type': 'image_url',
                                'image_url': {'url': f'data:{media_type};base64,{data}'},
                            })
                        elif item.get('type') == 'document':
                            result.append(convert_document_to_openai(item.get('source'), item.get('title')))
                        elif item.get('type') == 'tool_reference':
                            tool_name = item.get('tool_name', '')
                            result.append({'type': 'text', 'text': f'[ToolSearch] Tool now available: {tool_name}'})
                else:
                    result.append({'type': 'text', 'text': tool_content})
            elif block_type == 'tool_use':
                # Tool use in content - shouldn't happen in user messages
                result.append({
                    'type': 'text',
                    'text': f'[Tool call: {block.get("name")}({json.dumps(block.get("input", {}))})]',
                })
        elif isinstance(block, TextContent):
            result.append({'type': 'text', 'text': block.text})
        elif isinstance(block, ImageContent):
            result.append({
                'type': 'image_url',
                'image_url': {'url': f'data:{block.source.media_type};base64,{block.source.data}'},
            })
        elif isinstance(block, DocumentContent):
            result.append(convert_document_to_openai(block.source, block.title))
        elif isinstance(block, ToolResultContent):
            tool_content = block.content
            if isinstance(tool_content, list):
                for item in tool_content:
                    if isinstance(item, TextContent):
                        result.append({'type': 'text', 'text': item.text})
                    elif isinstance(item, ImageContent):
                        result.append({
                            'type': 'image_url',
                            'image_url': {'url': f'data:{item.source.media_type};base64,{item.source.data}'},
                        })
                    elif isinstance(item, DocumentContent):
                        result.append(convert_document_to_openai(item.source, item.title))
                    elif isinstance(item, ToolReferenceContent):
                        result.append({'type': 'text', 'text': f'[ToolSearch] Tool now available: {item.tool_name}'})
            else:
                result.append({'type': 'text', 'text': tool_content})
        elif isinstance(block, ToolUseContent):
            result.append({
                'type': 'text',
                'text': f'[Tool call: {block.name}({json.dumps(block.input)})]',
            })

    # Simplify if only one text block
    if len(result) == 1 and result[0].get('type') == 'text':
        return result[0]['text']

    return result


def convert_messages_to_openai(
    messages: list[Message],
    thinking_config: ThinkingConfig | None = None,
) -> list[dict[str, Any]]:
    """Convert Anthropic messages to OpenAI format.

    Handles the special case where tool_result blocks need to be converted
    to separate tool messages in OpenAI format.

    Args:
        messages: List of Anthropic messages
        thinking_config: How to encode thinking content in OpenAI format

    Returns:
        List of OpenAI format messages
    """
    thinking_config = thinking_config or get_thinking_config()
    result: list[dict[str, Any]] = []

    for msg in messages:
        if isinstance(msg, dict):
            role = msg.get('role', 'user')
            content = msg.get('content', '')
        else:
            role = msg.role
            content = msg.content

        # Check for tool results that need special handling
        if role == 'user' and isinstance(content, list):
            tool_results = []
            other_content = []

            for block in content:
                if isinstance(block, dict):
                    if block.get('type') == 'tool_result':
                        tool_results.append(block)
                    else:
                        other_content.append(block)
                elif isinstance(block, ToolResultContent):
                    tool_results.append({
                        'type': 'tool_result',
                        'tool_use_id': block.tool_use_id,
                        'content': block.content,
                        'is_error': block.is_error,
                    })
                else:
                    other_content.append(block)

            # Add tool result messages first
            for tool_result in tool_results:
                tool_content = tool_result.get('content', '')
                tool_parts: list[dict[str, Any]] = []

                if isinstance(tool_content, list):
                    for item in tool_content:
                        if isinstance(item, dict):
                            if item.get('type') == 'text':
                                tool_parts.append({'type': 'text', 'text': item.get('text', '')})
                            elif item.get('type') == 'image':
                                source = item.get('source', {})
                                media_type = source.get('media_type', 'image/png')
                                data = source.get('data', '')
                                tool_parts.append({
                                    'type': 'image_url',
                                    'image_url': {'url': f'data:{media_type};base64,{data}'},
                                })
                            elif item.get('type') == 'document':
                                tool_parts.append(convert_document_to_openai(item.get('source'), item.get('title')))
                            elif item.get('type') == 'tool_reference':
                                tool_name = item.get('tool_name', '')
                                tool_parts.append({
                                    'type': 'text',
                                    'text': f'[ToolSearch] Tool now available: {tool_name}',
                                })
                        elif isinstance(item, TextContent):
                            tool_parts.append({'type': 'text', 'text': item.text})
                        elif isinstance(item, ImageContent):
                            tool_parts.append({
                                'type': 'image_url',
                                'image_url': {'url': f'data:{item.source.media_type};base64,{item.source.data}'},
                            })
                        elif isinstance(item, DocumentContent):
                            tool_parts.append(convert_document_to_openai(item.source, item.title))
                        elif isinstance(item, ToolReferenceContent):
                            tool_parts.append({
                                'type': 'text',
                                'text': f'[ToolSearch] Tool now available: {item.tool_name}',
                            })
                else:
                    tool_parts.append({'type': 'text', 'text': tool_content})

                result.append({
                    'role': 'tool',
                    'tool_call_id': tool_result.get('tool_use_id', ''),
                    'content': tool_parts,
                })

            # Add remaining content as user message if any
            if other_content:
                result.append({
                    'role': 'user',
                    'content': convert_content_to_openai(other_content),
                })
        elif role == 'assistant' and isinstance(content, list):
            text_parts: list[str] = []
            thinking_blocks: list[dict[str, Any]] = []
            tool_calls: list[dict[str, Any]] = []

            for block in content:
                if isinstance(block, dict):
                    block_type = block.get('type')
                    if block_type == 'tool_use':
                        tool_calls.append({
                            'id': block.get('id', str(uuid.uuid4())),
                            'type': 'function',
                            'function': {
                                'name': block.get('name', ''),
                                'arguments': json.dumps(block.get('input', {})),
                            },
                        })
                    elif block_type == 'thinking' or block_type == 'redacted_thinking':
                        thinking_blocks.append(block)
                    elif block_type == 'text':
                        text_parts.append(block.get('text', ''))
                elif isinstance(block, ToolUseContent):
                    tool_calls.append({
                        'id': block.id,
                        'type': 'function',
                        'function': {
                            'name': block.name,
                            'arguments': json.dumps(block.input),
                        },
                    })
                elif isinstance(block, ThinkingContent):
                    thinking_blocks.append({
                        'type': 'thinking',
                        'thinking': block.thinking,
                        'signature': block.signature,
                    })
                elif isinstance(block, RedactedThinkingContent):
                    thinking_blocks.append({'type': 'redacted_thinking', 'data': block.data})
                elif isinstance(block, TextContent):
                    text_parts.append(block.text)

            joined_text = '\n'.join(text_parts)
            # Upstream Anthropic-format converters (litellm's anthropic_messages_pt, used by
            # Bedrock invoke / gcp_anthropic / anthropic routes) rewrite blank content into a
            # visible "[System: Empty message content sanitised to satisfy protocol]" text block.
            # null is the OpenAI-native way to say "tool calls only, no text".
            text_content: str | None = None if tool_calls and not joined_text.strip() else joined_text

            msg_dict: dict[str, Any] = {
                'role': 'assistant',
                'content': text_content,
            }

            if thinking_blocks:
                if thinking_config.format == 'blocks':
                    msg_dict[thinking_config.request_field] = thinking_blocks
                else:
                    reasoning_parts = [b.get('thinking', '') for b in thinking_blocks if b.get('type') == 'thinking']
                    if reasoning_parts:
                        msg_dict[thinking_config.request_field] = '\n'.join(reasoning_parts)

            if tool_calls:
                msg_dict['tool_calls'] = tool_calls

            result.append(msg_dict)
        else:
            result.append({
                'role': role,
                'content': convert_content_to_openai(content),
            })

    return result


def convert_system_to_openai(system: str | list[SystemContent] | None) -> str | None:
    """Convert Anthropic system prompt to OpenAI format.

    Args:
        system: Anthropic system prompt (string or list of system content blocks)

    Returns:
        OpenAI format system prompt (string or None)
    """
    if system is None:
        return None

    if isinstance(system, str):
        return system

    # Concatenate all text blocks
    parts = []
    for block in system:
        if isinstance(block, dict):
            parts.append(block.get('text', ''))
        elif isinstance(block, SystemContent):
            parts.append(block.text)

    return '\n'.join(parts)


def convert_system_content_to_openai(
    system: str | list[SystemContent] | None,
) -> str | list[dict[str, Any]] | None:
    """Convert top-level Anthropic system content without collapsing block boundaries."""
    if system is None or isinstance(system, str):
        return system

    blocks: list[dict[str, Any]] = []
    for block in system:
        if isinstance(block, dict):
            text = block.get('text', '')
            cache_control = block.get('cache_control')
        else:
            text = block.text
            cache_control = block.cache_control
        converted: dict[str, Any] = {'type': 'text', 'text': text}
        if cache_control is not None:
            converted['cache_control'] = cache_control
        blocks.append(converted)
    return blocks


def _tool_to_dict(tool: dict[str, Any] | Any) -> dict[str, Any]:
    if isinstance(tool, dict):
        return tool
    model_dump = getattr(tool, 'model_dump', None)
    if callable(model_dump):
        return model_dump(exclude_none=True)
    return {}


def _is_anthropic_server_tool(tool: dict[str, Any] | Any) -> bool:
    if isinstance(tool, AnthropicServerTool):
        return True
    tool_type = tool.get('type') if isinstance(tool, dict) else getattr(tool, 'type', None)
    return is_anthropic_server_tool_type(tool_type)


def convert_tools_to_openai(tools: list[dict[str, Any] | Any] | None) -> list[dict[str, Any]] | None:
    """Convert Anthropic tools to OpenAI format.

    Args:
        tools: Anthropic tool definitions

    Returns:
        OpenAI format tool definitions
    """
    if tools is None:
        return None

    result = []
    for tool in tools:
        if _is_anthropic_server_tool(tool):
            result.append(_tool_to_dict(tool))
            continue

        tool_data = _tool_to_dict(tool)
        name = tool_data.get('name', '')
        description = tool_data.get('description', '')
        input_schema = tool_data.get('input_schema', {})

        # Clean schema for compatibility
        cleaned_schema = _clean_schema(input_schema)

        result.append({
            'type': 'function',
            'function': {
                'name': name,
                'description': description,
                'parameters': cleaned_schema,
            },
        })

    return result


def _clean_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Clean JSON schema for OpenAI/Gemini compatibility.

    Removes unsupported fields like additionalProperties, default, etc.
    """
    if not isinstance(schema, dict):
        return schema

    result = {}
    for key, value in schema.items():
        # Skip unsupported fields
        if key in ('additionalProperties', 'default', '$schema'):
            continue

        if key == 'properties' and isinstance(value, dict):
            result[key] = {k: _clean_schema(v) for k, v in value.items()}
        elif key == 'items' and isinstance(value, dict) or isinstance(value, dict):
            result[key] = _clean_schema(value)
        elif isinstance(value, list):
            result[key] = [_clean_schema(item) if isinstance(item, dict) else item for item in value]
        else:
            result[key] = value

    return result


def convert_tool_choice_to_openai(tool_choice: dict[str, Any] | None) -> str | dict[str, Any] | None:
    """Convert Anthropic tool_choice to OpenAI format.

    Args:
        tool_choice: Anthropic tool_choice

    Returns:
        OpenAI format tool_choice
    """
    if tool_choice is None:
        return None

    choice_type = tool_choice.get('type') if isinstance(tool_choice, dict) else tool_choice.type

    if choice_type == 'auto':
        return 'auto'
    if choice_type == 'any':
        return 'required'
    if choice_type == 'tool':
        name = tool_choice.get('name') if isinstance(tool_choice, dict) else tool_choice.name
        return {'type': 'function', 'function': {'name': name}}

    return 'auto'


def _tools_summary(request: MessagesRequest) -> list[str]:
    """Names-only summary of the tools carried by this request."""
    names: list[str] = []
    for tool in request.tools or []:
        name = tool.get('name') if isinstance(tool, dict) else getattr(tool, 'name', None)
        if name:
            names.append(name)
    return names


def _system_summary(request: MessagesRequest) -> str | None:
    """Short preview of the system prompt, not the full text."""
    system = convert_system_to_openai(request.system)
    if not system:
        return None
    return f'<{len(system)} chars>'


def _dump_recent_message_conversion(request: MessagesRequest, thinking_config: ThinkingConfig | None = None) -> None:
    """Log the tail of the conversation and its openai translation.

    The Anthropic Messages API is stateless — every request carries the full
    history, so we cannot reliably compute "what's new since last time" from
    inside the proxy (clients may restart sessions or truncate history).
    Dumping the last two messages covers the usual pattern ``(assistant,
    user/tool_result)`` appended per turn, while still giving enough context
    to spot a conversion bug. Tools and system prompt are summarised rather
    than dumped in full to keep the log short.
    """
    if not conversion_logger.isEnabledFor(_conversion_log_level):
        return
    if not request.messages:
        return
    tail = request.messages[-2:]
    before = {
        'messages_tail': [m.model_dump(exclude_none=True) if hasattr(m, 'model_dump') else m for m in tail],
        'total_messages': len(request.messages),
        'tools': _tools_summary(request),
        'system': _system_summary(request),
    }
    after = {'messages_tail': convert_messages_to_openai(list(tail), thinking_config=thinking_config)}
    _dump(direction='anthropic->openai (last <=2 msgs)', before=before, after=after)


def anthropic_to_openai(
    request: MessagesRequest,
    thinking_config: ThinkingConfig | None = None,
) -> dict[str, Any]:
    """Convert Anthropic MessagesRequest to OpenAI Chat Completion format.

    Args:
        request: Anthropic MessagesRequest
        thinking_config: How to encode thinking content in OpenAI format

    Returns:
        OpenAI Chat Completion request parameters
    """
    thinking_config = thinking_config or get_thinking_config()
    _dump_recent_message_conversion(request, thinking_config=thinking_config)

    messages = convert_messages_to_openai(request.messages, thinking_config=thinking_config)

    # Add system message if present
    system = convert_system_content_to_openai(request.system)
    if system:
        messages.insert(0, {'role': 'system', 'content': system})

    result: dict[str, Any] = {
        'model': request.model,
        'messages': messages,
    }

    if request.max_tokens is not None:
        result['max_tokens'] = request.max_tokens

    result['_client_has_thinking'] = request.thinking is not None
    result['_client_requests_haiku'] = 'haiku' in request.model.lower()

    if request.tools:
        tools_data = [t.model_dump() if hasattr(t, 'model_dump') else t for t in request.tools]
        result['tools'] = convert_tools_to_openai(tools_data)

    if request.tool_choice:
        choice_data = (
            request.tool_choice.model_dump() if hasattr(request.tool_choice, 'model_dump') else request.tool_choice
        )
        result['tool_choice'] = convert_tool_choice_to_openai(choice_data)

    if request.temperature is not None:
        result['temperature'] = request.temperature

    if request.top_p is not None:
        result['top_p'] = request.top_p

    if request.stop_sequences:
        result['stop'] = request.stop_sequences

    return result


# ============== OpenAI -> Anthropic Conversion ==============


def _warn_unconsumed_thinking(message: dict[str, Any], consumed_field: str) -> None:
    """Warn if the message contains a known thinking field that was not consumed."""
    for field in KNOWN_THINKING_FIELDS - {consumed_field}:
        if message.get(field):
            logger.warning(
                'Response contains "%s" but ThinkingConfig reads "%s". Possible thinking content dropped.',
                field,
                consumed_field,
            )


def convert_openai_content_to_anthropic(
    message: dict[str, Any],
    thinking_config: ThinkingConfig | None = None,
) -> list[TextContent | ThinkingContent | RedactedThinkingContent | ToolUseContent]:
    """Convert OpenAI response content to Anthropic format.

    Args:
        message: The OpenAI ``choices[0].message`` dict.
        thinking_config: How to read thinking content from OpenAI format.

    Returns:
        List of Anthropic content blocks
    """
    thinking_config = thinking_config or get_thinking_config()
    content = message.get('content')
    tool_calls = message.get('tool_calls')
    thinking_data = message.get(thinking_config.response_field)

    _warn_unconsumed_thinking(message, thinking_config.response_field)

    result: list[TextContent | ThinkingContent | RedactedThinkingContent | ToolUseContent] = []

    if thinking_data:
        if thinking_config.format == 'blocks' and isinstance(thinking_data, list):
            for block in thinking_data:
                block_type = block.get('type') if isinstance(block, dict) else None
                if block_type == 'thinking':
                    result.append(
                        ThinkingContent(
                            type='thinking',
                            thinking=block.get('thinking', ''),
                            signature=block.get('signature', '') if thinking_config.preserve_signature else '',
                        )
                    )
                elif block_type == 'redacted_thinking' and thinking_config.support_redacted:
                    result.append(RedactedThinkingContent(type='redacted_thinking', data=block.get('data', '')))
        else:
            result.append(ThinkingContent(type='thinking', thinking=str(thinking_data)))

    if content:
        result.append(TextContent(type='text', text=content))

    if tool_calls:
        for call in tool_calls:
            function = call.get('function', {})
            try:
                arguments = json.loads(function.get('arguments', '{}'))
            except json.JSONDecodeError:
                arguments = {}

            result.append(
                ToolUseContent(
                    type='tool_use',
                    id=call.get('id', str(uuid.uuid4())),
                    name=function.get('name', ''),
                    input=arguments,
                )
            )

    return result


StopReason = Literal['end_turn', 'max_tokens', 'stop_sequence', 'tool_use']


def convert_stop_reason(finish_reason: str | None) -> StopReason:
    """Convert OpenAI finish_reason to Anthropic stop_reason.

    Args:
        finish_reason: OpenAI finish reason

    Returns:
        Anthropic stop reason
    """
    mapping: dict[str, StopReason] = {
        'stop': 'end_turn',
        'length': 'max_tokens',
        'tool_calls': 'tool_use',
        'content_filter': 'end_turn',
        'function_call': 'tool_use',
    }
    return mapping.get(finish_reason or '', 'end_turn')


def openai_to_anthropic(
    response: dict[str, Any] | Any,
    model: str,
    thinking_config: ThinkingConfig | None = None,
) -> MessagesResponse:
    """Convert OpenAI Chat Completion response to Anthropic MessagesResponse.

    Args:
        response: OpenAI Chat Completion response (dict or Pydantic model)
        model: Original model name from the request
        thinking_config: How to read thinking content from OpenAI format

    Returns:
        Anthropic MessagesResponse
    """
    thinking_config = thinking_config or get_thinking_config()

    # Handle both dict and object responses - convert Pydantic models to dict
    model_dump_fn = getattr(response, 'model_dump', None)
    response_dict: dict[str, Any] = (
        model_dump_fn() if model_dump_fn is not None and callable(model_dump_fn) else response
    )
    # Anthropic clients use this id to delimit assistant turns, while an OpenAI
    # backend may reuse one response id for the whole Gateway session.
    message_id = f'msg_{uuid.uuid4().hex[:24]}'

    choices = response_dict.get('choices', [])
    if not choices:
        return MessagesResponse(
            id=message_id,
            model=model,
            content=[TextContent(type='text', text='')],
            stop_reason='end_turn',
            stop_sequence=None,
            usage=Usage(
                input_tokens=0,
                output_tokens=0,
                cache_creation_input_tokens=0,
                cache_read_input_tokens=0,
            ),
        )

    choice = choices[0]
    message = choice.get('message', {})

    content = convert_openai_content_to_anthropic(message, thinking_config=thinking_config)

    # Ensure at least one content block
    if not content:
        content = [TextContent(type='text', text='')]

    # Handle usage - use 'or {}' to handle both missing key and None value
    usage_data = response_dict.get('usage') or {}

    prompt_tokens = usage_data.get('prompt_tokens') or 0
    completion_tokens = usage_data.get('completion_tokens') or 0

    # Extract cache tokens if available
    cache_creation = 0
    cache_read = 0
    if 'prompt_tokens_details' in usage_data and usage_data['prompt_tokens_details']:
        details = usage_data['prompt_tokens_details']
        cache_creation = details.get('cache_creation_tokens') or 0
        cache_read = details.get('cached_tokens') or 0

    anthropic_response = MessagesResponse(
        id=message_id,
        model=model,
        content=content,
        stop_reason=convert_stop_reason(choice.get('finish_reason')),
        stop_sequence=None,
        usage=Usage(
            input_tokens=max(prompt_tokens - cache_read - cache_creation, 0),
            output_tokens=completion_tokens,
            cache_creation_input_tokens=cache_creation,
            cache_read_input_tokens=cache_read,
        ),
    )
    # Only dump the new assistant message (and its content blocks), not the
    # surrounding usage/id metadata or the entire raw OpenAI response.
    _dump(
        direction='openai->anthropic (new message)',
        before={'message': message, 'finish_reason': choice.get('finish_reason')},
        after={'content': [c.model_dump(exclude_none=True) for c in content]},
    )
    return anthropic_response


__all__ = [
    'anthropic_to_openai',
    'convert_content_to_openai',
    'convert_document_to_openai',
    'convert_messages_to_openai',
    'convert_openai_content_to_anthropic',
    'convert_stop_reason',
    'convert_system_content_to_openai',
    'convert_system_to_openai',
    'convert_tool_choice_to_openai',
    'convert_tools_to_openai',
    'openai_to_anthropic',
    'set_conversion_log_level',
]
