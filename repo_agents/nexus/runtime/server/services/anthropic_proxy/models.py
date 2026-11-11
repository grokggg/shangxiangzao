"""Pydantic models for Anthropic Messages API.

These models define the request and response structures for the Anthropic Messages API,
which is used by Claude Code and other Anthropic clients.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


# ============== Content Block Types ==============


class TextContent(BaseModel):
    """Text content block."""

    type: Literal['text'] = 'text'
    text: str


class ImageSource(BaseModel):
    """Image source for base64 encoded images."""

    type: Literal['base64'] = 'base64'
    media_type: str  # e.g., "image/jpeg", "image/png"
    data: str  # base64 encoded image data


class ImageContent(BaseModel):
    """Image content block."""

    type: Literal['image'] = 'image'
    source: ImageSource


class DocumentSource(BaseModel):
    """Source for a document content block.

    Claude Code's Read tool emits base64 PDF sources, e.g.
    ``{"type": "base64", "media_type": "application/pdf", "data": "JVBERi0..."}``.
    Anthropic also allows ``text``/``url``/``content`` document sources, so the
    model only requires the discriminating ``type`` and stays permissive
    (``extra='allow'``) about the rest.
    """

    model_config = ConfigDict(extra='allow')

    type: str  # 'base64' | 'text' | 'url' | 'content'
    media_type: str | None = None  # e.g. "application/pdf", "text/plain"
    data: str | None = None  # base64 encoded document data (type == 'base64')
    text: str | None = None  # inline text (type == 'text')
    url: str | None = None  # remote document url (type == 'url')


class DocumentContent(BaseModel):
    """Document content block (e.g. a PDF read by Claude Code's Read tool).

    Claude Code turns a read PDF into a ``document`` block and sends it back to
    the model (typically inside a ``tool_result``) so the model can inspect the
    file. The proxy accepts it here and converts it to an OpenAI ``file`` part.
    """

    model_config = ConfigDict(extra='allow')

    type: Literal['document'] = 'document'
    source: DocumentSource
    title: str | None = None


class ThinkingContent(BaseModel):
    """Thinking content block (from reasoning models)."""

    type: Literal['thinking'] = 'thinking'
    thinking: str
    signature: str = ''


class RedactedThinkingContent(BaseModel):
    """Redacted thinking content block (bedrock-claude)."""

    type: Literal['redacted_thinking'] = 'redacted_thinking'
    data: str


class ToolUseContent(BaseModel):
    """Tool use content block (in assistant messages)."""

    type: Literal['tool_use'] = 'tool_use'
    id: str
    name: str
    input: dict[str, Any]


class ToolResultContent(BaseModel):
    """Tool result content block (in user messages)."""

    type: Literal['tool_result'] = 'tool_result'
    tool_use_id: str
    content: str | list[TextContent | ImageContent | DocumentContent | ToolReferenceContent]
    is_error: bool = False


class ToolReferenceContent(BaseModel):
    """Tool reference block returned by ToolSearch inside a tool_result.

    Signals that a deferred tool is now available. Claude Code sends the
    referenced tool's full schema in the next request's ``tools`` array,
    so the proxy only needs to pass this through (or render as text).
    """

    type: Literal['tool_reference'] = 'tool_reference'
    tool_name: str


ContentBlock = (
    TextContent
    | ImageContent
    | DocumentContent
    | ThinkingContent
    | RedactedThinkingContent
    | ToolUseContent
    | ToolResultContent
)


# ============== Message Types ==============


class Message(BaseModel):
    """A message in the conversation."""

    role: Literal['user', 'assistant', 'system']
    content: str | list[ContentBlock]


# ============== Tool Definition ==============


class ToolInputSchema(BaseModel):
    """JSON Schema for tool input."""

    type: Literal['object'] = 'object'
    properties: dict[str, Any] = Field(default_factory=dict)
    required: list[str] = Field(default_factory=list)


class Tool(BaseModel):
    """Tool definition for function calling."""

    model_config = ConfigDict(extra='allow')

    name: str
    description: str
    input_schema: ToolInputSchema


ANTHROPIC_SERVER_TOOL_TYPE_PREFIXES = ('web_search_',)


def is_anthropic_server_tool_type(tool_type: Any) -> bool:
    """Return whether a tool type is an Anthropic server-side tool supported by this proxy."""
    return isinstance(tool_type, str) and tool_type.startswith(ANTHROPIC_SERVER_TOOL_TYPE_PREFIXES)


class AnthropicServerTool(BaseModel):
    """Anthropic server-side tool definition.

    Hosted tools such as web_search are executed by Anthropic-compatible
    backends and therefore do not have client-side input schemas.
    """

    model_config = ConfigDict(extra='allow')

    type: str
    name: str

    @model_validator(mode='after')
    def validate_server_tool_type(self) -> AnthropicServerTool:
        if not is_anthropic_server_tool_type(self.type):
            raise ValueError(f'Unsupported Anthropic server-side tool type: {self.type!r}')
        return self


ToolDefinition = AnthropicServerTool | Tool


class ToolChoice(BaseModel):
    """Tool choice configuration."""

    type: Literal['auto', 'any', 'tool']
    name: str | None = None  # Required when type is 'tool'


# ============== System Content ==============


class SystemContent(BaseModel):
    """System content block with optional cache control."""

    type: Literal['text'] = 'text'
    text: str
    cache_control: dict[str, str] | None = None


# ============== Request Model ==============


class MessagesRequest(BaseModel):
    """Request model for POST /v1/messages."""

    model: str
    messages: list[Message]
    max_tokens: int | None = None
    system: str | list[SystemContent] | None = None
    tools: list[ToolDefinition] | None = None
    tool_choice: ToolChoice | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    stop_sequences: list[str] | None = None
    stream: bool = False
    metadata: dict[str, Any] | None = None
    thinking: dict[str, Any] | None = None


# ============== Response Model ==============


class Usage(BaseModel):
    """Token usage statistics."""

    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


class MessagesResponse(BaseModel):
    """Response model for POST /v1/messages."""

    id: str
    type: Literal['message'] = 'message'
    role: Literal['assistant'] = 'assistant'
    model: str
    content: list[TextContent | ThinkingContent | RedactedThinkingContent | ToolUseContent]
    stop_reason: Literal['end_turn', 'max_tokens', 'stop_sequence', 'tool_use', 'refusal'] | None
    stop_sequence: str | None = None
    stop_details: dict[str, Any] | None = None
    usage: Usage


# ============== Thinking Config ==============


KNOWN_THINKING_FIELDS = frozenset({'reasoning_content', 'thinking_blocks'})


class ThinkingConfig(BaseModel):
    """Declares how a backend provider encodes thinking/reasoning content."""

    format: Literal['flat', 'blocks'] = 'flat'
    """``flat``: single string field (hunyuan/DeepSeek).
    ``blocks``: structured array with signature + redacted support (bedrock-claude)."""

    response_field: str = 'reasoning_content'
    """Field name on the OpenAI message dict that carries thinking content."""

    request_field: str = 'reasoning_content'
    """Field name to write thinking content into on outbound OpenAI messages."""

    preserve_signature: bool = False
    """Whether to preserve the ``signature`` field on thinking blocks."""

    support_redacted: bool = False
    """Whether to handle ``redacted_thinking`` blocks."""


THINKING_PRESETS: dict[str, ThinkingConfig] = {
    'hunyuan': ThinkingConfig(
        format='flat',
        response_field='reasoning_content',
        request_field='reasoning_content',
    ),
    'bedrock-claude': ThinkingConfig(
        format='blocks',
        response_field='thinking_blocks',
        request_field='thinking_blocks',
        preserve_signature=True,
        support_redacted=True,
    ),
}

DEFAULT_THINKING_PRESET = 'hunyuan'


def get_thinking_config(preset_or_config: str | dict[str, Any] | ThinkingConfig | None = None) -> ThinkingConfig:
    """Resolve a preset name, raw dict, or ThinkingConfig into a ThinkingConfig."""
    if preset_or_config is None:
        return THINKING_PRESETS[DEFAULT_THINKING_PRESET]
    if isinstance(preset_or_config, ThinkingConfig):
        return preset_or_config
    if isinstance(preset_or_config, str):
        if preset_or_config in THINKING_PRESETS:
            return THINKING_PRESETS[preset_or_config]
        raise ValueError(f'Unknown thinking preset: {preset_or_config!r}. Available: {list(THINKING_PRESETS)}')
    return ThinkingConfig(**preset_or_config)


# ============== Error Response ==============


class ErrorDetail(BaseModel):
    """Error detail."""

    type: str
    message: str


class ErrorResponse(BaseModel):
    """Error response model."""

    type: Literal['error'] = 'error'
    error: ErrorDetail


__all__ = [
    'ContentBlock',
    'DEFAULT_THINKING_PRESET',
    'ANTHROPIC_SERVER_TOOL_TYPE_PREFIXES',
    'AnthropicServerTool',
    'DocumentContent',
    'DocumentSource',
    'ErrorDetail',
    'ErrorResponse',
    'ImageContent',
    'ImageSource',
    'KNOWN_THINKING_FIELDS',
    'Message',
    'MessagesRequest',
    'MessagesResponse',
    'RedactedThinkingContent',
    'SystemContent',
    'TextContent',
    'ThinkingConfig',
    'ThinkingContent',
    'THINKING_PRESETS',
    'Tool',
    'ToolChoice',
    'ToolDefinition',
    'ToolInputSchema',
    'ToolReferenceContent',
    'ToolResultContent',
    'ToolUseContent',
    'Usage',
    'get_thinking_config',
    'is_anthropic_server_tool_type',
]
