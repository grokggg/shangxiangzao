"""Simulated Anthropic SSE streaming.

Converts a complete MessagesResponse into an Anthropic-compatible SSE event stream.
This enables streaming clients (like Claude Code) to work with non-streaming backends
by generating the expected Server-Sent Events protocol from a complete response.

Event sequence per message:
    message_start -> [content_block_start -> ping -> content_block_delta(s) -> content_block_stop]* ->
    message_delta -> message_stop -> [DONE]
"""

from __future__ import annotations

import asyncio
import json

from collections.abc import AsyncIterator, Mapping
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from .models import MessagesResponse


CHUNK_SIZE = 20


def _sse_event(event: str, data: dict | str) -> str:
    """Format a single SSE event.

    Args:
        event: Event type name
        data: Event payload (dict will be JSON-serialized)

    Returns:
        Formatted SSE event string
    """
    payload = json.dumps(data, ensure_ascii=False) if isinstance(data, dict) else data
    return f'event: {event}\ndata: {payload}\n\n'


async def simulate_streaming(response: MessagesResponse) -> AsyncIterator[str]:
    """Generate Anthropic SSE events from a complete MessagesResponse.

    Splits text content into small chunks and tool_use JSON arguments into chunks
    to simulate incremental streaming output.

    Args:
        response: Complete Anthropic MessagesResponse

    Yields:
        SSE-formatted event strings
    """
    # message_start: initial metadata with input_tokens only
    yield _sse_event(
        'message_start',
        {
            'type': 'message_start',
            'message': {
                'id': response.id,
                'type': 'message',
                'role': 'assistant',
                'content': [],
                'model': response.model,
                'stop_reason': None,
                'stop_sequence': None,
                'usage': {
                    'input_tokens': response.usage.input_tokens,
                    'output_tokens': 0,
                    'cache_creation_input_tokens': response.usage.cache_creation_input_tokens,
                    'cache_read_input_tokens': response.usage.cache_read_input_tokens,
                },
            },
        },
    )

    for index, block in enumerate(response.content):
        if block.type == 'redacted_thinking':
            yield _sse_event(
                'content_block_start',
                {
                    'type': 'content_block_start',
                    'index': index,
                    'content_block': {'type': 'redacted_thinking', 'data': block.data},  # type: ignore[union-attr]
                },
            )
            yield _sse_event('ping', {'type': 'ping'})

        elif block.type == 'thinking':
            yield _sse_event(
                'content_block_start',
                {
                    'type': 'content_block_start',
                    'index': index,
                    'content_block': {'type': 'thinking', 'thinking': '', 'signature': ''},
                },
            )
            yield _sse_event('ping', {'type': 'ping'})

            thinking: str = block.thinking  # type: ignore[union-attr]
            for i in range(0, max(len(thinking), 1), CHUNK_SIZE):
                chunk = thinking[i : i + CHUNK_SIZE]
                if chunk:
                    yield _sse_event(
                        'content_block_delta',
                        {
                            'type': 'content_block_delta',
                            'index': index,
                            'delta': {'type': 'thinking_delta', 'thinking': chunk},
                        },
                    )
                    await asyncio.sleep(0)

            # Emit signature if present
            signature = getattr(block, 'signature', None)
            if signature:
                yield _sse_event(
                    'content_block_delta',
                    {
                        'type': 'content_block_delta',
                        'index': index,
                        'delta': {'type': 'signature_delta', 'signature': signature},
                    },
                )

        elif block.type == 'text':
            yield _sse_event(
                'content_block_start',
                {
                    'type': 'content_block_start',
                    'index': index,
                    'content_block': {'type': 'text', 'text': ''},
                },
            )
            yield _sse_event('ping', {'type': 'ping'})

            text: str = block.text  # type: ignore[union-attr]
            for i in range(0, max(len(text), 1), CHUNK_SIZE):
                chunk = text[i : i + CHUNK_SIZE]
                if chunk:
                    yield _sse_event(
                        'content_block_delta',
                        {
                            'type': 'content_block_delta',
                            'index': index,
                            'delta': {'type': 'text_delta', 'text': chunk},
                        },
                    )
                    await asyncio.sleep(0)

        elif block.type == 'tool_use':
            yield _sse_event(
                'content_block_start',
                {
                    'type': 'content_block_start',
                    'index': index,
                    'content_block': {
                        'type': 'tool_use',
                        'id': block.id,  # type: ignore[union-attr]
                        'name': block.name,  # type: ignore[union-attr]
                        'input': {},
                    },
                },
            )
            yield _sse_event('ping', {'type': 'ping'})

            args_json = json.dumps(block.input, ensure_ascii=False)  # type: ignore[union-attr]
            for i in range(0, max(len(args_json), 1), CHUNK_SIZE):
                chunk = args_json[i : i + CHUNK_SIZE]
                if chunk:
                    yield _sse_event(
                        'content_block_delta',
                        {
                            'type': 'content_block_delta',
                            'index': index,
                            'delta': {'type': 'input_json_delta', 'partial_json': chunk},
                        },
                    )
                    await asyncio.sleep(0)

        # content_block_stop for every block
        yield _sse_event(
            'content_block_stop',
            {
                'type': 'content_block_stop',
                'index': index,
            },
        )

    # message_delta: final stop_reason and output_tokens
    message_delta = {
        'stop_reason': response.stop_reason,
        'stop_sequence': response.stop_sequence,
    }
    if response.stop_details is not None:
        message_delta['stop_details'] = response.stop_details

    yield _sse_event(
        'message_delta',
        {
            'type': 'message_delta',
            'delta': message_delta,
            'usage': {
                'output_tokens': response.usage.output_tokens,
            },
        },
    )

    # message_stop
    yield _sse_event('message_stop', {'type': 'message_stop'})

    # Stream terminator
    yield 'data: [DONE]\n\n'


async def simulate_native_streaming(response: Mapping[str, object]) -> AsyncIterator[str]:
    """Encode a provider-owned Anthropic response without narrowing it through local models."""

    raw_usage = response.get('usage')
    usage = dict(raw_usage) if isinstance(raw_usage, Mapping) else {}
    start_usage = dict(usage)
    start_usage.setdefault('input_tokens', 0)
    start_usage['output_tokens'] = 0
    start_usage.setdefault('cache_creation_input_tokens', 0)
    start_usage.setdefault('cache_read_input_tokens', 0)

    start_message = dict(response)
    start_message['content'] = []
    start_message['stop_reason'] = None
    start_message['stop_sequence'] = None
    start_message.pop('stop_details', None)
    start_message['usage'] = start_usage
    yield _sse_event('message_start', {'type': 'message_start', 'message': start_message})

    raw_content = response.get('content')
    content = raw_content if isinstance(raw_content, list) else []
    for index, raw_block in enumerate(content):
        block: dict[str, object] = (
            {str(key): value for key, value in raw_block.items()}
            if isinstance(raw_block, Mapping)
            else {'type': 'unknown', 'value': raw_block}
        )
        block_type = block.get('type')
        start_block = dict(block)
        deltas: list[dict[str, object]] = []

        text = block.get('text')
        thinking = block.get('thinking')
        tool_input = block.get('input')
        if block_type == 'text' and isinstance(text, str):
            start_block['text'] = ''
            deltas = [{'type': 'text_delta', 'text': text[i : i + CHUNK_SIZE]} for i in range(0, len(text), CHUNK_SIZE)]
        elif block_type == 'thinking' and isinstance(thinking, str):
            signature = block.get('signature')
            start_block['thinking'] = ''
            start_block['signature'] = ''
            deltas = [
                {'type': 'thinking_delta', 'thinking': thinking[i : i + CHUNK_SIZE]}
                for i in range(0, len(thinking), CHUNK_SIZE)
            ]
            if isinstance(signature, str) and signature:
                deltas.append({'type': 'signature_delta', 'signature': signature})
        elif block_type == 'tool_use' and isinstance(tool_input, Mapping):
            arguments = json.dumps(tool_input, ensure_ascii=False)
            start_block['input'] = {}
            deltas = [
                {'type': 'input_json_delta', 'partial_json': arguments[i : i + CHUNK_SIZE]}
                for i in range(0, len(arguments), CHUNK_SIZE)
            ]

        yield _sse_event(
            'content_block_start',
            {'type': 'content_block_start', 'index': index, 'content_block': start_block},
        )
        yield _sse_event('ping', {'type': 'ping'})
        for delta in deltas:
            yield _sse_event(
                'content_block_delta',
                {'type': 'content_block_delta', 'index': index, 'delta': delta},
            )
            await asyncio.sleep(0)
        yield _sse_event('content_block_stop', {'type': 'content_block_stop', 'index': index})

    message_delta = {
        'stop_reason': response.get('stop_reason'),
        'stop_sequence': response.get('stop_sequence'),
    }
    if 'stop_details' in response:
        message_delta['stop_details'] = response['stop_details']

    known_usage_fields = {
        'input_tokens',
        'output_tokens',
        'cache_creation_input_tokens',
        'cache_read_input_tokens',
    }
    delta_usage = {'output_tokens': usage.get('output_tokens', 0)}
    delta_usage.update({key: value for key, value in usage.items() if key not in known_usage_fields})
    yield _sse_event(
        'message_delta',
        {'type': 'message_delta', 'delta': message_delta, 'usage': delta_usage},
    )
    yield _sse_event('message_stop', {'type': 'message_stop'})
    yield 'data: [DONE]\n\n'


__all__ = [
    'simulate_native_streaming',
    'simulate_streaming',
]
