from __future__ import annotations

import time
import uuid

from typing import Any

from nexus.trace import Event, get_current_span_id, update_span


STREAM_EVENT_CLIENT_ARG_KEYS = {
    'stream',
    'stream_events_include_raw',
    'stream_events_max_delta_chars',
}


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, 'model_dump'):
        dumped = value.model_dump()
        if isinstance(dumped, dict):
            return dumped
    return {}


def _get_nested(data: dict[str, Any], *keys: str) -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _compact_delta(value: Any, *, max_chars: int) -> tuple[Any, dict[str, Any]]:
    if not isinstance(value, str) or max_chars <= 0 or len(value) <= max_chars:
        return value, {}
    return value[:max_chars], {
        'delta_truncated': True,
        'delta_original_length': len(value),
    }


class LLMStreamEmitter:
    def __init__(
        self,
        *,
        provider: str,
        model: str | None,
        client_key: str | None = None,
        stream_id: str | None = None,
        max_delta_chars: int = 4096,
        include_raw: bool = False,
    ) -> None:
        self.provider = provider
        self.model = model
        self.client_key = client_key
        self.stream_id = stream_id or uuid.uuid4().hex
        self.max_delta_chars = max_delta_chars
        self.include_raw = include_raw
        self._sequence = 0

    def emit(self, name: str, **attributes: Any) -> None:
        self._sequence += 1
        attrs: dict[str, Any] = {
            'llm.stream_id': self.stream_id,
            'llm.sequence': self._sequence,
            'llm.provider': self.provider,
        }
        if self.model is not None:
            attrs['llm.model'] = self.model
        if self.client_key is not None:
            attrs['llm.client_key'] = self.client_key

        for key, value in attributes.items():
            if key == 'raw' and not self.include_raw:
                continue
            if key == 'delta':
                value, extra = _compact_delta(value, max_chars=self.max_delta_chars)
                attrs.update(extra)
            attrs[_attribute_key(key)] = value

        span_id = get_current_span_id()
        if span_id is None:
            return
        update_span(
            span_id,
            events=[Event(time_unix_nano=time.time_ns(), name=name, attributes=attrs)],
        )


def _attribute_key(key: str) -> str:
    if key in {'delta', 'response_id', 'finish_reason'} or key.startswith('error.') or key.startswith('usage.'):
        return key
    if key.startswith('llm.'):
        return key
    return f'llm.{key}'


class ChatCompletionStreamEventEmitter:
    def __init__(self, *, emitter: LLMStreamEmitter) -> None:
        self.emitter = emitter
        self.response_id: str | None = None
        self._started = False
        self._tool_calls: dict[tuple[int, int], dict[str, Any]] = {}
        self._done = False

    def consume_chunk(self, chunk: Any) -> None:
        data = _as_dict(chunk)
        if not data:
            return
        self.response_id = data.get('id') or self.response_id
        if not self._started:
            self._started = True
            self.emitter.emit('chat.completion.created', response_id=self.response_id)

        for raw_choice in data.get('choices') or []:
            choice = _as_dict(raw_choice)
            index = int(choice.get('index') or 0)
            delta = _as_dict(choice.get('delta') or {})

            role = delta.get('role')
            if isinstance(role, str):
                self.emitter.emit(
                    'chat.completion.delta',
                    response_id=self.response_id,
                    **{
                        'choice.index': index,
                        'delta.role': role,
                    },
                )

            content = delta.get('content')
            if isinstance(content, str) and content:
                self.emitter.emit(
                    'chat.completion.delta',
                    response_id=self.response_id,
                    **{
                        'choice.index': index,
                        'content.index': 0,
                    },
                    delta=content,
                )

            reasoning_delta = self._reasoning_delta(delta)
            if reasoning_delta:
                self.emitter.emit(
                    'chat.completion.reasoning.delta',
                    response_id=self.response_id,
                    **{
                        'choice.index': index,
                        'content.index': 0,
                    },
                    delta=reasoning_delta,
                )

            self._consume_tool_calls(index, delta)

    def _reasoning_delta(self, delta: dict[str, Any]) -> str | None:
        for key in ('reasoning_content', 'reasoning', 'thinking'):
            value = delta.get(key)
            if isinstance(value, str) and value:
                return value
        return None

    def _consume_tool_calls(self, choice_index: int, delta: dict[str, Any]) -> None:
        tool_calls = delta.get('tool_calls')
        if not isinstance(tool_calls, list):
            return
        for raw_tool_call in tool_calls:
            tool_call = _as_dict(raw_tool_call)
            tool_index = int(tool_call.get('index') or 0)
            key = (choice_index, tool_index)
            state = self._tool_calls.setdefault(
                key,
                {
                    'id': None,
                    'type': 'function',
                    'function': {'name': None, 'arguments': ''},
                },
            )
            if tool_call.get('id'):
                state['id'] = tool_call['id']
            if tool_call.get('type'):
                state['type'] = tool_call['type']
            function = _as_dict(tool_call.get('function') or {})
            if function.get('name'):
                state['function']['name'] = function['name']
            arguments_delta = function.get('arguments')
            if isinstance(arguments_delta, str) and arguments_delta:
                state['function']['arguments'] += arguments_delta
                tool_call_id = state['id'] or f'call_{choice_index}_{tool_index}'
                self.emitter.emit(
                    'chat.completion.tool_call.delta',
                    response_id=self.response_id,
                    call_id=state['id'],
                    **{
                        'choice.index': choice_index,
                        'tool_call.index': tool_index,
                        'tool_call.id': tool_call_id,
                        'tool_call.type': state['type'],
                        'tool_call.function.name': state['function']['name'],
                        'tool_call.function.arguments_delta': arguments_delta,
                    },
                    delta=arguments_delta,
                )

    def finish(self, response_json: dict[str, Any]) -> None:
        if self._done:
            return
        self.emitter.emit('chat.completion.completed', **self._completion_attributes(response_json))
        self._done = True

    def fail(self, error: BaseException) -> None:
        self.emitter.emit(
            'chat.completion.failed',
            **{
                'error.type': type(error).__name__,
                'error.message': str(error),
            },
        )

    def _completion_attributes(self, response_json: dict[str, Any]) -> dict[str, Any]:
        attrs: dict[str, Any] = {'response_id': response_json.get('id') or self.response_id}
        finish_reasons = [
            choice.get('finish_reason')
            for choice in (_as_dict(raw_choice) for raw_choice in response_json.get('choices') or [])
            if choice.get('finish_reason')
        ]
        if finish_reasons:
            attrs['finish_reason'] = finish_reasons[0]
        usage = response_json.get('usage')
        if isinstance(usage, dict):
            attrs['usage.prompt_tokens'] = usage.get('prompt_tokens')
            attrs['usage.completion_tokens'] = usage.get('completion_tokens')
            attrs['usage.total_tokens'] = usage.get('total_tokens')
            reasoning_tokens = _get_nested(usage, 'completion_tokens_details', 'reasoning_tokens')
            if reasoning_tokens is not None:
                attrs['usage.reasoning_output_tokens'] = reasoning_tokens
        return attrs
