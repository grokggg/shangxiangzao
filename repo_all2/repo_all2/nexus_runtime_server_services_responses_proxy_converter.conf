"""Convert between Codex Responses API payloads and OpenAI chat completions."""

from __future__ import annotations

import json
import logging
import uuid

from typing import Any


logger = logging.getLogger(__name__)


def responses_to_openai(
    request_body: dict[str, Any],
    *,
    collect_custom_tools: set[str] | None = None,
) -> dict[str, Any]:
    """Convert a Responses API request into OpenAI chat completion parameters.

    ``collect_custom_tools``: optional output set. When provided, the names of tools whose
    Responses type is ``custom`` (e.g. Codex's freeform ``apply_patch``) are added to it, so
    the caller can pass the same set to ``openai_to_responses_events`` and have the matching
    function_call restored as a ``custom_tool_call`` on the return trip. Defaults to None →
    behavior identical to before (no collection).
    """
    openai_params: dict[str, Any] = {
        'messages': _build_messages(request_body),
    }

    if request_body.get('model') is not None:
        openai_params['model'] = request_body['model']

    tools = _convert_tools(request_body.get('tools'), collect_custom_tools=collect_custom_tools)
    if tools:
        openai_params['tools'] = tools

    if request_body.get('tool_choice') is not None:
        openai_params['tool_choice'] = request_body['tool_choice']

    if request_body.get('parallel_tool_calls') is not None:
        openai_params['parallel_tool_calls'] = request_body['parallel_tool_calls']

    return openai_params


def openai_to_responses_events(
    openai_response: dict[str, Any],
    *,
    custom_tool_names: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Convert an OpenAI chat completion response into Responses API SSE events.

    ``custom_tool_names``: names of tools that were originally Responses ``custom`` tools (e.g.
    Codex freeform ``apply_patch``), collected by ``responses_to_openai``. A tool_call whose
    name is in this set is emitted as a ``custom_tool_call`` (the form Codex >=0.136 requires;
    its apply_patch handler only accepts ToolPayload::Custom) instead of a ``function_call``.
    Defaults to None → every tool_call becomes a function_call, identical to before.
    """
    response_id = openai_response.get('id') or f'resp_{uuid.uuid4().hex}'
    events: list[dict[str, Any]] = [
        {
            'type': 'response.created',
            'response': {
                'id': response_id,
            },
        }
    ]

    choices = openai_response.get('choices') or []
    for choice in choices:
        message = choice.get('message') or {}

        reasoning_items = message.get('reasoning_items') or []
        emitted_reasoning = False
        if isinstance(reasoning_items, list):
            for reasoning_item in reasoning_items:
                item = _normalize_reasoning_item(reasoning_item)
                if item is None:
                    continue
                events.append({
                    'type': 'response.output_item.done',
                    'item': item,
                })
                emitted_reasoning = True

        reasoning_content = message.get('reasoning_content')
        if reasoning_content and not emitted_reasoning:
            events.append({
                'type': 'response.output_item.done',
                'item': _reasoning_item_from_text(str(reasoning_content)),
            })

        text = _assistant_content_to_text(message.get('content'))
        if text:
            events.append({
                'type': 'response.output_item.done',
                'item': {
                    'type': 'message',
                    'role': 'assistant',
                    'id': f'msg_{uuid.uuid4().hex}',
                    'content': [
                        {
                            'type': 'output_text',
                            'text': text,
                        }
                    ],
                },
            })

        tool_calls = message.get('tool_calls') or []
        for tool_call in tool_calls:
            function = tool_call.get('function') or {}
            name = function.get('name', '')
            call_id = tool_call.get('id') or f'call_{uuid.uuid4().hex}'
            if custom_tool_names and name in custom_tool_names:
                # Original tool was a Responses custom tool (e.g. apply_patch). Restore it as a
                # custom_tool_call; Codex >=0.136 routes only custom_tool_call to the apply_patch
                # handler (function_call would raise "apply_patch handler received unsupported
                # payload"). _extract_custom_tool_input recovers the raw input from the function
                # arguments we downgraded on the request side.
                events.append({
                    'type': 'response.output_item.done',
                    'item': {
                        'type': 'custom_tool_call',
                        'call_id': call_id,
                        'name': name,
                        'input': _extract_custom_tool_input(function.get('arguments')),
                    },
                })
            else:
                if custom_tool_names and len(custom_tool_names) == 1 and len(tool_calls) == 1:
                    # The request advertised exactly one custom tool but this call's name did not
                    # match (host may have namespaced/renamed it). Emitting function_call here
                    # silently reintroduces the 0.136 "unsupported payload" crash, so warn.
                    only_custom = next(iter(custom_tool_names))
                    logger.warning(
                        'responses_proxy: tool_call name %r did not match the sole custom tool %r; '
                        'emitting function_call may break Codex >=0.136 apply_patch',
                        name,
                        only_custom,
                    )
                events.append({
                    'type': 'response.output_item.done',
                    'item': {
                        'type': 'function_call',
                        'call_id': call_id,
                        'name': name,
                        'arguments': _ensure_json_string(function.get('arguments')),
                    },
                })

    usage = _convert_usage(openai_response.get('usage'))
    events.append({
        'type': 'response.completed',
        'response': {
            'id': response_id,
            'usage': usage,
        },
    })

    return events


def _build_messages(request_body: dict[str, Any]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []

    instructions = request_body.get('instructions')
    if isinstance(instructions, str) and instructions.strip():
        messages.append({'role': 'system', 'content': instructions})

    raw_input = request_body.get('input')
    if isinstance(raw_input, str):
        messages.append({'role': 'user', 'content': raw_input})
        return messages

    if not isinstance(raw_input, list):
        return messages

    pending_assistant_message: dict[str, Any] | None = None
    pending_tool_calls: list[dict[str, Any]] = []
    pending_reasoning_items: list[dict[str, Any]] = []

    def flush_pending_assistant() -> None:
        nonlocal pending_assistant_message, pending_tool_calls, pending_reasoning_items
        if pending_assistant_message is not None:
            message = dict(pending_assistant_message)
            if pending_reasoning_items:
                message['reasoning_items'] = pending_reasoning_items
                _attach_reasoning_content(message, pending_reasoning_items)
            if pending_tool_calls:
                message['tool_calls'] = pending_tool_calls
            messages.append(message)
            pending_assistant_message = None
            pending_tool_calls = []
            pending_reasoning_items = []
            return
        if pending_tool_calls or pending_reasoning_items:
            message = {
                'role': 'assistant',
                'content': '',
            }
            if pending_reasoning_items:
                message['reasoning_items'] = pending_reasoning_items
                _attach_reasoning_content(message, pending_reasoning_items)
            if pending_tool_calls:
                message['tool_calls'] = pending_tool_calls
            messages.append(message)
            pending_tool_calls = []
            pending_reasoning_items = []

    for item in raw_input:
        if not isinstance(item, dict):
            continue

        item_type = item.get('type')
        if item_type == 'reasoning':
            reasoning_item = _normalize_reasoning_item(item)
            if reasoning_item is not None:
                pending_reasoning_items.append(reasoning_item)
            continue

        if item_type in {'function_call', 'custom_tool_call'}:
            pending_tool_calls.append(_convert_tool_call_item(item))
            continue

        if item_type == 'message':
            message = _convert_message_item(item)
            if message is not None:
                if message.get('role') == 'assistant':
                    if pending_assistant_message is not None or pending_tool_calls:
                        flush_pending_assistant()
                    pending_assistant_message = message
                else:
                    flush_pending_assistant()
                    messages.append(message)
        elif item_type in {'function_call_output', 'custom_tool_call_output'}:
            flush_pending_assistant()
            messages.append(_convert_tool_output_item(item))
        elif item_type in {'tool_search_call', 'tool_search_output'}:
            # tool_search is dropped on the request side (see _convert_tools); drop its replayed
            # history items loudly too, rather than silently skipping them.
            # TODO(tool_search): when full tool_search support lands (see _convert_tools TODO),
            # these items must round-trip instead of being dropped.
            logger.warning('responses_proxy: dropping input item type=%s', item_type)
        elif item_type is not None:
            logger.warning('responses_proxy: dropping unknown input item type=%s', item_type)

    flush_pending_assistant()
    return messages


def _convert_message_item(item: dict[str, Any]) -> dict[str, Any] | None:
    role = item.get('role', 'user')
    if role == 'developer':
        role = 'system'

    content = _convert_content(item.get('content'))
    if content is None:
        return None

    return {
        'role': role,
        'content': content,
    }


def _convert_content(content: Any) -> str | list[dict[str, Any]] | None:
    if isinstance(content, str):
        return content

    if not isinstance(content, list):
        return ''

    has_images = any(isinstance(part, dict) and part.get('type') == 'input_image' for part in content)
    if has_images:
        parts: list[dict[str, Any]] = []
        for part in content:
            if not isinstance(part, dict):
                continue
            part_type = part.get('type')
            if part_type in {'input_text', 'output_text', 'text'}:
                parts.append({'type': 'text', 'text': part.get('text', '')})
            elif part_type == 'input_image':
                parts.append({'type': 'image_url', 'image_url': {'url': part.get('image_url', '')}})
        return parts

    text_parts = []
    for part in content:
        if not isinstance(part, dict):
            continue
        if part.get('type') in {'input_text', 'output_text', 'text'}:
            text_parts.append(part.get('text', ''))

    return '\n\n'.join(text_parts)


def _convert_tool_call_item(item: dict[str, Any]) -> dict[str, Any]:
    item_type = item.get('type')
    if item_type == 'custom_tool_call':
        arguments = json.dumps({'input': item.get('input', '')}, ensure_ascii=False)
    else:
        arguments = _ensure_json_string(item.get('arguments'))

    return {
        'id': item.get('call_id') or f'call_{uuid.uuid4().hex}',
        'type': 'function',
        'function': {
            'name': item.get('name', ''),
            'arguments': arguments,
        },
    }


def _convert_tool_output_item(item: dict[str, Any]) -> dict[str, Any]:
    return {
        'role': 'tool',
        'tool_call_id': item.get('call_id', ''),
        'content': _tool_output_to_text(item.get('output')),
    }


def _tool_output_to_text(output: Any) -> str:
    if output is None:
        return ''
    if isinstance(output, str):
        return output
    if isinstance(output, list):
        text_parts = []
        for part in output:
            if isinstance(part, dict):
                if 'text' in part:
                    text_parts.append(str(part.get('text', '')))
                else:
                    text_parts.append(json.dumps(part, ensure_ascii=False))
            else:
                text_parts.append(str(part))
        return '\n'.join(text_parts)
    return json.dumps(output, ensure_ascii=False)


def _convert_tools(
    tools: Any,
    *,
    collect_custom_tools: set[str] | None = None,
) -> list[dict[str, Any]] | None:
    if not isinstance(tools, list):
        return None

    converted_tools = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue

        tool_type = tool.get('type')
        if tool_type == 'function':
            name = tool.get('name')
            if not name:
                continue
            converted_tools.append({
                'type': 'function',
                'function': {
                    'name': name,
                    'description': tool.get('description', ''),
                    'parameters': tool.get('parameters')
                    or {
                        'type': 'object',
                        'properties': {},
                        'additionalProperties': False,
                    },
                },
            })
        elif tool_type == 'custom':
            name = tool.get('name')
            if not name:
                continue
            # Chat completions has no freeform/custom tool. Downgrade to a function with a single
            # raw `input` string. Record the name so openai_to_responses_events can restore the
            # response-side function_call back into a custom_tool_call.
            if collect_custom_tools is not None:
                collect_custom_tools.add(name)
            converted_tools.append({
                'type': 'function',
                'function': {
                    'name': name,
                    'description': tool.get('description', ''),
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'input': {
                                'type': 'string',
                                'description': 'Raw tool input.',
                            }
                        },
                        'required': ['input'],
                        'additionalProperties': False,
                    },
                },
            })
        elif tool_type == 'tool_search':
            # tool_search is a client-executed Responses tool: Codex runs BM25 search locally over
            # "deferred" tools and replays tool_search_call/tool_search_output itself. We drop it
            # loudly (not silently) here; the model just won't see this tool.
            # TODO(tool_search): support an `enable_tool_search` switch so models that handle it
            # (e.g. GPT-5.5) can use it while others (e.g. HY3) stay off. This is NOT a simple
            # passthrough — full support needs: (1) restoring the response-side function_call into
            # a `tool_search_call` (ResponseItem::ToolSearchCall has a different shape than a
            # function call), (2) a deferred-tool registry on the proxy side, and (3) round-tripping
            # tool_search_output (the matched tools) back into the next turn's tools. Tracked in
            # repo_qa/nexus/commit_2_codex_tool_protocol_plan.md §2.4.
            logger.warning('responses_proxy: dropping unsupported tool type=tool_search (name=%s)', tool.get('name'))
        else:
            logger.warning('responses_proxy: dropping unknown tool type=%s', tool_type)

    return converted_tools or None


def _summary_from_text(text: str) -> list[dict[str, str]]:
    if not text:
        return []
    return [{'type': 'summary_text', 'text': text}]


def _attach_reasoning_content(message: dict[str, Any], reasoning_items: list[dict[str, Any]]) -> None:
    """Mirror replayed reasoning_items into the assistant message's reasoning_content.

    HY-style chat models continue their chain-of-thought across turns only when the prior turn's
    reasoning is replayed in the ``reasoning_content`` field (a plain string) alongside ``content``;
    they never read the Responses ``reasoning_items`` structure. We keep ``reasoning_items`` intact
    (GPT-5.5 / Responses models continue via its encrypted_content + item ids) and ADD a flattened
    ``reasoning_content`` next to it -- each item's ``summary[].text`` (falling back to
    ``content[].text``). litellm's Responses path simply drops this extra field, so GPT-5.5 is
    unaffected (verified on the wire: reasoning_items count and encrypted_content are unchanged),
    while chat models pick it up. Only set when there is textual reasoning and the message does not
    already carry reasoning_content (opaque encrypted_content-only items yield nothing)."""
    if message.get('reasoning_content'):
        return
    parts: list[str] = []
    for item in reasoning_items:
        if not isinstance(item, dict):
            continue
        chunks: list[str] = []
        for key in ('summary', 'content'):
            section = item.get(key)
            if not isinstance(section, list):
                continue
            for entry in section:
                if isinstance(entry, dict):
                    text = entry.get('text')
                    if isinstance(text, str) and text:
                        chunks.append(text)
            if chunks:
                # Prefer summary; only fall back to content when summary had no text.
                break
        if chunks:
            parts.append('\n'.join(chunks))
    if parts:
        message['reasoning_content'] = '\n'.join(parts)


def _reasoning_item_from_text(text: str) -> dict[str, Any]:
    return {
        'type': 'reasoning',
        'id': f'rs_{uuid.uuid4().hex}',
        'summary': _summary_from_text(text),
    }


def _normalize_reasoning_item(item: Any) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None

    normalized: dict[str, Any] = {
        'type': 'reasoning',
        'id': item.get('id') or f'rs_{uuid.uuid4().hex}',
    }

    summary = item.get('summary')
    if isinstance(summary, list):
        normalized['summary'] = summary
    elif isinstance(summary, str):
        normalized['summary'] = _summary_from_text(summary)
    else:
        normalized['summary'] = []

    content = item.get('content')
    if isinstance(content, list):
        normalized['content'] = content

    encrypted_content = item.get('encrypted_content')
    if encrypted_content:
        normalized['encrypted_content'] = encrypted_content

    status = item.get('status')
    if status:
        normalized['status'] = status

    return normalized


def _assistant_content_to_text(content: Any) -> str:
    if content is None:
        return ''
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, dict):
                if part.get('type') in {'text', 'output_text'}:
                    text_parts.append(part.get('text', ''))
                elif 'text' in part:
                    text_parts.append(str(part.get('text', '')))
        return '\n'.join(text_parts)
    return str(content)


def _convert_usage(usage: Any) -> dict[str, Any]:
    if not isinstance(usage, dict):
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
    else:
        prompt_tokens = int(usage.get('prompt_tokens') or 0)
        completion_tokens = int(usage.get('completion_tokens') or 0)
        total_tokens = int(usage.get('total_tokens') or (prompt_tokens + completion_tokens))

    return {
        'input_tokens': prompt_tokens,
        'input_tokens_details': None,
        'output_tokens': completion_tokens,
        'output_tokens_details': None,
        'total_tokens': total_tokens,
    }


def _ensure_json_string(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value if value is not None else {}, ensure_ascii=False)


def _extract_custom_tool_input(arguments: Any) -> str:
    """Recover the raw input string for a restored custom_tool_call.

    On the request side a custom tool is downgraded to a function with a single ``input`` string
    param (see _convert_tools). The model usually replies with arguments ``{"input": "..."}``, but
    it may not follow that shape, so handle each case without ever turning a non-patch value into
    a string that codex would try to parse as a patch.
    """
    if arguments is None:
        return ''  # do NOT json.dumps(None) -> "null"; codex would treat it as a patch
    if not isinstance(arguments, str):
        if isinstance(arguments, dict) and isinstance(arguments.get('input'), str):
            return arguments['input']
        return json.dumps(arguments, ensure_ascii=False)
    # arguments is a str: either {"input": "..."} JSON, or a bare patch body.
    try:
        obj = json.loads(arguments)
    except (ValueError, TypeError):
        return arguments  # bare patch (not JSON) -> pass through verbatim
    if isinstance(obj, dict):
        if isinstance(obj.get('input'), str):
            return obj['input']  # normal: {"input": "*** Begin Patch..."}
        # dict but no "input" key (model used a different schema, e.g. {"patch": ...}).
        # Do not feed the whole JSON as a patch; prefer the sole string value, else warn.
        str_vals = [v for v in obj.values() if isinstance(v, str)]
        if len(str_vals) == 1:
            logger.warning('responses_proxy: custom tool arguments missing "input", using sole string field')
            return str_vals[0]
        logger.warning('responses_proxy: custom tool arguments has no usable input field: %s', list(obj))
        return arguments  # last resort (may still fail downstream, but logged)
    return arguments  # JSON but not a dict (rare) -> verbatim


__all__ = [
    'openai_to_responses_events',
    'responses_to_openai',
]
