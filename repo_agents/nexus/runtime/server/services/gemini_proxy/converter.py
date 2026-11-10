"""Convert between Gemini and OpenAI-style chat payloads."""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import time
import uuid

from collections.abc import AsyncIterator
from typing import Any


_STREAM_TEXT_CHUNK_SIZE = 64


def gemini_request_to_openai(path: str, request_body: dict[str, Any]) -> dict[str, Any]:
    """Convert a Gemini request body into OpenAI chat completion params."""
    openai_params: dict[str, Any] = {
        'messages': _build_messages(request_body),
    }

    model = _extract_model_from_path(path)
    if model:
        openai_params['model'] = model

    tools = _convert_tools(request_body.get('tools'))
    if tools:
        openai_params['tools'] = tools

    generation_config = request_body.get('generationConfig') or request_body.get('generation_config')
    if isinstance(generation_config, dict):
        _copy_if_present(generation_config, openai_params, 'temperature')
        _copy_if_present(generation_config, openai_params, 'topP', target_key='top_p')
        _copy_if_present(generation_config, openai_params, 'topK', target_key='top_k')
        _copy_if_present(generation_config, openai_params, 'candidateCount', target_key='n')
        _copy_if_present(generation_config, openai_params, 'maxOutputTokens', target_key='max_tokens')
        stop_sequences = generation_config.get('stopSequences')
        if isinstance(stop_sequences, list) and stop_sequences:
            openai_params['stop'] = [str(item) for item in stop_sequences]

    tool_config = request_body.get('toolConfig') or request_body.get('tool_config')
    tool_choice = _gemini_tool_config_to_openai(tool_config)
    if tool_choice is not None:
        openai_params['tool_choice'] = tool_choice

    return openai_params


def openai_request_to_gemini(
    *,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    tool_choice: Any = None,
    temperature: Any = None,
    top_p: Any = None,
    top_k: Any = None,
    max_tokens: Any = None,
    stop: Any = None,
    n: Any = None,
) -> dict[str, Any]:
    """Convert OpenAI chat completion params into a Gemini request body."""
    tool_name_by_id: dict[str, str] = {}
    system_parts: list[dict[str, Any]] = []
    contents: list[dict[str, Any]] = []

    for raw_message in messages:
        if not isinstance(raw_message, dict):
            continue

        role = str(raw_message.get('role') or 'user')
        if role == 'system':
            system_parts.extend(_openai_content_to_parts(raw_message.get('content')))
            continue

        if role == 'assistant':
            assistant_parts: list[dict[str, Any]] = []
            reasoning_content = raw_message.get('reasoning_content')
            if reasoning_content:
                assistant_parts.append({'text': str(reasoning_content), 'thought': True})
            assistant_parts.extend(_openai_content_to_parts(raw_message.get('content')))
            for tool_call in raw_message.get('tool_calls') or []:
                if not isinstance(tool_call, dict):
                    continue
                function = tool_call.get('function') or {}
                name = str(function.get('name') or '')
                if not name:
                    continue
                call_id = str(tool_call.get('id') or f'call_{uuid.uuid4().hex}')
                tool_name_by_id[call_id] = name
                assistant_parts.append({
                    'functionCall': {
                        'name': name,
                        'args': _parse_json_or_raw(function.get('arguments')),
                    }
                })
            if assistant_parts:
                contents.append({'role': 'model', 'parts': assistant_parts})
            continue

        if role == 'tool':
            tool_call_id = str(raw_message.get('tool_call_id') or '')
            name = tool_name_by_id.get(tool_call_id) or str(raw_message.get('name') or '')
            function_response: dict[str, Any] = {
                'response': _tool_output_to_gemini(raw_message.get('content')),
            }
            if name:
                function_response['name'] = name
            contents.append({'role': 'user', 'parts': [{'functionResponse': function_response}]})
            continue

        user_parts = _openai_content_to_parts(raw_message.get('content'))
        if user_parts:
            contents.append({'role': 'user', 'parts': user_parts})

    payload: dict[str, Any] = {'contents': contents}
    if system_parts:
        payload['systemInstruction'] = {'parts': system_parts}

    gemini_tools = openai_tools_to_gemini(tools)
    if gemini_tools:
        payload['tools'] = gemini_tools

    tool_config = _openai_tool_choice_to_gemini(tool_choice)
    if tool_config:
        payload['toolConfig'] = tool_config

    generation_config: dict[str, Any] = {}
    _copy_if_present_locals(temperature, generation_config, 'temperature')
    _copy_if_present_locals(top_p, generation_config, 'topP')
    _copy_if_present_locals(top_k, generation_config, 'topK')
    _copy_if_present_locals(max_tokens, generation_config, 'maxOutputTokens')
    _copy_if_present_locals(n, generation_config, 'candidateCount')
    if stop:
        generation_config['stopSequences'] = stop if isinstance(stop, list) else [str(stop)]
    if generation_config:
        payload['generationConfig'] = generation_config

    return payload


def gemini_response_to_openai(response_body: dict[str, Any], model: str | None = None) -> dict[str, Any]:
    """Convert a Gemini response into an OpenAI-style chat completion payload."""
    candidate = ((response_body.get('candidates') or [{}])[:1] or [{}])[0]
    content = ((candidate.get('content') or {}) if isinstance(candidate, dict) else {}) or {}
    parts = content.get('parts') or []

    text_parts: list[str] = []
    thought_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []

    for part in parts:
        if not isinstance(part, dict):
            continue

        # Gemini reasoning models use 'thought' flag on text parts
        if part.get('thought') and isinstance(part.get('text'), str):
            thought_parts.append(part['text'])
            continue

        text = part.get('text')
        if isinstance(text, str):
            text_parts.append(text)
            continue

        function_call = part.get('functionCall') or part.get('function_call')
        if isinstance(function_call, dict):
            tool_calls.append({
                'id': function_call.get('id') or f'call_{uuid.uuid4().hex}',
                'type': 'function',
                'function': {
                    'name': function_call.get('name', ''),
                    'arguments': _ensure_json_string(function_call.get('args') or function_call.get('arguments')),
                },
            })

    finish_reason = _convert_finish_reason(candidate.get('finishReason'))
    usage_metadata = response_body.get('usageMetadata') or {}
    prompt_tokens = int(usage_metadata.get('promptTokenCount') or 0)
    completion_tokens = int(usage_metadata.get('candidatesTokenCount') or 0)
    total_tokens = int(usage_metadata.get('totalTokenCount') or (prompt_tokens + completion_tokens))

    message: dict[str, Any] = {
        'role': 'assistant',
        'content': '\n'.join(text_parts) if text_parts else '',
    }
    if thought_parts:
        message['reasoning_content'] = '\n'.join(thought_parts)
    if tool_calls:
        message['tool_calls'] = tool_calls

    resolved_model = model or response_body.get('modelVersion')

    return {
        'id': response_body.get('responseId') or f'chatcmpl_{uuid.uuid4().hex}',
        'object': 'chat.completion',
        'created': int(time.time()),
        'model': resolved_model,
        'choices': [
            {
                'index': 0,
                'finish_reason': finish_reason,
                'message': message,
            }
        ],
        'usage': {
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens,
            'total_tokens': total_tokens,
        },
    }


def openai_response_to_gemini(response_body: dict[str, Any], model: str | None = None) -> dict[str, Any]:
    """Convert an OpenAI-style chat completion response into Gemini format."""
    choice = ((response_body.get('choices') or [{}])[:1] or [{}])[0]
    message = (choice.get('message') or {}) if isinstance(choice, dict) else {}
    content = message.get('content')
    tool_calls = message.get('tool_calls') or []

    parts: list[dict[str, Any]] = []
    reasoning_content = message.get('reasoning_content')
    if reasoning_content:
        parts.append({'text': str(reasoning_content), 'thought': True})

    if content:
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get('type') == 'text' and isinstance(item.get('text'), str):
                    parts.append({'text': item['text']})
        else:
            parts.append({'text': str(content)})

    for tool_call in tool_calls:
        if not isinstance(tool_call, dict):
            continue
        function = tool_call.get('function') or {}
        name = str(function.get('name') or '')
        if not name:
            continue
        parts.append({
            'functionCall': {
                'id': str(tool_call.get('id') or f'call_{uuid.uuid4().hex}'),
                'name': name,
                'args': _parse_json_or_raw(function.get('arguments')),
            }
        })

    usage = response_body.get('usage') or {}
    prompt_tokens = int(usage.get('prompt_tokens') or 0)
    completion_tokens = int(usage.get('completion_tokens') or 0)
    total_tokens = int(usage.get('total_tokens') or (prompt_tokens + completion_tokens))

    resolved_model = model or response_body.get('model')
    created = int(response_body.get('created') or time.time())
    create_time = dt.datetime.fromtimestamp(created, tz=dt.timezone.utc).isoformat().replace('+00:00', 'Z')

    return {
        'candidates': [
            {
                'content': {
                    'role': 'model',
                    'parts': parts or [{'text': ''}],
                },
                'finishReason': _convert_finish_reason_to_gemini(choice.get('finish_reason')),
            }
        ],
        'createTime': create_time,
        'modelVersion': resolved_model,
        'responseId': response_body.get('id') or f'resp_{uuid.uuid4().hex}',
        'usageMetadata': {
            'promptTokenCount': prompt_tokens,
            'candidatesTokenCount': completion_tokens,
            'totalTokenCount': total_tokens,
        },
    }


async def simulate_gemini_streaming(response_body: dict[str, Any]) -> AsyncIterator[str]:
    """Generate Gemini-style SSE events from a complete response."""
    candidate = ((response_body.get('candidates') or [{}])[:1] or [{}])[0]
    content = (candidate.get('content') or {}) if isinstance(candidate, dict) else {}
    parts = content.get('parts') or []

    emitted = False
    for part in parts:
        if not isinstance(part, dict):
            continue
        if isinstance(part.get('text'), str):
            text = part['text']
            if not text:
                continue
            for index in range(0, len(text), _STREAM_TEXT_CHUNK_SIZE):
                chunk = _gemini_response_chunk(
                    response_body,
                    {'text': text[index : index + _STREAM_TEXT_CHUNK_SIZE]},
                )
                emitted = True
                yield f'data: {json.dumps(chunk, ensure_ascii=False)}\n\n'
                await asyncio.sleep(0)
            continue

        if part.get('functionCall'):
            emitted = True
            yield f'data: {json.dumps(_gemini_response_chunk(response_body, part), ensure_ascii=False)}\n\n'
            await asyncio.sleep(0)

    final_chunk = _gemini_response_chunk(
        response_body,
        None,
        finish_reason=candidate.get('finishReason'),
        usage_metadata=response_body.get('usageMetadata'),
    )
    if emitted:
        yield f'data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n'
    else:
        yield f'data: {json.dumps(response_body, ensure_ascii=False)}\n\n'


def merge_gemini_sse_chunks(chunks: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Merge Gemini SSE chunks into a single response-like payload."""
    if not chunks:
        return None

    merged: dict[str, Any] = {}
    merged_candidate: dict[str, Any] = {'content': {'parts': []}}

    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue

        for key in ('responseId', 'modelVersion', 'createTime'):
            if key in chunk:
                merged[key] = chunk[key]

        if 'usageMetadata' in chunk:
            merged['usageMetadata'] = chunk['usageMetadata']

        candidate = ((chunk.get('candidates') or [{}])[:1] or [{}])[0]
        if not isinstance(candidate, dict):
            continue

        if 'finishReason' in candidate:
            merged_candidate['finishReason'] = candidate['finishReason']

        content = candidate.get('content') or {}
        parts = content.get('parts') or []
        merged_parts = merged_candidate['content']['parts']

        for part in parts:
            if not isinstance(part, dict):
                continue
            text = part.get('text')
            if (
                isinstance(text, str)
                and merged_parts
                and isinstance(merged_parts[-1], dict)
                and set(merged_parts[-1].keys()) == {'text'}
            ):
                merged_parts[-1]['text'] += text
            else:
                merged_parts.append(part)

    merged['candidates'] = [merged_candidate]
    return merged


def _build_messages(request_body: dict[str, Any]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []

    system_instruction = request_body.get('systemInstruction') or request_body.get('system_instruction')
    if isinstance(system_instruction, dict):
        system_text = _parts_to_text(system_instruction.get('parts'))
        if system_text:
            messages.append({'role': 'system', 'content': system_text})

    raw_contents = request_body.get('contents')
    if not isinstance(raw_contents, list):
        return messages

    for content in raw_contents:
        if not isinstance(content, dict):
            continue

        role = str(content.get('role') or 'user')
        parts = content.get('parts') or []
        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        tool_outputs: list[dict[str, Any]] = []

        thought_parts: list[str] = []

        for part in parts:
            if not isinstance(part, dict):
                continue

            # Gemini reasoning models use 'thought' flag on text parts
            if part.get('thought') and isinstance(part.get('text'), str):
                thought_parts.append(part['text'])
                continue

            text = part.get('text')
            if isinstance(text, str):
                text_parts.append(text)
                continue

            function_call = part.get('functionCall') or part.get('function_call')
            if isinstance(function_call, dict):
                tool_calls.append({
                    'id': function_call.get('id') or f'call_{uuid.uuid4().hex}',
                    'type': 'function',
                    'function': {
                        'name': function_call.get('name', ''),
                        'arguments': _ensure_json_string(function_call.get('args') or function_call.get('arguments')),
                    },
                })
                continue

            function_response = part.get('functionResponse') or part.get('function_response')
            if isinstance(function_response, dict):
                tool_outputs.append({
                    'role': 'tool',
                    'tool_call_id': function_response.get('id')
                    or function_response.get('name')
                    or f'call_{uuid.uuid4().hex}',
                    'content': _tool_output_to_text(
                        function_response.get('response') or function_response.get('output')
                    ),
                })

        if role == 'model':
            if text_parts or tool_calls or thought_parts:
                assistant_message: dict[str, Any] = {
                    'role': 'assistant',
                    'content': '\n'.join(text_parts) if text_parts else '',
                }
                if thought_parts:
                    assistant_message['reasoning_content'] = '\n'.join(thought_parts)
                if tool_calls:
                    assistant_message['tool_calls'] = tool_calls
                messages.append(assistant_message)
            continue

        if text_parts:
            messages.append({'role': 'user', 'content': '\n'.join(text_parts)})
        messages.extend(tool_outputs)

    return messages


def openai_tools_to_gemini(raw_tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
    if not isinstance(raw_tools, list):
        return None

    declarations: list[dict[str, Any]] = []
    for tool in raw_tools:
        if not isinstance(tool, dict) or tool.get('type') != 'function':
            continue
        function = tool.get('function') or {}
        name = function.get('name')
        if not name:
            continue
        declarations.append({
            'name': name,
            'description': function.get('description', ''),
            'parameters': _clean_schema(
                function.get('parameters')
                or {
                    'type': 'object',
                    'properties': {},
                    'additionalProperties': False,
                }
            ),
        })

    if not declarations:
        return None
    return [{'functionDeclarations': declarations}]


def _convert_tools(raw_tools: Any) -> list[dict[str, Any]] | None:
    if not isinstance(raw_tools, list):
        return None

    converted_tools: list[dict[str, Any]] = []
    for tool in raw_tools:
        if not isinstance(tool, dict):
            continue

        declarations = tool.get('functionDeclarations') or tool.get('function_declarations')
        if not isinstance(declarations, list):
            continue

        for declaration in declarations:
            if not isinstance(declaration, dict):
                continue
            name = declaration.get('name')
            if not name:
                continue
            converted_tools.append({
                'type': 'function',
                'function': {
                    'name': name,
                    'description': declaration.get('description', ''),
                    'parameters': declaration.get('parameters')
                    or declaration.get('parametersJsonSchema')
                    or {
                        'type': 'object',
                        'properties': {},
                        'additionalProperties': False,
                    },
                },
            })

    return converted_tools or None


def _parts_to_text(parts: Any) -> str:
    if not isinstance(parts, list):
        return ''
    text_parts: list[str] = []
    for part in parts:
        if isinstance(part, dict) and isinstance(part.get('text'), str):
            text_parts.append(part['text'])
    return '\n'.join(text_parts)


def _clean_schema(schema: Any) -> Any:
    if not isinstance(schema, dict):
        return schema

    result: dict[str, Any] = {}
    for key, value in schema.items():
        if key in {'$schema', 'additionalProperties', 'default'}:
            continue
        if isinstance(value, dict):
            result[key] = _clean_schema(value)
            continue
        if isinstance(value, list):
            result[key] = [_clean_schema(item) if isinstance(item, dict) else item for item in value]
            continue
        result[key] = value
    return result


def _openai_content_to_parts(content: Any) -> list[dict[str, Any]]:
    if content is None:
        return []
    if isinstance(content, str):
        return [{'text': content}]
    if not isinstance(content, list):
        return [{'text': str(content)}]

    parts: list[dict[str, Any]] = []
    for item in content:
        if not isinstance(item, dict):
            parts.append({'text': str(item)})
            continue
        if item.get('type') == 'text' and isinstance(item.get('text'), str):
            parts.append({'text': item['text']})
            continue
        image_url = item.get('image_url')
        if item.get('type') == 'image_url' and isinstance(image_url, dict):
            inline_data = _image_url_to_inline_data(image_url.get('url'))
            if inline_data is not None:
                parts.append({'inlineData': inline_data})
    return parts


def _image_url_to_inline_data(url: Any) -> dict[str, Any] | None:
    if not isinstance(url, str) or not url.startswith('data:') or ';base64,' not in url:
        return None
    prefix, encoded = url.split(';base64,', 1)
    mime_type = prefix.removeprefix('data:') or 'application/octet-stream'
    return {
        'mimeType': mime_type,
        'data': encoded,
    }


def _tool_output_to_text(output: Any) -> str:
    if output is None:
        return ''
    if isinstance(output, str):
        return output
    if isinstance(output, list):
        return '\n'.join(_tool_output_to_text(item) for item in output)
    return json.dumps(output, ensure_ascii=False)


def _ensure_json_string(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value or {}, ensure_ascii=False)


def _tool_output_to_gemini(value: Any) -> Any:
    if isinstance(value, str):
        return _parse_json_or_raw(value)
    return value if value is not None else {}


def _parse_json_or_raw(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _extract_model_from_path(path: str) -> str | None:
    marker = '/models/'
    if marker not in path:
        return None
    suffix = path.split(marker, 1)[1]
    model = suffix.split(':', 1)[0]
    return model or None


def _convert_finish_reason(reason: Any) -> str:
    normalized = str(reason or '').upper()
    if normalized in {'STOP', 'MAX_TOKENS', 'SAFETY', 'RECITATION'}:
        return normalized.lower()
    if normalized == 'MALFORMED_FUNCTION_CALL':
        return 'tool_calls'
    return 'tool_calls' if normalized else 'stop'


def _convert_finish_reason_to_gemini(reason: Any) -> str:
    normalized = str(reason or '').lower()
    if normalized == 'length':
        return 'MAX_TOKENS'
    if normalized in {'content_filter', 'safety'}:
        return 'SAFETY'
    return 'STOP'


def _copy_if_present(
    source: dict[str, Any], target: dict[str, Any], source_key: str, target_key: str | None = None
) -> None:
    value = source.get(source_key)
    if value is not None:
        target[target_key or source_key] = value


def _copy_if_present_locals(value: Any, target: dict[str, Any], target_key: str) -> None:
    if value is not None:
        target[target_key] = value


def _gemini_tool_config_to_openai(tool_config: Any) -> Any:
    if not isinstance(tool_config, dict):
        return None
    function_config = tool_config.get('functionCallingConfig') or tool_config.get('function_calling_config')
    if not isinstance(function_config, dict):
        return None
    mode = str(function_config.get('mode') or '').upper()
    if mode == 'NONE':
        return 'none'
    if mode == 'ANY':
        allowed_names = function_config.get('allowedFunctionNames') or function_config.get('allowed_function_names')
        if isinstance(allowed_names, list) and len(allowed_names) == 1 and allowed_names[0]:
            return {'type': 'function', 'function': {'name': str(allowed_names[0])}}
        return 'required'
    if mode == 'AUTO':
        return 'auto'
    return None


def _openai_tool_choice_to_gemini(tool_choice: Any) -> dict[str, Any] | None:
    if tool_choice in (None, 'auto'):
        return None
    if tool_choice == 'none':
        return {'functionCallingConfig': {'mode': 'NONE'}}
    if tool_choice in {'required', 'any'}:
        return {'functionCallingConfig': {'mode': 'ANY'}}
    if isinstance(tool_choice, dict):
        function = tool_choice.get('function') or {}
        name = function.get('name')
        if name:
            return {
                'functionCallingConfig': {
                    'mode': 'ANY',
                    'allowedFunctionNames': [str(name)],
                }
            }
    return None


def _gemini_response_chunk(
    response_body: dict[str, Any],
    part: dict[str, Any] | None,
    *,
    finish_reason: Any = None,
    usage_metadata: Any = None,
) -> dict[str, Any]:
    chunk: dict[str, Any] = {
        'candidates': [
            {
                'content': {
                    'role': 'model',
                    'parts': [part] if part is not None else [],
                }
            }
        ],
        'responseId': response_body.get('responseId') or f'resp_{uuid.uuid4().hex}',
        'modelVersion': response_body.get('modelVersion'),
    }
    if response_body.get('createTime'):
        chunk['createTime'] = response_body['createTime']
    if finish_reason is not None:
        chunk['candidates'][0]['finishReason'] = finish_reason
    if usage_metadata is not None:
        chunk['usageMetadata'] = usage_metadata
    return chunk
