"""InferenceBackend protocol and implementations.

All backends implement a unified `create_completion()` method returning `BackendResult`.
- Token-level backends (HyTunerServerBackend, VllmCompletionsBackend) accept
  `prompt_ids` and return token_ids/log_probs/routed_experts.
- Message-level backend (OpenAIAPIBackend) accepts `messages` and returns message/usage.
"""

import asyncio
import json

from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable
from urllib.parse import quote

import httpx

from nexus.gateway.types import BackendResult


_TOKEN_PREFIX = 'token_id:'
VLLM_SAMPLING_ARGS = frozenset({
    'bad_words',
    'best_of',
    'detokenize',
    'early_stopping',
    'frequency_penalty',
    'guided_choice',
    'guided_decoding',
    'guided_json',
    'guided_regex',
    'ignore_eos',
    'include_stop_str_in_output',
    'length_penalty',
    'logprobs',
    'max_tokens',
    'min_p',
    'min_tokens',
    'n',
    'presence_penalty',
    'prompt_logprobs',
    'repetition_penalty',
    'seed',
    'skip_special_tokens',
    'spaces_between_special_tokens',
    'stop',
    'stop_token_ids',
    'temperature',
    'top_k',
    'top_p',
    'truncate_prompt_tokens',
    'use_beam_search',
})


def _parse_token_id(token: str) -> int:
    if token.startswith(_TOKEN_PREFIX):
        return int(token[len(_TOKEN_PREFIX) :])
    raise ValueError(f'expected "token_id:<int>" (return_tokens_as_token_ids), got {token!r}')


def _vllm_sampling_params(request_kwargs: dict[str, Any] | None) -> dict[str, Any]:
    """Project a resolved request onto the token backend contract."""
    return {
        key: value for key, value in (request_kwargs or {}).items() if key in VLLM_SAMPLING_ARGS and value is not None
    }


def _backend_request_id(request_id: str, request_meta: dict[str, Any]) -> str:
    """Derive one physical rollout request ID from declared agent execution fields."""
    declared_chain = request_meta.get('declared_chain')
    declared_chain_kind = declared_chain.get('kind') if isinstance(declared_chain, Mapping) else None
    if not isinstance(declared_chain_kind, str) or declared_chain_kind not in {'main', 'subagent'}:
        return request_id

    agent_id = request_meta.get('agent_id')
    iteration = request_meta.get('iteration')
    retry = request_meta.get('retry')
    if not isinstance(agent_id, str) or not agent_id:
        raise ValueError('declared agent request metadata requires a non-empty string agent_id')
    if type(iteration) is not int or iteration < 0:
        raise ValueError('declared agent request metadata requires a non-negative integer iteration')
    if type(retry) is not int or retry < 0:
        raise ValueError('declared agent request metadata requires a non-negative integer retry')
    invocation_kind = request_meta.get('invocation_kind', 'completion')
    if not isinstance(invocation_kind, str) or not invocation_kind:
        raise ValueError('declared agent request metadata requires a non-empty string invocation_kind')

    return ':'.join((
        quote(request_id, safe='-._~'),
        quote(agent_id, safe='-._~'),
        str(iteration),
        str(retry),
        quote(invocation_kind, safe='-._~'),
    ))


@runtime_checkable
class InferenceBackend(Protocol):
    """Execute a resolved request in one backend-specific representation.

    ``request_meta`` is a caller-owned dict forwarded end to end
    (client -> session -> processor -> backend) without gateway interpretation —
    integrations ride it for per-request context (e.g. HyTuner's ``validate``
    flag selecting the train/val generation channel).
    """

    async def create_completion(
        self,
        request_id: str,
        *,
        prompt_ids: list[int] | None = None,
        messages: list[dict] | None = None,
        request_kwargs: dict[str, Any] | None = None,
        image_data: Any | None = None,
        video_data: Any | None = None,
        audio_data: Any | None = None,
        priority: int = 0,
        request_meta: dict[str, Any] | None = None,
    ) -> BackendResult: ...


class HyTunerServerBackend:
    """Adapt HyTuner's complete generation endpoint to the Gateway backend interface."""

    def __init__(self, server_manager: Any):
        self._server_manager = server_manager

    async def create_completion(
        self,
        request_id: str,
        *,
        prompt_ids: list[int] | None = None,
        messages: list[dict] | None = None,
        request_kwargs: dict[str, Any] | None = None,
        image_data: Any | None = None,
        video_data: Any | None = None,
        audio_data: Any | None = None,
        priority: int = 0,
        request_meta: dict[str, Any] | None = None,
    ) -> BackendResult:
        assert prompt_ids is not None, 'HyTunerServerBackend requires prompt_ids'
        sampling_params = _vllm_sampling_params(request_kwargs)
        request_metadata = dict(request_meta or {})
        output = await self._server_manager.generate_tokens(
            request_id=_backend_request_id(request_id, request_metadata),
            prompt_ids=list(prompt_ids),
            sampling_params=sampling_params,
            image_data=image_data,
            video_data=video_data,
            audio_data=audio_data,
            priority=priority,
            request_meta=request_metadata,
        )
        token_ids = list(output.token_ids or [])
        stop_reason = output.stop_reason or 'stop'
        if (
            stop_reason in {'completed', 'aborted'}
            and sampling_params.get('max_tokens') is not None
            and len(token_ids) == sampling_params['max_tokens']
        ):
            stop_reason = 'length'
        return BackendResult(
            token_ids=token_ids,
            log_probs=list(output.log_probs) if output.log_probs is not None else None,
            routed_experts=output.routed_experts,
            stop_reason=stop_reason,
            usage={
                'prompt_tokens': len(prompt_ids),
                'completion_tokens': len(token_ids),
                'total_tokens': len(prompt_ids) + len(token_ids),
            },
            extra_fields=dict(output.extra_fields or {}),
        )


class VllmCompletionsBackend:
    """Wraps one vLLM OpenAI-compatible endpoint for the gateway token lane."""

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        topk_logprobs: int | None = None,
        max_model_len: int | None = None,
        timeout: float | None = None,
        max_retries: int = 3,
    ):
        self._base_url = base_url.rstrip('/')
        self._model = model
        self._topk = topk_logprobs
        self._max_model_len = max_model_len
        self._max_retries = max_retries
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(timeout))

    async def create_completion(
        self,
        request_id: str,
        *,
        prompt_ids: list[int] | None = None,
        messages: list[dict] | None = None,
        request_kwargs: dict[str, Any] | None = None,
        image_data: Any | None = None,
        video_data: Any | None = None,
        audio_data: Any | None = None,
        priority: int = 0,
        request_meta: dict[str, Any] | None = None,
    ) -> BackendResult:
        if prompt_ids is None:
            raise ValueError('prompt_ids is required for the token-level vLLM backend')
        if image_data is not None or video_data is not None or audio_data is not None:
            raise NotImplementedError('VllmCompletionsBackend does not support multi-modal inputs')
        params = _vllm_sampling_params(request_kwargs)
        max_tokens = params.pop('max_tokens', 4096)
        if self._max_model_len is not None:
            max_tokens = min(max_tokens, self._max_model_len - len(prompt_ids))
        if max_tokens <= 0:
            raise ValueError(
                f'request {request_id}: prompt of {len(prompt_ids)} tokens leaves no room to generate '
                f'(max_model_len={self._max_model_len})'
            )
        body: dict[str, Any] = {
            'model': self._model,
            'prompt': prompt_ids,
            'max_tokens': max_tokens,
            # logprobs must be requested to recover generated token ids
            'logprobs': self._topk if self._topk else 0,
            'return_tokens_as_token_ids': True,
            'skip_special_tokens': False,
        }
        for key in ('temperature', 'top_p', 'top_k', 'presence_penalty', 'frequency_penalty', 'seed', 'stop'):
            if key in params and params[key] is not None:
                body[key] = params[key]

        data = await self._post_with_retry(body, request_id)
        choices = data.get('choices') or []
        if len(choices) != 1:
            raise RuntimeError(f'vLLM completions returned {len(choices)} choices (expected 1)')
        choice = choices[0]
        finish_reason = choice.get('finish_reason')
        wire_text = choice.get('text')
        if not isinstance(wire_text, str):
            raise RuntimeError('vLLM completions response is missing choice.text')
        logprobs = choice.get('logprobs') or {}
        tokens = logprobs.get('tokens') or []
        if wire_text and not tokens:
            raise RuntimeError('vLLM response is missing generated token IDs')
        token_ids = [_parse_token_id(token) for token in tokens]
        raw_logprobs = logprobs.get('token_logprobs') or []
        # per-token alignment check; a missing logprob is a missing fact — never fabricate 0.0
        if len(raw_logprobs) != len(token_ids):
            raise RuntimeError(f'logprobs misaligned: {len(raw_logprobs)} vs {len(token_ids)} tokens')
        if any(lp is None for lp in raw_logprobs):
            raise RuntimeError('vLLM returned None logprob for a generated token')
        log_probs = [float(lp) for lp in raw_logprobs]
        extra_fields: dict[str, Any] = {}
        top = logprobs.get('top_logprobs')
        if self._topk and token_ids:
            # requested numeric facts must arrive complete; empty/missing is a failure, not a skip
            if not top or len(top) != len(token_ids):
                raise RuntimeError(f'top_logprobs missing or misaligned: {len(top or [])} vs {len(token_ids)} tokens')
            topk_token_ids: list[list[int]] = []
            topk_logprobs: list[list[float]] = []
            for entry in top:
                entry = entry or {}
                topk_token_ids.append([_parse_token_id(token) for token in entry])
                topk_logprobs.append([float(v) for v in entry.values()])
            # legacy wire keys expected by the training consumer
            extra_fields['resp_topk_logprob_indice'] = topk_token_ids
            extra_fields['resp_topk_logprob_value'] = topk_logprobs

        return BackendResult(
            token_ids=token_ids,
            log_probs=log_probs or None,
            stop_reason=finish_reason,
            wire_text=wire_text,
            usage=data.get('usage'),
            id=data.get('id'),
            created=data.get('created'),
            model=data.get('model'),
            extra_fields=extra_fields,
        )

    async def _post_with_retry(self, body: dict[str, Any], request_id: str) -> dict[str, Any]:
        url = f'{self._base_url}/v1/completions'
        last_error: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                response = await self._client.post(url, json=body, headers={'X-Request-Id': request_id})
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as error:
                status = error.response.status_code
                if status < 500 and status != 429:  # client errors do not heal on retry
                    raise RuntimeError(
                        f'vLLM completions rejected ({status}) for request {request_id}: {error.response.text[:200]}'
                    ) from error
                last_error = error
            except httpx.TransportError as error:
                last_error = error
            if attempt + 1 < self._max_retries:
                await asyncio.sleep(1.0 * (attempt + 1))
        raise RuntimeError(
            f'vLLM completions failed after {self._max_retries} attempts for {request_id}: {last_error!r}'
        )

    async def aclose(self) -> None:
        await self._client.aclose()


class OpenAIAPIBackend:
    """Message-level backend wrapping an OpenAI-compatible endpoint.

    Used for Nexus pure inference. No abort concept.
    Returns BackendResult with message/usage filled, token_ids as None.
    """

    def __init__(self, openai_client):
        self._client = openai_client

    @staticmethod
    def _messages_for_openai_wire(messages: list[dict]) -> list[dict]:
        def _serialize_tool_call(tool_call: dict) -> dict:
            function = tool_call.get('function')
            if not isinstance(function, dict):
                return tool_call
            arguments = function.get('arguments')
            if not isinstance(arguments, (dict, list)):
                return tool_call
            return {
                **tool_call,
                'function': {**function, 'arguments': json.dumps(arguments, ensure_ascii=False)},
            }

        return [
            {**message, 'tool_calls': [_serialize_tool_call(tool_call) for tool_call in message['tool_calls']]}
            if message.get('tool_calls')
            else message
            for message in messages
        ]

    async def create_completion(
        self,
        request_id: str,
        *,
        prompt_ids: list[int] | None = None,
        messages: list[dict] | None = None,
        request_kwargs: dict[str, Any] | None = None,
        image_data: Any | None = None,
        video_data: Any | None = None,
        audio_data: Any | None = None,
        priority: int = 0,
        request_meta: dict[str, Any] | None = None,
    ) -> BackendResult:
        assert messages is not None, 'OpenAIAPIBackend requires messages'

        kwargs = dict(request_kwargs or {})
        wire_messages = self._messages_for_openai_wire(messages)

        completion = await self._client.create_completion(messages=wire_messages, **kwargs)
        choice = completion.choices[0]
        msg = choice.message

        if hasattr(msg, 'model_dump'):
            msg_dict = msg.model_dump(exclude_none=True)
        else:
            msg_dict = {'role': 'assistant', 'content': getattr(msg, 'content', None)}
        msg_dict.setdefault('role', 'assistant')
        msg_dict.setdefault('content', getattr(msg, 'content', None))
        reasoning_content = getattr(msg, 'reasoning_content', None)
        if reasoning_content is None:
            provider_fields = getattr(msg, 'provider_specific_fields', None) or {}
            if isinstance(provider_fields, dict):
                reasoning_content = provider_fields.get('reasoning_content')
        if reasoning_content is not None:
            msg_dict['reasoning_content'] = reasoning_content
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            msg_dict['tool_calls'] = [tc.model_dump() for tc in msg.tool_calls]

        usage_dict = None
        if hasattr(completion, 'usage') and completion.usage:
            usage_dict = (
                completion.usage.model_dump() if hasattr(completion.usage, 'model_dump') else dict(completion.usage)
            )
        choice_logprobs = getattr(choice, 'logprobs', None)
        if hasattr(choice_logprobs, 'model_dump'):
            choice_logprobs = choice_logprobs.model_dump()
        choice_provider_fields = getattr(choice, 'provider_specific_fields', None)
        if hasattr(choice_provider_fields, 'model_dump'):
            choice_provider_fields = choice_provider_fields.model_dump()

        return BackendResult(
            message=msg_dict,
            usage=usage_dict,
            stop_reason=getattr(choice, 'finish_reason', None),
            id=getattr(completion, 'id', None),
            created=getattr(completion, 'created', None),
            model=getattr(completion, 'model', None),
            object=getattr(completion, 'object', None),
            system_fingerprint=getattr(completion, 'system_fingerprint', None),
            service_tier=getattr(completion, 'service_tier', None),
            choice_logprobs=choice_logprobs,
            choice_provider_specific_fields=choice_provider_fields
            if isinstance(choice_provider_fields, dict)
            else None,
        )
