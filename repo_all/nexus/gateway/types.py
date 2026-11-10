"""Core data structures for the Gateway module."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, model_validator


@dataclass(frozen=True)
class FrozenPrefix:
    """The chain's frozen physical stream as per-message spans and token IDs.

    Spans and IDs are exactly as recorded at append time and are never re-derived
    or re-encoded. Semantic message matching authorizes reuse; canonical rendering
    supplies only the continuation and may not replace this stream. Internal span
    boundaries carry no authority: only the concatenation is load-bearing, and it
    ends at the last sampled token.
    """

    count: int
    text: str
    templated_messages: tuple[str, ...]
    message_token_ids: tuple[tuple[int, ...], ...]


class BackendResult(BaseModel):
    """Unified backend return type.

    Token-level backends (vLLM replica) fill token_ids/log_probs/routed_experts.
    Message-level backends (OpenAI API) fill message/usage.
    """

    # Token-level output (vLLM replica fills these)
    token_ids: list[int] | None = None
    log_probs: list[float] | None = None
    routed_experts: Any | None = None
    stop_reason: str | None = None
    wire_text: str | None = None
    inference_time: float | None = None
    # Message-level output (API fills these)
    message: dict | None = None
    usage: dict | None = None
    id: str | None = None
    created: int | None = None
    model: str | None = None
    object: str | None = None
    system_fingerprint: str | None = None
    service_tier: str | None = None
    choice_logprobs: Any | None = None
    choice_provider_specific_fields: dict[str, Any] | None = None
    # Shared
    extra_fields: dict[str, Any] = {}

    @model_validator(mode='after')
    def validate_token_facts(self) -> BackendResult:
        if self.token_ids is None:
            return self
        token_count = len(self.token_ids)
        if self.log_probs is not None and len(self.log_probs) != token_count:
            raise ValueError(f'log_probs length {len(self.log_probs)} != token_ids length {token_count}')
        indices = self.extra_fields.get('resp_topk_logprob_indice')
        values = self.extra_fields.get('resp_topk_logprob_value')
        if (indices is None) != (values is None):
            raise ValueError('top-k indices and values must be present together')
        if indices is not None and values is not None:
            if len(indices) != token_count or len(values) != token_count:
                raise ValueError('top-k rows must align with token_ids')
            for row, (index_row, value_row) in enumerate(zip(indices, values, strict=True)):
                if len(index_row) != len(value_row):
                    raise ValueError(f'top-k row {row} width mismatch')
        return self


class PromptRender(BaseModel):
    """Exact tokenizer-template output for one model request."""

    text: str
    token_ids: list[int]
    templated_messages: list[str]
    message_token_ids: list[list[int]]
    generation_prompt: str
    generation_prompt_ids: list[int]

    @model_validator(mode='after')
    def validate_partition(self) -> PromptRender:
        if len(self.templated_messages) != len(self.message_token_ids):
            raise ValueError('templated message and token counts differ')
        if ''.join(self.templated_messages) + self.generation_prompt != self.text:
            raise ValueError('templated messages do not reconstruct the rendered prompt')
        token_ids = [token for message_ids in self.message_token_ids for token in message_ids]
        if token_ids + self.generation_prompt_ids != self.token_ids:
            raise ValueError('templated message tokens do not reconstruct the prompt token IDs')
        return self


class ProcessorResult(BaseModel):
    """Return type of Processor.create_completion(). Transparent to session layer."""

    message: dict  # OpenAI-format assistant message
    # Token-level (only filled by token path)
    token_ids: list[int] | None = None
    prompt_ids: list[int] | None = None
    raw_prompt_ids: list[int] | None = None
    log_probs: list[float] | None = None
    routed_experts: Any | None = None
    stop_reason: str | None = None
    completion_text: str | None = None
    prompt_render: PromptRender | None = None
    inference_time: float | None = None
    # Message-level (only filled by message path)
    usage: dict | None = None
    id: str | None = None
    created: int | None = None
    model: str | None = None
    object: str | None = None
    system_fingerprint: str | None = None
    service_tier: str | None = None
    choice_logprobs: Any | None = None
    choice_provider_specific_fields: dict[str, Any] | None = None
    # Shared
    extra_fields: dict[str, Any] = {}


class GatewayCompletionResult(BaseModel):
    """External return type of create_completion."""

    message: dict
    token_ids: list[int] | None = None
    log_probs: list[float] | None = None
    routed_experts: Any | None = None
    stop_reason: str | None = None
    usage: dict | None = None
    id: str | None = None
    created: int | None = None
    model: str | None = None
    object: str | None = None
    system_fingerprint: str | None = None
    service_tier: str | None = None
    choice_logprobs: Any | None = None
    choice_provider_specific_fields: dict[str, Any] | None = None
    extra_fields: dict[str, Any] = {}
    recorded: bool = True
    inference_time: float | None = None
