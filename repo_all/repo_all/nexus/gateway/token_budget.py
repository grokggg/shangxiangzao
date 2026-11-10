"""Context-aware token budget accounting for one Gateway session."""

from __future__ import annotations

from typing import Any

from nexus.core import ContextLengthExceededError
from nexus.utils.token_budget_config import TokenBudgetComponent, parse_token_budget_include


class TokenBudgetState:
    """Track trajectory and context-local token consumption."""

    def __init__(
        self,
        *,
        max_model_len: int | None = None,
        token_budget: int | None = None,
        token_budget_include: list[str] | None = None,
    ):
        self._max_model_len = max_model_len
        self._token_budget = token_budget
        self._token_budget_components = parse_token_budget_include(token_budget_include)
        self._initial_prompt_lens: dict[Any, int] = {}
        self._token_consumed = 0
        self._segment_consumed_tokens: dict[Any, int] = {}
        self._accounted_prompt_ids: dict[Any, list[int]] = {}

    def prepare_generation(
        self,
        context_key: Any,
        raw_prompt_ids: list[int],
        prompt_ids: list[int],
        request_kwargs: dict[str, Any],
        uses_default_max_tokens: bool,
    ) -> dict[str, Any]:
        # Budget checks are read-only: compare against the last successfully committed stream.
        previous_ids = self._accounted_prompt_ids.get(context_key, [])
        common = self._common_prefix_length(previous_ids, raw_prompt_ids)
        extends_previous = bool(previous_ids) and common == len(previous_ids)
        initial_prompt_len = self._initial_prompt_lens.get(context_key, len(prompt_ids))
        if previous_ids and not extends_previous:
            # Treat a rewritten prompt as a candidate baseline; record_generation commits it on success.
            initial_prompt_len = len(prompt_ids)

        params = dict(request_kwargs)
        if self._max_model_len is not None and uses_default_max_tokens:
            context_max_tokens = self._max_model_len - initial_prompt_len
            configured_max_tokens = params.get('max_tokens')
            params['max_tokens'] = (
                context_max_tokens
                if configured_max_tokens is None
                else min(int(configured_max_tokens), context_max_tokens)
            )
        if not uses_default_max_tokens or params.get('max_tokens') is None:
            return params

        remaining = int(params['max_tokens']) - (len(prompt_ids) - initial_prompt_len)
        if self._token_budget is not None:
            remaining = min(remaining, self._token_budget - self._token_consumed)
        if remaining <= 0:
            raise ContextLengthExceededError(
                f'Gateway context length exceeded: max_tokens={params["max_tokens"]}, '
                f'initial_prompt_len={initial_prompt_len}, prompt_ids={len(prompt_ids)}'
            )
        params['max_tokens'] = remaining
        return params

    def update_prompt(self, context_key: Any, raw_prompt_ids: list[int], prompt_ids: list[int]) -> None:
        self._account_prompt(context_key, raw_prompt_ids, consume=True)
        self._initial_prompt_lens.setdefault(context_key, len(prompt_ids))

    def record_generation(
        self,
        context_key: Any,
        raw_prompt_ids: list[int],
        prompt_ids: list[int],
        generated_ids: list[int],
        *,
        output_component: TokenBudgetComponent = TokenBudgetComponent.AGENT_OUTPUT,
    ) -> None:
        self._account_prompt(context_key, raw_prompt_ids, consume=False)
        self._initial_prompt_lens.setdefault(context_key, len(prompt_ids))
        if output_component in self._token_budget_components:
            self._token_consumed += len(generated_ids)
        self._segment_consumed_tokens[context_key] = self._segment_consumed_tokens.get(context_key, 0) + len(
            generated_ids
        )
        self._accounted_prompt_ids[context_key] = list(raw_prompt_ids) + generated_ids

    def segment_consumed_tokens(self, context_key: Any) -> int:
        return self._segment_consumed_tokens.get(context_key, 0)

    def set_segment_consumed_tokens(self, context_key: Any, value: int) -> None:
        self._segment_consumed_tokens[context_key] = value

    def _account_prompt(self, context_key: Any, prompt_ids: list[int], *, consume: bool) -> None:
        previous_ids = self._accounted_prompt_ids.get(context_key, [])
        common = self._common_prefix_length(previous_ids, prompt_ids)
        extends_previous = bool(previous_ids) and common == len(previous_ids)
        if previous_ids and not extends_previous:
            self._initial_prompt_lens.pop(context_key, None)
        added = len(prompt_ids) - common if consume and extends_previous else 0
        if TokenBudgetComponent.TOOL_OBSERVATION in self._token_budget_components:
            self._token_consumed += added
        self._segment_consumed_tokens[context_key] = self._segment_consumed_tokens.get(context_key, 0) + added
        self._accounted_prompt_ids[context_key] = list(prompt_ids)

    @staticmethod
    def _common_prefix_length(previous_ids: list[int], prompt_ids: list[int]) -> int:
        common = 0
        for previous, current in zip(previous_ids, prompt_ids, strict=False):
            if previous != current:
                break
            common += 1
        return common
