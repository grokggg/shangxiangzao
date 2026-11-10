"""LLM Request Handler.

Handles LLM requests from the runtime side via Socket.IO.
This enables installed agents running in sandboxed runtimes to
make LLM calls through the host-side OpenAI client.
"""

import asyncio
import contextlib
import copy
import json
import logging
import os
import time

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal, cast

from nexus.core.openai_client import (
    CONTEXT_LENGTH_EXCEEDED_CODE,
    ContextLengthExceededError,
    OpenAICompatibleClient,
)
from nexus.runtime.server.rpc_replay_guard import RPCReplayGuard
from nexus.runtime.server.services.rsio_llm_bridge import RSIOLLMConfig


if TYPE_CHECKING:
    from nexus.runtime import Runtime


logger = logging.getLogger(__name__)

NoThinkingRequestArgsMode = Literal['replace', 'inherit']

_HEARTBEAT_FAILURE_TIMEOUT = 5 * 60.0
# Keep a just-delivered result long enough to absorb a resend that was already
# in transit when the worker received the final response.
_FINAL_REPLAY_GRACE_SECONDS = 5.0
_UNSUPPORTED_COMPLETION_FIELDS = frozenset({
    'metadata',
    'service_tier',
    'store',
    'stream_options',
})


def sanitize_openai_params(data: dict[str, Any]) -> dict[str, Any]:
    """Drop OpenAI-only request fields unsupported by generic host-side clients."""
    sanitized = dict(data)
    dropped = sorted(key for key in _UNSUPPORTED_COMPLETION_FIELDS if key in sanitized)
    for key in dropped:
        sanitized.pop(key, None)

    if dropped:
        logger.info('Dropping unsupported completion fields from runtime request: %s', ', '.join(dropped))

    return sanitized


class LLMRequestHandler:
    """Handles LLM requests from the runtime side.

    This class manages the Socket.IO event handling for LLM requests
    coming from proxy servers running in the runtime container.
    """

    def __init__(
        self,
        runtime: 'Runtime',
        openai_client: OpenAICompatibleClient,
        no_thinking_request_args: dict[str, Any] | None = None,
        no_thinking_request_args_mode: NoThinkingRequestArgsMode = 'replace',
        haiku_model: str | None = None,
        preserve_thinking: bool = False,
        on_tool_call: Callable[[str], None] | None = None,
    ) -> None:
        self.runtime = runtime
        self.openai_client = openai_client
        self.no_thinking_request_args = no_thinking_request_args
        if no_thinking_request_args_mode not in ('replace', 'inherit'):
            raise ValueError(
                f'no_thinking_request_args_mode must be replace or inherit, got {no_thinking_request_args_mode!r}'
            )
        self.no_thinking_request_args_mode = no_thinking_request_args_mode
        self.haiku_model = haiku_model
        self.preserve_thinking = preserve_thinking
        # Optional host-side observer of tool_call names. Used by kilo to
        # detect 'suggest' tool calls and abort the long-running invoke early
        # (see installed_common.call_runtime_agent and kilo/agent.py).
        self.on_tool_call = on_tool_call
        # Cache of past assistant turns' thinking_blocks and any message-level
        # thought signatures. Each entry carries stable surface identifiers from
        # the assistant message, so re-injection can avoid positional guesses
        # when a scaffold rewrites or drops turns.
        self._thinking_cache: list[dict[str, Any]] = []
        # Same idea as _thinking_cache, but for OpenAI Responses reasoning
        # items carried on chat-like messages by litellm's openai/responses
        # bridge. Runtime chat-proxy scaffolds often drop these non-standard
        # fields, so we conservatively restore them before the next host call.
        self._reasoning_items_cache: list[dict[str, Any]] = []
        self.pending_requests: dict[str, asyncio.Future] = {}
        self._guard = RPCReplayGuard(ttl_seconds=300.0)
        self._report_interval = RSIOLLMConfig().report_interval
        self._handler_registered: bool = False
        # Outcome of every request this handler settled, keyed by request_id and ordered by
        # first completion: error code for a failure, None for a success.
        #
        # An agent whose loop runs inside the sandbox cannot `except ContextLengthExceededError`
        # the way in-process agents (mini_swe_agent, tau2, ace) do: the error crosses the
        # Socket.IO boundary and reaches the agent as an opaque HTTP 500, so it can only report a
        # generic failure. Only the host still knows the true cause.
        #
        # This must stay per-request rather than a handler-wide flag: one handler serves every
        # request of a single call_runtime_agent, so a run where an early call is refused on
        # budget and a later one succeeds would otherwise report the whole run as truncated.
        self._request_outcomes: dict[str, str | None] = {}

    @property
    def request_outcomes(self) -> dict[str, str | None]:
        """Per-request outcomes in completion order: request_id -> error code, or None on success."""
        return dict(self._request_outcomes)

    @property
    def last_request_error_code(self) -> str | None:
        """Error code of the most recently settled request, or None if it succeeded.

        A caller classifying why a sandboxed agent exited should prefer evidence from the agent's
        own event stream, which names the call it actually died on. This is the host-side
        approximation for agents that report nothing usable: the last call to settle is the one
        the agent was on when it gave up. It can miss under concurrency — a call refused on budget
        may be followed by an unrelated success — so it is a fallback, never an override.
        """
        if not self._request_outcomes:
            return None
        return next(reversed(self._request_outcomes.values()))

    def _record_outcome(self, request_id: str, error_code: str | None) -> None:
        # Keyed by request_id so a resend of an already-settled call neither duplicates an entry
        # nor reorders the one already recorded.
        if request_id not in self._request_outcomes:
            self._request_outcomes[request_id] = error_code

    @staticmethod
    def _first_tool_call_id(msg: dict[str, Any]) -> str | None:
        for tc in msg.get('tool_calls') or []:
            tc_id = (tc or {}).get('id')
            if tc_id:
                return str(tc_id)
        return None

    @staticmethod
    def _tool_call_ids(msg: dict[str, Any]) -> list[str]:
        ids: list[str] = []
        for tc in msg.get('tool_calls') or []:
            tc_id = (tc or {}).get('id')
            if tc_id:
                ids.append(str(tc_id))
        return ids

    @staticmethod
    def _normalize_tool_call_arguments(arguments: Any) -> str:
        """Canonicalize a tool_call's arguments for cross-scaffold comparison.

        Chat-proxy scaffolds (opencode, kilo) round-trip tool_calls through their
        own JSON encoders, which preserve the semantic arguments but reorder object
        keys. Parse and re-dump with sorted keys so two semantically identical
        argument strings compare equal; fall back to the raw string when it is not
        valid JSON.
        """
        if arguments is None:
            return ''
        if isinstance(arguments, str):
            try:
                parsed = json.loads(arguments)
            except (ValueError, TypeError):
                return arguments
        else:
            parsed = arguments
        try:
            return json.dumps(parsed, sort_keys=True, ensure_ascii=False)
        except (TypeError, ValueError):
            return str(arguments)

    @classmethod
    def _tool_calls_for_match(cls, msg: dict[str, Any]) -> list[dict[str, Any]]:
        """Return ordered, normalized tool_calls for conservative state matching.

        Tool call order is part of the replay surface. Do not sort or compare
        as a set: if a scaffold reorders calls, skip injection rather than
        risk attaching opaque reasoning state to the wrong assistant turn.

        Each entry is normalized to ``{id, name, arguments}`` so that fields a
        scaffold may add or drop on the round-trip (e.g. the streaming ``index``)
        and JSON key ordering inside ``arguments`` do not defeat the match. The
        anchor remains the globally-unique tool ``id``; name/arguments are a
        secondary cross-check, not a positional guess.
        """
        tool_calls: list[dict[str, Any]] = []
        for tc in msg.get('tool_calls') or []:
            if not isinstance(tc, dict):
                continue
            fn = tc.get('function') or {}
            tool_calls.append({
                'id': str(tc['id']) if tc.get('id') else None,
                'name': fn.get('name') if isinstance(fn, dict) else None,
                'arguments': cls._normalize_tool_call_arguments(fn.get('arguments') if isinstance(fn, dict) else None),
            })
        return tool_calls

    @staticmethod
    def _plaintext_content(msg: dict[str, Any]) -> str | None:
        """Extract comparable assistant-visible text from OpenAI-style content."""
        content = msg.get('content')
        if content is None:
            return None
        if isinstance(content, str):
            return content or None
        if isinstance(content, list):
            parts: list[str] = []
            for part in content:
                if isinstance(part, str):
                    parts.append(part)
                    continue
                if not isinstance(part, dict):
                    continue
                text = part.get('text') or part.get('input_text')
                if isinstance(text, str):
                    parts.append(text)
            return ''.join(parts) if parts else None
        return None

    @staticmethod
    def _thinking_entry_has_payload(entry: dict[str, Any]) -> bool:
        """Whether a cache entry still carries something worth replaying."""
        return bool(entry.get('blocks')) or bool(entry.get('provider_specific_fields'))

    def _find_cached_thinking_entry(
        self,
        *,
        msg: dict[str, Any],
        used: list[bool],
    ) -> dict[str, Any] | None:
        """Find a cached thinking entry without falling back to turn position."""
        tc_id = self._first_tool_call_id(msg)
        content = self._plaintext_content(msg)
        if tc_id:
            matches = [
                (j, entry)
                for j, entry in enumerate(self._thinking_cache)
                if not used[j]
                and self._thinking_entry_has_payload(entry)
                and entry.get('tool_call_id') == tc_id
                and entry.get('content') == content
            ]
            if len(matches) != 1:
                return None

            j, entry = matches[0]
            used[j] = True
            return entry

        if not content:
            return None

        matches = [
            (j, entry)
            for j, entry in enumerate(self._thinking_cache)
            if not used[j]
            and not entry.get('tool_call_id')
            and entry.get('content') == content
            and self._thinking_entry_has_payload(entry)
        ]
        if len(matches) != 1:
            return None

        j, entry = matches[0]
        used[j] = True
        return entry

    def _inject_thinking_blocks(self, messages: list[dict[str, Any]]) -> int:
        """Restore thinking state (blocks and/or message-level signatures).

        Returns the number of assistant messages that received an injection.
        Matching is intentionally conservative: tool-call turns require both
        tool_call_id and exact plaintext content to match. Plain assistant turns
        without tool calls can match by unique exact plaintext content. Missing
        matches are skipped rather than falling back to turn position, because a
        wrong signed thinking block is worse than none.
        """
        if not self._thinking_cache:
            return 0
        injected = 0
        used = [False] * len(self._thinking_cache)
        for msg in messages:
            if msg.get('role') != 'assistant':
                continue
            if 'thinking_blocks' in msg and msg['thinking_blocks']:
                continue

            entry = self._find_cached_thinking_entry(msg=msg, used=used)
            if entry is None:
                continue

            restored = False
            if entry.get('blocks'):
                msg['thinking_blocks'] = [dict(b) for b in entry['blocks']]
                restored = True
            if entry.get('provider_specific_fields') and not msg.get('provider_specific_fields'):
                msg['provider_specific_fields'] = copy.deepcopy(entry['provider_specific_fields'])
                restored = True
            if restored:
                injected += 1
        return injected

    @staticmethod
    def _message_thought_signatures(msg: dict[str, Any]) -> list[str]:
        """Gemini's message-level thought signatures, if this response carries any.

        Anthropic-style providers sign each thinking block. Gemini instead leaves
        ``thinking_blocks[].signature`` unset and returns opaque signatures for the
        non-functionCall parts under ``provider_specific_fields.thought_signatures``
        (tool-call signatures ride on the tool_call itself). Both have to be replayed
        verbatim or the model loses the reasoning state behind the turn.
        """
        provider_fields = msg.get('provider_specific_fields')
        if not isinstance(provider_fields, dict):
            return []
        signatures = provider_fields.get('thought_signatures')
        if not isinstance(signatures, list):
            return []
        return [str(s) for s in signatures if s]

    def _cache_thinking_from_response(self, response_data: dict[str, Any]) -> tuple[int, str | None]:
        """Cache this assistant turn's thinking state for later replay.

        Signed blocks are cached as-is. When the provider signs at the message level
        instead, the unsigned blocks are kept alongside those signatures so the pair
        can be restored together — dropping either half loses the reasoning state.
        """
        choices = response_data.get('choices') or []
        if not choices:
            return 0, None
        msg = choices[0].get('message') or {}
        blocks_raw = [b for b in msg.get('thinking_blocks') or [] if isinstance(b, dict)]
        blocks = [dict(b) for b in blocks_raw if b.get('signature')]

        signatures = self._message_thought_signatures(msg)
        provider_fields = {'thought_signatures': signatures} if signatures else None
        if provider_fields and not blocks:
            blocks = [dict(b) for b in blocks_raw]

        if not blocks and not provider_fields:
            return 0, None

        tc_id: str | None = None
        for tc in msg.get('tool_calls') or []:
            if (tc or {}).get('id'):
                tc_id = str(tc['id'])
                break
        self._thinking_cache.append({
            'tool_call_id': tc_id,
            'content': self._plaintext_content(msg),
            'blocks': blocks,
            'provider_specific_fields': provider_fields,
        })
        return len(blocks) or len(signatures), tc_id

    @staticmethod
    def _normalize_reasoning_item(item: dict[str, Any]) -> dict[str, Any] | None:
        """Copy a Responses reasoning item without leaking invalid/null fields."""
        if not isinstance(item, dict):
            return None

        encrypted_content = item.get('encrypted_content')
        item_id = item.get('id')
        if encrypted_content and not item_id:
            logger.warning('preserve_reasoning: skipping encrypted reasoning item without id')
            return None

        normalized: dict[str, Any] = {'type': item.get('type') or 'reasoning'}
        if item_id:
            normalized['id'] = str(item_id)

        summary = item.get('summary')
        if isinstance(summary, list):
            normalized['summary'] = copy.deepcopy(summary)
        elif summary is not None:
            normalized['summary'] = summary

        content = item.get('content')
        if isinstance(content, list):
            normalized['content'] = copy.deepcopy(content)
        elif content is not None:
            normalized['content'] = content

        if encrypted_content:
            normalized['encrypted_content'] = str(encrypted_content)

        status = item.get('status')
        if status:
            normalized['status'] = status

        return normalized

    def _match_tool_reasoning_entry(
        self,
        *,
        tool_call_ids: list[str],
        tool_calls: list[dict[str, Any]],
        content: str | None,
        used: list[bool],
    ) -> int | None:
        """Index of the unique cached entry for this assistant turn's tool calls.

        Anchored on the ordered, globally-unique tool ``id`` list; the normalized
        tool_calls (name + canonical arguments, see ``_tool_calls_for_match``) are a
        secondary cross-check so a scaffold's added ``index`` field or reordered
        argument keys do not defeat the match. When the cached entry also carries
        plaintext content it must equal the message's content. Returns None when the
        match is absent or ambiguous (never guesses by position).
        """
        matches: list[int] = []
        for j, entry in enumerate(self._reasoning_items_cache):
            if used[j] or not entry.get('items'):
                continue
            if entry.get('tool_call_ids') != tool_call_ids:
                continue
            if entry.get('tool_calls') != tool_calls:
                continue
            entry_content = entry.get('content')
            if entry_content and content and entry_content != content:
                continue
            matches.append(j)
        return matches[0] if len(matches) == 1 else None

    def _match_content_reasoning_entry(
        self,
        *,
        content: str | None,
        used: list[bool],
        exclude: int | None = None,
    ) -> int | None:
        """Index of the unique content-only cached entry matching ``content``.

        A chat-proxy scaffold (e.g. opencode) collapses a multi-choice Responses
        reply -- a text choice plus a separate function_call choice -- into one
        assistant message carrying both content and tool_calls. The reasoning that
        rode on the text choice was cached as a content-only entry (no tool ids), so
        it must be recovered by exact plaintext content rather than tool id. Returns
        None when absent or ambiguous.
        """
        if not content:
            return None
        matches = [
            j
            for j, entry in enumerate(self._reasoning_items_cache)
            if not used[j]
            and j != exclude
            and not entry.get('tool_call_ids')
            and entry.get('content') == content
            and entry.get('items')
        ]
        return matches[0] if len(matches) == 1 else None

    def _collect_cached_reasoning(self, *, msg: dict[str, Any], used: list[bool]) -> list[dict[str, Any]]:
        """Gather cached reasoning_items for one assistant message, in replay order.

        A single scaffold-rewritten assistant message can need reasoning from more
        than one cached turn: the text choice's reasoning (matched by content) and
        the tool-call choice's reasoning (matched by tool id). They are returned in
        temporal order -- content reasoning first, then tool-call reasoning -- so
        litellm's Responses bridge replays them in the same order it emitted them
        (all reasoning_items lower ahead of the function_calls). Matched entries are
        marked used so they are not double-injected.
        """
        tool_call_ids = self._tool_call_ids(msg)
        content = self._plaintext_content(msg)
        collected: list[dict[str, Any]] = []

        if tool_call_ids:
            tool_idx = self._match_tool_reasoning_entry(
                tool_call_ids=tool_call_ids,
                tool_calls=self._tool_calls_for_match(msg),
                content=content,
                used=used,
            )
            # Only pull a separate content-only entry for the merged multi-choice
            # shape, where the tool entry carries no content of its own. If the tool
            # entry already owns this content (HY-style single choice), a same-text
            # content-only entry would be a coincidental different turn -- skip it.
            tool_entry = self._reasoning_items_cache[tool_idx] if tool_idx is not None else None
            if content and not (tool_entry and tool_entry.get('content')):
                content_idx = self._match_content_reasoning_entry(content=content, used=used, exclude=tool_idx)
                if content_idx is not None:
                    used[content_idx] = True
                    collected.extend(self._reasoning_items_cache[content_idx]['items'])
            if tool_idx is not None:
                used[tool_idx] = True
                collected.extend(self._reasoning_items_cache[tool_idx]['items'])
            return collected

        content_idx = self._match_content_reasoning_entry(content=content, used=used)
        if content_idx is not None:
            used[content_idx] = True
            collected.extend(self._reasoning_items_cache[content_idx]['items'])
        return collected

    def _inject_reasoning_items(self, messages: list[dict[str, Any]]) -> int:
        """Restore cached Responses reasoning_items onto assistant messages."""
        if not self._reasoning_items_cache:
            return 0

        injected = 0
        used = [False] * len(self._reasoning_items_cache)
        for msg in messages:
            if msg.get('role') != 'assistant':
                continue
            if msg.get('reasoning_items'):
                continue

            items = self._collect_cached_reasoning(msg=msg, used=used)
            if items:
                msg['reasoning_items'] = copy.deepcopy(items)
                injected += 1
        return injected

    def _cache_reasoning_items_from_response(self, response_data: dict[str, Any]) -> tuple[int, str | None]:
        """Cache this assistant turn's Responses reasoning_items for replay."""
        total_items = 0
        first_key: str | None = None

        for choice in response_data.get('choices') or []:
            msg = (choice or {}).get('message') or {}
            raw_items = msg.get('reasoning_items') or []
            if not isinstance(raw_items, list):
                continue

            items = [
                normalized
                for item in raw_items
                if isinstance(item, dict)
                for normalized in [self._normalize_reasoning_item(item)]
                if normalized is not None
            ]
            if not items:
                continue

            tool_call_ids = self._tool_call_ids(msg)
            tool_calls = self._tool_calls_for_match(msg)
            content = self._plaintext_content(msg)
            self._reasoning_items_cache.append({
                'tool_call_ids': tool_call_ids,
                'tool_calls': tool_calls,
                'content': content,
                'items': items,
            })
            total_items += len(items)
            if first_key is None:
                first_key = ','.join(tool_call_ids) if tool_call_ids else '<content>'

        return total_items, first_key

    @staticmethod
    def _maybe_tamper_for_test(messages: list[dict[str, Any]]) -> int | None:
        """Optional fault injector: corrupt the Nth assistant turn's signature.

        Activated by env var ``NEXUS_TAMPER_THINKING_INDEX`` (0-based index over
        assistant messages that carry thinking_blocks). Used to verify that
        Bedrock validates each historical signature independently and reports
        the precise message index. Returns the assistant index that was tampered,
        or None if disabled / not applicable.
        """
        raw = os.environ.get('NEXUS_TAMPER_THINKING_INDEX')
        if raw is None:
            return None
        try:
            target = int(raw)
        except ValueError:
            return None
        seen = 0
        for msg in messages:
            if msg.get('role') != 'assistant':
                continue
            tbs = msg.get('thinking_blocks') or []
            if not tbs:
                continue
            if seen == target:
                first = tbs[0]
                if isinstance(first, dict) and first.get('signature'):
                    sig = first['signature']
                    # Flip one base64 char so the signature is still well-formed
                    # base64 but cryptographically invalid.
                    flip = 'B' if sig[0] != 'B' else 'C'
                    first['signature'] = flip + sig[1:]
                    return target
            seen += 1
        return None

    async def _produce_response(
        self, request_id: str, data: dict[str, Any], *, rsio_mode: bool = True
    ) -> dict[str, Any]:
        """Run the host-side LLM call and return the response dump.

        ``data`` has already been copied and had ``rsio`` / ``request_id``
        stripped by :meth:`handle_llm_request`. Raises on failure. The periodic
        ``running`` heartbeat is emitted only in ``rsio_mode`` — an old worker
        cannot digest it.
        """
        requested_model = data.get('model')
        host_request_args = self.openai_client.config.request_args
        host_model = host_request_args.get('model')
        if host_model:
            data.pop('model', None)
        # Preserve the historical handler contract: request_args.reasoning_effort
        # is host-pinned, while max_tokens remains a default unless the client
        # explicitly declares it under enforced_request_args.
        if host_request_args.get('reasoning_effort') is not None:
            data.pop('reasoning_effort', None)

        try:
            client_has_thinking = data.pop('_client_has_thinking', None)
            client_requests_haiku = data.pop('_client_requests_haiku', None)

            if client_has_thinking is False and self.no_thinking_request_args is not None:
                no_think_args = copy.deepcopy(self.no_thinking_request_args)
                if self.no_thinking_request_args_mode == 'inherit':
                    no_think_overrides = no_think_args
                    no_think_args = copy.deepcopy(host_request_args)
                    no_think_args.update(no_think_overrides)
                    for field in ('thinking', 'reasoning_effort', 'output_config'):
                        if field not in no_think_overrides:
                            no_think_args.pop(field, None)
                    extra_body = no_think_args.get('extra_body')
                    override_extra_body = no_think_overrides.get('extra_body')
                    if isinstance(extra_body, dict):
                        extra_body = dict(extra_body)
                        for field in ('thinking', 'output_config'):
                            if not isinstance(override_extra_body, dict) or field not in override_extra_body:
                                extra_body.pop(field, None)
                        if extra_body:
                            no_think_args['extra_body'] = extra_body
                        else:
                            no_think_args.pop('extra_body', None)
                if client_requests_haiku and self.haiku_model:
                    no_think_args.pop('model', None)
                data['_request_args_override'] = no_think_args

            if client_requests_haiku and self.haiku_model:
                data['model'] = self.haiku_model

            if 'request_meta' in data and not getattr(self.openai_client, 'supports_request_meta', False):
                data.pop('request_meta', None)

            final_model = data.get('model') or host_model or requested_model
            thinking_str = 'no_thinking' if client_has_thinking is False else 'thinking'
            haiku_str = ' [haiku]' if client_requests_haiku else ''
            logger.info(
                f'Handling LLM request: {request_id}, '
                f'requested_model={requested_model}, final_model={final_model}, '
                f'{thinking_str}{haiku_str}'
            )

            request_data = sanitize_openai_params(data)

            if self.preserve_thinking and isinstance(request_data.get('messages'), list):
                injected = self._inject_thinking_blocks(request_data['messages'])
                if injected:
                    logger.info(
                        'preserve_thinking: injected thinking_blocks into %d assistant message(s) [%s]',
                        injected,
                        request_id,
                    )
                reasoning_injected = self._inject_reasoning_items(request_data['messages'])
                if reasoning_injected:
                    logger.info(
                        'preserve_reasoning: injected reasoning_items into %d assistant message(s) [%s]',
                        reasoning_injected,
                        request_id,
                    )
                tampered = self._maybe_tamper_for_test(request_data['messages'])
                if tampered is not None:
                    logger.warning(
                        'preserve_thinking: TAMPERED signature on assistant turn #%d for diagnostics [%s]',
                        tampered,
                        request_id,
                    )

            heartbeat = asyncio.create_task(self._run_heartbeat(request_id)) if rsio_mode else None
            try:
                response = await self.openai_client.create_completion(**request_data)
            finally:
                if heartbeat is not None:
                    heartbeat.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await heartbeat

            response_dump = response.model_dump() if hasattr(response, 'model_dump') else response

            if self.preserve_thinking and isinstance(response_dump, dict):
                cached_n, cached_key = self._cache_thinking_from_response(response_dump)
                if cached_n:
                    logger.info(
                        'preserve_thinking: cached %d thinking block(s) (key=%s) [%s]',
                        cached_n,
                        cached_key or '<content>',
                        request_id,
                    )
                reasoning_cached_n, reasoning_cached_key = self._cache_reasoning_items_from_response(response_dump)
                if reasoning_cached_n:
                    logger.info(
                        'preserve_reasoning: cached %d reasoning item(s) (key=%s) [%s]',
                        reasoning_cached_n,
                        reasoning_cached_key or '<unknown>',
                        request_id,
                    )

            # Notify host-side tool_call observer (used by kilo to detect 'suggest').
            if self.on_tool_call is not None and isinstance(response_dump, dict):
                for choice in response_dump.get('choices') or []:
                    message = (choice or {}).get('message') or {}
                    for tc in message.get('tool_calls') or []:
                        fn = (tc or {}).get('function') or {}
                        fn_name = fn.get('name') or ''
                        if fn_name:
                            try:
                                self.on_tool_call(fn_name)
                            except Exception:
                                logger.exception('on_tool_call callback raised; ignoring')

            if self.openai_client.config.client_args.get('log_llm_usage'):
                rd = response_dump
                usage = (rd.get('usage') or {}) if isinstance(rd, dict) else {}
                cache_read = usage.get('cache_read_input_tokens', 0) or 0
                cache_creation = usage.get('cache_creation_input_tokens', 0) or 0
                input_tokens = usage.get('prompt_tokens', 0) or usage.get('input_tokens', 0) or 0
                output_tokens = usage.get('completion_tokens', 0) or usage.get('output_tokens', 0) or 0
                total_cached = cache_read + cache_creation
                hit_rate = (cache_read / total_cached * 100) if total_cached > 0 else 0
                prompt_details = usage.get('prompt_tokens_details') or {}
                input_details = usage.get('input_tokens_details') or {}
                cached_tokens = 0
                if isinstance(prompt_details, dict):
                    cached_tokens = (
                        prompt_details.get('gpu_cached_tokens', 0) or prompt_details.get('cached_tokens', 0) or 0
                    )
                if not cached_tokens and isinstance(input_details, dict):
                    cached_tokens = (
                        input_details.get('gpu_cached_tokens', 0) or input_details.get('cached_tokens', 0) or 0
                    )
                cached_input_rate = (
                    (cached_tokens / input_tokens * 100) if input_tokens > 0 and cached_tokens > 0 else 0
                )
                logger.info(
                    'LLM usage [%s]: input=%d output=%d cache_read=%d cache_creation=%d hit_rate=%.1f%% cached_tokens=%d cached_input_rate=%.1f%%',
                    request_id,
                    input_tokens,
                    output_tokens,
                    cache_read,
                    cache_creation,
                    hit_rate,
                    cached_tokens,
                    cached_input_rate,
                )

            return response_dump

        except Exception as e:
            logger.error(f'LLM request {request_id} failed: {e}')
            raise

    async def _emit_running(self, request_id: str, *, kind: str = 'heartbeat') -> bool:
        """Emit one ``running`` report for *request_id*.

        ``kind`` is only for the local log line (``ack`` vs ``heartbeat``); the
        wire payload is identical. Both submit acknowledgements and periodic
        heartbeats stay at DEBUG to avoid flooding normal logs. Only ever
        called in RSIO mode — an old worker
        cannot digest a ``running`` report (it has no ``success`` field) and
        would mistake it for a failed final.

        Delivery is best-effort: if the socket is currently disconnected the
        report is dropped silently. The worker's rsio driver will resend the
        original request when it stops hearing heartbeats, and the replay
        guard will short-circuit the duplicate to the cached final anyway.
        """
        logger.debug('rsio.host: running %s request_id=%s', kind, request_id)
        try:
            await self.runtime.emit(
                'llm_response',
                {'request_id': request_id, 'rsio': {'kind': 'running'}},
            )
        except Exception as exc:
            logger.warning('rsio.host: running %s emit dropped request_id=%s (%s)', kind, request_id, exc)
            return False
        return True

    async def _run_heartbeat(self, request_id: str) -> None:
        """Periodically report that this request is still being processed."""
        failure_started_at: float | None = None
        try:
            while True:
                await asyncio.sleep(self._report_interval)
                if await self._emit_running(request_id):
                    failure_started_at = None
                    continue

                now = time.monotonic()
                if failure_started_at is None:
                    failure_started_at = now
                elif now - failure_started_at >= _HEARTBEAT_FAILURE_TIMEOUT:
                    logger.warning(
                        'rsio.host: heartbeat stopped after %.0fs of consecutive emit failures request_id=%s',
                        now - failure_started_at,
                        request_id,
                    )
                    return
        except asyncio.CancelledError:
            raise

    async def _emit_final(
        self,
        request_id: str,
        *,
        success: bool,
        response: dict[str, Any] | None = None,
        error: str | None = None,
        error_code: str | None = None,
        rsio_mode: bool = True,
    ) -> bool:
        """Emit the terminal ``llm_response`` for *request_id*.

        Delivery is best-effort: if the socket is disconnected right now, we
        log and swallow the exception. The worker will resend the request and
        the replay guard turns that into a fresh ``_emit_final`` call — which
        can then land on a reconnected socket. Raising here instead would
        break the enclosing ``handle_llm_request`` and leave the worker
        without an eventual final anyway. A return value of ``True`` means the
        Socket.IO emit completed without raising; it is not an end-to-end ACK.
        """
        payload: dict[str, Any] = {'request_id': request_id, 'success': success}
        if rsio_mode:
            # Old workers ignore an unknown ``rsio`` sub-field on the final, so
            # it is harmless to include — but we omit it in legacy mode to keep
            # the wire identical to the pre-RSIO response.
            payload['rsio'] = {'kind': 'final'}
        if success:
            payload['response'] = response
        else:
            payload['error'] = error or 'Unknown error'
            if error_code:
                # Machine-readable companion to the free-text ``error``. The worker turns it back
                # into a typed failure so the in-sandbox proxy can name the cause to its agent,
                # which otherwise only ever sees an opaque HTTP 500. Old workers ignore it.
                payload['error_code'] = error_code
        logger.debug('rsio.host: emit final request_id=%s success=%s rsio=%s', request_id, success, rsio_mode)
        try:
            await self.runtime.emit('llm_response', payload)
        except Exception as exc:
            logger.warning('rsio.host: final emit dropped request_id=%s (%s); worker will resend', request_id, exc)
            return False
        return True

    async def handle_llm_request(self, data: dict[str, Any]) -> None:
        """Handle an LLM request from the runtime side.

        Dedups overlapping retries by ``request_id`` (same host process), emits
        ``running`` heartbeats while the model call runs, and always attempts a
        terminal ``llm_response``. A completed replay result is retained until
        that final emit completes without raising.

        Backward compatibility (new host + old worker): the inbound request
        carries an ``rsio`` sub-field only from a new worker. When it is absent
        the caller is an old worker that cannot digest ``running`` heartbeats
        and does not expect ``rsio`` on the response — so we skip the heartbeat
        and emit a plain legacy final.
        """
        data = dict(data)
        rsio = data.pop('rsio', None)  # protocol envelope — never forward to the provider
        rsio_mode = rsio is not None  # absent ⇒ old worker ⇒ legacy responses
        request_id = data.pop('request_id', None) or '<no-request-id>'
        attempt = (rsio or {}).get('attempt')
        logger.debug(
            'rsio.host: recv llm_request request_id=%s attempt=%s mode=%s',
            request_id,
            attempt,
            'rsio' if rsio_mode else 'legacy',
        )

        ran = False

        async def once() -> dict[str, Any]:
            nonlocal ran
            ran = True
            # Ack with a `running` report before starting the model call, and
            # only on the owning frame (guard runs `once` once per request_id).
            # This tells the caller "received, working on it" so its probe window
            # switches from ack_timeout to the longer report_timeout — a normal
            # multi-second LLM call is no longer mistaken for a lost request and
            # needlessly resent. In-flight/cached resends don't re-ack: the owner
            # already did, and a resend gets the final directly anyway.
            # Skip entirely for an old worker (see method docstring).
            if rsio_mode:
                await self._emit_running(request_id, kind='ack')
            return await self._produce_response(request_id, data, rsio_mode=rsio_mode)

        try:
            result = await self._guard.run('llm_request', request_id, once)
        except Exception as e:
            error_code = CONTEXT_LENGTH_EXCEEDED_CODE if isinstance(e, ContextLengthExceededError) else None
            self._record_outcome(request_id, error_code)
            logger.warning('rsio.host: request_id=%s failed (code=%s): %s', request_id, error_code, e)
            delivered = await self._emit_final(
                request_id,
                success=False,
                error=str(e),
                error_code=error_code,
                rsio_mode=rsio_mode,
            )
            if delivered:
                await self._guard.discard('llm_request', request_id, delay=_FINAL_REPLAY_GRACE_SECONDS)
            return
        self._record_outcome(request_id, None)
        if not ran:
            # Guard replayed an in-flight/cached result: this is a resend of a
            # request already being handled or already completed. The model call
            # was NOT re-executed — we just re-emit the final (a prior final may
            # have been the frame that got lost and triggered this resend).
            logger.info('rsio.host: dedup replay request_id=%s attempt=%s (no model re-call)', request_id, attempt)
        # Re-emit final on every attempt (cache hit or fresh run): a prior final
        # may have been the message that got lost and triggered the resend.
        delivered = await self._emit_final(request_id, success=True, response=result, rsio_mode=rsio_mode)
        if delivered:
            await self._guard.discard('llm_request', request_id, delay=_FINAL_REPLAY_GRACE_SECONDS)

    def register(self) -> None:
        """Register the LLM request handler with the runtime."""
        if self._handler_registered:
            return

        self.runtime.on('llm_request', cast(Any, self.handle_llm_request))
        self._handler_registered = True
        logger.info('LLM request handler registered on runtime')

    def unregister(self) -> None:
        """Unregister the LLM request handler."""
        if not self._handler_registered:
            return

        self.runtime.off('llm_request', cast(Any, self.handle_llm_request))
        self._handler_registered = False
        logger.debug('LLM request handler unregistered')
