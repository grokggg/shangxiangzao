"""Processor: model I/O with two internal paths.

Token-level path (training / vLLM replica):
    apply_chat_template -> tokenize -> backend.create_completion(prompt_ids=...) -> decode -> parse

Message-level path (pure inference / OpenAI API):
    backend.create_completion(messages=...) -> pass-through

Path selection: determined by whether a tokenizer is provided at construction time.
"""

import inspect
import logging
import os
import time
import uuid

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from nexus.gateway.backend import InferenceBackend
from nexus.gateway.types import BackendResult, FrozenPrefix, ProcessorResult, PromptRender


logger = logging.getLogger(__name__)


_PrepareGeneration = Callable[[list[int], list[int], dict[str, Any], bool], dict[str, Any]]


@dataclass(frozen=True)
class _ToolParseRequest:
    tools: list[dict[str, Any]] | None
    chat_template_kwargs: dict[str, Any]


class Processor:
    """Model I/O processor with two paths.

    The path is determined at construction time based on whether a tokenizer is provided.
    Token-level path handles template/tokenize/decode/parse; message-level path is pass-through.
    """

    def __init__(
        self,
        backend: InferenceBackend,
        # Token-level deps (optional — determines path)
        tokenizer=None,
        hf_processor=None,
        tool_parser=None,
        template_parser=None,
        apply_chat_template_kwargs: dict[str, Any] | None = None,
        process_vision_info_fn=None,
        # Shared
        priority: int = 0,
    ):
        self._backend = backend
        self._tokenizer = tokenizer
        self._hf_processor = hf_processor
        self._tool_parser = tool_parser
        self._template_parser = template_parser
        self._apply_chat_template_kwargs = {'is_training': True, **(apply_chat_template_kwargs or {})}
        self._process_vision_info_fn = process_vision_info_fn
        self._priority = priority
        # Turn-boundary seam policy; see _build_prompt_render. The default prefers an
        # exact direct extension and may insert a verified closure seam after a divergent
        # frozen stream. GATEWAY_LEGACY_SEAM_COMPAT=1 instead reproduces the legacy
        # HyTunerClient canonical-prefix subtraction for every reused prefix.
        # Delete the switch once the legacy subtraction and its goldens are retired.
        self._legacy_seam_compat = os.environ.get('GATEWAY_LEGACY_SEAM_COMPAT', '') == '1'
        self._is_token_level = tokenizer is not None

    @property
    def is_token_level(self) -> bool:
        return self._is_token_level

    async def create_completion(
        self,
        request_id: str,
        messages: list[dict],
        request_kwargs: dict[str, Any] | None = None,
        priority: int | None = None,
        prepare_generation: _PrepareGeneration | None = None,
        uses_default_max_tokens: bool = True,
        request_meta: dict[str, Any] | None = None,
        frozen_prefix: FrozenPrefix | None = None,
    ) -> ProcessorResult:
        """Unified entry point. Dispatches based on is_token_level.

        ``request_meta`` is caller-owned and forwarded to the backend untouched
        (gateway never interprets it) — e.g. HyTuner's ``validate`` flag rides it
        to the generation endpoint.
        """
        execution_request = dict(request_kwargs or {})
        tools = execution_request.get('tools')

        if self._is_token_level:
            return await self._token_level_completion(
                request_id,
                messages,
                tools,
                execution_request,
                priority=priority,
                prepare_generation=prepare_generation,
                uses_default_max_tokens=uses_default_max_tokens,
                request_meta=request_meta,
                frozen_prefix=frozen_prefix,
            )
        return await self._message_level_completion(
            request_id,
            messages,
            execution_request,
            request_meta=request_meta,
        )

    # ---- Token-level path ----

    async def _token_level_completion(
        self,
        request_id: str,
        messages: list[dict],
        tools: list[dict] | None,
        request_kwargs: dict[str, Any],
        priority: int | None,
        prepare_generation: _PrepareGeneration | None,
        uses_default_max_tokens: bool,
        request_meta: dict[str, Any] | None = None,
        frozen_prefix: FrozenPrefix | None = None,
    ) -> ProcessorResult:
        """Template -> tokenize -> backend.create_completion(prompt_ids) -> decode -> parse."""
        # 1. Render & tokenize
        raw_ids, expanded_ids, mm_data, prompt_render = self._render_and_tokenize(
            messages,
            tools,
            frozen_prefix=frozen_prefix,
        )
        if prepare_generation is not None:
            request_kwargs = prepare_generation(
                raw_ids,
                expanded_ids,
                request_kwargs,
                uses_default_max_tokens,
            )

        # 2. Call backend with prompt_ids
        inference_started = time.perf_counter()
        result: BackendResult = await self._backend.create_completion(
            request_id,
            prompt_ids=expanded_ids,
            request_kwargs=request_kwargs,
            image_data=mm_data.get('image_data'),
            video_data=mm_data.get('video_data'),
            audio_data=mm_data.get('audio_data'),
            priority=self._priority if priority is None else priority,
            request_meta=request_meta,
        )
        inference_time = result.inference_time
        if inference_time is None:
            abort_wait_time = float((result.extra_fields or {}).get('_abort_wait_time') or 0.0)
            inference_time = max(0.0, time.perf_counter() - inference_started - abort_wait_time)
        extra_fields = dict(result.extra_fields)

        # 3. Decode
        tokenizer = self._tokenizer
        assert tokenizer is not None
        if result.token_ids is None:
            raise RuntimeError('Token-level backend returned no token IDs')
        token_ids = list(result.token_ids)
        text = tokenizer.decode(token_ids, skip_special_tokens=False)
        wire_text = (
            result.wire_text if result.wire_text is not None else tokenizer.decode(token_ids, skip_special_tokens=True)
        )

        # 4. Parse tool calls
        raw_tool = None
        tool_calls = None
        content = wire_text
        if self._tool_parser and token_ids:
            try:
                raw_tool, content, tool_calls = await self._parse_tool_calls(token_ids, wire_text, tools)
            except Exception as exc:
                if result.stop_reason != 'length':
                    raise
                logger.warning(
                    'Tool-call parsing failed after output reached max_tokens; preserving raw partial response: %s: %s',
                    type(exc).__name__,
                    exc,
                )
            else:
                hidden_suffix = text[len(wire_text) :] if text.startswith(wire_text) else ''
                if tool_calls and hidden_suffix and isinstance(content, str) and content.endswith(hidden_suffix):
                    content = content[: -len(hidden_suffix)]

        # 5. Build message
        message: dict[str, Any] = {'role': 'assistant', 'content': content}
        if self._tool_parser:
            message['raw_response'] = wire_text
            if raw_tool is not None:
                message['raw_tool'] = raw_tool
            think_end = getattr(self._tool_parser, 'think_end_token', '</think>')
            if isinstance(content, str) and think_end in content:
                reasoning, content = content.split(think_end, 1)
                think_start = getattr(self._tool_parser, 'think_start_token', '<think>')
                if reasoning.startswith(think_start):
                    reasoning = reasoning[len(think_start) :]
                message['reasoning_content'] = reasoning or None
                message['content'] = content
        if tool_calls:
            message['tool_calls'] = tool_calls

        return ProcessorResult(
            message=message,
            token_ids=list(token_ids),
            prompt_ids=expanded_ids,
            raw_prompt_ids=raw_ids,
            log_probs=result.log_probs,
            routed_experts=result.routed_experts,
            stop_reason=result.stop_reason,
            completion_text=text,
            prompt_render=prompt_render,
            inference_time=inference_time,
            usage=result.usage,
            id=result.id,
            created=result.created,
            model=result.model,
            object=result.object,
            system_fingerprint=result.system_fingerprint,
            service_tier=result.service_tier,
            choice_logprobs=result.choice_logprobs,
            choice_provider_specific_fields=result.choice_provider_specific_fields,
            extra_fields=extra_fields,
        )

    def _render_and_tokenize(
        self,
        messages: list[dict],
        tools: list[dict] | None,
        add_generation_prompt: bool = True,
        frozen_prefix: FrozenPrefix | None = None,
    ) -> tuple[list[int], list[int], dict[str, Any], PromptRender]:
        """Render and tokenize one request, preserving exact message ownership."""
        tokenizer = self._tokenizer
        assert tokenizer is not None
        if (
            self._hf_processor
            and self._process_vision_info_fn
            and any(isinstance(message.get('content'), list) for message in messages)
        ):
            raise NotImplementedError('Per-message multi-modal prompt expansion is not supported yet')

        text = self._render_chat(
            messages,
            tools,
            add_generation_prompt=add_generation_prompt,
        )
        prompt_render = self._build_prompt_render(
            messages,
            tools,
            text=text,
            add_generation_prompt=add_generation_prompt,
            frozen=frozen_prefix,
        )

        raw_ids = list(prompt_render.token_ids)
        expanded_ids = list(raw_ids)

        return raw_ids, expanded_ids, {}, prompt_render

    def _build_prompt_render(
        self,
        messages: list[dict],
        tools: list[dict] | None,
        *,
        text: str,
        add_generation_prompt: bool,
        frozen: FrozenPrefix | None = None,
    ) -> PromptRender:
        """Build the exact model prompt, with message spans as best-effort attribution.

        A semantic frozen prefix is immutable physical history. Canonical rendering
        supplies only its continuation; it may never replace the sampled stream.
        """
        start = 0
        anchor = ''
        seam = ''
        frozen_text = ''
        templated_messages: list[str] = []
        message_token_ids: list[list[int]] = []
        if frozen is not None:
            if frozen.count > len(messages) or not frozen.text:
                raise ValueError('Frozen prefix does not match the request')
            start = frozen.count
            frozen_text = frozen.text
            templated_messages = list(frozen.templated_messages)
            message_token_ids = [list(ids) for ids in frozen.message_token_ids]

            if self._legacy_seam_compat:
                if not text.startswith(frozen.text):
                    self._observe_splice_divergence(frozen, text)
                anchor = self._render_chat(messages[: frozen.count], tools, add_generation_prompt=False)
                if not text.startswith(anchor):
                    raise ValueError('Tokenizer template does not expose a legacy-compatible continuation tail')
            elif text.startswith(frozen.text):
                # The canonical request already byte-extends the sampled stream.
                # Its direct suffix includes any closure that follows the frozen tail.
                anchor = frozen.text
            else:
                self._observe_splice_divergence(frozen, text)
                anchor = self._render_chat(messages[: frozen.count], tools, add_generation_prompt=False)
                if not text.startswith(anchor):
                    raise ValueError('Tokenizer template does not expose a legacy-compatible continuation tail')
                seam = self._locate_seam(frozen.text, anchor) or ''

        canonical_tail = text[len(anchor) :]
        continuation = seam + canonical_tail

        structured = None
        if frozen is None:
            structured = self._structured_prompt_spans(
                messages,
                tools,
                text=text,
                add_generation_prompt=add_generation_prompt,
            )
        if structured is not None:
            tail_spans, generation_prompt = structured
        else:
            rendered_inputs = None
            try:
                tail_spans, previous_text = self._discovered_spans(
                    messages,
                    tools,
                    start=start,
                    previous_text=anchor,
                )
                rendered_inputs = previous_text
                if not text.startswith(previous_text):
                    raise ValueError('Tokenizer template does not expose stable per-message prefixes')
                generation_prompt = text[len(previous_text) :]
                if tail_spans:
                    tail_spans[0] = seam + tail_spans[0]
                else:
                    generation_prompt = seam + generation_prompt
                if ''.join(tail_spans) + generation_prompt != continuation:
                    raise ValueError('Attributed spans do not reconstruct the physical continuation')
            except Exception:
                # Generic span ownership is bookkeeping. Preserve the exact
                # generation-prompt suffix when it can still be derived, then
                # conservatively attribute the fused input tail to its last message.
                tail_spans = [''] * (len(messages) - start)
                generation_prompt = continuation
                if tail_spans:
                    if not add_generation_prompt:
                        tail_spans[-1] = continuation
                        generation_prompt = ''
                    else:
                        if rendered_inputs is None:
                            try:
                                rendered_inputs = self._render_chat(
                                    messages,
                                    tools,
                                    add_generation_prompt=False,
                                )
                            except Exception:
                                rendered_inputs = None
                        if (
                            rendered_inputs is not None
                            and text.startswith(rendered_inputs)
                            and rendered_inputs.startswith(anchor)
                        ):
                            input_tail = seam + rendered_inputs[len(anchor) :]
                            candidate_generation_prompt = text[len(rendered_inputs) :]
                            if input_tail + candidate_generation_prompt == continuation:
                                tail_spans[-1] = input_tail
                                generation_prompt = candidate_generation_prompt

        tail_ids, generation_prompt_ids = self._canonical_span_ids(tail_spans, generation_prompt)
        templated_messages += tail_spans
        message_token_ids += tail_ids
        token_ids = [token for message_ids in message_token_ids for token in message_ids] + generation_prompt_ids
        return PromptRender(
            text=frozen_text + continuation,
            token_ids=token_ids,
            templated_messages=templated_messages,
            message_token_ids=message_token_ids,
            generation_prompt=generation_prompt,
            generation_prompt_ids=generation_prompt_ids,
        )

    def _discovered_spans(
        self,
        messages: list[dict],
        tools: list[dict] | None,
        *,
        start: int,
        previous_text: str,
    ) -> tuple[list[str], str]:
        """Per-message spans for ``messages[start:]`` by monotone prefix rendering.

        Returns the discovered spans and the final rendered prefix. A moved block
        closure is relocated within the discovered region only; any divergence that
        would reach ``previous_text`` itself fails loudly.
        """
        spans: list[str] = []
        for index in range(start, len(messages)):
            prefix = self._render_chat(
                messages[: index + 1],
                tools,
                add_generation_prompt=False,
            )
            if prefix.startswith(previous_text):
                spans.append(prefix[len(previous_text) :])
            else:
                # A closed block prefix is not a prefix of the full text until its
                # closure relocates past the content appended by later messages.
                closure = self._relocatable_closure(previous_text, prefix, spans)
                if closure is None:
                    raise ValueError('Tokenizer template does not expose stable per-message prefixes')
                spans[-1] = spans[-1][: -len(closure)]
                spans.append(prefix[len(previous_text) - len(closure) :])
            previous_text = prefix
        return spans, previous_text

    # Seam location bounds: how far back a template block closure may reach, and
    # how many shared trailing bytes anchor the frozen tail in the re-render.
    _SEAM_SEARCH_LIMIT = 64
    _SEAM_ANCHOR_LENGTH = 4

    def _locate_seam(self, frozen_text: str, rendered_prefix: str) -> str | None:
        """The block closure the re-render appends after the frozen tail.

        Normalization differences live inside a block, so the block's trailing
        bytes are shared between the sampled stream and the re-render. The
        smallest suffix of the re-rendered prefix whose removal re-exposes the
        frozen tail is the closure owed to the head of the tail.
        """
        for seam_length in range(min(self._SEAM_SEARCH_LIMIT, len(rendered_prefix)) + 1):
            body = rendered_prefix[: len(rendered_prefix) - seam_length]
            shared = 0
            for frozen_char, body_char in zip(reversed(frozen_text), reversed(body), strict=False):
                if frozen_char != body_char:
                    break
                shared += 1
            if shared >= min(len(frozen_text), self._SEAM_ANCHOR_LENGTH):
                return rendered_prefix[len(body) :]
        return None

    def _observe_splice_divergence(self, frozen: FrozenPrefix, text: str) -> None:
        """Record an expected physical divergence without changing splice behavior."""
        offset = 0
        for frozen_char, text_char in zip(frozen.text, text, strict=False):
            if frozen_char != text_char:
                break
            offset += 1
        logger.warning(
            'gateway frozen splice (No Worries): re-render diverges from the sampled stream at byte %d '
            '(sampled %r vs re-render %r); resending sampled bytes',
            offset,
            frozen.text[offset : offset + 24],
            text[offset : offset + 24],
        )

    @staticmethod
    def _relocatable_closure(
        previous_text: str,
        prefix: str,
        templated_messages: list[str],
    ) -> str | None:
        """Smallest block closure the template moved past newly appended content.

        Accepts only ``previous == stable + closure`` versus ``prefix == stable +
        appended + closure`` with the closure wholly owned by the immediately
        preceding message span; any other divergence stays a hard failure at the
        call site. The search starts at the structural minimum (the divergence
        point) so content bytes that merely coincide with the closure's tail are
        never pulled out of the preceding span.
        """
        if not templated_messages or len(prefix) <= len(previous_text):
            return None
        last_span = templated_messages[-1]
        divergence = 0
        for previous_char, prefix_char in zip(previous_text, prefix, strict=False):
            if previous_char != prefix_char:
                break
            divergence += 1
        for length in range(len(previous_text) - divergence, min(len(last_span), len(previous_text)) + 1):
            if length <= 0:
                continue
            closure = previous_text[len(previous_text) - length :]
            if not prefix.endswith(closure):
                continue
            stable_length = len(previous_text) - length
            if len(prefix) - length <= stable_length:
                continue
            if prefix.startswith(previous_text[:stable_length]):
                return closure
        return None

    def _structured_prompt_spans(
        self,
        messages: list[dict],
        tools: list[dict] | None,
        *,
        text: str,
        add_generation_prompt: bool,
    ) -> tuple[list[str], str] | None:
        parser = self._template_parser
        if parser is None or not messages:
            return None

        rendered_messages = parser.inject_tool_prompt(messages, tools) if tools is not None else messages
        if len(rendered_messages) != len(messages):
            return None

        templated_messages: list[str] = []
        for message in rendered_messages:
            render_message = getattr(parser, f'parse_{message["role"]}', None)
            if not callable(render_message):
                return None
            templated_messages.append(render_message(message))

        generation_prompt = getattr(parser, 'generation_prompt', None) if add_generation_prompt else ''
        if not isinstance(generation_prompt, str):
            return None
        # A parser that offers per-role spans has claimed the structured capability;
        # inconsistent spans are a parser bug and must not fall back to prefix probing.
        if not text.endswith(generation_prompt):
            raise ValueError('Template parser spans do not reconstruct the rendered prompt')
        message_text = text[: len(text) - len(generation_prompt)] if generation_prompt else text
        rendered_body = ''.join(templated_messages)
        if not message_text.endswith(rendered_body):
            raise ValueError('Template parser spans do not reconstruct the rendered prompt')
        leading_text = message_text[: len(message_text) - len(rendered_body)] if rendered_body else message_text
        templated_messages[0] = leading_text + templated_messages[0]
        return templated_messages, generation_prompt

    def _render_chat(
        self,
        messages: list[dict],
        tools: list[dict] | None,
        *,
        add_generation_prompt: bool,
    ) -> str:
        tokenizer = self._tokenizer
        assert tokenizer is not None
        render_messages = messages
        if any(
            isinstance(message.get('content'), list)
            and all(isinstance(block, dict) and block.get('type') == 'text' for block in message['content'])
            for message in messages
        ):
            render_messages = []
            for message in messages:
                content = message.get('content')
                if isinstance(content, list) and all(
                    isinstance(block, dict) and block.get('type') == 'text' for block in content
                ):
                    message = {
                        **message,
                        'content': ''.join(block.get('text', '') for block in content if isinstance(block, dict)),
                    }
                render_messages.append(message)
        if self._template_parser:
            msgs = render_messages
            if tools is not None:
                msgs = self._template_parser.inject_tool_prompt(msgs, tools)
            return self._template_parser.parse(
                msgs,
                add_generation_prompt=add_generation_prompt,
                is_first_msg=True,
            )
        return tokenizer.apply_chat_template(
            render_messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            tools=tools,
            **self._apply_chat_template_kwargs,
        )

    def _encode_text(self, text: str) -> list[int]:
        tokenizer = self._tokenizer
        assert tokenizer is not None
        return list(tokenizer.encode(text, add_special_tokens=False))

    def _canonical_span_ids(
        self,
        span_texts: list[str],
        generation_prompt: str,
    ) -> tuple[list[list[int]], list[int]]:
        """Tokenize the rendered spans canonically, then attribute ids back to spans.

        Independent per-span encodes can diverge from the whole-text tokenization at
        boundaries without a special-token barrier (e.g. consecutive tool responses the
        template folds into one block): BPE merges across the cut and the engine would
        see a non-canonical token stream. The stream the model sees is the hard
        invariant; span id-grouping is internal bookkeeping. So: encode the whole tail
        once, then split the ids at span text offsets. A token straddling a boundary is
        attributed to the earlier span — its boundary is then not id-aligned, which is
        acceptable because span boundaries carry no authority (FrozenPrefix docstring)
        and histories never split between messages the template fused into one block.
        """
        full = ''.join(span_texts) + generation_prompt
        canonical = self._encode_text(full)
        try:
            per_span = [self._encode_text(text) for text in span_texts]
            generation_ids = self._encode_text(generation_prompt)
            if [t for ids in per_span for t in ids] + generation_ids == canonical:
                return per_span, generation_ids

            tokenizer = self._tokenizer
            assert tokenizer is not None
            starts: list[int] | None = None
            try:
                enc = tokenizer(full, add_special_tokens=False, return_offsets_mapping=True)
                if list(enc['input_ids']) == canonical:
                    starts = [offset[0] for offset in enc['offset_mapping']]
            except Exception:
                starts = None
            if starts is None:
                # Fallback: byte-level BPE decodes each id to its exact text piece.
                starts = []
                pos = 0
                for token_id in canonical:
                    starts.append(pos)
                    pos += len(tokenizer.decode([token_id]))

            boundaries = []
            acc = 0
            for text in span_texts:
                acc += len(text)
                boundaries.append(acc)
            groups: list[list[int]] = [[] for _ in span_texts]
            generation_group: list[int] = []
            cursor = 0
            for token_id, start in zip(canonical, starts, strict=True):
                while cursor < len(boundaries) and start >= boundaries[cursor]:
                    cursor += 1
                (groups[cursor] if cursor < len(groups) else generation_group).append(token_id)
            return groups, generation_group
        except Exception:
            # Canonical encoding is authoritative; span grouping is bookkeeping.
            groups = [[] for _ in span_texts]
            if generation_prompt or not groups:
                return groups, canonical
            groups[-1] = canonical
            return groups, []

    async def _parse_tool_calls(
        self,
        token_ids: list[int],
        text: str,
        tools: list[dict] | None,
    ) -> tuple[Any, str, list[dict] | None]:
        """Parse tool calls from generated text/tokens."""
        if not self._tool_parser:
            return None, text, None
        extract = getattr(self._tool_parser, 'extract_tool_calls', None)
        if not callable(extract):
            raise RuntimeError('Gateway tool_parser does not expose extract_tool_calls')
        request = _ToolParseRequest(
            tools=tools,
            chat_template_kwargs=dict(self._apply_chat_template_kwargs),
        )
        parameters = inspect.signature(extract).parameters.values()
        accepts_request = any(
            parameter.name == 'request' or parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters
        )
        parsed = extract(token_ids, request=request) if accepts_request else extract(token_ids)
        if inspect.isawaitable(parsed):
            parsed = await parsed
        raw_tool = parsed[0] if isinstance(parsed, tuple) and parsed else None
        content, tool_calls = self._normalize_tool_parse_result(parsed, text)
        return raw_tool, content, tool_calls

    @staticmethod
    def _normalize_tool_parse_result(parsed: Any, fallback_text: str) -> tuple[str, list[dict] | None]:
        """Normalize parser output to (assistant_content, OpenAI tool_calls)."""
        content = fallback_text
        calls = parsed
        if isinstance(parsed, tuple):
            if len(parsed) >= 2:
                content = parsed[1]
            if len(parsed) >= 3:
                calls = parsed[2]

        if not calls:
            return content, None

        normalized = []
        for call in calls:
            if isinstance(call, dict):
                if 'function' in call:
                    normalized.append(call)
                    continue
                name = call.get('name', '')
                arguments = call.get('arguments', '')
            else:
                dumped = call.model_dump() if hasattr(call, 'model_dump') else {}
                name = dumped.get('name', getattr(call, 'name', ''))
                arguments = dumped.get('arguments', getattr(call, 'arguments', ''))

            normalized.append({
                'id': f'call_{uuid.uuid4().hex[:24]}',
                'type': 'function',
                'function': {
                    'name': name,
                    'arguments': arguments,
                },
            })

        return content, normalized

    # ---- Message-level path ----

    async def _message_level_completion(
        self,
        request_id: str,
        messages: list[dict],
        request_kwargs: dict[str, Any],
        request_meta: dict[str, Any] | None = None,
    ) -> ProcessorResult:
        """Forward messages to backend. Backend returns a parsed message."""
        inference_started = time.perf_counter()
        result: BackendResult = await self._backend.create_completion(
            request_id,
            messages=messages,
            request_kwargs=request_kwargs,
            request_meta=request_meta,
        )
        inference_time = time.perf_counter() - inference_started
        if result.message is None:
            raise RuntimeError('Message-level backend returned no message')

        extra_fields = dict(result.extra_fields)
        return ProcessorResult(
            message=result.message,
            inference_time=inference_time,
            usage=result.usage,
            stop_reason=result.stop_reason,
            id=result.id,
            created=result.created,
            model=result.model,
            object=result.object,
            system_fingerprint=result.system_fingerprint,
            service_tier=result.service_tier,
            choice_logprobs=result.choice_logprobs,
            choice_provider_specific_fields=result.choice_provider_specific_fields,
            extra_fields=extra_fields,
        )
