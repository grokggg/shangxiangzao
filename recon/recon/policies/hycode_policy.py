"""Direct event-fact policy for the HyCode 0.1.8 wire protocol."""

from __future__ import annotations

import json
import re

from dataclasses import replace
from typing import Any

from nexus.gateway.scaffold_policy import (
    AnchoredIdentity,
    BoundaryFact,
    ChildPromptCandidate,
    DeliveryReceipt,
    GenericScaffoldPolicy,
    InputMessage,
    NotificationDelivery,
    OutputMessage,
    RequestEvent,
    ResponseEvent,
    ResponsePolicy,
    RetrievalIntent,
    SidechainMembership,
    SpawnBinding,
    SpawnIntent,
    protocol_message_identity,
)


HYCODE_SIGNALS_VERSION = '0.1.8'

COMPACT_SUMMARY_MARKER = 'This session is being continued from a previous conversation that ran out of context'
COMPACT_TRIGGER_MARKER = 'CRITICAL: Respond with TEXT ONLY. Do NOT call any tools.'
ASYNC_LAUNCH_ACK = 'Async agent launched successfully.'

_AGENT_ID_LINE = re.compile(r'(?m)^agentId:\s*([^\s(<]+)\s+\((?:internal ID\b|use SendMessage\b)')
_TASK_NOTIFICATION = re.compile(r'<task-notification>(.*?)</task-notification>', re.DOTALL)
_TERMINAL_TASK_STATUSES = frozenset({'completed', 'failed', 'killed'})

# A small wire grammar, not a second policy DSL. All required fragments must
# occur in the selected message; ``line`` allows a flattened system prompt.
_UTILITY_MARKERS = (
    ('session_title', 'system', 'line', ('Generate a concise, sentence-case title (3-7 words)',)),
    (
        'session_memory',
        'user',
        'last',
        (
            'IMPORTANT: This message and these instructions are NOT part of the actual user conversation.',
            '"note-taking"',
            '"session notes extraction"',
        ),
    ),
    (
        'magic_docs',
        'user',
        'last',
        (
            'IMPORTANT: This message and these instructions are NOT part of the actual user conversation.',
            '"documentation updates"',
            '"magic docs"',
        ),
    ),
)


class HyCodeScaffoldPolicy(GenericScaffoldPolicy):
    """Translate pinned HyCode wire signals into scaffold-neutral facts."""

    response_policy = ResponsePolicy.PREFER_EXTENDED_ELSE_LAST

    def __init__(self) -> None:
        super().__init__(policy_id='hycode')

    @staticmethod
    def _text_blocks(content: Any) -> list[str]:
        if isinstance(content, str):
            return [content]
        if not isinstance(content, list):
            return []
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.extend(HyCodeScaffoldPolicy._text_blocks(item.get('text')))
                parts.extend(HyCodeScaffoldPolicy._text_blocks(item.get('content')))
        return parts

    @classmethod
    def _text(cls, message: dict[str, Any]) -> str:
        return '\n'.join(cls._text_blocks(message.get('content')))

    @staticmethod
    def _message_identity(message: dict[str, Any]):
        if message.get('role') != 'assistant' or not message.get('tool_calls'):
            return protocol_message_identity(message)
        normalized = dict(message)
        if normalized.get('content') in (None, ''):
            normalized['content'] = None
        return protocol_message_identity(normalized)

    @staticmethod
    def _argument(function: dict[str, Any], key: str) -> str:
        try:
            arguments = json.loads(function.get('arguments') or '{}')
        except (TypeError, ValueError):
            return ''
        value = arguments.get(key) if isinstance(arguments, dict) else None
        return value if isinstance(value, str) else ''

    @staticmethod
    def _tag(text: str, name: str) -> str | None:
        match = re.search(rf'<{re.escape(name)}>(.*?)</{re.escape(name)}>', text, re.DOTALL)
        return match.group(1).strip() if match is not None else None

    @classmethod
    def _utility_membership(cls, messages: tuple[dict[str, Any], ...]) -> SidechainMembership | None:
        for kind, role, position, required in _UTILITY_MARKERS:
            candidates = messages[-1:] if position == 'last' else messages
            for message in candidates:
                if message.get('role') != role:
                    continue
                text = cls._text(message)
                if position == 'line':
                    matched = any(all(fragment in line for fragment in required) for line in text.splitlines())
                else:
                    matched = all(fragment in text for fragment in required)
                if matched:
                    return SidechainMembership(
                        chain_id='',
                        kind=kind,
                        detector=f'marker:hycode-{HYCODE_SIGNALS_VERSION}',
                    )
        return None

    @classmethod
    def _message_facts(cls, messages, anchors):
        spawns = []
        bindings = []
        retrievals = []
        receipts = []
        notifications = []
        tool_names: dict[str, str] = {}

        for message, anchor in zip(messages, anchors, strict=True):
            for call in message.get('tool_calls') or []:
                function = call.get('function') or {}
                call_id = call.get('id')
                name = function.get('name')
                if not isinstance(call_id, str) or not isinstance(name, str):
                    continue
                tool_names[call_id] = name
                if name in ('Agent', 'Task'):
                    prompt = ' '.join(cls._argument(function, 'prompt').split())
                    if prompt:
                        spawns.append(SpawnIntent(anchor, call_id, prompt))
                elif name == 'TaskOutput':
                    agent_id = cls._argument(function, 'task_id')
                    if agent_id:
                        retrievals.append(RetrievalIntent(anchor, call_id, agent_id))

        for message, anchor in zip(messages, anchors, strict=True):
            text = cls._text(message)
            call_id = message.get('tool_call_id')
            tool_name = tool_names.get(call_id) if isinstance(call_id, str) else None
            if message.get('role') == 'tool' and isinstance(call_id, str):
                if tool_name in ('Agent', 'Task'):
                    receipts.append(DeliveryReceipt(anchor, call_id, not text.lstrip().startswith(ASYNC_LAUNCH_ACK)))
                    match = _AGENT_ID_LINE.search(text)
                    if match is not None:
                        bindings.append(SpawnBinding(anchor, call_id, match.group(1)))
                elif tool_name == 'TaskOutput':
                    receipts.append(DeliveryReceipt(anchor, call_id, cls._tag(text, 'retrieval_status') == 'success'))

            if message.get('role') != 'user':
                continue
            for body in _TASK_NOTIFICATION.findall(text):
                if cls._tag(body, 'task-type') is not None:
                    continue
                agent_id = cls._tag(body, 'task-id')
                status = cls._tag(body, 'status')
                if agent_id is None or status not in _TERMINAL_TASK_STATUSES:
                    continue
                tool_use_id = cls._tag(body, 'tool-use-id')
                if tool_use_id is not None and tool_names.get(tool_use_id) not in (None, 'Agent', 'Task'):
                    continue
                notifications.append(NotificationDelivery(anchor, agent_id))
                if tool_use_id is not None:
                    bindings.append(SpawnBinding(anchor, tool_use_id, agent_id))

        return tuple(spawns), tuple(bindings), tuple(retrievals), tuple(receipts), tuple(notifications)

    @classmethod
    def _child_prompts(cls, messages: tuple[dict[str, Any], ...]):
        for index, message in enumerate(messages):
            if message.get('role') != 'user':
                continue
            candidates = []
            for block in cls._text_blocks(message.get('content')):
                if block.lstrip().startswith('<system-reminder>'):
                    continue
                prompt = ' '.join(block.split())
                if prompt:
                    candidates.append(ChildPromptCandidate(InputMessage(index), prompt, len(candidates)))
            if candidates:
                return tuple(candidates)
        return ()

    @classmethod
    def _boundary_facts(cls, messages: tuple[dict[str, Any], ...]):
        user_indexes = [index for index, message in enumerate(messages) if message.get('role') == 'user']
        if not user_indexes:
            return ()
        facts = []
        first, last = user_indexes[0], user_indexes[-1]
        if any(
            block.lstrip().startswith(COMPACT_SUMMARY_MARKER)
            for block in cls._text_blocks(messages[first].get('content'))
        ):
            facts.append(
                BoundaryFact(
                    'hycode_compaction_resume',
                    InputMessage(first),
                    'compaction_resume',
                    'compaction',
                )
            )
        if any(
            block.lstrip().startswith(COMPACT_TRIGGER_MARKER)
            for block in cls._text_blocks(messages[last].get('content'))
        ):
            facts.append(
                BoundaryFact(
                    'hycode_summary_inference',
                    InputMessage(last),
                    'summary_inference',
                    'compaction',
                    at_request_edge=True,
                )
            )
        return tuple(facts)

    def inspect_event(self, event):
        facts = super().inspect_event(event)
        request = event.request if isinstance(event, ResponseEvent) else event
        if not isinstance(request, RequestEvent):
            return facts

        meta = request.request_meta or {}
        agent_id = self._optional_str(meta.get('agent_id'))
        membership = self._utility_membership(request.messages)
        if membership is None and agent_id is not None:
            membership = SidechainMembership(
                chain_id=agent_id,
                kind='subagent',
                detector='header:agent_id',
            )
        chain_key = 'main' if membership is None else (membership.chain_id or f'utility:{membership.kind}')
        context_key = (self.policy_id, chain_key, membership is not None)

        if isinstance(event, ResponseEvent):
            messages = (event.message,)
            anchors = (OutputMessage(),)
            boundary_facts = ()
            child_prompts = ()
        else:
            messages = event.messages
            anchors = tuple(InputMessage(index) for index in range(len(messages)))
            boundary_facts = self._boundary_facts(messages)
            child_prompts = (
                self._child_prompts(messages) if membership is not None and membership.kind == 'subagent' else ()
            )
        spawns, bindings, retrievals, receipts, notifications = self._message_facts(messages, anchors)
        timeline = dict(facts.timeline)
        if timeline:
            timeline['is_sidechain'] = membership is not None
        return replace(
            facts,
            message_identities=tuple(
                AnchoredIdentity(anchor, self._message_identity(message))
                for message, anchor in zip(messages, anchors, strict=True)
            ),
            context_key=context_key,
            membership=membership,
            spawn_intents=spawns,
            spawn_bindings=bindings,
            retrieval_intents=retrievals,
            delivery_receipts=receipts,
            notification_deliveries=notifications,
            boundary_facts=boundary_facts,
            child_prompts=child_prompts,
            timeline=tuple(timeline.items()),
        )


HYCODE_POLICY = HyCodeScaffoldPolicy()
