"""Stateless scaffold event inspection for direct Trajectory assembly."""

from __future__ import annotations

import json

from collections.abc import Hashable
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from typing import Any, Generic, Protocol, TypeAlias, TypeVar, runtime_checkable

from nexus.gateway.trajectory import NodeId


class ResponsePolicy(str, Enum):
    KEEP_ALL = 'keep_all'
    PREFER_EXTENDED_ELSE_LAST = 'prefer_extended_else_last'


@dataclass(frozen=True)
class InputMessage:
    index: int


@dataclass(frozen=True)
class OutputMessage:
    pass


@dataclass(frozen=True)
class ObservationMessage:
    pass


UnboundAnchor: TypeAlias = InputMessage | OutputMessage | ObservationMessage
AnchorT = TypeVar('AnchorT')


@dataclass(frozen=True)
class BoundAnchor:
    node_id: NodeId


@dataclass(frozen=True)
class AnchoredIdentity:
    anchor: UnboundAnchor
    value: Hashable


@dataclass(frozen=True)
class SidechainMembership:
    chain_id: str
    kind: str = 'sidechain'
    parent_chain_id: str | None = None
    detector: str | None = None
    attributes: tuple[tuple[str, Hashable], ...] = ()


@dataclass(frozen=True)
class SpawnIntent(Generic[AnchorT]):
    anchor: AnchorT
    tool_use_id: str
    child_prompt: str


@dataclass(frozen=True)
class SpawnBinding(Generic[AnchorT]):
    """A scaffold-declared mapping from one spawn call to its runtime agent id."""

    anchor: AnchorT
    tool_use_id: str
    agent_id: str


@dataclass(frozen=True)
class RetrievalIntent(Generic[AnchorT]):
    anchor: AnchorT
    tool_use_id: str
    agent_id: str


@dataclass(frozen=True)
class DeliveryReceipt(Generic[AnchorT]):
    anchor: AnchorT
    tool_use_id: str
    delivered: bool


@dataclass(frozen=True)
class NotificationDelivery(Generic[AnchorT]):
    anchor: AnchorT
    agent_id: str


@dataclass(frozen=True)
class ChildPromptCandidate(Generic[AnchorT]):
    anchor: AnchorT
    value: str
    order: int


@dataclass(frozen=True)
class DeclaredChainClaim:
    chain_id: str
    kind: str = 'sidechain'
    parent_chain_id: str | None = None


@dataclass(frozen=True)
class ParentClaim:
    parent_chain_id: str


@dataclass(frozen=True)
class BoundaryFact:
    rule_name: str
    anchor: UnboundAnchor
    segment_kind: str
    cause: str
    from_request_context: bool = False
    at_request_edge: bool = False


@dataclass(frozen=True)
class RequestEvent:
    messages: tuple[dict[str, Any], ...]
    tools: list[dict[str, Any]] | None
    request_meta: dict[str, Any] | None


@dataclass(frozen=True)
class ResponseEvent:
    request: RequestEvent
    message: dict[str, Any]
    response_meta: dict[str, Any] | None = None
    start_time: float | None = None
    end_time: float | None = None
    status: str | None = None


@dataclass(frozen=True)
class ObservationEvent:
    message: dict[str, Any]
    parent_messages: tuple[dict[str, Any], ...] | None
    tools: list[dict[str, Any]] | None
    request_meta: dict[str, Any] | None
    start_time: float | None = None
    end_time: float | None = None
    status: str | None = None


@dataclass(frozen=True)
class EventFacts:
    message_identities: tuple[AnchoredIdentity, ...] = ()
    tools_identity: Hashable = None
    context_key: Hashable = 'main'
    request_id: str | None = None
    agent_id: str | None = None
    membership: SidechainMembership | None = None
    spawn_intents: tuple[SpawnIntent[UnboundAnchor], ...] = ()
    spawn_bindings: tuple[SpawnBinding[UnboundAnchor], ...] = ()
    retrieval_intents: tuple[RetrievalIntent[UnboundAnchor], ...] = ()
    delivery_receipts: tuple[DeliveryReceipt[UnboundAnchor], ...] = ()
    notification_deliveries: tuple[NotificationDelivery[UnboundAnchor], ...] = ()
    declared_claim: DeclaredChainClaim | None = None
    parent_claim: ParentClaim | None = None
    boundary_facts: tuple[BoundaryFact, ...] = ()
    child_prompts: tuple[ChildPromptCandidate[UnboundAnchor], ...] = ()
    timeline: tuple[tuple[str, Hashable], ...] = ()
    iteration: int | None = None
    retry: int | None = None
    sequence: int | None = None


@dataclass(frozen=True)
class ScaffoldContext:
    """Request-scoped policy facts that may be reused across wire events."""

    request_meta: dict[str, Any] | None
    context_key: Hashable
    request_id: str | None = None
    agent_id: str | None = None
    membership: SidechainMembership | None = None
    declared_claim: DeclaredChainClaim | None = None
    parent_claim: ParentClaim | None = None
    iteration: int | None = None
    retry: int | None = None
    sequence: int | None = None


@runtime_checkable
class ScaffoldPolicy(Protocol):
    policy_id: str
    response_policy: ResponsePolicy
    block_terminator_ids: tuple[int, ...]
    message_key: str
    tools_key: str

    def inspect_event(
        self,
        event: RequestEvent | ResponseEvent | ObservationEvent,
    ) -> EventFacts: ...


def _freeze_value(value: Any) -> Hashable:
    if isinstance(value, dict):
        return tuple((key, _freeze_value(item)) for key, item in sorted(value.items()))
    if isinstance(value, list):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, Hashable):
        return value
    return repr(value)


def _exact_value(value: Any) -> Any:
    if hasattr(value, 'model_dump'):
        return _exact_value(value.model_dump(exclude_none=False))
    if isinstance(value, dict):
        return {key: _exact_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_exact_value(item) for item in value]
    return value


def _canonical_arguments(arguments: Any) -> Any:
    if isinstance(arguments, str):
        try:
            return _canonical_arguments(json.loads(arguments))
        except (json.JSONDecodeError, ValueError):
            return arguments
    if isinstance(arguments, dict):
        return {key: _canonical_arguments(arguments[key]) for key in sorted(arguments)}
    if isinstance(arguments, list):
        return [_canonical_arguments(item) for item in arguments]
    return arguments


def protocol_message_identity(message: Any) -> Hashable:
    raw = _exact_value(message)
    if not isinstance(raw, dict):
        raw = dict(raw)
    normalized = {key: raw[key] for key in ('role', 'content', 'tool_calls', 'tool_call_id', 'name') if key in raw}
    if isinstance(normalized.get('tool_calls'), list):
        projected_calls = []
        for tool_call in normalized['tool_calls']:
            if not isinstance(tool_call, dict):
                projected_calls.append(tool_call)
                continue
            projected = {key: tool_call[key] for key in ('id', 'type', 'function') if key in tool_call}
            function = projected.get('function')
            if isinstance(function, dict):
                projected_function = {key: function[key] for key in ('name', 'arguments') if key in function}
                if 'arguments' in projected_function:
                    projected_function['arguments'] = _canonical_arguments(projected_function['arguments'])
                projected['function'] = projected_function
            projected_calls.append(projected)
        normalized['tool_calls'] = projected_calls
    return _freeze_value(normalized)


class GenericScaffoldPolicy:
    """Strict protocol-message identity with no scaffold semantics."""

    response_policy = ResponsePolicy.KEEP_ALL
    block_terminator_ids: tuple[int, ...] = ()
    message_key = 'protocol-message-v1'
    tools_key = 'exact-tools-v1'

    def __init__(self, *, policy_id: str = 'generic'):
        self.policy_id = policy_id

    def inspect_event(self, event: RequestEvent | ResponseEvent | ObservationEvent) -> EventFacts:
        if isinstance(event, RequestEvent):
            identities = tuple(
                AnchoredIdentity(InputMessage(index), protocol_message_identity(message))
                for index, message in enumerate(event.messages)
            )
            tools = event.tools
            request_meta = event.request_meta
            timeline = {}
        elif isinstance(event, ResponseEvent):
            identities = (AnchoredIdentity(OutputMessage(), protocol_message_identity(event.message)),)
            tools = event.request.tools
            request_meta = event.request.request_meta
            timeline = self._timeline(
                'llm_request',
                request_meta,
                event.start_time,
                event.end_time,
                event.status,
                event.response_meta,
            )
        else:
            identities = (AnchoredIdentity(ObservationMessage(), protocol_message_identity(event.message)),)
            tools = event.tools
            request_meta = event.request_meta
            timeline = self._timeline(
                'observation',
                request_meta,
                event.start_time,
                event.end_time,
                event.status,
                {'type': 'agent_observation'},
            )
        meta = request_meta or {}
        return EventFacts(
            message_identities=identities,
            tools_identity=_freeze_value(_exact_value(tools)),
            context_key=(self.policy_id, 'main', False),
            request_id=self._optional_str(meta.get('request_id')),
            agent_id=self._optional_str(meta.get('agent_id')),
            timeline=tuple(timeline.items()),
            iteration=self._optional_int(meta.get('iteration')),
            retry=self._optional_int(meta.get('retry')),
            sequence=self._optional_int(meta.get('sequence') or meta.get('seq')),
        )

    @staticmethod
    def _optional_int(value: Any) -> int | None:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _optional_float(value: Any) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _optional_str(value: Any) -> str | None:
        return None if value is None else str(value)

    @staticmethod
    def _copy_value(value: Any) -> Any:
        if hasattr(value, 'model_dump'):
            return GenericScaffoldPolicy._copy_value(value.model_dump(exclude_none=True))
        if isinstance(value, dict):
            return {key: GenericScaffoldPolicy._copy_value(item) for key, item in value.items() if item is not None}
        if isinstance(value, list):
            return [GenericScaffoldPolicy._copy_value(item) for item in value]
        return deepcopy(value)

    @staticmethod
    def _copy_attributes(value: Any) -> dict[str, Any]:
        if hasattr(value, 'model_dump'):
            value = value.model_dump(exclude_none=True)
        if not isinstance(value, dict):
            return {}
        return GenericScaffoldPolicy._copy_value(value)

    @staticmethod
    def _timeline(
        event_type: str,
        request_meta: dict[str, Any] | None,
        start_time: float | None,
        end_time: float | None,
        status: str | None,
        attributes: Any,
    ) -> dict[str, Any]:
        meta = request_meta or {}
        event_start = GenericScaffoldPolicy._optional_float(meta.get('start_time', start_time))
        event_end = GenericScaffoldPolicy._optional_float(meta.get('end_time', end_time))
        duration = GenericScaffoldPolicy._optional_float(meta.get('duration'))
        if duration is None and event_start is not None and event_end is not None:
            duration = event_end - event_start
        merged_attributes = GenericScaffoldPolicy._copy_attributes(meta.get('timeline_attributes'))
        merged_attributes.update(GenericScaffoldPolicy._copy_attributes(meta.get('attributes')))
        merged_attributes.update(GenericScaffoldPolicy._copy_attributes(attributes))
        request_id = GenericScaffoldPolicy._optional_str(meta.get('request_id'))
        agent_id = GenericScaffoldPolicy._optional_str(meta.get('agent_id'))
        values = {
            'event_type': str(meta.get('event_type') or event_type),
            'request_id': request_id,
            'agent_id': agent_id,
            'parent_agent_id': GenericScaffoldPolicy._optional_str(
                meta.get('parent_agent_id') or meta.get('parent_id')
            ),
            'is_sidechain': False,
            'start_time': event_start,
            'end_time': event_end,
            'duration': duration,
            'status': GenericScaffoldPolicy._optional_str(meta.get('status') or status),
            'span_id': GenericScaffoldPolicy._optional_str(meta.get('span_id')),
            'parent_span_id': GenericScaffoldPolicy._optional_str(meta.get('parent_span_id')),
            'iteration': GenericScaffoldPolicy._optional_int(meta.get('iteration')),
            'retry': GenericScaffoldPolicy._optional_int(meta.get('retry')),
            'sequence': GenericScaffoldPolicy._optional_int(meta.get('sequence') or meta.get('seq')),
            'attributes': merged_attributes or None,
        }
        has_facts = bool(merged_attributes or request_id or agent_id or meta.get('event_type'))
        has_timing = event_start is not None or event_end is not None or duration is not None
        if not has_facts and not has_timing:
            return {}
        return {key: value for key, value in values.items() if value is not None}


_SCAFFOLD_POLICIES: dict[str, ScaffoldPolicy] = {}


def register_scaffold_policy(policy: ScaffoldPolicy) -> None:
    policy_id = policy.policy_id
    if policy_id in _SCAFFOLD_POLICIES:
        raise ValueError(f'Duplicate gateway scaffold_policy={policy_id!r}')
    _SCAFFOLD_POLICIES[policy_id] = policy


def build_scaffold_policy(policy: ScaffoldPolicy | str | None) -> ScaffoldPolicy:
    if policy is None:
        raise ValueError('Gateway requires an explicit scaffold_policy')
    if isinstance(policy, str):
        import nexus.gateway.policies  # noqa: F401  (built-in policy registration)

        resolved = _SCAFFOLD_POLICIES.get(policy)
        if resolved is None:
            raise ValueError(f'Unknown gateway scaffold_policy={policy!r}')
        return resolved
    if not isinstance(policy, ScaffoldPolicy):
        raise TypeError(f'Invalid gateway scaffold_policy: {policy!r}')
    return policy
