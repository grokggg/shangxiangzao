"""Record Gateway events directly into one live public Trajectory."""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import replace
from typing import Any, Literal

from nexus.gateway.runtime_state import (
    NodeRuntimeFacts,
    PendingBoundary,
    RuntimeFactOccurrence,
    RuntimeState,
)
from nexus.gateway.scaffold_policy import (
    BoundAnchor,
    BoundaryFact,
    EventFacts,
    InputMessage,
    ObservationEvent,
    ObservationMessage,
    OutputMessage,
    RequestEvent,
    ResponseEvent,
    ScaffoldContext,
    ScaffoldPolicy,
    SidechainMembership,
    build_scaffold_policy,
)
from nexus.gateway.trajectory import (
    InvocationFacts,
    NodeId,
    ProviderFacts,
    SegmentBoundary,
    TokenAccount,
    Trajectory,
    TrajectoryNode,
)
from nexus.gateway.types import FrozenPrefix, PromptRender
from nexus.gateway.wire import copy_message, copy_messages, copy_value, copy_wire_value


_ANCHORED_FACT_FIELDS = (
    'spawn_intents',
    'spawn_bindings',
    'retrieval_intents',
    'delivery_receipts',
    'notification_deliveries',
    'child_prompts',
)


class TrajectoryAssembler:
    """Own and mutate one live Trajectory plus private scalar runtime state."""

    def __init__(
        self,
        scaffold_policy: ScaffoldPolicy | str | None = None,
        *,
        session_id: str = '',
    ):
        self._policy = build_scaffold_policy(scaffold_policy)
        self.trajectory = Trajectory(session_id=session_id, state='active')
        self._runtime_state = RuntimeState()

    def extract_context(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        request_meta: dict[str, Any] | None = None,
    ) -> ScaffoldContext:
        facts = self._inspect_request(messages, tools, request_meta)
        return ScaffoldContext(
            request_meta=copy_wire_value(request_meta),
            context_key=facts.context_key,
            request_id=facts.request_id,
            agent_id=facts.agent_id,
            membership=facts.membership,
            declared_claim=facts.declared_claim,
            parent_claim=facts.parent_claim,
            iteration=facts.iteration,
            retry=facts.retry,
            sequence=facts.sequence,
        )

    @staticmethod
    def context_key(context: ScaffoldContext) -> Hashable:
        return context.context_key

    def frozen_prefix(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        *,
        context: ScaffoldContext | None = None,
        request_meta: dict[str, Any] | None = None,
    ) -> FrozenPrefix | None:
        request_event = self._request_event(messages, tools, request_meta)
        request_facts = self._event_facts(request_event, context)
        cursor_id = self._runtime_state.lookup_cache.last_llm_by_context.get(request_facts.context_key)
        if cursor_id is None:
            return None
        cursor_facts = self._runtime_state.facts.by_node[cursor_id]
        if cursor_facts.tools_identity != request_facts.tools_identity:
            return None
        identities = self._input_identities(request_facts, len(messages))
        matched: list[TrajectoryNode] = []
        for node_id, identity in zip(self._node_path(cursor_id), identities, strict=False):
            node = self.trajectory.nodes[node_id]
            account = node.token_account
            if (
                node.templated_message is None
                or account is None
                or account.token_ids is None
                or self._runtime_state.facts.by_node[node_id].identity != identity
            ):
                break
            matched.append(node)
        if not matched:
            return None
        frozen = self._frozen_prefix(matched)
        return frozen if frozen.text else None

    @staticmethod
    def _frozen_prefix(nodes: list[TrajectoryNode]) -> FrozenPrefix:
        spans: list[str] = []
        token_ids: list[tuple[int, ...]] = []
        for node in nodes:
            account = node.token_account
            assert node.templated_message is not None
            assert account is not None
            assert account.token_ids is not None
            spans.append(node.templated_message)
            token_ids.append(tuple(account.token_ids))
        return FrozenPrefix(
            count=len(nodes),
            text=''.join(spans),
            templated_messages=tuple(spans),
            message_token_ids=tuple(token_ids),
        )

    def record_completion(
        self,
        messages: list[dict],
        output_message: dict,
        request_kwargs: dict[str, Any] | None = None,
        stop_reason: str | None = None,
        token_training: dict[str, Any] | None = None,
        response_meta: dict[str, Any] | None = None,
        completion_text: str | None = None,
        prompt_render: PromptRender | None = None,
        context: ScaffoldContext | None = None,
        request_meta: dict[str, Any] | None = None,
        start_time: float | None = None,
        end_time: float | None = None,
    ) -> None:
        input_messages = copy_messages(messages)
        request_sequence = self._runtime_state.next_request_sequence
        self._runtime_state.next_request_sequence += 1
        tools = (request_kwargs or {}).get('tools')
        request_event = self._request_event(input_messages, tools, request_meta)
        request_facts = self._event_facts(request_event, context)
        normalized_training = self._normalize_token_training(token_training) if token_training is not None else None
        self._validate_prompt_inputs(normalized_training, prompt_render, completion_text)

        current_id = self._find_or_create_path(
            input_messages,
            request_facts,
            request_sequence,
            prompt_render,
        )

        glue_ids: list[int] = []
        if normalized_training is not None:
            assert prompt_render is not None
            glue_ids = list(prompt_render.generation_prompt_ids)
            if self._input_token_ids(current_id) + glue_ids != normalized_training['prompt_ids']:
                raise ValueError('Assembly path does not reconstruct the model prompt IDs')

        output = copy_message(output_message)
        output.setdefault('role', 'assistant')
        response_attributes = {'type': 'agent_inference'}
        response_attributes.update({
            key: copy_value(value) for key, value in (response_meta or {}).items() if value is not None
        })
        response_facts = self._event_facts(
            ResponseEvent(
                request=request_event,
                message=output,
                response_meta=response_attributes,
                start_time=start_time,
                end_time=end_time,
                status='OK',
            ),
            context,
        )
        boundary = self._request_boundary(request_facts, current_id)
        boundary = self._merge_boundaries(boundary, self._output_boundary(response_facts))
        boundary = self._consume_pending_boundary(boundary)
        physical_boundary = self._physical_boundary(
            request_facts=request_facts,
            current_id=current_id,
            token_training=normalized_training,
        )
        boundary = self._merge_boundaries(boundary, physical_boundary)

        output_identity = self._identity_for(response_facts, OutputMessage())
        output_id = self._new_node(
            parent_id=current_id,
            original_message=output,
            generated_by='llm',
            identity=output_identity,
            event_facts=response_facts,
            templated_message=(prompt_render.generation_prompt + (completion_text or ''))
            if prompt_render is not None
            else completion_text,
        )
        output_node = self.trajectory.nodes[output_id]
        output_node.request_message_overrides = {
            node_id: copy_message(message)
            for node_id, message in zip(self._node_path(current_id), input_messages, strict=True)
            if message != self.trajectory.nodes[node_id].original_message
        }
        wire_kwargs = copy_wire_value(request_kwargs or {})
        output_node.request_kwargs = wire_kwargs or None
        output_node.provider_facts = self._provider_facts(output, stop_reason, response_meta)
        output_node.timeline = [dict(response_facts.timeline)] if response_facts.timeline else []
        output_node.invocation = self._invocation(response_facts)
        if boundary is not None:
            output_node.segment_boundary = copy_value(boundary.boundary)
            output_facts = self._runtime_state.facts.by_node[output_id]
            self._runtime_state.facts.by_node[output_id] = replace(
                output_facts,
                boundary_claims=tuple(
                    (rule_name, output_id if anchor == OutputMessage() else anchor)
                    for rule_name, anchor in boundary.claims
                ),
            )
        if normalized_training is not None:
            output_node.token_account = self._token_account(glue_ids, normalized_training)
        self._update_cursors(output_id)

    def record_observation(
        self,
        content: str | None = None,
        token_ids: list[int] | None = None,
        role: str = 'tool',
        parent_messages: list[dict] | None = None,
        tools: list[dict] | None = None,
        context: ScaffoldContext | None = None,
        request_meta: dict[str, Any] | None = None,
        start_time: float | None = None,
        end_time: float | None = None,
        status: str | None = None,
        message: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        if token_ids:
            raise ValueError('observation token IDs are derived from the next rendered prompt')
        if message is None:
            if content is None:
                raise ValueError('record_observation requires content or message')
            observation = {'role': role, 'content': content}
        else:
            observation = copy_message(message)
            if content is not None and observation.get('content') != content:
                raise ValueError('content differs from the explicit observation message')
            observation.setdefault('role', role)

        observation_event = ObservationEvent(
            message=observation,
            parent_messages=tuple(copy_messages(parent_messages)) if parent_messages is not None else None,
            tools=copy_wire_value(tools),
            request_meta=copy_wire_value(request_meta),
            start_time=start_time,
            end_time=end_time,
            status=status,
        )
        if context is None and parent_messages is None and not request_meta:
            observation_facts = self._contextless_observation_facts(observation_event)
        else:
            observation_facts = self._event_facts(observation_event, context)
        if parent_messages is not None:
            parent_event = self._request_event(parent_messages, tools, request_meta)
            parent_facts = self._event_facts(parent_event, context)
            parent_id = self._find_path_node(parent_facts, len(parent_messages))
            if parent_id is None:
                raise ValueError('parent_messages do not match this trajectory')
        elif context is None and request_meta:
            parent_id = self._runtime_state.lookup_cache.last_node_by_context.get(observation_facts.context_key)
            if parent_id is None:
                raise ValueError('no trajectory path matches the observation context')
        elif context is None and not request_meta:
            parent_id = self._runtime_state.lookup_cache.last_node
        else:
            parent_id = self._runtime_state.lookup_cache.last_node_by_context.get(observation_facts.context_key)
            if parent_id is None:
                raise ValueError('no trajectory path matches the observation context')

        identity = self._identity_for(observation_facts, ObservationMessage())
        candidates = self._find_children(parent_id, identity)
        if candidates:
            observation_id = candidates[0]
            self._merge_node_facts(observation_id, observation_facts)
        else:
            observation_id = self._new_node(
                parent_id=parent_id,
                original_message=observation,
                generated_by='observation',
                identity=identity,
                event_facts=observation_facts,
            )
        observation_node = self.trajectory.nodes[observation_id]
        observation_node.generated_by = 'observation'
        if observation_facts.timeline:
            observation_node.timeline.append(dict(observation_facts.timeline))
        self._update_cursors(observation_id)
        return self._wire_messages_for_node(parent_id) + [copy_message(observation)]

    def finalize_segment(self) -> None:
        self._runtime_state.pending_boundary = PendingBoundary(
            SegmentBoundary(starts_new_segment=True, causes=('explicit',))
        )

    def freeze(
        self,
        *,
        session_id: str | None = None,
        state: str | None = None,
        session_timeline: list[dict[str, Any]] | None = None,
        latest_messages: list[dict[str, Any]] | None = None,
        termination_reason: str | None = None,
        metrics: dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> Trajectory:
        from nexus.gateway.trajectory_freeze import freeze_trajectory

        return freeze_trajectory(
            self.trajectory,
            self._runtime_state,
            self._policy.response_policy,
            session_id=session_id,
            state=state,
            session_timeline=session_timeline,
            latest_messages=latest_messages,
            termination_reason=termination_reason,
            metrics=metrics,
            extra=extra,
        )

    def _inspect_request(
        self,
        messages: list[dict],
        tools: list[dict] | None,
        request_meta: dict[str, Any] | None,
    ) -> EventFacts:
        return self._event_facts(self._request_event(messages, tools, request_meta))

    def _event_facts(
        self,
        event: RequestEvent | ResponseEvent | ObservationEvent,
        supplied: ScaffoldContext | None = None,
    ) -> EventFacts:
        if supplied is not None:
            if isinstance(event, RequestEvent):
                event = replace(event, request_meta=copy_wire_value(supplied.request_meta))
            elif isinstance(event, ResponseEvent):
                event = replace(
                    event,
                    request=replace(event.request, request_meta=copy_wire_value(supplied.request_meta)),
                )
            else:
                event = replace(event, request_meta=copy_wire_value(supplied.request_meta))
        facts = self._policy.inspect_event(event)
        if supplied is not None:
            facts = replace(
                facts,
                context_key=supplied.context_key,
                request_id=supplied.request_id,
                agent_id=supplied.agent_id,
                membership=supplied.membership,
                declared_claim=supplied.declared_claim,
                parent_claim=supplied.parent_claim,
                iteration=supplied.iteration,
                retry=supplied.retry,
                sequence=supplied.sequence,
                timeline=self._timeline_with_context(facts.timeline, supplied),
            )
        self._validate_event_facts(event, facts)
        return facts

    @staticmethod
    def _timeline_with_context(
        timeline: tuple[tuple[str, Hashable], ...],
        context: ScaffoldContext,
    ) -> tuple[tuple[str, Hashable], ...]:
        if not timeline:
            return ()
        projected = dict(timeline)
        for key, value in (
            ('request_id', context.request_id),
            ('agent_id', context.agent_id),
            ('iteration', context.iteration),
            ('retry', context.retry),
            ('sequence', context.sequence),
        ):
            if value is None:
                projected.pop(key, None)
            else:
                projected[key] = value
        projected['is_sidechain'] = context.membership is not None
        canonical_order = (
            'event_type',
            'request_id',
            'agent_id',
            'parent_agent_id',
            'is_sidechain',
            'start_time',
            'end_time',
            'duration',
            'status',
            'span_id',
            'parent_span_id',
            'iteration',
            'retry',
            'sequence',
            'attributes',
        )
        ordered = [(key, projected.pop(key)) for key in canonical_order if key in projected]
        ordered.extend(projected.items())
        return tuple(ordered)

    def _contextless_observation_facts(self, event: ObservationEvent) -> EventFacts:
        inspected = self._policy.inspect_event(event)
        facts = replace(
            inspected,
            context_key=None,
            request_id=None,
            agent_id=None,
            membership=None,
            declared_claim=None,
            parent_claim=None,
            timeline=(),
            iteration=None,
            retry=None,
            sequence=None,
        )
        self._validate_event_facts(event, facts)
        return facts

    @staticmethod
    def _validate_event_facts(
        event: RequestEvent | ResponseEvent | ObservationEvent,
        facts: EventFacts,
    ) -> None:
        if isinstance(event, RequestEvent):
            expected_anchors = {InputMessage(index) for index in range(len(event.messages))}
        elif isinstance(event, ResponseEvent):
            expected_anchors = {OutputMessage()}
        else:
            expected_anchors = {ObservationMessage()}
        identity_anchors = [item.anchor for item in facts.message_identities]
        if len(identity_anchors) != len(expected_anchors) or set(identity_anchors) != expected_anchors:
            event_name = type(event).__name__
            raise ValueError(f'{event_name} message identities do not match its messages')

        for field_name in _ANCHORED_FACT_FIELDS:
            for fact in getattr(facts, field_name):
                if fact.anchor not in expected_anchors:
                    raise ValueError(f'{field_name} anchor is not emitted by this event')
        for fact in facts.boundary_facts:
            if fact.anchor not in expected_anchors:
                raise ValueError('boundary_facts anchor is not emitted by this event')

    @staticmethod
    def _request_event(
        messages: list[dict],
        tools: list[dict] | None,
        request_meta: dict[str, Any] | None,
    ) -> RequestEvent:
        return RequestEvent(
            messages=tuple(copy_messages(messages)),
            tools=copy_wire_value(tools),
            request_meta=copy_wire_value(request_meta),
        )

    @staticmethod
    def _identity_for(facts: EventFacts, anchor: InputMessage | OutputMessage | ObservationMessage) -> Hashable:
        return next(item.value for item in facts.message_identities if item.anchor == anchor)

    @staticmethod
    def _input_identities(facts: EventFacts, message_count: int) -> list[Hashable]:
        by_index: dict[int, Hashable] = {}
        for item in facts.message_identities:
            if not isinstance(item.anchor, InputMessage):
                raise ValueError('Request message identities must use InputMessage anchors')
            if item.anchor.index in by_index:
                raise ValueError(f'Duplicate InputMessage identity for index {item.anchor.index}')
            by_index[item.anchor.index] = item.value
        expected = set(range(message_count))
        if set(by_index) != expected:
            raise ValueError('Scaffold policy must emit exactly one identity for every input message')
        return [by_index[index] for index in range(message_count)]

    def _new_node(
        self,
        *,
        parent_id: NodeId | None,
        original_message: dict[str, Any],
        generated_by: Literal['llm', 'observation', 'client'],
        identity: Hashable,
        event_facts: EventFacts,
        message_id: int | None = None,
        templated_message: str | None = None,
        token_account: TokenAccount | None = None,
    ) -> NodeId:
        state = self._runtime_state
        node_id = state.next_node_id
        state.next_node_id += 1
        if message_id is None:
            message_id = state.next_message_id
            state.next_message_id += 1
        self.trajectory.nodes[node_id] = TrajectoryNode(
            node_id=node_id,
            message_id=message_id,
            parent=parent_id,
            generated_by=generated_by,
            children=[],
            original_message=copy_message(original_message),
            templated_message=templated_message,
            token_account=token_account,
        )
        if parent_id is None:
            self.trajectory.roots.append(node_id)
        else:
            self.trajectory.nodes[parent_id].children.append(node_id)
        state.facts.by_node[node_id] = self._bound_node_facts(node_id, identity, event_facts)
        return node_id

    def _bound_node_facts(
        self,
        node_id: NodeId,
        identity: Hashable,
        event_facts: EventFacts,
        *,
        include_ownership: bool = True,
    ) -> NodeRuntimeFacts:
        bindings: dict[Any, NodeId] = {item.anchor: node_id for item in event_facts.message_identities}
        anchored = {
            field_name: tuple(
                replace(fact, anchor=BoundAnchor(node_id))
                for fact in getattr(event_facts, field_name)
                if bindings.get(fact.anchor) == node_id
            )
            for field_name in _ANCHORED_FACT_FIELDS
        }
        occurrence_facts = [
            fact
            for fact in (
                event_facts.membership if include_ownership else None,
                *anchored['spawn_intents'],
                *anchored['spawn_bindings'],
                *anchored['retrieval_intents'],
                *anchored['delivery_receipts'],
                *anchored['notification_deliveries'],
                event_facts.declared_claim if include_ownership else None,
                event_facts.parent_claim if include_ownership else None,
                *anchored['child_prompts'],
            )
            if fact is not None
        ]
        occurrences: list[RuntimeFactOccurrence] = []
        for fact in occurrence_facts:
            occurrences.append(RuntimeFactOccurrence(self._runtime_state.next_fact_sequence, fact))
            self._runtime_state.next_fact_sequence += 1
        return NodeRuntimeFacts(
            identity=identity,
            tools_identity=event_facts.tools_identity,
            context_key=event_facts.context_key,
            occurrences=tuple(occurrences),
        )

    def _merge_node_facts(self, node_id: NodeId, event_facts: EventFacts, index: int | None = None) -> None:
        current = self._runtime_state.facts.by_node[node_id]
        anchor = InputMessage(index) if index is not None else ObservationMessage()
        selected = replace(
            event_facts,
            message_identities=tuple(item for item in event_facts.message_identities if item.anchor == anchor),
        )
        incoming = self._bound_node_facts(node_id, current.identity, selected, include_ownership=False)
        self._runtime_state.facts.by_node[node_id] = replace(
            current,
            occurrences=current.occurrences + incoming.occurrences,
        )

    def _find_or_create_path(
        self,
        messages: list[dict],
        facts: EventFacts,
        request_sequence: int,
        render: PromptRender | None,
    ) -> NodeId | None:
        if render is not None and len(messages) != len(render.templated_messages):
            raise ValueError('PromptRender message count does not match the request')
        logical_ids = self._logical_message_ids(facts, len(messages)) if render is not None else [None] * len(messages)
        templated_messages = render.templated_messages if render is not None else [None] * len(messages)
        message_token_ids = render.message_token_ids if render is not None else [None] * len(messages)
        current_id: NodeId | None = None
        for index, (message, templated, token_ids, logical_id) in enumerate(
            zip(messages, templated_messages, message_token_ids, logical_ids, strict=True)
        ):
            identity = self._identity_for(facts, InputMessage(index))
            semantic_children = self._find_children(current_id, identity)
            if render is None:
                child_id = self._contextual_match(semantic_children, facts.context_key)
            else:
                token_matches = []
                for candidate_id in semantic_children:
                    account = self.trajectory.nodes[candidate_id].token_account
                    if account is not None and account.token_ids == token_ids:
                        token_matches.append(candidate_id)
                child_id = self._contextual_match(
                    token_matches,
                    facts.context_key,
                )
            if child_id is None and render is not None:
                child_id = next(
                    (
                        candidate_id
                        for candidate_id in semantic_children
                        if self.trajectory.nodes[candidate_id].generated_by != 'llm'
                        and self.trajectory.nodes[candidate_id].templated_message is None
                        and self.trajectory.nodes[candidate_id].token_account is None
                        and not self.trajectory.nodes[candidate_id].children
                    ),
                    None,
                )
                if child_id is not None:
                    child = self.trajectory.nodes[child_id]
                    child.templated_message = templated
                    child.token_account = TokenAccount(token_ids=list(token_ids or []))
            if child_id is None:
                anchor = InputMessage(index)
                child_id = self._new_node(
                    parent_id=current_id,
                    original_message=message,
                    generated_by='client',
                    identity=identity,
                    event_facts=replace(
                        facts,
                        message_identities=tuple(item for item in facts.message_identities if item.anchor == anchor),
                    ),
                    message_id=logical_id,
                    templated_message=templated,
                    token_account=TokenAccount(token_ids=list(token_ids or [])) if render is not None else None,
                )
            else:
                self._record_variant(child_id, message, request_sequence)
                self._merge_node_facts(child_id, facts, index)
            current_id = child_id
        return current_id

    def _logical_message_ids(self, facts: EventFacts, message_count: int) -> list[int | None]:
        candidates: list[NodeId | None] = [None]
        result: list[int | None] = []
        for identity in self._input_identities(facts, message_count):
            matches = [
                child_id for candidate_id in candidates for child_id in self._find_children(candidate_id, identity)
            ]
            message_ids = {self.trajectory.nodes[node_id].message_id for node_id in matches}
            message_id = next(iter(message_ids)) if len(message_ids) == 1 else None
            if message_id is None:
                contextual = {
                    self.trajectory.nodes[node_id].message_id
                    for node_id in matches
                    if self._runtime_state.facts.by_node[node_id].context_key == facts.context_key
                }
                message_id = next(iter(contextual)) if len(contextual) == 1 else None
            result.append(message_id)
            candidates = matches
        return result

    def _find_children(self, parent_id: NodeId | None, identity: Hashable) -> list[NodeId]:
        children = self.trajectory.roots if parent_id is None else self.trajectory.nodes[parent_id].children
        return [node_id for node_id in children if self._runtime_state.facts.by_node[node_id].identity == identity]

    def _contextual_match(self, matches: list[NodeId], context_key: Hashable) -> NodeId | None:
        if not matches:
            return None
        return next(
            (node_id for node_id in matches if self._runtime_state.facts.by_node[node_id].context_key == context_key),
            matches[0],
        )

    def _find_path_node(self, facts: EventFacts, message_count: int) -> NodeId | None:
        candidates: list[NodeId | None] = [None]
        for identity in self._input_identities(facts, message_count):
            candidates = [
                child_id for candidate_id in candidates for child_id in self._find_children(candidate_id, identity)
            ]
            if not candidates:
                return None
        contextual = [
            node_id
            for node_id in candidates
            if node_id is not None and self._runtime_state.facts.by_node[node_id].context_key == facts.context_key
        ]
        if not contextual:
            return None
        if len(contextual) > 1:
            raise ValueError('parent_messages match multiple physical trajectory paths')
        return contextual[0]

    def _node_path(self, node_id: NodeId | None) -> list[NodeId]:
        path: list[NodeId] = []
        current_id = node_id
        while current_id is not None:
            path.append(current_id)
            current_id = self.trajectory.nodes[current_id].parent
        path.reverse()
        return path

    def _path_divergence_message_index(self, previous_id: NodeId, current_id: NodeId | None) -> int | None:
        previous_path = self._node_path(previous_id)
        current_path = self._node_path(current_id)
        for index, (previous_node_id, current_node_id) in enumerate(zip(previous_path, current_path, strict=False)):
            if (
                self._runtime_state.facts.by_node[previous_node_id].identity
                != self._runtime_state.facts.by_node[current_node_id].identity
            ):
                return index
        return len(current_path) if len(current_path) < len(previous_path) else None

    def _record_variant(self, node_id: NodeId, message: dict[str, Any], request_sequence: int) -> None:
        node = self.trajectory.nodes[node_id]
        wire = copy_message(message)
        if wire == node.original_message:
            return
        for variant in node.as_sent_variants:
            if variant['message'] == wire:
                variant['occurrences'] += 1
                return
        node.as_sent_variants.append({'message': wire, 'first_request_seq': request_sequence, 'occurrences': 1})

    def _update_cursors(self, node_id: NodeId) -> None:
        state = self._runtime_state
        cache = state.lookup_cache
        facts = replace(
            state.facts.by_node[node_id],
            cursor_sequence=state.next_cursor_sequence,
        )
        state.next_cursor_sequence += 1
        state.facts.by_node[node_id] = facts
        node = self.trajectory.nodes[node_id]
        cache.observe(node_id, node, facts)

    def _wire_messages_for_node(self, node_id: NodeId | None) -> list[dict[str, Any]]:
        path = self._node_path(node_id)
        for index in range(len(path) - 1, -1, -1):
            llm = self.trajectory.nodes[path[index]]
            if llm.generated_by != 'llm':
                continue
            inputs = [
                copy_message(
                    llm.request_message_overrides.get(input_id, self.trajectory.nodes[input_id].original_message)
                )
                for input_id in path[:index]
            ]
            return inputs + [copy_message(self.trajectory.nodes[item_id].original_message) for item_id in path[index:]]
        return [copy_message(self.trajectory.nodes[item_id].original_message) for item_id in path]

    @staticmethod
    def _validate_prompt_inputs(
        training: dict[str, Any] | None,
        render: PromptRender | None,
        completion_text: str | None,
    ) -> None:
        if training is not None and render is None:
            raise ValueError('PromptRender is required for token-level recording')
        if render is None:
            return
        if training is None:
            raise ValueError('token_training is required with PromptRender')
        if completion_text is None:
            raise ValueError('completion_text is required with PromptRender')
        if training['prompt_ids'] != render.token_ids:
            raise ValueError('PromptRender token IDs differ from the model prompt IDs')
        if training['raw_prompt_ids'] != render.token_ids:
            if not training['raw_prompt_ids']:
                raise ValueError('raw_prompt_ids are required for token-level recording')
            raise NotImplementedError('Per-message multi-modal token ownership is not supported yet')

    def _request_boundary(self, facts: EventFacts, current_id: NodeId | None) -> PendingBoundary | None:
        path = self._node_path(current_id)
        anchor_bindings = {InputMessage(index): node_id for index, node_id in enumerate(path)}
        matches: list[tuple[BoundaryFact, Hashable]] = []
        for fact in facts.boundary_facts:
            anchor_id = anchor_bindings.get(fact.anchor)
            if anchor_id is None:
                continue
            anchor_context = self._runtime_state.facts.by_node[anchor_id].context_key
            if (
                facts.membership is not None
                and facts.declared_claim is None
                and not fact.from_request_context
                and anchor_context != facts.context_key
            ):
                continue
            if self._path_has_claim(current_id, fact.rule_name, anchor_id):
                continue
            matches.append((fact, anchor_id))
        return self._boundary_from_matches(matches)

    @classmethod
    def _output_boundary(cls, facts: EventFacts) -> PendingBoundary | None:
        return cls._boundary_from_matches([(fact, fact.anchor) for fact in facts.boundary_facts])

    @staticmethod
    def _boundary_from_matches(matches: list[tuple[BoundaryFact, Hashable]]) -> PendingBoundary | None:
        if not matches:
            return None
        consumed = list(matches)
        kinds = {fact.segment_kind for fact, _ in matches}
        if len(kinds) != 1:
            request_matches = [item for item in matches if item[0].from_request_context]
            edge_matches = [item for item in matches if item[0].at_request_edge]
            if request_matches and len({item[0].segment_kind for item in request_matches}) == 1:
                matches = request_matches
            elif edge_matches and len({item[0].segment_kind for item in edge_matches}) == 1:
                matches = edge_matches
            else:
                names = ', '.join(fact.rule_name for fact, _ in matches)
                raise ValueError(f'Boundary rules start incompatible segment kinds: {names}')
        rules = list(dict.fromkeys(fact.rule_name for fact, _ in matches))
        causes = tuple(dict.fromkeys(fact.cause for fact, _ in matches))
        return PendingBoundary(
            SegmentBoundary(
                starts_new_segment=True,
                kind=matches[0][0].segment_kind,
                causes=causes,
                evidence={'boundary_rules': rules},
            ),
            tuple((fact.rule_name, anchor) for fact, anchor in consumed),
        )

    def _path_has_claim(self, node_id: NodeId | None, rule_name: str, anchor_id: NodeId) -> bool:
        current_id = node_id
        while current_id is not None:
            if (rule_name, anchor_id) in self._runtime_state.facts.by_node[current_id].boundary_claims:
                return True
            current_id = self.trajectory.nodes[current_id].parent
        return False

    def _consume_pending_boundary(self, boundary: PendingBoundary | None) -> PendingBoundary | None:
        pending = self._runtime_state.pending_boundary
        self._runtime_state.pending_boundary = None
        return self._merge_boundaries(boundary, pending)

    @staticmethod
    def _merge_boundaries(
        primary: PendingBoundary | None,
        extra: PendingBoundary | None,
    ) -> PendingBoundary | None:
        if extra is None or not extra.boundary.starts_new_segment:
            return primary
        if primary is None or not primary.boundary.starts_new_segment:
            return extra
        primary_boundary = primary.boundary
        extra_boundary = extra.boundary
        evidence = {**extra_boundary.evidence, **primary_boundary.evidence}
        if primary_boundary.evidence.get('physical_only') is not True:
            evidence.pop('physical_only', None)
        causes = tuple(dict.fromkeys(extra_boundary.causes + primary_boundary.causes))
        return PendingBoundary(
            SegmentBoundary(
                starts_new_segment=True,
                kind=primary_boundary.kind or extra_boundary.kind,
                causes=causes,
                evidence=evidence,
            ),
            tuple(dict.fromkeys(extra.claims + primary.claims)),
        )

    def _physical_boundary(
        self,
        *,
        request_facts: EventFacts,
        current_id: NodeId | None,
        token_training: dict[str, Any] | None,
    ) -> PendingBoundary | None:
        nearest_id = next(
            (
                llm_id
                for node_id in reversed(self._node_path(current_id))
                if (
                    llm_id := self._runtime_state.lookup_cache.llm_by_message_id.get(
                        self.trajectory.nodes[node_id].message_id
                    )
                )
                is not None
            ),
            None,
        )
        if nearest_id is not None and self._chain_identity(nearest_id) == self._event_chain_identity(request_facts):
            previous_id = nearest_id
        else:
            previous_id = self._runtime_state.lookup_cache.last_llm_by_context.get(request_facts.context_key)
        if previous_id is None:
            return None
        previous = self.trajectory.nodes[previous_id]
        previous_account = previous.token_account
        previous_is_token = previous_account is not None
        current_is_token = token_training is not None
        if previous_is_token != current_is_token:
            return self._physical_restart('representation_restart', 'training_representation_discontinuity')
        previous_facts = self._runtime_state.facts.by_node[previous_id]
        if previous_facts.context_key != request_facts.context_key:
            return self._physical_restart('render_restart', 'render_context_discontinuity')
        if not current_is_token:
            if previous_facts.tools_identity == request_facts.tools_identity:
                return None
            return self._physical_restart('render_restart', 'render_context_discontinuity')

        assert token_training is not None
        assert previous_account is not None
        prompt_ids = token_training['prompt_ids']
        previous_prompt = (
            self.trajectory.prompt_ids(previous_id)
            if previous_account.token_ids is not None
            else self._input_token_ids(previous.parent)
        )
        previous_stream = previous_prompt + previous_account.generated_ids + previous_account.trailing_ids
        if prompt_ids == previous_prompt or prompt_ids[: len(previous_stream)] == previous_stream:
            return None
        divergence_message_index = self._path_divergence_message_index(previous_id, current_id)
        divergence = self._divergence_attribution(
            prompt_ids=prompt_ids,
            previous_stream=previous_stream,
            previous_tools=previous_facts.tools_identity,
            current_tools=request_facts.tools_identity,
            path_divergence_message_index=divergence_message_index,
        )
        cause = 'history_rewrite' if divergence == 'history-rewrite' else 'render_divergence'
        evidence: dict[str, Any] = {'divergence': divergence, 'physical_only': True}
        if divergence == 'history-rewrite':
            evidence['path_divergence_message_index'] = divergence_message_index
        return PendingBoundary(
            SegmentBoundary(
                starts_new_segment=True,
                kind='render_restart',
                causes=(cause,),
                evidence=evidence,
            )
        )

    @staticmethod
    def _physical_restart(kind: str, cause: str) -> PendingBoundary:
        return PendingBoundary(
            SegmentBoundary(
                starts_new_segment=True,
                kind=kind,
                causes=(cause,),
                evidence={'physical_only': True},
            )
        )

    def _divergence_attribution(
        self,
        *,
        prompt_ids: list[int],
        previous_stream: list[int],
        previous_tools: Hashable,
        current_tools: Hashable,
        path_divergence_message_index: int | None,
    ) -> str:
        divergence_start = 0
        for previous_token, current_token in zip(previous_stream, prompt_ids, strict=False):
            if previous_token != current_token:
                break
            divergence_start += 1
        delta = tuple(prompt_ids[divergence_start:])
        terminator = tuple(self._policy.block_terminator_ids)
        if terminator and delta[: len(terminator)] == terminator:
            return 'terminator-residue'
        if previous_tools != current_tools:
            return 'context-change'
        if path_divergence_message_index is not None:
            return 'history-rewrite'
        return 'assumption-violation'

    def _chain_identity(self, node_id: NodeId) -> tuple[Any, ...]:
        memberships = self._runtime_state.facts.by_node[node_id].facts_of_type(SidechainMembership)
        membership = memberships[0][1] if memberships else None
        if membership is None:
            return ('main',)
        return (membership.chain_id, membership.kind, membership.detector)

    @staticmethod
    def _event_chain_identity(facts: EventFacts) -> tuple[Any, ...]:
        membership = facts.membership
        if membership is None:
            return ('main',)
        return (membership.chain_id, membership.kind, membership.detector)

    def _input_token_ids(self, node_id: NodeId | None) -> list[int]:
        token_ids: list[int] = []
        for path_id in self._node_path(node_id):
            account = self.trajectory.nodes[path_id].token_account
            if account is not None and account.token_ids is not None:
                token_ids.extend(account.token_ids)
        return token_ids

    @staticmethod
    def _provider_facts(
        output: dict[str, Any],
        stop_reason: str | None,
        response_meta: dict[str, Any] | None,
    ) -> ProviderFacts | None:
        meta = response_meta or {}
        facts = ProviderFacts(
            model=copy_value(meta.get('model')),
            response_id=copy_value(meta.get('response_id')),
            usage=copy_value(meta.get('usage')),
            finish_reason=TrajectoryAssembler._finish_reason(output, stop_reason),
            inference_time=copy_value(meta.get('inference_time')),
            abort_wait_time=copy_value(meta.get('abort_wait_time')),
            num_preempted=copy_value(meta.get('num_preempted')),
        )
        return facts if any(value is not None for value in vars(facts).values()) else None

    @staticmethod
    def _invocation(facts: EventFacts) -> InvocationFacts | None:
        if facts.iteration is None and facts.retry is None:
            return None
        return InvocationFacts(iteration=facts.iteration, retry=facts.retry)

    @staticmethod
    def _token_account(glue_ids: list[int], training: dict[str, Any]) -> TokenAccount:
        response_ids = list(training.get('response_ids') or [])
        extra = training.get('extra_fields') or {}
        indices = extra.get('resp_topk_logprob_indice')
        values = extra.get('resp_topk_logprob_value')
        return TokenAccount(
            token_ids=list(glue_ids) + response_ids,
            gen_start=len(glue_ids),
            gen_len=len(response_ids),
            logprobs=list(training['response_logprobs']) if training.get('response_logprobs') else None,
            topk_indices=[list(row) for row in indices] if indices else None,
            topk_values=[list(row) for row in values] if values else None,
            routed_experts=copy_value(training.get('routed_experts')),
            multi_modal=copy_value(training.get('multi_modal_data')),
        )

    @staticmethod
    def _normalize_token_training(token_training: dict[str, Any]) -> dict[str, Any]:
        response_logprobs = token_training.get('response_logprobs')
        return {
            'prompt_ids': list(token_training.get('prompt_ids') or []),
            'raw_prompt_ids': list(token_training.get('raw_prompt_ids') or []),
            'response_ids': list(token_training.get('response_ids') or []),
            'response_logprobs': list(response_logprobs) if response_logprobs is not None else None,
            'routed_experts': token_training.get('routed_experts'),
            'multi_modal_data': copy_value(token_training.get('multi_modal_data')),
            'extra_fields': copy_value(token_training.get('extra_fields') or {}),
        }

    @staticmethod
    def _finish_reason(message: dict[str, Any], stop_reason: str | None) -> str:
        if stop_reason == 'length':
            return 'length'
        if message.get('tool_calls'):
            return 'tool_calls'
        return stop_reason or 'stop'
