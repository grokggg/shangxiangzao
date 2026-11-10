# /// script
# dependencies = ["langfuse>=4.0.0"]
#
# [tool.nexus]
# trajectory_observers = { langfuse = "nexus.gateway.observers.langfuse" }
# ///

from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import logging
import math
import os
import threading
import uuid

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, cast

from nexus.gateway.observers.base import TrajectoryObserver, TrajectoryUpdate
from nexus.gateway.observers.registry import trajectory_observer_registry
from nexus.gateway.trajectory import MAIN_CHAIN_ID, Chain, Trajectory, TrajectoryNode


logger = logging.getLogger(__name__)

_TAIJI_CONTRACT = 'episode_trace_generation_turn'
_TAIJI_METADATA_ENV = {
    'instance_id': 'TJ_INSTANCE_ID',
    'task_id': 'TJ_TASK_ID',
    'task_name': 'TJ_TASK_NAME',
    'workspace_id': 'TJ_WORKSPACE_ID',
}


@dataclass(frozen=True)
class LangfuseTrajectoryConfig:
    public_key: str = ''
    secret_key: str = ''
    host: str = ''
    trace_name: str = ''
    session_id: str = ''
    tags: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    flush_on_final: bool = True
    max_tool_chars: int = 8000
    max_content_chars: int = 0

    @classmethod
    def from_dict(cls, value: dict[str, Any] | None) -> LangfuseTrajectoryConfig:
        data = dict(value or {})
        data.pop('enabled', None)
        if 'trace_metadata' in data:
            if 'metadata' in data:
                raise ValueError('Configure either metadata or trace_metadata, not both')
            data['metadata'] = data.pop('trace_metadata')
        if 'tags' in data:
            data['tags'] = tuple(data['tags'] or ())
        return cls(**data)


@dataclass(frozen=True)
class _Observation:
    key: str
    parent_key: str | None
    kind: str
    name: str
    input: Any = None
    output: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)
    model: str | None = None
    model_parameters: dict[str, Any] | None = None
    usage: dict[str, int] | None = None
    complete: bool = False

    def payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {'input': self.input, 'output': self.output, 'metadata': self.metadata}
        if self.model:
            payload['model'] = self.model
        if self.model_parameters:
            payload['model_parameters'] = self.model_parameters
        if self.usage and self.usage.get('total', 0) > 0:
            payload['usage_details'] = self.usage
        return payload


@dataclass(frozen=True)
class _Presentation:
    observations: tuple[_Observation, ...]
    session_id: str
    trace_name: str
    tags: tuple[str, ...]
    score: float | None
    score_comment: str


@dataclass
class _ExportState:
    trace_id: str
    last_revision: int = -1
    handles: dict[str, Any] = field(default_factory=dict)
    fingerprints: dict[str, str] = field(default_factory=dict)
    ended: set[str] = field(default_factory=set)
    order: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class _Turn:
    node: TrajectoryNode
    chain_id: str
    turn: int
    global_step: int
    request_messages: list[dict[str, Any]]
    tool_calls: list[dict[str, Any]]


class LangfusePresentationProjector:
    """Project explicit Trajectory facts into the legacy Langfuse page contract."""

    def __init__(self, config: LangfuseTrajectoryConfig):
        self._config = config

    def project(self, trajectory: Trajectory, *, final: bool) -> _Presentation:
        metadata = self._trace_metadata()
        trace_name = self._config.trace_name or str(metadata.get('instance_id') or metadata.get('task_name') or '')
        trace_name = trace_name or 'Agent Run'
        session_id = self._config.session_id or str(metadata.get('episode_id') or trajectory.session_id)
        turns = self._turns(trajectory)
        observations = [self._trace(trajectory, metadata, trace_name, turns, final)]
        observations.extend(self._chain(trajectory, MAIN_CHAIN_ID, turns, metadata, final, 'trace'))
        model = self._last_model(turns)
        tags = self._dedupe([
            *self._config.tags,
            metadata.get('instance_id'),
            metadata.get('task_id'),
            metadata.get('task_name'),
            metadata.get('workspace_id'),
            model.removeprefix('openai/') if model else None,
        ])
        return _Presentation(
            observations=tuple(observations),
            session_id=session_id,
            trace_name=trace_name,
            tags=tuple(tags),
            score=self._finite_number(trajectory.final_reward),
            score_comment=self._score_comment(metadata),
        )

    def _chain(
        self,
        trajectory: Trajectory,
        chain_id: str,
        turns: list[_Turn],
        trace_metadata: dict[str, Any],
        final: bool,
        parent_key: str,
    ) -> list[_Observation]:
        chain = trajectory.chains.get(chain_id, Chain(chain_id=chain_id))
        chain_turns = [turn for turn in turns if turn.chain_id == chain_id]
        agent_key = 'agent:main' if chain_id == MAIN_CHAIN_ID else f'agent:{chain_id}'
        parent_turn = self._spawn_turn(chain, turns)
        agent_input, agent_name = self._agent_identity(trajectory, chain)
        metadata: dict[str, Any] = {'source': 'langfuse_trajectory_observer'}
        if chain_id != MAIN_CHAIN_ID:
            metadata.update(self._episode_fields(trace_metadata))
            metadata.update({
                'parent_turn': None if parent_turn is None else parent_turn.turn,
                'parent_step': None if parent_turn is None else parent_turn.turn,
                'parent_global_step': None if parent_turn is None else parent_turn.global_step,
                'description': agent_name.removeprefix('Subagent: '),
                'subagent_type': chain.kind,
            })
        if final or self._chain_complete(trajectory, chain):
            usage_turns = turns if chain_id == MAIN_CHAIN_ID else chain_turns
            usage = self._total_usage(usage_turns)
            model = self._last_model(usage_turns)
            last_text = self._message_text(chain_turns[-1].node.original_message) if chain_turns else ''
            metadata.update({
                'total_steps': len(chain_turns),
                'model': model,
                'model_parameters': self._model_parameters(chain_turns[-1].node.request_kwargs) if chain_turns else {},
                'total_usage': usage,
                'cache_hit_rate': round(usage['cached_tokens'] / usage['input'], 3) if usage['input'] else 0,
                'tools': self._tools_summary(chain_turns),
            })
            agent_output: Any = (
                {'total_steps': len(chain_turns), 'model': model}
                if chain_id == MAIN_CHAIN_ID
                else {'result': self._trunc(last_text, 2000)}
            )
        else:
            agent_output = None
        observations = [
            _Observation(
                key=agent_key,
                parent_key=parent_key,
                kind='span',
                name='Agent' if chain_id == MAIN_CHAIN_ID else agent_name,
                input=agent_input,
                output=agent_output,
                metadata=self._safe(metadata),
                complete=final or (chain_id != MAIN_CHAIN_ID and self._chain_complete(trajectory, chain)),
            )
        ]
        observations.extend(self._prompt_observations(chain_turns, agent_key))
        child_chains = sorted(
            (item for item in trajectory.chains.values() if item.parent_chain_id == chain_id),
            key=lambda item: item.chain_id,
        )
        children_by_spawn: dict[int, list[Chain]] = {}
        for child in child_chains:
            if child.spawn and child.spawn.node is not None:
                children_by_spawn.setdefault(child.spawn.node, []).append(child)
        previous_count = 0
        previous_turn: _Turn | None = None
        for turn in chain_turns:
            generation = self._generation(turn, trace_metadata, agent_key, previous_count, previous_turn)
            observations.append(generation)
            previous_count = len(turn.request_messages)
            previous_turn = turn
            for tool_call in turn.tool_calls:
                child = next(
                    (
                        item
                        for item in children_by_spawn.get(turn.node.node_id, [])
                        if item.spawn and item.spawn.tool_use_id == tool_call['id']
                    ),
                    None,
                )
                tool = self._tool(trajectory, turn, tool_call, trace_metadata, final, child)
                observations.append(tool)
                if child is not None:
                    observations.extend(self._chain(trajectory, child.chain_id, turns, trace_metadata, final, tool.key))
        if final and chain_id == MAIN_CHAIN_ID and chain_turns:
            last = chain_turns[-1]
            text = self._message_text(last.node.original_message)
            if text and not last.tool_calls:
                observations.append(
                    _Observation(
                        key='final-response',
                        parent_key=agent_key,
                        kind='generation',
                        name='Final Response',
                        input={'note': 'Final assistant response (no tool calls)'},
                        output={'role': 'assistant', 'content': self._trunc(text, self._config.max_content_chars)},
                        metadata={
                            **self._turn_metadata(last, trace_metadata),
                            'is_final': True,
                            'total_steps': len(chain_turns),
                            'source': 'realtime',
                        },
                        model=self._last_model(chain_turns),
                        model_parameters=self._model_parameters(last.node.request_kwargs),
                        complete=True,
                    )
                )
        return observations

    def _trace(
        self,
        trajectory: Trajectory,
        metadata: dict[str, Any],
        trace_name: str,
        turns: list[_Turn],
        final: bool,
    ) -> _Observation:
        main_turns = [turn for turn in turns if turn.chain_id == MAIN_CHAIN_ID]
        task = self._first_user_input(main_turns)
        trace_input: Any = task or {'note': 'SWE-bench agent run'}
        root_metadata: dict[str, Any] = {**metadata, 'source': 'langfuse_trajectory_observer'}
        output = None
        if final:
            usage = self._total_usage(turns)
            model = self._last_model(turns)
            last_output = self._message_text(main_turns[-1].node.original_message) if main_turns else ''
            root_metadata.update(copy.deepcopy(trajectory.extra))
            root_metadata.update({
                'total_steps': len(main_turns),
                'model': model,
                'model_full': model,
                'num_rounds': len(main_turns),
                'total_usage': usage,
                'cache_hit_rate': round(usage['cached_tokens'] / usage['input'], 3) if usage['input'] else 0,
                'termination_reason': trajectory.termination_reason,
                'metrics': trajectory.metrics,
            })
            if trajectory.final_reward is not None:
                root_metadata['score'] = trajectory.final_reward
            output = {
                'total_steps': len(main_turns),
                'last_output': self._trunc(last_output, 2000),
                'total_usage': usage,
            }
        return _Observation(
            key='trace',
            parent_key=None,
            kind='span',
            name=trace_name,
            input=self._safe(trace_input),
            output=self._safe(output),
            metadata=self._safe(root_metadata),
            complete=final,
        )

    def _prompt_observations(self, turns: list[_Turn], parent_key: str) -> list[_Observation]:
        first = next((turn for turn in turns if (turn.node.request_kwargs or {}).get('tools')), None)
        if first is None:
            return []
        result: list[_Observation] = []
        user_created = False
        for index, message in enumerate(first.request_messages):
            role = message.get('role')
            if role == 'system' and not user_created:
                text = self._message_text(message)
                if text:
                    result.append(
                        _Observation(
                            key=f'{parent_key}:system:{index}',
                            parent_key=parent_key,
                            kind='span',
                            name='System Prompt',
                            input={'content': self._trunc(text, self._config.max_content_chars)},
                            metadata={'length': len(text)},
                            complete=True,
                        )
                    )
            elif role == 'user' and not user_created:
                user_created = True
                user = self._user_input(message)
                result.append(
                    _Observation(
                        key=f'{parent_key}:user',
                        parent_key=parent_key,
                        kind='span',
                        name='User Prompt',
                        input=user,
                        metadata={
                            'length': len(str(user.get('task', ''))),
                            'block_count': len(message.get('content', []))
                            if isinstance(message.get('content'), list)
                            else 0,
                        },
                        complete=True,
                    )
                )
                break
        tools = (first.node.request_kwargs or {}).get('tools') or []
        if tools:
            result.append(
                _Observation(
                    key=f'{parent_key}:tools',
                    parent_key=parent_key,
                    kind='span',
                    name='Tools Definition',
                    input=self._safe({'tools': tools}),
                    metadata={'tool_count': len(tools)},
                    complete=True,
                )
            )
        return result

    def _generation(
        self,
        turn: _Turn,
        trace_metadata: dict[str, Any],
        parent_key: str,
        previous_count: int,
        previous_turn: _Turn | None,
    ) -> _Observation:
        message = turn.node.original_message
        tool_names = [item['name'] for item in turn.tool_calls]
        label = f'Step {turn.turn}'
        if tool_names:
            label += f': {", ".join(tool_names[:4])}{"..." if len(tool_names) > 4 else ""}'
        request = {
            'messages': [self._truncate_message(item) for item in turn.request_messages],
            'total_messages': len(turn.request_messages),
            'new_messages': len(turn.request_messages) - previous_count,
        }
        tools = (turn.node.request_kwargs or {}).get('tools') or []
        if tools:
            request['tools'] = self._safe(tools)
        output: dict[str, Any] = {}
        thinking = self._thinking(message)
        text = self._message_text(message)
        if thinking:
            output['thinking'] = self._trunc(thinking, self._config.max_content_chars)
        if text:
            output['text'] = self._trunc(text, self._config.max_content_chars)
        output_signatures = self._output_signatures(message)
        output_reasoning = self._output_reasoning_items(message)
        input_signatures = self._input_signatures(turn.request_messages)
        input_reasoning = self._input_reasoning_items(turn.request_messages)
        if output_signatures:
            output['thinking_signatures'] = output_signatures
        if output_reasoning:
            output['reasoning_items'] = output_reasoning
        if input_signatures:
            request['thinking_signatures'] = input_signatures
        if input_reasoning:
            last_index = max(item['message_index'] for item in input_reasoning)
            request['input_reasoning_items'] = [item for item in input_reasoning if item['message_index'] == last_index]
        if turn.tool_calls:
            output['tool_calls'] = [
                {
                    'name': item['name'],
                    'args': {key: self._trunc(value, 300) for key, value in item['arguments'].items()}
                    if isinstance(item['arguments'], dict)
                    else {'raw': self._trunc(item['arguments'], 500)},
                }
                for item in turn.tool_calls
            ]
        usage = self._usage(None if turn.node.provider_facts is None else turn.node.provider_facts.usage)
        metadata = {
            **self._turn_metadata(turn, trace_metadata),
            'tool_count': len(turn.tool_calls),
            'tool_names': tool_names,
            'parallel': len(turn.tool_calls) > 1,
            'source': 'realtime',
        }
        if usage.get('cached_tokens', 0):
            metadata['cached_tokens'] = usage['cached_tokens']
        if previous_turn is not None:
            previous_signatures = self._output_signatures(previous_turn.node.original_message)
            if previous_signatures:
                indexed = {item['signature']: item for item in input_signatures}
                round_trip = [
                    {
                        'tool_call_id': item.get('tool_call_id'),
                        'found': item['signature'] in indexed,
                        'landed_at_message_index': indexed.get(item['signature'], {}).get('message_index'),
                    }
                    for item in previous_signatures
                ]
                metadata['signature_round_trip'] = round_trip
                signature_hits = sum(1 for item in round_trip if item['found'])
                metadata['signature_round_trip_rate'] = f'{signature_hits}/{len(round_trip)}'
            previous_reasoning = self._output_reasoning_items(previous_turn.node.original_message)
            if previous_reasoning:
                available = {self._reasoning_key(item): item for item in input_reasoning if self._reasoning_key(item)}
                round_trip = []
                for item in previous_reasoning:
                    key = self._reasoning_key(item)
                    hit = available.get(key)
                    round_trip.append({
                        'id': item.get('id'),
                        'match_on': None if key is None else key[0],
                        'found': hit is not None,
                        'tool_call_ids': item.get('tool_call_ids') or [],
                        'landed_tool_call_ids': None if hit is None else hit.get('tool_call_ids'),
                        'landed_at_message_index': None if hit is None else hit.get('message_index'),
                        'landed_at_reasoning_index': None if hit is None else hit.get('reasoning_index'),
                    })
                metadata['reasoning_round_trip'] = round_trip
                reasoning_hits = sum(1 for item in round_trip if item['found'])
                metadata['reasoning_round_trip_rate'] = f'{reasoning_hits}/{len(round_trip)}'
        duration = self._duration(turn.node)
        if duration is not None:
            metadata['duration_s'] = round(duration, 2)
        provider = turn.node.provider_facts
        return _Observation(
            key=f'generation:{turn.node.node_id}',
            parent_key=parent_key,
            kind='generation',
            name=label,
            input=self._safe(request),
            output=self._safe(output),
            metadata=self._safe(metadata),
            model=None if provider is None else provider.model,
            model_parameters=self._model_parameters(turn.node.request_kwargs),
            usage=usage,
            complete=True,
        )

    def _tool(
        self,
        trajectory: Trajectory,
        turn: _Turn,
        tool_call: dict[str, Any],
        trace_metadata: dict[str, Any],
        final: bool,
        child: Chain | None,
    ) -> _Observation:
        result = self._tool_result(trajectory, tool_call['id'], child)
        complete = result is not None or final
        output: Any = result
        if final and result is None:
            output = '(agent finished before tool result)'
        return _Observation(
            key=f'tool:{turn.node.node_id}:{tool_call["id"]}',
            parent_key=f'generation:{turn.node.node_id}',
            kind='tool',
            name=f'Tool: {tool_call["name"]}',
            input=self._safe(
                {key: self._trunc(value, self._config.max_tool_chars) for key, value in tool_call['arguments'].items()}
                if isinstance(tool_call['arguments'], dict)
                else {'raw': self._trunc(tool_call['arguments'], self._config.max_tool_chars)}
            ),
            output=self._trunc(output, self._config.max_tool_chars) if output is not None else None,
            metadata={
                **self._turn_metadata(turn, trace_metadata),
                'tool_call_id': tool_call['id'],
                'tool_name': tool_call['name'],
                'is_agent_call': child is not None,
                'is_parallel': len(turn.tool_calls) > 1,
                'source': 'realtime',
            },
            complete=complete,
        )

    def _turns(self, trajectory: Trajectory) -> list[_Turn]:
        llm_nodes = [node for node in trajectory.iter_nodes() if node.generated_by == 'llm']
        turn_by_chain: dict[str, int] = {}
        turns: list[_Turn] = []
        for global_step, node in enumerate(llm_nodes, start=1):
            chain_id = trajectory.node_chain(node.node_id)
            turn_by_chain[chain_id] = turn_by_chain.get(chain_id, 0) + 1
            turns.append(
                _Turn(
                    node=node,
                    chain_id=chain_id,
                    turn=turn_by_chain[chain_id],
                    global_step=global_step,
                    request_messages=self._request_messages(trajectory, node),
                    tool_calls=self._tool_calls(node.original_message),
                )
            )
        return turns

    @staticmethod
    def _request_messages(trajectory: Trajectory, node: TrajectoryNode) -> list[dict[str, Any]]:
        node_ids: list[int] = []
        current = node.parent
        while current is not None:
            node_ids.append(current)
            current = trajectory.nodes[current].parent
        node_ids.reverse()
        return [
            copy.deepcopy(node.request_message_overrides.get(node_id, trajectory.nodes[node_id].original_message))
            for node_id in node_ids
        ]

    @staticmethod
    def _tool_calls(message: dict[str, Any]) -> list[dict[str, Any]]:
        result = []
        for call in message.get('tool_calls') or []:
            function = call.get('function') or {}
            name = function.get('name') or call.get('name')
            if not name:
                continue
            raw = function.get('arguments', call.get('arguments', {}))
            try:
                arguments = json.loads(raw) if isinstance(raw, str) else raw
            except (TypeError, json.JSONDecodeError):
                arguments = {'raw': str(raw)}
            result.append({'id': str(call.get('id') or ''), 'name': str(name), 'arguments': arguments})
        return result

    @staticmethod
    def _tool_result(trajectory: Trajectory, tool_call_id: str, child: Chain | None) -> Any:
        if child and child.join and child.join.node in trajectory.nodes:
            return trajectory.nodes[child.join.node].original_message.get('content')
        for node in trajectory.iter_nodes():
            message = node.original_message
            if message.get('role') == 'tool' and message.get('tool_call_id') == tool_call_id:
                return message.get('content')
        return None

    @staticmethod
    def _output_signatures(message: dict[str, Any]) -> list[dict[str, Any]]:
        first_call = next(
            (str(call['id']) for call in message.get('tool_calls') or [] if isinstance(call, dict) and call.get('id')),
            None,
        )
        result = []
        for index, block in enumerate(message.get('thinking_blocks') or []):
            if isinstance(block, dict) and block.get('signature'):
                result.append({
                    'tool_call_id': first_call,
                    'signature_index': index,
                    'signature': str(block['signature']),
                })
        return result

    @staticmethod
    def _input_signatures(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        result = []
        for message_index, message in enumerate(messages):
            if message.get('role') != 'assistant':
                continue
            first_call = next(
                (
                    str(call['id'])
                    for call in message.get('tool_calls') or []
                    if isinstance(call, dict) and call.get('id')
                ),
                None,
            )
            blocks = list(message.get('thinking_blocks') or [])
            content = message.get('content')
            if isinstance(content, list):
                blocks = [*content, *blocks]
            signature_index = 0
            for block in blocks:
                if isinstance(block, dict) and block.get('signature'):
                    result.append({
                        'message_index': message_index,
                        'tool_call_id': first_call,
                        'signature_index': signature_index,
                        'signature': str(block['signature']),
                    })
                    signature_index += 1
        return result

    @staticmethod
    def _output_reasoning_items(message: dict[str, Any]) -> list[dict[str, Any]]:
        call_ids = [
            str(call['id']) for call in message.get('tool_calls') or [] if isinstance(call, dict) and call.get('id')
        ]
        result = []
        for index, item in enumerate(message.get('reasoning_items') or []):
            if not isinstance(item, dict):
                continue
            result.append({**copy.deepcopy(item), 'tool_call_ids': call_ids, 'reasoning_index': index})
        return result

    @staticmethod
    def _input_reasoning_items(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        result = []
        for message_index, message in enumerate(messages):
            call_ids = [
                str(call['id']) for call in message.get('tool_calls') or [] if isinstance(call, dict) and call.get('id')
            ]
            for reasoning_index, item in enumerate(message.get('reasoning_items') or []):
                if isinstance(item, dict):
                    result.append({
                        **copy.deepcopy(item),
                        'message_index': message_index,
                        'tool_call_ids': call_ids,
                        'reasoning_index': reasoning_index,
                    })
        return result

    @staticmethod
    def _reasoning_key(item: dict[str, Any]) -> tuple[str, str] | None:
        if item.get('encrypted_content'):
            return ('encrypted_content', str(item['encrypted_content']))
        if item.get('id'):
            return ('id', str(item['id']))
        return None

    def _agent_identity(self, trajectory: Trajectory, chain: Chain) -> tuple[Any, str]:
        if chain.chain_id == MAIN_CHAIN_ID:
            return None, 'Agent'
        arguments: Any = {}
        if chain.spawn and chain.spawn.node in trajectory.nodes:
            spawn_message = trajectory.nodes[chain.spawn.node].original_message
            call = next(
                (item for item in self._tool_calls(spawn_message) if item['id'] == chain.spawn.tool_use_id), None
            )
            if call is not None:
                arguments = call['arguments']
        description = arguments.get('description', chain.chain_id) if isinstance(arguments, dict) else chain.chain_id
        prompt = arguments.get('prompt', description) if isinstance(arguments, dict) else description
        return {'prompt': self._trunc(prompt, self._config.max_content_chars)}, f'Subagent: {description}'

    @staticmethod
    def _spawn_turn(chain: Chain, turns: list[_Turn]) -> _Turn | None:
        if chain.spawn is None or chain.spawn.node is None:
            return None
        return next((turn for turn in turns if turn.node.node_id == chain.spawn.node), None)

    @staticmethod
    def _chain_complete(trajectory: Trajectory, chain: Chain) -> bool:
        return chain.join is not None and chain.join.node in trajectory.nodes

    @staticmethod
    def _message_text(message: dict[str, Any]) -> str:
        content = message.get('content')
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return '\n'.join(
                str(block.get('text') or '')
                for block in content
                if isinstance(block, dict) and block.get('type') == 'text'
            )
        return '' if content is None else str(content)

    @staticmethod
    def _thinking(message: dict[str, Any]) -> str:
        return str(
            message.get('reasoning_content')
            or (message.get('provider_specific_fields') or {}).get('reasoning_content')
            or ''
        )

    def _truncate_message(self, message: dict[str, Any]) -> dict[str, Any]:
        result = copy.deepcopy(message)
        limit = self._config.max_tool_chars if message.get('role') == 'tool' else self._config.max_content_chars
        if isinstance(result.get('content'), str):
            result['content'] = self._trunc(result['content'], limit)
        return self._safe(result)

    def _first_user_input(self, turns: list[_Turn]) -> dict[str, Any] | None:
        if not turns:
            return None
        message = next((item for item in turns[0].request_messages if item.get('role') == 'user'), None)
        return None if message is None else self._user_input(message)

    def _user_input(self, message: dict[str, Any]) -> dict[str, Any]:
        content = message.get('content')
        if isinstance(content, str):
            return {'task': self._trunc(content, self._config.max_content_chars)}
        blocks = content if isinstance(content, list) else []
        texts = [
            str(block.get('text') or '') for block in blocks if isinstance(block, dict) and block.get('type') == 'text'
        ]
        task = next(
            (text for text in texts if text and not text.strip().startswith('<system-reminder>')), '\n'.join(texts)
        )
        context_blocks = [
            {'type': 'text', 'text': self._trunc(text, 500)}
            for text in texts
            if text.strip().startswith('<system-reminder>')
        ]
        result: dict[str, Any] = {'task': self._trunc(task, self._config.max_content_chars)}
        if context_blocks:
            result['context_blocks'] = context_blocks
        return result

    @staticmethod
    def _model_parameters(request_kwargs: dict[str, Any] | None) -> dict[str, Any]:
        kwargs = request_kwargs or {}
        result = {key: kwargs[key] for key in ('temperature', 'max_tokens') if kwargs.get(key) is not None}
        extra_body = kwargs.get('extra_body') or {}
        chat_kwargs = extra_body.get('chat_template_kwargs') or {} if isinstance(extra_body, dict) else {}
        for key in ('reasoning_effort', 'interleaved_thinking'):
            if chat_kwargs.get(key) is not None:
                result[key] = chat_kwargs[key]
        headers = kwargs.get('extra_headers') or {}
        if isinstance(headers, dict) and headers.get('StrategyType'):
            result['StrategyType'] = headers['StrategyType']
        return result

    def _tools_summary(self, turns: list[_Turn]) -> list[dict[str, str]]:
        tools = next(
            (
                (turn.node.request_kwargs or {}).get('tools')
                for turn in turns
                if (turn.node.request_kwargs or {}).get('tools')
            ),
            [],
        )
        result = []
        for tool in tools or []:
            function = tool.get('function') or {}
            name = function.get('name') or tool.get('name')
            if name:
                result.append({
                    'name': str(name),
                    'description': self._trunc(function.get('description') or tool.get('description') or '', 120),
                })
        return result

    @staticmethod
    def _usage(value: dict[str, Any] | None) -> dict[str, int]:
        value = value if isinstance(value, dict) else {}
        details = value.get('prompt_tokens_details') or {}
        cached = (
            details.get('gpu_cached_tokens', 0) or details.get('cached_tokens', 0) or 0
            if isinstance(details, dict)
            else 0
        )
        input_tokens = value.get('prompt_tokens', value.get('input_tokens', 0)) or 0
        output_tokens = value.get('completion_tokens', value.get('output_tokens', 0)) or 0
        return {
            'input': int(input_tokens),
            'output': int(output_tokens),
            'total': int(value.get('total_tokens', value.get('total', input_tokens + output_tokens)) or 0),
            'cached_tokens': int(cached),
        }

    def _total_usage(self, turns: list[_Turn]) -> dict[str, int]:
        total = {'input': 0, 'output': 0, 'total': 0, 'cached_tokens': 0}
        for turn in turns:
            usage = self._usage(None if turn.node.provider_facts is None else turn.node.provider_facts.usage)
            for key in total:
                total[key] += usage[key]
        return total

    @staticmethod
    def _last_model(turns: list[_Turn]) -> str:
        return next(
            (
                turn.node.provider_facts.model
                for turn in reversed(turns)
                if turn.node.provider_facts and turn.node.provider_facts.model
            ),
            '',
        )

    @staticmethod
    def _duration(node: TrajectoryNode) -> float | None:
        for event in node.timeline:
            start = LangfusePresentationProjector._finite_number(event.get('start_time'))
            end = LangfusePresentationProjector._finite_number(event.get('end_time'))
            if start is not None and end is not None:
                return max(0.0, end - start)
        return None

    @staticmethod
    def _finite_number(value: Any) -> float | None:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if math.isfinite(number) else None

    def _trace_metadata(self) -> dict[str, Any]:
        env = {key: value for key, name in _TAIJI_METADATA_ENV.items() if (value := os.environ.get(name))}
        metadata = {**env, **copy.deepcopy(self._config.metadata)}
        metadata['contract'] = _TAIJI_CONTRACT
        metadata.setdefault('episode', 0)
        if metadata.get('instance_id') and not metadata.get('episode_id'):
            metadata['episode_id'] = f'{metadata["instance_id"]}-ep{metadata["episode"]}'
        return metadata

    @staticmethod
    def _episode_fields(metadata: dict[str, Any]) -> dict[str, Any]:
        result = {'contract': _TAIJI_CONTRACT, 'episode': metadata.get('episode', 0)}
        if metadata.get('episode_id'):
            result['episode_id'] = metadata['episode_id']
        return result

    def _turn_metadata(self, turn: _Turn, metadata: dict[str, Any]) -> dict[str, Any]:
        return {**self._episode_fields(metadata), 'turn': turn.turn, 'step': turn.turn, 'global_step': turn.global_step}

    @staticmethod
    def _score_comment(metadata: dict[str, Any]) -> str:
        parts = []
        if metadata.get('instance_id'):
            parts.append(f'instance={metadata["instance_id"]}')
        parts.append(f'episode={metadata.get("episode", 0)}')
        if metadata.get('task_name'):
            parts.append(f'task={metadata["task_name"]}')
        return ', '.join(parts)

    @staticmethod
    def _dedupe(items: list[Any]) -> list[str]:
        return list(dict.fromkeys(str(item) for item in items if item is not None and str(item)))

    @staticmethod
    def _trunc(value: Any, limit: int) -> str:
        if value is None:
            return ''
        text = str(value)
        return text if limit <= 0 or len(text) <= limit else text[:limit] + f'...[{len(text)} chars]'

    def _safe(self, value: Any) -> Any:
        try:
            serialized = json.dumps(value, ensure_ascii=False, default=str)
            normalized = json.loads(serialized)
        except (TypeError, ValueError):
            return str(value)
        limit = self._config.max_content_chars
        if limit > 0 and len(serialized) > limit:
            return {'truncated_json': serialized[:limit], 'original_chars': len(serialized)}
        return normalized


@trajectory_observer_registry.register('langfuse')
class LangfuseTrajectoryObserver(TrajectoryObserver):
    _MAX_FINALIZED_IDS = 4096

    def __init__(
        self,
        config: LangfuseTrajectoryConfig | dict[str, Any] | None = None,
        *,
        client: Any = None,
        **config_values: Any,
    ):
        if config is not None and config_values:
            raise ValueError('Pass either config or keyword configuration values, not both')
        raw = config if config is not None else config_values
        self.config = raw if isinstance(raw, LangfuseTrajectoryConfig) else LangfuseTrajectoryConfig.from_dict(raw)
        self.projector = LangfusePresentationProjector(self.config)
        self._client = client
        self._states: dict[str, _ExportState] = {}
        self._finalized_ids: OrderedDict[str, None] = OrderedDict()
        self._disabled = False
        self._dirty = False
        self._lock = threading.RLock()

    def on_trajectory_reset(self, trajectory_id: str) -> None:
        with self._lock:
            state = self._states.pop(trajectory_id, None)
            if state is not None:
                self._end_all(state)
            self._finalized_ids.pop(trajectory_id, None)

    def on_update(self, update: TrajectoryUpdate) -> None:
        with self._lock:
            if update.trajectory_id in self._finalized_ids or not self._ensure_client():
                return
            state = self._states.get(update.trajectory_id)
            if state is not None and update.revision <= state.last_revision:
                return
            state = state or self._new_state(update.trajectory_id)
            self._states[update.trajectory_id] = state
            self._apply(update.snapshot, state, final=False)
            state.last_revision = update.revision

    def on_final(self, trajectory: Trajectory) -> None:
        with self._lock:
            trajectory_id = trajectory.session_id
            if trajectory_id in self._finalized_ids or not self._ensure_client():
                return
            state = self._states.get(trajectory_id) or self._new_state(trajectory_id)
            self._states[trajectory_id] = state
            self._apply(trajectory, state, final=True)
            self._states.pop(trajectory_id, None)
            self._finalized_ids[trajectory_id] = None
            while len(self._finalized_ids) > self._MAX_FINALIZED_IDS:
                self._finalized_ids.popitem(last=False)

    def close(self) -> None:
        with self._lock:
            for state in self._states.values():
                self._end_all(state)
            self._states.clear()
            self._finalized_ids.clear()
            if self._client is not None and self._dirty:
                self._flush()

    def _apply(self, trajectory: Trajectory, state: _ExportState, *, final: bool) -> None:
        presentation = self.projector.project(trajectory, final=final)
        completed: list[str] = []
        for spec in presentation.observations:
            fingerprint = self._fingerprint(spec)
            if spec.key in state.ended:
                continue
            if state.fingerprints.get(spec.key) != fingerprint:
                try:
                    self._upsert(state, spec, presentation)
                    state.fingerprints[spec.key] = fingerprint
                    self._dirty = True
                except Exception:
                    logger.warning('Failed to export Langfuse presentation node %s', spec.key, exc_info=True)
                    continue
            if spec.complete:
                completed.append(spec.key)
        for key in reversed(completed):
            self._end(state, key)
        if not final:
            return
        self._end_all(state)
        if presentation.score is not None:
            try:
                self._client.create_score(
                    trace_id=state.trace_id,
                    name='episode_reward',
                    value=presentation.score,
                    comment=presentation.score_comment,
                )
                self._dirty = True
            except Exception:
                logger.warning('Failed to upload Gateway trajectory reward', exc_info=True)
        if self.config.flush_on_final and self._dirty:
            self._flush()

    def _upsert(self, state: _ExportState, spec: _Observation, presentation: _Presentation) -> None:
        handle = state.handles.get(spec.key)
        if handle is not None:
            handle.update(**spec.payload())
            return
        parent = state.handles.get(spec.parent_key or '')
        if spec.parent_key is not None and parent is None:
            raise RuntimeError(f'Langfuse parent {spec.parent_key!r} was not created before {spec.key!r}')
        if parent is not None:
            handle = parent.start_observation(name=spec.name, as_type=spec.kind, **spec.payload())
        else:
            client = cast(Any, self._client)
            with self._trace_attributes(presentation):
                handle = client.start_observation(
                    trace_context=self._trace_context(state.trace_id),
                    name=spec.name,
                    as_type=spec.kind,
                    **spec.payload(),
                )
        state.handles[spec.key] = handle
        state.order.append(spec.key)

    def _new_state(self, trajectory_id: str) -> _ExportState:
        seed = f'{trajectory_id}::{uuid.uuid4().hex}'
        creator = getattr(self._client, 'create_trace_id', None)
        trace_id = (
            creator(seed=seed) if callable(creator) else hashlib.blake2b(seed.encode(), digest_size=16).hexdigest()
        )
        return _ExportState(trace_id=trace_id)

    @staticmethod
    def _trace_context(trace_id: str) -> Any:
        try:
            from langfuse.types import TraceContext

            return TraceContext(trace_id=trace_id, parent_span_id='')
        except ImportError:
            return {'trace_id': trace_id, 'parent_span_id': ''}

    @staticmethod
    def _trace_attributes(presentation: _Presentation) -> Any:
        try:
            from langfuse import propagate_attributes

            return propagate_attributes(
                tags=list(presentation.tags),
                session_id=presentation.session_id or None,
                trace_name=presentation.trace_name or None,
            )
        except ImportError:
            return contextlib.nullcontext()

    def _end(self, state: _ExportState, key: str) -> None:
        if key in state.ended or key not in state.handles:
            return
        try:
            state.handles[key].end()
            state.ended.add(key)
            self._dirty = True
        except Exception:
            logger.warning('Failed to end Langfuse presentation node %s', key, exc_info=True)

    def _end_all(self, state: _ExportState) -> None:
        for key in reversed(state.order):
            self._end(state, key)

    def _ensure_client(self) -> bool:
        if self._disabled:
            return False
        if self._client is not None:
            return True
        public_key = self.config.public_key or os.environ.get('LANGFUSE_PUBLIC_KEY', '')
        secret_key = self.config.secret_key or os.environ.get('LANGFUSE_SECRET_KEY', '')
        host = (
            self.config.host
            or os.environ.get('LANGFUSE_HOST', '')
            or os.environ.get('LANGFUSE_BASE_URL', '')
            or 'https://cloud.langfuse.com'
        )
        if not public_key or not secret_key:
            logger.warning('Langfuse credentials not configured, trajectory observer disabled')
            self._disabled = True
            return False
        try:
            from langfuse import Langfuse

            self._client = Langfuse(public_key=public_key, secret_key=secret_key, host=host)
            return True
        except Exception:
            logger.warning('Failed to initialize Langfuse trajectory observer', exc_info=True)
            self._disabled = True
            return False

    def _flush(self) -> None:
        try:
            self._client.flush()
            self._dirty = False
        except Exception:
            logger.warning('Failed to flush Langfuse trajectory observer', exc_info=True)

    @staticmethod
    def _fingerprint(spec: _Observation) -> str:
        encoded = json.dumps(
            {
                'parent': spec.parent_key,
                'kind': spec.kind,
                'name': spec.name,
                'payload': spec.payload(),
                'complete': spec.complete,
            },
            sort_keys=True,
            ensure_ascii=False,
            default=str,
        ).encode()
        return hashlib.sha256(encoded).hexdigest()
