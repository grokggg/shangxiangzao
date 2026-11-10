"""Bridge white-box run spans into trajectory-level facts."""

import copy
import json
import logging

from typing import Any

from nexus.gateway.observers.hub import get_current_trajectory_observation_hub
from nexus.gateway.scaffold_policy import protocol_message_identity
from nexus.gateway.session_manager import session_manager
from nexus.gateway.trajectory import Trajectory
from nexus.trace.events import SpanEndEvent, SpanStartEvent
from nexus.trace.processors.base import SpanProcessor
from nexus.trace.types import Span
from nexus.utils.tool_analyzer import build_turn_stats_from_messages, summarize_tool_call_stats


logger = logging.getLogger(__name__)


class NexusRunTracker(SpanProcessor):
    _TIMELINE_TYPES = {'task_phase', 'agent_inference', 'agent_observation', 'agent_compaction'}

    def __init__(
        self,
        progress_state: dict[str, Any] | None = None,
        *,
        trajectory_id: str | None = None,
    ):
        self._progress_state = progress_state
        self._trajectory_id = trajectory_id
        self.task_name: str | None = None
        self.termination_reason: str | None = None
        self.observation_time = 0.0
        self.reward_time = 0.0
        self.runtime_id: str | None = None
        self.runtime_ids: list[str] = []
        self.eval_runtime_id: str | None = None
        self.agent_timeline: list[dict[str, Any]] = []
        self.context_ready = False

    def on_start(self, span: Span, event: SpanStartEvent) -> None:
        del event
        if span.attributes.get('type') == 'task':
            self.task_name = span.attributes.get('name')

    def on_end(self, span: Span, event: SpanEndEvent) -> None:
        del event
        attributes = span.attributes
        span_type = attributes.get('type')
        if (
            span_type in self._TIMELINE_TYPES
            and span.end_time_unix_nano is not None
            and (span_type != 'agent_compaction' or attributes.get('compacted'))
        ):
            name = span.name
            timeline_type = span_type
            if span_type == 'agent_observation':
                name = attributes.get('tool_name', 'observation')
            elif span_type == 'agent_inference':
                name = 'gen'
            elif span_type == 'agent_compaction':
                name = 'compaction'
            elif span_type == 'task_phase':
                timeline_type = self.task_name
                name = attributes.get('phase', span.name)
            item = {
                'type': timeline_type,
                'name': name,
                'start_time': span.start_time_unix_nano / 1e9,
                'end_time': span.end_time_unix_nano / 1e9,
            }
            tool_args = attributes.get('hytuner.tool_args')
            if tool_args is not None:
                try:
                    item['tool_args'] = json.loads(tool_args) if isinstance(tool_args, str) else tool_args
                except json.JSONDecodeError:
                    item['tool_args'] = tool_args
            self.agent_timeline.append(item)

        reason = attributes.get('hytuner.termination_reason') or attributes.get('agent.termination_reason')
        if reason:
            self.termination_reason = reason
        observation_time = attributes.get('hytuner.observation_time') or attributes.get('agent.observation_time')
        if observation_time is not None:
            self.observation_time += float(observation_time)
        reward_time = attributes.get('hytuner.reward_time')
        if reward_time is not None:
            self.reward_time += float(reward_time)
        self._record_runtime(attributes.get('hytuner.runtime_id'))
        eval_runtime_id = attributes.get('hytuner.eval_runtime_id')
        if eval_runtime_id:
            self.eval_runtime_id = eval_runtime_id
            if self._progress_state is not None:
                self._progress_state['eval_runtime_id'] = eval_runtime_id

    async def capture_context(self, context: Any) -> None:
        self.context_ready = True
        self._record_runtime(context.get_runtime_id('main'))
        eval_runtime_id = context.get_runtime_id('eval')
        if eval_runtime_id:
            self.eval_runtime_id = eval_runtime_id
            if self._progress_state is not None:
                self._progress_state['eval_runtime_id'] = eval_runtime_id

    async def capture_run_output(self, context: Any, output: Any) -> None:
        await self.capture_context(context)
        messages = getattr(getattr(output, 'agent_output', None), 'main_trajectory', None)
        self._import_terminal_observations(messages)

    def _import_terminal_observations(self, messages: Any) -> None:
        if self._trajectory_id is None or not isinstance(messages, list):
            return
        session = session_manager.get_session(self._trajectory_id)
        if session is None:
            return

        recorded = session.get_trajectory().to_messages()
        extends_recorded = len(messages) >= len(recorded) and all(
            isinstance(reported, dict) and protocol_message_identity(reported) == protocol_message_identity(canonical)
            for reported, canonical in zip(messages, recorded, strict=False)
        )
        if not extends_recorded:
            logger.warning(
                'Gateway run output trajectory does not extend the recorded transcript for %s; '
                'terminal observations were not imported',
                self._trajectory_id,
            )
            return
        observations = messages[len(recorded) :]
        if not observations:
            return
        if any(
            not isinstance(message, dict) or message.get('role') not in {'tool', 'user'} for message in observations
        ):
            logger.warning(
                'Gateway run output for %s has an unsupported terminal input suffix; it was not imported',
                self._trajectory_id,
            )
            return

        parent_messages = copy.deepcopy(recorded)
        for observation in observations:
            session.record_observation(message=observation, parent_messages=parent_messages)
            parent_messages.append(copy.deepcopy(observation))

    def finalize_trajectory(
        self,
        trajectory: Trajectory | None,
        trajectory_id: str,
        *,
        termination_reason: str,
        final_reward: float | None = None,
        extra: dict[str, Any] | None = None,
        allow_empty: bool = False,
    ) -> Trajectory | None:
        trajectory_extra = copy.deepcopy(extra or {})
        if self.runtime_id:
            trajectory_extra['runtime_id'] = self.runtime_id
        if self.runtime_ids:
            trajectory_extra['runtime_ids'] = list(self.runtime_ids)
        trajectory_extra['eval_runtime_id'] = self.eval_runtime_id or 'unknown'
        trajectory_extra['agent_timeline'] = copy.deepcopy(self.agent_timeline)
        metrics = {
            'abort_wait_time': 0.0,
            'llm_inference_time': 0.0,
            'observation_time': self.observation_time,
            'reward_time': self.reward_time,
        }

        if trajectory is None and (self.context_ready or allow_empty):
            trajectory = Trajectory(
                session_id=trajectory_id,
                latest_messages=[],
                final_reward=final_reward,
                termination_reason=termination_reason,
                metrics=metrics,
                extra=trajectory_extra,
            )
        elif trajectory is None:
            return None

        metrics['abort_wait_time'] = sum(
            float(node.provider_facts.abort_wait_time or 0.0)
            for node in trajectory.nodes.values()
            if node.provider_facts is not None
        )
        metrics['llm_inference_time'] = sum(
            float(node.provider_facts.inference_time or 0.0)
            for node in trajectory.nodes.values()
            if node.provider_facts is not None
        )

        if trajectory.latest_messages is None:
            logger.warning(
                'Gateway trajectory %s has no latest_messages; tool statistics will use an empty transcript',
                trajectory_id,
            )
        messages = copy.deepcopy(trajectory.latest_messages or [])
        try:
            tool_call_stats = self._tool_call_stats(messages)
            trajectory_extra['tool_stats'] = self._tool_stats(messages)
        except Exception:
            logger.exception('Failed to build tool statistics for Gateway trajectory %s', trajectory_id)
            tool_call_stats = []
            trajectory_extra['tool_stats'] = {}
        if tool_call_stats:
            trajectory_extra['tool_call_stats'] = tool_call_stats
            trajectory_extra['tool_call_summary'] = summarize_tool_call_stats(tool_call_stats)

        trajectory.metrics = copy.deepcopy(metrics)
        trajectory.extra.update(copy.deepcopy(trajectory_extra))
        trajectory.termination_reason = termination_reason
        trajectory.final_reward = final_reward
        hub = get_current_trajectory_observation_hub()
        if hub is not None:
            hub.publish_final(trajectory)
        return trajectory

    def _tool_call_stats(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
        starts = [index for index, message in enumerate(messages) if message.get('role') == 'assistant']
        stats = []
        for offset, start in enumerate(starts):
            end = starts[offset + 1] if offset + 1 < len(starts) else len(messages)
            turn = build_turn_stats_from_messages(messages[start:end], task_name=self.task_name)
            if turn is None:
                return None
            stats.append(turn)
        return stats

    @staticmethod
    def _tool_stats(messages: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
        # ***** FOLLOWS HytunerProcessor._parse_tool_stats (nexus/trace/processors/
        # hytuner.py) — stat shape and failure classification must stay identical or
        # the two lanes' ``tool_stats`` diverge; the shared classifiers are imported
        # from that module instead of re-implemented here.
        from nexus.trace.processors.hytuner import _classify_visit_failure, _is_tool_failure

        def _empty_bucket() -> dict[str, int]:
            return {'total': 0, 'success': 0, 'failure': 0, 'failure_readpage': 0, 'failure_summary_llm': 0}

        stats: dict[str, dict[str, int]] = {}
        call_names: dict[str, str] = {}
        for message in messages:
            if message.get('role') == 'assistant':
                for call in message.get('tool_calls') or []:
                    name = call.get('function', {}).get('name', 'unknown')
                    call_names[call.get('id', '')] = name
                    stats.setdefault(name, _empty_bucket())['total'] += 1
            elif message.get('role') == 'tool':
                name = call_names.get(message.get('tool_call_id', ''), message.get('name', 'unknown'))
                bucket = stats.setdefault(name, _empty_bucket())
                content = str(message.get('content', ''))
                if _is_tool_failure(name, content):
                    bucket['failure'] += 1
                    if name == 'visit':
                        bucket[f'failure_{_classify_visit_failure(content)}'] += 1
                else:
                    bucket['success'] += 1
        # ***** END FOLLOWS (per-tool stats parity with the legacy lane).
        return stats

    def _record_runtime(self, runtime_id: str | None) -> None:
        if not runtime_id:
            return
        self.runtime_id = runtime_id
        if runtime_id not in self.runtime_ids:
            self.runtime_ids.append(runtime_id)
        if self._progress_state is not None:
            self._progress_state['runtime_id'] = runtime_id
