"""GatewaySession: per-trajectory interaction container.

Composes Processor + TrajectoryAssembler to manage a single trajectory's model
interactions and trace assembly.
"""

import asyncio
import copy
import logging
import threading
import time

from typing import Any

from nexus.gateway.observers.base import TrajectoryUpdate
from nexus.gateway.observers.hub import get_current_trajectory_observation_hub
from nexus.gateway.processor import Processor
from nexus.gateway.scaffold_policy import ScaffoldPolicy
from nexus.gateway.token_budget import TokenBudgetState
from nexus.gateway.trajectory import Trajectory
from nexus.gateway.trajectory_assembler import TrajectoryAssembler
from nexus.gateway.types import GatewayCompletionResult
from nexus.utils.token_budget_config import TokenBudgetComponent


logger = logging.getLogger(__name__)


logger = logging.getLogger(__name__)


class GatewaySession:
    """Per-trajectory interaction container.

    Each session corresponds to one trajectory_id (e.g., one agent episode).
    It owns a TrajectoryAssembler (token-level trace) and delegates model I/O to a
    shared Processor.

    The session also owns the per-trajectory ``TokenBudgetState`` (legacy
    HyTunerClient budget semantics). Per-run budget parameters ride in via
    ``metadata['token_budget_config']`` carries the model limit, trajectory budget,
    and included token sources.
    """

    def __init__(
        self,
        session_id: str,
        processor: Processor,
        metadata: dict | None = None,
        scaffold_policy: ScaffoldPolicy | str | None = None,
    ):
        self.session_id = session_id
        hub = get_current_trajectory_observation_hub()
        if hub is not None:
            hub.begin_trajectory(session_id)
        self._processor = processor
        self._segment_assembler = TrajectoryAssembler(scaffold_policy=scaffold_policy, session_id=session_id)
        session_metadata = copy.deepcopy(metadata or {})
        token_budget_config = session_metadata.pop('token_budget_config', {})
        self._metadata = session_metadata
        self._ended = False
        self._termination_reason: str | None = None
        self._latest_messages: list[dict] = []
        self._timeline: list[dict] = []
        self._lock = threading.RLock()
        self._in_flight = 0
        self._pending_completions: dict[asyncio.Task, str] = {}
        self._aborted_request_labels: list[str] = []
        self._token_budget_state = TokenBudgetState(
            max_model_len=token_budget_config.get('max_model_len'),
            token_budget=token_budget_config.get('token_budget'),
            token_budget_include=token_budget_config.get('token_budget_include'),
        )

    @property
    def is_ended(self) -> bool:
        return self._ended

    @property
    def segment_assembler(self) -> TrajectoryAssembler:
        return self._segment_assembler

    async def create_completion(
        self,
        messages: list[dict],
        request_kwargs: dict[str, Any] | None = None,
        priority: int | None = None,
        uses_default_max_tokens: bool = True,
        request_meta: dict[str, Any] | None = None,
    ) -> GatewayCompletionResult:
        """Run a single completion through the processor and record the trace."""
        with self._lock:
            if self._ended:
                raise RuntimeError(f'Session {self.session_id} already ended')
            recording_messages = copy.deepcopy(messages)
            execution_messages = copy.deepcopy(messages)
            request_snapshot = copy.deepcopy(request_kwargs)
            execution_request = copy.deepcopy(request_kwargs)
            recording_meta = copy.deepcopy(request_meta)
            execution_meta = copy.deepcopy(request_meta)
            tools = request_snapshot.get('tools') if request_snapshot is not None else None
            request_facts = self._segment_assembler.extract_context(
                recording_messages,
                tools=tools,
                request_meta=recording_meta,
            )
            context_key = request_facts.context_key
            frozen_prefix = self._segment_assembler.frozen_prefix(
                recording_messages,
                tools=tools,
                context=request_facts,
            )
            self._in_flight += 1
            task = asyncio.current_task()
            if task is not None:
                request_label = (
                    str(request_facts.request_id)
                    if request_facts.request_id is not None
                    else f'{request_facts.agent_id or "main"}#{request_facts.sequence}'
                )
                self._pending_completions[task] = request_label
        t_start = time.time()
        try:
            processor_kwargs: dict[str, Any] = {
                'request_id': self.session_id,
                'messages': execution_messages,
                'request_kwargs': execution_request,
                'priority': priority,
            }
            if isinstance(self._processor, Processor):
                processor_kwargs['frozen_prefix'] = frozen_prefix
                processor_kwargs['prepare_generation'] = lambda *args: self._token_budget_state.prepare_generation(
                    context_key, *args
                )
                processor_kwargs['uses_default_max_tokens'] = uses_default_max_tokens
                processor_kwargs['request_meta'] = execution_meta
            result = await self._processor.create_completion(**processor_kwargs)
            t_end = time.time()
            completion = GatewayCompletionResult(
                message=result.message,
                token_ids=result.token_ids,
                log_probs=result.log_probs,
                routed_experts=result.routed_experts,
                stop_reason=result.stop_reason,
                usage=result.usage,
                id=result.id,
                created=result.created,
                model=result.model,
                object=result.object,
                system_fingerprint=result.system_fingerprint,
                service_tier=result.service_tier,
                choice_logprobs=result.choice_logprobs,
                choice_provider_specific_fields=result.choice_provider_specific_fields,
                extra_fields=result.extra_fields,
                inference_time=getattr(result, 'inference_time', None),
            )
            token_training = None
            if self._processor.is_token_level:
                token_training = {
                    'prompt_ids': result.prompt_ids or [],
                    'raw_prompt_ids': result.raw_prompt_ids or [],
                    'response_ids': result.token_ids or [],
                    'response_logprobs': result.log_probs,
                    'routed_experts': result.routed_experts,
                    'extra_fields': result.extra_fields,
                }
            backend_abort_wait = float((result.extra_fields or {}).get('_abort_wait_time') or 0.0)
            with self._lock:
                if self._processor.is_token_level:
                    invocation_kind = (
                        execution_meta.get('invocation_kind') if isinstance(execution_meta, dict) else None
                    )
                    self._token_budget_state.record_generation(
                        context_key,
                        list(result.raw_prompt_ids or []),
                        list(result.prompt_ids or []),
                        list(result.token_ids or []),
                        output_component=(
                            TokenBudgetComponent.COMPACTION_OUTPUT
                            if invocation_kind == 'summary_inference'
                            else TokenBudgetComponent.AGENT_OUTPUT
                        ),
                    )
                self._segment_assembler.record_completion(
                    messages=recording_messages,
                    output_message=result.message,
                    request_kwargs=request_snapshot,
                    stop_reason=result.stop_reason,
                    token_training=token_training,
                    response_meta={
                        'model': result.model,
                        'response_id': result.id,
                        'usage': result.usage,
                        'inference_time': getattr(result, 'inference_time', None),
                        'abort_wait_time': backend_abort_wait,
                        'num_preempted': (result.extra_fields or {}).get('_num_preempted'),
                    },
                    completion_text=result.completion_text,
                    prompt_render=result.prompt_render,
                    context=request_facts,
                    request_meta=recording_meta,
                    start_time=t_start,
                    end_time=t_end,
                )
                self._latest_messages = copy.deepcopy(recording_messages) + [copy.deepcopy(result.message)]
                self._record_timeline_event(
                    'inference',
                    'create_completion',
                    t_start,
                    t_end,
                    request_id=request_facts.request_id,
                    agent_id=request_facts.agent_id,
                    is_sidechain=request_facts.membership is not None,
                    iteration=request_facts.iteration,
                    retry=request_facts.retry,
                    sequence=request_facts.sequence,
                )
                observation_snapshot = self._observation_snapshot_locked()
            self._publish_observation_snapshot(observation_snapshot)
            return completion
        finally:
            with self._lock:
                self._in_flight -= 1
                if task is not None:
                    self._pending_completions.pop(task, None)

    def update_token_consumption(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        request_meta: dict[str, Any] | None = None,
    ) -> None:
        if not self._processor.is_token_level:
            raise RuntimeError('Token consumption tracking requires a token-level processor')
        context = self._segment_assembler.extract_context(messages, tools, request_meta)
        context_key = context.context_key
        # Same ruler as the request path: the accounting must measure the frozen
        # sampled stream + rendered tail the next request actually resends. A fresh
        # canonical render is not a prefix-extension of the spliced stream whenever an
        # emission violates render(parse(x)) == x, which would break update_prompt's
        # extends-previous accounting (double count).
        frozen_prefix = self._segment_assembler.frozen_prefix(messages, tools=tools, context=context)
        raw_ids, expanded_ids, _, _ = self._processor._render_and_tokenize(messages, tools, frozen_prefix=frozen_prefix)
        self._token_budget_state.update_prompt(context_key, raw_ids, expanded_ids)

    def segment_consumed_tokens(self, request_meta: dict[str, Any] | None = None) -> int:
        context = self._segment_assembler.extract_context([], request_meta=request_meta)
        return self._token_budget_state.segment_consumed_tokens(context.context_key)

    def set_segment_consumed_tokens(
        self,
        value: int,
        request_meta: dict[str, Any] | None = None,
    ) -> None:
        context = self._segment_assembler.extract_context([], request_meta=request_meta)
        self._token_budget_state.set_segment_consumed_tokens(context.context_key, value)

    def record_observation(
        self,
        content: str | None = None,
        token_ids: list[int] | None = None,
        role: str = 'tool',
        parent_messages: list[dict] | None = None,
        request_meta: dict[str, Any] | None = None,
        start_time: float | None = None,
        end_time: float | None = None,
        status: str | None = None,
        message: dict[str, Any] | None = None,
    ) -> None:
        """Record an observation (tool/env output) in the trajectory.

        Called by control flow to inform Gateway about non-LLM tokens.
        """
        with self._lock:
            wire_messages = self._segment_assembler.record_observation(
                content=content,
                token_ids=token_ids,
                role=role,
                message=message,
                parent_messages=parent_messages,
                request_meta=request_meta,
                start_time=start_time,
                end_time=end_time,
                status=status,
            )
            observation = copy.deepcopy(message) if message is not None else {'role': role, 'content': content}
            observation.setdefault('role', role)
            if parent_messages is not None:
                self._latest_messages = copy.deepcopy(parent_messages) + [observation]
            else:
                self._latest_messages = wire_messages
            observation_snapshot = self._observation_snapshot_locked()
        self._publish_observation_snapshot(observation_snapshot)

    def finalize_segment(self) -> None:
        """Mark a segment boundary (e.g., context compaction)."""
        with self._lock:
            self._segment_assembler.finalize_segment()
            observation_snapshot = self._observation_snapshot_locked()
        self._publish_observation_snapshot(observation_snapshot)

    def get_trajectory(self, *, state: str | None = None, copy_inputs: bool = True) -> Trajectory:
        """Resolve the trajectory, copying live inputs unless explicitly disabled.

        ``copy_inputs=False`` mutates and returns the session's live trajectory.
        """
        from nexus.gateway.trajectory_freeze import freeze_trajectory

        with self._lock:
            inputs = (
                self._segment_assembler.trajectory,
                self._segment_assembler._runtime_state,
                self._timeline,
                self._latest_messages,
                self._metadata,
                self._ended,
                self._termination_reason,
            )
            if copy_inputs:
                inputs = copy.deepcopy(inputs)
            (
                live_trajectory,
                runtime_state,
                timeline,
                latest_messages,
                metadata,
                ended,
                termination_reason,
            ) = inputs
            response_policy = self._segment_assembler._policy.response_policy
            resolved_state = state if state is not None else ('completed' if ended else 'active')

            def materialize() -> Trajectory:
                return freeze_trajectory(
                    live_trajectory,
                    runtime_state,
                    response_policy,
                    state=resolved_state,
                    session_timeline=timeline,
                    latest_messages=latest_messages,
                    termination_reason=termination_reason,
                    extra=metadata,
                    copy_inputs=False,
                )

            if not copy_inputs:
                return materialize()
        return materialize()

    def end(
        self,
        termination_reason: str | None = None,
        *,
        final_reward: float | None = None,
        metrics: dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> Trajectory:
        """End the session and return its canonical trajectory."""
        with self._lock:
            if self._in_flight:
                raise RuntimeError(f'Session {self.session_id} has {self._in_flight} in-flight request(s)')
            trajectory = self.get_trajectory(state='completed', copy_inputs=False)
            trajectory.final_reward = final_reward
            trajectory.termination_reason = termination_reason
            trajectory.metrics = copy.deepcopy(metrics or {})
            trajectory.extra.update(copy.deepcopy(extra or {}))
            if self._aborted_request_labels:
                trajectory.extra['aborted_in_flight_requests'] = list(self._aborted_request_labels)
            self._ended = True
            self._termination_reason = termination_reason
            return trajectory

    async def abort_in_flight(self) -> list[str]:
        """Cancel and drain completions whose consumer no longer exists.

        For teardown paths where the scaffold process is known to be gone (killed on
        timeout, runtime lost): its pending completions can never be consumed, so this
        does NOT wait for the backend to finish generating them. Cancellation is
        native asyncio task cancellation: a CancelledError raised at each task's await
        point unwinds the backend call (closing its connection and freeing the slot)
        and runs the ``finally`` that releases the in-flight count. ``end()``
        afterwards sees a quiescent session and keeps its strict no-in-flight
        contract; a session that was NOT aborted still fails fast on a leaked
        request. Returns the aborted request labels (also recorded on the trajectory
        as ``extra['aborted_in_flight_requests']`` at end).
        """
        # Snapshot the outstanding completions under the lock. Skip our own task (an
        # abort issued from inside a completion must not cancel itself) and anything
        # already finished.
        with self._lock:
            me = asyncio.current_task()
            orphans = {
                task: label for task, label in self._pending_completions.items() if task is not me and not task.done()
            }
        if not orphans:
            return []
        # asyncio cancellation is two-phase: cancel() only schedules a CancelledError
        # at each task's await point...
        for task in orphans:
            task.cancel()
        # ...and this is the second phase: wait until every cancelled task has
        # unwound, running the finally that releases its in-flight count.
        # return_exceptions=True collects their CancelledErrors instead of re-raising
        # them here. After this line the session is quiescent.
        await asyncio.gather(*orphans, return_exceptions=True)
        labels = sorted(orphans.values())
        with self._lock:
            self._aborted_request_labels.extend(labels)
        logger.warning(
            'Session %s: aborted %d in-flight request(s): %s',
            self.session_id,
            len(labels),
            labels,
        )
        return labels

    def _observation_snapshot_locked(self) -> TrajectoryUpdate | None:
        hub = get_current_trajectory_observation_hub()
        if hub is None:
            return None
        revision = hub.reserve_update(self.session_id)
        if revision is None:
            return None
        return TrajectoryUpdate(snapshot=self.get_trajectory(copy_inputs=True), revision=revision)

    @staticmethod
    def _publish_observation_snapshot(update: TrajectoryUpdate | None) -> None:
        hub = get_current_trajectory_observation_hub()
        if update is not None and hub is not None:
            hub.publish_update(update)

    def _record_timeline_event(
        self,
        event_type: str,
        name: str,
        start: float,
        end: float,
        **meta: Any,
    ) -> None:
        clean_meta = {key: value for key, value in meta.items() if value is not None}
        self._timeline.append({
            'type': event_type,
            'name': name,
            'start': start,
            'end': end,
            'duration': end - start,
            **clean_meta,
        })
