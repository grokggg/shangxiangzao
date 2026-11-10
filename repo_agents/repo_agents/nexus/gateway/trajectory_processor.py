"""Finalize a Gateway session through the standard span processor lifecycle."""

from __future__ import annotations

from typing import Any

from nexus.gateway.nexus_run_tracker import NexusRunTracker
from nexus.gateway.session_manager import session_manager
from nexus.trace.events import SpanEndEvent
from nexus.trace.processors.base import span_processor_registry
from nexus.trace.types import Span


@span_processor_registry.register('gateway_trajectory')
class GatewayTrajectoryProcessor(NexusRunTracker):
    """Return one canonical Gateway trajectory as a processor result."""

    def __init__(self, *, trajectory_id: str) -> None:
        if not trajectory_id:
            raise ValueError('GatewayTrajectoryProcessor requires trajectory_id')
        super().__init__(trajectory_id=trajectory_id)
        self._terminal_messages: list[dict[str, Any]] | None = None
        self._task_output: dict[str, Any] | None = None
        self._result: dict[str, Any] | None = None

    def on_end(self, span: Span, event: SpanEndEvent) -> None:
        super().on_end(span, event)
        span_type = span.attributes.get('type')
        output = span.attributes.get('outputs')

        if span_type == 'agent' and isinstance(output, dict):
            messages = output.get('main_trajectory')
            if isinstance(messages, list):
                self._terminal_messages = messages
        elif span_type == 'task':
            if isinstance(output, dict):
                self._task_output = output
            if span.status.code == 'ERROR' and self.termination_reason is None:
                self.termination_reason = 'UNHANDLED_ERROR'

    def finalize(self) -> None:
        trajectory_id = self._trajectory_id
        if trajectory_id is None:
            raise ValueError('trajectory_id must not be None before finalizing.')
        if not session_manager.has_session(trajectory_id):
            self._result = None
            return

        self._import_terminal_observations(self._terminal_messages)
        termination_reason = self.termination_reason or 'ENV_DONE'
        task_output = self._task_output or {}
        final_reward = task_output.get('score')
        extra = task_output.get('extra_fields')
        trajectory = session_manager.end_session(trajectory_id, termination_reason=termination_reason)
        trajectory = self.finalize_trajectory(
            trajectory,
            trajectory_id,
            termination_reason=termination_reason,
            final_reward=final_reward,
            extra=extra if isinstance(extra, dict) else None,
        )
        if trajectory is None:
            raise RuntimeError('finalize_trajectory returned None, cannot generate result')
        self._result = trajectory.dump()

    def get_result(self) -> dict[str, Any] | None:
        return self._result
