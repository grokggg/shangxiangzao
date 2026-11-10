"""GatewaySessionManager: trajectory_id -> GatewaySession mapping with lazy creation.

The module-level ``session_manager`` below is THE process-wide singleton: every
GatewayOpenAIClient (one per trajectory) and every runner-level lifecycle call
(end_session / leak accounting) goes through it. Sessions store per-trajectory
data; the manager itself holds no model I/O state — each session receives its
Processor from the client that creates it.
"""

from nexus.gateway.processor import Processor
from nexus.gateway.scaffold_policy import ScaffoldPolicy
from nexus.gateway.session import GatewaySession
from nexus.gateway.trajectory import Trajectory


class GatewaySessionManager:
    """Manages trajectory_id -> GatewaySession mapping.

    Sessions are created lazily on first access via get_or_create().
    Control flow does not need to explicitly call create_session().
    """

    def __init__(self):
        self._sessions: dict[str, GatewaySession] = {}

    def get_or_create(
        self,
        trajectory_id: str,
        *,
        processor: Processor | None = None,
        metadata: dict | None = None,
        scaffold_policy: ScaffoldPolicy | str | None = None,
    ) -> GatewaySession:
        """Get existing session or create a new one for the given trajectory_id.

        ``processor`` is required on the creating call (the per-trajectory client
        passes its own); subsequent lookups may omit it.
        """
        session = self._sessions.get(trajectory_id)
        if session is None:
            if processor is None:
                raise ValueError(
                    f'No active session for trajectory_id={trajectory_id!r} and no processor '
                    'given to create one (lookups without processor require an existing session)'
                )
            session = GatewaySession(
                session_id=trajectory_id,
                processor=processor,
                metadata=metadata,
                scaffold_policy=scaffold_policy,
            )
            self._sessions[trajectory_id] = session
        elif processor is not None and session._processor is not processor:
            raise RuntimeError(f'Active Gateway session already owns trajectory_id={trajectory_id!r}')
        return session

    def get_session(self, trajectory_id: str) -> GatewaySession | None:
        """Return the active session for trajectory_id without creating one."""
        return self._sessions.get(trajectory_id)

    def has_session(self, trajectory_id: str) -> bool:
        return trajectory_id in self._sessions

    async def abort_in_flight(self, trajectory_id: str) -> list[str]:
        """Cancel a session's orphaned in-flight requests before ending it.

        No-op (empty list) when the session does not exist or nothing is pending.
        See :meth:`GatewaySession.abort_in_flight`.
        """
        session = self._sessions.get(trajectory_id)
        if session is None:
            return []
        return await session.abort_in_flight()

    def end_session(
        self,
        trajectory_id: str,
        termination_reason: str | None = None,
        *,
        final_reward: float | None = None,
        metrics: dict | None = None,
        extra: dict | None = None,
    ) -> Trajectory:
        """End a session and return its trajectory. Removes it from active sessions."""
        session = self._sessions.get(trajectory_id)
        if session is None:
            raise ValueError(f'No active session for trajectory_id={trajectory_id}')
        try:
            return session.end(
                termination_reason=termination_reason,
                final_reward=final_reward,
                metrics=metrics,
                extra=extra,
            )
        finally:
            self._sessions.pop(trajectory_id, None)

    def end_all(self, termination_reason: str | None = None) -> dict[str, Trajectory]:
        """End all active sessions and return them by trajectory id."""
        trajectories = {}
        first_error: BaseException | None = None
        for tid in list(self._sessions.keys()):
            try:
                trajectories[tid] = self.end_session(tid, termination_reason)
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise first_error
        return trajectories

    @property
    def active_count(self) -> int:
        return len(self._sessions)


# Process-wide singleton. Import this, not the class, for all runtime use.
session_manager = GatewaySessionManager()
