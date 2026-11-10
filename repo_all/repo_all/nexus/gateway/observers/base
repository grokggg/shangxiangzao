from __future__ import annotations

from dataclasses import dataclass

from nexus.gateway.trajectory import Trajectory


@dataclass(frozen=True)
class TrajectoryUpdate:
    """One immutable-in-practice view of a live canonical Trajectory Tree."""

    snapshot: Trajectory
    revision: int

    @property
    def trajectory_id(self) -> str:
        return self.snapshot.session_id


class TrajectoryObserver:
    """Read-only, best-effort consumer of canonical Trajectory Trees.

    One observer may receive interleaved publications from many trajectories.
    Implementations must isolate mutable projection state by ``session_id``.
    """

    def on_update(self, update: TrajectoryUpdate) -> None:
        """Consume an isolated active Trajectory snapshot."""

    def on_final(self, trajectory: Trajectory) -> None:
        """Reconcile one finalized canonical Trajectory."""

    def on_trajectory_reset(self, trajectory_id: str) -> None:
        """Discard stale state before an explicit trajectory id is reused."""

    def close(self) -> None:
        """Release observer-owned process or execution-context resources."""
