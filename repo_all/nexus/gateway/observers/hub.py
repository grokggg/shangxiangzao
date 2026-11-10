from __future__ import annotations

import copy
import logging
import threading

from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from nexus.gateway.observers.base import TrajectoryObserver, TrajectoryUpdate
from nexus.gateway.trajectory import Trajectory


logger = logging.getLogger(__name__)


class TrajectoryObservationHub:
    """Run-scoped multicast boundary between Trajectory Trees and observers."""

    def __init__(self, observers: Iterable[TrajectoryObserver]) -> None:
        unique_observers: list[TrajectoryObserver] = []
        for observer in observers:
            if not any(candidate is observer for candidate in unique_observers):
                unique_observers.append(observer)
        self._observers = tuple(unique_observers)
        self._revisions: dict[str, int] = {}
        self._finalized: set[str] = set()
        self._lock = threading.RLock()
        self._closed = False

    @property
    def has_observers(self) -> bool:
        with self._lock:
            return not self._closed and bool(self._observers)

    def begin_trajectory(self, trajectory_id: str) -> None:
        """Begin a new publication stream, allowing an explicit id to be reused."""
        with self._lock:
            if self._closed:
                return
            self._revisions.pop(trajectory_id, None)
            self._finalized.discard(trajectory_id)
            observers = self._observers
        for observer in observers:
            self._invoke(
                observer,
                'on_trajectory_reset',
                trajectory_id,
                trajectory_id=trajectory_id,
            )

    def reserve_update(self, trajectory_id: str) -> int | None:
        """Reserve an update revision while the producer holds its Tree lock."""
        with self._lock:
            if self._closed or not self._observers or trajectory_id in self._finalized:
                return None
            revision = self._revisions.get(trajectory_id, 0) + 1
            self._revisions[trajectory_id] = revision
            return revision

    def publish_update(self, update: TrajectoryUpdate) -> bool:
        """Multicast an already-versioned active Tree snapshot."""
        with self._lock:
            if self._closed or update.trajectory_id in self._finalized:
                return False
            observers = self._observers
        for observer in observers:
            isolated_update = TrajectoryUpdate(snapshot=copy.deepcopy(update.snapshot), revision=update.revision)
            self._invoke(
                observer,
                'on_update',
                isolated_update,
                trajectory_id=update.trajectory_id,
                revision=update.revision,
            )
        return bool(observers)

    def publish_final(self, trajectory: Trajectory) -> bool:
        """Publish one finalized Trajectory exactly once in this Run."""
        trajectory_id = trajectory.session_id
        with self._lock:
            if self._closed or trajectory_id in self._finalized:
                return False
            self._finalized.add(trajectory_id)
            self._revisions.pop(trajectory_id, None)
            observers = self._observers
        for observer in observers:
            self._invoke(observer, 'on_final', copy.deepcopy(trajectory), trajectory_id=trajectory_id)
        return bool(observers)

    def close(self) -> None:
        """Close this Run's observers once and reject further publications."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            observers = self._observers
            self._observers = ()
            self._revisions.clear()
            self._finalized.clear()
        for observer in observers:
            self._close_observer(observer)

    @staticmethod
    def _invoke(
        observer: TrajectoryObserver,
        method: str,
        value: Any,
        *,
        trajectory_id: str,
        revision: int | None = None,
    ) -> None:
        try:
            getattr(observer, method)(value)
        except Exception:
            suffix = '' if revision is None else f' revision {revision}'
            logger.warning(
                'Trajectory observer %s failed for %s%s',
                type(observer).__name__,
                trajectory_id,
                suffix,
                exc_info=True,
            )

    @staticmethod
    def _close_observer(observer: TrajectoryObserver) -> None:
        try:
            observer.close()
        except Exception:
            logger.warning('Failed to close trajectory observer %s', type(observer).__name__, exc_info=True)


_current_hub: ContextVar[TrajectoryObservationHub | None] = ContextVar('trajectory_observation_hub', default=None)


def get_current_trajectory_observation_hub() -> TrajectoryObservationHub | None:
    """Return the Hub bound to the current Run, if trajectory observation is active."""
    return _current_hub.get()


@contextmanager
def trajectory_observer_context(
    observers: Iterable[TrajectoryObserver],
    *,
    close_on_exit: bool = True,
) -> Iterator[TrajectoryObservationHub]:
    """Bind one independent TrajectoryObservationHub to the current Run."""
    hub = TrajectoryObservationHub(observers)
    token = _current_hub.set(hub)
    try:
        yield hub
    finally:
        _current_hub.reset(token)
        if close_on_exit:
            hub.close()
