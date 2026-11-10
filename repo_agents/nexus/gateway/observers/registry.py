from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from nexus.core.registry import Registry
from nexus.gateway.observers.base import TrajectoryObserver


def validate_trajectory_observer(obj: Any, _name: str, _meta: dict) -> None:
    if not isinstance(obj, type) or not issubclass(obj, TrajectoryObserver):
        raise TypeError(f'TrajectoryObserver must be a subclass of TrajectoryObserver, got {obj!r}')


trajectory_observer_registry = Registry[type[TrajectoryObserver]](
    'trajectory_observer',
    validate_trajectory_observer,
)


def create_trajectory_observers(configs: Mapping[str, Any]) -> dict[str, TrajectoryObserver]:
    """Instantiate registered observers from config objects or plain mappings."""
    observers: dict[str, TrajectoryObserver] = {}
    for key, config in configs.items():
        name = getattr(config, 'name', None)
        args = getattr(config, 'args', None)
        if isinstance(config, Mapping):
            name = config.get('name')
            args = config.get('args', {})
        if not isinstance(name, str) or not name:
            raise ValueError(f'Trajectory observer {key!r} requires a non-empty name')
        if not isinstance(args, Mapping):
            raise TypeError(f'Trajectory observer {key!r} args must be a mapping')
        observer_cls = trajectory_observer_registry.get(name)
        observers[key] = observer_cls(**dict(args))
    return observers
