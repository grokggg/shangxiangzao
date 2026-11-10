"""RuntimeProvider registry for managing runtime provider classes.

This module provides the registry for runtime providers.
"""

from __future__ import annotations

from typing import Any

from nexus.core.registry import Registry
from nexus.runtime.providers.base import RuntimeProvider


def validate_runtime_provider(obj: Any, _name: str, _meta: dict) -> None:
    """Validate that an object is a valid RuntimeProvider class."""
    if not isinstance(obj, type):
        raise TypeError(f'RuntimeProvider must be a class, got {type(obj)}')
    if not issubclass(obj, RuntimeProvider):
        raise TypeError(f'RuntimeProvider must be a subclass of RuntimeProvider, got {obj}')


runtime_provider_registry = Registry[type[RuntimeProvider]]('runtime_provider', validate_runtime_provider)


__all__ = [
    'runtime_provider_registry',
    'validate_runtime_provider',
]
