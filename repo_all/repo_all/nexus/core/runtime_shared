"""Ephemeral values shared by Host and Runtime components."""

from __future__ import annotations

from typing import Any, TypeVar


_T = TypeVar('_T')


class SharedValueNotFoundError(KeyError):
    """Raised when a named shared value is not available."""


class RuntimeSharedStore:
    """A small process-local store for named values shared within one run."""

    def __init__(self) -> None:
        self._values: dict[str, Any] = {}

    def set(self, name: str, value: Any) -> None:
        if not name:
            raise ValueError('Shared value name must be non-empty')
        self._values[name] = value

    def get(self, name: str, expected_type: type[_T] | None = None) -> _T | Any:
        try:
            value = self._values[name]
        except KeyError as exc:
            raise SharedValueNotFoundError(name) from exc
        if expected_type is not None and not isinstance(value, expected_type):
            raise TypeError(f'Shared value {name!r} has type {type(value).__name__}, expected {expected_type.__name__}')
        return value

    def get_optional(self, name: str, expected_type: type[_T] | None = None) -> _T | Any | None:
        if name not in self._values:
            return None
        return self.get(name, expected_type)

    def discard(self, name: str) -> None:
        self._values.pop(name, None)

    def clear(self) -> None:
        self._values.clear()

    def __contains__(self, name: object) -> bool:
        return name in self._values


__all__ = ['RuntimeSharedStore', 'SharedValueNotFoundError']
