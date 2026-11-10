"""Lossless copying for Gateway wire values."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


def copy_messages(messages: list[dict]) -> list[dict]:
    return [copy_message(message) for message in messages]


def copy_message(message: Any) -> dict:
    copied = copy_wire_value(message)
    if isinstance(copied, dict):
        return copied
    return dict(copied)


def copy_wire_value(value: Any) -> Any:
    """Copy renderer-relevant wire facts without collapsing null and absent."""
    if hasattr(value, 'model_dump'):
        return copy_wire_value(value.model_dump(exclude_none=False))
    if isinstance(value, dict):
        return {key: copy_wire_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [copy_wire_value(item) for item in value]
    return deepcopy(value)


def copy_value(value: Any) -> Any:
    if hasattr(value, 'model_dump'):
        return copy_value(value.model_dump(exclude_none=True))
    if isinstance(value, dict):
        return {key: copy_value(item) for key, item in value.items() if item is not None}
    if isinstance(value, list):
        return [copy_value(item) for item in value]
    return deepcopy(value)
