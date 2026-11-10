"""CLI command registry support."""

from __future__ import annotations

import argparse

from collections.abc import Callable
from typing import Any, Protocol

from nexus.core.registry import Registry


CommandHandler = Callable[[argparse.Namespace], int]
CommandParserRegistrar = Callable[[argparse._SubParsersAction, str], None]


class CommandSpec(Protocol):
    """Protocol for extension-provided CLI command registrations."""

    register_parser: CommandParserRegistrar
    handle_command: CommandHandler


def validate_command(obj: Any, name: str, meta: dict) -> None:
    """Validate extension CLI command registrations."""
    if not callable(getattr(obj, 'register_parser', None)):
        raise TypeError(f"Command '{name}' register_parser must be callable")
    if not callable(getattr(obj, 'handle_command', None)):
        raise TypeError(f"Command '{name}' handle_command must be callable")


command_registry = Registry[CommandSpec]('command', validate_command)


__all__ = [
    'CommandHandler',
    'CommandParserRegistrar',
    'CommandSpec',
    'command_registry',
    'validate_command',
]
