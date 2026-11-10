"""Tool registry for nexus framework."""

from __future__ import annotations

import inspect

from collections.abc import Callable
from typing import Any, get_type_hints

from pydantic import BaseModel

from nexus.core.registry import ProcessorResult, Registry
from nexus.trace import observe

from .types import ToolContext, ToolFunc, ToolMeta, ToolResponse, ToolType


def _validate_tool_func(func: Callable[..., Any], name: str, meta: dict[str, Any]) -> ProcessorResult:
    """Validate and process tool function during registration, and wrap with observe."""
    func_name = getattr(func, '__name__', '<unknown>')
    tool_name = name or func_name

    if not inspect.iscoroutinefunction(func):
        raise TypeError(f"Tool '{tool_name}' must be an async function")

    hints = get_type_hints(func)
    params = list(inspect.signature(func).parameters.keys())

    if len(params) < 2:
        raise TypeError(
            f"Tool '{func_name}' must have at least 2 parameters: (params: InputModel, context: ToolContext)"
        )

    first_param = params[0]
    if first_param not in hints:
        raise TypeError(f"Tool '{func_name}' first parameter must have a type annotation")

    input_type = hints[first_param]
    if not (isinstance(input_type, type) and issubclass(input_type, BaseModel)):
        raise TypeError(f"Tool '{func_name}' first parameter must be a Pydantic BaseModel, got {input_type}")

    second_param = params[1]
    if second_param not in hints or hints[second_param] != ToolContext:
        raise TypeError(
            f"Tool '{func_name}' second parameter must be ToolContext, got {hints.get(second_param, 'no annotation')}"
        )

    if 'return' not in hints:
        raise TypeError(f"Tool '{func_name}' must have a return type annotation")
    if hints['return'] != ToolResponse:
        raise TypeError(f"Tool '{func_name}' must return ToolResponse, got {hints['return']}")

    description = meta.get('description', inspect.getdoc(func) or 'No description')
    tool_type = ToolType(meta.get('tool_type', ToolType.LOCAL))
    schema_visible_name = meta.get('name', name)
    if not schema_visible_name:
        raise TypeError(f"Tool '{tool_name}' meta.name must be non-empty")
    parallel_func = meta.get('parallel_func')

    _span_name = f'tool.{name.replace("/", ".")}'
    wrapped_func = observe(name=_span_name, attributes={'type': 'tool', 'name': name})(func)
    wrapped_func.__tool_input_model__ = input_type
    wrapped_func.__tool_meta__ = ToolMeta(
        name=schema_visible_name,
        description=description,
        tool_type=tool_type,
        input_model=input_type,
        func=wrapped_func,
        parallel_func=parallel_func,
    )

    return ProcessorResult(wrapped_obj=wrapped_func, meta=meta)


tool_registry = Registry[ToolFunc]('tool', _validate_tool_func)


def get_tool_meta(tool: ToolFunc) -> ToolMeta:
    """Get metadata for a registered tool."""
    return tool.__tool_meta__


def get_tool_schema(tool: ToolFunc) -> dict[str, Any]:
    """Get OpenAI function schema for a tool."""
    return get_tool_meta(tool).schema


__all__ = [
    'get_tool_meta',
    'get_tool_schema',
    'tool_registry',
]
