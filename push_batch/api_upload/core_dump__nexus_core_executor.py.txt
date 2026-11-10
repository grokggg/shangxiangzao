"""
Executor is a callable unit registered on the runtime side, typically serving as
the underlying implementation of tools. The interface requires exactly one Pydantic
model as input with no other parameters, and returns a single Pydantic model as output.
"""

import inspect

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel

from nexus.core.registry import Registry


def validate_executor(obj: Any, _name: str, _meta: dict) -> None:
    if not callable(obj):
        raise TypeError(f'Executor must be callable, got {type(obj)}')

    sig = inspect.signature(obj)
    params = [p for p in sig.parameters.values() if p.name != 'self']

    if len(params) != 1:
        raise TypeError(f'Executor must have exactly one parameter (excluding self), got {len(params)}')

    param = params[0]
    if param.annotation is inspect.Parameter.empty:
        raise TypeError('Executor parameter must have type annotation')

    input_type = param.annotation
    if not (isinstance(input_type, type) and issubclass(input_type, BaseModel)):
        raise TypeError(f'Executor parameter must be a Pydantic BaseModel subclass, got {input_type}')

    if sig.return_annotation is inspect.Signature.empty:
        raise TypeError('Executor must have return type annotation')

    return_type = sig.return_annotation
    if not (isinstance(return_type, type) and issubclass(return_type, BaseModel)):
        raise TypeError(f'Executor return type must be a Pydantic BaseModel subclass, got {return_type}')


executor_registry = Registry[Callable]('executor', validate_executor)
