"""Base class for executor request models with response type tracking."""

from __future__ import annotations

from typing import ClassVar, Generic, TypeVar

from pydantic import BaseModel


ResponseT = TypeVar('ResponseT', bound=BaseModel)


class ExecutorRequest(BaseModel, Generic[ResponseT]):
    """Executor request that carries its response type and dispatch name.

    Subclasses **must** set two ``ClassVar`` fields::

        class RunCommandRequest(ExecutorRequest[RunCommandResponse]):
            executor_name: ClassVar[str] = 'run_command'
            response_type: ClassVar[type[BaseModel]] = RunCommandResponse

    This enables :meth:`ExecutorMixin.call` to dispatch and validate
    the response with full static type inference.
    """

    executor_name: ClassVar[str]
    response_type: ClassVar[type[BaseModel]]


__all__ = ['ExecutorRequest', 'ResponseT']
