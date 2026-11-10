"""Local runtime — executes everything in the current process."""

from __future__ import annotations

import asyncio
import inspect

from typing import TYPE_CHECKING, Any

from nexus.core.registry import Registry
from nexus.runtime.client.base import Runtime


if TYPE_CHECKING:
    from nexus.runtime.models import InvokeRequest


class LocalRuntime(Runtime):
    """Local runtime that executes everything in the current process.

    This runtime does not call a remote runtime server. All calls are
    resolved and executed directly via :meth:`_invoke`.
    """

    def __init__(self, resource_spec: dict[str, int] | None = None) -> None:
        super().__init__()
        self.resource_spec = resource_spec or {}
        self._ensure_executors_loaded()

    @staticmethod
    def _ensure_executors_loaded() -> None:
        """Import executor modules so they are registered in executor_registry."""
        Registry.import_module('nexus.runtime.executors')

    async def _invoke(self, request: InvokeRequest) -> Any:
        """Resolve and execute a function locally (no serialization over the wire)."""
        from nexus.runtime.models import coerce_args, resolve_func

        func = resolve_func(request.import_path)
        args, kwargs = request.unpack_payload()
        args, kwargs = coerce_args(func, args, kwargs)

        if inspect.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        return await asyncio.to_thread(func, *args, **kwargs)


__all__ = ['LocalRuntime']
