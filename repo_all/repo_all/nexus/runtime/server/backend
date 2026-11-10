"""Executor backend abstraction for transparent dispatch.

Two backends share the same ``call(method, params)`` interface:
- ``InProcessBackend``: resolves from executor_registry and calls directly.
- ``WorkerBackend``: forwards via Socket.IO to a worker process (stub, completed later).
"""

import inspect
import logging

from typing import Any, Protocol, runtime_checkable

from nexus.core import executor_registry


logger = logging.getLogger(__name__)


@runtime_checkable
class ExecutorBackend(Protocol):
    """Abstract interface for executor dispatch."""

    async def call(self, method: str, params: dict[str, Any]) -> dict[str, Any]: ...


class InProcessBackend:
    """Dispatch executor calls within the current process."""

    async def call(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        # Try dotted group name first (e.g. "browser.browser_goto"), then bare name
        executor = None
        for name in (method, method.split('.')[-1] if '.' in method else method):
            try:
                executor = executor_registry.get(name)
                break
            except Exception:
                continue

        if executor is None:
            available = list(executor_registry.get_all().keys())
            raise ValueError(f"Executor '{method}' not found. Available: {available}")

        sig = inspect.signature(executor)
        param_list = [p for p in sig.parameters.values() if p.name != 'self']
        input_type = param_list[0].annotation
        parsed_arg = input_type.model_validate(params)
        result = await executor(parsed_arg)

        if hasattr(result, 'model_dump'):
            return result.model_dump()
        if isinstance(result, dict):
            return result
        return {'data': result}


class WorkerBackend:
    """Dispatch executor calls to a worker process via Socket.IO.

    Stub implementation — completed in the Worker Process work unit.
    """

    def __init__(self, sio: Any, worker_sid: str, namespace: str = '/worker') -> None:
        self._sio = sio
        self._worker_sid = worker_sid
        self._namespace = namespace

    async def call(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        result = await self._sio.call(
            'execute',
            {'method': method, 'params': params},
            to=self._worker_sid,
            namespace=self._namespace,
        )
        if not result.get('success', False):
            raise RuntimeError(f"Worker executor '{method}' failed: {result.get('error', 'unknown')}")
        return result.get('data', {})
