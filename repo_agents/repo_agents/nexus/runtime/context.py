"""RuntimeControl contextvar for runtime-side code.

Code executing inside the runtime process (e.g., invoke targets) can access
executor capabilities through this contextvar instead of importing executor
modules directly.  This keeps business code decoupled from the runtime
implementation and allows transparent dispatch through InProcess or Worker
backends.
"""

from __future__ import annotations

import asyncio
import contextvars
import functools
import logging
import uuid

from collections.abc import Callable
from typing import Any, TypeVar

import socketio

from nexus.core.runtime_shared import RuntimeSharedStore
from nexus.dataset.task_files import RuntimeTaskFiles
from nexus.runtime.client.base import ExecutorMixin
from nexus.runtime.server.backend import ExecutorBackend, InProcessBackend


logger = logging.getLogger(__name__)
_T = TypeVar('_T')
_SHARED_VALUE_REQUEST_TIMEOUT_SECONDS = 60.0

_runtime_control: contextvars.ContextVar[RuntimeControl | None] = contextvars.ContextVar(
    '_runtime_control', default=None
)


def require_runtime_control() -> RuntimeControl:
    """Get the current RuntimeControl or raise."""
    ctrl = _runtime_control.get()
    if ctrl is None:
        raise RuntimeError(
            'No RuntimeControl available in current context. '
            'Ensure the code is running inside a runtime invoke handler.'
        )
    return ctrl


def set_runtime_control(ctrl: RuntimeControl) -> contextvars.Token[RuntimeControl | None]:
    """Set the RuntimeControl for the current async context."""
    return _runtime_control.set(ctrl)


def get_runtime_control() -> RuntimeControl | None:
    """Get the current RuntimeControl, or None if not set."""
    return _runtime_control.get()


class SIOBridge:
    """Unified Socket.IO event interface for RuntimeControl.

    Abstracts the difference between ``AsyncServer`` (Base process) and
    ``AsyncClient`` (Worker process) so that runtime-side code (proxies,
    services) can use a single ``emit`` / ``on`` API regardless of where it
    runs.

    Handler signature is always ``async def handler(data)`` — no ``sid``.

    Base:   ``SIOBridge(server)``            — emit broadcasts on default ``/``
    Worker: ``SIOBridge(client, '/worker')`` — emit on ``/worker``, Base relay
            forwards to Host and back.
    """

    def __init__(
        self,
        sio: socketio.AsyncServer | socketio.AsyncClient,
        namespace: str | None = None,
    ) -> None:
        self._sio = sio
        self._ns = namespace
        self._is_server = isinstance(sio, socketio.AsyncServer)

    async def emit(self, event: str, data: Any = None) -> None:
        """Emit an event.  Works identically in Base and Worker."""
        await self._sio.emit(event, data, namespace=self._ns)

    async def call(self, event: str, data: Any = None, **kwargs: Any) -> Any:
        """Emit an event and wait for acknowledgement (request-response).

        On Worker (AsyncClient) this calls the Base server directly.
        On Base (AsyncServer) the caller must pass ``to=sid`` via *kwargs*.
        """
        return await self._sio.call(event, data, namespace=self._ns, **kwargs)

    def on(self, event: str, handler: Any = None) -> Any:
        """Register an event handler (decorator or direct call).

        The handler signature is always ``async def handler(data)`` — the
        bridge strips the ``sid`` argument on the server side so callers
        never need to care.
        """

        def decorator(fn: Any) -> Any:
            if self._is_server:

                @functools.wraps(fn)
                async def _server_wrapper(sid: str, *args: Any) -> Any:  # noqa: ARG001
                    return await fn(*args)

                self._sio.on(event, _server_wrapper, namespace=self._ns)
            else:
                self._sio.on(event, fn, namespace=self._ns)
            return fn

        return decorator if handler is None else decorator(handler)


class RuntimeControl(ExecutorMixin):
    """Capability object providing executor access for runtime-side code.

    Inherits all typed convenience methods from ExecutorMixin (run_command,
    browser_goto, gui_click, etc.) — no need to duplicate them here.

    Attributes:
        sio: Unified :class:`SIOBridge` for emitting / receiving events.
             Proxy services (AnthropicProxy, OpenAIProxy, etc.) use this to
             relay LLM requests back to the Host.
    """

    def __init__(
        self,
        backend: ExecutorBackend | None = None,
        sio: SIOBridge | Any = None,
        *,
        runtime_shared: RuntimeSharedStore | None = None,
    ) -> None:
        self._backend = backend or InProcessBackend()
        self.sio = sio
        self.runtime_shared = runtime_shared or RuntimeSharedStore()
        self._shared_value_locks: dict[str, asyncio.Lock] = {}
        self._shared_value_waiters: dict[str, asyncio.Future[Any]] = {}
        self.task_files = RuntimeTaskFiles(sio=sio)

    async def get_shared_value(
        self,
        name: str,
        expected_type: type[_T],
        *,
        decode: Callable[[Any], _T],
    ) -> _T:
        """Return a cached shared value, requesting it from Host on first use."""

        cached = self.runtime_shared.get_optional(name, expected_type)
        if cached is not None:
            return cached

        lock = self._shared_value_locks.setdefault(name, asyncio.Lock())
        async with lock:
            cached = self.runtime_shared.get_optional(name, expected_type)
            if cached is not None:
                return cached
            payload = await self._request_host_shared_value(name)
            value = decode(payload)
            if not isinstance(value, expected_type):
                raise TypeError(
                    f'Decoder for shared value {name!r} returned {type(value).__name__}, '
                    f'expected {expected_type.__name__}'
                )
            self.runtime_shared.set(name, value)
            return value

    async def _request_host_shared_value(self, name: str) -> Any:
        if self.sio is None:
            raise RuntimeError(f'Runtime shared value {name!r} requires a Socket.IO bridge')

        request_id = uuid.uuid4().hex
        loop = asyncio.get_running_loop()
        future: asyncio.Future[Any] = loop.create_future()
        self._shared_value_waiters[request_id] = future
        try:
            await self.sio.emit('shared-value:request', {'request_id': request_id, 'name': name})
            return await asyncio.wait_for(future, timeout=_SHARED_VALUE_REQUEST_TIMEOUT_SECONDS)
        finally:
            self._shared_value_waiters.pop(request_id, None)

    def provide_shared_value(self, request_id: str, value: Any) -> None:
        """Resolve a pending Host shared value request inside this Runtime process."""

        future = self._shared_value_waiters.get(request_id)
        if future is None:
            raise KeyError(f'Unknown Runtime shared value request: {request_id}')
        if not future.done():
            future.set_result(value)

    def fail_shared_value(self, request_id: str, error: str) -> None:
        """Fail a pending Host shared value request inside this Runtime process."""

        future = self._shared_value_waiters.get(request_id)
        if future is None:
            raise KeyError(f'Unknown Runtime shared value request: {request_id}')
        if not future.done():
            future.set_exception(RuntimeError(error))

    async def call_executor(self, executor_name: str, arg: Any) -> Any:
        """Call an executor by name with a dict or Pydantic model arg."""
        params = arg.model_dump() if hasattr(arg, 'model_dump') else arg
        return await self._backend.call(executor_name, params)


async def provide_runtime_shared_value(request_id: str, value: Any) -> None:
    """Host-invoked helper that delivers a shared value to a waiting Runtime call."""

    require_runtime_control().provide_shared_value(request_id, value)


async def fail_runtime_shared_value(request_id: str, error: str) -> None:
    """Host-invoked helper that fails a waiting Runtime shared value request."""

    require_runtime_control().fail_shared_value(request_id, error)


__all__ = [
    'RuntimeControl',
    'SIOBridge',
    'fail_runtime_shared_value',
    'get_runtime_control',
    'provide_runtime_shared_value',
    'require_runtime_control',
    'set_runtime_control',
]
