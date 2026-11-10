"""Worker SDK — runs inside a worker process, connects back to Base via SIO.

Pre-installed in the base image. The worker process loads this module,
connects to the Base server's ``/worker`` namespace, and handles ``execute``
events by dispatching to the local executor registry.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import logging
import os
import sys

from typing import Any

import socketio

from nexus.core import executor_registry, import_from_path
from nexus.runtime.capabilities import runtime_capabilities
from nexus.trace.processors.base import SpanProcessor
from nexus.trace.processors.socketio import DEFAULT_SPAN_QUEUE_SIZE, truncate_span_event_data
from nexus.utils.otel.lifecycle import async_force_flush_best_effort
from nexus.utils.otel.propagation import use_remote_context
from nexus.utils.otel.transport import transport_span
from nexus.utils.socketio import AsyncSocketIOEmitter


logger = logging.getLogger(__name__)

# Contextvar for worker-side code to emit events back to Base
_worker_emit: contextvars.ContextVar[Any] = contextvars.ContextVar('_worker_emit', default=None)


def get_worker_emit() -> Any:
    """Get the emit callable if running inside a worker process."""
    return _worker_emit.get()


class WorkerSDK:
    """Socket.IO client that connects a worker process to the Base server.

    Usage (inside worker process)::

        sdk = WorkerSDK(base_url='http://localhost:32000', venv_id='abc123')
        await sdk.start()
        # ... runs until stopped
        await sdk.stop()
    """

    def __init__(self, base_url: str, venv_id: str, worker_id: str | None = None) -> None:
        self.base_url = base_url
        self.venv_id = venv_id
        self.worker_id = worker_id or f'worker-{venv_id}'
        self._sio = socketio.AsyncClient(logger=False, engineio_logger=False)
        self._connected = asyncio.Event()
        self._span_processor = _SIOSpanProcessor(self._sio, self.worker_id)
        self._span_processor_registered = False
        from nexus.runtime.context import RuntimeControl, SIOBridge
        from nexus.runtime.server.backend import InProcessBackend

        self._runtime_control = RuntimeControl(
            InProcessBackend(),
            sio=SIOBridge(self._sio, namespace='/worker'),
        )
        self._setup_handlers()

    def _setup_handlers(self) -> None:
        ns = '/worker'

        @self._sio.on('connect', namespace=ns)
        async def on_connect() -> None:
            logger.info('Worker %s connected to Base', self.worker_id)
            self._connected.set()

        @self._sio.on('disconnect', namespace=ns)
        async def on_disconnect() -> None:
            logger.warning('Worker %s disconnected from Base', self.worker_id)
            self._connected.clear()

        @self._sio.on('execute', namespace=ns)
        async def on_execute(data: dict) -> dict:
            """Handle an execute request from the Base server."""
            method: str = data.get('method', '')
            params: dict = data.get('params', {})
            logger.info('Worker %s executing %s', self.worker_id, method)
            try:
                result = await self._dispatch(method, params)
                response = {'success': True, 'data': result}
            except Exception as exc:
                logger.error('Worker %s execute %s failed: %s', self.worker_id, method, exc)
                response = {'success': False, 'error': str(exc)}
            try:
                await self._span_processor.flush()
            except Exception as exc:
                logger.error('Worker %s trace flush failed: %s', self.worker_id, exc)
                if response['success']:
                    response = {'success': False, 'error': f'{type(exc).__name__}: {exc}'}
            return response

        @self._sio.on('trace:flush', namespace=ns)
        async def on_trace_flush(_data: dict | None = None) -> dict[str, Any]:
            try:
                await self._span_processor.flush()
                await async_force_flush_best_effort()
                return {'ok': True}
            except Exception as exc:
                return {'ok': False, 'error': f'{type(exc).__name__}: {exc}'}

        @self._sio.on('invoke', namespace=ns)
        async def on_invoke(data: dict) -> None:
            """Handle an invoke request (pickle-based) from the Base server.

            Uses the shared ``execute_invoked_func`` core so behaviour stays
            identical to the Base server's ``_execute_invoke``.

            Emits ``invoke:result`` instead of returning via ACK so there is
            no SIO-level timeout on the Base side.
            """
            import_path: str = data.get('import_path', '')
            request_id: str = data.get('request_id', '')
            logger.info('Worker %s invoke %s (rid=%s)', self.worker_id, import_path, request_id)

            try:
                from nexus.runtime.models import (
                    execute_invoked_func,
                    invoke_response_from_exception,
                    pack_result_for_transport,
                )
                from nexus.trace import RemoteTraceContext

                trace_ctx_raw = data.get('trace_context')
                trace_ctx = RemoteTraceContext.model_validate(trace_ctx_raw) if trace_ctx_raw else None

                payload = data.get('payload', '')
                payload_ref = data.get('payload_ref')
                carrier = trace_ctx.otel_carrier if trace_ctx is not None else None
                with use_remote_context(carrier, fallback_only=False):
                    with transport_span(
                        'runtime.worker.invoke',
                        kind='server',
                        attributes={
                            'rpc.system': 'socket.io',
                            'rpc.method': 'invoke',
                            'nexus.worker.id': self.worker_id,
                        },
                    ):
                        # Resolve the full input inside the server span so transport failures
                        # are recorded on the request that triggered them.
                        if payload_ref and not payload:
                            import base64

                            import aiohttp as _aiohttp

                            async with _aiohttp.ClientSession() as session:
                                resp = await session.get(f'{self.base_url}/api/v1/blob/{payload_ref}')
                                resp.raise_for_status()
                                blob = await resp.read()
                            payload = base64.b64encode(blob).decode('utf-8')

                        result = await execute_invoked_func(
                            import_path,
                            payload,
                            self._runtime_control,
                            trace_context=trace_ctx,
                        )

                        response, blob_id, blob_bytes = pack_result_for_transport(result, request_id=request_id)
                        if blob_bytes is not None:
                            from nexus.runtime.models import upload_blob

                            await upload_blob(f'{self.base_url}/api/v1/blob/{blob_id}', blob_bytes)

            except Exception as exc:
                logger.error('Worker %s invoke %s failed: %s', self.worker_id, import_path, exc)
                response = invoke_response_from_exception(exc, request_id=request_id)

            try:
                await self._span_processor.flush()
            except Exception as exc:
                logger.error('Worker %s trace flush failed: %s', self.worker_id, exc)
                if response.success:
                    from nexus.runtime.models import invoke_response_from_exception

                    response = invoke_response_from_exception(exc, request_id=request_id)

            await self._sio.emit('invoke:result', response.model_dump(), namespace='/worker')

    async def _dispatch(self, method: str, params: dict) -> Any:
        """Dispatch an execute request to the local executor registry."""
        import inspect

        # Try group-style first (e.g. "browser.browser_goto")
        executor = None
        for name in (method, method.split('.')[-1] if '.' in method else method):
            try:
                executor = executor_registry.get(name)
                break
            except Exception:
                continue

        if executor is None and ('::' in method or '.' in method):
            with contextlib.suppress(Exception):
                executor = import_from_path(method)

        if executor is None:
            raise ValueError(f"Worker cannot resolve method '{method}'")

        sig = inspect.signature(executor)
        param_list = [p for p in sig.parameters.values() if p.name != 'self']
        if param_list and hasattr(param_list[0].annotation, 'model_validate'):
            parsed = param_list[0].annotation.model_validate(params)
            result = await executor(parsed)
        else:
            result = await executor(**params) if params else await executor()

        if hasattr(result, 'model_dump'):
            return result.model_dump()
        if isinstance(result, dict):
            return result
        return {'data': result}

    def _connect_auth(self) -> dict[str, Any]:
        return {
            'venv_id': self.venv_id,
            'worker_id': self.worker_id,
            'capabilities': runtime_capabilities(),
        }

    async def start(self) -> None:
        """Connect to Base server and register as a worker."""
        # Inject emit callable into contextvar
        _worker_emit.set(self._sio.emit)

        # Setup SIO log handler to forward logs to Base.
        # The root logger defaults to WARNING — lower it so INFO/DEBUG
        # records propagate to the handler.
        root = logging.getLogger()
        root.setLevel(logging.DEBUG)
        handler = _SIOLogHandler(self._sio, self.worker_id, self.venv_id)
        root.addHandler(handler)

        # Setup span processor to forward trace events to Base.
        from nexus.trace.tracer import add_span_processor

        await self._span_processor.start()
        add_span_processor(self._span_processor)
        self._span_processor_registered = True

        await self._sio.connect(
            self.base_url,
            namespaces=['/worker'],
            auth=self._connect_auth(),
        )
        await self._connected.wait()
        logger.info('Worker %s ready', self.worker_id)

    async def stop(self) -> None:
        """Disconnect from Base server."""
        if self._span_processor_registered:
            from nexus.trace.tracer import remove_span_processor

            remove_span_processor(self._span_processor)
            self._span_processor_registered = False
        try:
            await self._span_processor.close()
            await async_force_flush_best_effort()
        finally:
            if self._sio.connected:
                await self._sio.disconnect()
        logger.info('Worker %s stopped', self.worker_id)

    async def run_forever(self) -> None:
        """Block until disconnected."""
        await self._sio.wait()


class _SIOLogHandler(logging.Handler):
    """Forward Python logging records to Base server via SIO."""

    def __init__(self, sio: socketio.AsyncClient, worker_id: str, venv_id: str) -> None:
        super().__init__()
        self._sio = sio
        self._worker_id = worker_id
        self._venv_id = venv_id

    def emit(self, record: logging.LogRecord) -> None:
        if not self._sio.connected:
            return
        try:
            from nexus.utils.truncate import truncate_string

            msg = truncate_string(self.format(record))
            asyncio.get_event_loop().create_task(
                self._sio.emit(
                    'worker_log',
                    {
                        'worker_id': self._worker_id,
                        'venv_id': self._venv_id,
                        'level': record.levelname,
                        'name': record.name,
                        'message': msg,
                    },
                    namespace='/worker',
                )
            )
        except Exception:
            pass


class _SIOSpanProcessor(SpanProcessor):
    """Forward span events from worker to Base server via SIO."""

    def __init__(
        self,
        sio: socketio.AsyncClient,
        worker_id: str,
        *,
        max_queue_size: int = DEFAULT_SPAN_QUEUE_SIZE,
        batch_size: int = 50,
        flush_interval: float = 0.1,
        request_timeout: float = 60.0,
    ) -> None:
        self._sio = sio
        self._worker_id = worker_id
        self._request_timeout = request_timeout
        self._emitter = AsyncSocketIOEmitter(
            sio,
            'span',
            max_queue_size=max_queue_size,
            batch_size=batch_size,
            flush_interval=flush_interval,
            emit_func=self._emit_batch,
        )

    def _emit(self, data: dict) -> None:
        payload = truncate_span_event_data(data)
        payload['worker_id'] = self._worker_id
        self._emitter.enqueue(payload)

    async def _emit_batch(self, event: str, data: dict[str, Any]) -> None:
        response = await self._sio.call(
            event,
            data,
            namespace='/worker',
            timeout=self._request_timeout,
        )
        if not isinstance(response, dict) or not response.get('ok'):
            error = response.get('error') if isinstance(response, dict) else None
            raise RuntimeError(error or 'Worker span relay failed')

    def on_start(self, span, event) -> None:  # noqa: ANN001
        self._emit(event.model_dump())

    def on_update(self, span, event) -> None:  # noqa: ANN001
        self._emit(event.model_dump())

    def on_end(self, span, event) -> None:  # noqa: ANN001
        self._emit(event.model_dump())

    async def start(self) -> None:
        await self._emitter.start()

    async def flush(self) -> None:
        await self._emitter.flush()

    async def close(self) -> None:
        await self._emitter.close()


async def worker_main(base_url: str, venv_id: str, worker_id: str | None = None) -> None:
    """Entry point for a worker process."""
    sdk = WorkerSDK(base_url, venv_id, worker_id)
    try:
        await sdk.start()
        await sdk.run_forever()
    finally:
        await sdk.stop()


if __name__ == '__main__':
    os.environ.pop('PYTHONSAFEPATH', None)
    # python -m nexus.runtime.worker.sdk <base_url> <venv_id> [worker_id]
    if len(sys.argv) < 3:
        print(f'Usage: {sys.argv[0]} <base_url> <venv_id> [worker_id]')
        sys.exit(1)
    asyncio.run(worker_main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None))
