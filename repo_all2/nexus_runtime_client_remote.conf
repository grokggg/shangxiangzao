"""Remote runtime — HTTP invoke + Socket.IO events."""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import logging
import os
import random
import uuid

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

import aiohttp
import socketio

from nexus.runtime.capabilities import supports_trace_flush
from nexus.runtime.client.base import Runtime
from nexus.runtime.client.cancellation import RemoteChildOperationCancelledError, await_remote_operation
from nexus.runtime.client.models import VenvHandle
from nexus.trace import RemoteTraceContext
from nexus.utils.async_http import AsyncHTTP
from nexus.utils.event_loop import get_current_task_cancellation_count
from nexus.utils.otel.propagation import inject_carrier


if TYPE_CHECKING:
    from nexus.runtime.models import InvokeRequest


logger = logging.getLogger(__name__)

# Uvicorn closes idle HTTP keep-alive connections after five seconds by default.
# Expire pooled runtime connections first so a model/tool pause cannot race a
# server-side close and surface as an intermittent ServerDisconnectedError.
http_client = AsyncHTTP(
    timeout=aiohttp.ClientTimeout(total=None),
    keepalive_timeout=4.0,
)

# Dedicated loggers routed into python-socketio / engineio so their traces
# (PING/PONG, transport error, polling reconnects, packet flow) appear under
# our namespace instead of the library defaults. Bump/silence them
# independently of ``remote`` without editing this file.
_sio_logger = logging.getLogger('nexus.runtime.client.remote.socketio')
_eio_logger = logging.getLogger('nexus.runtime.client.remote.engineio')


def _sio_debug_enabled() -> bool:
    """True when the socket.io / engine.io transport traces should flow.

    Off by default because each connection emits a PING/PONG pair every 15 s
    and every packet is logged. Flip on with ``NEXUS_RUNTIME_SIO_DEBUG=1`` when
    investigating a disconnect storm or wire-level issue.
    """
    return os.getenv('NEXUS_RUNTIME_SIO_DEBUG', '').strip().lower() in {'1', 'true', 'yes', 'on'}


@dataclass
class RemoteRuntimeConfig:
    """Configuration for RemoteRuntime.

    Combines HTTP invoke settings with Socket.IO event streaming settings.
    """

    base_url: str = ''
    request_timeout: float = 1800.0  # 总超时（秒），默认 30 分钟
    invoke_wait: float = 30.0  # Prefer: wait=N — long-poll 等待秒数
    poll_interval: float = 2.0  # HTTP 轮询间隔（秒）
    auto_connect_ws: bool = True  # 是否自动连接 SIO（事件流 + push 通知）
    socketio_request_timeout: float = 60.0  # Engine.IO HTTP polling/connect request timeout
    ws_reconnection: bool = True
    ws_reconnection_attempts: int = 60
    ws_reconnection_delay: float = 1.0
    ws_reconnection_delay_max: float = 10.0
    enable_runtime_logs: bool = True
    default_headers: dict[str, str] = field(default_factory=dict)

    # --- Initial SIO connection retry (sandbox startup) ---
    initial_connect_attempts: int = 20
    initial_connect_delay: float = 1.0
    initial_connect_delay_max: float = 10.0
    initial_connect_jitter: float = 0.5
    initial_connect_ns_wait_timeout: float = 3.0


def _venv_handle_from_dict(data: dict[str, Any]) -> VenvHandle:
    return VenvHandle(
        venv_id=data.get('venv_id', ''),
        state=data.get('state', 'unknown'),
        python_version=data.get('python_version', ''),
        artifacts=data.get('artifacts', []),
        error=data.get('error'),
    )


class RemoteRuntime(Runtime):
    """Unified remote runtime client.

    - **HTTP** for invoke (``Prefer: wait=N`` long-poll -> async fallback -> polling).
    - **Socket.IO** (optional) for real-time events (logs, spans) and push notification
      of async invoke completion (``invoke:done:{rid}``).

    Example::

        runtime = RemoteRuntime(base_url='http://sandbox:8000')

        result = await runtime.invoke('mod::func', 1, 2)
        response = await runtime.run_command(command='ls -la')

        await runtime.ws_connect()
        runtime.on('custom_event', handler)
        await runtime.ws_disconnect()
    """

    def __init__(
        self,
        base_url: str = '',
        request_timeout: float = 1800.0,
        resource_spec: dict[str, int] | None = None,
        enable_runtime_logs: bool = True,
        auto_connect_ws: bool = True,
        default_headers: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.config = RemoteRuntimeConfig(
            base_url=base_url,
            request_timeout=request_timeout,
            enable_runtime_logs=enable_runtime_logs,
            auto_connect_ws=auto_connect_ws,
            default_headers=default_headers or {},
            **kwargs,
        )
        self._timeout = aiohttp.ClientTimeout(total=request_timeout)
        self.resource_spec = resource_spec or {}

        # Socket.IO (optional — for events + push)
        self._sio: socketio.AsyncClient | None = None
        self._ws_connected: bool = False
        self._disconnect_event: asyncio.Event = asyncio.Event()
        self._sio_auto_connect_disabled: bool = False
        self._event_listeners: dict[str, list[Callable[..., Awaitable[None]]]] = {}
        self._span_replay_lock = asyncio.Lock()
        self._span_replay_condition = asyncio.Condition()
        self._last_replayed_span_batch_id = 0
        self._span_batch_baseline_established = False
        self._span_replay_error: tuple[int | None, Exception] | None = None
        self._trace_flush_supported: bool | None = None
        self._remote_operation_started_without_sio = False

    @property
    def base_url(self) -> str | None:
        return self.config.base_url

    async def _race_with_unhealthy(self, coro: Awaitable[Any], *, op_desc: str) -> Any:
        """Run *coro* and fail-fast if the runtime is marked unhealthy while it's in flight.

        The coroutine is wrapped in a task. If ``_unhealthy_event`` fires first,
        the in-flight task is cancelled and :class:`RuntimeUnhealthyError` is raised.
        """
        self._raise_if_unhealthy()
        op_task = asyncio.ensure_future(coro)
        unhealthy_wait = asyncio.create_task(self._unhealthy_event.wait())
        try:
            done, _ = await asyncio.wait({op_task, unhealthy_wait}, return_when=asyncio.FIRST_COMPLETED)
            if unhealthy_wait in done and op_task not in done:
                op_task.cancel()
                with contextlib.suppress(BaseException):
                    await op_task
                logger.warning('Aborting %s due to unhealthy runtime', op_desc)
                self._raise_if_unhealthy()
            return op_task.result()
        finally:
            if not unhealthy_wait.done():
                unhealthy_wait.cancel()

    # ==================== Invoke (HTTP long-poll + polling) ====================

    async def _invoke(self, request: InvokeRequest) -> Any:
        """Send an InvokeRequest via HTTP with long-poll + async fallback.

        1. ``POST /api/v1/invoke`` with ``Prefer: wait=N``.
        2. If ``200`` — fast path, result inline.
        3. If ``202`` — slow path, wait via SIO push or HTTP polling.
        """
        from nexus.runtime.models import BLOB_THRESHOLD_BYTES, InvokeResponse

        self._raise_if_unhealthy()
        await self._try_auto_connect_sio()

        # Fill defaults
        if not request.request_id:
            request = request.model_copy(update={'request_id': uuid.uuid4().hex})
        if not request.trace_context:
            request = request.model_copy(update={'trace_context': RemoteTraceContext.from_current_span()})
        request_timeout = (
            request.request_timeout if request.request_timeout is not None else self.config.request_timeout
        )
        timeout = aiohttp.ClientTimeout(total=request_timeout)

        # Offload large payload to blob store so the JSON body stays small.
        if request.payload and len(request.payload) > BLOB_THRESHOLD_BYTES:
            import base64

            from nexus.runtime.models import upload_blob

            blob_id = uuid.uuid4().hex
            blob_url = f'{self.config.base_url}/api/v1/blob/{blob_id}'
            blob_data = base64.b64decode(request.payload)
            # Free the base64 string before uploading to reduce peak memory.
            request = request.model_copy(update={'payload': '', 'payload_ref': blob_id})
            await await_remote_operation(
                upload_blob(blob_url, blob_data, timeout=timeout),
                logger=logger,
                phase='invoke_blob_upload',
                base_url=self.config.base_url,
                request_id=request.request_id,
            )
            del blob_data

        rid = request.request_id
        url = f'{self.config.base_url}/api/v1/invoke'
        headers = dict(self.config.default_headers or {})
        headers['Prefer'] = f'wait={min(self.config.invoke_wait, request_timeout)}'

        if request.trace_context is not None:
            carrier = inject_carrier()
            if carrier:
                request = request.model_copy(
                    update={
                        'trace_context': request.trace_context.model_copy(update={'otel_carrier': carrier}),
                    }
                )

        logger.info('invoke: POST %s import_path=%s rid=%s', url, request.import_path, rid)

        if self.config.auto_connect_ws and not self._ws_connected:
            self._remote_operation_started_without_sio = True
        resp = await await_remote_operation(
            self._race_with_unhealthy(
                http_client.post(url, json=request.model_dump(), timeout=timeout, headers=headers),
                op_desc=f'invoke POST rid={rid}',
            ),
            logger=logger,
            phase='invoke_post',
            base_url=self.config.base_url,
            request_id=rid,
        )

        if resp.status_code == 200:
            # Fast path — completed within the wait window
            response = InvokeResponse(**resp.json())
            return await self._resolve_response(response, timeout=timeout)

        # 202 — async job, wait for completion
        return await self._wait_for_job(rid or '', request_timeout=request_timeout, timeout=timeout)

    async def _wait_for_job(self, rid: str, *, request_timeout: float, timeout: aiohttp.ClientTimeout) -> Any:
        """Wait for an async invoke job to complete via SIO push or HTTP polling."""
        from nexus.runtime.models import InvokeResponse

        deadline = asyncio.get_event_loop().time() + request_timeout
        unhealthy_wait = asyncio.create_task(self._unhealthy_event.wait())

        # Try SIO push first (non-blocking race with polling)
        if self._ws_connected:
            result_future: asyncio.Future[dict] = asyncio.get_event_loop().create_future()

            def _on_done(data: dict) -> None:
                if not result_future.done():
                    result_future.set_result(data)

            unsub = self.on(f'invoke:done:{rid}', _on_done)
            try:
                # Reconciliation poll: the listener is already registered, so any
                # emit from here on is captured by `_on_done`. This single poll
                # recovers a result whose `invoke:done` event was emitted BEFORE
                # registration (lost wakeup) — the server writes `_jobs` before
                # it emits, so a completed job is always visible here.
                poll_url = f'{self.config.base_url}/api/v1/invoke/{rid}'
                poll_data: dict | None = None
                try:
                    resp = await await_remote_operation(
                        http_client.get(poll_url, timeout=timeout, headers=self.config.default_headers or None),
                        logger=logger,
                        phase='invoke_reconcile_get',
                        base_url=self.config.base_url,
                        request_id=rid,
                    )
                    poll_data = resp.json()
                except Exception as e:
                    # Poll request/parse failure is non-fatal — fall through to SIO wait.
                    logger.warning('invoke: reconcile poll failed for rid=%s (%s); waiting on SIO', rid, e)

                if poll_data is not None and poll_data.get('status', '') in ('done', 'error'):
                    # Resolve OUTSIDE the except: for an error job this re-raises the
                    # remote exception, which must propagate (not be swallowed as a poll failure).
                    response = InvokeResponse(**poll_data['response'])
                    return await self._resolve_response(response, timeout=timeout)

                remaining = max(deadline - asyncio.get_event_loop().time(), 0)
                done, _ = await await_remote_operation(
                    asyncio.wait(
                        {result_future, unhealthy_wait},
                        timeout=remaining,
                        return_when=asyncio.FIRST_COMPLETED,
                    ),
                    logger=logger,
                    phase='invoke_sio_wait',
                    base_url=self.config.base_url,
                    request_id=rid,
                )
                if unhealthy_wait in done:
                    self._raise_if_unhealthy()
                if result_future in done:
                    response = InvokeResponse(**result_future.result())
                    return await self._resolve_response(response, timeout=timeout)
                raise TimeoutError(f'invoke: job {rid} timed out')
            finally:
                unsub()
                if not unhealthy_wait.done():
                    unhealthy_wait.cancel()

        try:
            # Fallback: HTTP polling
            poll_url = f'{self.config.base_url}/api/v1/invoke/{rid}'
            while asyncio.get_event_loop().time() < deadline:
                self._raise_if_unhealthy()
                remaining = max(deadline - asyncio.get_event_loop().time(), 0)
                with contextlib.suppress(asyncio.TimeoutError):
                    await await_remote_operation(
                        asyncio.wait_for(
                            asyncio.shield(self._unhealthy_event.wait()),
                            timeout=min(self.config.poll_interval, remaining),
                        ),
                        logger=logger,
                        phase='invoke_poll_wait',
                        base_url=self.config.base_url,
                        request_id=rid,
                    )
                self._raise_if_unhealthy()
                if asyncio.get_event_loop().time() >= deadline:
                    break
                resp = await await_remote_operation(
                    http_client.get(poll_url, timeout=timeout, headers=self.config.default_headers or None),
                    logger=logger,
                    phase='invoke_poll_get',
                    base_url=self.config.base_url,
                    request_id=rid,
                )
                data = resp.json()
                status = data.get('status', '')
                if status in ('done', 'error'):
                    response = InvokeResponse(**data['response'])
                    return await self._resolve_response(response, timeout=timeout)

            raise TimeoutError(f'invoke: job {rid} timed out after polling')
        finally:
            if not unhealthy_wait.done():
                unhealthy_wait.cancel()

    async def _resolve_response(self, response: Any, *, timeout: aiohttp.ClientTimeout | None = None) -> Any:
        """Deserialize an InvokeResponse, fetching blob if result_ref is set."""
        from nexus.runtime.models import unpack_response

        if response.result_ref and not response.result:
            blob_url = f'{self.config.base_url}/api/v1/blob/{response.result_ref}'
            blob_resp = await await_remote_operation(
                http_client.get(
                    blob_url,
                    timeout=timeout or self._timeout,
                    headers=self.config.default_headers or None,
                ),
                logger=logger,
                phase='invoke_blob_get',
                base_url=self.config.base_url,
                request_id=getattr(response, 'request_id', None),
            )
            import pickle

            return pickle.loads(blob_resp.content)

        return unpack_response(response)

    # ==================== Venv Management ====================

    _VENV_WAIT: float = 60.0  # default long-poll wait for venv creation

    async def create_venv(
        self,
        spec: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> VenvHandle:
        url = f'{self.config.base_url}/api/v1/venv'
        payload = spec or kwargs
        headers = dict(self.config.default_headers or {})
        headers['Prefer'] = f'wait={self._VENV_WAIT}'

        resp = await http_client.post(url, json=payload, timeout=self._timeout, headers=headers)
        data = resp.json()

        if resp.status_code == 200:
            return _venv_handle_from_dict(data)

        # 202 — creation still in progress, wait for completion
        vid = data.get('venv_id', '')
        return await self._wait_for_venv(vid)

    async def _wait_for_venv(self, vid: str) -> VenvHandle:
        """Wait for venv creation via SIO push or HTTP polling."""
        deadline = asyncio.get_event_loop().time() + self.config.request_timeout

        if self._ws_connected:
            result_future: asyncio.Future[dict] = asyncio.get_event_loop().create_future()

            def _on_done(data: dict) -> None:
                if not result_future.done():
                    result_future.set_result(data)

            unsub = self.on(f'venv:done:{vid}', _on_done)
            try:
                remaining = deadline - asyncio.get_event_loop().time()
                data = await asyncio.wait_for(result_future, timeout=max(remaining, 0))
                return _venv_handle_from_dict(data)
            except asyncio.TimeoutError:
                raise TimeoutError(f'create_venv: venv {vid} timed out') from None
            finally:
                unsub()

        # Fallback: HTTP polling
        poll_url = f'{self.config.base_url}/api/v1/venv/{vid}'
        while asyncio.get_event_loop().time() < deadline:
            await asyncio.sleep(self.config.poll_interval)
            resp = await http_client.get(poll_url, timeout=self._timeout, headers=self.config.default_headers or None)
            data = resp.json()
            state = data.get('state', '')
            if state in ('ready', 'failed'):
                return _venv_handle_from_dict(data)

        raise TimeoutError(f'create_venv: venv {vid} timed out after polling')

    async def destroy_venv(self, venv_id: str) -> None:
        url = f'{self.config.base_url}/api/v1/venv/{venv_id}'
        await http_client.delete(url, timeout=self._timeout, headers=self.config.default_headers or None)

    # ==================== Socket.IO Connection Management ====================

    def _ensure_sio_client(self) -> socketio.AsyncClient:
        """Create the SIO client (lazy) and register event handlers."""
        if self._sio is None:
            # Pull client-side receive buffer up to match the server: base's
            # SIO server + uvicorn WS layer both accept 100 MiB, but aiohttp's
            # ``ws_connect`` defaults to a 4 MiB ``max_msg_size`` and would
            # tear the connection down (as ``transport error`` — the message
            # never reaches Engine.IO) on any single frame past that. Push it
            # up so large LLM responses / worker log batches don't
            # false-alarm the reconnect loop.
            debug = _sio_debug_enabled()
            self._sio = socketio.AsyncClient(
                reconnection=self.config.ws_reconnection,
                reconnection_attempts=self.config.ws_reconnection_attempts,
                reconnection_delay=self.config.ws_reconnection_delay,
                reconnection_delay_max=self.config.ws_reconnection_delay_max,
                request_timeout=self.config.socketio_request_timeout,
                # Pass our logger instances when the debug env var is on
                # (traces routed to our namespace); otherwise pass ``False`` so
                # the library falls back to its own ERROR-level default logger
                # and stays quiet.
                logger=_sio_logger if debug else False,
                engineio_logger=_eio_logger if debug else False,
                websocket_extra_options={'max_msg_size': 100 * 1024 * 1024},
            )

            @self._sio.event
            async def connect():
                self._ws_connected = True
                transport = self._sio.transport() if self._sio is not None else 'unknown'
                logger.info(f'socketio: connected to {self.config.base_url} via {transport}')

            @self._sio.event
            async def disconnect(reason: Any = None):
                self._ws_connected = False
                self._disconnect_event.set()
                if reason is not None:
                    logger.info('socketio: disconnected from %s reason=%s', self.config.base_url, reason)
                else:
                    logger.info('socketio: disconnected from %s', self.config.base_url)

            if self.config.enable_runtime_logs:

                @self._sio.on('log')
                async def on_log(data: dict) -> None:
                    for log_data in data.get('items', []):
                        self._handle_runtime_log(log_data)
                    await self._dispatch_event_listeners('log', data)

                @self._sio.on('worker_log')
                async def on_worker_log(data: dict) -> None:
                    self._handle_worker_log(data)
                    await self._dispatch_event_listeners('worker_log', data)

            @self._sio.on('span')
            async def on_span(data: dict) -> None:
                dispatch_to_listeners = False
                async with self._span_replay_lock:
                    raw_batch_id = data.get('batch_id')
                    batch_id = (
                        raw_batch_id
                        if isinstance(raw_batch_id, int) and not isinstance(raw_batch_id, bool) and raw_batch_id >= 0
                        else None
                    )
                    replay_batch = batch_id is not None
                    if batch_id is None:
                        self._record_span_replay_error(
                            None,
                            ValueError('runtime span batch has missing or invalid batch_id'),
                        )
                    else:
                        replay_batch = self._accept_span_batch(batch_id)
                    try:
                        if replay_batch:
                            items = data.get('items')
                            if not isinstance(items, list):
                                self._record_span_replay_error(
                                    batch_id, ValueError('runtime span batch items must be a list')
                                )
                                items = []
                            for span_event in items:
                                try:
                                    self._handle_span_event(span_event)
                                except Exception as exc:
                                    self._record_span_replay_error(batch_id, exc)
                                    logger.exception('Failed to replay runtime span event')
                            dispatch_to_listeners = not data.get('barrier')
                    finally:
                        if batch_id is not None:
                            await self._mark_span_batch_replayed(batch_id)
                if dispatch_to_listeners:
                    await self._dispatch_event_listeners('span', data)

        return self._sio

    async def _try_auto_connect_sio(self) -> None:
        """Best-effort auto-connect SIO for events while preserving caller cancellation."""
        if not self.config.auto_connect_ws or self._sio_auto_connect_disabled or self._ws_connected:
            return
        try:
            sio = self._ensure_sio_client()
            await await_remote_operation(
                sio.connect(self.config.base_url, headers=self.config.default_headers or None),
                logger=logger,
                phase='sio_auto_connect',
                base_url=self.config.base_url,
            )
            logger.info(f'socketio: auto-connected to {self.config.base_url}')
        except (RemoteChildOperationCancelledError, asyncio.CancelledError) as error:
            cancellation_count = get_current_task_cancellation_count()
            if isinstance(error, asyncio.CancelledError) and (cancellation_count is None or cancellation_count > 0):
                raise
            self._sio_auto_connect_disabled = True
            with contextlib.suppress(Exception):
                await sio.disconnect()
            self._sio = None
            logger.warning(
                'socketio: auto-connect child operation was cancelled. '
                'Real-time log streaming disabled; invoke will use HTTP polling fallback.'
            )
        except Exception as e:
            self._sio_auto_connect_disabled = True
            logger.warning(
                f'socketio: auto-connect failed ({type(e).__name__}: {e}). '
                f'Real-time log streaming disabled; invoke will use HTTP polling fallback.'
            )

    async def _ensure_sio_connected(self) -> socketio.AsyncClient:
        """Connect SIO with exponential backoff (for sandbox startup delays)."""
        sio = self._ensure_sio_client()
        if self._ws_connected:
            return sio

        max_attempts = self.config.initial_connect_attempts
        base_delay = self.config.initial_connect_delay
        delay_max = self.config.initial_connect_delay_max
        jitter = self.config.initial_connect_jitter
        ns_wait_timeout = self.config.initial_connect_ns_wait_timeout
        last_error: Exception | None = None

        for attempt in range(1, max_attempts + 1):
            try:
                if sio.connected:
                    logger.warning('socketio: client reports connected but _ws_connected is False; resetting')
                    with contextlib.suppress(Exception):
                        await sio.disconnect()

                await sio.connect(
                    self.config.base_url,
                    wait_timeout=ns_wait_timeout,
                    headers=self.config.default_headers or None,
                )
                logger.info(f'socketio: connected to {self.config.base_url}')
                return sio
            except Exception as e:
                last_error = e
                logger.warning(f'socketio: attempt {attempt}/{max_attempts} failed: {type(e).__name__}: {e}')

                err_msg = str(e)
                if (
                    'namespace' in err_msg.lower()
                    or 'closed' in err_msg.lower()
                    or 'already connected' in err_msg.lower()
                ):
                    logger.info('socketio: forcing client reset')
                    await self._force_reset_sio_client()
                    sio = self._ensure_sio_client()

                if attempt < max_attempts:
                    raw_delay = min(base_delay * (2 ** (attempt - 1)), delay_max)
                    await asyncio.sleep(raw_delay * random.uniform(1 - jitter, 1 + jitter))

        raise RuntimeError(
            f'socketio: failed to connect to {self.config.base_url} after {max_attempts} attempts: {last_error}'
        )

    async def _force_reset_sio_client(self) -> None:
        """Discard and re-create the SIO client after unrecoverable state."""
        if self._sio is not None:
            with contextlib.suppress(Exception):
                if self._sio.connected:
                    await self._sio.disconnect()
            self._sio = None
        self._ws_connected = False

    def _handle_runtime_log(self, log_data: dict) -> None:
        try:
            level = log_data.get('level', 'INFO')
            name = log_data.get('name', 'runtime')
            message = log_data.get('message', '')
            runtime_logger = logging.getLogger(name)
            level_map = {
                'DEBUG': logging.DEBUG,
                'INFO': logging.INFO,
                'WARNING': logging.WARNING,
                'ERROR': logging.ERROR,
                'CRITICAL': logging.CRITICAL,
            }
            runtime_logger.log(level_map.get(level, logging.INFO), message)
        except Exception as e:
            logger.error(f'Failed to handle runtime log event: {e}')

    def _handle_worker_log(self, log_data: dict) -> None:
        """Surface a worker-process log record forwarded via the ``worker_log`` event.

        The worker's ``_SIOLogHandler`` emits one record at a time as
        ``{worker_id, venv_id, level, name, message}`` (not the batched
        ``{items:[...]}`` shape used by ``log``). Re-log it under the record's
        original logger name so callers see the true origin (e.g.
        ``nexus.runtime.server.services.rsio_llm_bridge``) instead of a
        catch-all bucket, and prefix the message with the worker id.
        """
        try:
            level = log_data.get('level', 'INFO')
            worker_id = log_data.get('worker_id', 'worker')
            name = log_data.get('name') or 'nexus.runtime.worker'
            message = log_data.get('message', '')
            level_map = {
                'DEBUG': logging.DEBUG,
                'INFO': logging.INFO,
                'WARNING': logging.WARNING,
                'ERROR': logging.ERROR,
                'CRITICAL': logging.CRITICAL,
            }
            logging.getLogger(name).log(level_map.get(level, logging.INFO), '[%s] %s', worker_id, message)
        except Exception as e:
            logger.error(f'Failed to handle worker log event: {e}')

    def _handle_span_event(self, data: dict) -> None:
        from nexus.trace import SpanController, SpanEndEvent, SpanStartEvent, SpanUpdateEvent

        event_type = data.get('type', '').upper()
        if event_type == 'START':
            event = SpanStartEvent.model_validate(data)
        elif event_type == 'UPDATE':
            event = SpanUpdateEvent.model_validate(data)
        elif event_type == 'END':
            event = SpanEndEvent.model_validate(data)
        else:
            raise ValueError(f'unknown runtime span event type: {event_type or "missing"}')
        controller = SpanController.get_instance_sync()
        if event_type != 'START' and event.span_id not in controller.active_span_dict:
            raise RuntimeError(f'runtime span {event.span_id} has no active START event')
        controller.add_event(event, origin='replay')
        logger.debug(f'Replayed span event: {event_type} {data.get("span_id")}')

    def _record_span_replay_error(self, batch_id: int | None, error: Exception) -> None:
        if self._span_replay_error is None:
            self._span_replay_error = (batch_id, error)

    def _accept_span_batch(self, batch_id: int) -> bool:
        if not self._span_batch_baseline_established:
            self._span_batch_baseline_established = True
            return True
        expected_batch_id = self._last_replayed_span_batch_id + 1
        if batch_id == expected_batch_id:
            return True
        if batch_id <= self._last_replayed_span_batch_id:
            self._record_span_replay_error(
                batch_id,
                RuntimeError(f'duplicate or out-of-order span batch {batch_id}'),
            )
            return False
        self._record_span_replay_error(
            batch_id,
            RuntimeError(f'runtime span batch gap: expected {expected_batch_id}, received {batch_id}'),
        )
        return True

    async def _mark_span_batch_replayed(self, batch_id: int) -> None:
        async with self._span_replay_condition:
            self._last_replayed_span_batch_id = max(self._last_replayed_span_batch_id, batch_id)
            self._span_replay_condition.notify_all()

    async def _wait_for_span_batch(self, batch_id: int, *, timeout: float) -> None:
        async with self._span_replay_condition:
            if self._last_replayed_span_batch_id >= batch_id:
                return
            if timeout <= 0:
                raise asyncio.TimeoutError
            await asyncio.wait_for(
                self._span_replay_condition.wait_for(lambda: self._last_replayed_span_batch_id >= batch_id),
                timeout=timeout,
            )

    def _raise_span_replay_error(self, batch_id: int) -> None:
        failure = self._span_replay_error
        if failure is None:
            return
        error_batch_id, error = failure
        if error_batch_id is None or error_batch_id <= batch_id:
            raise RuntimeError(f'Runtime span replay failed: {error}') from error

    async def ws_connect(self) -> None:
        """Establish Socket.IO connection (for events, not required for invoke)."""
        await self._ensure_sio_connected()

    async def ws_disconnect(self) -> None:
        if self._sio is not None:
            if hasattr(self._sio, 'shutdown'):
                await self._sio.shutdown()
            else:
                await self._sio.disconnect()

    async def _supports_trace_flush(self) -> bool:
        if self._trace_flush_supported is not None:
            return self._trace_flush_supported

        timeout = aiohttp.ClientTimeout(total=min(self.config.socketio_request_timeout, 5.0))
        response = await await_remote_operation(
            http_client.get(
                f'{self.config.base_url}/health',
                timeout=timeout,
                headers=self.config.default_headers or None,
            ),
            logger=logger,
            phase='trace_health',
            base_url=self.config.base_url,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise RuntimeError('Runtime health response must be a JSON object')
        self._trace_flush_supported = supports_trace_flush(payload.get('capabilities'))
        logger.info(
            'Runtime %s Socket.IO trace flush capability: %s',
            self.config.base_url,
            'supported' if self._trace_flush_supported else 'legacy/unsupported',
        )
        return self._trace_flush_supported

    async def drain_trace(self, timeout: float | None = None) -> None:
        """Wait for the server transport and local replay of all preceding span events."""
        if not await self._supports_trace_flush():
            return
        if not self.config.auto_connect_ws:
            raise RuntimeError('Cannot drain runtime trace when Socket.IO is disabled')
        if self._remote_operation_started_without_sio:
            raise RuntimeError('Cannot drain runtime trace: remote operation started before Socket.IO connected')
        if not self._ws_connected:
            await await_remote_operation(
                self._ensure_sio_connected(),
                logger=logger,
                phase='trace_sio_connect',
                base_url=self.config.base_url,
            )
        assert self._sio is not None
        timeout_seconds = timeout if timeout is not None else self.config.socketio_request_timeout
        deadline = asyncio.get_running_loop().time() + timeout_seconds
        response = await await_remote_operation(
            self._sio.call(
                'trace:flush',
                {},
                timeout=timeout_seconds,
            ),
            logger=logger,
            phase='trace_flush',
            base_url=self.config.base_url,
        )
        if not isinstance(response, dict) or not response.get('ok'):
            error = response.get('error') if isinstance(response, dict) else None
            raise RuntimeError(error or 'Runtime trace barrier failed')
        span_batch_id = response.get('span_batch_id')
        if not isinstance(span_batch_id, int) or isinstance(span_batch_id, bool) or span_batch_id < 0:
            raise RuntimeError('Runtime trace barrier returned an invalid span_batch_id')
        remaining = deadline - asyncio.get_running_loop().time()
        await await_remote_operation(
            self._wait_for_span_batch(span_batch_id, timeout=remaining),
            logger=logger,
            phase='trace_span_replay',
            base_url=self.config.base_url,
        )
        self._raise_span_replay_error(span_batch_id)

    @property
    def ws_connected(self) -> bool:
        return self._ws_connected

    # ==================== Event Management ====================

    def on(
        self,
        event: str,
        listener: Callable[..., None] | Callable[..., Awaitable[None]],
    ) -> Callable[[], None]:
        sio = self._ensure_sio_client()
        if not asyncio.iscoroutinefunction(listener) and not inspect.iscoroutinefunction(listener):
            original_listener = listener

            async def async_wrapper(*args, **kwargs):
                original_listener(*args, **kwargs)

            wrapper = async_wrapper
        else:
            wrapper = listener

        listeners = self._event_listeners.setdefault(event, [])
        listeners.append(wrapper)
        if event not in {'log', 'span'}:
            sio.on(event, self._make_event_dispatcher(event))

        def unsubscribe():
            event_listeners = self._event_listeners.get(event)
            if event_listeners and wrapper in event_listeners:
                event_listeners.remove(wrapper)
            if event_listeners == []:
                self._event_listeners.pop(event, None)

        return unsubscribe

    def _make_event_dispatcher(self, event: str) -> Callable[..., Awaitable[None]]:
        async def dispatch(*args: Any, **kwargs: Any) -> None:
            await self._dispatch_event_listeners(event, *args, **kwargs)

        return dispatch

    async def _dispatch_event_listeners(self, event: str, *args: Any, **kwargs: Any) -> None:
        for listener in list(self._event_listeners.get(event, ())):
            try:
                await listener(*args, **kwargs)
            except Exception:
                logger.debug('Socket.IO event listener failed: %s', event, exc_info=True)

    def once(
        self,
        event: str,
        listener: Callable[..., None] | Callable[..., Awaitable[None]],
    ) -> Callable[[], None]:
        triggered = False
        unsubscribe_func: Callable[[], None] | None = None

        async def once_wrapper(*args, **kwargs):
            nonlocal triggered
            if triggered:
                return
            triggered = True
            if unsubscribe_func:
                unsubscribe_func()
            result = listener(*args, **kwargs)
            if asyncio.iscoroutine(result) or inspect.isawaitable(result):
                await result

        unsubscribe_func = self.on(event, once_wrapper)
        return unsubscribe_func

    def off(
        self,
        event: str,
        listener: Callable[..., None] | Callable[..., Awaitable[None]] | None = None,
    ) -> None:
        if listener is None:
            self._event_listeners.pop(event, None)
            return
        listeners = self._event_listeners.get(event)
        registered_listener = cast(Callable[..., Awaitable[None]], listener)
        if listeners and registered_listener in listeners:
            listeners.remove(registered_listener)
        if listeners == []:
            self._event_listeners.pop(event, None)

    async def emit(
        self,
        event: str,
        *args: Any,
        ack_callback: Callable[..., None] | Callable[..., Awaitable[None]] | None = None,
    ) -> None:
        if not self._ws_connected:
            raise RuntimeError('Socket.IO is not connected. Call ws_connect() first.')
        sio = self._ensure_sio_client()
        data = args[0] if len(args) == 1 else args if args else None
        if ack_callback is not None:
            await sio.emit(event, data, callback=ack_callback)
        else:
            await sio.emit(event, data)


# ============== Deprecated Aliases ==============

HttpRuntimeConfig = RemoteRuntimeConfig
"""Deprecated: use :class:`RemoteRuntimeConfig`."""

HttpRuntime = RemoteRuntime
"""Deprecated: use :class:`RemoteRuntime`."""

SocketIoRuntimeConfig = RemoteRuntimeConfig
"""Deprecated: use :class:`RemoteRuntimeConfig`."""

SocketIoRuntime = RemoteRuntime
"""Deprecated: use :class:`RemoteRuntime`."""


__all__ = [
    'HttpRuntime',
    'HttpRuntimeConfig',
    'RemoteRuntime',
    'RemoteRuntimeConfig',
    'SocketIoRuntime',
    'SocketIoRuntimeConfig',
]
