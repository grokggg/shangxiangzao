"""Runtime server for executing tools and functions in sandbox environment.

This server runs inside a sandbox (container, VM, etc.) and handles:
- Unified invoke endpoint (import_path + pickle args)
- Tool execution with context passing (via import_path)
- Executor invocation via registry
- Real-time event communication via Socket.IO

Primary transport:
- POST /api/v1/invoke + rpc:invoke (Socket.IO).
Legacy:
- /api/v1/call_tool, /api/v1/call_executor and their SIO counterparts.
"""

import asyncio
import base64
import inspect
import logging
import os
import time
import traceback
import uuid

from contextlib import asynccontextmanager, suppress
from datetime import datetime, timezone
from importlib import resources
from typing import Any, TypeGuard

import aiohttp
import socketio

from fastapi import APIRouter, FastAPI, Request, WebSocket
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from pydantic import BaseModel

from nexus.core import (
    ToolContext,
    ToolResponse,
    executor_registry,
    get_tool_meta,
    import_from_path,
)
from nexus.core.registry import Registry
from nexus.runtime.capabilities import runtime_capabilities, supports_trace_flush
from nexus.runtime.client.models import (
    CallExecutorRequest,
)
from nexus.runtime.client.models import (
    CallToolRequest as ClientCallToolRequest,
)
from nexus.runtime.context import RuntimeControl, SIOBridge
from nexus.runtime.models import (
    BLOB_THRESHOLD_BYTES,
    InvokeJobInfo,
    InvokeRequest,
    InvokeResponse,
    JobStatus,
    execute_invoked_func,
    invoke_response_from_exception,
    pack_result_for_transport,
)
from nexus.runtime.server.backend import InProcessBackend
from nexus.runtime.server.debug import (
    aiomonitor_webui_port,
    build_debug_page,
    build_novnc_path,
    rewrite_aiomonitor_html,
    validate_debug_port,
)
from nexus.runtime.server.rpc_replay_guard import RPCReplayGuard
from nexus.trace import RemoteTraceContext
from nexus.trace.processors.socketio import SocketIOSpanBroadcaster
from nexus.trace.tracer import add_span_processor, remove_span_processor, with_remote_trace_context
from nexus.utils.log import setup_logging
from nexus.utils.otel.lifecycle import async_force_flush_best_effort
from nexus.utils.otel.propagation import inject_carrier, use_remote_context
from nexus.utils.otel.transport import transport_span
from nexus.utils.socketio import SafeAsyncServer, SocketIOEventLoggerAdapter


# ============== Activity Tracking ==============

_last_activity_time: float = time.time()
_server_start_time: float = time.time()


def update_activity() -> None:
    """Update last activity timestamp to current time.

    Note: No lock needed in async context - all coroutines run in single thread
    and simple assignment is atomic (no await between read and write).
    """
    global _last_activity_time
    _last_activity_time = time.time()


def get_last_activity_time() -> float:
    """Get the timestamp of last activity."""
    return _last_activity_time


def get_idle_seconds() -> float:
    """Get seconds since last activity."""
    return time.time() - _last_activity_time


def safe_repr(obj, max_len=10000):
    """安全打印对象，防止大对象或不可打印对象炸日志"""
    try:
        s = repr(obj)
        return s[:max_len] + '...' if len(s) > max_len else s
    except Exception:
        return f'<{type(obj).__name__}>'


def _get_env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _get_env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _get_env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {'1', 'true', 'yes', 'on'}


_SOCKETIO_PING_INTERVAL = _get_env_float('NEXUS_RUNTIME_SIO_PING_INTERVAL', 15.0)
_SOCKETIO_PING_TIMEOUT = _get_env_float('NEXUS_RUNTIME_SIO_PING_TIMEOUT', 120.0)
_ACTIVITY_HEARTBEAT_INTERVAL = _get_env_float('NEXUS_RUNTIME_ACTIVITY_HEARTBEAT_INTERVAL', 10.0)
_RUNTIME_AIOMONITOR_ENABLED = _get_env_bool('NEXUS_RUNTIME_AIOMONITOR_ENABLED', False)
_RUNTIME_AIOMONITOR_HOST = os.getenv('NEXUS_RUNTIME_AIOMONITOR_HOST', '127.0.0.1')
_RUNTIME_AIOMONITOR_BASE_PORT = _get_env_int('NEXUS_RUNTIME_AIOMONITOR_BASE_PORT', 50100)
_RUNTIME_AIOMONITOR_WEBUI_BASE_PORT = _get_env_int('NEXUS_RUNTIME_AIOMONITOR_WEBUI_BASE_PORT', 50200)
_RUNTIME_AIOMONITOR_CONSOLE_BASE_PORT = _get_env_int('NEXUS_RUNTIME_AIOMONITOR_CONSOLE_BASE_PORT', 50300)
_RPC_REPLAY_GUARD = RPCReplayGuard(
    ttl_seconds=_get_env_float('NEXUS_RUNTIME_RPC_REPLAY_TTL', 3600.0),
    max_entries=_get_env_int('NEXUS_RUNTIME_RPC_REPLAY_MAX_ENTRIES', 2048),
)


@asynccontextmanager
async def keep_activity_alive():
    """Refresh activity while a long-running RPC/HTTP handler is still active."""
    update_activity()
    if _ACTIVITY_HEARTBEAT_INTERVAL <= 0:
        try:
            yield
        finally:
            update_activity()
        return

    stop_event = asyncio.Event()

    async def heartbeat() -> None:
        while not stop_event.is_set():
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=_ACTIVITY_HEARTBEAT_INTERVAL)
            except asyncio.TimeoutError:
                update_activity()

    task = asyncio.create_task(heartbeat())
    try:
        yield
    finally:
        stop_event.set()
        with suppress(asyncio.CancelledError):
            await task
        update_activity()


# ============== Socket.IO Server ==============

sio = SafeAsyncServer(
    async_mode='asgi',
    cors_allowed_origins='*',
    # Route the socketio server's own traces through our namespace so they
    # inherit the root formatter (in particular ``TrimFormatter``) instead of
    # the library's default StreamHandler-to-stderr, which would print
    # untrimmed messages alongside our trimmed ones.
    logger=SocketIOEventLoggerAdapter(logging.getLogger('nexus.runtime.server.socketio'), {}),
    engineio_logger=False,
    ping_interval=_SOCKETIO_PING_INTERVAL,
    ping_timeout=_SOCKETIO_PING_TIMEOUT,
    monitor_clients=True,
    max_http_buffer_size=100 * 1024 * 1024,  # 100MB for large payloads
)

socketio_log_handler = setup_logging(socketio_server=sio, log_level=logging.DEBUG)

# Register span broadcaster for distributed tracing
# This broadcasts all span events to connected clients via Socket.IO
span_broadcaster = SocketIOSpanBroadcaster(sio, event_name='span')

# Import executors to register them
Registry.import_module('nexus.runtime.executors')


logger = logging.getLogger(__name__)
logger.info(
    'socketio: server heartbeat configured ping_interval=%ss ping_timeout=%ss',
    _SOCKETIO_PING_INTERVAL,
    _SOCKETIO_PING_TIMEOUT,
)


@sio.event
async def connect(sid: str, environ: dict, auth: dict | None = None):
    """Handle client connection."""
    update_activity()
    logger.info(f'socketio: client connected, sid={sid}')


@sio.event
async def disconnect(sid: str):
    """Handle client disconnection."""
    update_activity()
    logger.info(f'socketio: client disconnected, sid={sid}')


# Events to relay from Host (default ns) back to Workers
_RELAY_TO_WORKER = frozenset({'llm_response'})
_RELAY_TO_WORKER_PREFIXES = ('task_files:session_response:', 'task_files:local_file_response:')


@sio.on('*')
async def catch_all(event: str, sid: str, *args):
    """Catch-all: debug logging + relay whitelisted events to Workers."""
    update_activity()
    logger.debug(f'socketio: received event={event}, sid={sid}, args={safe_repr(args)}')
    if event in _RELAY_TO_WORKER or event.startswith(_RELAY_TO_WORKER_PREFIXES):
        await sio.emit(event, *args, namespace='/worker')


# ============== Socket.IO RPC Handlers ==============


@sio.on('rpc:call_tool')
async def sio_call_tool(sid: str, data: dict) -> dict:
    """Execute a tool by importing it from the given path via Socket.IO RPC.

    The tool function is loaded via import_path, validated, and executed.
    Returns result as ACK to the client's sio.call().
    """
    update_activity()
    try:
        request = ClientCallToolRequest(**data)
    except Exception as exc:
        logger.error(f'rpc:call_tool: invalid request from sid={sid}: {exc}')
        logger.error(f'rpc:call_tool: traceback:\n{traceback.format_exc()}')
        update_activity()
        return {
            'success': False,
            'error': str(exc),
            'detail': exc.__class__.__name__,
            'traceback': traceback.format_exc(),
        }

    logger.info(
        'rpc:call_tool: received request_id=%s, import_path=%s',
        request.request_id or '-',
        request.import_path,
    )

    async def execute_once() -> dict:
        t_start = time.monotonic()
        try:
            tool_func = import_from_path(request.import_path)
            meta = get_tool_meta(tool_func)

            logger.info(
                "rpc:call_tool: executing request_id=%s tool='%s', params=%s",
                request.request_id or '-',
                meta.name,
                safe_repr(request.params),
            )

            async def execute():
                parsed_params = meta.input_model.model_validate(request.params)
                return await tool_func(parsed_params, request.context)

            async with keep_activity_alive():
                if request.trace_context:
                    with with_remote_trace_context(
                        request.trace_context.trace_id,
                        request.trace_context.parent_span_id,
                        run_id=request.trace_context.run_id,
                        run_span_id=request.trace_context.run_span_id,
                        run_kind=request.trace_context.run_kind,
                        otel_carrier=request.trace_context.otel_carrier,
                    ):
                        result = await execute()
                else:
                    result = await execute()

            elapsed = time.monotonic() - t_start
            logger.info(
                "rpc:call_tool: request_id=%s tool='%s' executed successfully, elapsed=%.3fs",
                request.request_id or '-',
                meta.name,
                elapsed,
            )
            return {'success': True, 'data': result.model_dump()}
        except Exception as exc:
            elapsed = time.monotonic() - t_start
            logger.error(
                'rpc:call_tool: request_id=%s failed with %s: %s, elapsed=%.3fs',
                request.request_id or '-',
                exc.__class__.__name__,
                exc,
                elapsed,
            )
            logger.error(f'rpc:call_tool: traceback:\n{traceback.format_exc()}')
            return {
                'success': False,
                'error': str(exc),
                'detail': exc.__class__.__name__,
                'traceback': traceback.format_exc(),
            }

    return await _RPC_REPLAY_GUARD.run('rpc:call_tool', request.request_id, execute_once)


@sio.on('rpc:call_executor')
async def sio_call_executor(sid: str, data: dict) -> dict:
    """Invoke a registered executor via Socket.IO RPC.

    Returns result as ACK to the client's sio.call().
    """
    update_activity()
    try:
        request = CallExecutorRequest(**data)
    except Exception as exc:
        logger.error(f'rpc:call_executor: invalid request from sid={sid}: {exc}')
        logger.error(f'rpc:call_executor: traceback:\n{traceback.format_exc()}')
        update_activity()
        return {
            'success': False,
            'error': str(exc),
            'detail': exc.__class__.__name__,
            'traceback': traceback.format_exc(),
        }

    logger.info(
        'rpc:call_executor: received request_id=%s, executor_name=%s',
        request.request_id or '-',
        request.executor_name,
    )

    async def execute_once() -> dict:
        t_start = time.monotonic()
        try:
            executor = executor_registry.get(request.executor_name)
            if executor is None:
                available = list(executor_registry.get_all().keys())
                raise ValueError(f"Executor '{request.executor_name}' not found. Available: {available}")

            sig = inspect.signature(executor)
            input_type = list(sig.parameters.values())[0].annotation
            parsed_arg = input_type.model_validate(request.arg)

            logger.info(
                "rpc:call_executor: executing request_id=%s executor='%s', arg=%s",
                request.request_id or '-',
                request.executor_name,
                safe_repr(request.arg),
            )

            async def execute():
                return await executor(parsed_arg)

            async with keep_activity_alive():
                if request.trace_context:
                    with with_remote_trace_context(
                        request.trace_context.trace_id,
                        request.trace_context.parent_span_id,
                        run_id=request.trace_context.run_id,
                        run_span_id=request.trace_context.run_span_id,
                        run_kind=request.trace_context.run_kind,
                        otel_carrier=request.trace_context.otel_carrier,
                    ):
                        result = await execute()
                else:
                    result = await execute()

            elapsed = time.monotonic() - t_start
            logger.info(
                "rpc:call_executor: request_id=%s executor='%s' executed successfully, elapsed=%.3fs",
                request.request_id or '-',
                request.executor_name,
                elapsed,
            )

            if hasattr(result, 'model_dump'):
                return {'success': True, 'data': result.model_dump()}
            if isinstance(result, dict):
                return {'success': True, 'data': result}
            return {'success': True, 'data': result}
        except Exception as exc:
            elapsed = time.monotonic() - t_start
            logger.error(
                "rpc:call_executor: request_id=%s executor='%s' failed with %s: %s, elapsed=%.3fs",
                request.request_id or '-',
                request.executor_name,
                exc.__class__.__name__,
                exc,
                elapsed,
            )
            logger.error(f'rpc:call_executor: traceback:\n{traceback.format_exc()}')
            return {
                'success': False,
                'error': str(exc),
                'detail': exc.__class__.__name__,
                'traceback': traceback.format_exc(),
            }

    return await _RPC_REPLAY_GUARD.run('rpc:call_executor', request.request_id, execute_once)


# ============== Unified Invoke ==============

_backend = InProcessBackend()
_runtime_control = RuntimeControl(_backend, sio=SIOBridge(sio))

# --- Job store (in-memory, for async invoke polling) ---
# Maps request_id -> InvokeJobInfo.  Entries are ephemeral; a TTL sweep
# runs periodically to avoid unbounded growth.
_jobs: dict[str, InvokeJobInfo] = {}
_JOB_TTL_SECONDS: float = 600.0  # completed jobs kept for 10 min

# --- Blob store (in-memory, for large payload / result sideband) ---
_blobs: dict[str, bytes] = {}
_BLOB_TTL_SECONDS: float = 600.0

# BLOB_THRESHOLD_BYTES is imported from nexus.runtime.models

# Default long-poll wait when client doesn't send Prefer header.
_DEFAULT_WAIT_SECONDS: float = 30.0


@sio.on('rpc:invoke')
async def sio_invoke(sid: str, data: dict) -> dict:
    """Unified invoke handler via Socket.IO RPC.

    Executes synchronously and returns the InvokeResponse as ACK.
    Also emits ``invoke:done:{request_id}`` so that clients using the
    HTTP long-poll / polling path can be notified via SIO push.
    """
    update_activity()
    try:
        request = InvokeRequest(**data)
    except Exception as exc:
        logger.error(f'rpc:invoke: invalid request from sid={sid}: {exc}')
        return invoke_response_from_exception(exc, request_id=data.get('request_id')).model_dump()

    logger.info(
        'rpc:invoke: received request_id=%s, import_path=%s',
        request.request_id or '-',
        request.import_path,
    )

    async def execute_once() -> dict:
        response = await _execute_invoke(request)
        return response.model_dump()

    return await _RPC_REPLAY_GUARD.run('rpc:invoke', request.request_id, execute_once)


@sio.on('rpc:create_venv')
async def sio_create_venv(sid: str, data: dict) -> dict:
    """Create a worker venv via Socket.IO RPC.

    Blocks until the worker subprocess connects via SIO.
    """
    update_activity()
    try:
        from nexus.runtime.server.venv.models import VenvSpec

        spec = VenvSpec(**data)
        vid = uuid.uuid4().hex[:12]
        status = await _create_venv_full(vid, spec)
        return status.model_dump(mode='json')
    except Exception as exc:
        logger.error(f'rpc:create_venv: failed: {exc}')
        return {'error': str(exc)}


@sio.on('rpc:destroy_venv')
async def sio_destroy_venv(sid: str, data: dict) -> dict:
    """Destroy a worker venv via Socket.IO RPC."""
    update_activity()
    venv_id = data.get('venv_id', '')
    try:
        await _worker_launcher.stop_worker(f'worker-{venv_id}')
        await _venv_manager.destroy(venv_id)
        return {'status': 'destroyed', 'venv_id': venv_id}
    except Exception as exc:
        logger.error(f'rpc:destroy_venv: failed: {exc}')
        return {'error': str(exc)}


# Pending futures for emit-based worker invoke (no sio.call timeout).
_worker_invoke_futures: dict[str, asyncio.Future[dict]] = {}


async def _execute_invoke_in_worker(request: InvokeRequest) -> InvokeResponse:
    """Dispatch an invoke request to a worker process via Socket.IO.

    Uses emit + future instead of sio.call() so there is no artificial
    SIO-level timeout — the only timeout is from the caller (HTTP long-poll
    or host request_timeout), matching the Base execution path.
    """
    worker_id = f'worker-{request.venv_id}'
    worker_sid = _worker_sids.get(worker_id)
    if not worker_sid:
        raise ValueError(
            f'No connected worker for venv_id={request.venv_id!r}. Connected workers: {list(_worker_sids.keys())}'
        )

    logger.info(
        'invoke: dispatching to worker %s (sid=%s) import_path=%s',
        worker_id,
        worker_sid,
        request.import_path,
    )

    # Offload large payload to blob store so the SIO message stays small.
    if request.payload and len(request.payload) > BLOB_THRESHOLD_BYTES:
        import uuid as _uuid

        blob_id = _uuid.uuid4().hex
        _blobs[blob_id] = base64.b64decode(request.payload)
        request = request.model_copy(update={'payload': '', 'payload_ref': blob_id})

    carrier = request.trace_context.otel_carrier if request.trace_context is not None else None
    with use_remote_context(carrier, fallback_only=True):
        with transport_span(
            'runtime.worker.invoke',
            kind='client',
            attributes={
                'rpc.system': 'socket.io',
                'rpc.method': 'invoke',
                'nexus.worker.id': worker_id,
            },
        ):
            if request.trace_context is not None:
                fresh_carrier = inject_carrier()
                if fresh_carrier:
                    request = request.model_copy(
                        update={
                            'trace_context': request.trace_context.model_copy(update={'otel_carrier': fresh_carrier})
                        }
                    )

            rid = request.request_id or ''
            future: asyncio.Future[dict] = asyncio.get_event_loop().create_future()
            _worker_invoke_futures[rid] = future

            try:
                await sio.emit('invoke', request.model_dump(), to=worker_sid, namespace=worker_ns)
                result = await future
            finally:
                _worker_invoke_futures.pop(rid, None)

    if isinstance(result, dict) and 'success' in result:
        return InvokeResponse(**result)

    raise RuntimeError(f'Unexpected worker response: {result}')


async def _execute_invoke(request: InvokeRequest) -> InvokeResponse:
    """Execute an invoke under its transport context, using the payload carrier to repair conflicting traces."""
    carrier = request.trace_context.otel_carrier if request.trace_context is not None else None
    with use_remote_context(carrier, fallback_only=True, fallback_policy='same_trace_id'):
        with transport_span(
            'runtime.invoke',
            kind='server',
            attributes={
                'rpc.system': 'nexus.runtime',
                'rpc.method': 'invoke',
                'nexus.runtime.target': request.import_path,
                'nexus.request.id': request.request_id or '',
            },
        ):
            return await _execute_invoke_with_current_context(request)


async def _execute_invoke_with_current_context(request: InvokeRequest) -> InvokeResponse:
    """Shared invoke execution logic for HTTP, SIO, and async-job paths.

    Handles:
    - Worker venv dispatch (when ``venv_id`` is set).
    - ``payload_ref`` resolution from blob store.
    - Dual resolution: ``module::func`` via import_from_path, bare names via executor_registry.
    - Auto-coercion: dict args -> BaseModel subclasses.
    - Pickle: payload unpacking and result serialization.
    - Result blob offloading when the pickled result exceeds *BLOB_THRESHOLD_BYTES*.
    """

    # Dispatch to worker process when venv_id is set
    if request.venv_id:
        return await _execute_invoke_in_worker(request)

    t_start = time.monotonic()
    try:
        # 0. Resolve payload from blob store if payload_ref is set
        if request.payload_ref and not request.payload:
            blob = _blobs.pop(request.payload_ref, None)
            if blob is None:
                raise ValueError(f'Blob {request.payload_ref!r} not found or already consumed')
            request = request.model_copy(update={'payload': base64.b64encode(blob).decode('utf-8')})

        logger.info(
            'invoke: executing request_id=%s, import_path=%s',
            request.request_id or '-',
            request.import_path,
        )

        async with keep_activity_alive():
            result = await execute_invoked_func(
                request.import_path,
                request.payload,
                _runtime_control,
                trace_context=request.trace_context,
            )

        elapsed = time.monotonic() - t_start
        logger.info(
            'invoke: request_id=%s import_path=%s executed successfully, elapsed=%.3fs',
            request.request_id or '-',
            request.import_path,
            elapsed,
        )

        response, blob_id, blob_bytes = pack_result_for_transport(result, request_id=request.request_id)
        if blob_bytes is not None:
            assert blob_id is not None
            _blobs[blob_id] = blob_bytes

        return response

    except Exception as exc:
        elapsed = time.monotonic() - t_start
        logger.error(
            'invoke: request_id=%s failed with %s: %s, elapsed=%.3fs',
            request.request_id or '-',
            exc.__class__.__name__,
            exc,
            elapsed,
        )
        logger.error(f'invoke: traceback:\n{traceback.format_exc()}')
        return invoke_response_from_exception(exc, request_id=request.request_id)


async def _run_invoke_job(request: InvokeRequest) -> None:
    """Execute an invoke request as a background job, updating the job store.

    Also emits ``invoke:done:{request_id}`` via SIO so clients
    waiting on push notification are unblocked immediately.
    """
    rid = request.request_id or ''
    _jobs[rid] = InvokeJobInfo(request_id=rid, status=JobStatus.RUNNING)
    try:
        response = await _execute_invoke(request)
        status = JobStatus.DONE if response.success else JobStatus.ERROR
        _jobs[rid] = InvokeJobInfo(request_id=rid, status=status, response=response)
    except Exception as exc:
        response = invoke_response_from_exception(exc, request_id=rid)
        _jobs[rid] = InvokeJobInfo(request_id=rid, status=JobStatus.ERROR, response=response)

    # Push notification via SIO (best-effort)
    with suppress(Exception):
        await sio.emit(f'invoke:done:{rid}', response.model_dump())


# ============== FastAPI App ==============

_runtime_aiomonitor_handle: Any | None = None


async def start_runtime_aiomonitor() -> None:
    """Start aiomonitor for the runtime server event loop when explicitly enabled."""
    global _runtime_aiomonitor_handle
    if not _RUNTIME_AIOMONITOR_ENABLED:
        return
    try:
        import aiomonitor
    except ImportError:
        logger.warning('Runtime aiomonitor is enabled but the aiomonitor package is not installed')
        return
    try:
        _runtime_aiomonitor_handle = aiomonitor.start_monitor(
            asyncio.get_running_loop(),
            host=_RUNTIME_AIOMONITOR_HOST,
            port=_RUNTIME_AIOMONITOR_BASE_PORT,
            webui_port=_RUNTIME_AIOMONITOR_WEBUI_BASE_PORT,
            console_port=_RUNTIME_AIOMONITOR_CONSOLE_BASE_PORT,
            locals={'loop_name': 'runtime-server'},
        )
    except Exception:
        logger.exception(
            'Failed to start runtime aiomonitor on %s:%s/%s/%s',
            _RUNTIME_AIOMONITOR_HOST,
            _RUNTIME_AIOMONITOR_BASE_PORT,
            _RUNTIME_AIOMONITOR_WEBUI_BASE_PORT,
            _RUNTIME_AIOMONITOR_CONSOLE_BASE_PORT,
        )
        _runtime_aiomonitor_handle = None
        return
    logger.info(
        'Started runtime aiomonitor on %s:%s webui=%s console=%s',
        _RUNTIME_AIOMONITOR_HOST,
        _RUNTIME_AIOMONITOR_BASE_PORT,
        _RUNTIME_AIOMONITOR_WEBUI_BASE_PORT,
        _RUNTIME_AIOMONITOR_CONSOLE_BASE_PORT,
    )


async def stop_runtime_aiomonitor() -> None:
    global _runtime_aiomonitor_handle
    handle = _runtime_aiomonitor_handle
    _runtime_aiomonitor_handle = None
    close = getattr(handle, 'close', None)
    if close is not None:
        close()


@asynccontextmanager
async def runtime_lifespan(_app: FastAPI):
    if socketio_log_handler is not None:
        await socketio_log_handler.start_async()
    await span_broadcaster.start()
    add_span_processor(span_broadcaster)
    await start_runtime_aiomonitor()
    try:
        yield
    finally:
        remove_span_processor(span_broadcaster)
        try:
            await span_broadcaster.close()
        finally:
            try:
                if socketio_log_handler is not None:
                    await socketio_log_handler.close_async()
            finally:
                await stop_runtime_aiomonitor()


app = FastAPI(lifespan=runtime_lifespan)

# Mount MCP proxy ASGI app for streamable HTTP access to MCP servers
# Import is deferred because executors are registered above via Registry.import_module
from nexus.runtime.executors.mcp import mcp_proxy_asgi  # noqa: E402


app.mount('/mcp/proxy', mcp_proxy_asgi)
api_v1_router = APIRouter(prefix='/api/v1')


def _load_runtime_debug_html() -> str:
    return resources.files('nexus.runtime.server').joinpath('debug.html').read_text(encoding='utf-8')


def _executor_debug_specs() -> list[dict[str, Any]]:
    specs = []
    entries = sorted(executor_registry.list_entries(), key=lambda item: item.name)
    for entry in entries:
        name = entry.name
        executor = executor_registry.get(name)
        sig = inspect.signature(executor)
        params = [param for param in sig.parameters.values() if param.name != 'self']
        input_type = params[0].annotation if params else None
        response_type = getattr(input_type, 'response_type', None)

        request_schema = input_type.model_json_schema(by_alias=True) if _is_pydantic_model(input_type) else {}
        response_schema = response_type.model_json_schema(by_alias=True) if _is_pydantic_model(response_type) else {}

        specs.append({
            'name': name,
            'request_schema': request_schema,
            'response_schema': response_schema,
            'import_path': entry.import_path,
            'meta': entry.meta,
        })
    return specs


def _is_pydantic_model(value: Any) -> TypeGuard[type[BaseModel]]:
    return isinstance(value, type) and issubclass(value, BaseModel)


@app.middleware('http')
async def activity_tracking_middleware(request: Request, call_next):
    """Middleware to track activity on /call_* API endpoints.

    Updates last activity timestamp when request enters and exits,
    enabling idle time monitoring for auto-cleanup.
    """
    path = request.url.path
    # Track activity for /api/v1/call_* and /api/v1/invoke endpoints (not /health, etc.)
    is_call_endpoint = path.startswith('/api/v1/call_') or path == '/api/v1/invoke' or path.startswith('/api/v1/venv')
    if is_call_endpoint:
        update_activity()  # Track on request entry
        logger.debug(f'activity_tracking: request entered {path}')

    response = await call_next(request)

    if is_call_endpoint:
        update_activity()  # Track on request exit
        logger.debug(f'activity_tracking: request exited {path}')

    return response


@app.get('/health')
def health():
    """Health check endpoint with activity tracking info.

    Returns:
        status: Server status ('ok')
        last_activity_time: ISO format timestamp of last activity (UTC)
        idle_seconds: Seconds since last activity
        uptime_seconds: Seconds since server started
        capabilities: Supported optional runtime protocol versions
    """
    last_activity = get_last_activity_time()
    return {
        'status': 'ok',
        'last_activity_time': datetime.fromtimestamp(last_activity, tz=timezone.utc).isoformat(),
        'idle_seconds': round(get_idle_seconds(), 2),
        'uptime_seconds': round(time.time() - _server_start_time, 2),
        'capabilities': runtime_capabilities(),
    }


@app.get('/', response_class=HTMLResponse)
def debug_page():
    """Runtime executor debug UI."""
    return HTMLResponse(_load_runtime_debug_html())


@api_v1_router.get('/debug/executors')
def list_debug_executors():
    """Return executor schemas discovered from the executor registry."""
    return {'executors': _executor_debug_specs()}


@app.get('/debug/gui')
async def debug_gui(request: Request):
    debug_gui_path = request.url.path.rstrip('/')
    websocket_path = f'{debug_gui_path}/vnc/websockify'
    novnc_path = build_novnc_path(websocket_path, vnc_html_path=f'{debug_gui_path}/vnc/vnc.html')
    return HTMLResponse(build_debug_page(novnc_path=novnc_path, websocket_path=websocket_path))


def _novnc_http_origin() -> str:
    port = int(os.getenv('NEXUS_GUI_NOVNC_PORT', '6080'))
    return f'http://127.0.0.1:{port}'


def _novnc_ws_origin() -> str:
    port = int(os.getenv('NEXUS_GUI_NOVNC_PORT', '6080'))
    return f'ws://127.0.0.1:{port}'


@app.get('/debug/gui/vnc')
async def debug_gui_vnc_root(request: Request):
    return RedirectResponse(f'{request.url.path.rstrip("/")}/vnc.html')


@app.get('/debug/gui/vnc/{asset_path:path}')
async def debug_gui_vnc_proxy(asset_path: str, request: Request):
    upstream_url = f'{_novnc_http_origin()}/{asset_path}'
    if request.url.query:
        upstream_url = f'{upstream_url}?{request.url.query}'

    async with aiohttp.ClientSession() as session:
        async with session.get(upstream_url) as upstream:
            body = await upstream.read()
            passthrough_headers = {}
            for name in ('Content-Type', 'Cache-Control', 'ETag', 'Last-Modified'):
                value = upstream.headers.get(name)
                if value:
                    passthrough_headers[name] = value
            return Response(content=body, status_code=upstream.status, headers=passthrough_headers)


@app.websocket('/debug/gui/vnc/websockify')
async def debug_gui_vnc_websocket_proxy(websocket: WebSocket):
    await websocket.accept()
    upstream_url = f'{_novnc_ws_origin()}/websockify'
    if websocket.url.query:
        upstream_url = f'{upstream_url}?{websocket.url.query}'

    session = aiohttp.ClientSession()
    try:
        async with session.ws_connect(upstream_url) as upstream:

            async def _client_to_upstream() -> None:
                while True:
                    message = await websocket.receive()
                    message_type = message.get('type')
                    if message_type == 'websocket.disconnect':
                        break
                    if message.get('bytes') is not None:
                        await upstream.send_bytes(message['bytes'])
                    elif message.get('text') is not None:
                        await upstream.send_str(message['text'])

            async def _upstream_to_client() -> None:
                async for message in upstream:
                    if message.type == aiohttp.WSMsgType.BINARY:
                        await websocket.send_bytes(message.data)
                    elif message.type == aiohttp.WSMsgType.TEXT:
                        await websocket.send_text(message.data)
                    elif message.type in {aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSING, aiohttp.WSMsgType.CLOSED}:
                        break

            await asyncio.gather(_client_to_upstream(), _upstream_to_client())
    finally:
        await session.close()
        with suppress(Exception):
            await websocket.close()


def _aiomonitor_debug_base_path(request: Request, *, loop_name: str) -> str:
    header = request.headers.get('X-Nexus-Debug-Base-Path')
    if header:
        return header.rstrip('/')
    return f'/debug/aiomonitor/{loop_name}'


def _aiomonitor_upstream_url(*, loop_name: str, asset_path: str, query: str = '', websocket: bool = False) -> str:
    port = aiomonitor_webui_port(loop_name, base_port=_RUNTIME_AIOMONITOR_WEBUI_BASE_PORT)
    origin = f'ws://127.0.0.1:{port}' if websocket else f'http://127.0.0.1:{port}'
    path = f'/{asset_path.lstrip("/")}' if asset_path else '/'
    url = f'{origin}{path}'
    return f'{url}?{query}' if query else url


@app.get('/debug/aiomonitor/{loop_name}')
async def debug_aiomonitor_root(loop_name: str, request: Request):
    try:
        aiomonitor_webui_port(loop_name, base_port=_RUNTIME_AIOMONITOR_WEBUI_BASE_PORT)
    except ValueError as exc:
        return JSONResponse({'error': str(exc)}, status_code=404)
    return RedirectResponse(f'{request.url.path.rstrip("/")}/')


@app.api_route(
    '/debug/aiomonitor/{loop_name}/{asset_path:path}',
    methods=['GET', 'HEAD', 'POST', 'PUT', 'PATCH', 'DELETE', 'OPTIONS'],
)
async def debug_aiomonitor_http_proxy(loop_name: str, asset_path: str, request: Request):
    try:
        upstream_url = _aiomonitor_upstream_url(loop_name=loop_name, asset_path=asset_path, query=request.url.query)
    except ValueError as exc:
        return JSONResponse({'error': str(exc)}, status_code=404)

    base_path = _aiomonitor_debug_base_path(request, loop_name=loop_name)
    host = f'127.0.0.1:{aiomonitor_webui_port(loop_name, base_port=_RUNTIME_AIOMONITOR_WEBUI_BASE_PORT)}'
    headers = _proxy_request_headers(request, host=host)
    body = await request.body()
    async with aiohttp.ClientSession(auto_decompress=False) as session:
        try:
            async with session.request(request.method, upstream_url, data=body or None, headers=headers) as upstream:
                response_headers = _proxy_response_headers(upstream.headers)
                location = response_headers.get('Location')
                if location and location.startswith('/'):
                    response_headers['Location'] = f'{base_path}{location}'
                content = await upstream.read()
                content_type = upstream.headers.get('Content-Type', '')
                if 'text/html' in content_type.lower():
                    content = rewrite_aiomonitor_html(
                        content.decode(getattr(upstream, 'charset', None) or 'utf-8', errors='replace'),
                        base_path=base_path,
                    ).encode('utf-8')
                    response_headers.pop('Content-Type', None)
                    response_headers.pop('content-type', None)
                    return Response(content=content, status_code=upstream.status, media_type='text/html')
                return Response(content=content, status_code=upstream.status, headers=response_headers)
        except aiohttp.ClientError as exc:
            return JSONResponse(
                {
                    'error': f'Cannot connect to runtime aiomonitor loop {loop_name}',
                    'detail': f'{type(exc).__name__}: {exc}',
                },
                status_code=502,
            )


@app.websocket('/debug/aiomonitor/{loop_name}/{asset_path:path}')
async def debug_aiomonitor_websocket_proxy(loop_name: str, asset_path: str, websocket: WebSocket):
    try:
        upstream_url = _aiomonitor_upstream_url(
            loop_name=loop_name,
            asset_path=asset_path,
            query=websocket.url.query,
            websocket=True,
        )
    except ValueError:
        await websocket.close(code=1008)
        return

    await websocket.accept()
    host = f'127.0.0.1:{aiomonitor_webui_port(loop_name, base_port=_RUNTIME_AIOMONITOR_WEBUI_BASE_PORT)}'
    session = aiohttp.ClientSession()
    try:
        try:
            upstream_context = session.ws_connect(upstream_url, headers=_proxy_websocket_headers(websocket, host=host))
            upstream = await upstream_context.__aenter__()
        except aiohttp.ClientError:
            await websocket.close(code=1011)
            return
        try:

            async def _client_to_upstream() -> None:
                while True:
                    message = await websocket.receive()
                    message_type = message.get('type')
                    if message_type == 'websocket.disconnect':
                        break
                    if message.get('bytes') is not None:
                        await upstream.send_bytes(message['bytes'])
                    elif message.get('text') is not None:
                        await upstream.send_str(message['text'])

            async def _upstream_to_client() -> None:
                async for message in upstream:
                    if message.type == aiohttp.WSMsgType.BINARY:
                        await websocket.send_bytes(message.data)
                    elif message.type == aiohttp.WSMsgType.TEXT:
                        await websocket.send_text(message.data)
                    elif message.type in {aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSING, aiohttp.WSMsgType.CLOSED}:
                        break

            await _run_websocket_bridge(_client_to_upstream(), _upstream_to_client())
        finally:
            await upstream_context.__aexit__(None, None, None)
    finally:
        await session.close()
        with suppress(Exception):
            await websocket.close()


def _debug_port_origin(port: int) -> str:
    return f'http://127.0.0.1:{port}'


def _debug_port_ws_origin(port: int) -> str:
    return f'ws://127.0.0.1:{port}'


def _debug_port_upstream_url(*, port: int, asset_path: str, query: str = '', websocket: bool = False) -> str:
    """Build a runtime-local upstream URL, stripping the public debug prefix."""
    origin = _debug_port_ws_origin(port) if websocket else _debug_port_origin(port)
    path = f'/{asset_path.lstrip("/")}' if asset_path else '/'
    url = f'{origin}{path}'
    return f'{url}?{query}' if query else url


def _proxy_request_headers(request: Request, *, host: str) -> dict[str, str]:
    skip = {
        'host',
        'transfer-encoding',
        'connection',
        'upgrade',
        'sec-websocket-key',
        'sec-websocket-protocol',
        'x-nexus-debug-base-path',
    }
    headers = {key: value for key, value in request.headers.items() if key.lower() not in skip}
    headers['Host'] = host
    return headers


def _proxy_response_headers(headers: Any) -> dict[str, str]:
    skip = {'connection', 'content-length', 'keep-alive', 'te', 'trailer', 'transfer-encoding', 'upgrade'}
    return {key: value for key, value in headers.items() if key.lower() not in skip}


def _proxy_websocket_headers(websocket: WebSocket, *, host: str) -> dict[str, str]:
    skip = {'host', 'origin', 'connection', 'upgrade', 'sec-websocket-key', 'sec-websocket-protocol'}
    headers = {key: value for key, value in websocket.headers.items() if key.lower() not in skip}
    headers['Host'] = host
    headers['Origin'] = f'http://{host}'
    return headers


def _is_websocket_close_error(exc: BaseException) -> bool:
    if isinstance(exc, (ConnectionError, aiohttp.ClientError)):
        return True
    return isinstance(exc, RuntimeError) and any(
        text in str(exc).lower() for text in ('closing transport', 'closed', 'disconnect')
    )


async def _run_websocket_bridge(*coroutines: Any) -> None:
    tasks = [asyncio.create_task(coroutine) for coroutine in coroutines]
    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    error: BaseException | None = None
    for task in done:
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            if _is_websocket_close_error(exc):
                pass
            else:
                error = exc
    for task in pending:
        task.cancel()
    for task in pending:
        with suppress(asyncio.CancelledError):
            try:
                await task
            except Exception as exc:
                if not _is_websocket_close_error(exc):
                    error = error or exc
    if error is not None:
        raise error


@app.get('/debug/ports/{port}')
@app.get('/debug/runs/{run_id}/ports/{port}')
async def debug_port_proxy_root(port: int, request: Request, run_id: str | None = None):
    _ = run_id
    validate_debug_port(port)
    return RedirectResponse(f'{request.url.path}/')


@app.api_route(
    '/debug/ports/{port}/{asset_path:path}',
    methods=['GET', 'HEAD', 'POST', 'PUT', 'PATCH', 'DELETE', 'OPTIONS'],
)
@app.api_route(
    '/debug/runs/{run_id}/ports/{port}/{asset_path:path}',
    methods=['GET', 'HEAD', 'POST', 'PUT', 'PATCH', 'DELETE', 'OPTIONS'],
)
async def debug_port_http_proxy(port: int, asset_path: str, request: Request, run_id: str | None = None):
    """Reverse proxy a runtime-local HTTP service through the runtime server.

    Public debug prefixes are stripped before forwarding so tools like
    code-server can continue serving from their normal root path.
    """
    _ = (asset_path, run_id)
    try:
        validate_debug_port(port)
    except ValueError as exc:
        return Response(str(exc), status_code=400)

    upstream_url = _debug_port_upstream_url(port=port, asset_path=asset_path, query=request.url.query)

    host = f'127.0.0.1:{port}'
    headers = _proxy_request_headers(request, host=host)
    body = await request.body()
    async with aiohttp.ClientSession(auto_decompress=False) as session:
        try:
            async with session.request(request.method, upstream_url, data=body or None, headers=headers) as upstream:
                response_headers = _proxy_response_headers(upstream.headers)
                return Response(content=await upstream.read(), status_code=upstream.status, headers=response_headers)
        except aiohttp.ClientError as exc:
            return JSONResponse(
                {
                    'error': f'Cannot connect to runtime-local debug port {port}',
                    'detail': f'{type(exc).__name__}: {exc}',
                },
                status_code=502,
            )


@app.websocket('/debug/ports/{port}/{asset_path:path}')
@app.websocket('/debug/runs/{run_id}/ports/{port}/{asset_path:path}')
async def debug_port_websocket_proxy(
    port: int,
    asset_path: str,
    websocket: WebSocket,
    run_id: str | None = None,
):
    _ = (asset_path, run_id)
    try:
        validate_debug_port(port)
    except ValueError:
        await websocket.close(code=1008)
        return

    await websocket.accept()
    upstream_url = _debug_port_upstream_url(
        port=port,
        asset_path=asset_path,
        query=websocket.url.query,
        websocket=True,
    )

    session = aiohttp.ClientSession()
    try:
        try:
            upstream_context = session.ws_connect(
                upstream_url,
                headers=_proxy_websocket_headers(websocket, host=f'127.0.0.1:{port}'),
            )
            upstream = await upstream_context.__aenter__()
        except aiohttp.ClientError:
            await websocket.close(code=1011)
            return
        try:

            async def _client_to_upstream() -> None:
                while True:
                    message = await websocket.receive()
                    message_type = message.get('type')
                    if message_type == 'websocket.disconnect':
                        break
                    if message.get('bytes') is not None:
                        await upstream.send_bytes(message['bytes'])
                    elif message.get('text') is not None:
                        await upstream.send_str(message['text'])

            async def _upstream_to_client() -> None:
                async for message in upstream:
                    if message.type == aiohttp.WSMsgType.BINARY:
                        await websocket.send_bytes(message.data)
                    elif message.type == aiohttp.WSMsgType.TEXT:
                        await websocket.send_text(message.data)
                    elif message.type in {aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSING, aiohttp.WSMsgType.CLOSED}:
                        break

            await _run_websocket_bridge(_client_to_upstream(), _upstream_to_client())
        finally:
            await upstream_context.__aexit__(None, None, None)
    finally:
        await session.close()
        with suppress(Exception):
            await websocket.close()


class ErrorResponse(BaseModel):
    error: str
    detail: str | None = None
    traceback: str | None = None


@app.exception_handler(Exception)
async def global_exception_handler(_request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error=str(exc), detail=exc.__class__.__name__, traceback=traceback.format_exc()
        ).model_dump(),
    )


# ============== HTTP RPC Endpoints ==============


class CallToolRequest(BaseModel):
    """Request to execute a tool with context (server-side model for HTTP endpoints).

    Uses import_path to load the tool function dynamically,
    ensuring the runtime server executes the same code as the agent.
    """

    import_path: str  # e.g., "nexus.extensions.tools.sandbox::bash_tool"
    params: dict[str, Any]
    context: ToolContext
    trace_context: RemoteTraceContext | None = None


@api_v1_router.post('/call_tool', response_model=ToolResponse)
async def call_tool(request: CallToolRequest):
    """Execute a tool by importing it from the given path.

    The tool function is loaded via import_path, validated, and executed.
    Context updates in the response will be sent back to the caller.
    """
    logger.info(f'call_tool: received request, import_path={request.import_path}')
    t_start = time.monotonic()
    tool_func = import_from_path(request.import_path)
    meta = get_tool_meta(tool_func)

    logger.info(f"call_tool: executing tool '{meta.name}', params={request.params}")

    async def execute():
        parsed_params = meta.input_model.model_validate(request.params)
        return await tool_func(parsed_params, request.context)

    try:
        # Use remote trace context if provided, so spans are linked to client's trace
        if request.trace_context:
            with with_remote_trace_context(
                request.trace_context.trace_id,
                request.trace_context.parent_span_id,
                run_id=request.trace_context.run_id,
                run_span_id=request.trace_context.run_span_id,
                run_kind=request.trace_context.run_kind,
                otel_carrier=request.trace_context.otel_carrier,
            ):
                result = await execute()
        else:
            result = await execute()
        elapsed = time.monotonic() - t_start
        logger.info(f"call_tool: '{meta.name}' executed successfully, elapsed={elapsed:.3f}s")
        return result
    except Exception as exc:
        elapsed = time.monotonic() - t_start
        logger.error(f"call_tool: '{meta.name}' failed with {exc.__class__.__name__}: {exc}, elapsed={elapsed:.3f}s")
        logger.error(f'call_tool: traceback:\n{traceback.format_exc()}')
        raise


@api_v1_router.post('/call_executor', response_model=Any)
async def call_executor(request: CallExecutorRequest):
    logger.info(f'call_executor: received request, executor_name={request.executor_name}')
    t_start = time.monotonic()

    executor = executor_registry.get(request.executor_name)
    if executor is None:
        available = list(executor_registry.get_all().keys())
        logger.error(f"call_executor: executor '{request.executor_name}' not found, available: {available}")
        raise ValueError(f"Executor '{request.executor_name}' not found. Available: {available}")

    sig = inspect.signature(executor)
    input_type = list(sig.parameters.values())[0].annotation
    parsed_arg = input_type.model_validate(request.arg)

    logger.info(f"call_executor: executing '{request.executor_name}', arg={request.arg}")

    async def execute():
        return await executor(parsed_arg)

    try:
        # Use remote trace context if provided, so spans are linked to client's trace
        if request.trace_context:
            with with_remote_trace_context(
                request.trace_context.trace_id,
                request.trace_context.parent_span_id,
                run_id=request.trace_context.run_id,
                run_span_id=request.trace_context.run_span_id,
                run_kind=request.trace_context.run_kind,
                otel_carrier=request.trace_context.otel_carrier,
            ):
                result = await execute()
        else:
            result = await execute()
        elapsed = time.monotonic() - t_start
        logger.info(f"call_executor: '{request.executor_name}' executed successfully, elapsed={elapsed:.3f}s")
        return result
    except Exception as exc:
        elapsed = time.monotonic() - t_start
        logger.error(
            f"call_executor: '{request.executor_name}' failed with {exc.__class__.__name__}: {exc}, elapsed={elapsed:.3f}s"
        )
        logger.error(f'call_executor: traceback:\n{traceback.format_exc()}')
        raise


def _parse_prefer_wait(request: Request, *, default: float | None = None) -> float:
    """Extract wait seconds from ``Prefer: wait=N`` header."""
    prefer = request.headers.get('prefer', '')
    for part in prefer.split(','):
        part = part.strip()
        if part.startswith('wait='):
            try:
                return float(part[5:])
            except ValueError:
                pass
    return default if default is not None else _DEFAULT_WAIT_SECONDS


@api_v1_router.post('/invoke')
async def http_invoke(body: InvokeRequest, raw_request: Request):
    """Unified invoke endpoint with long-poll support.

    Behaviour depends on the ``Prefer: wait=N`` header:

    - **wait > 0** (default 30s): hold the connection up to *N* seconds.
      If the call completes within the window, return ``200`` with
      :class:`InvokeResponse`.  Otherwise return ``202`` with
      :class:`InvokeJobInfo` (status=running) and continue in background.
    - **wait = 0**: always async — return ``202`` immediately.
    """
    rid = body.request_id or uuid.uuid4().hex
    body = body.model_copy(update={'request_id': rid})
    wait = _parse_prefer_wait(raw_request)

    if wait <= 0:
        # Fully async — fire and forget, return 202 immediately
        _jobs[rid] = InvokeJobInfo(request_id=rid, status=JobStatus.PENDING)
        asyncio.create_task(_run_invoke_job(body))
        return JSONResponse(
            status_code=202,
            content=InvokeJobInfo(request_id=rid, status=JobStatus.PENDING).model_dump(),
        )

    # Long-poll: race execution against the wait timeout
    task = asyncio.create_task(_execute_invoke(body))
    done, _ = await asyncio.wait({task}, timeout=wait)

    if done:
        # Completed within the wait window — synchronous response
        response = task.result()
        return JSONResponse(status_code=200, content=response.model_dump())

    # Timed out — promote to background job, return 202
    _jobs[rid] = InvokeJobInfo(request_id=rid, status=JobStatus.RUNNING)

    async def _finish() -> None:
        try:
            response = await task
            status = JobStatus.DONE if response.success else JobStatus.ERROR
            _jobs[rid] = InvokeJobInfo(request_id=rid, status=status, response=response)
            with suppress(Exception):
                await sio.emit(f'invoke:done:{rid}', response.model_dump())
        except Exception as exc:
            response = invoke_response_from_exception(exc, request_id=rid)
            _jobs[rid] = InvokeJobInfo(request_id=rid, status=JobStatus.ERROR, response=response)
            with suppress(Exception):
                await sio.emit(f'invoke:done:{rid}', response.model_dump())

    asyncio.create_task(_finish())
    return JSONResponse(
        status_code=202,
        content=InvokeJobInfo(request_id=rid, status=JobStatus.RUNNING).model_dump(),
    )


@api_v1_router.get('/invoke/{request_id}')
async def http_invoke_poll(request_id: str):
    """Poll for the result of an async invoke job.

    Returns :class:`InvokeJobInfo` with the current status.
    When ``status`` is ``done`` or ``error``, ``response`` carries
    the final :class:`InvokeResponse`.
    """
    job = _jobs.get(request_id)
    if job is None:
        return JSONResponse(status_code=404, content={'error': f'Job {request_id} not found'})
    return job.model_dump()


# ============== Blob Sideband Endpoints ==============


@api_v1_router.post('/blob/{blob_id}')
async def upload_blob(blob_id: str, raw_request: Request):
    """Upload a binary blob for use as ``payload_ref`` in InvokeRequest."""
    data = await raw_request.body()
    _blobs[blob_id] = data
    return {'blob_id': blob_id, 'size': len(data)}


@api_v1_router.get('/blob/{blob_id}')
async def download_blob(blob_id: str):
    """Download a blob (e.g. a large ``result_ref``)."""
    data = _blobs.pop(blob_id, None)
    if data is None:
        return JSONResponse(status_code=404, content={'error': f'Blob {blob_id} not found'})
    from starlette.responses import Response

    return Response(content=data, media_type='application/octet-stream')


for name, entry in executor_registry.get_all().items():
    api_v1_router.add_api_route(
        path=f'/call_executor/{name}',
        endpoint=entry,
        methods=['POST'],
    )


# ============== Venv Management Endpoints ==============

from nexus.runtime.server.venv.manager import VenvManager  # noqa: E402
from nexus.runtime.server.venv.models import VenvSpec, VenvStatus  # noqa: E402
from nexus.runtime.server.worker.launcher import WorkerLauncher  # noqa: E402


_venv_manager = VenvManager()
_worker_launcher = WorkerLauncher()
_server_base_url: str = 'http://localhost:8000'

# Track worker SIO connections: worker_id -> sid
_worker_sids: dict[str, str] = {}
# Track capabilities by Socket.IO session. Missing means a legacy worker.
_worker_capabilities: dict[str, Any] = {}
# Signaled by worker_connect when a worker registers its SIO session
_worker_ready_events: dict[str, asyncio.Event] = {}


def configure(*, host: str = 'localhost', port: int = 8000) -> None:
    """Configure the server before starting uvicorn.

    Called from ``start.sh`` (or equivalent) so the server knows its own
    address and can pass it to worker processes.
    """
    global _server_base_url
    _server_base_url = f'http://{host}:{port}'
    logger.info('Server base_url for workers: %s', _server_base_url)


_VENV_DEFAULT_WAIT_SECONDS: float = 60.0


async def _create_venv_full(vid: str, spec: VenvSpec) -> VenvStatus:
    """Create venv + launch worker + wait for SIO connect.

    Shared by the HTTP and SIO create-venv handlers.
    """
    status = await _venv_manager.create(vid, spec, defer_ready=True)

    if status.state == 'creating':
        # Installation alone is not externally ready: invoke requires a
        # registered worker SID. Keep status creating until worker_connect.
        venv_dir = _venv_manager.root / vid
        worker_id = f'worker-{vid}'
        ready_ev = asyncio.Event()
        _worker_ready_events[worker_id] = ready_ev
        try:
            await _worker_launcher.start_worker(vid, venv_dir, base_url=_server_base_url, env_vars=spec.env_vars)
            await asyncio.wait_for(ready_ev.wait(), timeout=30.0)
            if worker_id not in _worker_sids:
                raise RuntimeError('Worker disconnected before becoming ready')
            status = status.model_copy(update={'state': 'ready'})
        except asyncio.TimeoutError:
            # Report what the worker printed before it went silent. Without this the
            # caller only learns "did not connect" and cannot tell an interpreter crash
            # from a slow start -- the exact reason the one worker-init failure in
            # r20final could not be attributed. Bounded and non-raising by contract; see
            # `read_startup_output`. Not left to the `except Exception` below, which would
            # replace this timeout error with the diagnostics error and lose the failure.
            detail = await _worker_launcher.read_startup_output(worker_id)
            error = 'Worker did not connect within 30s'
            if detail:
                error = f'{error}; {detail}'
            logger.warning('Worker venv %s: %s', vid, error)
            status = status.model_copy(update={'state': 'failed', 'error': error})
        except Exception as exc:
            status = status.model_copy(update={'state': 'failed', 'error': str(exc)})
        finally:
            _worker_ready_events.pop(worker_id, None)

    _venv_manager._venvs[vid] = status
    return status


@api_v1_router.post('/venv')
async def http_create_venv(spec: VenvSpec, raw_request: Request, venv_id: str | None = None):
    """Create a worker venv with long-poll support.

    Behaviour depends on the ``Prefer: wait=N`` header (default 60s):

    - **wait > 0**: hold the connection up to *N* seconds.
      If creation completes within the window, return ``200`` with
      :class:`VenvStatus`.  Otherwise return ``202`` with
      ``VenvStatus(state='creating')`` — poll ``GET /venv/{id}`` or
      listen for ``venv:done:{id}`` via SIO.
    - **wait = 0**: always async — return ``202`` immediately.
    """
    vid = venv_id or uuid.uuid4().hex[:12]

    # Register as 'creating' immediately so GET /venv/{vid} works
    status = VenvStatus(
        venv_id=vid,
        state='creating',
        python_version=spec.python_version,
        artifacts=[a.name for a in spec.artifacts],
        extra_requirements=spec.extra_requirements,
        created_at=datetime.now(timezone.utc),
    )
    _venv_manager._venvs[vid] = status

    wait = _parse_prefer_wait(raw_request, default=_VENV_DEFAULT_WAIT_SECONDS)

    if wait <= 0:
        asyncio.create_task(_create_venv_bg(vid, spec))
        return JSONResponse(status_code=202, content=status.model_dump(mode='json'))

    # Long-poll: race creation against wait timeout
    task = asyncio.create_task(_create_venv_full(vid, spec))
    done, _ = await asyncio.wait({task}, timeout=wait)

    if done:
        return JSONResponse(status_code=200, content=task.result().model_dump(mode='json'))

    # Timed out — continue in background, return 202
    async def _finish() -> None:
        try:
            final = await task
        except Exception as exc:
            final = VenvStatus(
                venv_id=vid,
                state='failed',
                python_version=spec.python_version,
                created_at=datetime.now(timezone.utc),
                error=str(exc),
            )
            _venv_manager._venvs[vid] = final
        with suppress(Exception):
            await sio.emit(f'venv:done:{vid}', final.model_dump(mode='json'))

    asyncio.create_task(_finish())
    return JSONResponse(status_code=202, content=status.model_dump(mode='json'))


async def _create_venv_bg(vid: str, spec: VenvSpec) -> None:
    """Background venv creation — updates status and emits SIO event."""
    try:
        final = await _create_venv_full(vid, spec)
    except Exception as exc:
        final = VenvStatus(
            venv_id=vid,
            state='failed',
            python_version=spec.python_version,
            created_at=datetime.now(timezone.utc),
            error=str(exc),
        )
        _venv_manager._venvs[vid] = final
    with suppress(Exception):
        await sio.emit(f'venv:done:{vid}', final.model_dump(mode='json'))


@api_v1_router.get('/venv/{venv_id}', response_model=VenvStatus)
async def get_venv(venv_id: str):
    status = _venv_manager.get(venv_id)
    if status is None:
        return JSONResponse(status_code=404, content={'error': f'Venv {venv_id} not found'})
    return status


@api_v1_router.delete('/venv/{venv_id}')
async def delete_venv(venv_id: str):
    await _worker_launcher.stop_worker(f'worker-{venv_id}')
    await _venv_manager.destroy(venv_id)
    return {'status': 'destroyed', 'venv_id': venv_id}


@api_v1_router.get('/venvs', response_model=list[VenvStatus])
async def list_venvs():
    return _venv_manager.list_all()


# ============== /worker SIO Namespace ==============

worker_ns = '/worker'


@sio.on('invoke:result', namespace=worker_ns)
async def _on_worker_invoke_result(sid: str, data: dict) -> None:
    """Receive the invoke result emitted by the worker (fire-and-forget style)."""
    rid = data.get('request_id', '')
    future = _worker_invoke_futures.get(rid)
    if future is not None and not future.done():
        future.set_result(data)
    else:
        logger.warning('invoke:result for unknown or already-resolved request_id=%s', rid)


@sio.on('connect', namespace=worker_ns)
async def worker_connect(sid: str, environ: dict, auth: dict | None = None):
    """Handle worker process connecting to /worker namespace."""
    worker_id = (auth or {}).get('worker_id', sid)
    venv_id = (auth or {}).get('venv_id', 'unknown')
    _worker_sids[worker_id] = sid
    _worker_capabilities[sid] = (auth or {}).get('capabilities')
    logger.info('Worker %s (venv=%s) connected, sid=%s', worker_id, venv_id, sid)
    # Signal waiters (create_venv blocks until this fires)
    ev = _worker_ready_events.pop(worker_id, None)
    if ev is not None:
        ev.set()


@sio.on('disconnect', namespace=worker_ns)
async def worker_disconnect(sid: str):
    """Handle worker process disconnecting."""
    _worker_capabilities.pop(sid, None)
    # Clean up HTTP routes registered by this worker
    for prefix, (_url, route_sid) in list(_worker_http_routes.items()):
        if route_sid == sid:
            del _worker_http_routes[prefix]
            logger.info('Worker HTTP route auto-cleaned: %s', prefix)
    # Remove from SID mapping
    for wid, wsid in list(_worker_sids.items()):
        if wsid == sid:
            del _worker_sids[wid]
            logger.info('Worker %s disconnected', wid)
            break


@sio.on('trace:flush')
async def trace_flush(_sid: str, _data: dict | None = None) -> dict[str, Any]:
    """Establish an ordered barrier across worker, host, and client span transports."""
    try:
        for worker_sid in set(_worker_sids.values()):
            if not supports_trace_flush(_worker_capabilities.get(worker_sid)):
                logger.info('Skipping trace flush for legacy worker sid=%s', worker_sid)
                continue
            response = await sio.call(
                'trace:flush',
                {},
                to=worker_sid,
                namespace=worker_ns,
                timeout=60,
            )
            if not isinstance(response, dict) or not response.get('ok'):
                error = response.get('error') if isinstance(response, dict) else None
                raise RuntimeError(error or f'Worker trace barrier failed for {worker_sid}')
        span_batch_id = await span_broadcaster.barrier()
        await async_force_flush_best_effort()
        return {'ok': True, 'span_batch_id': span_batch_id}
    except Exception as exc:
        logger.exception('Trace barrier failed')
        return {'ok': False, 'error': f'{type(exc).__name__}: {exc}'}


@sio.on('*', namespace=worker_ns)
async def worker_relay(event: str, sid: str, *args):
    """Auto-forward all worker events to Host (default namespace).

    Span events are routed through the broadcaster so they get batched
    into the ``{'items': [...]}`` format that clients expect.
    """
    if event == 'span' and args:
        payload = args[0]
        items = payload.get('items') if isinstance(payload, dict) else None
        if not isinstance(items, list):
            items = [payload]
        for item in items:
            if not isinstance(item, dict):
                raise TypeError(f'Invalid worker span event from {sid}')
            if not span_broadcaster.enqueue_event(item):
                raise RuntimeError(f'Failed to enqueue worker span event from {sid}')
        await span_broadcaster.flush()
        return {'ok': True}
    await sio.emit(event, *args)
    return {'ok': True}


# ============== Worker HTTP Route Forwarding ==============

# Route table: path_prefix -> (worker_base_url, worker_sid)
_worker_http_routes: dict[str, tuple[str, str]] = {}


@sio.on('worker:register_http_route', namespace=worker_ns)
async def _on_register_http_route(sid: str, data: dict) -> dict:
    """Worker registers an HTTP route prefix for reverse-proxy forwarding."""
    prefix = data['path_prefix']
    worker_url = data['worker_url']
    _worker_http_routes[prefix] = (worker_url, sid)
    logger.info('Worker HTTP route registered: %s → %s (sid=%s)', prefix, worker_url, sid)
    return {'success': True}


@sio.on('worker:unregister_http_route', namespace=worker_ns)
async def _on_unregister_http_route(sid: str, data: dict) -> dict:
    """Worker unregisters an HTTP route prefix."""
    prefix = data['path_prefix']
    _worker_http_routes.pop(prefix, None)
    logger.info('Worker HTTP route unregistered: %s', prefix)
    return {'success': True}


app.include_router(api_v1_router)


# ============== Worker Route Proxy ==============


class WorkerRouteProxy:
    """ASGI middleware that reverse-proxies matching HTTP requests to Worker HTTP servers.

    Workers register route prefixes via the ``worker:register_http_route``
    Socket.IO event.  This middleware intercepts matching HTTP requests and
    forwards them to the Worker's local HTTP server using ``aiohttp``,
    streaming the response back (including SSE).

    Non-matching requests pass through to the wrapped ASGI app unchanged.
    """

    def __init__(self, inner_app: Any) -> None:
        self.app = inner_app

    async def __call__(
        self,
        scope: dict[str, Any],
        receive: Any,
        send: Any,
    ) -> None:
        if scope['type'] == 'http':
            path: str = scope.get('path', '')
            match = self._match_route(path)
            if match is not None:
                await self._reverse_proxy(match, scope, receive, send)
                return
        await self.app(scope, receive, send)

    @staticmethod
    def _match_route(path: str) -> str | None:
        """Return the worker base URL if *path* matches a registered prefix."""
        for prefix, (worker_url, _sid) in _worker_http_routes.items():
            if path == prefix or path.startswith(prefix + '/'):
                return worker_url
        return None

    @staticmethod
    async def _reverse_proxy(
        worker_url: str,
        scope: dict[str, Any],
        receive: Any,
        send: Any,
    ) -> None:
        """Forward an HTTP request to *worker_url* and stream the response."""
        import aiohttp as _aiohttp

        # Read request body
        body = b''
        while True:
            msg = await receive()
            body += msg.get('body', b'')
            if not msg.get('more_body', False):
                break

        # Build target URL
        path: str = scope.get('path', '')
        query: bytes = scope.get('query_string', b'')
        url = f'{worker_url}{path}'
        if query:
            url += '?' + query.decode()

        # Forward headers (skip hop-by-hop)
        _skip = frozenset(('host', 'transfer-encoding'))
        headers: dict[str, str] = {}
        for raw_key, raw_val in scope.get('headers', []):
            name = raw_key.decode().lower()
            if name not in _skip:
                headers[raw_key.decode()] = raw_val.decode()

        try:
            async with _aiohttp.ClientSession() as session:
                async with session.request(
                    scope['method'],
                    url,
                    data=body or None,
                    headers=headers,
                ) as resp:
                    resp_headers = [
                        (k.encode(), v.encode())
                        for k, v in resp.headers.items()
                        if k.lower() not in ('transfer-encoding',)
                    ]
                    await send({
                        'type': 'http.response.start',
                        'status': resp.status,
                        'headers': resp_headers,
                    })
                    async for chunk in resp.content.iter_any():
                        await send({
                            'type': 'http.response.body',
                            'body': chunk,
                            'more_body': True,
                        })
                    await send({
                        'type': 'http.response.body',
                        'body': b'',
                        'more_body': False,
                    })
        except Exception:
            logger.exception('WorkerRouteProxy: failed to forward %s %s', scope['method'], url)
            from starlette.responses import Response

            error_resp = Response(status_code=502, content='Worker proxy error')
            await error_resp(scope, receive, send)


# ============== Socket.IO ASGI App ==============

# Wrap FastAPI app with Worker route proxy, then with Socket.IO.
# Order: SIO handles /socket.io/*, WorkerRouteProxy intercepts worker routes,
# everything else goes to FastAPI.
asgi_app = socketio.ASGIApp(sio, other_asgi_app=WorkerRouteProxy(app))
