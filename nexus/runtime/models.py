"""Stable protocol models for the unified invoke API.

These models define the wire format between Host and Runtime.
Function dispatch uses import_path strings (``module.path::func_name``)
or registry names (``executor_name``).
Return values, exceptions, **and arguments** all use pickle (stdlib)
for faithful object preservation.

Transport strategy:
- Small payloads travel inline (base64 pickle in JSON).
- Large payloads use HTTP blob sideband (``payload_ref`` / ``result_ref``).
- Long-running calls use ``Prefer: wait=N`` long-poll with async fallback.
"""

from __future__ import annotations

import base64
import enum
import pickle

from collections.abc import AsyncIterator, Callable
from typing import Any

from pydantic import BaseModel, Field

from nexus.trace import RemoteTraceContext


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Threshold above which payload / result is offloaded to blob store
# rather than sent inline (base64) over SIO / JSON.
BLOB_THRESHOLD_BYTES: int = 1 * 1024 * 1024  # 1 MB


# ---------------------------------------------------------------------------
# InvokeRequest / InvokeResponse
# ---------------------------------------------------------------------------


class InvokeRequest(BaseModel):
    """Unified request for remote function invocation.

    ``payload`` carries ``base64(pickle.dumps((args_tuple, kwargs_dict)))``.
    For large payloads, upload the blob via ``POST /api/v1/blob`` and set
    ``payload_ref`` instead (mutually exclusive with ``payload``).

    Use the :meth:`build` classmethod for convenient construction and
    :meth:`unpack_payload` to deserialize on the receiving end.
    """

    request_id: str | None = None
    import_path: str  # 'module.path::func_name' or registry name
    payload: str = ''  # base64(pickle.dumps((args, kwargs)))  — inline, small data
    payload_ref: str | None = None  # blob_id for large payloads (HTTP sideband)
    venv_id: str | None = None  # None = base process execution
    trace_context: RemoteTraceContext | None = None
    request_timeout: float | None = Field(None, gt=0)  # client-side invoke wait timeout in seconds

    # -- helpers ----------------------------------------------------------

    @classmethod
    def build(
        cls,
        target: str | Callable[..., Any],
        *args: Any,
        venv_id: str | None = None,
        request_id: str | None = None,
        trace_context: RemoteTraceContext | None = None,
        request_timeout: float | None = None,
        **kwargs: Any,
    ) -> InvokeRequest:
        """Construct an InvokeRequest with pickled args/kwargs.

        *target* may be an import-path string (``"mod.sub::func"``),
        a bare registry name (``"run_command"``), or a **callable** —
        in which case its import path is derived automatically.
        """
        import_path = _to_import_path(target)
        payload = base64.b64encode(pickle.dumps((args, kwargs))).decode('utf-8')
        return cls(
            import_path=import_path,
            payload=payload,
            venv_id=venv_id,
            request_id=request_id,
            trace_context=trace_context,
            request_timeout=request_timeout,
        )

    def unpack_payload(self) -> tuple[tuple, dict]:
        """Deserialize *payload* into ``(args, kwargs)``."""
        if not self.payload:
            return (), {}
        return pickle.loads(base64.b64decode(self.payload))


class InvokeResponse(BaseModel):
    """Unified response from remote function invocation.

    Uses pickle (base64-encoded) for result/exception to preserve object
    structure.  For large results the server stores the blob and returns
    ``result_ref`` instead of ``result``.
    """

    request_id: str | None = None
    success: bool
    result: str | None = None  # base64(pickle.dumps(return_value)) — inline
    result_ref: str | None = None  # blob_id for large results (HTTP sideband)
    exception: str | None = None  # base64(pickle.dumps(exc)) when success=False
    traceback: str | None = None  # traceback.format_exc() for logging


# ---------------------------------------------------------------------------
# Job status (for async invoke polling)
# ---------------------------------------------------------------------------


class JobStatus(str, enum.Enum):
    """Lifecycle states of an async invoke job."""

    PENDING = 'pending'
    RUNNING = 'running'
    DONE = 'done'
    ERROR = 'error'


class InvokeJobInfo(BaseModel):
    """Returned by ``GET /api/v1/invoke/{request_id}`` while polling.

    When ``status`` is ``done`` or ``error``, the ``response`` field
    carries the final :class:`InvokeResponse`.
    """

    request_id: str
    status: JobStatus
    response: InvokeResponse | None = None  # present when status in {done, error}


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class RemoteInvokeError(Exception):
    """Client-side exception wrapping a remote invocation failure.

    Carries the original exception object (deserialized from pickle) when available.
    """

    def __init__(self, message: str, *, cause: Exception | None = None, remote_traceback: str = '') -> None:
        self.cause = cause
        self.remote_traceback = remote_traceback
        super().__init__(message)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def invoke_response_from_exception(exc: Exception, *, request_id: str | None = None) -> InvokeResponse:
    """Build an InvokeResponse from a caught exception."""
    import traceback as tb_module

    return InvokeResponse(
        request_id=request_id,
        success=False,
        exception=base64.b64encode(pickle.dumps(exc)).decode('utf-8'),
        traceback=tb_module.format_exc(),
    )


def pack_result(result: Any, *, request_id: str | None = None) -> InvokeResponse:
    """Build a success InvokeResponse with pickled result."""
    return InvokeResponse(
        request_id=request_id,
        success=True,
        result=base64.b64encode(pickle.dumps(result)).decode('utf-8'),
    )


def pack_result_for_transport(
    result: Any, *, request_id: str | None = None
) -> tuple[InvokeResponse, str | None, bytes | None]:
    """Pack result, checking whether blob offloading is needed.

    Returns ``(response, blob_id, blob_bytes)``:

    - *blob_bytes* is not ``None`` → caller must store *blob_bytes* under
      *blob_id* (response already has ``result_ref`` set, ``result`` is ``None``).
    - *blob_bytes* is ``None`` → result is inline in ``response.result`` (base64).

    This avoids the base64-encode-then-decode round-trip that happens when
    ``pack_result`` is followed by a separate blob-offload check.
    """
    import uuid

    result_bytes = pickle.dumps(result)
    if len(result_bytes) > BLOB_THRESHOLD_BYTES:
        blob_id = uuid.uuid4().hex
        return (
            InvokeResponse(
                request_id=request_id,
                success=True,
                result_ref=blob_id,
            ),
            blob_id,
            result_bytes,
        )
    return (
        InvokeResponse(
            request_id=request_id,
            success=True,
            result=base64.b64encode(result_bytes).decode('utf-8'),
        ),
        None,
        None,
    )


async def upload_blob(url: str, data: bytes, *, timeout: Any = None) -> None:
    """Upload binary data to a blob endpoint via chunked HTTP POST.

    Uses chunked Transfer-Encoding to avoid aiohttp's Content-Length
    issues with payloads near the 2 GB (2^31) boundary.
    """
    import aiohttp

    chunk_size = 8 * 1024 * 1024  # 8 MB

    async def _iter() -> AsyncIterator[bytes]:
        for i in range(0, len(data), chunk_size):
            yield data[i : i + chunk_size]

    async with aiohttp.ClientSession() as session:
        async with session.post(url, data=_iter(), timeout=timeout) as resp:
            resp.raise_for_status()


def unpack_response(response: InvokeResponse) -> Any:
    """Deserialize an InvokeResponse — return the value or raise the exception."""
    if response.success:
        if response.result is None:
            return None
        return pickle.loads(base64.b64decode(response.result))

    if response.exception:
        try:
            exc = pickle.loads(base64.b64decode(response.exception))
        except Exception as decode_exc:
            raise RemoteInvokeError(
                f'Failed to deserialize remote exception: {decode_exc}',
                remote_traceback=response.traceback or '',
            ) from decode_exc
        if isinstance(exc, Exception):
            raise exc

    raise RemoteInvokeError(
        response.traceback or 'Remote invoke failed',
        remote_traceback=response.traceback or '',
    )


# ---------------------------------------------------------------------------
# Import-path resolution helpers
# ---------------------------------------------------------------------------


def _to_import_path(target: str | Callable[..., Any]) -> str:
    """Convert *target* to an import-path string.

    If *target* is already a string it is returned as-is.
    If it is a callable, :func:`get_import_path` derives the
    ``"module.path::name"`` form from its ``__module__`` and ``__qualname__``.
    """
    if isinstance(target, str):
        return target
    from nexus.core.registry import get_import_path

    return get_import_path(target)


# ---------------------------------------------------------------------------
# Shared resolve + auto-coerce logic (used by server and LocalRuntime)
# ---------------------------------------------------------------------------


def resolve_func(import_path: str) -> Any:
    """Resolve a callable from *import_path*.

    - If the path contains ``::`` it is treated as a full module path
      and loaded via :func:`import_from_path`.
    - Otherwise it is looked up in the executor registry by name.
    """
    if '::' in import_path:
        from nexus.core.registry import import_from_path

        return import_from_path(import_path)

    from nexus.core import executor_registry

    return executor_registry.get(import_path)


def coerce_args(func: Any, args: tuple, kwargs: dict) -> tuple[tuple, dict]:
    """Auto-coerce dict arguments to Pydantic models when the function
    signature expects a :class:`~pydantic.BaseModel` subclass.

    Returns a new ``(args, kwargs)`` tuple with coerced values.
    """
    import inspect

    sig = inspect.signature(func)
    params = [p for p in sig.parameters.values() if p.name != 'self']

    coerced_args = list(args)
    for i, (param, arg) in enumerate(zip(params, coerced_args, strict=False)):
        if isinstance(arg, dict) and _is_base_model(param.annotation):
            coerced_args[i] = param.annotation.model_validate(arg)

    coerced_kwargs = dict(kwargs)
    for key, value in coerced_kwargs.items():
        if key in sig.parameters and isinstance(value, dict) and _is_base_model(sig.parameters[key].annotation):
            coerced_kwargs[key] = sig.parameters[key].annotation.model_validate(value)

    return tuple(coerced_args), coerced_kwargs


def _is_base_model(annotation: Any) -> bool:
    """Return True if *annotation* is a BaseModel subclass (not an instance)."""
    try:
        return isinstance(annotation, type) and issubclass(annotation, BaseModel)
    except TypeError:
        return False


# ---------------------------------------------------------------------------
# Shared invoke execution core (used by Base server and Worker)
# ---------------------------------------------------------------------------


async def execute_invoked_func(
    import_path: str,
    payload: str,
    runtime_control: Any,
    trace_context: RemoteTraceContext | None = None,
) -> Any:
    """Shared invoke execution core used by both Base and Worker processes.

    Resolves the target function, unpacks pickled arguments, auto-coerces
    dict args to Pydantic models, injects RuntimeControl into the context,
    applies distributed trace context, and executes the function.
    Sync functions are offloaded to a thread via ``asyncio.to_thread``.

    Args:
        import_path: Function reference — ``'module::func'`` or bare executor name.
        payload: Base64-encoded pickled ``(args, kwargs)`` tuple.
        runtime_control: :class:`~nexus.runtime.context.RuntimeControl` instance
            to inject into the current async context.
        trace_context: Optional distributed trace context for span linking.

    Returns:
        The raw return value of the invoked function.
    """
    import asyncio
    import inspect

    from nexus.runtime.context import set_runtime_control
    from nexus.trace.tracer import with_remote_trace_context

    func = resolve_func(import_path)
    args, kwargs = pickle.loads(base64.b64decode(payload)) if payload else ((), {})
    args, kwargs = coerce_args(func, args, kwargs)

    set_runtime_control(runtime_control)

    async def _run() -> Any:
        if inspect.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        return await asyncio.to_thread(func, *args, **kwargs)

    if trace_context:
        with with_remote_trace_context(
            trace_context.trace_id,
            trace_context.parent_span_id,
            run_id=trace_context.run_id,
            run_span_id=trace_context.run_span_id,
            run_kind=trace_context.run_kind,
            otel_carrier=trace_context.otel_carrier,
        ):
            return await _run()
    return await _run()
