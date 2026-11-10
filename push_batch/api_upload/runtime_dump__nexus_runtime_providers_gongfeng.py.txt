"""Gongfeng-based RuntimeProvider for remote sandbox execution.

This provider manages remote sandbox environments via the sandbox v3 API.
It handles:
- Starting runtime instances via remote API
- Stopping runtime instances
- Python version matching to ensure serialization compatibility
- Provider-level image configuration with higher priority

Architecture:
This provider calls the Gongfeng unified sandbox service, which intelligently selects
from underlying resource pools including computing platforms and Gongfeng K8s clusters,
providing secure isolated code execution environments and compute resource scheduling.

Important:
- Automatically passes PYTHON_VERSION to ensure client/server compatibility
- Provider-level 'image' config takes precedence over task-level configure() calls
"""

# /// script
# dependencies = []
#
# [project]
# name = "gongfeng"
# authors = [
#     { name = "baynejiang", email = "baynejiang@tencent.com" },
#     { name = "tarschen", email = "tarschen@tencent.com" },
# ]
#
# [tool.nexus]
# runtime_providers = { gongfeng = "nexus.runtime.providers.gongfeng" }
# ///

from __future__ import annotations

import asyncio
import base64
import contextlib
import logging
import os
import random
import re
import sys
import time
import uuid

from typing import Any

import aiohttp

from nexus.runtime.client.base import Runtime
from nexus.runtime.client.remote import HttpRuntime, SocketIoRuntime
from nexus.runtime.providers.base import RuntimeProvider
from nexus.runtime.providers.docker import get_package_version
from nexus.runtime.providers.registry import runtime_provider_registry
from nexus.trace.tracer import get_current_span
from nexus.utils.async_http import http_client
from nexus.utils.common import coerce_bool, get_current_python_version
from nexus.utils.event_loop import create_detached_task
from nexus.utils.otel.auto import is_enabled as is_otel_enabled
from nexus.utils.otel.environment import merge_managed_otel_env
from nexus.utils.otel.redaction import exporter_credential_values, redact_exporter_credentials, redact_sensitive_text
from nexus.utils.paths import GONGFENG_AUTH_KEY_FILE


logger = logging.getLogger(__name__)


# Runtime status constants (string values are fixed by the sandbox v3 API proto; do not change).
RUNTIME_STATUS_QUEUED = 'queued'  # Waiting in the backend queue for resources (not yet started)
# IMPORTANT: 'scheduled' is a frequent source of user confusion. It does NOT mean "waiting to be
# scheduled" — it means the sandbox has ALREADY been scheduled and started, and is up and ready to
# serve requests. The waiting-in-queue state is RUNTIME_STATUS_QUEUED above. User-facing log lines
# deliberately say "started / running / READY" instead of echoing this raw term.
RUNTIME_STATUS_SCHEDULED = 'scheduled'  # Already scheduled + started + running (ready to serve)
RUNTIME_STATUS_STOPPED = 'stopped'  # Stopped
RUNTIME_STATUS_NOT_FOUND = 'not_found'  # Environment not found

_CAP_NAME_RE = re.compile(r'^[A-Z][A-Z0-9_]*$')


# Statuses that watchdog treats as definitive death. Explicit allowlist instead of
# "anything != scheduled" so future intermediate statuses (e.g. queued echoed back
# during a re-schedule, or new lifecycle phases) don't get misclassified as death.
WATCHDOG_DEAD_STATUSES: frozenset[str] = frozenset({RUNTIME_STATUS_STOPPED, RUNTIME_STATUS_NOT_FOUND})

# `detail` field values that indicate the sandbox has died and can no longer service
# requests. These come from the PROXY's error response body (not /runtime/get): when the
# watchdog GETs {base_url}/health and the proxy fails to forward to the dead pod, the proxy
# runs diagnoseProxyError → provider.Get() against K8s and returns a non-2xx body shaped
# {"error", "detail": "<reason>", "traceback", "exit_code"}. Observed reasons:
#   - OOMKilled:  container was killed by the OOM killer
#   - SandboxExit: the sandbox process exited abnormally (RestartPolicy=Never → no restart)
# The accompanying `exit_code` (K8s ContainerStateTerminated.ExitCode: 137=OOM, 143=SIGKILL,
# other non-zero=crash, 0=no exit-code info) is diagnostic only — it is folded into the death
# reason but never changes whether the runtime is marked unhealthy.
UNHEALTHY_DETAILS: frozenset[str] = frozenset({'OOMKilled', 'SandboxExit'})

# nexus runtime server health-check path, exposed inside the sandbox and reachable through
# the proxy at {base_url}/health. Must stay aligned with the backend's NexusHealthPath.
NEXUS_HEALTH_PATH = '/health'

# Per-probe timeout for the watchdog's GET {base_url}/health. INDEPENDENT of request_timeout
# (default 1800s): this bounds how long a single probe hangs when the server is wedged, so a
# stuck sandbox is detected in ~one poll cycle rather than up to half an hour later.
#
# MUST stay strictly larger than the sandbox proxy's Dialer.Timeout + its diagnose round-trip.
# For a dead sandbox the proxy first burns its FULL dial timeout, then runs diagnoseProxyError →
# provider.Get() to attach `detail` + `exit_code`, and only then writes the 502. A probe that
# expires before that point is cancelled client-side: we get a bare asyncio TimeoutError with no
# body, lose the diagnostic payload, and the death degrades from "killed on the next probe" to
# WATCHDOG_HEALTH_FAILURE_THRESHOLD × poll_interval (~16min).
#
# The proxy's dial budget is a CROSS-REPO dependency, not a local choice: sandbox
# apiserver/internal/service/proxy/proxy.go sets `net.Dialer{Timeout: 30 * time.Second}` (raised
# from 10s so that clients receive the 502+detail instead of stalling on kernel SYN retries).
# This value must be revisited whenever that one moves.
#
# 40s = 30s dial + ~10s for the provider.Get() round-trip and network jitter, and stays well
# under the 90s lower bound of the poll interval so probes never overlap. The previous 15s was
# sized against the old 10s dial budget and silently regressed when the proxy went to 30s:
# measured on prod sandbox nexus-d6a842b3078440439122e43dd41bd4c2, all 7 post-OOM probes came
# back as `detail=ClientCanceled` at elapsed=15.3~15.8s — our own deadline firing mid-diagnose —
# while the 502 carrying `OOMKilled`+`exit_code=137` needed elapsed≈30.0s.
WATCHDOG_HEALTH_TIMEOUT = 40.0

# How many CONSECUTIVE inconclusive probe failures (bare timeout / connection refused /
# proxy's own 5xx / non-2xx without a known detail) before declaring the sandbox dead. A
# failure carrying detail ∈ UNHEALTHY_DETAILS is a proxy-confirmed death and kills immediately,
# bypassing this counter. The counter resets on any 2xx.
#
# Sized to outlast the K8s livenessProbe on the gongfeng pod (period=60s, failureThreshold=5),
# which takes ~9min from a server wedge to surface detail+exit_code through the proxy
# (300s initialDelay + 5×60s failures + ~25s kubelet kill → K8s Terminated → provider.Get()
# loop). With watchdog poll_interval=120s + ±25% jitter (lower bound 90s), threshold=8 gives
# a 12min lower bound on counter-based death, leaving ~3min of headroom past probe completion
# so the next watchdog probe walks the fast path with detail=SandboxExit+exit_code=137.
#
# Earlier values were too tight: threshold=3 (6min) killed before probe even finished;
# threshold=5 (7.5min lower bound) raced with the probe's detail-write window — measured 3/3
# wedged sandboxes died via "unreachable" instead of SandboxExit. Threshold=8 verified with
# 3/3 PASS at elapsed 610/701/724s, all walking the fast path with exit_code=137.
WATCHDOG_HEALTH_FAILURE_THRESHOLD = 8

# Default interval between watchdog probes. Sized for large pools (up to ~8k concurrent
# sandboxes); the probe hits the proxy, which only touches K8s when forwarding actually fails,
# so a healthy sandbox costs nothing beyond a cheap proxied 200.
DEFAULT_WATCHDOG_POLL_INTERVAL = 120.0

# Fraction of the poll interval used as uniform ±jitter, to de-synchronize
# watchdogs across concurrent sandboxes.
WATCHDOG_POLL_JITTER = 0.25

# Interval between queue-state polls when acquire() is waiting for a queued sandbox
# to be scheduled. Deliberately kept as a module constant (not a constructor arg):
# this is an implementation detail of how the client implements the wait, not a
# user-facing knob. 15s is the empirical sweet spot — short enough that a user
# waiting on a sandbox doesn't perceive the lag (avg ~7.5s perceived latency),
# long enough that even thousands of concurrent waiters don't put meaningful load
# on the shared /runtime/get endpoint. ±25% jitter is applied identically to the
# watchdog to spread requests across the wall clock.
QUEUE_POLL_INTERVAL = 15.0
QUEUE_POLL_JITTER = 0.25

# How often to emit a human-readable "still queued, waited Ns" progress line while a sandbox
# sits in the backend queue. The per-poll /runtime/get calls themselves are logged at DEBUG
# (silent by default) to avoid spam; this INFO heartbeat is the ONLY queue-wait output a user
# sees at the default level, so they can tell the sandbox is queued-and-waiting (normal under
# load) rather than stuck. 60s keeps it reassuring without becoming noise.
QUEUE_PROGRESS_LOG_INTERVAL = 60.0

# `remark` values returned on /runtime/get when the queued sandbox has been
# given up by the backend. NOTE: must be matched with str.startswith, not equality —
# the proto documents the failure-during-stop variant as "{reason}(stop_failed:{error})",
# e.g. "queue_timeout(stop_failed:connection timeout)". A strict == comparison would
# silently miss these cases.
QUEUE_FAILURE_REMARK_PREFIXES: tuple[str, ...] = ('queue_timeout', 'queue_give_up')

# aiohttp sub-timeout values used in `_call_api`. Kept as named constants so the diagnostic
# logic can compare elapsed time against the actual configured budgets (rather than a magic
# 0.9 fraction of the total timeout, which silently lies if anyone tunes sock_read smaller).
API_CONNECT_TIMEOUT = 60.0
API_SOCK_CONNECT_TIMEOUT = 60.0
STOP_RUNTIME_MAX_ATTEMPTS = 5
STOP_RUNTIME_RETRY_BASE_SECONDS = 0.5
STOP_RUNTIME_RETRY_MAX_SECONDS = 4.0

# Linux capability names accepted by the sandbox API.
_CAP_NAME_RE = re.compile(r'^[A-Z][A-Z0-9_]*$')

# Upper bound for the sandbox ttl, mirroring the backend's MaxSuanliSandboxTTL (168h = 7 days).
# The backend rejects anything above it instead of silently truncating, so rejecting client-side
# costs the caller nothing and saves a doomed round-trip.
MAX_TTL_SECONDS = 168 * 3600

# Go duration grammar, narrowed to the units that make sense for a sandbox lifetime: one or more
# <number><unit> terms, optionally signed. Go itself also accepts ns/us/ms, but a sub-second
# lifetime can never produce a usable sandbox (scheduling alone takes tens of seconds), so those
# are rejected here rather than forwarded. Go has no day/week/year unit either, which is why '7d'
# gets its own error message below.
_GO_DURATION_UNITS: dict[str, float] = {
    's': 1.0,
    'm': 60.0,
    'h': 3600.0,
}
_GO_DURATION_TERM_RE = re.compile(r'(\d+(?:\.\d*)?|\.\d+)(s|m|h)')
_GO_DURATION_RE = re.compile(rf'^[+-]?(?:{_GO_DURATION_TERM_RE.pattern})+$')

# A day/week/year suffix ('7d') is the single most common ttl mistake, and the generic
# "not a Go duration" message doesn't tell the caller what to write instead.
# The trailing lookahead (rather than \b) keeps '1d12h' matching: a digit may legitimately follow
# the bogus unit when more terms come after it, but another letter means this isn't a unit at all.
_GO_DURATION_UNSUPPORTED_UNIT_RE = re.compile(r'\d\s*(?:days?|d|weeks?|w|years?|y)(?![a-z])', re.IGNORECASE)

# Max bytes of response body embedded into raised exception messages. In legacy disabled mode the full body is
# emitted by the `Gongfeng API response: ...` trace line above (DEBUG by default); the exception message is only
# meant to give upstream callers a glance, not the full payload — a multi-MB HTML
# error page propagating up through every `logger.exception` call site can DoS the log
# pipeline. 4KB fits typical JSON error envelopes ({code, msg, stack}) without truncating
# the actually-useful part, while still capping pathological cases.
API_ERROR_BODY_PREVIEW_LIMIT = 4096


def _load_gongfeng_token(token: str | None) -> str:
    """Resolve an explicit, environment, or default-file sandbox token."""

    for candidate in (token, os.getenv('GONGFENG_TOKEN')):
        if candidate and candidate.strip():
            return candidate.strip()

    try:
        file_token = GONGFENG_AUTH_KEY_FILE.read_text(encoding='utf-8').strip()
    except FileNotFoundError:
        file_token = ''
    except OSError as exc:
        raise ValueError(f'Failed to read Gongfeng token file {GONGFENG_AUTH_KEY_FILE}: {exc}') from exc
    if file_token:
        return file_token

    raise ValueError(
        'token is required for Gongfeng sandbox API authentication. '
        'Pass it explicitly, set GONGFENG_TOKEN, or create '
        f'{GONGFENG_AUTH_KEY_FILE}. '
        'Apply at: https://sbox.woa.com/?tab=resource . '
        'For any sandbox usage questions, consult the AI assistant: '
        'https://sbox.woa.com/?tab=ai-assistant'
    )


class RuntimeQueueWaitTimeoutError(RuntimeError):
    """Raised when the client's queue_wait_max budget is exhausted while the sandbox is still queued.

    The client will best-effort call /runtime/stop to cancel the queue entry before raising,
    so the runtime_id will not leak into the backend's runtime_queue table beyond this point.
    """


class RuntimeQueueFailedError(RuntimeError):
    """Raised when the backend reports the queued sandbox as terminally failed during polling.

    This covers backend-side give-ups (remark prefix in QUEUE_FAILURE_REMARK_PREFIXES, e.g.
    queue_timeout / queue_give_up), as well as any other status=stopped or status=not_found
    observed during the wait (the backend's success path is queued → scheduled, so stopped
    during the wait is always a failure with no self-healing possibility).
    """


def _truncate_body_for_error(text: str, limit: int = API_ERROR_BODY_PREVIEW_LIMIT) -> str:
    """Truncate a response body for embedding in exception messages.

    The full body is always available in the preceding ``Gongfeng API response: ...`` trace line
    (emitted at DEBUG by default), so callers reading the exception only need a glanceable preview here.
    """
    if text is None:
        return ''
    if len(text) <= limit:
        return text
    return (
        f'{text[:limit]}...(truncated, total {len(text)} chars; '
        f'see full body in the preceding "Gongfeng API response" DEBUG log)'
    )


def _current_w3c_traceparent() -> str | None:
    """Build a W3C ``traceparent`` from the current nexus span, or None if unavailable.

    nexus ids are base64url (``random_urlsafe_id``: trace 16 bytes, span 8 bytes) but
    W3C traceparent requires fixed-width hex, so we decode and re-encode. This lets the
    sandbox backend's oteltrpc ServerFilter continue the same trace with zero backend
    change. Returns None when there is no active span (e.g. background watchdog tasks) or
    the ids decode to the wrong length, so a malformed header is never sent.
    """
    span = get_current_span()
    if span is None:
        return None
    try:
        # '==' / '=' restore the padding that random_urlsafe_id stripped; a wrong-length
        # id either raises here or fails the length check below — both fall through to None.
        trace_id = base64.urlsafe_b64decode(span.trace_id + '==').hex()
        span_id = base64.urlsafe_b64decode(span.span_id + '=').hex()
    except (ValueError, TypeError):
        return None
    if len(trace_id) != 32 or len(span_id) != 16:
        return None
    # version 00, sampled flag 01 — propagating the trace at all means we want it collected.
    return f'00-{trace_id}-{span_id}-01'


def _redact_api_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return redact_exporter_credentials(payload)


def _api_payload_for_log(payload: dict[str, Any], *, otel_enabled: bool) -> dict[str, Any]:
    return _redact_api_payload(payload) if otel_enabled else payload


def _api_sensitive_values(payload: dict[str, Any], *, otel_enabled: bool, token: str) -> list[str]:
    values: list[str] = []
    if otel_enabled:
        values.extend(exporter_credential_values(os.environ))
        values.extend(exporter_credential_values(payload))
    if token:
        values.append(str(token))
    return list(dict.fromkeys(values))


def _api_response_text_for_log(text: str, *, sensitive_values: list[str]) -> str:
    # Sandbox errors may echo the submitted environment, including exporter credentials.
    return redact_sensitive_text(text, sensitive_values)


def get_default_runtime_image() -> str:
    """Get the default runtime image URL with current package version."""
    return f'mirrors.tencent.com/hunyuan_yanguan/nexus-runtime:{get_package_version()}'


def build_gongfeng_runtime_port_summary(runtime: Runtime) -> list[str]:
    """Build a human-readable summary of Gongfeng runtime and GUI endpoints."""
    base_url = runtime.base_url or ''
    lines = [
        f'Runtime server: {base_url}',
        f'Debug GUI: {base_url.rstrip("/")}/debug/gui',
    ]

    novnc_url = runtime.port_mappings.get(6080)
    if novnc_url:
        lines.append(f'noVNC: {novnc_url.rstrip("/")}/vnc.html')

    vnc_url = runtime.port_mappings.get(5900)
    if vnc_url:
        lines.append(f'VNC: {vnc_url.removeprefix("http://").removeprefix("https://")}')

    return lines


@runtime_provider_registry.register('gongfeng')
class GongfengRuntimeProvider(RuntimeProvider):
    """Gongfeng-based RuntimeProvider for remote sandbox execution.

    This provider creates sandbox environments by calling the Gongfeng sandbox v3 API.
    It provides the same interface as DockerRuntimeProvider but manages remote resources.

    Config options:
        api_base_url: Base URL for the sandbox API (required)
        token: Authentication token for sandbox API (required, can also be set via GONGFENG_TOKEN env var.
               Apply at https://sbox.woa.com/?tab=resource)
        image: Main container image (takes precedence over configure() if set)
        runtime_image: Runtime image with /nix code. Pass the literal string 'current' (recommended)
                       to use the image matching the currently installed nexus package version;
                       leave unset for the same default. Pin to an explicit tag only when you
                       need to lock to a specific version.
        python_version: Python version to use in runtime (default: current interpreter version)
        runtime_provider: Resource scheduling strategy (gongfeng/suanli, default: empty for auto selection)
                          Note: Different from runtime_provider_name, which selects the RuntimeProvider implementation
        enable_dind: Provider-level default for Docker-in-Docker. Can be overridden per runtime via
                     runtime_configs.<name>.enable_dind / acquire(enable_dind=...).
        runtime_disable_aslr: Provider-level default for disabling ASLR in the runtime container.
                     When true, the backend force-routes to the suanli provider (sandbox-sdk only).
                     Can be overridden per runtime via runtime_configs.<name>.runtime_disable_aslr /
                     acquire(runtime_disable_aslr=...).
        network_policy: Provider-level default egress network policy dict (default: None). Shape:
                     {'default_action': 'allow'|'deny', 'egress': [{'action': 'allow'|'deny', 'target': '<domain>'}]}.
                     When set, the backend force-routes to the suanli provider. Can be overridden per
                     runtime via acquire(network_policy=...).
        cap_add: Provider-level default list of extra Linux capabilities to add to the sandbox main
                     container (default: None). Names are uppercase-underscore WITHOUT the CAP_ prefix,
                     e.g. ['NET_ADMIN', 'SYS_PTRACE']. suanli-only: a non-empty value force-routes the
                     backend to the suanli provider. Can be overridden per runtime via acquire(cap_add=...).
        ttl: Provider-level default sandbox lifetime as a Go duration string, e.g. '30m' / '12h' /
                     '168h' (default: 12h when unset). Go duration has no day unit, so 7 days must be
                     written '168h', NOT '7d'. The maximum is 168h and a larger value is rejected. Can
                     be overridden per runtime via acquire(ttl=...); pass '' there to fall back to the
                     default lifetime.
        resource_spec: Resource specification dict with 'cpu' (cores, 0-32, default 2), 'memory' (GB, 0-64, default 10)
                          and 'gpu' (GPU card count, default 0; do not specify unless needed)
                          Requires a token with resource quotas (cpu, gpu, etc.) applied at
                          https://sbox.woa.com/?tab=resource
                          Note: Please request resources based on actual usage to avoid waste
        working_dir: Container working directory (optional)
        auto_connect_ws: Whether to automatically connect Socket.IO for real-time logs and events (default: True)
        socketio_request_timeout: Engine.IO HTTP polling/connect request timeout in seconds (default: 60.0)
        rpc_protocol: RPC transport protocol, 'http' (default) or 'socketio'
        batch_id: Batch identifier for grouping runtime operations (default: auto-generated UUID)
        purpose: Purpose of the runtime, one of 'prod', 'eval', 'test' (default: 'test' if not specified)
                 - prod: Production tasks, typically launched via the Hunyuan platform
                 - eval: Evaluation tasks, typically launched via the evaluation platform
                 - test: Offline testing / experimentation / validation tasks
        print_port_mappings: Whether to print resolved runtime/debug GUI/noVNC/VNC URLs to stderr
        watchdog_poll_interval: Seconds between proxied {base_url}/health probes for each active
                 runtime. The probe hits the nexus server inside the pod through the proxy, so it
                 detects a wedged/crashed server (not just a stopped pod). Effective interval is
                 randomized within ±25% to avoid thundering herd across concurrent sandboxes. Default: 120s.
        queue_wait_max: Maximum wall-clock seconds the client will poll /runtime/get while the
                 backend keeps a /runtime/start request queued (status=queued, empty url). The
                 client is response-driven: it ALWAYS waits when the backend enqueues, whatever
                 the reason (token quota_full_policy=queue, high-load admission throttle, ...).
                 Default: 0, interpreted as "no client-side deadline" — poll until the sandbox is
                 scheduled or the backend gives up (the backend's 48h queue retention is the hard
                 ceiling, after which the entry is auto-purged with remark=queue_timeout). Set a
                 positive value to cap the wait (e.g. a realtime task that should fail fast).
                 Independent of `request_timeout`, which only bounds a single RPC call's duration.
                 The polling interval inside this wait is a fixed internal detail (~15s with ±25%
                 jitter); only the total wait budget is exposed.

    Three timeout dimensions to keep distinct:
        - request_timeout: single-RPC HTTP timeout (start / get / stop each independent)
        - queue_wait_max:  total wall-clock budget across many /runtime/get polls while waiting
        - backend 48h:     hard ceiling enforced by sandbox-offline worker on runtime_queue rows

    Acquire params:
        image: Docker image to run for this runtime instance (used only if provider-level image not set)
        runtime_image: Runtime image (optional, uses provider default). Pass 'current' (recommended)
                       to use the image matching the currently installed nexus package version.
        environment: Environment variables to pass to the container
        runtime_id: Optional custom runtime ID (auto-generated if not provided)
        exposed_ports: List of additional internal ports to expose (optional, each gets auto-assigned host port)
                      Example: [8023, 8080, 8888]
        enable_dind: Whether to enable Docker-in-Docker for this sandbox. When true, the provider injects
                     NEXUS_ENABLE_DIND=true and forwards enable_dind to the sandbox scheduler.
        runtime_disable_aslr: Whether to disable ASLR in the runtime container. When true, forwarded
                     to the sandbox scheduler as runtime_disable_aslr and force-routes to suanli.
        cap_add: Extra Linux capabilities for the sandbox main container (optional, overrides provider
                     default). Uppercase-underscore names without the CAP_ prefix, e.g. ['NET_ADMIN'].
                     suanli-only: a non-empty value force-routes the backend to the suanli provider.
        ttl: Sandbox lifetime for this runtime as a Go duration string (optional, overrides provider
                     default), e.g. '30m' / '168h'. Pass '' to drop a provider-level default and fall
                     back to the default 12h lifetime.
        resource_spec: Resource specification (optional, uses provider default). Supports an optional
                     'gpu_card_type' (str) key alongside 'cpu'/'memory'/'gpu' to pin the GPU card type.
        working_dir: Container working directory (optional, uses provider default)
        auto_connect_ws: Whether to automatically connect Socket.IO (default: provider-level setting)
        socketio_request_timeout: Engine.IO HTTP polling/connect request timeout (default: provider-level setting)
        queue_wait_max: Override the provider-level queue_wait_max for this acquire() call.
                        Useful when different runtimes have different SLA expectations
                        (e.g. eval task may wait 4h, prod realtime task only 5min).
    """

    def __init__(
        self,
        api_base_url: str,
        token: str | None = None,
        image: str | None = None,
        runtime_image: str | None = None,
        python_version: str | None = None,
        runtime_provider: str | None = None,
        enable_dind: bool | str = False,
        runtime_disable_aslr: bool | str = False,
        resource_spec: dict[str, Any] | None = None,
        exposed_ports: list[int] | None = None,
        working_dir: str | None = None,
        request_timeout: float = 1800.0,
        auto_connect_ws: bool = True,
        socketio_request_timeout: float = 60.0,
        rpc_protocol: str = 'http',
        batch_id: str | None = None,
        need_transparent_proxy: list[str] | None = None,
        network_policy: dict[str, Any] | None = None,
        cap_add: list[str] | None = None,
        ttl: str | None = None,
        purpose: str | None = None,
        print_port_mappings: bool = True,
        watchdog_poll_interval: float = DEFAULT_WATCHDOG_POLL_INTERVAL,
        queue_wait_max: float = 0,
        startup_jitter_max: float = 0.0,
        **config: Any,
    ) -> None:
        """Initialize GongfengRuntimeProvider.

        Args:
            api_base_url: Base URL for the sandbox API, e.g. 'http://sandbox.example.com'
            token: Authentication token for sandbox API (required, can also be set via GONGFENG_TOKEN env var.
                   Apply at https://sbox.woa.com/?tab=resource)
            image: Main container image (required)
            runtime_image: Runtime image. Pass 'current' (recommended) to use the image matching
                           the currently installed nexus package version; same as leaving it unset.
                           Pin to an explicit tag only when locking to a specific version.
            python_version: Python version to use in runtime (default: current interpreter version)
            runtime_provider: Resource scheduling strategy (gongfeng/suanli, default: empty for auto selection)
            enable_dind: Provider-level default for Docker-in-Docker. Can be overridden per runtime via
                         runtime_configs.<name>.enable_dind / acquire(enable_dind=...).
            runtime_disable_aslr: Provider-level default for disabling ASLR in the runtime container.
                         When true, the backend force-routes to suanli. Can be overridden per runtime
                         via runtime_configs.<name>.runtime_disable_aslr / acquire(runtime_disable_aslr=...).
            resource_spec: Resource specification with 'cpu' (cores, 0-32, default 2), 'memory' (GB, 0-64, default 10)
                           and 'gpu' (GPU card count, default 0; do not specify unless needed)
                           Requires a token with resource quotas (cpu, gpu, etc.) applied at
                           https://sbox.woa.com/?tab=resource
                           Note: Please request resources based on actual usage to avoid waste
            exposed_ports: List of additional internal ports to expose (default: empty)
            working_dir: Container working directory (optional)
            request_timeout: API request timeout in seconds
            auto_connect_ws: Whether to automatically connect Socket.IO for real-time logs and events (default: True)
            socketio_request_timeout: Engine.IO HTTP polling/connect request timeout in seconds (default: 60.0)
            rpc_protocol: RPC transport protocol, 'http' (default) or 'socketio'
            batch_id: Batch identifier for grouping runtime operations (default: auto-generated UUID)
            need_transparent_proxy: List of transparent proxy protocols to enable (default: None,
                                 disabled). Allowed values: 'http', 'https', 'tcp' (at most 3).
                                 When set, the backend enables NetworkSidecarEnabled on the
                                 computing platform. Can be overridden per-runtime via acquire().
            network_policy: Egress network policy dict (default: None, disabled). Shape:
                                 {'default_action': 'allow'|'deny', 'egress': [{'action': 'allow'|'deny',
                                 'target': '<target>'}, ...]}. default_action is the fallback when no
                                 egress rule matches (defaults to 'allow' if omitted); egress rules are
                                 evaluated in order, first match wins. target accepts an exact domain
                                 (example.com), a wildcard domain (*.example.com), an IPv4 (1.2.3.4), a
                                 CIDR (10.0.0.0/8) or a host:port (api.example.com:443) — NOT a URL (no
                                 scheme/path); CIDR cannot carry a port and IPv6 is unsupported.
                                 When set, the backend force-routes to the suanli provider (do not also
                                 pass a non-suanli runtime_provider, or the backend rejects with
                                 InvalidParam). Validated client-side; can be overridden per-runtime via
                                 acquire(). See https://iwiki.woa.com/p/4021452048 for the full spec.

                                 Two common shapes:
                                 - Allowlist (deny by default, only let a few through):
                                     {'default_action': 'deny',
                                      'egress': [{'action': 'allow', 'target': '*.tencent.com'},
                                                 {'action': 'allow', 'target': '*.woa.com'}]}
                                 - Blocklist (allow by default, only block a few):
                                     {'default_action': 'allow',
                                      'egress': [{'action': 'deny', 'target': 'evil.example.com'}]}
            cap_add: Extra Linux capabilities to add to the sandbox main container (default: None,
                                 disabled). Names use uppercase-underscore style WITHOUT the CAP_ prefix,
                                 e.g. ['NET_ADMIN', 'SYS_PTRACE']. Common values: NET_ADMIN (network config),
                                 SYS_PTRACE (debug/strace), SYS_ADMIN (mount and other privileged ops),
                                 NET_RAW (raw sockets). suanli-only feature: a non-empty value makes the
                                 backend force-route to the suanli provider — if you also pass a non-suanli
                                 runtime_provider it is silently overridden to suanli (the backend logs a
                                 warning; it does NOT reject). Validated client-side; can be overridden
                                 per-runtime via acquire().
            ttl: Sandbox lifetime as a Go duration string (default: None, meaning the standard 12h
                                 lifetime applies). Accepts one or more <number><unit> terms using the
                                 s / m / h units, e.g. '30m', '12h', '1h30m', '168h'. There is no day
                                 unit: 7 days must be written '168h', and '7d' is rejected with a hint.
                                 Sub-second units are rejected too — a lifetime that short can never
                                 yield a usable sandbox. The maximum is 168h; anything larger is
                                 rejected rather than silently truncated. Can be overridden
                                 per-runtime via acquire(ttl=...), where '' means "clear the provider
                                 default and fall back to the standard lifetime".

                                 Note on expiry: when the lifetime runs out the sandbox is reclaimed,
                                 and nexus notices it through the health watchdog — the runtime is marked
                                 dead with an "unreachable" reason after several failed probes (minutes
                                 later), not with an explicit "ttl expired". The configured ttl is echoed
                                 in the "Sandbox 'x': creating (...)" log line so such a death can be
                                 matched against the requested lifetime.
            purpose: Purpose of the runtime, one of 'prod', 'eval', 'test' (default: 'test' if not specified)
                     - prod: Production tasks, typically launched via the Hunyuan platform
                     - eval: Evaluation tasks, typically launched via the evaluation platform
                     - test: Offline testing / experimentation / validation tasks
            print_port_mappings: Whether to print resolved runtime and GUI port URLs to stderr
            watchdog_poll_interval: Seconds between proxied {base_url}/health probes for each active
                     runtime. The probe hits the nexus server inside the pod through the proxy, so it
                     detects a wedged/crashed server (not just a stopped pod). Effective interval is
                     randomized within ±25% to avoid thundering herd across concurrent sandboxes. Default: 120s.
            queue_wait_max: Total wall-clock budget (in seconds) the client will spend polling
                     /runtime/get while the sandbox sits queued in the backend. The client is
                     response-driven and always polls when the backend enqueues (status=queued).
                     Default 0 means "no client-side deadline" — poll until scheduled or the
                     backend gives up. Set a positive value to cap the wait. The polling interval
                     is a fixed internal detail (~15s with ±25% jitter), not exposed.
            startup_jitter_max: Maximum seconds to sleep (uniform random) before each acquire()
                     call, to de-synchronize concurrent runtime starts and avoid thundering herd
                     on the sandbox API. Default 0 means no jitter.
            **config: Additional config options
        """
        super().__init__(**config)

        # Validate required parameters
        if not api_base_url:
            raise ValueError('api_base_url is required')
        resolved_token = _load_gongfeng_token(token)

        self.api_base_url = api_base_url.rstrip('/')
        self.token = resolved_token
        self.agentflow_type = 'nexus'  # Fixed value for nexus
        self.image = image
        # 支持显式写 runtime_image: "current"，等价于使用当前版本的镜像
        self.runtime_image = (
            get_default_runtime_image() if (not runtime_image or runtime_image == 'current') else runtime_image
        )
        self.python_version = python_version or get_current_python_version()
        self.runtime_provider = runtime_provider
        self.enable_dind = coerce_bool(enable_dind, name='enable_dind')
        self.runtime_disable_aslr = coerce_bool(runtime_disable_aslr, name='runtime_disable_aslr')
        self.resource_spec = resource_spec
        self.exposed_ports = exposed_ports or []
        self.working_dir = working_dir
        try:
            self.request_timeout = float(request_timeout)
        except (TypeError, ValueError) as exc:
            raise ValueError(f'Invalid request_timeout: {request_timeout!r}. Expected a number.') from exc
        self.auto_connect_ws = auto_connect_ws
        self.rpc_protocol = rpc_protocol
        self.batch_id = batch_id or str(uuid.uuid4())
        self.need_transparent_proxy = self._validate_transparent_proxy(need_transparent_proxy)
        self.network_policy = self._validate_network_policy(network_policy)
        self.cap_add = self._validate_cap_add(cap_add)
        self.ttl = self._validate_ttl(ttl)
        self.print_port_mappings = print_port_mappings

        try:
            self.watchdog_poll_interval = float(watchdog_poll_interval)
        except (TypeError, ValueError) as exc:
            raise ValueError(f'Invalid watchdog_poll_interval: {watchdog_poll_interval!r}. Expected a number.') from exc
        if self.watchdog_poll_interval <= 0:
            raise ValueError(f'watchdog_poll_interval must be > 0, got {self.watchdog_poll_interval}')
        self.socketio_request_timeout = socketio_request_timeout

        # queue_wait_max: 0 (default) means "poll with no client-side deadline" — the client is
        # response-driven and always waits when the backend enqueues. > 0 caps the wait. Negative
        # is rejected to fail fast on obvious typos like -1.
        try:
            self.queue_wait_max = float(queue_wait_max)
        except (TypeError, ValueError) as exc:
            raise ValueError(f'Invalid queue_wait_max: {queue_wait_max!r}. Expected a number.') from exc
        if self.queue_wait_max < 0:
            raise ValueError(f'queue_wait_max must be >= 0, got {self.queue_wait_max}')

        self.startup_jitter_max = float(startup_jitter_max)

        # Validate and store purpose
        valid_purposes = ('prod', 'eval', 'test')
        if purpose is not None and purpose not in valid_purposes:
            raise ValueError(f'Invalid purpose: {purpose!r}, must be one of {valid_purposes}')
        self.purpose = purpose

        # Validate python_version format
        if '.' not in self.python_version:
            raise ValueError(
                f'Invalid python_version format: {self.python_version}, expected "major.minor" like "3.12"'
            )
        parts = self.python_version.split('.')
        if len(parts) != 2 or not all(part.isdigit() for part in parts):
            raise ValueError(
                f'Invalid python_version format: {self.python_version}, expected "major.minor" like "3.12"'
            )

        # Track runtime_id -> name mapping
        self._runtime_ids: dict[str, str] = {}  # name -> runtime_id
        # Lock for acquire to prevent concurrent access to same name
        self._acquire_locks: dict[str, asyncio.Lock] = {}
        # Background watchdog tasks keyed by runtime name
        self._watchdog_tasks: dict[str, asyncio.Task[None]] = {}

    @staticmethod
    def _validate_transparent_proxy(protocols: list[str] | None) -> list[str] | None:
        """Validate need_transparent_proxy protocols."""
        if not protocols:
            return None
        allowed = {'http', 'https', 'tcp'}
        if len(protocols) > 3:
            raise ValueError('need_transparent_proxy supports at most 3 protocols')
        normalized = []
        for p in protocols:
            p = p.strip().lower()
            if p not in allowed:
                raise ValueError(f'need_transparent_proxy only supports: http, https, tcp; got {p!r}')
            if p not in normalized:
                normalized.append(p)
        return normalized

    @staticmethod
    def _validate_network_policy(network_policy: dict[str, Any] | None) -> dict[str, Any] | None:
        """Validate and normalize an egress network policy.

        Mirrors the backend's ValidateNetworkPolicy so an obviously malformed policy fails fast on
        the client instead of after a round-trip InvalidParam. ``None`` / empty returns ``None`` (the
        policy is simply not sent). A non-empty policy is normalized to only the proto-known fields
        (``default_action`` + ``egress`` rules), dropping any stray keys so the transmitted payload
        stays clean.

        Note: passing a non-empty network_policy makes the backend force-route to the suanli
        provider; combining it with an explicit non-suanli ``runtime_provider`` is rejected
        server-side with InvalidParam. We intentionally do not duplicate that cross-field rule here,
        consistent with how enable_dind / need_transparent_proxy leave provider-conflict checks to
        the backend.

        Args:
            network_policy: Egress policy dict, or None.

        Returns:
            Normalized policy dict, or None when nothing should be sent.

        Raises:
            ValueError: When the policy is structurally invalid (bad default_action/action,
                non-list egress, or a rule with a missing/empty target).
        """
        if not network_policy:
            return None
        if not isinstance(network_policy, dict):
            raise ValueError(f'network_policy must be a dict, got {type(network_policy).__name__}')

        allowed_actions = {'allow', 'deny'}
        # default_action omitted -> 'allow', matching the docstring, the iwiki spec
        # (https://iwiki.woa.com/p/4021452048) and the backend ValidateNetworkPolicy. An explicitly
        # supplied bad value still fails below (omission = caller didn't opt in; a wrong value = typo).
        default_action = network_policy.get('default_action', 'allow')
        if default_action not in allowed_actions:
            raise ValueError(
                f'network_policy.default_action must be one of {sorted(allowed_actions)}, got {default_action!r}'
            )

        raw_egress = network_policy.get('egress') or []
        if not isinstance(raw_egress, list):
            raise ValueError(f'network_policy.egress must be a list, got {type(raw_egress).__name__}')

        normalized_egress: list[dict[str, str]] = []
        for idx, rule in enumerate(raw_egress):
            if not isinstance(rule, dict):
                raise ValueError(f'network_policy.egress[{idx}] must be a dict, got {type(rule).__name__}')
            action = rule.get('action')
            if action not in allowed_actions:
                raise ValueError(
                    f'network_policy.egress[{idx}].action must be one of {sorted(allowed_actions)}, got {action!r}'
                )
            target = rule.get('target')
            if not isinstance(target, str) or not target.strip():
                raise ValueError(f'network_policy.egress[{idx}].target must be a non-empty string, got {target!r}')
            normalized_egress.append({'action': action, 'target': target})

        return {'default_action': default_action, 'egress': normalized_egress}

    @staticmethod
    def _validate_cap_add(cap_add: list[str] | None) -> list[str] | None:
        """Validate and normalize a cap_add (extra Linux capabilities) list.

        Mirrors the backend's expectation for StartRuntimeRequest.cap_add so an obviously
        malformed value fails fast client-side instead of after a round-trip. ``None`` / empty
        returns ``None`` (the field is simply not sent, preserving the old behavior). Capability
        names must be uppercase-underscore style WITHOUT the ``CAP_`` prefix (e.g. 'NET_ADMIN',
        'SYS_PTRACE') — the backend rejects 'CAP_NET_ADMIN' / 'net_admin'.

        Note: a non-empty cap_add makes the backend force-route to the suanli provider (mirroring
        network_policy). It does NOT reject a co-supplied non-suanli ``runtime_provider`` — the
        backend's ResolveProvider silently overrides it to suanli (logging a warning). We leave that
        cross-field routing to the backend, consistent with how enable_dind / need_transparent_proxy
        / network_policy defer provider selection.

        Validation order is deliberate: ``None`` is handled explicitly (not via ``not``) BEFORE the
        list-type check, so a falsy-but-wrong value like ``''`` / ``{}`` / ``0`` raises instead of
        being silently swallowed as "unset". An empty list ``[]`` is the one falsy value treated as
        "clear / unset" (returns None), so acquire(cap_add=[]) can clear a provider-level default.

        Args:
            cap_add: List of capability names, or None.

        Returns:
            Deduplicated list of capability names, or None when nothing should be sent
            (``None`` input or an empty list).

        Raises:
            ValueError: When cap_add is not a list, or any entry is not a non-empty string in the
                expected uppercase-underscore form (must match ``[A-Z][A-Z0-9_]*``, no CAP_ prefix).
        """
        if cap_add is None:
            return None
        if not isinstance(cap_add, list):
            raise ValueError(f'cap_add must be a list, got {type(cap_add).__name__}')
        if not cap_add:
            # Empty list == "clear / not configured": nothing to send. Checked after the type
            # guard so falsy non-list values ('' / {} / 0) still raise above.
            return None

        normalized: list[str] = []
        for idx, cap in enumerate(cap_add):
            if not isinstance(cap, str) or not cap.strip():
                raise ValueError(f'cap_add[{idx}] must be a non-empty string, got {cap!r}')
            cap = cap.strip()
            if cap.startswith('CAP_'):
                raise ValueError(f'cap_add[{idx}]={cap!r} must not carry the CAP_ prefix (e.g. use "NET_ADMIN")')
            if not _CAP_NAME_RE.match(cap):
                raise ValueError(
                    f'cap_add[{idx}]={cap!r} must be an uppercase-underscore capability name '
                    f'matching [A-Z][A-Z0-9_]* (e.g. "NET_ADMIN"); no lowercase, hyphen, space or leading digit'
                )
            if cap not in normalized:
                normalized.append(cap)
        return normalized

    @staticmethod
    def _validate_ttl(ttl: str | None) -> str | None:
        """Validate and normalize a sandbox ttl (lifetime) Go duration string.

        Mirrors the backend's ``do.ValidateTTL`` so a malformed lifetime fails before the sandbox
        start request is even built. ``None`` and ``''`` both return ``None`` (the field is simply
        not sent, so the backend applies its own default); ``''`` is the value acquire() callers use
        to clear a provider-level default, mirroring ``cap_add=[]``.

        Note: a non-empty ttl makes the backend force-route to the suanli provider, the same way
        cap_add / network_policy do. That cross-field routing is left to the backend.

        Args:
            ttl: Go duration string, or None.

        Returns:
            The stripped duration string, or None when nothing should be sent.

        Raises:
            ValueError: When ttl is not a string, uses a day/week/year unit that Go duration does
                not support, is not a valid Go duration, is non-positive, or exceeds
                :data:`MAX_TTL_SECONDS`.
        """
        if ttl is None:
            return None
        if not isinstance(ttl, str):
            raise ValueError(f'ttl must be a Go duration string such as "12h", got {type(ttl).__name__}')
        ttl = ttl.strip()
        if not ttl:
            return None

        if _GO_DURATION_UNSUPPORTED_UNIT_RE.search(ttl):
            raise ValueError(
                f'ttl={ttl!r} uses a day/week/year unit, which Go duration does not support; '
                f'express it in hours instead (7 days is "168h")'
            )
        if not _GO_DURATION_RE.match(ttl):
            raise ValueError(
                f'ttl={ttl!r} must be a Go duration string made of <number><unit> terms '
                f'(units: s, m, h — sub-second units are not accepted), e.g. "30m", "12h", "1h30m", "168h"'
            )

        seconds = sum(float(value) * _GO_DURATION_UNITS[unit] for value, unit in _GO_DURATION_TERM_RE.findall(ttl))
        if ttl.startswith('-'):
            seconds = -seconds
        if seconds <= 0:
            raise ValueError(f'ttl={ttl!r} must be positive')
        if seconds > MAX_TTL_SECONDS:
            raise ValueError(f'ttl={ttl!r} must not exceed {MAX_TTL_SECONDS // 3600}h')
        return ttl

    def _create_runtime(
        self,
        url: str,
        resource_spec: dict[str, Any] | None = None,
        auto_connect_ws: bool | None = None,
        socketio_request_timeout: float | None = None,
    ) -> Runtime:
        """Create the appropriate runtime client based on rpc_protocol setting.

        Args:
            url: Base URL of the runtime server
            resource_spec: Resource specification (optional)
            auto_connect_ws: Whether to auto-connect Socket.IO (only used for HttpRuntime)
            socketio_request_timeout: Engine.IO HTTP polling/connect request timeout (optional)

        Returns:
            HttpRuntime or SocketIoRuntime instance
        """
        actual_socketio_request_timeout = (
            socketio_request_timeout if socketio_request_timeout is not None else self.socketio_request_timeout
        )
        if self.rpc_protocol == 'socketio':
            return SocketIoRuntime(
                base_url=url,
                request_timeout=self.request_timeout,
                resource_spec=resource_spec,
                enable_runtime_logs=True,
                socketio_request_timeout=actual_socketio_request_timeout,
            )
        return HttpRuntime(
            base_url=url,
            request_timeout=self.request_timeout,
            resource_spec=resource_spec,
            auto_connect_ws=auto_connect_ws if auto_connect_ws is not None else self.auto_connect_ws,
            socketio_request_timeout=actual_socketio_request_timeout,
        )

    async def _call_api(
        self,
        endpoint: str,
        payload: dict[str, Any],
        timeout: float | None = None,
        log_level: int = logging.DEBUG,
    ) -> dict[str, Any]:
        """Call the sandbox API.

        Args:
            endpoint: API endpoint path, e.g. '/sandbox/v3/runtime/start'
            payload: Request body
            timeout: Request timeout, defaults to self.request_timeout
            log_level: Level for the routine request/response trace lines (full payload + body).
                Defaults to DEBUG: these raw API dumps are diagnostic detail, not something the
                average user should parse at INFO. The human-readable lifecycle lines ("Sandbox
                'x': creating ... / READY ...") already carry the key facts (image, resource_spec,
                url, runtime_id) at INFO; open DEBUG to see request/response diagnostics. When OTel is enabled,
                environment values and response bodies are redacted because the backend can echo credentials. Error
                paths (non-200, timeout, code != 0) ignore this and always log at warning/exception
                level, so failures stay visible regardless.

        Returns:
            API response data

        Raises:
            RuntimeError: When API request fails or returns non-zero code
        """
        url = f'{self.api_base_url}{endpoint}'
        timeout = timeout or self.request_timeout

        otel_enabled = is_otel_enabled()
        sensitive_values = _api_sensitive_values(payload, otel_enabled=otel_enabled, token=self.token)
        logger.log(
            log_level,
            'Gongfeng API request: %s, payload: %s',
            url,
            _api_payload_for_log(payload, otel_enabled=otel_enabled),
        )

        start = time.monotonic()
        try:
            client_timeout = aiohttp.ClientTimeout(
                total=timeout,
                connect=API_CONNECT_TIMEOUT,
                sock_read=timeout,
                sock_connect=API_SOCK_CONNECT_TIMEOUT,
            )
            headers = {'token': f'{self.token}'}
            if not otel_enabled:
                # Preserve the legacy Nexus-ID conversion exactly when OTel is disabled.
                traceparent = _current_w3c_traceparent()
                if traceparent:
                    headers['traceparent'] = traceparent
            resp = await http_client.post(url, json=payload, timeout=client_timeout, headers=headers)

            response_text = _api_response_text_for_log(resp.text, sensitive_values=sensitive_values)
            logger.log(log_level, 'Gongfeng API response: status=%s, body=%s', resp.status_code, response_text)

            # Check HTTP status code before parsing JSON
            # Note: full status & body have been logged above via the 'Gongfeng API response' line (DEBUG by default)
            if resp.status_code != 200:
                raise RuntimeError(
                    f'HTTP error (status={resp.status_code}) for {url}: {_truncate_body_for_error(response_text)}'
                )

            try:
                data = resp.json()
            except Exception as json_err:
                if otel_enabled:
                    safe_json_error = redact_sensitive_text(repr(json_err), sensitive_values)
                    logger.error('Failed to parse response as JSON: url=%s, error=%s', url, safe_json_error)
                    raise RuntimeError(
                        f'Failed to parse API response as JSON for {url}: {safe_json_error}, '
                        f'body={_truncate_body_for_error(response_text)!r}'
                    ) from None
                logger.exception('Failed to parse response as JSON: url=%s, error=%r', url, json_err)
                raise RuntimeError(
                    f'Failed to parse API response as JSON for {url}: {json_err!r}, '
                    f'body={_truncate_body_for_error(response_text)!r}'
                ) from json_err
        except RuntimeError:
            # RuntimeErrors above (HTTP non-200, JSON parse failure) already carry a precisely
            # constructed message and have been logged. Re-raise as-is so the bare
            # `except Exception` below doesn't double-wrap them into a generic
            # 'API request failed: RuntimeError(RuntimeError(...))' and re-emit a duplicate
            # `logger.exception` traceback.
            raise
        except (aiohttp.ServerTimeoutError, asyncio.TimeoutError) as e:
            elapsed = time.monotonic() - start

            # Surface the underlying trigger (ConnectionRefusedError, socket.gaierror, OSError,
            # ...) which aiohttp hangs off __cause__. Bounded loop guards against pathological
            # self-referential cycles; real chains are 2–3 deep.
            root: BaseException = e
            for _ in range(16):
                nxt = root.__cause__
                if nxt is None or nxt is root:
                    break
                root = nxt

            # Diagnose which timeout budget actually fired. Comparing `elapsed` against the
            # configured sub-timeouts directly (rather than a fraction of `total`) keeps the
            # diagnosis correct even if connect/sock_connect get tuned independently in the future.
            # The 1.5x multiplier absorbs normal scheduling jitter / event-loop lag without
            # bleeding into the next bucket.
            sub_timeout_ceiling = max(API_CONNECT_TIMEOUT, API_SOCK_CONNECT_TIMEOUT) * 1.5
            if elapsed <= sub_timeout_ceiling and timeout > sub_timeout_ceiling:
                # Fired well before total could plausibly have been hit, and we're inside the
                # connect/sock_connect window — almost always local IO/event-loop stall or an
                # unreachable gateway, NOT a slow sandbox.
                diagnosis = (
                    f'sub-timeout triggered at ~{elapsed:.1f}s '
                    f'(connect={API_CONNECT_TIMEOUT}s / sock_connect={API_SOCK_CONNECT_TIMEOUT}s); '
                    f'likely local IO/event-loop stall or unreachable gateway, '
                    f'NOT sandbox response delay'
                )
            else:
                # Burned (close to) the full budget — sandbox accepted the TCP connection but
                # never returned a response in time.
                diagnosis = (
                    f'total timeout reached after {elapsed:.1f}s '
                    f'(sock_read={timeout}s); '
                    f'sandbox accepted the connection but did not respond in time'
                )

            logger.exception(
                'API request timeout: url=%s, elapsed=%.1fs, total_timeout=%ss, error=%r, root_cause=%r, diagnosis=%s',
                url,
                elapsed,
                timeout,
                e,
                root,
                diagnosis,
            )
            raise RuntimeError(
                f'API request timeout for {url} '
                f'(elapsed={elapsed:.1f}s, total_timeout={timeout}s, '
                f'connect={API_CONNECT_TIMEOUT}s, sock_connect={API_SOCK_CONNECT_TIMEOUT}s, '
                f'sock_read={timeout}s): '
                f'{type(e).__name__}: {e!r}, root_cause={type(root).__name__}: {root!r}. '
                f'Diagnosis: {diagnosis}'
            ) from e
        except aiohttp.ClientError as e:
            elapsed = time.monotonic() - start
            logger.exception('API client error: url=%s, elapsed=%.1fs, error=%r', url, elapsed, e)
            raise RuntimeError(f'API client error for {url} (elapsed={elapsed:.1f}s): {type(e).__name__}: {e!r}') from e
        except Exception as e:
            elapsed = time.monotonic() - start
            logger.exception('API request failed: url=%s, elapsed=%.1fs, error=%r', url, elapsed, e)
            raise RuntimeError(
                f'API request failed for {url} (elapsed={elapsed:.1f}s): {type(e).__name__}: {e!r}'
            ) from e

        code = data.get('code', -1)
        msg = data.get('msg', 'Unknown error')

        if code != 0:
            safe_msg = redact_sensitive_text(str(msg), sensitive_values)
            raise RuntimeError(f'Gongfeng API error (code={code}): {safe_msg}')

        return data

    def _start_payload_extras(self, merged: dict[str, Any]) -> dict[str, Any]:
        """Extra Start JSON fields assembled by a subclass.

        Gongfeng has nothing extra. Taiji uses this to attach ``taiji_app_group`` /
        ``taiji_app_group_location`` from provider defaults merged with configure/acquire params.
        """
        return {}

    async def _start_runtime(
        self,
        image: str,
        runtime_image: str | None = None,
        environment: dict[str, str] | None = None,
        runtime_id: str | None = None,
        exposed_ports: list[int] | None = None,
        resource_spec: dict[str, Any] | None = None,
        working_dir: str | None = None,
        need_transparent_proxy: list[str] | None = None,
        network_policy: dict[str, Any] | None = None,
        cap_add: list[str] | None = None,
        ttl: str | None = None,
        enable_dind: bool = False,
        runtime_disable_aslr: bool = False,
        extra_payload: dict[str, Any] | None = None,
    ) -> tuple[str, str, dict[int, str], str]:
        """Start a runtime.

        Note: When the backend allocates immediately, this method returns once the runtime is
        fully started and ready (non-empty url). When the token has quota_full_policy=queue
        and the quota is exhausted, the backend returns immediately with an empty url (and a
        msg starting with "queued"); the caller (_acquire_impl) is then responsible for polling
        /runtime/get until the queue clears. The underlying computing platform and Gongfeng
        sandbox service guarantee runtime availability for the scheduled case; the queued case
        carries no such guarantee until promoted.

        Note on status inference: the Start response (proto StartRuntimeResponse_Data) does NOT
        carry a status field — only runtime_id, url, port_mappings. We synthesize a status here
        from url emptiness so the caller has a uniform contract; do not rely on
        result_data['status'] in this method (only GetRuntime carries it).

        Args:
            image: Main image
            runtime_image: Runtime image
            environment: Environment variables
            runtime_id: Optional runtime ID
            exposed_ports: List of additional internal ports to expose
            resource_spec: Resource specification with 'cpu', 'memory' and 'gpu' (default 0, do not specify unless needed)
            working_dir: Container working directory
            need_transparent_proxy: Transparent proxy protocols to enable (already validated)
            network_policy: Egress network policy dict (already validated/normalized); non-empty
                forces backend routing to the suanli provider
            cap_add: Extra Linux capabilities to add to the sandbox main container (already validated);
                non-empty forces backend routing to the suanli provider (suanli-only feature)
            ttl: Sandbox lifetime as a Go duration string (already validated); non-empty forces backend
                routing to the suanli provider. Omitted means the backend applies its own default
            enable_dind: Whether to request Docker-in-Docker scheduler support and runtime bootstrap
            runtime_disable_aslr: Whether to disable ASLR in the runtime container; true force-routes
                the backend to the suanli provider (sandbox-sdk only)
            extra_payload: Optional extra Start JSON fields from a subclass hook. Empty values are
                omitted. Gongfeng always passes None.

        Raises:
            ValueError: If both enable_dind and need_transparent_proxy are set — DinD injects
                explicit HTTP_PROXY env into inner containers (路线 A) while transparent proxy
                hijacks egress at the network layer (路线 B/C); the two are mutually exclusive
                (gVisor doc: docs/dind-inner-egress-user-guide.md §2.3).

        Returns:
            (runtime_id, url, port_mappings, status) tuple.
            - status=scheduled: runtime is ready, url and port_mappings are populated.
            - status=queued: backend enqueued the request, url is empty and port_mappings is empty;
              caller must poll /runtime/get until promotion.
            port_mappings: dict mapping internal_port -> external_url
        """
        payload: dict[str, Any] = {
            'agentflow_type': self.agentflow_type,
            'image': image,
            'batch_id': self.batch_id,
        }

        # DinD (路线 A: 显式 proxy env 注入) 和 transparent proxy (路线 B/C: 网络层劫持)
        # 是两条互斥的内层出网机制——同时启用会让 inner container 同时收到 dind-runc
        # 注入的 HTTP_PROXY env 和 sidecar 透明劫持，两者冲突。详见 gVisor 文档
        # docs/dind-inner-egress-user-guide.md §2.3。
        if enable_dind and need_transparent_proxy:
            raise ValueError(
                'enable_dind and need_transparent_proxy are mutually exclusive: '
                'DinD injects explicit HTTP_PROXY env into inner containers while '
                'transparent proxy hijacks egress at the network layer; pick one.'
            )

        if runtime_image or self.runtime_image:
            payload['runtime_image'] = runtime_image or self.runtime_image

        # Merge user environment with PYTHON_VERSION
        # PYTHON_VERSION ensures client/server serialization compatibility
        # Always use provider's python_version to ensure compatibility
        merged_environment = {**(environment or {}), 'PYTHON_VERSION': self.python_version}
        if enable_dind:
            merged_environment['NEXUS_ENABLE_DIND'] = 'true'
            payload['enable_dind'] = True
        if runtime_disable_aslr:
            payload['runtime_disable_aslr'] = True
        payload['environment'] = merged_environment

        if runtime_id:
            payload['runtime_id'] = runtime_id

        if self.runtime_provider:
            payload['runtime_provider'] = self.runtime_provider

        # Add resource_spec if provided (parameter takes precedence over provider default)
        actual_resource_spec = resource_spec or self.resource_spec
        if actual_resource_spec:
            payload['resource_spec'] = actual_resource_spec

        # Add exposed_ports for additional port mappings (used by WebArena etc.)
        if exposed_ports:
            payload['exposed_ports'] = exposed_ports

        # Add working_dir if provided (parameter takes precedence over provider default)
        actual_working_dir = working_dir or self.working_dir
        if actual_working_dir:
            payload['working_dir'] = actual_working_dir

        if self.purpose:
            payload['purpose'] = self.purpose

        # Pass transparent proxy protocols to backend
        if need_transparent_proxy:
            payload['need_transparent_proxy'] = need_transparent_proxy

        # Pass egress network policy to backend; non-empty forces routing to the suanli provider
        if network_policy:
            payload['network_policy'] = network_policy

        # Pass extra Linux capabilities to backend; non-empty forces routing to the suanli provider
        if cap_add:
            payload['cap_add'] = cap_add

        # Pass the requested sandbox lifetime to backend; non-empty forces routing to the suanli
        # provider. Omitting it (rather than sending the backend's 12h default) is deliberate: the
        # backend treats "absent" and "explicitly 12h" differently for routing and for how its
        # cleanup job decides expiry.
        if ttl:
            payload['ttl'] = ttl

        # Subclass hook (Taiji app group / location). Empty values stay omitted so the backend
        # applies its own defaults; Gongfeng itself always passes None.
        if extra_payload:
            for key, value in extra_payload.items():
                if value:
                    payload[key] = value

        payload['client_version'] = f'nexus/{get_package_version()}'

        data = await self._call_api('/sandbox/v3/runtime/start', payload)

        result_data = data.get('data', {})
        runtime_id = result_data.get('runtime_id')
        url = result_data.get('url') or ''
        # The Start response (per proto StartRuntimeResponse_Data) carries only runtime_id,
        # url and port_mappings — there is NO status field. Empty url is the queue signal:
        # the service explicitly returns url="" + msg="queued, please poll..." when the
        # request was enqueued due to quota_full_policy=queue. We infer status from url
        # emptiness alone (not from msg wording, which could change). For non-empty url
        # the runtime is fully scheduled and ready to dial. (Don't read result_data['status']
        # here — the backend's GetRuntime response carries that, but Start does not.)
        status = RUNTIME_STATUS_SCHEDULED if url else RUNTIME_STATUS_QUEUED

        if not runtime_id:
            raise RuntimeError(f'Invalid API response: missing runtime_id in data: {result_data}')

        # Parse port_mappings from API response
        # Format: {"8023": "http://xxx:32001", "8080": "http://xxx:32002"}
        raw_port_mappings = result_data.get('port_mappings', {})
        port_mappings: dict[int, str] = {}
        for internal_port, external_url in raw_port_mappings.items():
            try:
                port_mappings[int(internal_port)] = external_url
            except (ValueError, TypeError):
                logger.warning('Invalid port mapping key: %s -> %s', internal_port, external_url)

        return runtime_id, url, port_mappings, status

    async def _get_runtime(self, runtime_id: str, log_level: int = logging.DEBUG) -> dict[str, Any]:
        """Get runtime information.

        Args:
            runtime_id: Runtime ID
            log_level: Level for the underlying request/response trace lines (see _call_api).
                Defaults to DEBUG — raw /runtime/get dumps are diagnostic detail; the lifecycle
                lines carry the user-facing state. Watchdog / queue-wait pollers also pass DEBUG
                explicitly to make the intent obvious at the call site.

        Returns:
            Runtime info dict containing status, url and other fields
        """
        data = await self._call_api('/sandbox/v3/runtime/get', {'runtime_id': runtime_id}, log_level=log_level)
        return data.get('data', {})

    async def _stop_runtime(self, runtime_ids: list[str]) -> None:
        """Stop runtimes.

        Args:
            runtime_ids: List of runtime IDs to stop
        """
        payload = {'runtime_ids': runtime_ids, 'batch_id': self.batch_id}
        for attempt in range(1, STOP_RUNTIME_MAX_ATTEMPTS + 1):
            try:
                await self._call_api('/sandbox/v3/runtime/stop', payload)
                return
            except RuntimeError:
                if attempt >= STOP_RUNTIME_MAX_ATTEMPTS:
                    raise
                delay = min(
                    STOP_RUNTIME_RETRY_BASE_SECONDS * (2 ** (attempt - 1)),
                    STOP_RUNTIME_RETRY_MAX_SECONDS,
                )
                logger.warning(
                    'Failed to stop %d runtime(s), retrying in %.1fs (attempt %d/%d)',
                    len(runtime_ids),
                    delay,
                    attempt,
                    STOP_RUNTIME_MAX_ATTEMPTS,
                    exc_info=True,
                )
                await asyncio.sleep(delay)

    async def acquire(self, name: str, **params: Any) -> Runtime:
        """Acquire a new runtime instance.

        Args:
            name: Unique name for this runtime instance
            **params: Runtime parameters:
                - image: Docker image to run (required)
                - runtime_image: Runtime image (optional)
                - environment: Environment variables (optional)
                - runtime_id: Custom runtime ID (optional, auto-generated)
                - exposed_ports: List of additional internal ports to expose (optional)
                - enable_dind: Enable Docker-in-Docker for this runtime (optional, default: False)
                - runtime_disable_aslr: Disable ASLR in the runtime container (optional, default: False).
                  When true, forwarded to the sandbox scheduler and force-routes to suanli.
                - network_policy: Egress network policy dict (optional, overrides provider default).
                  Shape: {'default_action': 'allow'|'deny', 'egress': [{'action': ..., 'target': ...}]}.
                  Non-empty forces backend routing to the suanli provider.
                - cap_add: Extra Linux capabilities list (optional, overrides provider default), e.g.
                  ['NET_ADMIN']. suanli-only; non-empty forces backend routing to the suanli provider.
                - ttl: Sandbox lifetime as a Go duration string (optional, overrides provider default),
                  e.g. '30m' / '168h' ('7d' is not valid Go duration). Pass '' to clear a provider
                  default and fall back to the standard 12h lifetime.
                - resource_spec: Resource specification (optional), may include 'gpu_card_type' (str)

        Note:
            For ``runtime_image``, prefer passing 'current' (or omitting it) so the runtime image
            tracks the installed nexus package version. Hard-coded tags should only be used when
            you need to pin to a specific version.

        Returns:
            Runtime instance (HttpRuntime or SocketIoRuntime) connected to the container
        """
        # Use lock to ensure serialized access for the same name
        if name not in self._acquire_locks:
            self._acquire_locks[name] = asyncio.Lock()

        async with self._acquire_locks[name]:
            return await self._acquire_impl(name, **params)

    async def _acquire_impl(self, name: str, **params: Any) -> Runtime:
        """Internal implementation of acquire."""
        if self.startup_jitter_max > 0:
            jitter = random.uniform(0, self.startup_jitter_max)
            logger.info(
                'Startup jitter: sleeping %.1fs before runtime acquire (max=%.0fs)', jitter, self.startup_jitter_max
            )
            await asyncio.sleep(jitter)

        merged = self.get_params(name, **params)

        # If image is configured at provider level, it takes precedence over configure() settings
        # This ensures provider-level config overrides task-level settings
        image = self.image or merged.get('image')

        if not image:
            raise ValueError('image parameter is required for GongfengRuntimeProvider.acquire()')

        runtime_image = merged.get('runtime_image')
        # 支持 runtime_image: "current"，转为版本化镜像；None 或空则用 provider 默认
        if not runtime_image or runtime_image == 'current':
            runtime_image = self.runtime_image
        environment = merge_managed_otel_env(
            merged.get('environment'),
            process_role='runtime-base',
            service_name='nexus-runtime-base',
        )
        custom_runtime_id = merged.get('runtime_id')
        client_owns_runtime_id = custom_runtime_id is None
        requested_runtime_id = custom_runtime_id or f'nexus-{uuid.uuid4().hex}'
        exposed_ports = merged.get('exposed_ports')
        resource_spec = merged.get('resource_spec')
        working_dir = merged.get('working_dir')
        auto_connect_ws = merged.get('auto_connect_ws', self.auto_connect_ws)
        socketio_request_timeout = merged.get('socketio_request_timeout', self.socketio_request_timeout)
        queue_wait_max = merged.get('queue_wait_max', self.queue_wait_max)
        need_transparent_proxy = merged.get('need_transparent_proxy', self.need_transparent_proxy)
        network_policy = self._validate_network_policy(merged.get('network_policy', self.network_policy))
        cap_add = self._validate_cap_add(merged.get('cap_add', self.cap_add))
        ttl = self._validate_ttl(merged.get('ttl', self.ttl))
        enable_dind = coerce_bool(merged.get('enable_dind', self.enable_dind), name='enable_dind')
        runtime_disable_aslr = coerce_bool(
            merged.get('runtime_disable_aslr', self.runtime_disable_aslr), name='runtime_disable_aslr'
        )

        # Check if runtime with this name already exists
        if name in self._runtime_ids:
            existing_runtime_id = self._runtime_ids[name]
            try:
                runtime_info = await self._get_runtime(existing_runtime_id)
                status = runtime_info.get('status', '')
                url = runtime_info.get('url', '')

                if status == RUNTIME_STATUS_SCHEDULED and url:
                    logger.info("Sandbox '%s': reusing existing runtime (runtime_id=%s)", name, existing_runtime_id)
                    runtime = self._create_runtime(
                        url,
                        auto_connect_ws=auto_connect_ws,
                        socketio_request_timeout=socketio_request_timeout,
                    )
                    self._active_runtimes[name] = runtime
                    return runtime
            except RuntimeError:
                # Runtime may no longer exist, clean up and create new one
                self._runtime_ids.pop(name, None)
                self._active_runtimes.pop(name, None)
            else:
                # Runtime exists but status is not scheduled or url is missing, clean up
                self._runtime_ids.pop(name, None)
                self._active_runtimes.pop(name, None)

        actual_resource_spec = resource_spec or self.resource_spec
        actual_exposed_ports = exposed_ports or self.exposed_ports
        actual_working_dir = working_dir or self.working_dir
        logger.info(
            "Sandbox '%s': creating (image=%s, runtime_image=%s, resource_spec=%s, exposed_ports=%s, working_dir=%s, enable_dind=%s, runtime_disable_aslr=%s, ttl=%s)",
            name,
            image,
            runtime_image,
            actual_resource_spec,
            actual_exposed_ports,
            actual_working_dir,
            enable_dind,
            runtime_disable_aslr,
            ttl or 'backend default',
        )
        try:
            runtime_id, url, port_mappings, status = await self._start_runtime(
                image=image,
                runtime_image=runtime_image,
                environment=environment,
                runtime_id=requested_runtime_id,
                exposed_ports=actual_exposed_ports,
                resource_spec=actual_resource_spec,
                working_dir=actual_working_dir,
                need_transparent_proxy=need_transparent_proxy,
                network_policy=network_policy,
                cap_add=cap_add,
                ttl=ttl,
                enable_dind=enable_dind,
                runtime_disable_aslr=runtime_disable_aslr,
                extra_payload=self._start_payload_extras(merged),
            )
        except asyncio.CancelledError:
            if client_owns_runtime_id:
                await self._stop_ambiguous_start(requested_runtime_id)
            raise
        except Exception:
            if client_owns_runtime_id:
                await self._stop_ambiguous_start(requested_runtime_id)
            raise

        if status == RUNTIME_STATUS_QUEUED:
            # Response-driven: an empty url (status=queued) is the only signal we key off. We do
            # NOT care WHY the backend enqueued (token quota_full_policy=queue, high-load admission
            # throttle, ...) — whenever it enqueues, we poll /runtime/get until the sandbox is
            # scheduled, fails, or the (optional) queue_wait_max budget elapses.
            #
            # Register runtime_id BEFORE waiting so release()/release_all() can cancel the
            # queue entry if the user bails (Ctrl-C, asyncio cancel, explicit release).
            # Do NOT register _active_runtimes yet — there is no Runtime instance to expose,
            # and get_active_runtimes() should not advertise an un-ready entry.
            self._runtime_ids[name] = runtime_id
            # _wait_for_scheduled owns the entire queue-wait narrative (QUEUED entry line,
            # periodic "still waiting Ns" heartbeat, and the promotion line), so we don't
            # duplicate it here.
            url, port_mappings = await self._wait_for_scheduled(name, runtime_id, queue_wait_max)

        # Create runtime client and record it (pass resource_spec for service configuration)
        runtime = self._create_runtime(
            url,
            resource_spec=actual_resource_spec,
            auto_connect_ws=auto_connect_ws,
            socketio_request_timeout=socketio_request_timeout,
        )

        # Populate port_mappings for exposed ports (for WebArena browser access etc.)
        for internal_port, external_url in port_mappings.items():
            runtime.port_mappings[internal_port] = external_url
            logger.info('Port mapping: %s -> %s', internal_port, external_url)

        if self.print_port_mappings:
            with contextlib.suppress(OSError):
                print('\n'.join(build_gongfeng_runtime_port_summary(runtime)), file=sys.stderr)

        self._active_runtimes[name] = runtime
        self._runtime_ids[name] = runtime_id

        # Start background watchdog so we fail fast if the sandbox dies.
        self._start_watchdog(name, runtime, runtime_id)

        logger.info("Sandbox '%s': READY at %s (runtime_id=%s)", name, url, runtime_id)
        return runtime

    async def _stop_ambiguous_start(self, runtime_id: str) -> None:
        timeout = min(max(self.request_timeout, 5.0), 60.0)
        try:
            await asyncio.wait_for(self._stop_runtime([runtime_id]), timeout=timeout)
        except Exception as exc:
            logger.warning(
                'Failed to stop runtime %s after an ambiguous start failure within %.1fs: %s',
                runtime_id,
                timeout,
                exc,
            )

    @staticmethod
    def _next_queue_delay() -> float:
        """Return the next sleep duration for queue polling, jittered ±QUEUE_POLL_JITTER."""
        return QUEUE_POLL_INTERVAL * random.uniform(1.0 - QUEUE_POLL_JITTER, 1.0 + QUEUE_POLL_JITTER)

    async def _wait_for_scheduled(
        self, name: str, runtime_id: str, queue_wait_max: float
    ) -> tuple[str, dict[int, str]]:
        """Poll /runtime/get until the queued sandbox becomes scheduled, or fail.

        Polling is driven by QUEUE_POLL_INTERVAL with ±25% jitter. The per-poll /runtime/get
        calls are logged at DEBUG (silent by default) so a long queue wait doesn't spam the log
        with identical bodies; instead this method emits a small set of human-readable lifecycle
        lines at INFO — a QUEUED entry line, a periodic "still queued, waited Ns" heartbeat, and
        the final promotion line — so a user can always tell the sandbox is queued-and-waiting
        (normal under load) rather than stuck. Transient errors from /runtime/get (network blips,
        5xx) are logged and tolerated — they count toward the deadline but do not abort the wait.
        Cancellation (CancelledError from the event loop) triggers a best-effort /runtime/stop to
        cancel the queue entry before re-raising, so we never leak rows in the backend's
        runtime_queue table.

        Args:
            name: Human-readable runtime name, used to make the lifecycle log lines greppable.
            runtime_id: The runtime ID returned by /runtime/start with status=queued.
            queue_wait_max: Total wall-clock budget in seconds. <= 0 means "no client-side
                deadline" — poll until the sandbox is scheduled or the backend gives up.

        Returns:
            (url, port_mappings) once status=scheduled is observed.

        Raises:
            RuntimeQueueWaitTimeoutError: a positive queue_wait_max elapsed while still queued.
            RuntimeQueueFailedError: backend transitioned to stopped/not_found (with or
                without queue_timeout/queue_give_up remark); backend success path is
                queued → scheduled, so any stopped during the wait is terminal.
        """
        # queue_wait_max <= 0 means "no client-side deadline": keep polling until the backend
        # schedules the sandbox or gives up on it (status -> stopped/not_found). Only a positive
        # budget arms the wait-timeout branch below.
        unbounded = queue_wait_max <= 0
        started = time.monotonic()
        deadline = started + queue_wait_max
        wait_desc = 'no deadline' if unbounded else f'up to {queue_wait_max:.0f}s'
        logger.info(
            "Sandbox '%s': QUEUED, waiting for backend resources (runtime_id=%s, %s). "
            'This is normal under load — the sandbox is not stuck; polling until it starts running.',
            name,
            runtime_id,
            wait_desc,
        )
        next_progress = started + QUEUE_PROGRESS_LOG_INTERVAL
        try:
            while True:
                try:
                    info = await self._get_runtime(runtime_id, log_level=logging.DEBUG)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    # Transient API error (network blip / 5xx / parse failure). The error has
                    # already been logged inside _call_api with full context; we only swallow
                    # it here so a single bad poll doesn't abandon the wait. The error still
                    # counts toward the deadline via the sleep below.
                    logger.warning("Sandbox '%s': queue poll failed transiently, will retry: %s", name, exc)
                    info = {}

                status = info.get('status', '')
                if status == RUNTIME_STATUS_SCHEDULED:
                    url = info.get('url') or ''
                    if not url:
                        # Should not happen per backend contract (scheduled implies url),
                        # but if it does we keep polling rather than crash — the next poll
                        # will likely see the url populated.
                        logger.warning("Sandbox '%s': reported status=scheduled with empty url, continuing poll", name)
                    else:
                        raw_port_mappings = info.get('port_mappings') or {}
                        port_mappings: dict[int, str] = {}
                        for internal_port, external_url in raw_port_mappings.items():
                            try:
                                port_mappings[int(internal_port)] = external_url
                            except (ValueError, TypeError):
                                logger.warning('Invalid port mapping key: %s -> %s', internal_port, external_url)
                        logger.info(
                            "Sandbox '%s': started after queuing %.0fs (now running)",
                            name,
                            time.monotonic() - started,
                        )
                        return url, port_mappings
                elif status in WATCHDOG_DEAD_STATUSES:
                    # Backend gave up on us. Reading `remark` lets the caller distinguish
                    # queue_timeout (waited 48h, never scheduled) from queue_give_up
                    # (worker tried to start and failed repeatedly).
                    remark = info.get('remark') or ''
                    is_queue_failure = any(remark.startswith(p) for p in QUEUE_FAILURE_REMARK_PREFIXES)
                    logger.warning(
                        "Sandbox '%s': QUEUE FAILED — backend gave up (status=%s, remark=%s) after %.0fs",
                        name,
                        status,
                        remark,
                        time.monotonic() - started,
                    )
                    raise RuntimeQueueFailedError(
                        f'sandbox {runtime_id} (name={name!r}) transitioned to status={status!r} during queue '
                        f'wait (remark={remark!r}, queue_failure={is_queue_failure})'
                    )
                # else: status=queued, empty, or unknown intermediate — keep polling.

                if not unbounded and time.monotonic() >= deadline:
                    # Budget exhausted. Best-effort cancel the queue entry so it doesn't sit
                    # in the backend's runtime_queue table forever (well, until 48h auto-purge).
                    logger.info(
                        "Sandbox '%s': queue_wait_max (%.0fs) exhausted; cancelling queue entry",
                        name,
                        queue_wait_max,
                    )
                    try:
                        await self._stop_runtime([runtime_id])
                    except Exception as stop_exc:
                        logger.warning(
                            "Sandbox '%s': failed to cancel queued runtime after wait timeout: %s", name, stop_exc
                        )
                    raise RuntimeQueueWaitTimeoutError(
                        f'sandbox {runtime_id} (name={name!r}) still queued after queue_wait_max={queue_wait_max}s; '
                        f'queue entry has been cancelled'
                    )

                # Periodic reassurance heartbeat (the per-poll get is DEBUG/silent). Advance past
                # every elapsed multiple so a slow poll cycle doesn't fire several lines at once.
                now = time.monotonic()
                if now >= next_progress:
                    logger.info(
                        "Sandbox '%s': still QUEUED, waited %.0fs so far (not stuck — waiting for resources)",
                        name,
                        now - started,
                    )
                    while next_progress <= now:
                        next_progress += QUEUE_PROGRESS_LOG_INTERVAL

                await asyncio.sleep(self._next_queue_delay())
        except asyncio.CancelledError:
            # Caller bailed (Ctrl-C, asyncio task cancel). Cancel the backend queue entry
            # so we don't leak rows. Suppress errors during cleanup — cancellation should
            # propagate regardless of cleanup success.
            logger.info("Sandbox '%s': queue wait cancelled by caller; cancelling queue entry", name)
            try:
                await self._stop_runtime([runtime_id])
            except Exception as stop_exc:
                logger.warning(
                    "Sandbox '%s': failed to cancel queued runtime during cancellation cleanup: %s", name, stop_exc
                )
            raise

    def _start_watchdog(self, name: str, runtime: Runtime, runtime_id: str) -> None:
        """Start a background task that polls runtime state and marks it unhealthy on death."""
        existing = self._watchdog_tasks.pop(name, None)
        if existing is not None and not existing.done():
            existing.cancel()
        task = create_detached_task(
            self._watchdog_loop(name, runtime, runtime_id),
            name=f'gongfeng-watchdog[{name}:{runtime_id}]',
        )
        self._watchdog_tasks[name] = task

    def _next_watchdog_delay(self) -> float:
        """Return the next sleep duration, jittered ±WATCHDOG_POLL_JITTER around the interval."""
        base = self.watchdog_poll_interval
        return base * random.uniform(1.0 - WATCHDOG_POLL_JITTER, 1.0 + WATCHDOG_POLL_JITTER)

    async def _watchdog_loop(self, name: str, runtime: Runtime, runtime_id: str) -> None:
        """Probe the sandbox's real health via the proxy and mark ``runtime`` unhealthy on death.

        Probes ``GET {base_url}/health`` (proxied to the nexus server inside the pod) rather than
        the control-plane ``/runtime/get``, whose status is a stale DB snapshot that never reflects
        a pod that OOM'd / crashed / wedged. When the server is dead the proxy fails to forward and
        returns a non-2xx body carrying ``detail`` ∈ :data:`UNHEALTHY_DETAILS` (proxy-confirmed
        death → kill immediately). Inconclusive failures (bare timeout / connection refused /
        proxy's own 5xx) only kill after :data:`WATCHDOG_HEALTH_FAILURE_THRESHOLD` consecutive
        misses, so a proxy blip doesn't take down a live task.
        """
        # One-shot liveness contract at INFO: tell the user the sentinel is in place and that
        # silence == healthy (routine probes below are DEBUG; only a death logs at WARNING), so a
        # long quiet stretch isn't mistaken for a hang. Emitted before the initial jittered sleep
        # so it appears right after the sandbox is READY, not up to one poll interval later.
        logger.info(
            "Sandbox '%s': health watchdog started (probing %s every ~%.0fs; logs only on failure)",
            name,
            NEXUS_HEALTH_PATH,
            self.watchdog_poll_interval,
        )
        # First probe is jittered across [0, interval] so 8k watchdogs don't all fire
        # their first request at the same wall-clock moment.
        await asyncio.sleep(random.uniform(0, self.watchdog_poll_interval))

        base_url = runtime.base_url
        if not base_url:
            # Should not happen — RemoteRuntime subclasses always carry a base_url. Bail out of
            # the watchdog rather than crash-loop; the runtime stays usable, just unwatched.
            logger.warning(
                "Sandbox '%s' (%s): watchdog has no base_url to probe; health watchdog disabled", name, runtime_id
            )
            return
        health_url = f'{base_url.rstrip("/")}{NEXUS_HEALTH_PATH}'
        health_timeout = aiohttp.ClientTimeout(total=WATCHDOG_HEALTH_TIMEOUT)

        consecutive_failures = 0
        try:
            while not runtime.is_unhealthy:
                try:
                    resp = await http_client.get(health_url, timeout=health_timeout)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    # Inconclusive failure (timeout / connection refused / DNS). Could be a wedged
                    # server OR a transient network/proxy blip — only kill after a streak.
                    consecutive_failures += 1
                    logger.debug(
                        'Watchdog %s: /health probe failed (%d/%d): %s',
                        name,
                        consecutive_failures,
                        WATCHDOG_HEALTH_FAILURE_THRESHOLD,
                        exc,
                    )
                    if consecutive_failures >= WATCHDOG_HEALTH_FAILURE_THRESHOLD:
                        runtime.mark_unhealthy(
                            f"Sandbox '{name}' ({runtime_id}) DIED: {NEXUS_HEALTH_PATH} unreachable for "
                            f'{consecutive_failures} consecutive probes ({exc!r}); '
                            f'marking unhealthy so in-flight tasks fail fast'
                        )
                        return
                    await asyncio.sleep(self._next_watchdog_delay())
                    continue

                if 200 <= resp.status_code < 300:
                    # Server answered — alive. Reset the streak so isolated earlier blips don't
                    # accumulate toward a false death.
                    consecutive_failures = 0
                    await asyncio.sleep(self._next_watchdog_delay())
                    continue

                # Non-2xx: the proxy reached us but couldn't serve. Read the proxy's diagnostic
                # `detail` — a known reason is a confirmed death and kills immediately; anything
                # else (proxy's own 5xx, unexpected body) is inconclusive and counts toward the
                # streak like a transient failure. `exit_code` (when present) is the container's
                # K8s ContainerStateTerminated.ExitCode — 137=OOM, 143=SIGKILL, other non-zero=
                # crash, 0=no exit-code info (e.g. Waiting-phase error) — surfaced into the death
                # reason purely for diagnosis; it does not change the kill decision.
                detail = ''
                exit_code = None
                with contextlib.suppress(Exception):
                    body = resp.json() or {}
                    detail = body.get('detail') or ''
                    exit_code = body.get('exit_code')
                exit_desc = f', exit_code={exit_code}' if exit_code else ''
                if detail in UNHEALTHY_DETAILS:
                    runtime.mark_unhealthy(
                        f"Sandbox '{name}' ({runtime_id}) DIED: proxy reported detail={detail!r} "
                        f'(status={resp.status_code}{exit_desc}); marking unhealthy so in-flight tasks fail fast'
                    )
                    return

                consecutive_failures += 1
                logger.debug(
                    'Watchdog %s: /health returned status=%s detail=%r exit_code=%s (%d/%d)',
                    name,
                    resp.status_code,
                    detail,
                    exit_code,
                    consecutive_failures,
                    WATCHDOG_HEALTH_FAILURE_THRESHOLD,
                )
                if consecutive_failures >= WATCHDOG_HEALTH_FAILURE_THRESHOLD:
                    runtime.mark_unhealthy(
                        f"Sandbox '{name}' ({runtime_id}) DIED: {NEXUS_HEALTH_PATH} returned "
                        f'status={resp.status_code}{exit_desc} for {consecutive_failures} consecutive probes; '
                        f'marking unhealthy so in-flight tasks fail fast'
                    )
                    return
                await asyncio.sleep(self._next_watchdog_delay())
        except asyncio.CancelledError:
            logger.debug('Watchdog for %s cancelled', name)
            raise
        except Exception:
            logger.exception('Watchdog for %s crashed', name)

    async def _release_impl(self, name: str) -> None:
        """Release a runtime instance.

        Args:
            name: Name of the runtime instance to release
        """
        if name not in self._runtime_ids:
            logger.warning('Runtime %s not found in tracking, skipping release', name)
            return

        await self._stop_watchdog(name)

        # Explicitly disconnect Socket.IO client to avoid CLOSE_WAIT connections
        if name in self._active_runtimes:
            runtime = self._active_runtimes[name]
            try:
                await runtime.ws_disconnect()
                logger.debug('Socket.IO disconnected for runtime %s', name)
            except asyncio.CancelledError as e:
                logger.debug('Failed to disconnect Socket.IO for runtime %s: %s', name, e)
            except Exception as e:
                logger.debug('Failed to disconnect Socket.IO for runtime %s: %s', name, e)

        runtime_id = self._runtime_ids[name]
        logger.info("Sandbox '%s': stopping (runtime_id=%s)...", name, runtime_id)

        await self._stop_runtime([runtime_id])

        self._runtime_ids.pop(name, None)
        self._active_runtimes.pop(name, None)
        logger.info("Sandbox '%s': released", name)

    async def _stop_watchdog(self, name: str) -> None:
        """Cancel and await the watchdog task for a runtime, if running."""
        task = self._watchdog_tasks.pop(name, None)
        if task is None or task.done():
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.debug('Watchdog %s shutdown error: %s', name, e)

    async def _release_all_impl(self) -> None:
        """Release all runtime instances.
        Batch stop all running runtimes, more efficient than releasing one by one.
        """
        if not self._runtime_ids:
            return

        # Cancel all watchdog tasks before stopping runtimes.
        for name in list(self._watchdog_tasks.keys()):
            await self._stop_watchdog(name)

        # Explicitly disconnect all Socket.IO clients to avoid CLOSE_WAIT connections
        for name, runtime in list(self._active_runtimes.items()):
            try:
                await runtime.ws_disconnect()
                logger.debug('Socket.IO disconnected for runtime %s', name)
            except Exception as e:
                logger.debug('Failed to disconnect Socket.IO for runtime %s: %s', name, e)

        runtime_ids = list(self._runtime_ids.values())
        logger.info('Stopping all runtimes: %s', runtime_ids)

        await self._stop_runtime(runtime_ids)

        self._runtime_ids.clear()
        self._active_runtimes.clear()
        logger.info('All runtimes released')

    def get_runtime_id(self, name: str) -> str | None:
        """Return the Gongfeng-issued runtime ID (e.g. ``nexus-XXXX``) for ``name``.

        Returns ``None`` when no active runtime is registered under ``name``.
        """
        return self._runtime_ids.get(name)

    async def get_runtime_info(self, name: str) -> dict[str, Any] | None:
        """Get detailed information for a specific runtime.

        Args:
            name: Runtime name

        Returns:
            Runtime info dict, or None if not found
        """
        if name not in self._runtime_ids:
            return None

        runtime_id = self._runtime_ids[name]
        try:
            return await self._get_runtime(runtime_id)
        except RuntimeError:
            return None

    async def list_runtimes(
        self,
        runtime_ids: list[str] | None = None,
        batch_id: str | None = None,
        page: int = 1,
        per_page: int = 10,
    ) -> tuple[list[dict[str, Any]], int]:
        """List runtimes.

        Args:
            runtime_ids: List of runtime IDs to query (optional)
            batch_id: Batch identifier to query (optional)
            page: Page number
            per_page: Items per page

        Returns:
            (runtime list, total count) tuple
        """
        payload: dict[str, Any] = {
            'page': page,
            'per_page': per_page,
        }
        if runtime_ids:
            payload['runtime_ids'] = runtime_ids
        if batch_id:
            payload['batch_id'] = batch_id

        data = await self._call_api('/sandbox/v3/runtime/list', payload)
        result_data = data.get('data', {})
        return result_data.get('runtimes', []), result_data.get('total', 0)


__all__ = [
    'GongfengRuntimeProvider',
    'RUNTIME_STATUS_NOT_FOUND',
    'RUNTIME_STATUS_QUEUED',
    'RUNTIME_STATUS_SCHEDULED',
    'RUNTIME_STATUS_STOPPED',
    'RuntimeQueueFailedError',
    'RuntimeQueueWaitTimeoutError',
    'build_gongfeng_runtime_port_summary',
]
