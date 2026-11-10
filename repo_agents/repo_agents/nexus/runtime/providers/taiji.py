"""Taiji-backed RuntimeProvider for remote sandbox execution.

Thin specialization of :class:`~nexus.runtime.providers.gongfeng.GongfengRuntimeProvider`: it
calls the same sandbox v3 API (``/sandbox/v3/runtime/start|get|stop``) and reuses its queue
wait, health watchdog, timeout diagnostics and credential redaction wholesale. Differences:

- ``runtime_provider`` is pinned to ``taiji`` so the backend schedules onto the Taiji pool.
- ``resource_spec.gpu_card_type`` is required and must be a Taiji ``gpuType`` enum value.
  The Start payload still uses sandbox v3 field names (``gpu`` / ``gpu_card_type``); the
  backend maps them to CreateServing ``gpuNum`` / ``gpuType``.
- ``ttl`` uses the same Go duration grammar as Gongfeng, but the backend cap is 48h
  (not 168h). Omitted ttl is not sent; the backend persists 48h.
- ``taiji_app_group`` / ``taiji_app_group_location`` are required and must be set as a pair.
- ``purpose`` is recommended (``prod`` / ``eval`` / ``test``). The backend stores it for
  audit; Taiji does not use it for token-quota checks. Omitted purpose is not sent.
- ``token`` is required (same resolution as Gongfeng: explicit / ``GONGFENG_TOKEN`` /
  default file). Nexus only checks that a non-empty token can be resolved; it does not
  query or validate token-side CPU/GPU quota.

Capability fields Gongfeng already forwards (``enable_dind``, ``cap_add``, ``ttl``,
``need_transparent_proxy``, ``network_policy``, ``runtime_disable_aslr``) are accepted.
The backend keeps an explicit ``runtime_provider=taiji`` on Taiji and maps them to
CreateServing ``sandboxConfig`` / ASLR annotation.
"""

# /// script
# dependencies = []
#
# [project]
# name = "taiji"
# authors = [
#     { name = "baynejiang" },
# ]
#
# [tool.nexus]
# runtime_providers = { taiji = "nexus.runtime.providers.taiji" }
# ///

from __future__ import annotations

from typing import Any

from nexus.runtime.client.base import Runtime
from nexus.runtime.providers.gongfeng import (
    _GO_DURATION_TERM_RE,
    _GO_DURATION_UNITS,
    GongfengRuntimeProvider,
)
from nexus.runtime.providers.registry import runtime_provider_registry


# Value sent as the start request's ``runtime_provider`` field. This is what makes the backend
# schedule onto Taiji instead of gongfeng / suanli.
BACKEND_PROVIDER_NAME = 'taiji'

# Mirrors sandbox do.MaxTaijiSandboxTTL (48h). Parent MAX_TTL_SECONDS is 168h for suanli/gongfeng.
MAX_TAIJI_TTL_SECONDS = 48 * 3600

# Taiji CreateServing resourceConfig.gpuType enum. Sandbox v3 keeps the field name gpu_card_type.
TAIJI_GPU_CARD_TYPES: frozenset[str] = frozenset({
    'P40',
    'M40',
    'V100',
    'EF_T10',
    'T4',
    'P4',
    'ASCEND910B',
    'A100',
    'A100PRO',
    'A800',
    'H800',
    'H20',
    'L20',
})
_TAIJI_GPU_CARD_TYPE_LIST = ', '.join(sorted(TAIJI_GPU_CARD_TYPES))


def _gpu_card_type(spec: Any) -> str:
    if not isinstance(spec, dict):
        return ''
    raw = spec.get('gpu_card_type')
    if raw is None:
        return ''
    return str(raw).strip()


def _reject_gpu_card_type(spec: Any, *, where: str, required: bool) -> None:
    """Reject missing / unknown gpu_card_type. Does not validate gpu count."""
    if spec is None:
        if required:
            raise ValueError(
                f'TaijiRuntimeProvider requires resource_spec.gpu_card_type (passed via {where}). '
                f'Allowed values: {_TAIJI_GPU_CARD_TYPE_LIST}.'
            )
        return
    card = _gpu_card_type(spec)
    if not card:
        raise ValueError(
            f'TaijiRuntimeProvider requires resource_spec.gpu_card_type (passed via {where}). '
            f'Allowed values: {_TAIJI_GPU_CARD_TYPE_LIST}.'
        )
    if card not in TAIJI_GPU_CARD_TYPES:
        raise ValueError(
            f'TaijiRuntimeProvider resource_spec.gpu_card_type={card!r} is not a Taiji gpuType '
            f'(passed via {where}). Allowed values: {_TAIJI_GPU_CARD_TYPE_LIST}.'
        )


def _trim_text(value: Any) -> str:
    if value is None:
        return ''
    return str(value).strip()


def _validate_app_group_pair(app_group: str, location: str, *, where: str) -> None:
    if not app_group or not location:
        raise ValueError(f'taiji_app_group and taiji_app_group_location are required (passed via {where})')


def _reject_partial_app_group_override(params: dict[str, Any], *, where: str) -> None:
    if ('taiji_app_group' in params) != ('taiji_app_group_location' in params):
        raise ValueError(f'taiji_app_group and taiji_app_group_location are required (passed via {where})')


def _reject_unsupported(params: dict[str, Any], *, where: str) -> None:
    """Raise if ``params`` pins a conflicting target pool or an invalid GPU spec."""
    # Checked on every entry point, not just __init__: a gongfeng config carried over intact keeps
    # its runtime_provider line, and the parent quietly ignores the key outside __init__, so the
    # caller would believe they redirected the sandbox while it still lands on Taiji.
    requested = params.get('runtime_provider')
    if requested and requested != BACKEND_PROVIDER_NAME:
        raise ValueError(
            f'TaijiRuntimeProvider pins runtime_provider={BACKEND_PROVIDER_NAME!r}, got {requested!r} '
            f'(passed via {where}). Drop runtime_provider, or set runtime_provider_name: gongfeng to '
            f'schedule onto {requested!r} instead.'
        )

    if 'resource_spec' in params:
        _reject_gpu_card_type(params.get('resource_spec'), where=where, required=True)


@runtime_provider_registry.register('taiji')
class TaijiRuntimeProvider(GongfengRuntimeProvider):
    """RuntimeProvider that schedules sandboxes onto the Taiji resource pool.

    Same API, lifecycle and watchdog as :class:`GongfengRuntimeProvider`; only the target pool
    and a few Taiji-specific constraints differ.

    Supported config and acquire params (semantics identical to the gongfeng provider unless noted):
        api_base_url, token, image, runtime_image, python_version, resource_spec,
        exposed_ports, working_dir, environment, runtime_id, batch_id, purpose,
        request_timeout, queue_wait_max, watchdog_poll_interval, auto_connect_ws,
        socketio_request_timeout, rpc_protocol, print_port_mappings, startup_jitter_max,
        enable_dind, runtime_disable_aslr, cap_add, ttl, need_transparent_proxy, network_policy,
        taiji_app_group, taiji_app_group_location

    ``token`` is required. Resolution matches the parent; token-side quota is not checked.

    ``resource_spec.gpu_card_type`` is required for Taiji and must be one of
    ``TAIJI_GPU_CARD_TYPES``. Use sandbox v3 names (``gpu`` / ``gpu_card_type``), not
    CreateServing names (``gpuNum`` / ``gpuType``).

    ``ttl`` omits the field when unset (backend persists 48h) and rejects anything above 48h.

    ``taiji_app_group`` and ``taiji_app_group_location`` are required and must be set as a pair.

    Rejected params:
        runtime_provider — pinned to 'taiji' by this class
    """

    def __init__(self, api_base_url: str, **config: Any) -> None:
        _reject_unsupported(config, where='TaijiRuntimeProvider(...)')
        app_group = _trim_text(config.pop('taiji_app_group', ''))
        location = _trim_text(config.pop('taiji_app_group_location', ''))
        _validate_app_group_pair(app_group, location, where='TaijiRuntimeProvider(...)')
        config.pop('runtime_provider', None)
        super().__init__(api_base_url=api_base_url, runtime_provider=BACKEND_PROVIDER_NAME, **config)
        self.taiji_app_group = app_group
        self.taiji_app_group_location = location

    def configure(self, name: str, **params: Any) -> None:
        """Pre-configure a runtime, still pinning the target pool and GPU card type.

        Guarded separately from ``acquire`` because ``runtime_configs.<name>`` in a YAML config
        lands here, which is the likeliest place for a gongfeng config to be copied over intact.
        """
        _reject_unsupported(params, where=f'runtime_configs.{name}')
        _reject_partial_app_group_override(params, where=f'runtime_configs.{name}')
        app_group, location = self._resolve_app_group_pair(params)
        _validate_app_group_pair(app_group, location, where=f'runtime_configs.{name}')
        super().configure(name, **params)

    async def acquire(self, name: str, **params: Any) -> Runtime:
        _reject_unsupported(params, where=f'acquire({name!r})')
        merged = self.get_params(name, **params)
        spec = merged.get('resource_spec') or self.resource_spec
        _reject_gpu_card_type(spec, where=f'acquire({name!r})', required=True)
        _reject_partial_app_group_override(params, where=f'acquire({name!r})')
        app_group, location = self._resolve_app_group_pair(merged)
        _validate_app_group_pair(app_group, location, where=f'acquire({name!r})')
        return await super().acquire(name, **params)

    def _resolve_app_group_pair(self, params: dict[str, Any]) -> tuple[str, str]:
        app_group = _trim_text(params['taiji_app_group']) if 'taiji_app_group' in params else self.taiji_app_group
        location = (
            _trim_text(params['taiji_app_group_location'])
            if 'taiji_app_group_location' in params
            else self.taiji_app_group_location
        )
        return app_group, location

    def _start_payload_extras(self, merged: dict[str, Any]) -> dict[str, Any]:
        app_group, location = self._resolve_app_group_pair(merged)
        _validate_app_group_pair(app_group, location, where='acquire')
        return {
            'taiji_app_group': app_group,
            'taiji_app_group_location': location,
        }

    @staticmethod
    def _validate_ttl(ttl: str | None) -> str | None:
        """Same Go duration grammar as Gongfeng, capped at the Taiji backend's 48h."""
        validated = GongfengRuntimeProvider._validate_ttl(ttl)
        if validated is None:
            return None
        seconds = sum(
            float(value) * _GO_DURATION_UNITS[unit] for value, unit in _GO_DURATION_TERM_RE.findall(validated)
        )
        if seconds > MAX_TAIJI_TTL_SECONDS:
            raise ValueError(f'taiji ttl must not exceed 48h, got {validated!r}')
        return validated


__all__ = [
    'BACKEND_PROVIDER_NAME',
    'MAX_TAIJI_TTL_SECONDS',
    'TAIJI_GPU_CARD_TYPES',
    'TaijiRuntimeProvider',
]
