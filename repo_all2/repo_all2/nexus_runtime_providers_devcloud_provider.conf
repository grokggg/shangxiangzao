"""DevCloud macOS RuntimeProvider."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import random
import time

from collections.abc import Mapping
from typing import Any, cast

import aiohttp

from nexus.runtime.providers.base import InvalidPersistentRuntimeError, RuntimeProvider
from nexus.runtime.providers.devcloud.client import DevCloudRuntime
from nexus.runtime.providers.registry import runtime_provider_registry
from nexus.utils.async_http import http_client


logger = logging.getLogger(__name__)


_FAILED_STATUSES = {'failed', 'error', 'canceled', 'cancelled'}


@runtime_provider_registry.register('devcloud')
class DevCloudRuntimeProvider(RuntimeProvider):
    """Provider for DevCloud macOS OSWorld sandboxes."""

    DEFAULT_MODEL = 'VMware'
    DEFAULT_LEASE_TIME = 300
    DEFAULT_SERVER_PORT = 5000
    DEFAULT_REQUEST_TIMEOUT = 1800.0
    DEFAULT_INSTANCE_READY_TIMEOUT = 300.0
    TASK_POLL_INTERVAL = 2.0

    def __init__(
        self,
        *,
        app_id: str | None = None,
        app_token: str | None = None,
        sandbox_api_base_url: str,
        control_api_base_url: str,
        project: str,
        pipeline_id: str,
        build_id: str,
        vm_seq_id: str,
        model: str = DEFAULT_MODEL,
        lease_time: int = DEFAULT_LEASE_TIME,
        server_port: int = DEFAULT_SERVER_PORT,
        request_timeout: float = DEFAULT_REQUEST_TIMEOUT,
        instance_ready_timeout: float = DEFAULT_INSTANCE_READY_TIMEOUT,
        **config: Any,
    ) -> None:
        super().__init__(**config)
        self.app_id = app_id
        self.app_token = app_token

        self.sandbox_api_base_url = sandbox_api_base_url.rstrip('/')
        self.control_api_base_url = control_api_base_url.rstrip('/')
        self.project = project
        self.pipeline_id = pipeline_id
        self.build_id = build_id
        self.vm_seq_id = vm_seq_id
        self.model = model
        self.lease_time = int(lease_time)
        self.server_port = int(server_port)
        self.request_timeout = float(request_timeout)
        self.instance_ready_timeout = float(instance_ready_timeout)

        self._task_ids: dict[str, str] = {}
        self._vm_ids: dict[str, str] = {}
        self._identity: dict[str, dict[str, Any]] = {}
        self._metadata: dict[str, dict[str, Any]] = {}
        self._external_instance_names: set[str] = set()
        self._acquire_locks: dict[str, asyncio.Lock] = {}

    def _build_auth_headers(self, *, timestamp: str | None = None, random_digits: str | None = None) -> dict[str, str]:
        resolved_timestamp = timestamp or str(int(time.time()))
        resolved_random = random_digits or ''.join(str(random.randint(0, 9)) for _ in range(8))
        raw = f'{self.app_token}{resolved_timestamp}{resolved_random}'
        enckey = hashlib.md5(raw.encode('utf-8')).hexdigest().lower()
        return {
            'APPID': str(self.app_id),
            'TIMESTP': resolved_timestamp,
            'RANDOM': resolved_random,
            'ENCKEY': enckey,
        }

    def get_runtime_id(self, name: str) -> str | None:
        return self._vm_ids.get(name) or self._task_ids.get(name)

    async def _call_api(self, base_url: str, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f'{base_url.rstrip("/")}{path}'
        timeout = aiohttp.ClientTimeout(total=self.request_timeout)
        resp = await http_client.post(url, json=payload, headers=self._build_auth_headers(), timeout=timeout)
        if resp.status_code < 200 or resp.status_code >= 300:
            raise RuntimeError(f'DevCloud HTTP error {resp.status_code} for {path}: {resp.text[:512]}')
        try:
            data = resp.json()
        except Exception as exc:
            raise RuntimeError(f'DevCloud returned invalid JSON for {path}: {resp.text[:512]}') from exc
        if data.get('actionCode') != 200:
            raise RuntimeError(
                f'DevCloud API error for {path}: actionCode={data.get("actionCode")} '
                f'actionMessage={data.get("actionMessage", "")}'
            )
        return data

    async def acquire(self, name: str, **params: Any) -> DevCloudRuntime:
        if name not in self._acquire_locks:
            self._acquire_locks[name] = asyncio.Lock()
        async with self._acquire_locks[name]:
            return await self._acquire_impl(name, **params)

    async def _acquire_impl(self, name: str, **params: Any) -> DevCloudRuntime:
        if name in self._active_runtimes:
            return cast(DevCloudRuntime, self._active_runtimes[name])

        merged = self.get_params(name, **params)
        runtime_credential = merged.get('runtime_credential')
        instance_ip = ''
        server_port = int(merged.get('server_port', self.server_port))
        if runtime_credential is not None:
            if not isinstance(runtime_credential, Mapping):
                raise InvalidPersistentRuntimeError('DevCloud runtime_credential must be a mapping')
            instance_ip = str(runtime_credential.get('instance_ip') or '').strip()
            if not instance_ip:
                raise InvalidPersistentRuntimeError('DevCloud runtime_credential requires instance_ip')
            server_port = int(runtime_credential.get('server_port', server_port))
        elif merged.get('instance_ip') is not None:
            raise ValueError('DevCloud instance_ip is not supported; use runtime_credential.instance_ip')
        instance_ready_timeout = float(merged.get('instance_ready_timeout', self.instance_ready_timeout))
        if instance_ip:
            base_url = f'http://{instance_ip}:{server_port}'
            await self._wait_for_osworld_ready(base_url, instance_ready_timeout)
            runtime = DevCloudRuntime(
                base_url=base_url,
                request_timeout=self.request_timeout,
                metadata={'ip': instance_ip, 'reused': True},
            )
            self._active_runtimes[name] = runtime
            self._external_instance_names.add(name)
            logger.info("Reused DevCloud runtime '%s' at %s", name, base_url)
            return runtime

        if not self.app_id:
            raise ValueError('app_id is required for DevCloudRuntimeProvider when creating an instance')
        if not self.app_token:
            raise ValueError('app_token is required for DevCloudRuntimeProvider when creating an instance')

        identity = {
            'project': merged.get('project', self.project),
            'pipelineId': merged.get('pipeline_id', self.pipeline_id),
            'buildId': merged.get('build_id', self.build_id),
            'vmSeqId': merged.get('vm_seq_id', self.vm_seq_id),
        }
        self._identity[name] = identity
        model = merged.get('model', self.model)
        lease_time = int(merged.get('lease_time', self.lease_time))

        create_payload = {**identity, 'env': {'DEVCLOUD_MODEL': model}, 'leaseTime': lease_time}
        data = await self._call_api(self.sandbox_api_base_url, '/mac/sandboxCreate', create_payload)
        task_id = str((data.get('data') or {}).get('taskId') or '')
        if not task_id:
            raise RuntimeError(f'DevCloud create response missing taskId: {data}')
        self._task_ids[name] = task_id

        try:
            result = await self._wait_for_task(task_id)
            ip = str(result.get('ip') or '')
            if not ip:
                raise RuntimeError(f'DevCloud task {task_id} succeeded without ip: {result}')
            vm_id = result.get('id')
            if vm_id is not None:
                self._vm_ids[name] = str(vm_id)
            self._metadata[name] = result

            base_url = f'http://{ip}:{server_port}'
            await self._wait_for_osworld_ready(base_url, instance_ready_timeout)
            runtime = DevCloudRuntime(
                base_url=base_url,
                request_timeout=self.request_timeout,
                metadata={**result, 'taskId': task_id},
            )
            self._active_runtimes[name] = runtime
            logger.info("DevCloud '%s': READY at %s metadata=%s", name, base_url, result)
            return runtime
        except Exception:
            await self._best_effort_delete(name)
            raise

    async def _wait_for_task(self, task_id: str) -> dict[str, Any]:
        deadline = asyncio.get_running_loop().time() + self.request_timeout
        while True:
            data = await self._call_api(self.control_api_base_url, '/mac/task/result', {'taskId': task_id})
            result = data.get('data') or {}
            status = str(result.get('status') or '').lower()
            if status == 'succeeded':
                return result
            if status in _FAILED_STATUSES:
                raise RuntimeError(f'DevCloud task {task_id} failed with status={status}: {result}')
            if asyncio.get_running_loop().time() >= deadline:
                raise TimeoutError(f'DevCloud task {task_id} did not finish within {self.request_timeout}s')
            await asyncio.sleep(self.TASK_POLL_INTERVAL)

    async def _wait_for_osworld_ready(self, base_url: str, timeout_seconds: float) -> None:
        timeout = aiohttp.ClientTimeout(total=10)
        deadline = asyncio.get_running_loop().time() + timeout_seconds
        last_error = ''
        while True:
            try:
                resp = await http_client.get(f'{base_url}/health', timeout=timeout)
                if resp.status_code == 200 and resp.content:
                    return
                last_error = f'HTTP {resp.status_code}: {resp.text[:200]}'
            except Exception as exc:
                last_error = f'{type(exc).__name__}: {exc}'
            if asyncio.get_running_loop().time() >= deadline:
                raise TimeoutError(f'DevCloud OSWorld runtime not ready at {base_url}/health: {last_error}')
            await asyncio.sleep(self.TASK_POLL_INTERVAL)

    async def _release_impl(self, name: str) -> None:
        self._active_runtimes.pop(name, None)
        if name in self._external_instance_names:
            self._external_instance_names.discard(name)
            logger.info("Detached reused DevCloud runtime '%s' without deleting its instance", name)
            return
        await self._best_effort_delete(name)

    async def _best_effort_delete(self, name: str) -> None:
        vm_id = self._vm_ids.pop(name, None)
        identity = self._identity.pop(name, None)
        self._task_ids.pop(name, None)
        self._metadata.pop(name, None)

        if vm_id:
            payload: dict[str, Any] = {'id': vm_id, 'force': True}
        elif identity:
            payload = {**identity, 'force': True}
        else:
            return

        try:
            await self._call_api(self.control_api_base_url, '/mac/vm/delete', payload)
        except Exception as exc:
            logger.warning("DevCloud '%s': delete failed: %s", name, exc)


__all__ = ['DevCloudRuntimeProvider']
