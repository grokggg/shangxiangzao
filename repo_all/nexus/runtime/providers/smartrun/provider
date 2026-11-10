"""SmartRun real-device RuntimeProvider."""

from __future__ import annotations

import asyncio
import logging
import os
import random

from typing import Any, cast

from nexus.runtime.providers.base import RuntimeProvider
from nexus.runtime.providers.registry import runtime_provider_registry
from nexus.runtime.providers.smartrun.client import SmartRunRuntime
from nexus.utils.async_http import http_client


logger = logging.getLogger(__name__)


@runtime_provider_registry.register('smartrun')
class SmartRunRuntimeProvider(RuntimeProvider):
    """Provider for SmartRun real devices (iOS / Android).

    Manages the full device lifecycle: assigns a device on acquire() and
    releases it on release(). Devices are identified by session_id (SmartRun's
    lease handle) and device_id (the physical device identifier).

    Config options:
        token: SmartRun API token (or set SMARTRUN_TOKEN env var)
        api_base_url: SmartRun API base URL (default: 'https://smartrun.woa.com')
        platform: Default platform filter ('ios' / 'android' / 'harmony')
        brand: Default brand filter (e.g. 'Apple')
        model: Default model filter (e.g. 'iPhone11,2')
        duration: Default session duration in seconds (default: 1800)
        request_timeout: HTTP request timeout in seconds (default: 60.0)
        startup_jitter_max: Max random sleep before acquire() to de-sync batch starts

    Acquire params (all optional, override provider defaults):
        device_id: Specific device ID; if omitted a free device is auto-assigned
        platform: Platform filter
        brand: Brand filter
        model: Model filter
        duration: Session duration override
        purpose: Usage description for SmartRun statistics

    Example:
        provider = SmartRunRuntimeProvider(token='mcp_xxx', platform='ios')
        runtime = await provider.acquire('ep0')
        await runtime.gui_screenshot()
        await provider.release('ep0')
    """

    def __init__(
        self,
        token: str | None = None,
        api_base_url: str = 'https://smartrun.woa.com',
        platform: str | None = None,
        brand: str | None = None,
        model: str | None = None,
        duration: int = 1800,
        request_timeout: float = 60.0,
        startup_jitter_max: float = 0.0,
        **config: Any,
    ) -> None:
        super().__init__(**config)
        resolved_token = token or os.getenv('SMARTRUN_TOKEN')
        if not resolved_token:
            raise ValueError(
                'token is required for SmartRunRuntimeProvider. '
                'Pass it explicitly or set the SMARTRUN_TOKEN environment variable.'
            )
        self.token = resolved_token
        self.api_base_url = api_base_url.rstrip('/')
        self.platform = platform
        self.brand = brand
        self.model = model
        self.duration = duration
        self.request_timeout = request_timeout
        self.startup_jitter_max = startup_jitter_max

        self._session_ids: dict[str, str] = {}  # name -> session_id
        self._device_ids: dict[str, str] = {}  # name -> device_id
        self._acquire_locks: dict[str, asyncio.Lock] = {}

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    def get_runtime_id(self, name: str) -> str | None:
        """Return the session_id for the runtime registered under ``name``."""
        return self._session_ids.get(name)

    def get_device_id(self, name: str) -> str | None:
        """Return the device_id for the runtime registered under ``name``."""
        return self._device_ids.get(name)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def acquire(self, name: str, **params: Any) -> SmartRunRuntime:
        """Assign a device and return a connected SmartRunRuntime.

        Args:
            name: Unique name for this runtime instance.
            **params: device_id, platform, brand, model, duration, purpose.

        Returns:
            SmartRunRuntime connected to the assigned device.
        """
        if name not in self._acquire_locks:
            self._acquire_locks[name] = asyncio.Lock()

        async with self._acquire_locks[name]:
            return await self._acquire_impl(name, **params)

    async def _acquire_impl(self, name: str, **params: Any) -> SmartRunRuntime:
        if self.startup_jitter_max > 0:
            jitter = random.uniform(0, self.startup_jitter_max)
            logger.info('SmartRun acquire jitter: sleeping %.1fs (name=%s)', jitter, name)
            await asyncio.sleep(jitter)

        # Reuse existing active runtime for the same name
        if name in self._active_runtimes:
            logger.info("SmartRun '%s': reusing existing runtime (session=%s)", name, self._session_ids[name])
            return cast(SmartRunRuntime, self._active_runtimes[name])

        merged = self.get_params(name, **params)

        # Build assign_device arguments
        assign_args: dict[str, Any] = {
            'json_output': True,
            'duration': merged.get('duration', self.duration),
            'auto_renew': True,
        }
        device_id = merged.get('device_id')
        if device_id:
            assign_args['device_id'] = device_id

        platform = merged.get('platform', self.platform)
        if platform:
            assign_args['platform'] = platform

        brand = merged.get('brand', self.brand)
        if brand:
            assign_args['brand'] = brand

        model = merged.get('model', self.model)
        if model:
            assign_args['model'] = model

        purpose = merged.get('purpose')
        if purpose:
            assign_args['purpose'] = purpose

        logger.info("SmartRun '%s': assigning device (args=%s)", name, assign_args)
        data = await self._call_mcp('assign_device', assign_args)

        assigned_device_id: str = data['device']['device_id']
        session_id: str = data['session']['session_id']
        screen_size_str: str = data['device'].get('screen_size', '1080x1920')
        assigned_platform: str = data['device'].get('platform', platform or 'ios')

        # Parse "WxH" -> (W, H)
        try:
            w, h = (int(v) for v in screen_size_str.split('x'))
            screen_size: tuple[int, int] = (w, h)
        except (ValueError, AttributeError):
            screen_size = (1080, 1920)
            logger.warning(
                "SmartRun '%s': could not parse screen_size %r, using %s", name, screen_size_str, screen_size
            )

        runtime = SmartRunRuntime(
            device_id=assigned_device_id,
            session_id=session_id,
            screen_size=screen_size,
            platform=assigned_platform,
            api_base_url=self.api_base_url,
            token=self.token,
        )
        await runtime.connect()

        self._active_runtimes[name] = runtime
        self._session_ids[name] = session_id
        self._device_ids[name] = assigned_device_id

        logger.info(
            "SmartRun '%s': READY device=%s session=%s platform=%s screen=%s",
            name,
            assigned_device_id,
            session_id,
            assigned_platform,
            screen_size,
        )
        return runtime

    async def _release_impl(self, name: str) -> None:
        device_id = self._device_ids.pop(name, None)
        session_id = self._session_ids.pop(name, None)
        self._active_runtimes.pop(name, None)

        if not device_id:
            logger.warning("SmartRun release '%s': no device_id found, skipping", name)
            return

        logger.info("SmartRun '%s': releasing device=%s session=%s", name, device_id, session_id)
        try:
            await self._call_mcp('release_device', {'device_id': device_id, 'json_output': True})
            logger.info("SmartRun '%s': released", name)
        except Exception as e:
            logger.warning("SmartRun '%s': release_device failed (device will auto-expire): %s", name, e)

    async def _release_all_impl(self) -> None:
        names = list(self._device_ids.keys())
        await asyncio.gather(*[self._release_impl(n) for n in names], return_exceptions=True)

    # ------------------------------------------------------------------
    # Internal HTTP helper
    # ------------------------------------------------------------------

    async def _call_mcp(self, tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Send a JSON-RPC request to SmartRun MCP and return the parsed result dict."""
        import json

        url = f'{self.api_base_url}/mcp'
        headers = {
            'Authorization': f'Bearer {self.token}',
            'Content-Type': 'application/json',
        }
        payload = {
            'jsonrpc': '2.0',
            'method': 'tools/call',
            'params': {'name': tool, 'arguments': arguments},
            'id': 1,
        }

        resp = await http_client.post(url, json=payload, headers=headers)
        if resp.status_code != 200:
            raise RuntimeError(f'SmartRun HTTP error {resp.status_code} for tool={tool}: {resp.text[:512]}')

        data_line = ''
        for line in resp.text.splitlines():
            if line.startswith('data:'):
                data_line = line[5:].strip()
                break

        if not data_line:
            raise RuntimeError(f'SmartRun no data line for tool={tool}')

        rpc = json.loads(data_line)
        result = rpc.get('result', {})
        if result.get('isError'):
            content = result.get('content', [{}])
            msg = content[0].get('text', 'unknown error') if content else 'unknown error'
            raise RuntimeError(f'SmartRun {tool} error: {msg}')

        content = result.get('content', [{}])
        text = content[0].get('text', '{}') if content else '{}'
        return json.loads(text)


__all__ = ['SmartRunRuntimeProvider']
