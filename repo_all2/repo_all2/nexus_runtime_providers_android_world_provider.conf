"""Android World runtime provider."""

from __future__ import annotations

import logging

from typing import Any

from nexus.runtime.providers.android_world.client import AndroidWorldRuntime
from nexus.runtime.providers.base import RuntimeProvider
from nexus.runtime.providers.registry import runtime_provider_registry


logger = logging.getLogger(__name__)


@runtime_provider_registry.register('android_world')
class AndroidWorldProvider(RuntimeProvider):
    """Provider for Android World benchmark devices.

    Creates :class:`AndroidWorldRuntime` instances that connect to an Android
    World backend over REST for task-based evaluation. The device lifecycle is
    owned by the backend; this provider only opens / closes the HTTP session.

    Config options:
        host: Android World API host (default: 'http://api.cloudtest.woa.com')
        api_key: API authentication key (optional; sent as Bearer token)
        default_seed: Default seed for task initialization (default: 0)

    Acquire params (override provider defaults):
        device_id: Android World device ID (required, integer)
        host: Override config host
        api_key: Override config api_key

    Example:
        provider = AndroidWorldProvider(host='http://api.cloudtest.woa.com', api_key='xxx')
        runtime = await provider.acquire('android', device_id=39810)
        await runtime.gui_screenshot()
        await provider.release('android')
    """

    def __init__(
        self,
        *,
        host: str = 'http://api.cloudtest.woa.com',
        api_key: str | None = None,
        default_seed: int = 0,
        **config: Any,
    ) -> None:
        super().__init__(**config)
        self.host = host
        self.api_key = api_key
        self.default_seed = default_seed

    async def acquire(self, name: str, **params: Any) -> AndroidWorldRuntime:
        """Create and connect an AndroidWorldRuntime for ``name``."""
        if name in self._active_runtimes:
            logger.info("Android World '%s': reusing existing runtime", name)
            return self._active_runtimes[name]  # type: ignore[return-value]

        merged = self.get_params(name, **params)

        device_id = merged.get('device_id')
        if device_id is None:
            raise ValueError('device_id is required for AndroidWorldProvider.acquire()')
        # device_id often arrives as a string from ${oc.env:...} interpolation.
        try:
            device_id = int(device_id)
        except (TypeError, ValueError) as exc:
            raise ValueError(f'device_id must be an integer, got {device_id!r}') from exc

        runtime = AndroidWorldRuntime(
            device_id=device_id,
            host=merged.get('host', self.host),
            api_key=merged.get('api_key', self.api_key),
        )
        await runtime.connect()
        self._active_runtimes[name] = runtime
        logger.info("Android World '%s': READY device=%s", name, device_id)
        return runtime

    async def _release_impl(self, name: str) -> None:
        runtime = self._active_runtimes.pop(name, None)
        if runtime and isinstance(runtime, AndroidWorldRuntime):
            await runtime.disconnect()
            logger.info("Android World '%s': released", name)


__all__ = ['AndroidWorldProvider']
