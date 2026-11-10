"""CloudTest Android runtime provider."""

from __future__ import annotations

import logging

from typing import Any

from nexus.runtime.providers.android_cloudtest.client import CloudTestAndroidRuntime
from nexus.runtime.providers.base import RuntimeProvider
from nexus.runtime.providers.registry import runtime_provider_registry


logger = logging.getLogger(__name__)


@runtime_provider_registry.register('android_cloudtest')
class CloudTestAndroidProvider(RuntimeProvider):
    """Provider for CloudTest-based Android cloud devices.

    This provider creates CloudTestAndroidRuntime instances that connect to
    real Android devices via Tencent's CloudTest API.

    Config options:
        host: CloudTest API host (default: 'http://api.cloudtest.woa.com/')
        user_id: CloudTest user ID (optional, can be set via params)
        user_key: CloudTest user key (optional, can be set via params)
        is_public_cloud: Whether to use public cloud (default: False)
        project: Project name for public cloud (optional)

    Acquire params:
        device_id: CloudTest device ID (required, integer)
        host: Override config host
        user_id: Override config user_id
        user_key: Override config user_key
        is_public_cloud: Override config value
        project: Override config value

    Example:
        provider = CloudTestAndroidProvider(
            host='http://api.cloudtest.woa.com/',
            user_id='xxx',
            user_key='xxx',
        )
        runtime = await provider.acquire('main', device_id=28093)
        await runtime.gui_click(720, 1600)
        await provider.release('main')
    """

    def __init__(
        self,
        *,
        host: str = 'http://api.cloudtest.woa.com/',
        user_id: str | None = None,
        user_key: str | None = None,
        is_public_cloud: bool = False,
        project: str | None = None,
        **config: Any,
    ) -> None:
        super().__init__(**config)
        self.host = host
        self.user_id = user_id
        self.user_key = user_key
        self.is_public_cloud = is_public_cloud
        self.project = project

    async def acquire(self, name: str, **params: Any) -> CloudTestAndroidRuntime:
        merged = self.get_params(name, **params)

        device_id = merged.get('device_id')
        if device_id is None:
            raise ValueError('device_id is required for CloudTestAndroidProvider.acquire()')

        runtime = CloudTestAndroidRuntime(
            device_id=device_id,
            host=merged.get('host', self.host),
            user_id=merged.get('user_id', self.user_id),
            user_key=merged.get('user_key', self.user_key),
            is_public_cloud=merged.get('is_public_cloud', self.is_public_cloud),
            project=merged.get('project', self.project),
        )

        await runtime.connect()
        self._active_runtimes[name] = runtime
        logger.info(f'Acquired CloudTest Android runtime: {name} -> device {device_id}')
        return runtime

    async def _release_impl(self, name: str) -> None:
        runtime = self._active_runtimes.pop(name, None)
        if runtime and isinstance(runtime, CloudTestAndroidRuntime):
            await runtime.disconnect()
            logger.info(f'Released CloudTest Android runtime: {name}')


__all__ = [
    'CloudTestAndroidProvider',
]
