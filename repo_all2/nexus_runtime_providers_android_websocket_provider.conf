"""WebSocket Android sandbox runtime provider."""

from __future__ import annotations

import logging

from typing import Any

from nexus.runtime.providers.android_websocket.client import WebSocketAndroidRuntime
from nexus.runtime.providers.base import RuntimeProvider
from nexus.runtime.providers.registry import runtime_provider_registry


logger = logging.getLogger(__name__)


@runtime_provider_registry.register('android_websocket')
class WebSocketAndroidProvider(RuntimeProvider):
    """Provider for WebSocket-based Android sandbox environments.

    This provider creates WebSocketAndroidRuntime instances that connect to
    sandbox servers via WebSocket for controlling Android devices.

    Config options:
        ws_url: WebSocket server URL (required, e.g., 'ws://9.151.13.55:8788')
        instructions: Default task instructions
        with_history: Whether to keep history (default: True)
        model_version: Model version string (default: 'no')
        manual_vlm: Whether to use manual VLM mode (default: True)

    Acquire params:
        instance_id: Sandbox instance ID (required, e.g., 'cai-1383819211-xxxxx')
        instructions: Task instructions (optional, overrides config)
        with_history: Override config value
        model_version: Override config value
        manual_vlm: Override config value

    Example:
        provider = WebSocketAndroidProvider(
            ws_url='ws://9.151.13.55:8788',
            instructions=['自动化数据采集'],
        )
        runtime = await provider.acquire('main', instance_id='cai-1383819211-xxxxx')
        await runtime.gui_click(500, 300)
        await provider.release('main')
    """

    def __init__(
        self,
        ws_url: str,
        *,
        instructions: list[str] | None = None,
        with_history: bool = True,
        model_version: str = 'no',
        manual_vlm: bool = True,
        **config: Any,
    ) -> None:
        super().__init__(**config)
        self.ws_url = ws_url
        self.instructions = instructions
        self.with_history = with_history
        self.model_version = model_version
        self.manual_vlm = manual_vlm

    async def acquire(self, name: str, **params: Any) -> WebSocketAndroidRuntime:
        merged = self.get_params(name, **params)

        instance_id = merged.get('instance_id')
        if not instance_id:
            raise ValueError('instance_id is required for WebSocketAndroidProvider.acquire()')

        runtime = WebSocketAndroidRuntime(
            ws_url=self.ws_url,
            instance_id=instance_id,
            instructions=merged.get('instructions', self.instructions),
            with_history=merged.get('with_history', self.with_history),
            model_version=merged.get('model_version', self.model_version),
            manual_vlm=merged.get('manual_vlm', self.manual_vlm),
        )

        await runtime.connect()
        self._active_runtimes[name] = runtime
        logger.info(f'Acquired WebSocket Android runtime: {name} -> {instance_id}')
        return runtime

    async def _release_impl(self, name: str) -> None:
        runtime = self._active_runtimes.pop(name, None)
        if runtime and isinstance(runtime, WebSocketAndroidRuntime):
            await runtime.disconnect()
            logger.info(f'Released WebSocket Android runtime: {name}')


__all__ = [
    'WebSocketAndroidProvider',
]
