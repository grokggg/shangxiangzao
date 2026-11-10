"""AGR Mobile sandbox runtime provider."""

from __future__ import annotations

import asyncio
import logging

from typing import Any

from nexus.runtime.providers.agr_mobile.client import AGRMobileRuntime
from nexus.runtime.providers.base import RuntimeProvider
from nexus.runtime.providers.registry import runtime_provider_registry


logger = logging.getLogger(__name__)


def _load_e2b_sandbox_class():
    try:
        from e2b import Sandbox

        return Sandbox
    except ImportError as e:
        raise ImportError('e2b is not installed. Install it manually: pip install e2b>=0.17.0') from e


@runtime_provider_registry.register('agr_mobile')
class AGRMobileProvider(RuntimeProvider):
    """Provider for AGR Mobile sandbox environments.

    Supports two acquisition modes:
    1. Connect to an existing sandbox via ``sandbox_id``.
    2. Create a new sandbox from ``sandbox_template`` and then connect to it.

    When the provider creates the sandbox itself, it also owns cleanup and will
    kill that sandbox during ``release()``.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        domain: str | None = None,
        sandbox_template: str | None = None,
        template_timeout: int = 1800,
        request_timeout: float = 1800.0,
        appium_port: int = AGRMobileRuntime.APPIUM_PORT,
        ready_timeout: int = AGRMobileRuntime.APPIUM_READY_TIMEOUT,
        **config: Any,
    ) -> None:
        super().__init__(**config)
        self.api_key = api_key
        self.domain = domain
        self.sandbox_template = sandbox_template
        self.template_timeout = int(template_timeout)
        self.request_timeout = float(request_timeout)
        self.appium_port = appium_port
        self.ready_timeout = ready_timeout
        self._created_sandboxes: dict[str, Any] = {}

    async def acquire(self, name: str, **params: Any) -> AGRMobileRuntime:
        merged = self.get_params(name, **params)

        api_key = merged.get('api_key', self.api_key)
        domain = merged.get('domain', self.domain)
        sandbox_id = merged.get('sandbox_id')
        sandbox_template = merged.get('sandbox_template', self.sandbox_template)

        if not sandbox_id:
            if not sandbox_template:
                raise ValueError('sandbox_id or sandbox_template is required for AGRMobileProvider.acquire()')
            if not api_key:
                raise ValueError('api_key is required when creating a sandbox via sandbox_template')

            sandbox_cls = _load_e2b_sandbox_class()
            create_payload: dict[str, Any] = {
                'template': sandbox_template,
                'timeout': int(merged.get('template_timeout', self.template_timeout)),
                'api_key': api_key,
                'request_timeout': float(merged.get('request_timeout', self.request_timeout)),
            }
            if domain:
                create_payload['domain'] = domain

            logger.info('Creating AGR mobile sandbox: %s', {'template': sandbox_template, 'domain': domain})
            sandbox = await asyncio.to_thread(sandbox_cls.create, **create_payload)
            sandbox_id = getattr(sandbox, 'sandbox_id', None)
            if not sandbox_id:
                raise RuntimeError('Created AGR mobile sandbox did not expose sandbox_id')
            self._created_sandboxes[name] = sandbox

        runtime = AGRMobileRuntime(
            sandbox_id=sandbox_id,
            api_key=api_key,
            domain=domain,
            appium_port=int(merged.get('appium_port', self.appium_port)),
            ready_timeout=int(merged.get('ready_timeout', self.ready_timeout)),
        )

        await runtime.connect()
        self._active_runtimes[name] = runtime
        logger.info('Acquired AGR Mobile runtime: %s -> sandbox %s', name, sandbox_id)
        return runtime

    async def _release_impl(self, name: str) -> None:
        runtime = self._active_runtimes.pop(name, None)
        if runtime and isinstance(runtime, AGRMobileRuntime):
            await runtime.disconnect()

        sandbox = self._created_sandboxes.pop(name, None)
        if sandbox is not None:
            try:
                await asyncio.to_thread(sandbox.kill)
            except Exception as exc:
                logger.warning('Failed to kill AGR mobile sandbox %s: %s', name, exc)

        if runtime:
            logger.info(f'Released AGR Mobile runtime: {name}')


__all__ = [
    'AGRMobileProvider',
]
