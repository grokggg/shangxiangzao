from __future__ import annotations

import asyncio

from pathlib import Path

import aiohttp

from nexus.runtime.bridge.adapters.base import BasePosixBridgeAdapter, BridgeConnectionInfo
from nexus.runtime.providers.ags.osworld import (
    AGSOSWorldExecutorRuntime,
    AGSOSWorldRuntime,
    require_ags_osworld_runtime,
)


OSWORLD_BRIDGE_UPLOAD_TIMEOUT = 300.0
OSWORLD_BRIDGE_UPLOAD_ATTEMPTS = 3
OSWORLD_BRIDGE_UPLOAD_RETRY_DELAY = 2.0


class AGSOSWorldBridgeAdapter(BasePosixBridgeAdapter):
    target_label = 'OSWorld'
    # AGS OSWorld's gateway only routes specific ports; 9281 is the one exposed for the bridge.
    bridge_port = 9281

    def __init__(self, runtime: AGSOSWorldExecutorRuntime) -> None:
        super().__init__(require_ags_osworld_runtime(runtime))
        self.runtime: AGSOSWorldRuntime

    def _connection_info(self) -> BridgeConnectionInfo:
        return BridgeConnectionInfo(
            base_url=self.runtime.handle.url(self.bridge_port),
            headers=self.runtime.handle.headers,
        )

    async def _upload_artifact(self, artifact_path: Path, remote_artifact: str) -> None:
        last_error: Exception | None = None
        for attempt in range(1, OSWORLD_BRIDGE_UPLOAD_ATTEMPTS + 1):
            try:
                form = aiohttp.FormData()
                form.add_field('file_path', remote_artifact)
                with artifact_path.open('rb') as file_data:
                    form.add_field(
                        'file_data',
                        file_data,
                        filename=artifact_path.name,
                        content_type='application/octet-stream',
                    )
                    response = await self.runtime._http_request(
                        'POST',
                        self.runtime.handle.url(self.runtime.server_port, '/setup/upload'),
                        data=form,
                        timeout=aiohttp.ClientTimeout(total=OSWORLD_BRIDGE_UPLOAD_TIMEOUT),
                    )
                if response.status_code >= 400:
                    raise RuntimeError(
                        f'OSWorld bridge artifact upload failed: HTTP {response.status_code} {response.text[:500]}'
                    )
                return
            except (aiohttp.ClientError, TimeoutError) as exc:
                last_error = exc
                if attempt >= OSWORLD_BRIDGE_UPLOAD_ATTEMPTS:
                    raise
                await asyncio.sleep(OSWORLD_BRIDGE_UPLOAD_RETRY_DELAY)
        if last_error is not None:
            raise last_error


__all__ = ['AGSOSWorldBridgeAdapter']
