from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

import aiohttp

from nexus.runtime.bridge.adapters.base import BasePosixBridgeAdapter, BridgeConnectionInfo
from nexus.runtime.providers.devcloud.client import DevCloudRuntime


async def _upload_file_to_setup(base_url: str, local_path: Path, remote_path: str, timeout: float) -> None:
    form = aiohttp.FormData()
    form.add_field('file_path', remote_path)
    with local_path.open('rb') as file_data:
        form.add_field(
            'file_data',
            file_data,
            filename=local_path.name,
            content_type='application/octet-stream',
        )
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f'{base_url.rstrip("/")}/setup/upload',
                data=form,
                timeout=aiohttp.ClientTimeout(total=timeout),
            ) as resp:
                text = await resp.text()
                if resp.status >= 400:
                    raise RuntimeError(f'DevCloud macOS bridge artifact upload failed: HTTP {resp.status} {text[:500]}')


class DevCloudMacOSBridgeAdapter(BasePosixBridgeAdapter):
    target_label = 'DevCloud macOS'

    def __init__(self, runtime: DevCloudRuntime) -> None:
        super().__init__(runtime)
        self.runtime: DevCloudRuntime

    def _connection_info(self) -> BridgeConnectionInfo:
        parsed = urlparse(self.runtime.base_url or '')
        return BridgeConnectionInfo(base_url=f'{parsed.scheme}://{parsed.hostname}:{self.bridge_port}')

    async def _upload_artifact(self, artifact_path: Path, remote_artifact: str) -> None:
        await _upload_file_to_setup(self.runtime.base_url or '', artifact_path, remote_artifact, timeout=300.0)


__all__ = ['DevCloudMacOSBridgeAdapter']
