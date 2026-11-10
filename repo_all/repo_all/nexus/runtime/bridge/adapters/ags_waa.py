from __future__ import annotations

import asyncio
import base64
import json
import uuid

from pathlib import Path

import aiohttp

from nexus.runtime.bridge.adapters.base import (
    BridgeConnectionInfo,
    BridgeLaunchOptions,
    BridgeLaunchResult,
    BridgeNotReadyError,
)
from nexus.runtime.providers.ags.waa import AGSRawWindowsAgentArenaRuntime, WAAExecutorRuntime, require_raw_waa_runtime


WAA_BRIDGE_PORT = 9281
WAA_BRIDGE_HOME = r'C:\Windows\Temp\nexus-llm-bridge'
WAA_BRIDGE_UPLOAD_TIMEOUT = 300.0


def _encoded_powershell(script: str) -> list[str]:
    encoded = base64.b64encode(script.encode('utf-16-le')).decode()
    return ['powershell.exe', '-NoProfile', '-NonInteractive', '-EncodedCommand', encoded]


def _ps_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


class AGSWAABridgeAdapter:
    target_label = 'WAA'
    bridge_port = WAA_BRIDGE_PORT
    bridge_home = WAA_BRIDGE_HOME

    def __init__(self, runtime: WAAExecutorRuntime) -> None:
        self.runtime: AGSRawWindowsAgentArenaRuntime = require_raw_waa_runtime(runtime)

    def _connection_info(self) -> BridgeConnectionInfo:
        return BridgeConnectionInfo(
            base_url=self.runtime.handle.url(self.bridge_port),
            headers=self.runtime.handle.headers,
        )

    async def _execute(self, script: str) -> dict:
        result = await self.runtime.execute(_encoded_powershell(script), shell=False)
        if not isinstance(result, dict):
            raise RuntimeError(f'WAA execute returned unexpected payload: {result!r}')
        return result

    async def upload_artifact(self, artifact_path: Path) -> None:
        remote_artifact = f'{self.bridge_home}.tar.gz'
        await self.stop_bridge()
        cleanup = (
            f'Remove-Item -LiteralPath {_ps_literal(self.bridge_home)}, {_ps_literal(remote_artifact)} '
            '-Recurse -Force -ErrorAction SilentlyContinue'
        )
        await self._execute(cleanup)

        form = aiohttp.FormData()
        form.add_field('file_path', remote_artifact)
        with artifact_path.open('rb') as file_data:
            form.add_field(
                'file_data',
                file_data,
                filename=artifact_path.name,
                content_type='application/octet-stream',
            )
            response = await self.runtime.request(
                'POST',
                '/setup/upload',
                port=self.runtime.server_port,
                data=form,
                timeout=aiohttp.ClientTimeout(total=WAA_BRIDGE_UPLOAD_TIMEOUT),
            )
        if response.status_code >= 400:
            raise RuntimeError(f'WAA bridge artifact upload failed: HTTP {response.status_code} {response.text[:500]}')

        temp_dir = r'C:\Windows\Temp'
        extract = await self._execute(
            f'New-Item -ItemType Directory -Force -Path {_ps_literal(temp_dir)} | Out-Null; '
            f'tar.exe -xzf {_ps_literal(remote_artifact)} -C {_ps_literal(temp_dir)}; '
            'if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }'
        )
        if int(extract.get('returncode', -1)) != 0:
            raise RuntimeError(f'WAA bridge artifact extract failed: {extract}')

    async def launch_bridge(self, options: BridgeLaunchOptions | None = None) -> BridgeLaunchResult:
        launch_options = options or BridgeLaunchOptions()
        launch_id = uuid.uuid4().hex
        launcher = rf'{self.bridge_home}\bin\launch_bridge.cmd'
        await self.runtime.setup_launch(
            [
                'cmd.exe',
                '/d',
                '/s',
                '/c',
                launcher,
                str(self.bridge_port),
                launch_options.conversion_log_level,
                launch_options.thinking_config,
                launch_id,
                str(launch_options.request_timeout),
            ],
            shell=False,
        )
        connection = self._connection_info()
        return BridgeLaunchResult(
            base_url=connection.base_url,
            local_base_url=f'http://127.0.0.1:{self.bridge_port}',
            headers=connection.headers,
            launch_id=launch_id,
        )

    async def wait_ready(self, launch: BridgeLaunchResult, timeout: float = 60.0) -> None:
        deadline = asyncio.get_running_loop().time() + timeout
        last_error = ''
        async with aiohttp.ClientSession(headers=launch.headers or None) as session:
            while asyncio.get_running_loop().time() < deadline:
                try:
                    async with session.get(f'{launch.base_url}/health', timeout=aiohttp.ClientTimeout(total=5)) as resp:
                        text = await resp.text()
                        if resp.status == 200:
                            try:
                                payload = json.loads(text)
                            except json.JSONDecodeError:
                                payload = {}
                            if (
                                payload.get('bridge') == 'nexus-llm-bridge'
                                and payload.get('launch_id') == launch.launch_id
                            ):
                                return
                        last_error = f'HTTP {resp.status} {text[:200]}'
                except Exception as exc:
                    last_error = f'{type(exc).__name__}: {exc}'
                await asyncio.sleep(1)
        logs = await self._collect_logs()
        raise BridgeNotReadyError(f'WAA bridge not ready after {timeout}s: {last_error}\n{logs[-2000:]}')

    async def _collect_logs(self) -> str:
        stdout_path = rf'{self.bridge_home}\bridge.log'
        stderr_path = rf'{self.bridge_home}\bridge.err.log'
        result = await self._execute(
            f'if (Test-Path -LiteralPath {_ps_literal(stdout_path)}) '
            f'{{ Get-Content -Raw -LiteralPath {_ps_literal(stdout_path)} }}; '
            f'if (Test-Path -LiteralPath {_ps_literal(stderr_path)}) '
            f'{{ Get-Content -Raw -LiteralPath {_ps_literal(stderr_path)} }}'
        )
        return str(result.get('output') or result.get('error') or '')

    async def stop_bridge(self) -> None:
        pid_path = rf'{self.bridge_home}\bridge.pid'
        await self._execute(
            f'if (Test-Path -LiteralPath {_ps_literal(pid_path)}) {{ '
            f'$processId = Get-Content -LiteralPath {_ps_literal(pid_path)}; '
            '& taskkill.exe /PID $processId /T /F 2>$null | Out-Null; '
            f'Remove-Item -LiteralPath {_ps_literal(pid_path)} -Force -ErrorAction SilentlyContinue '
            '}; '
            f'Get-NetTCPConnection -LocalPort {self.bridge_port} -State Listen -ErrorAction SilentlyContinue | '
            'ForEach-Object { & taskkill.exe /PID $_.OwningProcess /T /F 2>$null | Out-Null }'
        )


__all__ = ['AGSWAABridgeAdapter']
