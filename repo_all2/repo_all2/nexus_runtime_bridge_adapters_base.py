from __future__ import annotations

import asyncio
import json
import uuid

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import aiohttp

from nexus.utils.shell import quote_posix


DEFAULT_BRIDGE_PORT = 8765
DEFAULT_BRIDGE_HOME = '/tmp/nexus-llm-bridge'


@dataclass(frozen=True)
class BridgeLaunchOptions:
    conversion_log_level: str = 'INFO'
    thinking_config: str = 'hunyuan'
    request_timeout: float = 300.0


@dataclass(frozen=True)
class BridgeLaunchResult:
    base_url: str
    local_base_url: str
    headers: dict[str, str] | None = None
    launch_id: str = ''


@dataclass(frozen=True)
class BridgeConnectionInfo:
    """How the host reaches the sandbox-local bridge."""

    base_url: str
    headers: dict[str, str] | None = None


class BridgeNotReadyError(RuntimeError):
    """Raised when the sandbox-local bridge fails to serve /health in time.

    Deliberately not a ``TimeoutError`` subclass: this is a setup failure, not
    the runner's own budget expiring, so it must surface with its full message
    (including the bridge log tail) rather than be swallowed by a runner-level
    timeout handler.
    """


class InstalledAgentBridgeAdapter(Protocol):
    async def upload_artifact(self, artifact_path: Path) -> None: ...
    async def launch_bridge(self, options: BridgeLaunchOptions | None = None) -> BridgeLaunchResult: ...
    async def wait_ready(self, launch: BridgeLaunchResult, timeout: float = 60.0) -> None: ...
    async def stop_bridge(self) -> None: ...


class BasePosixBridgeAdapter:
    """Shared bridge lifecycle for POSIX native targets.

    Subclasses provide only target-specific artifact upload and host-facing
    connection info; launch, readiness polling, log collection and teardown
    are shared.
    """

    bridge_port: int = DEFAULT_BRIDGE_PORT
    bridge_home: str = DEFAULT_BRIDGE_HOME
    extract_timeout: float = 180.0
    target_label: str = 'bridge'

    def __init__(self, runtime: Any) -> None:
        self.runtime = runtime

    async def _upload_artifact(self, artifact_path: Path, remote_artifact: str) -> None:
        """Upload *artifact_path* to *remote_artifact* inside the sandbox."""
        raise NotImplementedError

    def _connection_info(self) -> BridgeConnectionInfo:
        """Return how the host reaches the launched bridge."""
        raise NotImplementedError

    async def upload_artifact(self, artifact_path: Path) -> None:
        remote_artifact = f'{self.bridge_home}.tar.gz'
        await self.stop_bridge()
        await self.runtime.run_command(
            command=f'rm -rf {quote_posix(self.bridge_home)} {quote_posix(remote_artifact)}',
            timeout=30.0,
        )
        await self._upload_artifact(artifact_path, remote_artifact)
        extract = await self.runtime.run_command(
            command=f'mkdir -p /tmp && tar -xzf {quote_posix(remote_artifact)} -C /tmp',
            timeout=self.extract_timeout,
        )
        if extract.return_code != 0:
            raise RuntimeError(f'{self.target_label} bridge artifact extract failed: {extract}')

    async def launch_bridge(self, options: BridgeLaunchOptions | None = None) -> BridgeLaunchResult:
        launch_options = options or BridgeLaunchOptions()
        launch_id = uuid.uuid4().hex
        cmd = (
            f'cd {quote_posix(self.bridge_home)} && '
            f'BRIDGE_HOME={quote_posix(self.bridge_home)} bin/launch_bridge.sh '
            f'{self.bridge_port} {quote_posix(launch_options.conversion_log_level)} '
            f'{quote_posix(launch_options.thinking_config)} {quote_posix(launch_id)} '
            f'{quote_posix(str(launch_options.request_timeout))}'
        )
        result = await self.runtime.run_command(command=cmd, timeout=30.0)
        if result.return_code != 0:
            raise RuntimeError(f'{self.target_label} bridge launch failed: {result}')
        connection = self._connection_info()
        return BridgeLaunchResult(
            base_url=connection.base_url,
            local_base_url=f'http://127.0.0.1:{self.bridge_port}',
            headers=connection.headers,
            launch_id=launch_id,
        )

    async def wait_ready(self, launch: BridgeLaunchResult, timeout: float = 60.0) -> None:
        deadline = asyncio.get_running_loop().time() + timeout
        async with aiohttp.ClientSession(headers=launch.headers or None) as session:
            last_error = ''
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
        raise BridgeNotReadyError(
            f'{self.target_label} bridge not ready after {timeout}s: {last_error}\n{logs[-2000:]}'
        )

    async def _collect_logs(self) -> str:
        result = await self.runtime.run_command(
            command=f'cat {quote_posix(self.bridge_home)}/bridge.log 2>/dev/null || true', timeout=30.0
        )
        return result.stdout or result.stderr or ''

    async def stop_bridge(self) -> None:
        pid_file = f'{quote_posix(self.bridge_home)}/bridge.pid'
        await self.runtime.run_command(
            command=(
                f'if [ -f {pid_file} ]; then '
                f'pid=$(cat {pid_file} 2>/dev/null); '
                'if [ -n "$pid" ]; then '
                'descendants() { '
                'for child in $(ps -eo pid=,ppid= | while read c p; do [ "$p" = "$1" ] && echo "$c"; done); do '
                'descendants "$child"; echo "$child"; '
                'done; '
                '}; '
                'children=$(descendants "$pid"); '
                '[ -z "$children" ] || kill -TERM $children 2>/dev/null || true; '
                'kill -TERM "$pid" 2>/dev/null || true; '
                'sleep 1; '
                '[ -z "$children" ] || kill -KILL $children 2>/dev/null || true; '
                'kill -KILL "$pid" 2>/dev/null || true; '
                'fi; '
                f'rm -f {pid_file}; '
                'fi; '
                f'if command -v lsof >/dev/null 2>&1; then '
                f'port_pids=$(lsof -tiTCP:{self.bridge_port} -sTCP:LISTEN 2>/dev/null || true); '
                '[ -z "$port_pids" ] || kill -KILL $port_pids 2>/dev/null || true; '
                f'elif command -v fuser >/dev/null 2>&1; then fuser -k {self.bridge_port}/tcp >/dev/null 2>&1 || true; '
                'fi'
            ),
            timeout=30.0,
        )


__all__ = [
    'BasePosixBridgeAdapter',
    'BridgeConnectionInfo',
    'BridgeLaunchOptions',
    'BridgeLaunchResult',
    'BridgeNotReadyError',
    'InstalledAgentBridgeAdapter',
]
