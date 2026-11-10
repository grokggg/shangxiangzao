"""Docker-based RuntimeProvider for local container execution.

This provider manages Docker containers to provide isolated runtime environments
for executing code. It handles:
- Building the nexus-runtime image if not present
- Creating a runtime container with /nix volume
- Starting user containers with volumes from the runtime container
"""

# /// script
# dependencies = []
#
# [project]
# name = "docker"
# authors = [
#     { name = "tarschen" },
#     { name = "baynejiang" },
# ]
#
# [tool.nexus]
# runtime_providers = { docker = "nexus.runtime.providers.docker" }
# ///

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import shlex
import shutil
import sys
import tempfile

from datetime import datetime, timezone
from importlib import metadata, resources
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit, urlunsplit

from nexus.runtime.client.base import Runtime
from nexus.runtime.client.remote import HttpRuntime
from nexus.runtime.providers.base import RuntimeProvider
from nexus.runtime.providers.registry import runtime_provider_registry
from nexus.utils.common import coerce_bool, get_current_python_version, run_shell_cmd


logger = logging.getLogger(__name__)

# Global locks for coordinating concurrent operations across all provider instances
_image_lock = asyncio.Lock()
_container_lock = asyncio.Lock()
DOCKER_CMD_ENV = 'NEXUS_DOCKER_CMD'


def _docker_shell(command: str) -> str:
    """Return a docker shell command honoring an optional command prefix.

    This is useful on hosts where Docker is only reachable through a wrapper
    such as ``sudo -n docker``.
    """
    docker_cmd = os.getenv(DOCKER_CMD_ENV, 'docker').strip() or 'docker'
    return f'{docker_cmd} {command}'


def _replace_url_host(url: str, display_host: str) -> str:
    parts = urlsplit(url)
    if not parts.scheme or not parts.hostname:
        return url

    netloc = display_host
    if parts.port is not None:
        netloc = f'{display_host}:{parts.port}'
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))


def _docker_environment_args(environment: dict[str, Any] | None) -> list[str]:
    """Return shell-safe docker ``--env`` arguments for a runtime environment."""
    if not environment:
        return []

    args: list[str] = []
    for key, value in environment.items():
        if not key or '=' in key:
            raise ValueError(f'Invalid Docker environment variable name: {key!r}')
        args.append(f'--env {shlex.quote(f"{key}={value}")}')
    return args


def build_runtime_port_summary(runtime: HttpRuntime, *, display_host: str) -> list[str]:
    """Build a human-readable summary of externally mapped runtime ports."""
    base_url = _replace_url_host(runtime.base_url or '', display_host)
    lines = [
        f'Runtime server: {base_url}',
        f'Debug GUI: {base_url.rstrip("/")}/debug/gui',
    ]

    novnc_url = runtime.port_mappings.get(6080)
    if novnc_url:
        lines.append(f'noVNC: {_replace_url_host(novnc_url, display_host).rstrip("/")}/vnc.html')

    vnc_url = runtime.port_mappings.get(5900)
    if vnc_url:
        lines.append(
            f'VNC: {_replace_url_host(vnc_url, display_host).removeprefix("http://").removeprefix("https://")}'
        )

    return lines


async def run_shell_cmd_with_streaming(cmd: str, cwd: str | None = None) -> None:
    """Run a shell command with real-time stdout/stderr streaming.

    Args:
        cmd: The shell command to execute
        cwd: Optional working directory for the command

    Raises:
        RuntimeError: If the command exits with a non-zero return code
    """
    logger.debug('Running shell command with streaming: %s', cmd)

    proc = await asyncio.create_subprocess_shell(
        cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=cwd,
    )

    # Stream output line by line
    assert proc.stdout is not None
    while True:
        line = await proc.stdout.readline()
        if not line:
            break
        # Print to stdout for real-time display
        print(line.decode('utf-8'), end='', flush=True)

    await proc.wait()

    if proc.returncode != 0:
        raise RuntimeError(f'Command failed with return code {proc.returncode}')


if TYPE_CHECKING:
    if sys.version_info >= (3, 11):
        from importlib.resources.abc import Traversable
    else:
        from importlib.abc import Traversable


def get_package_version() -> str:
    """Get the nexus package version."""
    # Prefer reading from pyproject.toml at project root (source checkout)
    pyproject_path = Path(__file__).parents[3] / 'pyproject.toml'
    if pyproject_path.is_file():
        return _read_package_version_from_pyproject(pyproject_path)

    # Fallback: look up installed distribution metadata
    for package_name in ['tencent-nexus', 'nexus-agent']:
        try:
            return metadata.version(package_name)
        except metadata.PackageNotFoundError:
            ...

    # Binary distributions keep minimal project metadata in nexus._bundle.
    try:
        bundle_pyproject = resources.files('nexus._bundle') / 'pyproject.toml'
        if bundle_pyproject.is_file():
            return _read_package_version_from_pyproject(bundle_pyproject)
    except (ModuleNotFoundError, TypeError):
        pass

    raise RuntimeError(
        f'Could not determine nexus package version: neither pyproject.toml at '
        f'{pyproject_path}, an installed distribution (tencent-nexus/nexus-agent), '
        'nor nexus._bundle/pyproject.toml was found.'
    )


def _read_package_version_from_pyproject(pyproject_path: Any) -> str:
    """Read ``project.version`` from a pyproject path or package resource."""
    if sys.version_info >= (3, 11):
        import tomllib
    else:
        import tomli as tomllib

    with pyproject_path.open('rb') as f:
        data = tomllib.load(f)
    try:
        return data['project']['version']
    except KeyError as e:
        raise RuntimeError(
            f'Could not determine nexus package version: {pyproject_path} is missing [project].version.'
        ) from e


def _is_installed_mode() -> bool:
    """Check if running in installed mode (nexus._bundle exists)."""
    try:
        resources.files('nexus._bundle')
        return True
    except (ModuleNotFoundError, TypeError):
        return False


def _get_nexus_package_path() -> Path:
    """Get the path to the nexus package directory."""
    nexus_pkg = resources.files('nexus')

    # In Python 3.13+, resources.files() may return a Path object directly
    if isinstance(nexus_pkg, Path):
        return nexus_pkg

    # In older versions or certain installation modes, it has a _path attribute
    if hasattr(nexus_pkg, '_path'):
        return Path(nexus_pkg._path)  # type: ignore[attr-defined]

    raise RuntimeError('Cannot determine nexus package path: unsupported resources.files() return type')


def _get_project_root() -> Path:
    """Get the project root directory (for development mode).

    Returns:
        Path to project root (parent of nexus package)

    Raises:
        RuntimeError: If pyproject.toml or deployment directory not found
    """
    pkg_path = _get_nexus_package_path()
    project_root = pkg_path.parent

    if not (project_root / 'pyproject.toml').exists():
        raise RuntimeError(
            f'Development mode detected but pyproject.toml not found at {project_root}. '
            'Please run from the project root directory.'
        )

    if not (project_root / 'deployment').exists():
        raise RuntimeError(
            f'Development mode detected but deployment directory not found at {project_root}. '
            'Please ensure the deployment directory exists.'
        )

    return project_root


async def image_exists(image_name: str) -> bool:
    """Check if a Docker image exists locally."""
    try:
        await run_shell_cmd(_docker_shell(f'image inspect {shlex.quote(image_name)} >/dev/null 2>&1'))
        return True
    except RuntimeError:
        return False


async def container_exists(container_name: str) -> bool:
    """Check if a Docker container exists (running or stopped)."""
    try:
        result = await run_shell_cmd(
            _docker_shell(f"ps -a --format '{{{{.Names}}}}' --filter 'name=^{container_name}$'")
        )
        return container_name in result.strip().split('\n')
    except RuntimeError:
        return False


async def container_running(container_name: str) -> bool:
    """Check if a Docker container is running."""
    try:
        result = await run_shell_cmd(_docker_shell(f"ps --format '{{{{.Names}}}}' --filter 'name=^{container_name}$'"))
        return container_name in result.strip().split('\n')
    except RuntimeError:
        return False


def _parse_docker_timestamp(timestamp: str) -> datetime:
    """Parse Docker ISO 8601 timestamp to timezone-aware UTC datetime.

    Docker returns timestamps like '2025-01-15T10:30:00.123456789Z' with nanosecond
    precision. Python's fromisoformat only supports up to 6 decimal places (microseconds),
    so we truncate the nanoseconds.
    """
    ts = timestamp.strip()
    # Handle 'Z' suffix (UTC indicator)
    if ts.endswith('Z'):
        ts = ts[:-1] + '+00:00'
    # Truncate nanoseconds to microseconds (fromisoformat supports up to 6 decimal places)
    if '.' in ts:
        parts = ts.split('.')
        tz_idx = len(parts[1])
        for i, c in enumerate(parts[1]):
            if c in ('+', '-'):
                tz_idx = i
                break
        # Keep only 6 decimal places for microseconds
        decimal_part = parts[1][:tz_idx][:6]
        tz_part = parts[1][tz_idx:] if tz_idx < len(parts[1]) else ''
        ts = f'{parts[0]}.{decimal_part}{tz_part}'
    dt = datetime.fromisoformat(ts)
    # Convert to UTC if it has timezone info
    if dt.tzinfo is not None:
        return dt.astimezone(timezone.utc)
    # Assume UTC if no timezone info
    return dt.replace(tzinfo=timezone.utc)


async def get_image_created_time(image_name: str) -> datetime | None:
    """Get the creation time of a Docker image as UTC datetime."""
    try:
        result = await run_shell_cmd(_docker_shell(f"inspect --format '{{{{.Created}}}}' {shlex.quote(image_name)}"))
        if result.strip():
            return _parse_docker_timestamp(result)
        return None
    except RuntimeError:
        return None


async def get_container_created_time(container_name: str) -> datetime | None:
    """Get the creation time of a Docker container as UTC datetime."""
    try:
        result = await run_shell_cmd(
            _docker_shell(f"inspect --format '{{{{.Created}}}}' {shlex.quote(container_name)}")
        )
        if result.strip():
            return _parse_docker_timestamp(result)
        return None
    except RuntimeError:
        return None


async def get_container_port(container_name: str, internal_port: int = 8000) -> int | None:
    """Get the host port mapped to the container's internal port."""
    try:
        result = await run_shell_cmd(_docker_shell(f'port {shlex.quote(container_name)} {internal_port}'))
        if not result.strip():
            return None
        # Output format: "0.0.0.0:32768" or "[::]:32768"
        port_mapping = result.strip().split('\n')[0]
        return int(port_mapping.split(':')[-1])
    except RuntimeError:
        return None


async def get_docker_bridge_gateway() -> str:
    """Get Docker bridge network gateway IP.

    This IP is accessible from both the host and containers using the default bridge network.
    """
    try:
        result = await run_shell_cmd(
            _docker_shell("network inspect bridge --format '{{range .IPAM.Config}}{{.Gateway}}{{end}}'")
        )
        gateway = result.strip()
        if gateway:
            return gateway
    except RuntimeError:
        pass
    return '172.17.0.1'  # Default fallback


async def build_runtime_image(
    image_name: str,
    force_rebuild: bool = False,
    build_context: Path | None = None,
    push: bool = False,
) -> None:
    """Build the nexus-runtime Docker image.

    Args:
        image_name: Name/tag for the built image
        force_rebuild: If True, rebuild even if image exists
        build_context: Optional custom build context directory.
                      If None, auto-detects based on installation mode.
        push: If True, push image to mirrors.tencent.com after build.
              Requires DOCKER_REGISTRY_USERNAME and DOCKER_REGISTRY_PASSWORD env vars.

    In installed mode (pip install nexus):
        - Extracts bundled files from nexus._bundle to a temp directory
        - Copies nexus package for installation in the image

    In development mode (pip install -e .):
        - Uses project root directory directly
        - Raises RuntimeError if pyproject.toml or deployment not found
    """
    # 如果开启 push 且是默认版本 tag，先检查远端是否已存在，已存在则跳过 build 和 push
    if push:
        version = get_package_version()
        default_tag = f'nexus-runtime:{version}'
        if image_name == default_tag and await _remote_image_exists(image_name):
            logger.info('Remote image %s already exists, skipping build and push', image_name)
            return

    if not force_rebuild and await image_exists(image_name):
        logger.info('Image %s already exists, skipping build', image_name)
        if push:
            await _push_image(image_name)
        return

    if build_context is not None:
        await _build_from_context(image_name, build_context)
    elif _is_installed_mode():
        await _build_from_package(image_name)
    else:
        project_root = _get_project_root()
        await _build_from_context(image_name, project_root)

    logger.info('Successfully built %s', image_name)

    if push:
        await _push_image(image_name)


async def _build_from_context(image_name: str, build_context: Path) -> None:
    """Build image from a local directory context."""
    dockerfile = build_context / 'deployment' / 'docker' / 'Dockerfile.runtime'
    logger.info('Building %s from %s...', image_name, build_context)
    await run_shell_cmd_with_streaming(
        _docker_shell(
            f'build -f {shlex.quote(str(dockerfile))} -t {shlex.quote(image_name)} {shlex.quote(str(build_context))}'
        )
    )


async def _build_from_package(image_name: str) -> None:
    """Build image by extracting files from installed package."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)

        bundle_pkg = resources.files('nexus._bundle')
        _copy_package_tree(bundle_pkg, tmppath)

        nexus_pkg = resources.files('nexus')
        _copy_package_tree(nexus_pkg, tmppath / 'nexus')
        _copy_compiled_package_artifacts(_get_nexus_package_path().parent, tmppath)

        dockerfile = tmppath / 'deployment' / 'docker' / 'Dockerfile.runtime'
        logger.info('Building %s from package resources...', image_name)
        await run_shell_cmd_with_streaming(
            _docker_shell(
                f'build -f {shlex.quote(str(dockerfile))} -t {shlex.quote(image_name)} {shlex.quote(str(tmppath))}'
            )
        )


def _copy_package_tree(pkg: Traversable, dest: Path) -> None:
    """Recursively copy package resources to a directory."""
    dest.mkdir(parents=True, exist_ok=True)
    for item in pkg.iterdir():
        item_dest = dest / item.name
        if item.is_file():
            if isinstance(item, Path):
                shutil.copy2(item, item_dest)
            else:
                item_dest.write_bytes(item.read_bytes())
        elif item.is_dir():
            _copy_package_tree(item, item_dest)


def _copy_compiled_package_artifacts(source_root: Path, dest: Path) -> None:
    """Copy importable compiled package artifacts into a runtime build context."""
    dest.mkdir(parents=True, exist_ok=True)
    for artifact in sorted(source_root.glob('nexus.*')):
        if artifact.suffix not in ('.so', '.pyd', '.pyi'):
            continue
        shutil.copy2(artifact, dest / artifact.name)


async def _remote_image_exists(image_name: str) -> bool:
    """Check if an image exists on mirrors.tencent.com.

    Args:
        image_name: Local image name/tag (e.g. nexus-runtime:0.1.0)

    Returns:
        True if the remote image exists.
    """
    registry = 'mirrors.tencent.com'
    namespace = 'hunyuan_yanguan'
    registry_image = f'{registry}/{namespace}/{image_name}'
    try:
        await run_shell_cmd(_docker_shell(f'manifest inspect {shlex.quote(registry_image)}'))
        return True
    except RuntimeError:
        return False


async def _push_image(image_name: str) -> None:
    """Push Docker image to mirrors.tencent.com.

    Reads credentials from environment variables:
    - DOCKER_REGISTRY_USERNAME (default: baynejiang)
    - DOCKER_REGISTRY_PASSWORD (required)

    Args:
        image_name: Name/tag of the image to push
    """
    import os

    registry = 'mirrors.tencent.com'
    namespace = 'hunyuan_yanguan'
    username = os.getenv('DOCKER_REGISTRY_USERNAME', 'baynejiang')
    password = os.getenv('DOCKER_REGISTRY_PASSWORD')

    if not password:
        logger.warning('DOCKER_REGISTRY_PASSWORD not set, skipping push')
        return

    # Login to registry
    logger.info('Logging in to %s...', registry)
    try:
        # Use asyncio.create_subprocess_shell to avoid password exposure in terminal
        # stdout/stderr inherit from parent to show login messages
        proc = await asyncio.create_subprocess_shell(
            _docker_shell(f'login --username {shlex.quote(username)} --password-stdin {shlex.quote(registry)}'),
            stdin=asyncio.subprocess.PIPE,
        )
        await proc.communicate(input=password.encode('utf-8'))

        if proc.returncode != 0:
            raise RuntimeError('Docker login failed')
    except Exception as e:
        logger.error('Failed to login to registry: %s', e)
        raise

    # Tag and push image
    registry_image = f'{registry}/{namespace}/{image_name}'
    logger.info('Tagging and pushing %s...', registry_image)
    try:
        await run_shell_cmd(_docker_shell(f'tag {shlex.quote(image_name)} {shlex.quote(registry_image)}'))
        await run_shell_cmd_with_streaming(_docker_shell(f'push {shlex.quote(registry_image)}'))
        logger.info('Successfully pushed %s', registry_image)
    except RuntimeError as e:
        logger.error('Failed to push image: %s', e)
        raise


@runtime_provider_registry.register('docker')
class DockerRuntimeProvider(RuntimeProvider):
    """Docker-based RuntimeProvider for local container execution.

    This provider creates Docker containers that connect to a shared runtime
    container providing the /nix volume with Python environments.

    Config options:
        image: Main container image (takes precedence over configure() if set)
        runtime_image: Image to use for the runtime container (default: nexus-runtime:{version})
        force_rebuild: Force rebuild of runtime image even if exists (default: False)
        build_context: Path to build context for building runtime image
                       (default: auto-detect, uses project root in dev mode or package resources)
        python_version: Python version to use in runtime (default: current interpreter version)
        runtime_port: Internal port for runtime service (default: 8000)
        resource_spec: Resource specification dict with 'cpu' (cores) and 'memory' (GB)
                       Note: For docker provider this is informational only, actual limits depend on host
        request_timeout: HTTP request timeout in seconds (default: 1800.0)
        working_dir: Container working directory (optional)
        auto_connect_ws: Whether to automatically connect Socket.IO for real-time logs and events (default: True)
        socketio_request_timeout: Engine.IO HTTP polling/connect request timeout in seconds (default: 60.0)
        docker_host: IP address or hostname of the Docker host (default: localhost)
                     Use this when Docker context points to a remote machine.
                     Example: "192.168.1.100" for a remote Docker daemon.
        enable_dind: Unsupported for the local Docker provider; use Gongfeng/gVisor for DinD.
        runtime_disable_aslr: Unsupported for the local Docker provider; use Gongfeng/gVisor.

    Acquire params:
        image: Docker image to run for this runtime instance (used only if provider-level image not set)
        port: Host port to expose (optional, default: auto-assigned)
        exposed_ports: List of additional internal ports to expose (optional, each gets auto-assigned host port)
                      Example: [8023, 8080, 8888]
        environment: Environment variables to pass to the container
        extra_args: Additional docker run arguments (optional)
        enable_dind: Unsupported for the local Docker provider; use Gongfeng/gVisor for DinD.
        runtime_disable_aslr: Unsupported for the local Docker provider; use Gongfeng/gVisor.
        run_id: Host-unique run identifier for container naming (optional)
        resource_spec: Resource specification (optional, uses provider default)
        enable_runtime_logs: Whether to receive and print logs from runtime server (default: True)
        auto_connect_ws: Whether to automatically connect Socket.IO (default: provider-level setting)
        socketio_request_timeout: Engine.IO HTTP polling/connect request timeout (default: provider-level setting)
    """

    def __init__(
        self,
        image: str | None = None,
        runtime_image: str | None = None,
        force_rebuild: bool = False,
        build_context: str | Path | None = None,
        python_version: str | None = None,
        runtime_port: int = 8000,
        resource_spec: dict[str, int] | None = None,
        request_timeout: float = 1800.0,
        exposed_ports: list[int] | None = None,
        print_port_mappings: bool = True,
        working_dir: str | None = None,
        auto_connect_ws: bool = True,
        socketio_request_timeout: float = 60.0,
        docker_host: str = 'localhost',
        enable_dind: bool | str = False,
        runtime_disable_aslr: bool | str = False,
        **config: Any,
    ) -> None:
        """Initialize DockerRuntimeProvider.

        Args:
            image: Main container image (takes precedence over configure() if set)
            runtime_image: Image to use for the runtime container (default: nexus-runtime:{version})
            force_rebuild: Force rebuild of runtime image even if exists (default: False)
            build_context: Path to build context for building runtime image
                           (default: auto-detect, uses project root in dev mode or package resources)
            python_version: Python version to use in runtime (default: current interpreter version)
            runtime_port: Internal port for runtime service (default: 8000)
            resource_spec: Resource specification with 'cpu' (cores) and 'memory' (GB)
                           Note: For docker provider this is informational only
            request_timeout: HTTP request timeout in seconds (default: 1800.0)
            exposed_ports: List of additional internal ports to expose (optional)
            print_port_mappings: Whether to print resolved runtime and GUI port URLs to stderr
            working_dir: Container working directory (optional)
            auto_connect_ws: Whether to automatically connect Socket.IO for real-time logs and events (default: True)
            socketio_request_timeout: Engine.IO HTTP polling/connect request timeout in seconds (default: 60.0)
            docker_host: IP address or hostname of the Docker host (default: localhost)
                         Use this when Docker context points to a remote machine.
            enable_dind: Unsupported for the local Docker provider; use Gongfeng/gVisor for DinD.
            runtime_disable_aslr: Unsupported for the local Docker provider; use Gongfeng/gVisor.
            **config: Additional config options
        """
        super().__init__(**config)
        if coerce_bool(enable_dind, name='enable_dind'):
            raise ValueError(
                'enable_dind is only supported by the Gongfeng/gVisor runtime path. '
                'The local Docker provider uses runc and must not start nested dockerd.'
            )
        if coerce_bool(runtime_disable_aslr, name='runtime_disable_aslr'):
            raise ValueError(
                'runtime_disable_aslr is only supported by the Gongfeng/gVisor runtime path. '
                'The local Docker provider has no sandbox-sdk passthrough for this flag.'
            )
        version = get_package_version()
        self.image = image
        # 支持显式写 runtime_image: "current"，等价于使用当前版本的镜像
        if runtime_image == 'current' or not runtime_image:
            self.runtime_image = f'nexus-runtime:{version}'
        else:
            self.runtime_image = runtime_image
        self.force_rebuild = force_rebuild
        self.build_context = Path(build_context) if build_context else None
        self.python_version = python_version or get_current_python_version()
        self.runtime_port = runtime_port
        self.resource_spec = resource_spec
        self.request_timeout = request_timeout
        self.exposed_ports = exposed_ports
        self.print_port_mappings = print_port_mappings
        self.working_dir = working_dir
        self.auto_connect_ws = auto_connect_ws
        self.socketio_request_timeout = socketio_request_timeout
        self.docker_host = docker_host

        # Container tracking
        self._runtime_container_name = f'nexus-runtime-{version.replace(".", "-")}'
        self._user_containers: dict[str, str] = {}  # name -> container_id

    async def _ensure_runtime_image(self) -> None:
        """Ensure the runtime image is built.

        Uses a global async lock to prevent concurrent builds.
        """
        async with _image_lock:
            await build_runtime_image(
                self.runtime_image,
                force_rebuild=self.force_rebuild,
                build_context=self.build_context,
            )

    async def _ensure_runtime_container(self) -> None:
        """Ensure the runtime container exists with /nix volume.

        If the runtime image was created after the runtime container,
        the container is recreated to use the newer image.

        Uses a global async lock to prevent concurrent container creation.
        """
        async with _container_lock:
            if await container_exists(self._runtime_container_name):
                image_time = await get_image_created_time(self.runtime_image)
                container_time = await get_container_created_time(self._runtime_container_name)

                if image_time and container_time and image_time > container_time:
                    logger.info(
                        'Runtime image %s (created %s) is newer than container %s (created %s), recreating container...',
                        self.runtime_image,
                        image_time,
                        self._runtime_container_name,
                        container_time,
                    )
                    await run_shell_cmd(_docker_shell(f'rm -f {shlex.quote(self._runtime_container_name)}'))
                else:
                    logger.info('Runtime container %s already exists', self._runtime_container_name)
                    return

            logger.info('Creating runtime container %s...', self._runtime_container_name)
            await run_shell_cmd(
                _docker_shell(
                    f'run -d --name {shlex.quote(self._runtime_container_name)} {shlex.quote(self.runtime_image)} true'
                )
            )
            logger.info('Runtime container %s created', self._runtime_container_name)

    async def acquire(self, name: str, **params: Any) -> Runtime:
        """Acquire a new Docker runtime instance.

        Args:
            name: Unique name for this runtime instance
            **params: Runtime parameters:
                - image: Docker image to run (used only if provider-level image not set)
                - port: Host port to expose (default: auto-assigned)
                - exposed_ports: List of additional internal ports to expose
                - extra_args: Additional docker run arguments
                - run_id: Host-unique run identifier (optional, used for container naming)
                - resource_spec: Resource specification (optional)
                - enable_runtime_logs: Whether to receive and print logs from runtime server (default: True)

        Returns:
            HttpRuntime connected to the container
        """
        merged = self.get_params(name, **params)

        # If image is configured at provider level, it takes precedence over configure() settings
        image = self.image or merged.get('image')
        if not image:
            raise ValueError('image parameter is required for DockerRuntimeProvider.acquire()')

        host_port = merged.get('port')
        exposed_ports = merged.get('exposed_ports')
        extra_args = merged.get('extra_args', [])
        environment = merged.get('environment')
        run_id = merged.get('run_id')
        resource_spec = merged.get('resource_spec') or self.resource_spec
        enable_runtime_logs = merged.get('enable_runtime_logs', True)
        auto_connect_ws = merged.get('auto_connect_ws', self.auto_connect_ws)
        socketio_request_timeout = merged.get('socketio_request_timeout', self.socketio_request_timeout)
        enable_dind = coerce_bool(merged.get('enable_dind', False), name='enable_dind')
        if enable_dind:
            raise ValueError(
                'enable_dind is only supported by the Gongfeng/gVisor runtime path. '
                'The local Docker provider uses runc and must not start nested dockerd.'
            )
        runtime_disable_aslr = coerce_bool(merged.get('runtime_disable_aslr', False), name='runtime_disable_aslr')
        if runtime_disable_aslr:
            raise ValueError(
                'runtime_disable_aslr is only supported by the Gongfeng/gVisor runtime path. '
                'The local Docker provider has no sandbox-sdk passthrough for this flag.'
            )

        # Ensure runtime infrastructure is ready
        await self._ensure_runtime_image()
        await self._ensure_runtime_container()

        # Container naming: nexus-{run_id}-{name} or nexus-user-{name} (fallback)
        container_name = f'nexus-{run_id}-{name}' if run_id else f'nexus-user-{name}'

        # Determine the host address for connecting to containers
        # For local Docker: use bridge gateway for container-to-container communication
        # For remote Docker: use the configured docker_host
        if self.docker_host == 'localhost':
            gateway = await get_docker_bridge_gateway()
        else:
            gateway = self.docker_host

        # Remove any existing container (stale volumes, old config, etc.)
        if await container_exists(container_name):
            if await container_running(container_name):
                logger.warning('Removing existing user container %s (will recreate with fresh volumes)', container_name)
            await run_shell_cmd(_docker_shell(f'rm -f {shlex.quote(container_name)}'))

        # Build docker run command
        cmd_parts = [
            _docker_shell('run -d'),
            f'--name {shlex.quote(container_name)}',
            f'--volumes-from {shlex.quote(self._runtime_container_name)}:ro',
            "--entrypoint ''",
        ]

        # Resource limits from resource_spec
        if resource_spec:
            if cpu := resource_spec.get('cpu'):
                cmd_parts.append(f'--cpus={cpu}')
            if memory := resource_spec.get('memory'):
                cmd_parts.append(f'--memory={memory}g')

        # Working directory
        actual_working_dir = merged.get('working_dir') or self.working_dir
        if actual_working_dir:
            cmd_parts.append(f'--workdir {shlex.quote(actual_working_dir)}')

        # Port mapping for main runtime port
        if host_port:
            cmd_parts.append(f'-p {host_port}:{self.runtime_port}')
        else:
            cmd_parts.append(f'-p {self.runtime_port}')  # Auto-assign host port

        # Port mappings for additional exposed ports
        actual_exposed_ports = exposed_ports or self.exposed_ports or []
        for port in actual_exposed_ports:
            cmd_parts.append(f'-p {port}')  # Auto-assign host port for each

        # Environment variables
        cmd_parts.extend(_docker_environment_args(environment))

        # Extra arguments
        for arg in extra_args:
            cmd_parts.append(shlex.quote(arg))

        # Image and start command
        cmd_parts.extend([
            shlex.quote(image),
            '/nix/runtime/nix-env/bin/bash',
            '/nix/runtime/nexus/deployment/runtime/start.sh',
            '-v',
            self.python_version,
            '-p',
            str(self.runtime_port),
            '-H',
            '0.0.0.0',
        ])

        logger.info('Starting container %s...', container_name)
        await run_shell_cmd(' '.join(cmd_parts))

        # Wait for container to be ready and get the assigned port
        await self._wait_for_container_ready(container_name, gateway)

        port = await get_container_port(container_name, self.runtime_port)
        if not port:
            raise RuntimeError(f'Failed to get port for container {container_name}')

        runtime = HttpRuntime(
            base_url=f'http://{gateway}:{port}',
            request_timeout=self.request_timeout,
            resource_spec=resource_spec,
            enable_runtime_logs=enable_runtime_logs,
            auto_connect_ws=auto_connect_ws,
            socketio_request_timeout=socketio_request_timeout,
        )

        # Populate port_mappings for exposed ports
        for internal_port in actual_exposed_ports:
            mapped_port = await get_container_port(container_name, internal_port)
            if mapped_port:
                runtime.port_mappings[internal_port] = f'http://{gateway}:{mapped_port}'
                logger.info('Port mapping: %s -> http://%s:%s', internal_port, gateway, mapped_port)

        self._active_runtimes[name] = runtime
        self._user_containers[name] = container_name

        logger.info('Runtime %s ready at http://%s:%s', name, gateway, port)
        if self.print_port_mappings:
            display_host = '127.0.0.1' if self.docker_host == 'localhost' else self.docker_host
            print(
                '\n'.join(build_runtime_port_summary(runtime, display_host=display_host)),
                file=sys.stderr,
                flush=True,
            )
        return runtime

    async def _wait_for_container_ready(
        self,
        container_name: str,
        gateway: str,
        timeout: float = 60.0,
        interval: float = 1.0,
    ) -> None:
        """Wait for container to be running and healthy."""
        import aiohttp

        elapsed = 0.0
        while elapsed < timeout:
            if not await container_running(container_name):
                # Check if container exited with error
                try:
                    logs = await run_shell_cmd(_docker_shell(f'logs --tail 50 {shlex.quote(container_name)}'))
                except RuntimeError as e:
                    logs = str(e)
                raise RuntimeError(f'Container {container_name} exited unexpectedly:\n{logs}')

            # Try to connect to the runtime
            port = await get_container_port(container_name, self.runtime_port)
            if port:
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.get(
                            f'http://{gateway}:{port}/health', timeout=aiohttp.ClientTimeout(total=2)
                        ):
                            return  # Container is ready
                except Exception:
                    pass  # Not ready yet

            await asyncio.sleep(interval)
            elapsed += interval

        raise TimeoutError(f'Container {container_name} did not become ready within {timeout}s')

    async def _release_impl(self, name: str) -> None:
        """Release a Docker runtime instance.

        Args:
            name: Name of the runtime instance to release
        """
        if name not in self._user_containers:
            return

        # Explicitly disconnect Socket.IO client to avoid CLOSE_WAIT connections
        if name in self._active_runtimes:
            runtime = self._active_runtimes[name]
            try:
                await runtime.ws_disconnect()
                logger.debug('Socket.IO disconnected for runtime %s', name)
            except Exception as e:
                logger.debug('Failed to disconnect Socket.IO for runtime %s: %s', name, e)

        container_name = self._user_containers[name]
        logger.info('Stopping container %s...', container_name)
        with contextlib.suppress(RuntimeError):
            await run_shell_cmd(_docker_shell(f'rm -f {shlex.quote(container_name)}'))

        self._user_containers.pop(name, None)
        self._active_runtimes.pop(name, None)
        logger.info('Runtime %s released', name)

    async def cleanup_runtime_container(self) -> None:
        """Remove the shared runtime container.

        This is typically called when completely done with all runtimes.
        """
        if await container_exists(self._runtime_container_name):
            logger.info('Removing runtime container %s...', self._runtime_container_name)
            with contextlib.suppress(RuntimeError):
                await run_shell_cmd(_docker_shell(f'rm -f {shlex.quote(self._runtime_container_name)}'))


__all__ = [
    'DockerRuntimeProvider',
    'build_runtime_image',
    'get_package_version',
    'get_current_python_version',
]
