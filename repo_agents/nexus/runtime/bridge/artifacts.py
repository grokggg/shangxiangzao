"""Build and cache native LLM Bridge artifacts."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import filelock

from nexus.runtime.bridge.runtime_kind import NativeBridgeBundleTarget, native_bridge_bundle_target_for_runtime
from nexus.utils.paths import CACHE_DIR as _NEXUS_CACHE_DIR


logger = logging.getLogger(__name__)


DEFAULT_BRIDGE_PORT = 8765
BRIDGE_RUNTIME_REQUIREMENTS = Path(__file__).with_name('requirements.txt')
CACHE_DIR = _NEXUS_CACHE_DIR / 'native-bridge-artifacts'

_ARTIFACT_NAMES: dict[NativeBridgeBundleTarget, str] = {
    NativeBridgeBundleTarget.LINUX_X86_64_PY310: 'nexus-llm-bridge-app-linux-x86_64-py310.tar.gz',
    NativeBridgeBundleTarget.WINDOWS_AMD64_PY310: 'nexus-llm-bridge-app-windows-amd64-py310.tar.gz',
    NativeBridgeBundleTarget.MACOS_X86_64_PY311: 'nexus-llm-bridge-app-macos-x86_64-py311.tar.gz',
}
_HASHED_SOURCE_PATHS = [
    'nexus/runtime/bridge',
    'nexus/runtime/server/services/anthropic_proxy',
    'nexus/runtime/server/services/rsio_llm_bridge.py',
    'nexus/utils/common.py',
    'nexus/utils/event_loop.py',
]
_SKIP_DIRS = frozenset({'__pycache__', '.git', '.hg', '.svn', 'node_modules'})
_SKIP_SUFFIXES = frozenset({'.pyc', '.pyo'})

_result_cache: dict[str, Path] = {}
_inflight: dict[str, asyncio.Future[Path]] = {}
_inflight_lock = asyncio.Lock()


@dataclass(frozen=True)
class _BundleTargetSpec:
    python_tag: str
    platform_tag: str | None
    python_executable: str = 'python3'


_BUNDLE_TARGET_SPECS: dict[NativeBridgeBundleTarget, _BundleTargetSpec] = {
    NativeBridgeBundleTarget.LINUX_X86_64_PY310: _BundleTargetSpec('py310', 'manylinux2014_x86_64'),
    NativeBridgeBundleTarget.WINDOWS_AMD64_PY310: _BundleTargetSpec('py310', 'win_amd64', 'python'),
    NativeBridgeBundleTarget.MACOS_X86_64_PY311: _BundleTargetSpec(
        'py311',
        'macosx_10_12_x86_64',
        python_executable='/usr/local/bin/python3',
    ),
}


def _coerce_bundle_target(target: str | NativeBridgeBundleTarget) -> NativeBridgeBundleTarget:
    if isinstance(target, NativeBridgeBundleTarget):
        return target
    return NativeBridgeBundleTarget(target)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _copy_bridge_sources(root: Path) -> None:
    src_root = root / 'src'
    src_root.mkdir(parents=True, exist_ok=True)
    repo_root = _repo_root()
    for rel in [
        'nexus/runtime/bridge',
        'nexus/runtime/server/services/anthropic_proxy',
    ]:
        src = repo_root / rel
        dst = src_root / rel
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))

    for rel in [
        'nexus/__init__.py',
        'nexus/runtime/__init__.py',
        'nexus/runtime/server/__init__.py',
        'nexus/runtime/server/services/__init__.py',
        'nexus/utils/__init__.py',
    ]:
        dst = src_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text('')

    for rel in [
        'nexus/utils/common.py',
        # anthropic_proxy.server uses create_detached_task when the classic
        # AnthropicProxy starts, so importing the shared server module requires it.
        'nexus/utils/event_loop.py',
        # anthropic_proxy.server imports RSIOLLMBridge at module load; vendor it so the
        # bundled bridge can start (it only needs stdlib + a TYPE_CHECKING SIOBridge).
        'nexus/runtime/server/services/rsio_llm_bridge.py',
    ]:
        src = repo_root / rel
        dst = src_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _write_manifest(root: Path, *, name: str, target: str, layout: str = 'src+vendor') -> None:
    manifest = {
        'name': name,
        'target': target,
        'entrypoint': 'python -m nexus.runtime.bridge',
        'layout': layout,
    }
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False))


def _write_launchers(root: Path, *, default_port: int, python_executable: str = 'python3') -> None:
    bin_dir = root / 'bin'
    bin_dir.mkdir(parents=True, exist_ok=True)
    launch_sh = bin_dir / 'launch_bridge.sh'
    launch_sh.write_text(
        '#!/usr/bin/env bash\n'
        'set -euo pipefail\n'
        'BRIDGE_HOME="${BRIDGE_HOME:-$(cd "$(dirname "$0")/.." && pwd)}"\n'
        f'BRIDGE_PORT="${{1:-{default_port}}}"\n'
        'CONVERSION_LOG_LEVEL="${2:-INFO}"\n'
        'THINKING_CONFIG="${3:-hunyuan}"\n'
        'LAUNCH_ID="${4:-}"\n'
        'REQUEST_TIMEOUT="${5:-300.0}"\n'
        'export PYTHONPATH="$BRIDGE_HOME/src:$BRIDGE_HOME/vendor"\n'
        'cd "$BRIDGE_HOME"\n'
        f'nohup {python_executable} -m nexus.runtime.bridge --host 0.0.0.0 --port "$BRIDGE_PORT" '
        '--conversion-log-level "$CONVERSION_LOG_LEVEL" --thinking-config "$THINKING_CONFIG" '
        '--launch-id "$LAUNCH_ID" --request-timeout "$REQUEST_TIMEOUT" '
        '> "$BRIDGE_HOME/bridge.log" 2>&1 &\n'
        'echo $! > "$BRIDGE_HOME/bridge.pid"\n'
        'echo launched\n'
    )
    launch_sh.chmod(0o755)

    launch_ps1 = bin_dir / 'launch_bridge.ps1'
    launch_ps1.write_text(
        'param(\n'
        f'    [int]$Port = {default_port},\n'
        "    [string]$ConversionLogLevel = 'INFO',\n"
        "    [string]$ThinkingConfig = 'hunyuan',\n"
        "    [string]$LaunchId = '',\n"
        "    [string]$RequestTimeout = '300.0'\n"
        ')\n'
        "$ErrorActionPreference = 'Stop'\n"
        "$BridgeHome = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path\n"
        '$env:PYTHONPATH = "$BridgeHome\\src;$BridgeHome\\vendor"\n'
        "$stdout = Join-Path $BridgeHome 'bridge.log'\n"
        "$stderr = Join-Path $BridgeHome 'bridge.err.log'\n"
        '$arguments = @(\n'
        "    '-m', 'nexus.runtime.bridge', '--host', '0.0.0.0', '--port', [string]$Port,\n"
        "    '--conversion-log-level', $ConversionLogLevel, '--thinking-config', $ThinkingConfig,\n"
        "    '--launch-id', $LaunchId, '--request-timeout', $RequestTimeout\n"
        ')\n'
        f"$process = Start-Process -FilePath '{python_executable}' -ArgumentList $arguments "
        '-WorkingDirectory $BridgeHome -RedirectStandardOutput $stdout -RedirectStandardError $stderr '
        '-WindowStyle Hidden -PassThru\n'
        "$process.Id | Set-Content -LiteralPath (Join-Path $BridgeHome 'bridge.pid') -Encoding ascii\n"
        "Write-Output 'launched'\n"
    )

    launch_cmd = bin_dir / 'launch_bridge.cmd'
    launch_cmd.write_text(
        '@echo off\n'
        'setlocal\n'
        'set "BRIDGE_HOME=%~dp0.."\n'
        f'set "BRIDGE_PORT=%~1"\nif "%BRIDGE_PORT%"=="" set "BRIDGE_PORT={default_port}"\n'
        'set "CONVERSION_LOG_LEVEL=%~2"\nif "%CONVERSION_LOG_LEVEL%"=="" set "CONVERSION_LOG_LEVEL=INFO"\n'
        'set "THINKING_CONFIG=%~3"\nif "%THINKING_CONFIG%"=="" set "THINKING_CONFIG=hunyuan"\n'
        'set "LAUNCH_ID=%~4"\n'
        'set "REQUEST_TIMEOUT=%~5"\nif "%REQUEST_TIMEOUT%"=="" set "REQUEST_TIMEOUT=300.0"\n'
        'powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass '
        '-File "%BRIDGE_HOME%\\bin\\launch_bridge.ps1" -Port %BRIDGE_PORT% '
        '-ConversionLogLevel "%CONVERSION_LOG_LEVEL%" -ThinkingConfig "%THINKING_CONFIG%" '
        '-LaunchId "%LAUNCH_ID%" -RequestTimeout "%REQUEST_TIMEOUT%"\n'
    )


def _bridge_runtime_requirements_file() -> Path:
    return BRIDGE_RUNTIME_REQUIREMENTS


def _populate_vendor(root: Path, *, python_tag: str, platform_tag: str | None) -> None:
    vendor = root / 'vendor'
    wheels = root / '.wheels'
    wheels.mkdir()
    cmd = [
        sys.executable,
        '-m',
        'pip',
        'download',
        '--only-binary=:all:',
        '--dest',
        str(wheels),
        '-r',
        str(_bridge_runtime_requirements_file()),
    ]
    if platform_tag:
        cmd.extend(['--platform', platform_tag, '--python-version', python_tag.replace('py', '')])
    # Capture pip's chatter so it never leaks to the run's stdout/stderr; surface it only on failure.
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(
            'Bridge vendor pip download failed (rc=%s):\n%s\n%s',
            result.returncode,
            result.stdout,
            result.stderr,
        )
        raise subprocess.CalledProcessError(result.returncode, cmd, result.stdout, result.stderr)
    for wheel in wheels.glob('*.whl'):
        with zipfile.ZipFile(wheel) as zf:
            zf.extractall(vendor)
    shutil.rmtree(wheels)


def build_app_bundle(
    *, output: Path, target: str | NativeBridgeBundleTarget, default_port: int, include_vendor: bool = True
) -> Path:
    bundle_target = _coerce_bundle_target(target)
    spec = _BUNDLE_TARGET_SPECS[bundle_target]
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / 'nexus-llm-bridge'
        root.mkdir()
        _copy_bridge_sources(root)
        (root / 'vendor').mkdir()
        if include_vendor:
            _populate_vendor(root, python_tag=spec.python_tag, platform_tag=spec.platform_tag)
        _write_launchers(root, default_port=default_port, python_executable=spec.python_executable)
        _write_manifest(root, name=output.name, target=bundle_target.value)
        with tarfile.open(output, 'w:gz') as tf:
            tf.add(root, arcname='nexus-llm-bridge')
    return output


def _iter_hashed_files(repo_root: Path):
    for rel in _HASHED_SOURCE_PATHS:
        path = repo_root / rel
        if path.is_file():
            yield path
            continue
        for file in sorted(path.rglob('*')):
            if not file.is_file():
                continue
            rel_file = file.relative_to(repo_root)
            if _SKIP_DIRS & set(rel_file.parts):
                continue
            if file.suffix in _SKIP_SUFFIXES:
                continue
            yield file


def _bridge_content_hash(target: NativeBridgeBundleTarget, default_port: int) -> str:
    h = hashlib.sha256()
    h.update(b'nexus-native-bridge-artifact-v1')
    h.update(target.encode())
    h.update(str(default_port).encode())

    repo_root = _repo_root()
    for file in _iter_hashed_files(repo_root):
        rel = file.relative_to(repo_root)
        h.update(str(rel).encode())
        h.update(file.read_bytes())
    return h.hexdigest()[:16]


def _find_cached_artifact(cache_key: str, target: NativeBridgeBundleTarget) -> Path | None:
    artifact = CACHE_DIR / cache_key / _ARTIFACT_NAMES[target]
    return artifact if artifact.is_file() else None


def _locked_build(target: NativeBridgeBundleTarget, default_port: int, cache_key: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    lock = filelock.FileLock(CACHE_DIR / f'{cache_key}.lock')

    with lock:
        cached = _find_cached_artifact(cache_key, target)
        if cached is not None:
            return cached

        artifact_name = _ARTIFACT_NAMES[target]
        with tempfile.TemporaryDirectory(prefix='nexus-bridge-build-') as td:
            tmp_output = Path(td) / artifact_name
            build_app_bundle(output=tmp_output, target=target, default_port=default_port)
            dest_dir = CACHE_DIR / cache_key
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / artifact_name
            shutil.copy2(tmp_output, dest)
            return dest


async def build_bridge_artifact_cached(runtime: Any, *, default_port: int = DEFAULT_BRIDGE_PORT) -> Path:
    """Build or reuse the native bridge artifact for *runtime*.

    This mirrors worker venv artifact caching:
    - content-addressed disk cache under ``/tmp/nexus/cache``;
    - file lock for cross-process singleflight;
    - in-process Future coalescing for concurrent coroutines.
    """
    target = native_bridge_bundle_target_for_runtime(runtime)
    content_hash = await asyncio.to_thread(_bridge_content_hash, target, default_port)
    cache_key = f'{target.value}-{content_hash}'

    cached_result = _result_cache.get(cache_key)
    if cached_result is not None and cached_result.exists():
        return cached_result

    async with _inflight_lock:
        fut = _inflight.get(cache_key)
        if fut is None:
            fut = asyncio.get_running_loop().create_future()
            _inflight[cache_key] = fut
            is_owner = True
        else:
            is_owner = False

    if not is_owner:
        return await asyncio.shield(fut)

    try:
        artifact = await asyncio.to_thread(_locked_build, target, default_port, cache_key)
        _result_cache[cache_key] = artifact
        if not fut.done():
            fut.set_result(artifact)
        return artifact
    except asyncio.CancelledError:
        if not fut.done():
            fut.cancel()
        raise
    except BaseException as exc:
        if not fut.done():
            fut.set_exception(exc)
            fut.exception()
        raise
    finally:
        async with _inflight_lock:
            if _inflight.get(cache_key) is fut:
                del _inflight[cache_key]


__all__ = ['build_app_bundle', 'build_bridge_artifact_cached']
