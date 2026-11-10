"""Venv lifecycle manager — create, install, symlink, destroy.

Strategy: symlink the nexus core-deps closure from the base venv (for
the worker SDK), then ``uv pip install`` artifacts with full dependency
resolution.  Pre-symlinked packages are recognised as installed by uv,
so only truly missing dependencies are downloaded.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import logging
import os
import re
import shutil
import sys
import tempfile
import zipfile

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.parser import BytesParser
from pathlib import Path
from typing import TYPE_CHECKING

from packaging.markers import default_environment
from packaging.requirements import Requirement as PkgRequirement
from packaging.utils import canonicalize_name, canonicalize_version, parse_wheel_filename

from nexus.runtime.server.venv.models import ArtifactSpec, VenvSpec, VenvStatus
from nexus.utils.paths import RUNTIME_DIR


logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from nexus.core.module_meta import ModuleMeta

# ---------------------------------------------------------------------------
# Base venv introspection
# ---------------------------------------------------------------------------

_BASE_SITE_PACKAGES: Path | None = None
_CURRENT_ENV: dict[str, str] | None = None


def _get_base_site_packages() -> Path:
    global _BASE_SITE_PACKAGES
    if _BASE_SITE_PACKAGES is None:
        import site

        paths = site.getsitepackages()
        _BASE_SITE_PACKAGES = (
            Path(paths[0])
            if paths
            else Path(sys.prefix) / 'lib' / f'python{sys.version_info.major}.{sys.version_info.minor}' / 'site-packages'
        )
    return _BASE_SITE_PACKAGES


def _get_current_env() -> dict[str, str]:
    """Cached ``packaging.markers.default_environment()``."""
    global _CURRENT_ENV
    environment = _CURRENT_ENV
    if environment is None:
        environment = dict(default_environment())
        _CURRENT_ENV = environment
    return environment


def _normalize(name: str) -> str:
    """PEP 503 normalize a distribution name."""
    return name.lower().replace('_', '-').replace('.', '-')


@dataclass
class _PkgEntry:
    """A package installed in base site-packages."""

    name: str  # normalised
    version: str
    requires: list[tuple[str, frozenset[str]]] = field(default_factory=list)  # [(dep_name, dep_extras)]
    extras_requires: dict[str, list[tuple[str, frozenset[str]]]] = field(default_factory=dict)  # extra -> deps
    dist_info: Path = field(default_factory=lambda: Path())
    top_level: list[Path] = field(default_factory=list)  # package dirs / .py files


def _build_base_index(base_sp: Path) -> dict[str, _PkgEntry]:
    """Index every installed distribution in *base_sp*."""
    index: dict[str, _PkgEntry] = {}
    for dist_info in base_sp.glob('*.dist-info'):
        metadata = dist_info / 'METADATA'
        if not metadata.exists():
            continue

        pkg_name: str | None = None
        pkg_version: str | None = None
        requires: list[tuple[str, frozenset[str]]] = []
        extras_requires: dict[str, list[tuple[str, frozenset[str]]]] = {}

        env = _get_current_env()
        for line in metadata.read_text().splitlines():
            if not line:
                break  # blank line = end of header section
            if line[0].isspace():
                continue  # continuation line
            if line.startswith('Name:'):
                pkg_name = _normalize(line.split(':', 1)[1].strip())
            elif line.startswith('Version:'):
                pkg_version = line.split(':', 1)[1].strip()
            elif line.startswith('Requires-Dist:'):
                dep_str = line[len('Requires-Dist:') :].strip()
                try:
                    req = PkgRequirement(dep_str)
                except Exception:
                    continue
                dep_name = _normalize(req.name)
                dep_extras = frozenset(req.extras)
                if req.marker is None:
                    requires.append((dep_name, dep_extras))
                elif 'extra' not in str(req.marker):
                    # Pure platform marker — evaluate now
                    if req.marker.evaluate(env):
                        requires.append((dep_name, dep_extras))
                else:
                    # Extra-conditional (may also have platform conditions)
                    for m in re.finditer(r"""extra\s*==\s*['"]([^'"]+)['"]""", str(req.marker)):
                        extra_name = m.group(1)
                        if req.marker.evaluate({**env, 'extra': extra_name}):
                            extras_requires.setdefault(extra_name, []).append((dep_name, dep_extras))

        if not pkg_name or not pkg_version:
            continue

        top_level = _find_top_level_entries(base_sp, dist_info)
        index[pkg_name] = _PkgEntry(
            name=pkg_name,
            version=pkg_version,
            requires=requires,
            extras_requires=extras_requires,
            dist_info=dist_info,
            top_level=top_level,
        )
    return index


_SKIP_NAMES = frozenset({'.', '..', '__pycache__'})


def _find_top_level_entries(site_packages: Path, dist_info: Path) -> list[Path]:
    """Determine the top-level filesystem entries belonging to *dist_info*."""
    # 1. top_level.txt (most reliable)
    top_level_txt = dist_info / 'top_level.txt'
    if top_level_txt.exists():
        entries: list[Path] = []
        for name in top_level_txt.read_text().splitlines():
            name = name.strip()
            if not name or name in _SKIP_NAMES or name.startswith('.'):
                continue
            # Exact match (directory or file) first, then glob for .so/.pyd extensions
            exact = site_packages / name
            if exact.exists():
                entries.append(exact)
            elif (site_packages / f'{name}.py').exists():
                entries.append(site_packages / f'{name}.py')
            else:
                # mypyc support modules: top_level.txt lists bare name,
                # filesystem has e.g. <hash>__mypyc.cpython-313-x86_64-linux-gnu.so
                for match in site_packages.glob(f'{name}.*'):
                    if match.suffix in ('.py',) or match.name.endswith('.dist-info'):
                        continue
                    entries.append(match)
                    break
        if entries:
            return entries

    # 2. Parse RECORD for top-level names
    record = dist_info / 'RECORD'
    if record.exists():
        top_names: set[str] = set()
        di_name = dist_info.name
        for line in record.read_text().splitlines():
            parts = line.split(',', 1)
            if not parts:
                continue
            entry = parts[0].strip()
            if not entry or entry.startswith(di_name):
                continue
            top = entry.split('/')[0]
            # Skip path traversals, hidden entries, and __pycache__
            if top.startswith('.') or top == '__pycache__':
                continue
            if '/' in entry or not top.endswith('.dist-info'):
                top_names.add(top)
        return [site_packages / n for n in sorted(top_names) if (site_packages / n).exists()]

    # 3. Guess from dist-info name
    pkg_stem = dist_info.name.split('-')[0]
    results: list[Path] = []
    for candidate_name in (pkg_stem, pkg_stem.replace('-', '_')):
        if candidate_name in _SKIP_NAMES or candidate_name.startswith('.'):
            continue
        for candidate in (site_packages / candidate_name, site_packages / f'{candidate_name}.py'):
            if candidate.exists():
                results.append(candidate)
    return results


def _compute_closure(
    index: dict[str, _PkgEntry],
    seeds: list[tuple[str, frozenset[str]]],
) -> tuple[set[str], set[str]]:
    """BFS transitive closure of *seeds* within *index*.

    Each seed is *(normalised_name, extras)*.  When a package is reached
    with extras, its extras-conditional deps are also traversed.

    Returns *(closure, missing)* where *missing* contains normalised names
    that were reachable but absent from *index*.
    """
    closure: set[str] = set()
    missing: set[str] = set()
    processed_extras: dict[str, set[str]] = {}
    queue: deque[tuple[str, frozenset[str]]] = deque(seeds)
    while queue:
        name, extras = queue.popleft()
        name = _normalize(name)
        first_visit = name not in closure
        seen = processed_extras.get(name, set())
        new_extras = extras - seen if extras else frozenset()

        if not first_visit and not new_extras:
            continue

        closure.add(name)
        processed_extras.setdefault(name, set()).update(extras)

        if name not in index:
            missing.add(name)
            continue

        entry = index[name]
        if first_visit:
            for dep_name, dep_extras in entry.requires:
                queue.append((dep_name, dep_extras))
        for extra in new_extras:
            for dep_name, dep_extras in entry.extras_requires.get(extra, []):
                queue.append((dep_name, dep_extras))
    return closure, missing


def _parse_wheel_requires(
    whl_path: Path,
    extras: list[str] | None = None,
) -> list[tuple[str, frozenset[str]]]:
    """Read ``Requires-Dist`` from a wheel's METADATA without installing it.

    Returns *(dep_name, dep_extras)* tuples.  Extras-conditional deps are
    only included if the corresponding extra is listed in *extras*.
    """
    extras_set = set(extras or [])
    env = _get_current_env()
    requires: list[tuple[str, frozenset[str]]] = []
    with zipfile.ZipFile(whl_path) as zf:
        for entry_name in zf.namelist():
            if entry_name.endswith('.dist-info/METADATA'):
                metadata = zf.read(entry_name).decode()
                for line in metadata.splitlines():
                    if not line:
                        break  # end of header section
                    if not line.startswith('Requires-Dist:'):
                        continue
                    dep_str = line[len('Requires-Dist:') :].strip()
                    try:
                        req = PkgRequirement(dep_str)
                    except Exception:
                        continue
                    dep_name = _normalize(req.name)
                    dep_extras = frozenset(req.extras)

                    if req.marker is None:
                        requires.append((dep_name, dep_extras))
                    elif 'extra' not in str(req.marker):
                        # Pure platform marker
                        if req.marker.evaluate(env):
                            requires.append((dep_name, dep_extras))
                    else:
                        # Extra-conditional — include only if requested extra matches
                        for extra_name in extras_set:
                            if req.marker.evaluate({**env, 'extra': extra_name}):
                                requires.append((dep_name, dep_extras))
                                break
                break  # only read the first METADATA found
    return requires


def _wheel_filename(path: Path) -> str:
    with zipfile.ZipFile(path) as archive:
        metadata_names = [name for name in archive.namelist() if name.endswith('.dist-info/METADATA')]
        wheel_names = [name for name in archive.namelist() if name.endswith('.dist-info/WHEEL')]
        if len(metadata_names) != 1 or len(wheel_names) != 1:
            raise ValueError('wheel_metadata_layout_invalid')
        metadata = BytesParser().parsebytes(archive.read(metadata_names[0]))
        wheel_metadata = BytesParser().parsebytes(archive.read(wheel_names[0]))

    distribution = metadata.get('Name')
    version = metadata.get('Version')
    tags = wheel_metadata.get_all('Tag') or []
    if not distribution or not version or not tags:
        raise ValueError('wheel_metadata_incomplete')
    build = wheel_metadata.get('Build')
    build_part = f'-{build}' if build else ''
    filename = (
        f'{canonicalize_name(distribution).replace("-", "_")}'
        f'-{canonicalize_version(version).replace("-", "_")}'
        f'{build_part}-{tags[0]}.whl'
    )
    parse_wheel_filename(filename)
    return filename


@dataclass
class _SymlinkResult:
    """Result of symlinking packages from base into worker venv."""

    artifact_missing: set[str]  # artifact deps not in base → affects --no-deps decision
    module_missing: list[str]  # module runtime deps not in base → need network install (original PEP 508 strings)


def _collect_module_runtime_seeds(
    whl_paths: list[Path],
    modules: list[str] | None = None,
) -> list[tuple[str, frozenset[str], str]]:
    """Collect module runtime deps as BFS seeds.

    If *whl_paths* contains a nexus wheel, modules are discovered from it
    (reflects the latest source code).  Otherwise falls back to discovering
    from the base venv's installed nexus package.

    Args:
        whl_paths: Artifact wheel paths (only nexus wheels are scanned).
        modules: Module name filter.  ``None`` = all, ``[]`` = none.

    Returns:
        List of *(normalised_name, extras, original_dep_string)* tuples.
    """
    if modules is not None and not modules:
        return []  # explicitly empty → no module deps

    try:
        all_metas: list[ModuleMeta] = _discover_modules_for_runtime_deps(whl_paths)
    except Exception:
        logger.warning('Failed to discover modules for runtime dependencies', exc_info=True)
        return []

    # Filter by module names
    if modules is not None:
        module_set = set(modules)
        all_metas = [m for m in all_metas if m.name in module_set]

    # Collect deps
    env = _get_current_env()
    seen: dict[str, tuple[frozenset[str], str]] = {}
    for meta in all_metas:
        for dep_str in meta.all_runtime_deps:
            try:
                req = PkgRequirement(dep_str)
            except Exception:
                continue
            if req.marker is not None and not req.marker.evaluate(env):
                continue
            norm = _normalize(req.name)
            if norm not in seen:
                seen[norm] = (frozenset(req.extras), dep_str)

    return [(n, ext, orig) for n, (ext, orig) in seen.items()]


def _is_nexus_whl(whl_path: Path) -> bool:
    """Check whether a wheel is the nexus-agent package."""
    stem = whl_path.stem.split('-')[0].lower().replace('-', '_')
    return stem in ('nexus_agent', 'nexus')


def _discover_modules_for_runtime_deps(whl_paths: list[Path]) -> list[ModuleMeta]:
    """Discover Nexus modules used for worker runtime dependency collection."""
    from nexus.core.module_meta import (
        discover_all_modules,
        discover_modules_from_compiled_manifest,
        discover_modules_from_whl,
    )

    for whl_path in whl_paths:
        if _is_nexus_whl(whl_path):
            modules = discover_modules_from_whl(whl_path)
            if modules:
                return modules

    modules = discover_modules_from_compiled_manifest()
    if modules:
        return modules

    return discover_all_modules()


# ---------------------------------------------------------------------------
# VenvManager
# ---------------------------------------------------------------------------


class VenvManager:
    """Manage worker venvs on the local filesystem.

    Each venv is stored under ``root / venv_id``.
    """

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or RUNTIME_DIR / 'venvs'
        self.root.mkdir(parents=True, exist_ok=True)
        self._venvs: dict[str, VenvStatus] = {}
        self._base_index: dict[str, _PkgEntry] | None = None

    # ---- public API ----

    async def create(
        self,
        venv_id: str,
        spec: VenvSpec,
        *,
        wheelhouse: Path | None = None,
        defer_ready: bool = False,
    ) -> VenvStatus:
        """Create a new venv according to *spec*.

        Flow:
          1. ``uv venv`` — bare Python environment
          2. Materialise artifact wheels on disk
          3. Symlink nexus core-deps closure + artifact deps from base
          4. ``uv pip install <whl>[extras]`` — full dep resolution; symlinked
             packages are seen as installed, only missing deps are downloaded
          5. Install extra requirements (if any)

        When *defer_ready* is true, successful installation remains in
        ``creating`` so the runtime server can publish ``ready`` only after its
        worker process has connected.
        """
        venv_dir = self.root / venv_id
        status = VenvStatus(
            venv_id=venv_id,
            state='creating',
            python_version=spec.python_version,
            artifacts=[a.name for a in spec.artifacts],
            extra_requirements=spec.extra_requirements,
            created_at=datetime.now(timezone.utc),
        )
        self._venvs[venv_id] = status

        try:
            python = self._resolve_python(spec.python_version)
            await self._create_bare_venv(python, venv_dir)

            # Materialise wheels
            whl_info: list[tuple[Path, ArtifactSpec]] = []
            for artifact in spec.artifacts:
                whl_path = await self._materialize_whl(artifact)
                whl_info.append((whl_path, artifact))

            # Collect module runtime deps (from nexus whl or base venv)
            module_deps = _collect_module_runtime_seeds(
                [whl_path for whl_path, _ in whl_info],
                modules=spec.modules,
            )

            if wheelhouse is not None:
                if not wheelhouse.is_dir():
                    raise ValueError('offline wheelhouse directory is unavailable')
                for whl_path, artifact in whl_info:
                    await self._install_artifact(venv_dir, whl_path, artifact, wheelhouse=wheelhouse)
                if module_deps:
                    await self._install_extra(
                        venv_dir,
                        [original for _, _, original in module_deps],
                        wheelhouse=wheelhouse,
                    )
                if spec.extra_requirements:
                    await self._install_extra(venv_dir, spec.extra_requirements, wheelhouse=wheelhouse)
                if not defer_ready:
                    status.state = 'ready'
                self._venvs[venv_id] = status
                return status

            # Symlink closure from base (artifact deps + module deps)
            result = self._symlink_from_base(venv_dir, whl_info, module_deps)

            # Install artifacts — no-deps fast path only if artifact closure is complete
            for whl_path, artifact in whl_info:
                await self._install_artifact(venv_dir, whl_path, artifact, no_deps=not result.artifact_missing)

            # Install missing module runtime deps via network
            if result.module_missing:
                await self._install_extra(venv_dir, result.module_missing)

            # Extra requirements (user-specified)
            if spec.extra_requirements:
                await self._install_extra(venv_dir, spec.extra_requirements)

            if not defer_ready:
                status.state = 'ready'
        except Exception as exc:
            status.state = 'failed'
            status.error = str(exc)
            logger.error('Failed to create venv %s: %s', venv_id, exc)

        self._venvs[venv_id] = status
        return status

    async def destroy(self, venv_id: str) -> None:
        """Remove a venv and its files."""
        status = self._venvs.pop(venv_id, None)
        if status:
            status.state = 'destroying'
        venv_dir = self.root / venv_id
        if venv_dir.exists():
            shutil.rmtree(venv_dir, ignore_errors=True)
        logger.info('Destroyed venv %s', venv_id)

    def get(self, venv_id: str) -> VenvStatus | None:
        return self._venvs.get(venv_id)

    def list_all(self) -> list[VenvStatus]:
        return list(self._venvs.values())

    # ---- internals: venv creation ----

    _NIX_BIN = Path('/nix/runtime/nix-env/bin')

    def _resolve_python(self, version: str) -> str:
        nix_versioned = self._NIX_BIN / f'python{version}'
        if nix_versioned.exists():
            return str(nix_versioned)
        if f'{sys.version_info.major}.{sys.version_info.minor}' == version:
            return sys.executable
        return f'python{version}'

    @staticmethod
    def _resolve_uv() -> str:
        nix_uv = VenvManager._NIX_BIN / 'uv'
        if nix_uv.exists():
            return str(nix_uv)
        found = shutil.which('uv')
        if found:
            return found
        return 'uv'

    @staticmethod
    def _uv_subprocess_env() -> dict[str, str]:
        """Return a writable uv environment for dynamic worker venv operations.

        The runtime image may expose a prewarmed ``UV_CACHE_DIR`` under
        ``/nix/runtime``. That path is useful at image build time, but can be
        read-only in remote sandboxes. Dynamic worker venv creation only needs
        a writable uv state/cache directory; dependency reuse comes from
        ``_symlink_from_base`` below.
        """
        env = os.environ.copy()
        cache_dir = RUNTIME_DIR / 'uv-cache'
        cache_dir.mkdir(parents=True, exist_ok=True)
        env['UV_CACHE_DIR'] = str(cache_dir)
        env['UV_PYTHON_DOWNLOADS'] = 'never'
        return env

    async def _create_bare_venv(self, python: str, venv_dir: Path) -> None:
        uv = self._resolve_uv()
        proc = await asyncio.create_subprocess_exec(
            uv,
            'venv',
            '--no-config',
            '--python',
            python,
            str(venv_dir),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=self._uv_subprocess_env(),
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f'uv venv failed: {stderr.decode()}')
        logger.info('Created bare venv at %s', venv_dir)

    # ---- internals: wheel materialisation ----

    async def _materialize_whl(self, artifact: ArtifactSpec) -> Path:
        """Get a wheel file on disk from ArtifactSpec (push or pull)."""
        tmp_dir = Path(tempfile.mkdtemp(prefix='nexus-whl-'))
        download_path = tmp_dir / f'{artifact.name}.whl'

        if artifact.whl_data:
            download_path.write_bytes(base64.b64decode(artifact.whl_data))
        elif artifact.whl_url:
            proc = await asyncio.create_subprocess_exec(
                'curl',
                '-fsSL',
                '-o',
                str(download_path),
                artifact.whl_url,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await proc.communicate()
            if proc.returncode != 0:
                raise RuntimeError(f'Failed to download wheel from {artifact.whl_url}: {stderr.decode()}')
        else:
            raise ValueError(f'ArtifactSpec for {artifact.name} has neither whl_data nor whl_url')

        if artifact.whl_sha256:
            digest = hashlib.sha256()
            with download_path.open('rb') as stream:
                while chunk := stream.read(1024 * 1024):
                    digest.update(chunk)
            actual = digest.hexdigest()
            if actual != artifact.whl_sha256:
                raise ValueError(f'SHA-256 mismatch for {artifact.name}: expected {artifact.whl_sha256}, got {actual}')

        whl_path = tmp_dir / _wheel_filename(download_path)
        download_path.replace(whl_path)
        return whl_path

    # ---- internals: symlink from base ----

    def _get_base_index(self) -> dict[str, _PkgEntry]:
        """Lazy-build and cache the base site-packages index."""
        if self._base_index is None:
            self._base_index = _build_base_index(_get_base_site_packages())
        return self._base_index

    def _symlink_from_base(
        self,
        venv_dir: Path,
        whl_info: list[tuple[Path, ArtifactSpec]],
        module_deps: list[tuple[str, frozenset[str], str]] | None = None,
    ) -> _SymlinkResult:
        """Symlink the nexus core-deps closure + artifact deps + module deps from base.

        Seeds for the BFS closure:
          - ``nexus-agent`` (infrastructure — worker SDK needs this)
          - Every ``Requires-Dist`` of every artifact wheel
          - Module runtime deps (from ``module_deps``)

        Returns a :class:`_SymlinkResult` with separate tracking:
          - ``artifact_missing``: artifact deps not in base (affects ``--no-deps``)
          - ``module_missing``: module runtime deps not in base (original PEP 508 strings for network install)
        """
        base_sp = _get_base_site_packages()
        worker_sp = self._get_site_packages(venv_dir)

        if not base_sp.exists():
            logger.warning('Base site-packages not found at %s, skipping symlink', base_sp)
            return _SymlinkResult(artifact_missing=set(), module_missing=[])

        index = self._get_base_index()

        # Exclude the artifacts themselves — they'll be installed by uv.
        artifact_names = {_normalize(whl_path.stem.split('-')[0]) for whl_path, _ in whl_info}

        # --- Phase 1: artifact closure (nexus-agent + artifact wheel deps) ---
        # Always include 'runtime' extras for nexus-agent: worker processes run
        # inside the runtime and may import runtime-only deps (e.g. fastapi for
        # proxy services used by claude_code, codex, gemini_cli, openclaw).
        artifact_seeds: list[tuple[str, frozenset[str]]] = [('nexus-agent', frozenset(['runtime']))]
        for whl_path, artifact in whl_info:
            artifact_seeds.extend(_parse_wheel_requires(whl_path, artifact.extras))

        artifact_closure, artifact_missing = _compute_closure(index, artifact_seeds)
        artifact_closure -= artifact_names
        artifact_missing -= artifact_names

        if artifact_missing:
            logger.warning(
                'Base venv missing %d artifact deps (will need full resolution): %s',
                len(artifact_missing),
                sorted(artifact_missing),
            )

        # --- Phase 2: module runtime deps closure ---
        module_missing_deps: list[str] = []
        module_closure: set[str] = set()
        if module_deps:
            module_seeds = [(name, extras) for name, extras, _ in module_deps]
            dep_map = {name: orig for name, _, orig in module_deps}

            # Exclude packages already in artifact closure (no need to re-process)
            module_seeds = [(n, e) for n, e in module_seeds if n not in artifact_closure]
            if module_seeds:
                mod_closure, mod_missing = _compute_closure(index, module_seeds)
                mod_closure -= artifact_names | artifact_closure
                mod_missing -= artifact_names | artifact_closure
                module_closure = mod_closure

                if mod_missing:
                    module_missing_deps = [dep_map[n] for n in mod_missing if n in dep_map]
                    logger.warning(
                        'Base venv missing %d module deps (will download): %s',
                        len(module_missing_deps),
                        module_missing_deps,
                    )

        # --- Symlink combined closure ---
        combined_closure = artifact_closure | module_closure
        worker_existing: set[str] = {e.name for e in worker_sp.iterdir()}
        count = 0
        for pkg_name in sorted(combined_closure):
            entry = index.get(pkg_name)
            if entry is None:
                continue
            # dist-info — COPY (uv needs real dirs to detect installed pkgs)
            if entry.dist_info.name not in worker_existing:
                shutil.copytree(entry.dist_info, worker_sp / entry.dist_info.name)
                worker_existing.add(entry.dist_info.name)
                count += 1
            # top-level package dirs / .py files — SYMLINK
            for tl in entry.top_level:
                if tl.name in _SKIP_NAMES or tl.name.startswith('.'):
                    continue
                if tl.name not in worker_existing:
                    os.symlink(tl, worker_sp / tl.name)
                    worker_existing.add(tl.name)
                    count += 1

        logger.info('Linked %d entries (%d packages) from base into %s', count, len(combined_closure), venv_dir)
        return _SymlinkResult(artifact_missing=artifact_missing, module_missing=module_missing_deps)

    # ---- internals: install ----

    async def _install_artifact(
        self,
        venv_dir: Path,
        whl_path: Path,
        artifact: ArtifactSpec,
        *,
        no_deps: bool = False,
        wheelhouse: Path | None = None,
    ) -> None:
        """Install an artifact wheel into *venv_dir*.

        When *no_deps* is True the BFS closure guarantees all transitive deps
        are pre-symlinked, so ``--no-deps`` skips resolution entirely.
        When *no_deps* is False (some deps missing from base) full resolution
        is used so uv downloads the missing packages.
        """
        install_arg = str(whl_path)
        if artifact.extras:
            extras_str = ','.join(artifact.extras)
            install_arg = f'{whl_path}[{extras_str}]'

        uv = self._resolve_uv()
        python = str(venv_dir / 'bin' / 'python')
        base_cmd = [uv, 'pip', 'install', '--no-config', '--link-mode', 'copy', '--python', python]

        if wheelhouse is not None:
            proc = await asyncio.create_subprocess_exec(
                *base_cmd,
                '--no-index',
                '--find-links',
                str(wheelhouse),
                install_arg,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=self._uv_subprocess_env(),
            )
            _, stderr = await proc.communicate()
            if proc.returncode != 0:
                raise RuntimeError(f'Failed to install artifact {artifact.name} from wheelhouse: {stderr.decode()}')
            logger.info('Installed artifact %s into %s from offline wheelhouse', artifact.name, venv_dir)
            return

        if no_deps:
            proc = await asyncio.create_subprocess_exec(
                *base_cmd,
                '--no-deps',
                install_arg,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=self._uv_subprocess_env(),
            )
            _, stderr = await proc.communicate()
            if proc.returncode == 0:
                logger.info('Installed artifact %s into %s (no-deps)', artifact.name, venv_dir)
                return
            logger.warning('no-deps install failed for %s, retrying with deps: %s', artifact.name, stderr.decode())

        # Full resolution — downloads any missing deps
        proc = await asyncio.create_subprocess_exec(
            *base_cmd,
            install_arg,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=self._uv_subprocess_env(),
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f'Failed to install artifact {artifact.name}: {stderr.decode()}')
        logger.info('Installed artifact %s into %s', artifact.name, venv_dir)

    async def _install_extra(
        self,
        venv_dir: Path,
        requirements: list[str],
        *,
        wheelhouse: Path | None = None,
    ) -> None:
        uv = self._resolve_uv()
        python = str(venv_dir / 'bin' / 'python')
        req_str = ' '.join(requirements)
        command = [
            uv,
            'pip',
            'install',
            '--no-config',
            '--link-mode',
            'copy' if wheelhouse is not None else 'symlink',
            '--python',
            python,
        ]
        if wheelhouse is not None:
            command.extend(['--no-index', '--find-links', str(wheelhouse)])
        command.extend(requirements)
        proc = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=self._uv_subprocess_env(),
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f'Failed to install extra requirements: {stderr.decode()}')
        logger.info('Installed extra requirements [%s]', req_str)

    # ---- internals: helpers ----

    def _get_site_packages(self, venv_dir: Path) -> Path:
        lib = venv_dir / 'lib'
        if lib.exists():
            for pydir in lib.iterdir():
                sp = pydir / 'site-packages'
                if sp.exists():
                    return sp
        return lib / f'python{sys.version_info.major}.{sys.version_info.minor}' / 'site-packages'


__all__ = ['VenvManager']
