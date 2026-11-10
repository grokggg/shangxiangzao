"""Artifact packaging — build wheels from source directories.

Three modes (checked in order):
1. Installed in site-packages (sibling ``.dist-info``) → repack with existing metadata
2. Has ``pyproject.toml`` → prefer ``uv build``, fallback to custom builder
3. Plain directory → treat as package itself, build wheel
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import logging
import shutil
import subprocess
import tempfile
import zipfile

from pathlib import Path

import filelock

from nexus.utils.paths import CACHE_DIR as _NEXUS_CACHE_DIR


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cached build_artifact
# ---------------------------------------------------------------------------

CACHE_DIR = _NEXUS_CACHE_DIR / 'artifacts'

# Process-level result cache: built wheel path for this process's source.
_result_cache: Path | None = None

# In-process coalescing: concurrent callers share one build Future.
_inflight: asyncio.Future[Path] | None = None
_inflight_lock = asyncio.Lock()


def _content_hash(source: Path) -> str:
    """Compute a content fingerprint of the source directory.

    Hashes all pyproject.toml files (root + extension modules) and all .py
    files.
    """
    ignore_spec = _load_ignore_spec(source)

    h = hashlib.sha256()
    for f in sorted(source.rglob('pyproject.toml')):
        rel = f.relative_to(source)
        if _ALWAYS_SKIP_DIRS & set(rel.parts):
            continue
        if ignore_spec is not None and ignore_spec.match_file(str(rel)):
            continue
        h.update(str(rel).encode())
        h.update(f.read_bytes())
    for f in sorted(source.rglob('*.py')):
        rel = f.relative_to(source)
        if _ALWAYS_SKIP_DIRS & set(rel.parts):
            continue
        if ignore_spec is not None and ignore_spec.match_file(str(rel)):
            continue
        h.update(str(rel).encode())
        h.update(f.read_bytes())

    dist_info = _find_dist_info(source)
    if dist_info is not None:
        h.update(f'dist-info:{dist_info.name}'.encode())
        for name in ('METADATA', 'WHEEL', 'top_level.txt', 'RECORD'):
            file = dist_info / name
            if file.exists():
                h.update(name.encode())
                h.update(file.read_bytes())

    return h.hexdigest()[:16]


def _find_cached_whl(content_hash: str) -> Path | None:
    """Look up a previously-built wheel in the disk cache."""
    d = CACHE_DIR / content_hash
    if d.exists():
        whls = list(d.glob('*.whl'))
        if whls:
            return whls[0]
    return None


def _locked_build(
    source: Path,
    *,
    exclude_patterns: list[str] | None = None,
) -> Path:
    """Resolve/hash source, then build under a file lock."""
    source = source.resolve()
    content_hash = _content_hash(source)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    lock = filelock.FileLock(CACHE_DIR / f'{content_hash}.lock')

    with lock:
        cached = _find_cached_whl(content_hash)
        if cached is not None:
            return cached

        whl = build_artifact(source, exclude_patterns=exclude_patterns)
        dest_dir = CACHE_DIR / content_hash
        dest_dir.mkdir(exist_ok=True)
        dest = dest_dir / whl.name
        shutil.copy2(whl, dest)
        return dest


async def build_artifact_cached(
    source: Path,
    *,
    exclude_patterns: list[str] | None = None,
) -> Path:
    """Build a wheel with 3-layer caching.

    1. Process-level result cache (instant)
    2. asyncio.Future coalescing (concurrent coroutines share one resolve/hash/build)
    3. filelock + disk cache (concurrent processes share one build)
    """
    global _inflight
    global _result_cache

    # Layer 1: process-level result cache
    if _result_cache is not None and _result_cache.exists():
        return _result_cache

    # Layer 2: in-process coalescing
    async with _inflight_lock:
        if _inflight is None:
            fut: asyncio.Future[Path] = asyncio.get_running_loop().create_future()
            _inflight = fut
            is_owner = True
        else:
            fut = _inflight
            is_owner = False

    if not is_owner:
        return await fut

    try:
        # Layer 3: file lock + disk cache (runs in thread to avoid blocking)
        whl = await asyncio.to_thread(_locked_build, source, exclude_patterns=exclude_patterns)
        _result_cache = whl
        fut.set_result(whl)
        return whl
    except Exception as e:
        fut.set_exception(e)
        raise
    finally:
        async with _inflight_lock:
            if _inflight is fut:
                _inflight = None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _find_uv() -> str | None:
    """Find the ``uv`` binary — nix path first, then PATH."""
    nix_uv = Path('/nix/runtime/nix-env/bin/uv')
    if nix_uv.exists():
        return str(nix_uv)
    return shutil.which('uv')


def _try_uv_build(source: Path) -> Path | None:
    """Try to build a wheel with ``uv build --wheel``.

    Returns the wheel path on success, ``None`` on failure or if uv is
    unavailable.  ``uv build`` handles build isolation and produces correct
    metadata including optional-dependencies / extras.
    """
    uv = _find_uv()
    if uv is None:
        return None

    out_dir = Path(tempfile.mkdtemp(prefix='nexus-build-'))
    try:
        result = subprocess.run(
            [uv, 'build', '--wheel', '--out-dir', str(out_dir)],
            cwd=str(source),
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            logger.debug('uv build failed: %s', result.stderr)
            return None
        wheels = list(out_dir.glob('*.whl'))
        return wheels[0] if wheels else None
    except Exception as exc:
        logger.debug('uv build error: %s', exc)
        return None


def build_artifact(
    source: Path,
    *,
    name: str | None = None,
    version: str | None = None,
    extras: list[str] | None = None,
    exclude_patterns: list[str] | None = None,
) -> Path:
    """Build a wheel from a local source directory.

    Args:
        source: Source code directory (project root, package dir, or installed package).
        name: Package name override (auto-detected from pyproject.toml or dir name).
        version: Version override (auto-detected from pyproject.toml, default ``'0.0.0'``).
        extras: Extras to note (recorded in ArtifactSpec, not the wheel itself).
        exclude_patterns: File glob patterns to exclude from the wheel.

    Returns:
        Path to the built ``.whl`` file.
    """
    source = source.resolve()
    if not source.is_dir():
        raise ValueError(f'Source must be a directory: {source}')

    # Mode 1: installed package in site-packages — repack with existing metadata
    dist_info = _find_dist_info(source)
    if dist_info is not None:
        return _repack_installed(source, dist_info, exclude_patterns)

    # Mode 2: project with pyproject.toml
    pyproject = source / 'pyproject.toml'
    if pyproject.exists():
        # Prefer uv build — handles build isolation + correct metadata (incl. extras)
        whl = _try_uv_build(source)
        if whl is not None:
            logger.info('Built wheel via uv build: %s', whl.name)
            return whl

        # Fallback: custom builder
        logger.info('uv build unavailable, using custom builder')
        meta = _read_pyproject(pyproject)
        pkg_name = name or meta.get('name') or source.name.replace('-', '_')
        pkg_version = version or meta.get('version') or '0.0.0'
        dependencies = meta.get('dependencies', [])
        optional_dependencies = meta.get('optional_dependencies')
        packages = _discover_packages(source)
        if not packages:
            raise RuntimeError(f'No Python packages found in {source}')
        return _build_wheel(
            pkg_name,
            pkg_version,
            packages,
            exclude_patterns,
            dependencies=dependencies,
            optional_dependencies=optional_dependencies,
        )

    # Mode 3: plain directory — treat as package itself
    pkg_name = name or source.name.replace('-', '_')
    pkg_version = version or '0.0.0'
    return _build_wheel(pkg_name, pkg_version, [source], exclude_patterns)


# ---------------------------------------------------------------------------
# Mode 1: repack installed package from site-packages
# ---------------------------------------------------------------------------

_ALWAYS_SKIP_DIRS = frozenset({'__pycache__', '.git', '.hg', '.svn', 'node_modules'})
_ALWAYS_SKIP_SUFFIXES = frozenset({'.pyc', '.pyo'})


def _is_excluded(file_path: Path, base: Path, exclude_patterns: list[str] | None) -> bool:
    """Check whether *file_path* should be excluded from the wheel."""
    if _ALWAYS_SKIP_DIRS & set(file_path.relative_to(base).parts):
        return True
    if file_path.suffix in _ALWAYS_SKIP_SUFFIXES:
        return True
    if exclude_patterns:
        rel = file_path.relative_to(base)
        if any(rel.match(p) for p in exclude_patterns):
            return True
    return False


def _find_dist_info(source: Path) -> Path | None:
    """Find the ``.dist-info`` directory for an installed package.

    Looks for ``{normalized_name}-*.dist-info`` in the same parent directory
    (typically ``site-packages/``).  If the distribution name differs from
    the import package name, falls back to ``top_level.txt`` / ``RECORD``
    ownership checks.
    """
    parent = source.parent
    normalized = source.name.replace('-', '_').lower()
    dist_infos: list[Path] = []

    for entry in parent.iterdir():
        if not (entry.is_dir() and entry.name.endswith('.dist-info')):
            continue
        dist_infos.append(entry)
        stem = entry.name.removesuffix('.dist-info')
        parts = stem.rsplit('-', 1)
        if len(parts) != 2:
            continue
        dist_name = parts[0].replace('-', '_').lower()
        if dist_name == normalized:
            return entry

    matches = [entry for entry in dist_infos if _dist_info_owns_top_level(entry, source.name)]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        logger.warning(
            'Multiple dist-info directories own top-level package %s: %s',
            source.name,
            [m.name for m in matches],
        )
    return None


def _dist_info_owns_top_level(dist_info: Path, top_level: str) -> bool:
    """Return whether *dist_info* declares ownership of a top-level package."""
    top_level_txt = dist_info / 'top_level.txt'
    if top_level_txt.exists():
        names = {line.strip() for line in top_level_txt.read_text().splitlines()}
        if top_level in names:
            return True

    record = dist_info / 'RECORD'
    if record.exists():
        package_prefix = f'{top_level}/'
        module_path = f'{top_level}.py'
        for line in record.read_text().splitlines():
            path = line.split(',', 1)[0]
            if path == module_path or path.startswith(package_prefix):
                return True

    return False


def _repack_installed(
    source: Path,
    dist_info: Path,
    exclude_patterns: list[str] | None = None,
) -> Path:
    """Repack an installed package and its ``.dist-info`` into a wheel."""
    meta = _parse_pkg_metadata(dist_info / 'METADATA')
    pkg_name = meta['name'].replace('-', '_')
    pkg_version = meta['version']

    # Read wheel tag from existing WHEEL file, default to pure-Python
    tag = 'py3-none-any'
    wheel_file = dist_info / 'WHEEL'
    if wheel_file.exists():
        for line in wheel_file.read_text().splitlines():
            if line.startswith('Tag:'):
                tag = line.split(':', 1)[1].strip()
                break

    dist_info_name = f'{pkg_name}-{pkg_version}.dist-info'
    whl_filename = f'{pkg_name}-{pkg_version}-{tag}.whl'

    out_dir = Path(tempfile.mkdtemp(prefix='nexus-build-'))
    whl_path = out_dir / whl_filename

    records: list[str] = []

    with zipfile.ZipFile(whl_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        # Package files
        for file_path in sorted(source.rglob('*')):
            if not file_path.is_file():
                continue
            if _is_excluded(file_path, source, exclude_patterns):
                continue
            arcname = f'{source.name}/{file_path.relative_to(source)}'
            data = file_path.read_bytes()
            zf.writestr(arcname, data)
            records.append(_record_entry(arcname, data))

        # dist-info files (skip RECORD — we regenerate it)
        for file_path in sorted(dist_info.rglob('*')):
            if not file_path.is_file() or file_path.name == 'RECORD':
                continue
            arcname = f'{dist_info_name}/{file_path.relative_to(dist_info)}'
            data = file_path.read_bytes()
            zf.writestr(arcname, data)
            records.append(_record_entry(arcname, data))

        # RECORD itself (listed without hash per spec)
        record_arcname = f'{dist_info_name}/RECORD'
        records.append(f'{record_arcname},,')
        zf.writestr(record_arcname, '\n'.join(records) + '\n')

    return whl_path


def _parse_pkg_metadata(metadata_path: Path) -> dict[str, str]:
    """Extract ``Name`` and ``Version`` from a dist-info ``METADATA`` file."""
    if not metadata_path.exists():
        raise RuntimeError(f'No METADATA found at {metadata_path}')
    result: dict[str, str] = {}
    for line in metadata_path.read_text().splitlines():
        if ':' not in line:
            continue
        key, _, value = line.partition(':')
        key_lower = key.strip().lower()
        if key_lower == 'name' and 'name' not in result:
            result['name'] = value.strip()
        elif key_lower == 'version' and 'version' not in result:
            result['version'] = value.strip()
        if 'name' in result and 'version' in result:
            break
    if 'name' not in result or 'version' not in result:
        raise RuntimeError(f'METADATA at {metadata_path} missing Name or Version')
    return result


# ---------------------------------------------------------------------------
# Mode 2: project with pyproject.toml
# ---------------------------------------------------------------------------


def _load_toml(path: Path) -> dict:
    """Load a TOML file, returning an empty dict on failure."""
    try:
        import tomllib
    except ModuleNotFoundError:
        try:
            import tomli as tomllib
        except ModuleNotFoundError:
            return {}
    try:
        with open(path, 'rb') as f:
            return tomllib.load(f)
    except Exception:
        return {}


def _read_pyproject(path: Path) -> dict:
    """Read ``name``, ``version``, ``dependencies``, and ``optional-dependencies`` from pyproject.toml."""
    data = _load_toml(path)
    project = data.get('project', {})
    result: dict = {}
    if 'name' in project:
        result['name'] = project['name']
    if 'version' in project:
        result['version'] = project['version']
    if 'dependencies' in project:
        result['dependencies'] = project['dependencies']
    if 'optional-dependencies' in project:
        result['optional_dependencies'] = project['optional-dependencies']
    return result


def _read_build_packages(source: Path) -> list[Path] | None:
    """Read explicit package declarations from pyproject.toml build backends.

    Checks Hatch, Setuptools, and Flit configurations in priority order.
    Returns ``None`` if no explicit declaration is found (signals auto-discovery).
    """
    pyproject = source / 'pyproject.toml'
    if not pyproject.exists():
        return None

    data = _load_toml(pyproject)
    if not data:
        return None

    # Hatch: [tool.hatch.build.targets.wheel].packages
    hatch_pkgs = (
        data.get('tool', {}).get('hatch', {}).get('build', {}).get('targets', {}).get('wheel', {}).get('packages')
    )
    if hatch_pkgs and isinstance(hatch_pkgs, list):
        resolved = [source / p for p in hatch_pkgs if (source / p).is_dir()]
        return resolved or None

    # Setuptools: [tool.setuptools.packages.find]
    find_cfg = data.get('tool', {}).get('setuptools', {}).get('packages', {})
    if isinstance(find_cfg, dict) and 'find' in find_cfg:
        find = find_cfg['find']
        where = find.get('where', ['.'])
        include = find.get('include', ['*'])
        exclude = find.get('exclude', [])
        search_roots = [source / w for w in (where if isinstance(where, list) else [where])]
        packages: list[Path] = []
        for root in search_roots:
            if not root.is_dir():
                continue
            for child in sorted(root.iterdir()):
                if not child.is_dir() or not (child / '__init__.py').exists():
                    continue
                name = child.name
                if any(_simple_match(name, p) for p in exclude):
                    continue
                if any(_simple_match(name, p) for p in include):
                    packages.append(child)
        return packages or None

    # Setuptools: explicit list [tool.setuptools.packages] = [...]
    if isinstance(find_cfg, list):
        resolved = [source / p.replace('.', '/') for p in find_cfg]
        return [p for p in resolved if p.is_dir()] or None

    # Flit: [tool.flit.module].name
    flit_name = data.get('tool', {}).get('flit', {}).get('module', {}).get('name')
    if flit_name:
        pkg = source / flit_name
        if pkg.is_dir():
            return [pkg]
        # src layout
        pkg = source / 'src' / flit_name
        if pkg.is_dir():
            return [pkg]

    return None


def _simple_match(name: str, pattern: str) -> bool:
    """Minimal glob-style match for setuptools include/exclude patterns."""
    if pattern == '*':
        return True
    if '*' not in pattern:
        return name == pattern
    # prefix* or *suffix
    if pattern.endswith('*'):
        return name.startswith(pattern[:-1])
    if pattern.startswith('*'):
        return name.endswith(pattern[1:])
    return name == pattern


def _load_ignore_spec(source: Path):
    """Load .gitignore and .dockerignore patterns as a pathspec matcher.

    Returns ``None`` if pathspec is unavailable or no ignore files exist.
    """
    try:
        import pathspec
    except ImportError:
        return None

    lines: list[str] = []
    for ignore_file in ('.gitignore', '.dockerignore'):
        path = source / ignore_file
        if path.exists():
            lines.extend(path.read_text().splitlines())

    if not lines:
        return None

    return pathspec.PathSpec.from_lines('gitwildmatch', lines)


def _discover_packages(source: Path) -> list[Path]:
    """Auto-discover Python packages in a project directory.

    Strategy (checked in order):
    1. Explicit build-backend declarations in pyproject.toml (Hatch/Setuptools/Flit)
    2. ``__init__.py``-based discovery filtered by .gitignore + .dockerignore rules
    3. Pure ``__init__.py`` heuristic (fallback when no ignore files exist)
    """
    # Strategy 1: explicit declarations
    explicit = _read_build_packages(source)
    if explicit is not None:
        return explicit

    # Determine search root (src-layout vs flat)
    src_dir = source / 'src'
    search_root = src_dir if src_dir.is_dir() else source

    # Strategy 2 & 3: auto-discovery with optional ignore filtering
    ignore_spec = _load_ignore_spec(source)

    packages: list[Path] = []
    for child in sorted(search_root.iterdir()):
        if not child.is_dir():
            continue
        if child.name.startswith(('.', '_')):
            continue
        if child.name == '__pycache__':
            continue
        if not (child / '__init__.py').exists():
            continue
        # Apply ignore rules if available
        if ignore_spec is not None:
            rel = child.relative_to(source)
            # pathspec expects forward-slash paths with trailing / for directories
            if ignore_spec.match_file(str(rel) + '/'):
                continue
        packages.append(child)
    return packages


# ---------------------------------------------------------------------------
# Shared wheel-building helpers
# ---------------------------------------------------------------------------


def _build_wheel(
    name: str,
    version: str,
    package_dirs: list[Path],
    exclude_patterns: list[str] | None = None,
    *,
    dependencies: list[str] | None = None,
    optional_dependencies: dict[str, list[str]] | None = None,
) -> Path:
    """Build a pure-Python wheel from one or more package directories."""
    normalized = name.replace('-', '_')
    tag = 'py3-none-any'
    dist_info_name = f'{normalized}-{version}.dist-info'
    whl_filename = f'{normalized}-{version}-{tag}.whl'

    out_dir = Path(tempfile.mkdtemp(prefix='nexus-build-'))
    whl_path = out_dir / whl_filename

    records: list[str] = []

    with zipfile.ZipFile(whl_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        # Package files
        for pkg_dir in package_dirs:
            for file_path in sorted(pkg_dir.rglob('*')):
                if not file_path.is_file():
                    continue
                if _is_excluded(file_path, pkg_dir, exclude_patterns):
                    continue
                arcname = f'{pkg_dir.name}/{file_path.relative_to(pkg_dir)}'
                data = file_path.read_bytes()
                zf.writestr(arcname, data)
                records.append(_record_entry(arcname, data))

        # METADATA
        metadata = _generate_metadata(name, version, dependencies, optional_dependencies)
        metadata_bytes = metadata.encode()
        arcname = f'{dist_info_name}/METADATA'
        zf.writestr(arcname, metadata_bytes)
        records.append(_record_entry(arcname, metadata_bytes))

        # WHEEL
        wheel = _generate_wheel_file(tag)
        wheel_bytes = wheel.encode()
        arcname = f'{dist_info_name}/WHEEL'
        zf.writestr(arcname, wheel_bytes)
        records.append(_record_entry(arcname, wheel_bytes))

        # top_level.txt
        top_level = '\n'.join(d.name for d in package_dirs) + '\n'
        top_level_bytes = top_level.encode()
        arcname = f'{dist_info_name}/top_level.txt'
        zf.writestr(arcname, top_level_bytes)
        records.append(_record_entry(arcname, top_level_bytes))

        # RECORD (self-entry has no hash per spec)
        record_arcname = f'{dist_info_name}/RECORD'
        records.append(f'{record_arcname},,')
        zf.writestr(record_arcname, '\n'.join(records) + '\n')

    return whl_path


def _generate_metadata(
    name: str,
    version: str,
    dependencies: list[str] | None = None,
    optional_dependencies: dict[str, list[str]] | None = None,
) -> str:
    lines = ['Metadata-Version: 2.1', f'Name: {name}', f'Version: {version}']
    for dep in dependencies or []:
        lines.append(f'Requires-Dist: {dep}')
    if optional_dependencies:
        for extra_name, extra_deps in sorted(optional_dependencies.items()):
            lines.append(f'Provides-Extra: {extra_name}')
            for dep in extra_deps:
                requirement, separator, marker = dep.partition(';')
                if separator:
                    lines.append(
                        f'Requires-Dist: {requirement.strip()}; ({marker.strip()}) and extra == "{extra_name}"'
                    )
                else:
                    lines.append(f'Requires-Dist: {dep}; extra == "{extra_name}"')
    return '\n'.join(lines) + '\n'


def _generate_wheel_file(tag: str) -> str:
    return f'Wheel-Version: 1.0\nGenerator: nexus\nRoot-Is-Purelib: true\nTag: {tag}\n'


def _record_entry(arcname: str, data: bytes) -> str:
    """Generate a single RECORD line: ``path,sha256=...,size``."""
    digest = hashlib.sha256(data).digest()
    b64 = base64.urlsafe_b64encode(digest).rstrip(b'=').decode('ascii')
    return f'{arcname},sha256={b64},{len(data)}'


__all__ = ['build_artifact', 'build_artifact_cached']
