"""Dependency checker for nexus extension modules.

Reads declared dependencies from module metadata (pyproject.toml / PEP 723)
and checks whether they are installed in the current environment.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys

from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path

from nexus.core.module_meta import ModuleMeta, discover_all_modules


logger = logging.getLogger(__name__)

# Map pip package names → importable top-level module names
# Only needed when they differ (e.g. Pillow → PIL)
_PACKAGE_TO_IMPORT: dict[str, str] = {
    'pillow': 'PIL',
    'polaris-cpp-py': 'polaris',
    'tencentcloud-sdk-python': 'tencentcloud',
    'python-dateutil': 'dateutil',
}


@dataclass
class DepStatus:
    """Result of checking one module's dependencies."""

    module: ModuleMeta
    missing_base: list[str] = field(default_factory=list)
    missing_host_extras: list[str] = field(default_factory=list)
    missing_runtime_extras: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.missing_base and not self.missing_host_extras

    @property
    def missing_host(self) -> list[str]:
        """All missing packages for host environment (base + host extras)."""
        return self.missing_base + self.missing_host_extras

    @property
    def missing_runtime(self) -> list[str]:
        """All missing packages for runtime environment (base + runtime extras)."""
        return self.missing_base + self.missing_runtime_extras


def _normalize_pkg_name(dep: str) -> str:
    """Extract the bare package name from a PEP 508 dependency string."""
    # Strip version specifiers, extras, environment markers
    for ch in '>=<!;[':
        idx = dep.find(ch)
        if idx != -1:
            dep = dep[:idx]
    return dep.strip().lower().replace('-', '-')


def _is_installed(dep: str) -> bool:
    """Check whether a dependency is installed."""
    pkg = _normalize_pkg_name(dep)
    try:
        distribution(pkg)
        return True
    except PackageNotFoundError:
        pass
    # Try common name transformations
    import_name = _PACKAGE_TO_IMPORT.get(pkg)
    if import_name:
        try:
            distribution(import_name)
            return True
        except PackageNotFoundError:
            pass
    return False


def check_module(meta: ModuleMeta) -> DepStatus:
    """Check whether a module's declared dependencies are installed."""
    status = DepStatus(module=meta)
    for dep in meta.base_deps:
        if not _is_installed(dep):
            status.missing_base.append(dep)
    for dep in meta.host_extras:
        if not _is_installed(dep):
            status.missing_host_extras.append(dep)
    for dep in meta.runtime_extras:
        if not _is_installed(dep):
            status.missing_runtime_extras.append(dep)
    return status


def check_all(project_root: Path | None = None) -> list[DepStatus]:
    """Check all modules and return their dependency statuses."""
    modules = discover_all_modules(project_root)
    return [check_module(m) for m in modules]


# ---------------------------------------------------------------------------
# Package installation
# ---------------------------------------------------------------------------


def _find_installer() -> tuple[str, ...] | None:
    """Return the command prefix for installing packages.

    Prefers ``uv pip install``, falls back to ``python -m pip install``.

    Both forms target ``sys.executable`` explicitly. Without ``--python``, uv resolves
    the target from the shell's activated environment (``VIRTUAL_ENV``, then
    ``CONDA_PREFIX``), which on a venv-on-conda layout installs into the conda env while
    Nexus keeps running from the project venv: the install reports success and the import
    still fails.
    """
    uv = shutil.which('uv')
    if uv:
        return (uv, 'pip', 'install', '--python', sys.executable)
    # Fallback: current interpreter's pip
    return (sys.executable, '-m', 'pip', 'install')


def install_packages(packages: list[str]) -> tuple[bool, str]:
    """Install *packages* via uv/pip.

    Returns ``(success, message)``.
    """
    cmd_prefix = _find_installer()
    if cmd_prefix is None:
        return False, 'Neither uv nor pip found'

    try:
        result = subprocess.run(
            [*cmd_prefix, *packages],
            capture_output=True,
            text=True,
            timeout=300,
        )
    except subprocess.TimeoutExpired:
        return False, 'Install timed out (300s)'
    except Exception as exc:
        return False, str(exc)

    if result.returncode != 0:
        stderr = result.stderr.strip()
        return False, stderr[:500] if stderr else f'exit code {result.returncode}'
    return True, result.stdout.strip()
