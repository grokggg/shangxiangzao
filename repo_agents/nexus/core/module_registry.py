"""Module registry for nexus extensions.

Provides a first-class "module" concept — each extension directory
(e.g., ``agents/swe_agent``, ``tasks/vita``) is a self-describing module
that declares its objects in ``pyproject.toml`` (or PEP 723 for single files).

``module_registry`` catalogs all modules for dependency management,
discovery, and the ``nexus modules`` CLI.  The primary entry point is
``discover_and_register()`` which scans for metadata files and populates
the object registries without importing any extension code.
"""

from __future__ import annotations

import logging
import os

from pathlib import Path


logger = logging.getLogger(__name__)

AUTO_INSTALL_ENV = 'NEXUS_AUTO_INSTALL'


class ModuleRegistry:
    """Catalog of extension modules.

    Modules are discovered from ``pyproject.toml`` / PEP 723 metadata
    via ``discover_and_register()``.  Object registries (agent, task, tool, …)
    are populated with lazy ``registry.add()`` entries pointing at the
    declared import paths — no extension code is imported at discovery time.
    """

    def __init__(self) -> None:
        from nexus.core.module_meta import ModuleMeta

        self._modules: dict[str, ModuleMeta] = {}  # keyed by module_path

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def discover_and_register(self, project_root: Path | None = None) -> None:
        """Scan for metadata files and register all declared objects.

        1. Calls :func:`discover_all_modules` — filesystem scan + TOML parse, NO imports.
        2. For each :class:`ModuleMeta`, calls ``registry.add()`` on the correct registries.
        """
        from nexus.core.module_meta import discover_all_modules, register_objects_from_meta

        modules = discover_all_modules(project_root)

        for meta in modules:
            if meta.module_path in self._modules:
                logger.debug('Module %r already discovered, skipping', meta.module_path)
                continue
            self._modules[meta.module_path] = meta
            register_objects_from_meta(meta)

        logger.info(
            'Discovered %d modules (%d objects)',
            len(self._modules),
            sum(sum(len(v) for v in m.objects.values()) for m in self._modules.values()),
        )

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def list_modules(self) -> list:
        """Return all discovered :class:`ModuleMeta` entries."""
        return list(self._modules.values())

    def get_module(self, module_path: str):
        """Get a :class:`ModuleMeta` by its dotted module path."""
        return self._modules.get(module_path)

    def find_module_by_import_path(self, import_path: str):
        """Find the :class:`ModuleMeta` that contains *import_path*.

        For example, ``nexus.extensions.agents.swe_agent.agent`` matches
        the module whose ``module_path`` is ``nexus.extensions.agents.swe_agent``.
        """
        module_path = import_path.split('::')[0] if '::' in import_path else import_path
        for meta in self._modules.values():
            if module_path == meta.module_path or module_path.startswith(meta.module_path + '.'):
                return meta
        return None

    # ------------------------------------------------------------------
    # Auto-install
    # ------------------------------------------------------------------

    @staticmethod
    def auto_install_enabled() -> bool:
        """Check whether auto-install is enabled via ``NEXUS_AUTO_INSTALL=1``."""
        return os.getenv(AUTO_INSTALL_ENV, '').strip() in ('1', 'true', 'yes')

    def try_auto_install(self, import_path: str) -> bool:
        """Find and install missing host deps for the module containing *import_path*.

        Returns ``True`` if packages were installed successfully (caller should retry import).
        Returns ``False`` if no module found, no deps missing, or install failed.
        """
        meta = self.find_module_by_import_path(import_path)
        if meta is None or not meta.host_deps:
            return False

        from nexus.core.dep_checker import check_module, install_packages

        status = check_module(meta)
        if not status.missing_host:
            return False  # declared deps are all installed; problem is elsewhere

        logger.info(
            'Auto-installing %d missing dep(s) for module %s: %s',
            len(status.missing_host),
            meta.name,
            ', '.join(status.missing_host),
        )
        ok, msg = install_packages(status.missing_host)
        if not ok:
            logger.error('Auto-install failed for %s: %s', meta.name, msg)
            return False

        logger.info('Auto-install succeeded for module %s', meta.name)
        return True

    def format_install_hint(self, import_path: str) -> str | None:
        """Return a human-readable install hint for the module containing *import_path*.

        Returns ``None`` if no module or no missing deps found.
        """
        meta = self.find_module_by_import_path(import_path)
        if meta is None:
            return None

        from nexus.core.dep_checker import check_module

        status = check_module(meta)
        if not status.missing_host:
            return None

        missing = ', '.join(status.missing_host)
        return (
            f'Module {meta.name!r} requires missing packages: {missing}\n'
            f'  To install:  nexus modules install {meta.name}\n'
            f'  Or enable:   {AUTO_INSTALL_ENV}=1'
        )


module_registry = ModuleRegistry()
