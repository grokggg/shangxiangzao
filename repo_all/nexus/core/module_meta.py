"""Module metadata reader for nexus extensions.

Reads per-module metadata from two standard formats:

- **Directory modules**: ``pyproject.toml`` in the module directory
- **Single-file modules**: PEP 723 inline script metadata (``# /// script`` blocks)

Both formats declare:

- ``[project].authors`` / ``[project].maintainers``: module authorship metadata
- ``dependencies``: base deps shared between host and runtime
- ``optional-dependencies.host``: host-only extras
- ``optional-dependencies.runtime``: runtime-only extras
- ``[tool.nexus]``: registered objects (agents, tasks, tools, etc.)
"""

from __future__ import annotations

import logging
import re
import zipfile

from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Literal


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModuleMeta:
    """Parsed metadata for one nexus module."""

    name: str  # e.g. "swe_agent"
    module_path: str  # dotted import path, e.g. "nexus.extensions.agents.swe_agent"
    fs_path: Path  # absolute path to dir or .py file
    description: str = ''
    version: str = '0.1.0'
    authors: tuple[dict[str, str], ...] = ()
    maintainers: tuple[dict[str, str], ...] = ()
    base_deps: tuple[str, ...] = ()  # [project].dependencies (shared between host and runtime)
    host_extras: tuple[str, ...] = ()  # [project.optional-dependencies].host
    runtime_extras: tuple[str, ...] = ()  # [project.optional-dependencies].runtime
    objects: dict[str, dict[str, str]] = field(default_factory=dict)  # registry_type -> {obj_name: import_path}
    kind: Literal['directory', 'single_file'] = 'directory'

    @property
    def host_deps(self) -> list[str]:
        """Host total = base + host extras."""
        return list(self.base_deps) + list(self.host_extras)

    @property
    def all_runtime_deps(self) -> list[str]:
        """Runtime total = base + runtime extras."""
        return list(self.base_deps) + list(self.runtime_extras)

    @property
    def category(self) -> str:
        """Infer the primary category from the module path."""
        p = self.module_path
        if '.agents.' in p or p.endswith('.agents'):
            return 'Agent'
        if '.tasks.' in p or p.endswith('.tasks'):
            return 'Task'
        if '.tools.' in p or p.endswith('.tools'):
            return 'Tool'
        if '.openai_clients.' in p or p.endswith('.openai_clients'):
            return 'LLM Client'
        if '.executors.' in p or p.endswith('.executors'):
            return 'Executor'
        if '.providers.' in p or p.endswith('.providers'):
            return 'Provider'
        if '.processors.' in p or p.endswith('.processors'):
            return 'Processor'
        if '.projects.' in p or p.endswith('.projects'):
            return 'Project'
        return 'Other'

    @property
    def object_type_tags(self) -> list[str]:
        """Return the distinct object type keys present (e.g. ['agents', 'tools'])."""
        return list(self.objects.keys())


# ---------------------------------------------------------------------------
# TOML loading (Python 3.11 tomllib or tomli fallback)
# ---------------------------------------------------------------------------


def _load_toml(path: Path) -> dict:
    """Load a TOML file, returning an empty dict on any failure."""
    try:
        import tomllib
    except ModuleNotFoundError:
        try:
            import tomli as tomllib
        except ModuleNotFoundError:
            logger.warning('Neither tomllib nor tomli available — cannot read %s', path)
            return {}
    try:
        with open(path, 'rb') as f:
            return tomllib.load(f)
    except Exception:
        logger.warning('Failed to parse TOML: %s', path, exc_info=True)
        return {}


def _parse_toml_string(text: str) -> dict:
    """Parse a TOML string, returning an empty dict on failure."""
    try:
        import tomllib
    except ModuleNotFoundError:
        try:
            import tomli as tomllib
        except ModuleNotFoundError:
            return {}
    try:
        return tomllib.loads(text)
    except Exception:
        logger.warning('Failed to parse TOML string', exc_info=True)
        return {}


def _project_people(project: dict, key: str) -> tuple[dict[str, str], ...]:
    """Return normalised PEP 621 people entries from ``authors`` or ``maintainers``."""
    people = project.get(key, [])
    if not isinstance(people, list):
        logger.warning('[project].%s should be a list of inline tables', key)
        return ()

    normalised: list[dict[str, str]] = []
    for person in people:
        if not isinstance(person, dict):
            logger.warning('[project].%s entry should be an inline table: %r', key, person)
            continue
        entry: dict[str, str] = {}
        for field_name in ('name', 'email'):
            value = person.get(field_name)
            if isinstance(value, str) and value:
                entry[field_name] = value
        if entry:
            normalised.append(entry)
    return tuple(normalised)


def _normalise_tool_nexus_objects(tool_nexus: dict[str, Any], *, meta_name: str) -> dict[str, dict[str, str]]:
    """Return registry object maps."""

    objects: dict[str, dict[str, str]] = {}
    for nexus_key, obj_map in tool_nexus.items():
        if not isinstance(obj_map, dict):
            logger.warning('[tool.nexus].%s in %s should be a table', nexus_key, meta_name)
            continue
        if nexus_key == 'run_parts':
            objects[nexus_key] = _normalise_run_parts(obj_map, meta_name=meta_name)
            continue

        normalised: dict[str, str] = {}
        for obj_name, import_path in obj_map.items():
            if isinstance(import_path, str):
                normalised[str(obj_name)] = import_path
            else:
                logger.warning(
                    '[tool.nexus].%s.%s in %s should be an import path string', nexus_key, obj_name, meta_name
                )
        objects[nexus_key] = normalised
    return objects


def _normalise_run_parts(obj_map: dict[str, Any], *, meta_name: str) -> dict[str, str]:
    """Normalize run part metadata entries into registry-ready ``name -> import`` mappings."""

    normalised: dict[str, str] = {}
    for obj_name, raw_value in obj_map.items():
        name = str(obj_name)
        if not isinstance(raw_value, str) or not raw_value:
            logger.warning('[tool.nexus].run_parts.%s in %s should be an import path string', name, meta_name)
            continue

        _add_object_mapping(normalised, name, raw_value, meta_name=meta_name)
    return normalised


def _add_object_mapping(objects: dict[str, str], name: str, import_path: str, *, meta_name: str) -> None:
    existing = objects.get(name)
    if existing is not None and existing != import_path:
        logger.warning('Duplicate object %s in %s maps to both %s and %s', name, meta_name, existing, import_path)
        return
    objects[name] = import_path


# ---------------------------------------------------------------------------
# Directory module parser
# ---------------------------------------------------------------------------


def _path_to_module_path(fs_path: Path, project_root: Path) -> str:
    """Convert an absolute filesystem path to a dotted module path.

    ``/project/nexus/extensions/agents/swe_agent`` → ``nexus.extensions.agents.swe_agent``
    """
    try:
        rel = fs_path.relative_to(project_root)
    except ValueError:
        return fs_path.name
    return '.'.join(rel.parts)


def _meta_from_toml_data(
    data: dict,
    *,
    fallback_name: str,
    module_path: str,
    fs_path: Path,
    kind: Literal['directory', 'single_file'] = 'directory',
) -> ModuleMeta | None:
    """Build a :class:`ModuleMeta` from parsed TOML data.

    Returns ``None`` if the data has no ``[tool.nexus]`` section.
    """
    tool_nexus = data.get('tool', {}).get('nexus', {})
    if not tool_nexus:
        return None

    project = data.get('project', {})
    optional_deps = project.get('optional-dependencies', {})

    name = project.get('name', fallback_name)
    return ModuleMeta(
        name=name,
        module_path=module_path,
        fs_path=fs_path,
        description=project.get('description', ''),
        version=project.get('version', '0.1.0'),
        authors=_project_people(project, 'authors'),
        maintainers=_project_people(project, 'maintainers'),
        base_deps=tuple(project.get('dependencies', [])),
        host_extras=tuple(optional_deps.get('host', [])),
        runtime_extras=tuple(optional_deps.get('runtime', [])),
        objects=_normalise_tool_nexus_objects(tool_nexus, meta_name=name),
        kind=kind,
    )


def parse_directory_module(dir_path: Path, project_root: Path) -> ModuleMeta | None:
    """Read a directory module's ``pyproject.toml`` and return :class:`ModuleMeta`.

    Returns ``None`` if no ``pyproject.toml`` exists.
    """
    toml_path = dir_path / 'pyproject.toml'
    if not toml_path.is_file():
        return None

    data = _load_toml(toml_path)
    if not data:
        return None

    return _meta_from_toml_data(
        data,
        fallback_name=dir_path.name,
        module_path=_path_to_module_path(dir_path, project_root),
        fs_path=dir_path.resolve(),
        kind='directory',
    )


# ---------------------------------------------------------------------------
# PEP 723 single-file parser
# ---------------------------------------------------------------------------

_PEP723_PATTERN = re.compile(
    r'^# /// script\s*\n((?:#[^\n]*\n)*?)# ///',
    re.MULTILINE,
)


def _extract_pep723_block(source: str) -> str | None:
    """Extract the TOML content from a PEP 723 ``# /// script`` block."""
    m = _PEP723_PATTERN.search(source)
    if m is None:
        return None
    lines = m.group(1).splitlines()
    # Strip leading '# ' or '#' from each line
    stripped: list[str] = []
    for line in lines:
        if line.startswith('# '):
            stripped.append(line[2:])
        elif line == '#':
            stripped.append('')
        else:
            stripped.append(line.lstrip('# '))
    return '\n'.join(stripped)


def _meta_from_pep723_data(
    data: dict,
    *,
    fallback_name: str,
    module_path: str,
    fs_path: Path,
) -> ModuleMeta | None:
    """Build a :class:`ModuleMeta` from parsed PEP 723 TOML data.

    Returns ``None`` if the data has no ``[tool.nexus]`` section.
    """
    tool_nexus = data.get('tool', {}).get('nexus', {})
    if not tool_nexus:
        return None

    project = data.get('project', {})
    optional_deps = data.get('optional-dependencies', {})
    name = project.get('name', fallback_name)
    return ModuleMeta(
        name=name,
        module_path=module_path,
        fs_path=fs_path,
        description=project.get('description', data.get('description', '')),
        version=project.get('version', data.get('version', '0.1.0')),
        authors=_project_people(project, 'authors'),
        maintainers=_project_people(project, 'maintainers'),
        base_deps=tuple(data.get('dependencies', [])),
        host_extras=tuple(optional_deps.get('host', [])),
        runtime_extras=tuple(optional_deps.get('runtime', [])),
        objects=_normalise_tool_nexus_objects(tool_nexus, meta_name=name),
        kind='single_file',
    )


def parse_pep723_metadata(file_path: Path, project_root: Path) -> ModuleMeta | None:
    """Read PEP 723 inline script metadata from a single ``.py`` file.

    Returns ``None`` if the file has no ``# /// script`` block.
    """
    try:
        source = file_path.read_text(encoding='utf-8')
    except OSError:
        return None

    toml_text = _extract_pep723_block(source)
    if toml_text is None:
        return None

    data = _parse_toml_string(toml_text)
    if not data:
        return None

    # Module path: file relative to project root, without .py suffix
    rel = file_path.relative_to(project_root)
    module_path = '.'.join(rel.with_suffix('').parts)

    return _meta_from_pep723_data(
        data,
        fallback_name=file_path.stem,
        module_path=module_path,
        fs_path=file_path.resolve(),
    )


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

# Module roots relative to the ``nexus/`` package directory.
# Each root is scanned for both directory modules (pyproject.toml with
# [tool.nexus]) and single-file modules (PEP 723 with [tool.nexus]).
# Adding a new module category only requires appending to this tuple.
MODULE_ROOTS: tuple[str, ...] = (
    'gateway',
    'gateway/observers',
    'extensions',  # Root-level composite integrations such as harbor_bridge.
    'extensions/agents',
    'extensions/tasks',
    'extensions/tools',
    'extensions/projects',
    'extensions/openai_clients',
    'runtime/providers',
    'runtime/executors',
    'runner/compose',
    'trace/processors',
)


def _get_nexus_root() -> Path:
    """Return the ``nexus/`` package directory."""
    import nexus as _nexus_pkg

    return Path(_nexus_pkg.__file__).parent


def _scan_root(root_dir: Path, project_root: Path) -> list[ModuleMeta]:
    """Scan a single module root for directory and single-file modules.

    For each child of *root_dir*:
    - Directories: try ``parse_directory_module`` (needs ``pyproject.toml``
      with ``[tool.nexus]``).
    - ``.py`` files: try ``parse_pep723_metadata`` (needs ``# /// script``
      block with ``[tool.nexus]``).

    Files without nexus metadata are silently skipped — no explicit exclude
    list is needed.
    """
    if not root_dir.is_dir():
        return []

    modules: list[ModuleMeta] = []
    for child in sorted(root_dir.iterdir()):
        if child.name.startswith(('_', '.')):
            continue
        if child.is_dir():
            meta = parse_directory_module(child, project_root)
        elif child.suffix == '.py':
            meta = parse_pep723_metadata(child, project_root)
        else:
            continue
        if meta is not None:
            modules.append(meta)
    return modules


def _dedupe_modules(modules: list[ModuleMeta]) -> list[ModuleMeta]:
    """Keep the first module for each module path."""
    seen: set[str] = set()
    result: list[ModuleMeta] = []
    for meta in modules:
        if meta.module_path in seen:
            continue
        seen.add(meta.module_path)
        result.append(meta)
    return result


def discover_all_modules(
    project_root: Path | None = None,
    *,
    include_compiled_manifest: bool = True,
    include_source: bool = True,
) -> list[ModuleMeta]:
    """Discover all nexus modules from :data:`MODULE_ROOTS`.

    Each root is scanned for directory modules (``pyproject.toml``) and
    single-file modules (PEP 723).  A child is recognised as a module only
    if its metadata contains a ``[tool.nexus]`` section.

    Args:
        project_root: Parent directory of the ``nexus/`` package.
            If *None*, derived from ``nexus.__file__``.  Works both in
            a source checkout and after ``pip install``.
    """
    modules: list[ModuleMeta] = []

    if include_compiled_manifest:
        modules.extend(discover_modules_from_compiled_manifest())

    if include_source:
        if project_root is None:
            nexus_root = _get_nexus_root()
            project_root = nexus_root.parent
        else:
            project_root = project_root.resolve()
            nexus_root = project_root / 'nexus'

        for rel_root in MODULE_ROOTS:
            modules.extend(_scan_root(nexus_root / rel_root, project_root))

    return _dedupe_modules(modules)


# Build a set of expected path prefixes inside the wheel, e.g. "nexus/extensions/agents/"
_WHL_MODULE_PREFIXES: tuple[str, ...] = tuple(f'nexus/{root}/' for root in MODULE_ROOTS)


def _is_direct_wheel_module_entry(entry_name: str) -> bool:
    """Mirror source discovery by accepting only direct children of a module root."""
    for prefix in _WHL_MODULE_PREFIXES:
        if not entry_name.startswith(prefix):
            continue
        relative = entry_name.removeprefix(prefix)
        if relative.endswith('.py') and '/' not in relative:
            return True
        parts = PurePosixPath(relative).parts
        if len(parts) == 2 and parts[-1] == 'pyproject.toml':
            return True
    return False


def discover_modules_from_whl(whl_path: Path) -> list[ModuleMeta]:
    """Discover nexus modules from ``pyproject.toml`` files inside a wheel.

    Scans for paths matching :data:`MODULE_ROOTS` patterns (e.g.
    ``nexus/extensions/tools/file_editor/pyproject.toml``).  Only entries
    with a ``[tool.nexus]`` section are returned.

    This is the wheel-based counterpart of :func:`discover_all_modules`.
    """
    modules: list[ModuleMeta] = []
    with zipfile.ZipFile(whl_path) as zf:
        for entry_name in zf.namelist():
            if not _is_direct_wheel_module_entry(entry_name):
                continue

            if entry_name.endswith('.py'):
                try:
                    toml_text = _extract_pep723_block(zf.read(entry_name).decode())
                except Exception:
                    continue
                if toml_text is None:
                    continue
                data = _parse_toml_string(toml_text)
                if not data:
                    continue
                module_path = entry_name.removesuffix('.py').replace('/', '.')
                meta = _meta_from_pep723_data(
                    data,
                    fallback_name=PurePosixPath(entry_name).stem,
                    module_path=module_path,
                    fs_path=whl_path,
                )
                if meta is not None:
                    modules.append(meta)
                continue

            if not entry_name.endswith('/pyproject.toml'):
                continue

            try:
                data = _parse_toml_string(zf.read(entry_name).decode())
            except Exception:
                continue
            if not data:
                continue

            # Derive module_path from zip entry:
            # "nexus/extensions/tools/file_editor/pyproject.toml"
            #  → "nexus.extensions.tools.file_editor"
            dir_part = entry_name.rsplit('/pyproject.toml', 1)[0]
            module_path = dir_part.replace('/', '.')
            fallback_name = dir_part.rsplit('/', 1)[-1]

            meta = _meta_from_toml_data(
                data,
                fallback_name=fallback_name,
                module_path=module_path,
                fs_path=whl_path,
                kind='directory',
            )
            if meta is not None:
                modules.append(meta)
    return modules


def modules_from_manifest_records(records: list[dict]) -> list[ModuleMeta]:
    """Convert compiled manifest records into :class:`ModuleMeta` objects."""
    modules: list[ModuleMeta] = []
    for record in records:
        name = str(record['name'])
        module_path = str(record['module_path'])
        kind = record.get('kind', 'directory')
        if kind not in ('directory', 'single_file'):
            kind = 'directory'
        modules.append(
            ModuleMeta(
                name=name,
                module_path=module_path,
                fs_path=Path(f'<compiled:{module_path}>'),
                description=str(record.get('description', '')),
                version=str(record.get('version', '0.1.0')),
                authors=tuple(record.get('authors', ())),
                maintainers=tuple(record.get('maintainers', ())),
                base_deps=tuple(record.get('base_deps', ())),
                host_extras=tuple(record.get('host_extras', ())),
                runtime_extras=tuple(record.get('runtime_extras', ())),
                objects=_normalise_tool_nexus_objects(dict(record.get('objects', {})), meta_name=name),
                kind=kind,
            )
        )
    return modules


def discover_modules_from_compiled_manifest() -> list[ModuleMeta]:
    """Discover built-in Nexus modules from the generated Nuitka manifest."""
    try:
        from nexus import _compiled_module_manifest
    except ImportError:
        return []

    records = getattr(_compiled_module_manifest, 'COMPILED_MODULES', [])
    return modules_from_manifest_records(records)


# ---------------------------------------------------------------------------
# Registry key mapping
# ---------------------------------------------------------------------------

# [tool.nexus] key → registry type_name
NEXUS_KEY_TO_REGISTRY_TYPE: dict[str, str] = {
    'agents': 'agent',
    'tasks': 'task',
    'tools': 'tool',
    'openai_clients': 'openai_client',
    'executors': 'executor',
    'runtime_providers': 'runtime_provider',
    'span_processors': 'span_processor',
    'trajectory_observers': 'trajectory_observer',
    'run_parts': 'run_part',
    'dataset_sources': 'dataset_source',
    'commands': 'command',
}


def _get_registry(type_name: str):
    """Get a registry singleton by its type_name. Imports lazily to avoid cycles."""
    if type_name == 'agent':
        from nexus.core import agent_registry

        return agent_registry
    if type_name == 'task':
        from nexus.core import task_registry

        return task_registry
    if type_name == 'tool':
        from nexus.core import tool_registry

        return tool_registry
    if type_name == 'openai_client':
        from nexus.core import openai_client_registry

        return openai_client_registry
    if type_name == 'executor':
        from nexus.core import executor_registry

        return executor_registry
    if type_name == 'command':
        from nexus.core import command_registry

        return command_registry
    if type_name == 'runtime_provider':
        from nexus.runtime.providers.registry import runtime_provider_registry

        return runtime_provider_registry
    if type_name == 'span_processor':
        from nexus.trace.processors.base import span_processor_registry

        return span_processor_registry
    if type_name == 'trajectory_observer':
        from nexus.gateway.observers.registry import trajectory_observer_registry

        return trajectory_observer_registry
    if type_name == 'run_part':
        from nexus.runner.compose.registry import run_part_registry

        return run_part_registry
    if type_name == 'dataset_source':
        from nexus.dataset.source import dataset_source_registry

        return dataset_source_registry
    raise ValueError(f'Unknown registry type: {type_name}')


def register_objects_from_meta(meta: ModuleMeta) -> None:
    """Call ``registry.add()`` for every object declared in a :class:`ModuleMeta`."""
    for nexus_key, obj_map in meta.objects.items():
        type_name = NEXUS_KEY_TO_REGISTRY_TYPE.get(nexus_key)
        if type_name is None:
            logger.warning('Unknown [tool.nexus] key %r in %s', nexus_key, meta.name)
            continue
        registry = _get_registry(type_name)
        for obj_name, import_path in obj_map.items():
            registry.add(obj_name, import_path)
