import importlib
import logging

from collections.abc import Callable
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

from nexus.utils.state import GLOBAL_STATE


logger = logging.getLogger(__name__)

# Context variable to track registrations during import_module
_import_registrations: ContextVar[list[dict[str, str]] | None] = ContextVar('_import_registrations', default=None)


class RegistryError(Exception):
    """Base exception for registry errors."""

    pass


class RegistryNotFoundError(RegistryError):
    """Raised when a registry type is not found."""

    pass


class EntryNotFoundError(RegistryError):
    """Raised when an entry is not found in a registry."""

    pass


class DuplicateEntryError(RegistryError):
    """Raised when attempting to register a duplicate entry."""

    pass


class DuplicateRegistryError(RegistryError):
    """Raised when attempting to create a duplicate registry."""

    pass


T = TypeVar('T')


@dataclass
class ProcessorResult:
    """Result returned by registry processor functions.

    Attributes:
        wrapped_obj: The processed/wrapped object to register (optional, defaults to original obj)
        meta: Metadata dict for the registered object (optional, defaults to original meta)
    """

    wrapped_obj: Any | None = None
    meta: dict | None = None


@dataclass
class RegistryEntry:
    name: str
    obj: Any = None  # None means not yet loaded
    import_path: str | None = None  # For lazy loading
    meta: dict = field(default_factory=dict)
    install_commands: list[str] = field(default_factory=list)

    @property
    def is_loaded(self) -> bool:
        return self.obj is not None


class Registry(Generic[T]):
    """
    Type-safe registry for a specific type of objects.

    Supports:
    - Decorator-based registration
    - Lazy loading from import paths
    - Type safety with generics
    - Metadata attachment
    """

    def __init__(
        self,
        type_name: str,
        processor: Callable[[T, str, dict], ProcessorResult | None] | None = None,
    ):
        """
        Initialize a registry.

        Args:
            type_name: Name of the registry type (e.g., 'agent', 'tool')
            processor: Optional processor that validates/transforms the object and its metadata.
                      Signature: processor(obj, name, meta) -> ProcessorResult | None
                      Should raise an exception if validation fails.
                      Return None to keep obj and meta unchanged, or return ProcessorResult to replace them.
        """
        self.type_name = type_name
        self.processor = processor
        self._registry: dict[str, RegistryEntry] = {}

        if 'registries' not in GLOBAL_STATE:
            GLOBAL_STATE['registries'] = {}
        if type_name in GLOBAL_STATE['registries']:
            raise DuplicateRegistryError(f"Registry type '{type_name}' already exists.")
        GLOBAL_STATE['registries'][type_name] = self

    def register(self, name: str, meta: dict | None = None):
        """
        Decorator to register an object.

        Usage:
            @registry.register('my_func')
            def my_func():
                pass

        Args:
            name: Unique name for this entry
            meta: Optional metadata dict

        Returns:
            Decorated function
        """
        if meta is None:
            meta = {}

        def wrapper(obj: T) -> T:
            processed_meta = meta
            wrapped_obj = obj  # By default, use original object

            if self.processor:
                try:
                    result = self.processor(obj, name, meta)
                    if result is not None:
                        if not isinstance(result, ProcessorResult):
                            raise TypeError(f'Processor must return ProcessorResult, got {type(result).__name__}')
                        # Use wrapped_obj if provided, otherwise keep original
                        if result.wrapped_obj is not None:
                            wrapped_obj = result.wrapped_obj
                        # Use meta if provided, otherwise keep original
                        if result.meta is not None:
                            processed_meta = result.meta
                except Exception as e:
                    raise RegistryError(f"Processing failed for '{name}' in registry '{self.type_name}': {e}") from e

            entry = self._registry.get(name, RegistryEntry(name=name))

            if entry.is_loaded:
                raise DuplicateEntryError(f"Entry '{name}' already exists in registry '{self.type_name}'")

            entry.import_path = get_import_path(obj)
            wrapped_obj.nexus_meta = {'name': name, 'type': self.type_name, 'meta': processed_meta}  # type: ignore[attr-defined]
            entry.obj = wrapped_obj
            self._registry[name] = entry

            # Track registration if we're inside an import_module call
            registrations = _import_registrations.get()
            if registrations is not None:
                registrations.append({'type': self.type_name, 'name': name})

            logger.debug(f"Registered '{name}' in registry '{self.type_name}'")
            return wrapped_obj

        return wrapper

    def add(self, name: str, import_path: str, *, meta: dict | None = None, install_commands: list[str] | None = None):
        """
        Add an entry for lazy loading.

        The object will be imported and loaded when first accessed via get().

        Args:
            name: Unique name for this entry
            import_path: Import path in format 'module.path::object_name'
            meta: Optional metadata dict
            install_commands: Optional list of package names to install if import fails
        """
        if name in self._registry:
            existing = self._registry[name]
            if not existing.is_loaded and existing.import_path == import_path:
                logger.debug(f"Lazy entry '{name}' already exists in registry '{self.type_name}'")
                return
            logger.warning(f"Entry '{name}' already exists in registry '{self.type_name}'")
            return

        self._registry[name] = RegistryEntry(
            name=name, obj=None, import_path=import_path, meta=meta or {}, install_commands=install_commands or []
        )
        logger.debug(f"Added lazy entry '{name}' in registry '{self.type_name}'")

    def get(self, name: str) -> T:
        """
        Get a registered object by name.

        If the object hasn't been loaded yet (added via add()),
        it will be imported and loaded automatically.

        Args:
            name: Name of the entry

        Returns:
            The registered object

        Raises:
            EntryNotFoundError: If name doesn't exist in registry
            ImportError: If lazy loading fails
        """
        if name not in self._registry:
            raise EntryNotFoundError(
                f"Entry '{name}' not found in registry '{self.type_name}'. Available: {list(self._registry.keys())}"
            )

        entry = self._registry[name]

        if entry.is_loaded:
            return entry.obj

        if not entry.import_path:
            raise ImportError(f"Entry '{name}' in registry '{self.type_name}' has no import path for lazy loading.")

        Registry.import_module(entry.import_path, entry.install_commands)

        if not entry.is_loaded:
            raise ImportError(f"Failed to load '{name}' from '{entry.import_path}'")

        logger.debug(f"Lazy loaded '{name}' from '{entry.import_path}'")
        return entry.obj

    def get_entry(self, name: str) -> RegistryEntry:
        """Get the registry entry for a name."""
        if name not in self._registry:
            raise EntryNotFoundError(
                f"Entry '{name}' not found in registry '{self.type_name}'. Available: {list(self._registry.keys())}"
            )
        return self._registry[name]

    def __contains__(self, name: str) -> bool:
        return name in self._registry

    def list_entries(self) -> list[RegistryEntry]:
        return list(self._registry.values())

    def get_all(self) -> dict[str, T]:
        # Snapshot keys: get() may trigger lazy imports that register new entries.
        # If that happens, the caller forgot to declare the entry in module metadata.
        keys_before = set(self._registry)
        result = {name: self.get(name) for name in list(self._registry)}
        new_keys = set(self._registry) - keys_before
        if new_keys:
            raise RuntimeError(
                f"Registry '{self.type_name}' grew during get_all(): new entries {sorted(new_keys)} "
                f'were registered by eager @register() but missing from module metadata. '
                f'Add them to the appropriate pyproject.toml or PEP 723 [tool.nexus] section.'
            )
        return result

    @staticmethod
    def import_module(import_path: str, install_commands: list[str] | None = None) -> list[dict[str, str]]:
        """
        Load an object from an import path. If import fails, try pip install with hints and try again.

        Args:
            import_path: Path in format 'module.path::object_name'
            install_commands: Package names to install and retry if import fails

        Returns:
            List of registrations made during import, each with 'type' and 'name' keys

        Raises:
            ImportError: If import fails
        """
        if install_commands is None:
            install_commands = []

        # Check if we're already inside an import_module call (nested import)
        parent_registrations = _import_registrations.get()
        is_nested = parent_registrations is not None

        # Set up tracking context - reuse parent's list if nested, otherwise create new
        if parent_registrations is not None:
            registrations = parent_registrations
            token = None
        else:
            registrations = []
            token = _import_registrations.set(registrations)

        # Strip '::object_name' suffix if present — add() stores the full
        # 'module.path::object' form, but importlib only needs the module.
        module_path = import_path.split('::')[0] if '::' in import_path else import_path

        try:
            try:
                importlib.import_module(module_path)
            except ModuleNotFoundError as e:
                if install_commands:
                    # Explicit install commands provided — use dep_checker for uv/pip
                    from nexus.core.dep_checker import install_packages

                    logger.info('Installing explicit deps for %s: %s', import_path, install_commands)
                    ok, msg = install_packages(install_commands)
                    if not ok:
                        raise ImportError(f'Module {import_path!r} not found and install failed: {msg}') from e
                    importlib.import_module(module_path)
                else:
                    # Try auto-install from module metadata
                    from nexus.core.module_registry import module_registry

                    if module_registry.auto_install_enabled() and module_registry.try_auto_install(import_path):
                        importlib.import_module(module_path)
                    else:
                        hint = module_registry.format_install_hint(import_path)
                        if hint:
                            raise ImportError(f'Failed to import {import_path!r}: {e}\n{hint}') from e
                        raise ImportError(f'Module {import_path!r} not found and no install commands provided') from e

            # Only log at the outermost level
            if not is_nested:
                if registrations:
                    count = len(registrations)
                    suffix = 's' if count > 1 else ''
                    logger.info(f"Imported '{import_path}' with {count} registration{suffix}:")
                    # Group by type
                    by_type: dict[str, list[str]] = {}
                    for reg in registrations:
                        by_type.setdefault(reg['type'], []).append(reg['name'])
                    for reg_type, names in by_type.items():
                        logger.info(f'  - [{reg_type}] {", ".join(names)}')
                else:
                    logger.info(f"Imported '{import_path}' with no new registrations")

            return registrations
        finally:
            if token is not None:
                _import_registrations.reset(token)


def get_import_path(obj: Any) -> str:
    """Get the import path for a callable object.

    Returns a string like "module.path::object_name" that can be used
    to import the object later.
    """
    module = getattr(obj, '__module__', None)
    qualname = getattr(obj, '__qualname__', None)

    if module is None:
        raise ValueError(f'Object {obj} has no __module__ attribute')
    if qualname is None:
        raise ValueError(f'Object {obj} has no __qualname__ attribute')

    return f'{module}::{qualname}'


def import_from_path(import_path: str) -> Any:
    """Import an object from its import path.

    Args:
        import_path: Path in format 'module.path::object_name'

    Returns:
        The imported object
    """
    if '::' not in import_path:
        raise ValueError(f"Invalid import_path format: {import_path}. Expected 'module.path::object_name'")

    module_path, obj_name = import_path.split('::', 1)

    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        raise ImportError(f"Failed to import module '{module_path}': {e}") from e

    if not hasattr(module, obj_name):
        raise AttributeError(f"Module '{module_path}' has no attribute '{obj_name}'")

    return getattr(module, obj_name)


__all__ = [
    # Exceptions
    'RegistryError',
    'RegistryNotFoundError',
    'EntryNotFoundError',
    'DuplicateEntryError',
    # Core classes
    'Registry',
    'RegistryEntry',
    'ProcessorResult',
    # Utility functions
    'get_import_path',
    'import_from_path',
]
