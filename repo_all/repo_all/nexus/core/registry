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
                        # Use wrapped_obj if provided, ot