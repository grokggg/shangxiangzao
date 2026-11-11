"""Base classes for runtime providers.

This module contains abstract base classes for runtime providers that manage
the lifecycle of runtime environments.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from nexus.runtime.client.base import Runtime


class InvalidPersistentRuntimeError(RuntimeError):
    """Raised when a persistent runtime can no longer be reused.

    Runtime-pool integrations use this exception as an explicit signal to
    invalidate the corresponding pool entry.  Transient provider errors must
    use their original exception types so the entry remains available.
    """


class RuntimeProvider(ABC):
    """Base class for runtime providers.

    RuntimeProvider handles the lifecycle of runtime environments:
    - Acquiring new runtime instances (e.g., creating sandboxes)
    - Releasing runtime instances when no longer needed
    - Managing configuration for environment provisioning

    Supports two-phase parameter passing:
        # Phase 1: Pre-configure default params for a runtime
        provider.configure('main', image='python:3.11', cpu=2)

        # Phase 2: Acquire with additional/override params
        runtime = await provider.acquire('main', memory='4G')
        # Final params: {image: 'python:3.11', cpu: 2, memory: '4G'}

    Subclasses should implement the actual provisioning logic for specific
    infrastructure (e.g., Docker containers, Kubernetes pods, cloud VMs).
    """

    def __init__(self, **config: Any) -> None:
        """Initialize the provider with configuration.

        Args:
            **config: Provider-specific configuration options
        """
        self.config = config
        self._active_runtimes: dict[str, Runtime] = {}
        self._runtime_params: dict[str, dict[str, Any]] = {}
        self._suppress_release: bool = False

    def configure(self, name: str, **params: Any) -> None:
        """Pre-configure default parameters for a runtime.

        These params will be merged with params passed to acquire().
        acquire() params take precedence over pre-configured params.

        Args:
            name: Runtime name to configure
            **params: Default parameters for this runtime
        """
        if name not in self._runtime_params:
            self._runtime_params[name] = {}
        self._runtime_params[name].update(params)

    def get_params(self, name: str, **override: Any) -> dict[str, Any]:
        """Get merged parameters for a runtime.

        Merges pre-configured params with override params.
        Override params take precedence.

        Args:
            name: Runtime name
            **override: Parameters to override/extend

        Returns:
            Merged parameters dict
        """
        base = self._runtime_params.get(name, {}).copy()
        base.update(override)
        return base

    @abstractmethod
    async def acquire(self, name: str, **params: Any) -> Runtime:
        """Acquire a new runtime instance.

        This method should provision a new environment and return a Runtime
        instance that can interact with it.

        Implementation should call self.get_params(name, **params) to get
        the merged parameters.

        Args:
            name: Unique name for this runtime instance
            **params: Additional parameters (merged with pre-configured)

        Returns:
            A Runtime instance connected to the provisioned environment
        """
        ...

    async def release(self, name: str, *, force: bool = False) -> bool:
        """Release a runtime instance.

        Args:
            name: Name of the runtime instance to release
            force: Bypass ``_suppress_release`` for lifecycle-required isolation

        Returns:
            True when provider cleanup ran, False when release was suppressed
        """
        if self._suppress_release and not force:
            return False
        await self._release_impl(name)
        return True

    @abstractmethod
    async def _release_impl(self, name: str) -> None:
        """Actual release logic — subclasses implement this.

        Args:
            name: Name of the runtime instance to release
        """
        ...

    async def release_all(self) -> None:
        """Release all active runtime instances. Respects _suppress_release flag."""
        if self._suppress_release:
            return
        await self._release_all_impl()

    async def _release_all_impl(self) -> None:
        """Default release_all: iterate and release each. Subclass may override."""
        names = list(self._active_runtimes.keys())
        for name in names:
            await self._release_impl(name)

    def get_active_runtimes(self) -> dict[str, Runtime]:
        """Get all currently active runtime instances.

        Returns:
            Dict mapping runtime names to Runtime instances
        """
        return self._active_runtimes.copy()

    def get_runtime_id(self, name: str) -> str | None:
        """Return the provider-native runtime ID for the runtime registered under ``name``.

        The "runtime ID" is whatever stable identifier the underlying
        infrastructure assigns to a sandbox/container (e.g. the Gongfeng
        ``runtime_id`` like ``nexus-XXXX``, a Docker container ID, an AGS
        ``sandbox_id``). Subclasses should override this method when they
        track such an identifier; the base implementation returns ``None``
        for providers that have no meaningful ID to report (e.g. local).

        Args:
            name: Runtime name as passed to :meth:`acquire` / :meth:`configure`.

        Returns:
            The native runtime ID, or ``None`` when no ID is tracked or the
            given ``name`` has no active runtime.
        """
        return None


__all__ = [
    'InvalidPersistentRuntimeError',
    'RuntimeProvider',
]
