"""Local RuntimeProvider for in-process runtime execution.

This provider does not create containers or remote sandboxes. It returns a
LocalRuntime instance that executes all operations in the current machine's
local environment.
"""

# /// script
# dependencies = []
#
# [project]
# name = "local"
# authors = [
#     { name = "nickklau" },
# ]
#
# [tool.nexus]
# runtime_providers = { local = "nexus.runtime.providers.local" }
# ///

from __future__ import annotations

from typing import Any

from nexus.runtime.client.base import Runtime
from nexus.runtime.client.local import LocalRuntime
from nexus.runtime.providers.base import RuntimeProvider
from nexus.runtime.providers.registry import runtime_provider_registry


@runtime_provider_registry.register('local')
class LocalRuntimeProvider(RuntimeProvider):
    """Runtime provider that executes everything locally.

    Config options:
        resource_spec: Optional resource specification metadata.

    Acquire params:
        resource_spec: Optional resource specification metadata overriding provider default.
    """

    def __init__(self, resource_spec: dict[str, int] | None = None, **config: Any) -> None:
        super().__init__(**config)
        self.resource_spec = resource_spec

    async def acquire(self, name: str, **params: Any) -> Runtime:
        """Acquire a local runtime instance.

        Reuses existing runtime for the same name when available.
        """
        if name in self._active_runtimes:
            return self._active_runtimes[name]

        merged = self.get_params(name, **params)
        resource_spec = merged.get('resource_spec') or self.resource_spec

        runtime = LocalRuntime(resource_spec=resource_spec)
        self._active_runtimes[name] = runtime
        return runtime

    async def _release_impl(self, name: str) -> None:
        """Release a local runtime instance."""
        self._active_runtimes.pop(name, None)


__all__ = [
    'LocalRuntimeProvider',
]
