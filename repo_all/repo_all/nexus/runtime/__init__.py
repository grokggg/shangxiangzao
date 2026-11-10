"""Runtime module for nexus framework.

This module provides runtime providers and clients for executing code
in isolated environments (containers, sandboxes, etc.).

Structure:
- client/: Host-side runtime client interfaces and implementations
- server/: Sandbox-side server, venv management, worker processes, LLM proxies
- protocol/: Shared wire format models + backward-compat shims
- providers/: Host-side factory layer (creates Runtime client instances)
- executors/: Executor implementations (command, terminal, file, jupyter, browser)

Concrete providers and executors are registered lazily via metadata discovery
(see ``module_registry.discover_and_register()``).
"""

# Core client types (used throughout the codebase)
from nexus.runtime.client.base import Runtime as Runtime
from nexus.runtime.providers.base import InvalidPersistentRuntimeError as InvalidPersistentRuntimeError
from nexus.runtime.providers.base import RuntimeProvider as RuntimeProvider
from nexus.runtime.providers.registry import runtime_provider_registry as runtime_provider_registry
