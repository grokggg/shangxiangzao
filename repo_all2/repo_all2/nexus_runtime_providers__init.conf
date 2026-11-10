"""Runtime providers module.

Provider classes are registered lazily via PEP 723 metadata discovery
(see ``module_registry.discover_and_register()``).  The actual modules
are imported only when the corresponding name is first requested via
``runtime_provider_registry.get()``.
"""

from nexus.runtime.providers.base import InvalidPersistentRuntimeError as InvalidPersistentRuntimeError
from nexus.runtime.providers.base import RuntimeProvider as RuntimeProvider
from nexus.runtime.providers.registry import runtime_provider_registry as runtime_provider_registry
from nexus.runtime.providers.registry import validate_runtime_provider as validate_runtime_provider
