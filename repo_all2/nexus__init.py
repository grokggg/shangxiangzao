from nexus.utils.otel.auto import initialize_if_enabled as _initialize_otel_if_enabled


_initialize_otel_if_enabled()
del _initialize_otel_if_enabled

# Discover and lazy-register all modules declared in MODULE_ROOTS.
# Must run after the imports above to avoid circular dependencies
# (registries in nexus.core must be initialized first).
from nexus.core.module_registry import module_registry as module_registry  # noqa: E402
from nexus.runner.batch import BatchItem as BatchItem  # noqa: E402
from nexus.runner.single import ModuleObjectConfig as ModuleObjectConfig  # noqa: E402
from nexus.runner.single import ProcessorConfig as ProcessorConfig  # noqa: E402
from nexus.runner.single import RunInput as RunInput  # noqa: E402


module_registry.discover_and_register()
