"""Runtime executors module.

This module contains executors that run in the runtime environment (sandbox/container).
Each executor handles a specific type of operation (command execution, file operations,
terminal sessions, etc.).

Model classes (request/response) are imported eagerly from the lightweight
``models`` sub-package — no Playwright/jupyter_client/tmux needed.

Executor functions are registered lazily via metadata discovery
(see ``module_registry.discover_and_register()``).  The actual
executor implementations are loaded on first ``executor_registry.get()`` call.
"""

from nexus.runtime.executors.models import *  # noqa: F401,F403
