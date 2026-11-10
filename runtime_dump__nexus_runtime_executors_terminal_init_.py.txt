"""Terminal executor package: single PTY + patched-bash capture (--wrap=execute_command/readline).

Importing the package registers all 8 executors (via executors module side
effects) and exposes TerminalSession for direct use/tests.
"""

# Re-export the request/response models from the package path. The legacy
# single-file terminal.py did `from ...models.terminal import *`, so upstream
# modules import these straight from `nexus.runtime.executors.terminal`
# (e.g. extensions/tasks/swe/swe_agent_rubrics/tools.py). Keep that surface.
from nexus.runtime.executors.models.terminal import *  # noqa: F401,F403
from nexus.runtime.executors.terminal.executors import (  # noqa: F401
    TERMINAL_SESSIONS,
    capture_pane,
    create_terminal_session,
    destroy_terminal_session,
    query_command_status,
    run_command_in_session,
    send_keys_to_session,
    start_command_in_session,
    terminate_session_processes,
)
from nexus.runtime.executors.terminal.session import TerminalSession  # noqa: F401
