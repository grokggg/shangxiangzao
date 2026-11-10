"""Locate the patched bash binary that ships the bash-capture mechanism.

The capture mechanism is compiled INTO the bash binary with link-time wrappers
(--wrap=execute_command and --wrap=readline). There is no cc fallback: both local
dev/test and the container use the nix-built patched bash.

Priority:
  1. NEXUS_BASH_BIN env var (if the file exists)
  2. Image-baked fixed path (/nix/runtime/nix-env/bin/bash, patched) if it exists
  3. Otherwise: raise BuiltinUnavailableError (fail-fast, no silent degradation)
"""

import os


_DEFAULT_FIXED_PATH = '/nix/runtime/nix-env/bin/bash'


class BuiltinUnavailableError(RuntimeError):
    """Raised when the patched bash binary cannot be located."""


def locate_patched_bash(*, fixed_path: str | None = None) -> str:
    """Return a usable patched-bash path, following the priority order.

    Raise BuiltinUnavailableError if none is found.
    """
    env_bash = os.environ.get('NEXUS_BASH_BIN')
    if env_bash and os.path.exists(env_bash):
        return env_bash

    fixed = fixed_path if fixed_path is not None else _DEFAULT_FIXED_PATH
    if os.path.exists(fixed):
        return fixed

    raise BuiltinUnavailableError(
        'patched bash not found: set NEXUS_BASH_BIN to a bash built with '
        'the bash-capture mechanism (--wrap=execute_command + --wrap=readline), or '
        'build the nexus runtime nix env. No cc fallback exists — --wrap is a '
        'link-time decision.'
    )
