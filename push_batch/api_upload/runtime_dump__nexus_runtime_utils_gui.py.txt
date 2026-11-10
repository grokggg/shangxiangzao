"""Shared shell helpers for Linux GUI session user commands."""

from __future__ import annotations

import shlex

from typing import Literal


MissingUserPolicy = Literal['error', 'create', 'fallback']


def session_user_command(
    command: str,
    *,
    user: str | None = 'user',
    home_dir: str | None = None,
    cwd: str | None = None,
    bash: str = 'bash',
    display: str = ':99',
    missing_user_policy: MissingUserPolicy = 'error',
) -> str:
    """Wrap a shell command so it runs with the selected desktop session user."""
    if not user:
        if cwd:
            return f'cd {shlex.quote(cwd)} && {command}'
        return command

    home_dir = home_dir or f'/home/{user}'
    cwd = cwd or home_dir
    user_q = shlex.quote(user)
    home_q = shlex.quote(home_dir)
    cwd_q = shlex.quote(cwd)
    bash_q = shlex.quote(bash)
    inner = shlex.quote(f'cd {cwd_q} && {command}')
    session_bus_probe = (
        'for pid in $(pgrep -u "$(id -u)" -f "[g]nome-session-binary|[g]nome-shell" 2>/dev/null); do '
        'tr "\\0" "\\n" <"/proc/${pid}/environ" 2>/dev/null | '
        'sed -n "s/^DBUS_SESSION_BUS_ADDRESS=//p" | head -n 1; '
        'done | head -n 1'
    )
    session_bus_probe_q = shlex.quote(session_bus_probe)
    desktop_env = (
        'XDG_CURRENT_DESKTOP="${XDG_CURRENT_DESKTOP:-GNOME}" '
        'DESKTOP_SESSION="${DESKTOP_SESSION:-${NEXUS_GUI_GNOME_SESSION_NAME:-ubuntu}}" '
        'GNOME_SHELL_SESSION_MODE="${GNOME_SHELL_SESSION_MODE:-${NEXUS_GUI_GNOME_SESSION_NAME:-ubuntu}}" '
        'GDK_BACKEND="${GDK_BACKEND:-x11}" '
        'GSK_RENDERER="${GSK_RENDERER:-cairo}" '
        'CLUTTER_BACKEND="${CLUTTER_BACKEND:-x11}" '
    )
    run_as_user = (
        f'uid="$(id -u {user_q})"; '
        f'mkdir -p "/run/user/${{uid}}" {home_q}; '
        f'chown {user_q}:{user_q} "/run/user/${{uid}}" {home_q} 2>/dev/null || true; '
        f'chmod 700 "/run/user/${{uid}}" 2>/dev/null || true; '
        f'session_bus="$(runuser -u {user_q} -- {bash_q} -lc {session_bus_probe_q} 2>/dev/null || true)"; '
        'dbus_env=""; '
        'if [ -n "$session_bus" ]; then dbus_env="DBUS_SESSION_BUS_ADDRESS=$session_bus"; fi; '
        f'runuser -u {user_q} -- env DISPLAY="${{DISPLAY:-{display}}}" XDG_RUNTIME_DIR="/run/user/${{uid}}" '
        f'{desktop_env}$dbus_env HOME={home_q} USER={user_q} LOGNAME={user_q} {bash_q} -lc {inner}'
    )

    if missing_user_policy == 'create':
        return (
            f'if ! id -u {user_q} >/dev/null 2>&1; then '
            f'if ! getent group {user_q} >/dev/null 2>&1; then groupadd --gid 1000 {user_q} 2>/dev/null || true; fi; '
            f'useradd --uid 1000 --create-home --shell /bin/bash {user_q} 2>/dev/null || '
            f'useradd --create-home --shell /bin/bash {user_q}; '
            f'fi; '
            f'{run_as_user}'
        )

    if missing_user_policy == 'fallback':
        return (
            f'if id {user_q} >/dev/null 2>&1; then '
            f'{run_as_user}; '
            f'else mkdir -p {home_q}; '
            f'HOME={home_q} USER={user_q} LOGNAME={user_q} DISPLAY="${{DISPLAY:-{display}}}" '
            f'{desktop_env}{bash_q} -lc {inner}; '
            f'fi'
        )

    return run_as_user


__all__ = ['MissingUserPolicy', 'session_user_command']
