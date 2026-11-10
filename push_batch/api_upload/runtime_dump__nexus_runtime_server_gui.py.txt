"""GUI helpers used by the runtime server GUI executors."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import importlib
import io
import os
import platform
import shlex
import shutil
import signal
import subprocess
import sys
import time
import types

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nexus.runtime.pyautogui_guard import PyAutoGUICodeValidationError, validate_pyautogui_code
from nexus.runtime.utils.gui import session_user_command
from nexus.utils.shell import resolve_first_existing_path


GUI_BASH = resolve_first_existing_path(['/nix/runtime/nix-env/bin/bash', '/bin/bash', 'bash'], fallback_to_last=True)
DISPLAY = os.getenv('DISPLAY', os.getenv('NEXUS_GUI_DISPLAY', ':99'))
SCREEN_WIDTH = int(os.getenv('NEXUS_GUI_SCREEN_WIDTH', '1440'))
SCREEN_HEIGHT = int(os.getenv('NEXUS_GUI_SCREEN_HEIGHT', '900'))
AUTOMATION_BACKEND_ENV = 'NEXUS_GUI_AUTOMATION_BACKEND'
AUTOMATION_BACKEND_AUTO = 'auto'
AUTOMATION_BACKEND_XDOTOOL = 'xdotool'
AUTOMATION_BACKEND_PYAUTOGUI = 'pyautogui'
DEFAULT_SESSION_USER = os.getenv('NEXUS_GUI_SESSION_USER') or 'user'
DEFAULT_SESSION_HOME_DIR = os.getenv('NEXUS_GUI_SESSION_HOME_DIR') or f'/home/{DEFAULT_SESSION_USER}'
_SCROT_SCREENSHOT_PATH = Path('/tmp/nexus-gui-screenshot.png')
_GUI_LAUNCH_LOG_DIR = Path('/tmp')
_GUI_LAUNCH_FAILURE_GRACE_SECONDS = float(os.getenv('NEXUS_GUI_LAUNCH_FAILURE_GRACE_SECONDS', '0.2'))
_GUI_STACK_SCRIPT_CANDIDATES = [
    Path('/nix/runtime/nexus/deployment/runtime/gui_stack.sh'),
    Path(__file__).resolve().parents[3] / 'deployment/runtime/gui_stack.sh',
]

KEY_MAP = {
    'Backspace': 'BackSpace',
    'Delete': 'Delete',
    'Down': 'Down',
    'Enter': 'Return',
    'Escape': 'Escape',
    'Home': 'Home',
    'Left': 'Left',
    'PageDown': 'Next',
    'PageUp': 'Prior',
    'Return': 'Return',
    'Right': 'Right',
    'Space': 'space',
    'Tab': 'Tab',
    'Up': 'Up',
}


@dataclass
class LaunchedApp:
    command: str
    process: subprocess.Popen[bytes]
    started_at: float


_APP_PROCESSES: dict[str, list[LaunchedApp]] = {}
_BUTTON_MAP = {'left': '1', 'middle': '2', 'right': '3'}


def _gui_env() -> dict[str, str]:
    """Build the environment for child GUI processes."""
    env = os.environ.copy()
    env['DISPLAY'] = DISPLAY
    return env


def _resolve_session_user(user: str | None = None, home_dir: str | None = None) -> tuple[str | None, str | None]:
    """Resolve the Linux user/home used for session-owned GUI processes."""
    resolved_user = user if user is not None else DEFAULT_SESSION_USER
    if not resolved_user:
        return None, None
    return resolved_user, home_dir or DEFAULT_SESSION_HOME_DIR or f'/home/{resolved_user}'


def _gui_stack_script() -> Path:
    """Resolve the script that starts the local Linux GUI stack."""
    for candidate in _GUI_STACK_SCRIPT_CANDIDATES:
        if candidate.is_file():
            return candidate
    return _GUI_STACK_SCRIPT_CANDIDATES[0]


def _normalize_key_name(key: str) -> str:
    """Map public key names onto the xdotool-specific variants."""
    return KEY_MAP.get(key, key)


def _requested_automation_backend() -> str:
    """Return the requested GUI automation backend name."""
    requested = os.getenv(AUTOMATION_BACKEND_ENV, AUTOMATION_BACKEND_AUTO).strip().lower()
    if requested in {AUTOMATION_BACKEND_AUTO, AUTOMATION_BACKEND_XDOTOOL, AUTOMATION_BACKEND_PYAUTOGUI}:
        return requested
    return AUTOMATION_BACKEND_AUTO


def _load_pyautogui() -> Any:
    """Import PyAutoGUI and configure it for unattended runtime use."""

    def _set_dynamic_attr(target: Any, name: str, value: Any) -> None:
        setattr(target, name, value)

    try:
        module = importlib.import_module('pyautogui')
    except SystemExit as exc:
        if 'MouseInfo' not in str(exc) and 'tkinter' not in str(exc):
            raise
        sys.modules.pop('pyautogui', None)
        mouseinfo_stub = types.ModuleType('mouseinfo')

        def _mouse_info_window():
            raise RuntimeError('MouseInfo is unavailable in the runtime GUI environment')

        _set_dynamic_attr(mouseinfo_stub, 'MouseInfoWindow', _mouse_info_window)
        sys.modules['mouseinfo'] = mouseinfo_stub
        module = importlib.import_module('pyautogui')
    _set_dynamic_attr(module, 'FAILSAFE', False)
    return module


def _automation_backend() -> XDoToolBackend | PyAutoGUIBackend:
    """Resolve the active automation backend, preferring PyAutoGUI in auto mode."""
    requested = _requested_automation_backend()
    if requested == AUTOMATION_BACKEND_XDOTOOL:
        return XDoToolBackend()
    if requested == AUTOMATION_BACKEND_PYAUTOGUI:
        return PyAutoGUIBackend(_load_pyautogui())
    try:
        return PyAutoGUIBackend(_load_pyautogui())
    except (Exception, SystemExit):
        pass
    return XDoToolBackend()


def _selected_automation_name() -> str:
    """Return the currently selected backend name without throwing in auto mode."""
    try:
        return _automation_backend().name
    except Exception:
        return AUTOMATION_BACKEND_XDOTOOL


def _run_gui_command(
    command: list[str], *, check: bool = True, timeout: float = 30.0
) -> subprocess.CompletedProcess[str]:
    """Run a GUI command inside the configured X11 display."""
    return subprocess.run(
        command,
        check=check,
        capture_output=True,
        text=True,
        env=_gui_env(),
        timeout=timeout,
    )


def _run_gui_shell(command: str, *, timeout: float = 30.0) -> subprocess.CompletedProcess[str]:
    """Run a shell command inside the configured GUI environment."""
    return subprocess.run(
        [GUI_BASH, '-lc', command],
        check=True,
        capture_output=True,
        text=True,
        env=_gui_env(),
        timeout=timeout,
    )


def _button_number(button: str) -> str:
    """Map a public mouse button name onto xdotool's button number."""
    try:
        return _BUTTON_MAP[button]
    except KeyError as exc:
        raise ValueError(f'Unsupported mouse button: {button!r}') from exc


def _require_coordinate_pair(x: int | None, y: int | None) -> tuple[int, int] | None:
    """Return coordinates when both are present; reject half-specified coordinates."""
    if x is None and y is None:
        return None
    if x is None or y is None:
        raise ValueError('Both x and y must be provided together')
    return x, y


def _start_stack_sync(
    *,
    desktop_session: str | None = None,
    session_user: str | None = None,
    session_home_dir: str | None = None,
    window_manager: str | None = None,
    screen_width: int | None = None,
    screen_height: int | None = None,
    **session_options: object,
) -> dict[str, object]:
    """Start Xvfb, optional desktop session, x11vnc, and noVNC on demand."""
    env = _gui_env()
    gnome_session_name = session_options.get('gnome_session_name')
    resolved_session_user, resolved_session_home = _resolve_session_user(session_user, session_home_dir)
    if desktop_session is not None:
        env['NEXUS_GUI_DESKTOP_SESSION'] = desktop_session
    if resolved_session_user:
        env['NEXUS_GUI_SESSION_USER'] = resolved_session_user
    if resolved_session_home:
        env['HOME'] = resolved_session_home
    if isinstance(gnome_session_name, str) and gnome_session_name:
        env['NEXUS_GUI_GNOME_SESSION_NAME'] = gnome_session_name
    if window_manager is not None:
        env['NEXUS_GUI_WINDOW_MANAGER'] = window_manager
    if screen_width is not None:
        env['NEXUS_GUI_SCREEN_WIDTH'] = str(screen_width)
    if screen_height is not None:
        env['NEXUS_GUI_SCREEN_HEIGHT'] = str(screen_height)

    script = _gui_stack_script()
    result = subprocess.run(
        [GUI_BASH, str(script), 'start'],
        check=False,
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )
    return {
        'success': result.returncode == 0,
        'display': env.get('NEXUS_GUI_DISPLAY', DISPLAY),
        'vnc_port': int(env.get('NEXUS_GUI_VNC_PORT', '5900')),
        'novnc_port': int(env.get('NEXUS_GUI_NOVNC_PORT', '6080')),
        'desktop_session': env.get('NEXUS_GUI_DESKTOP_SESSION') or None,
        'stdout': result.stdout,
        'stderr': result.stderr,
        'error': None if result.returncode == 0 else f'gui stack start failed with return code {result.returncode}',
    }


def _prune_dead_processes() -> None:
    """Remove cached application processes that have already exited."""
    for key, handles in list(_APP_PROCESSES.items()):
        alive = [handle for handle in handles if handle.process.poll() is None]
        if alive:
            _APP_PROCESSES[key] = alive
        else:
            _APP_PROCESSES.pop(key, None)


def _register_process(command: str, process: subprocess.Popen[bytes]) -> None:
    """Track a launched GUI application for later cleanup."""
    _prune_dead_processes()
    _APP_PROCESSES.setdefault(command, []).append(LaunchedApp(command=command, process=process, started_at=time.time()))


def _matching_handles(name: str) -> list[LaunchedApp]:
    """Find tracked processes whose command matches the requested app name."""
    _prune_dead_processes()
    needle = name.strip()
    matches: list[LaunchedApp] = []
    for command, handles in _APP_PROCESSES.items():
        if command == needle or needle in command:
            matches.extend(handles)
    return matches


def _terminate_handle(handle: LaunchedApp) -> None:
    """Terminate a tracked GUI process group."""
    if handle.process.poll() is not None:
        return
    try:
        os.killpg(handle.process.pid, signal.SIGTERM)
        handle.process.wait(timeout=5)
    except Exception:
        with contextlib.suppress(Exception):
            os.killpg(handle.process.pid, signal.SIGKILL)


class XDoToolBackend:
    """Concrete GUI automation backend implemented with xdotool and desktop CLI tools."""

    name = AUTOMATION_BACKEND_XDOTOOL

    def screen_size(self) -> tuple[int, int]:
        try:
            result = _run_gui_command(['xrandr', '--current'])
            for line in result.stdout.splitlines():
                if ' current ' in line:
                    fragment = line.split(' current ', 1)[1].split(',', 1)[0]
                    width_text, height_text = [part.strip() for part in fragment.split(' x ', 1)]
                    return int(width_text), int(height_text)
        except Exception:
            pass
        return SCREEN_WIDTH, SCREEN_HEIGHT

    def screenshot(self) -> tuple[bytes, int, int]:
        width, height = self.screen_size()
        if shutil.which('gnome-screenshot'):
            _run_gui_command(['gnome-screenshot', '-f', str(_SCROT_SCREENSHOT_PATH)])
        else:
            _run_gui_command(['scrot', '-z', '-q', '100', str(_SCROT_SCREENSHOT_PATH)])
        return _SCROT_SCREENSHOT_PATH.read_bytes(), width, height

    def move_to(self, x: int, y: int) -> None:
        _run_gui_command(['xdotool', 'mousemove', '--sync', str(x), str(y)])

    def click(self, x: int | None, y: int | None, *, button: str = 'left', num_clicks: int = 1) -> None:
        coords = _require_coordinate_pair(x, y)
        command = ['xdotool']
        if coords is not None:
            command.extend(['mousemove', '--sync', str(coords[0]), str(coords[1])])
        command.append('click')
        if num_clicks > 1:
            command.extend(['--repeat', str(num_clicks), '--delay', '120'])
        command.append(_button_number(button))
        _run_gui_command(command)

    def mouse_down(self, *, button: str = 'left') -> None:
        _run_gui_command(['xdotool', 'mousedown', _button_number(button)])

    def mouse_up(self, *, button: str = 'left') -> None:
        _run_gui_command(['xdotool', 'mouseup', _button_number(button)])

    def right_click(self, x: int | None = None, y: int | None = None) -> None:
        self.click(x, y, button='right')

    def double_click(self, x: int | None = None, y: int | None = None) -> None:
        coords = _require_coordinate_pair(x, y)
        command = ['xdotool']
        if coords is not None:
            command.extend(['mousemove', '--sync', str(coords[0]), str(coords[1])])
        command.extend(['click', '--repeat', '2', '--delay', '120', '1'])
        _run_gui_command(command)

    def long_click(self, x: int, y: int, *, duration_ms: int) -> None:
        duration = max(duration_ms, 0) / 1000.0
        _run_gui_shell(
            f'xdotool mousemove --sync {x} {y} mousedown 1 && sleep {duration:.3f} && xdotool mouseup 1',
            timeout=max(duration + 5.0, 30.0),
        )

    def drag(self, x1: int, y1: int, x2: int, y2: int, *, duration_ms: int) -> None:
        duration = max(duration_ms, 0) / 1000.0
        _run_gui_shell(
            f'xdotool mousemove --sync {x1} {y1} mousedown 1 && '
            f'sleep {duration:.3f} && '
            f'xdotool mousemove --sync {x2} {y2} mouseup 1',
            timeout=max(duration + 5.0, 30.0),
        )

    def drag_to(self, x: int, y: int, *, duration_ms: int) -> None:
        duration = max(duration_ms, 0) / 1000.0
        _run_gui_shell(
            f'xdotool mousedown 1 && sleep {duration:.3f} && xdotool mousemove --sync {x} {y} mouseup 1',
            timeout=max(duration + 5.0, 30.0),
        )

    def scroll(self, *, dx: int = 0, dy: int = 0) -> None:
        if dx:
            _run_gui_command(['xdotool', 'click', '6' if dx < 0 else '7'])
        if dy:
            _run_gui_command(['xdotool', 'click', '5' if dy < 0 else '4'])

    def type_text(self, text: str) -> None:
        _run_gui_command(
            ['xdotool', 'type', '--delay', '1', '--clearmodifiers', '--', text],
            timeout=max(len(text) / 20, 30.0),
        )

    def press_key(self, key: str) -> None:
        _run_gui_command(['xdotool', 'key', '--clearmodifiers', _normalize_key_name(key)])

    def key_down(self, key: str) -> None:
        _run_gui_command(['xdotool', 'keydown', _normalize_key_name(key)])

    def key_up(self, key: str) -> None:
        _run_gui_command(['xdotool', 'keyup', _normalize_key_name(key)])

    def hotkey(self, keys: list[str]) -> None:
        _run_gui_command(['xdotool', 'key', '--clearmodifiers', '+'.join(_normalize_key_name(key) for key in keys)])


class PyAutoGUIBackend:
    """Concrete GUI automation backend implemented with PyAutoGUI."""

    name = AUTOMATION_BACKEND_PYAUTOGUI

    def __init__(self, pyautogui: Any | None = None) -> None:
        self._pyautogui = pyautogui

    @property
    def pyautogui(self) -> Any:
        if self._pyautogui is None:
            self._pyautogui = _load_pyautogui()
        return self._pyautogui

    def screen_size(self) -> tuple[int, int]:
        size = self.pyautogui.size()
        return int(size[0]), int(size[1])

    def screenshot(self) -> tuple[bytes, int, int]:
        width, height = self.screen_size()
        image = self.pyautogui.screenshot()
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        return buffer.getvalue(), width, height

    def move_to(self, x: int, y: int) -> None:
        self.pyautogui.moveTo(x, y)

    def click(self, x: int | None, y: int | None, *, button: str = 'left', num_clicks: int = 1) -> None:
        coords = _require_coordinate_pair(x, y)
        kwargs: dict[str, object] = {'button': button, 'clicks': num_clicks}
        if coords is not None:
            kwargs.update({'x': coords[0], 'y': coords[1]})
        self.pyautogui.click(**kwargs)

    def mouse_down(self, *, button: str = 'left') -> None:
        self.pyautogui.mouseDown(button=button)

    def mouse_up(self, *, button: str = 'left') -> None:
        self.pyautogui.mouseUp(button=button)

    def right_click(self, x: int | None = None, y: int | None = None) -> None:
        self.click(x, y, button='right')

    def double_click(self, x: int | None = None, y: int | None = None) -> None:
        coords = _require_coordinate_pair(x, y)
        if coords is None:
            self.pyautogui.doubleClick()
        else:
            self.pyautogui.doubleClick(coords[0], coords[1])

    def long_click(self, x: int, y: int, *, duration_ms: int) -> None:
        duration = max(duration_ms, 0) / 1000.0
        self.pyautogui.mouseDown(x, y)
        time.sleep(duration)
        self.pyautogui.mouseUp(x, y)

    def drag(self, x1: int, y1: int, x2: int, y2: int, *, duration_ms: int) -> None:
        duration = max(duration_ms, 0) / 1000.0
        self.pyautogui.moveTo(x1, y1)
        self.pyautogui.dragTo(x2, y2, duration=duration, button='left')

    def drag_to(self, x: int, y: int, *, duration_ms: int) -> None:
        duration = max(duration_ms, 0) / 1000.0
        self.pyautogui.dragTo(x, y, duration=duration, button='left')

    def scroll(self, *, dx: int = 0, dy: int = 0) -> None:
        if dx:
            self.pyautogui.hscroll(dx)
        if dy:
            self.pyautogui.vscroll(dy)

    def type_text(self, text: str) -> None:
        self.pyautogui.typewrite(text, interval=0.001)

    def press_key(self, key: str) -> None:
        self.pyautogui.press(key.lower())

    def key_down(self, key: str) -> None:
        self.pyautogui.keyDown(key.lower())

    def key_up(self, key: str) -> None:
        self.pyautogui.keyUp(key.lower())

    def hotkey(self, keys: list[str]) -> None:
        self.pyautogui.hotkey(*(key.lower() for key in keys))

    def _runner_command(self, code: str) -> str:
        validate_pyautogui_code(code)
        encoded_code = base64.b64encode(code.encode('utf-8')).decode('ascii')
        runner = f"""\
import base64
import contextlib
import platform
import sys
import time
import types

def _load_pyautogui():
    try:
        import pyautogui as module
    except SystemExit as exc:
        if 'MouseInfo' not in str(exc) and 'tkinter' not in str(exc):
            raise
        sys.modules.pop('pyautogui', None)
        mouseinfo_stub = types.ModuleType('mouseinfo')
        def _mouse_info_window():
            raise RuntimeError('MouseInfo is unavailable in the runtime GUI environment')
        mouseinfo_stub.MouseInfoWindow = _mouse_info_window
        sys.modules['mouseinfo'] = mouseinfo_stub
        import pyautogui as module
    module.FAILSAFE = False
    _osworld_shift_chars = '~!@#$%^&*()_+' + chr(123) + chr(125) + '|:"<>?'
    _osworld_linux_shift_chars = '~!@#$%^&*()_+' + chr(123) + chr(125) + '|:">?'
    module.isShiftCharacter = lambda character: character.isupper() or character in (
        _osworld_linux_shift_chars if platform.system() == 'Linux' else _osworld_shift_chars
    )
    return module

with contextlib.redirect_stdout(sys.stderr):
    pyautogui = _load_pyautogui()

code = base64.b64decode({encoded_code!r}).decode('utf-8')
globals_dict = {{
    'pyautogui': pyautogui,
    'time': time,
    'platform': platform,
}}
exec(compile(code, '<gui_pyautogui>', 'exec'), globals_dict, globals_dict)
"""
        runner_arg = shlex.quote(runner)
        return (
            'python_bin=""; '
            'for candidate in '
            '/nix/runtime/nexus/.venv-3.13/bin/python '
            '/nix/runtime/nexus/.venv-3.12/bin/python '
            '/nix/runtime/nexus/.venv-3.11/bin/python '
            '/nix/runtime/nexus/.venv-3.10/bin/python '
            '/nix/runtime/nix-env/bin/python3 '
            '/usr/bin/python3; do '
            'if [ -x "$candidate" ]; then python_bin="$candidate"; break; fi; '
            'done; '
            'if [ -z "$python_bin" ]; then echo "No Python runtime found" >&2; exit 127; fi; '
            '"$python_bin" -c '
            f'{runner_arg}'
        )

    def _shell_command(self, code: str, *, user: str | None, home_dir: str | None) -> str:
        python_command = self._runner_command(code)
        if user and home_dir:
            return session_user_command(
                python_command,
                user=user,
                home_dir=home_dir,
                bash=GUI_BASH,
                display=DISPLAY,
                missing_user_policy='create',
            )
        return python_command

    def _run_code_shell(
        self, code: str, *, user: str | None = None, home_dir: str | None = None, timeout: float
    ) -> dict[str, object]:
        command = self._shell_command(code, user=user, home_dir=home_dir)
        result = subprocess.run(
            [GUI_BASH, '-lc', command],
            check=False,
            capture_output=True,
            text=True,
            env=_gui_env(),
            timeout=timeout,
        )
        return {
            'success': result.returncode == 0,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'error': None if result.returncode == 0 else result.stderr or f'Exit code {result.returncode}',
        }

    def run_code(
        self,
        code: str,
        *,
        user: str | None = None,
        home_dir: str | None = None,
        timeout: float = 30.0,
    ) -> dict[str, object]:
        try:
            validate_pyautogui_code(code)
        except PyAutoGUICodeValidationError as exc:
            error = str(exc)
            return {'success': False, 'stdout': '', 'stderr': '', 'error': error}
        resolved_user, resolved_home = _resolve_session_user(user, home_dir)
        return self._run_code_shell(code, user=resolved_user, home_dir=resolved_home, timeout=timeout)


def _screen_size_sync() -> tuple[int, int]:
    """Read the current X11 display size."""
    return _backend_call_sync('screen_size')


def _backend_call_sync(method: str, *args: object, **kwargs: object) -> Any:
    """Call one method on the selected GUI automation backend."""
    return getattr(_automation_backend(), method)(*args, **kwargs)


async def _backend_call(method: str, *args: object, **kwargs: object) -> Any:
    """Call one backend method without blocking the event loop."""
    return await asyncio.to_thread(_backend_call_sync, method, *args, **kwargs)


def _active_window_name() -> str | None:
    """Return the focused window title when one is available."""
    try:
        result = _run_gui_command(['xdotool', 'getwindowfocus', 'getwindowname'])
        title = result.stdout.strip()
        return title or None
    except Exception:
        return None


def _active_window_pid() -> str | None:
    """Return the PID of the focused window when one is available."""
    try:
        result = _run_gui_command(['xdotool', 'getwindowfocus', 'getwindowpid'])
        pid = result.stdout.strip()
        return pid or None
    except Exception:
        return None


def _screenshot_sync() -> tuple[bytes, int, int]:
    """Capture a PNG screenshot from the current desktop."""
    return _backend_call_sync('screenshot')


def _move_to_sync(x: int, y: int) -> None:
    """Move the cursor to the requested position."""
    _backend_call_sync('move_to', x, y)


def _click_sync(x: int | None, y: int | None, *, button: str = 'left', num_clicks: int = 1) -> None:
    """Perform a single mouse click."""
    _backend_call_sync('click', x, y, button=button, num_clicks=num_clicks)


def _run_pyautogui_sync(
    code: str, *, user: str | None = None, home_dir: str | None = None, timeout: float = 30.0
) -> dict[str, object]:
    """Execute arbitrary PyAutoGUI controller code in the runtime process."""
    return PyAutoGUIBackend().run_code(code, user=user, home_dir=home_dir, timeout=timeout)


def _launch_log_path(command: str) -> Path:
    """Return a short-lived log path for launch failures."""
    safe_name = ''.join(ch if ch.isalnum() else '-' for ch in command.strip())[:48].strip('-') or 'app'
    return _GUI_LAUNCH_LOG_DIR / f'nexus-gui-launch-{os.getpid()}-{int(time.time() * 1000)}-{safe_name}.log'


def _read_launch_log(path: Path) -> str:
    with contextlib.suppress(Exception):
        return path.read_text(encoding='utf-8', errors='replace').strip()
    return ''


def _launch_app_sync(command: str, *, user: str | None = None, home_dir: str | None = None) -> dict[str, object]:
    """Launch a desktop application and track its process handle."""
    resolved_user, resolved_home = _resolve_session_user(user, home_dir)
    launch_command = session_user_command(
        command,
        user=resolved_user,
        home_dir=resolved_home,
        bash=GUI_BASH,
        display=DISPLAY,
        missing_user_policy='create',
    )
    _GUI_LAUNCH_LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = _launch_log_path(command)
    log_file = log_path.open('ab')
    try:
        process = subprocess.Popen(
            [GUI_BASH, '-lc', launch_command],
            stdout=log_file,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            env=_gui_env(),
            preexec_fn=os.setsid,
        )
    finally:
        log_file.close()

    if _GUI_LAUNCH_FAILURE_GRACE_SECONDS > 0:
        time.sleep(_GUI_LAUNCH_FAILURE_GRACE_SECONDS)
    return_code = process.poll()
    if return_code is not None:
        output = _read_launch_log(log_path)
        if return_code == 0:
            return {
                'success': True,
                'error': None,
                'return_code': return_code,
                'log_path': str(log_path),
                'output': output,
            }
        error = output or f'Launch command exited immediately with return code {return_code}'
        return {'success': False, 'error': error, 'return_code': return_code, 'log_path': str(log_path)}

    _register_process(command, process)
    return {'success': True, 'error': None, 'log_path': str(log_path)}


def _close_app_sync(app_name: str) -> list[str]:
    """Close tracked applications and fall back to xdotool window search when needed."""
    closed: list[str] = []
    for handle in _matching_handles(app_name):
        _terminate_handle(handle)
        closed.append(handle.command)

    if closed:
        _prune_dead_processes()
        return closed

    safe_name = shlex.quote(app_name)
    with contextlib.suppress(Exception):
        _run_gui_shell(f'xdotool search --name {safe_name} windowclose %@')
        closed.append(app_name)

    return closed


def _reset_apps_sync() -> list[str]:
    """Terminate every tracked GUI process."""
    _prune_dead_processes()
    stopped: list[str] = []
    for handles in list(_APP_PROCESSES.values()):
        for handle in handles:
            if handle.process.poll() is None:
                _terminate_handle(handle)
                stopped.append(handle.command)
    _APP_PROCESSES.clear()
    return stopped


async def move_to(x: int, y: int) -> dict[str, object]:
    """Move the cursor to the requested position."""
    await _backend_call('move_to', x, y)
    return {'success': True}


async def click(
    x: int | None = None, y: int | None = None, *, button: str = 'left', num_clicks: int = 1
) -> dict[str, object]:
    """Perform a single click on the desktop."""
    await _backend_call('click', x, y, button=button, num_clicks=num_clicks)
    return {'success': True}


async def mouse_down(*, button: str = 'left') -> dict[str, object]:
    """Press and hold a mouse button."""
    await _backend_call('mouse_down', button=button)
    return {'success': True}


async def mouse_up(*, button: str = 'left') -> dict[str, object]:
    """Release a mouse button."""
    await _backend_call('mouse_up', button=button)
    return {'success': True}


async def right_click(x: int | None = None, y: int | None = None) -> dict[str, object]:
    """Perform a right click on the desktop."""
    await _backend_call('right_click', x, y)
    return {'success': True}


async def double_click(x: int | None = None, y: int | None = None) -> dict[str, object]:
    """Perform a double click on the desktop."""
    await _backend_call('double_click', x, y)
    return {'success': True}


async def long_click(x: int, y: int, *, duration_ms: int) -> dict[str, object]:
    """Perform a press-and-hold click on the desktop."""
    await _backend_call('long_click', x, y, duration_ms=duration_ms)
    return {'success': True}


async def drag(x1: int, y1: int, x2: int, y2: int, *, duration_ms: int) -> dict[str, object]:
    """Perform a drag gesture on the desktop."""
    await _backend_call('drag', x1, y1, x2, y2, duration_ms=duration_ms)
    return {'success': True}


async def drag_to(x: int, y: int, *, duration_ms: int) -> dict[str, object]:
    """Perform an OSWorld-style drag from the current cursor position."""
    await _backend_call('drag_to', x, y, duration_ms=duration_ms)
    return {'success': True}


async def scroll(*, dx: int = 0, dy: int = 0) -> dict[str, object]:
    """Scroll horizontally and/or vertically."""
    await _backend_call('scroll', dx=dx, dy=dy)
    return {'success': True}


async def type_text(text: str) -> dict[str, object]:
    """Type text into the focused application."""
    await _backend_call('type_text', text)
    return {'success': True}


async def press_key(key: str) -> dict[str, object]:
    """Press a special key in the focused application."""
    await _backend_call('press_key', key)
    return {'success': True}


async def key_down(key: str) -> dict[str, object]:
    """Press and hold a keyboard key."""
    await _backend_call('key_down', key)
    return {'success': True}


async def key_up(key: str) -> dict[str, object]:
    """Release a keyboard key."""
    await _backend_call('key_up', key)
    return {'success': True}


async def hotkey(keys: list[str]) -> dict[str, object]:
    """Press a keyboard shortcut."""
    await _backend_call('hotkey', keys)
    return {'success': True}


async def run_pyautogui(
    code: str,
    *,
    timeout: float = 30.0,
    user: str | None = None,
    home_dir: str | None = None,
) -> dict[str, object]:
    """Execute arbitrary PyAutoGUI controller code."""
    return await asyncio.wait_for(
        asyncio.to_thread(_run_pyautogui_sync, code, user=user, home_dir=home_dir, timeout=timeout),
        timeout=timeout,
    )


async def start_stack(
    *,
    desktop_session: str | None = None,
    session_user: str | None = None,
    session_home_dir: str | None = None,
    window_manager: str | None = None,
    screen_width: int | None = None,
    screen_height: int | None = None,
    **session_options: object,
) -> dict[str, object]:
    """Start the local GUI stack explicitly for runtimes that need desktop automation."""
    return await asyncio.to_thread(
        _start_stack_sync,
        desktop_session=desktop_session,
        session_user=session_user,
        session_home_dir=session_home_dir,
        window_manager=window_manager,
        screen_width=screen_width,
        screen_height=screen_height,
        **session_options,
    )


async def screenshot() -> tuple[bytes, int, int]:
    """Capture a desktop screenshot."""
    return await asyncio.to_thread(_screenshot_sync)


async def screen_size() -> tuple[int, int]:
    """Return the current desktop size."""
    return await asyncio.to_thread(_screen_size_sync)


async def launch_app(command: str, *, user: str | None = None, home_dir: str | None = None) -> dict[str, object]:
    """Launch a GUI application command."""
    return await asyncio.to_thread(_launch_app_sync, command, user=user, home_dir=home_dir)


async def close_app(app_name: str) -> dict[str, object]:
    """Close GUI applications that match the requested name."""
    closed = await asyncio.to_thread(_close_app_sync, app_name)
    return {'success': bool(closed), 'closed': closed}


async def reset_apps() -> dict[str, object]:
    """Terminate all tracked GUI applications."""
    stopped_packages = await asyncio.to_thread(_reset_apps_sync)
    return {'success': True, 'stopped_packages': stopped_packages}


async def get_device_info() -> dict[str, object]:
    """Return the current GUI runtime device metadata."""
    width, height = await screen_size()
    return {
        'success': True,
        'device_name': 'nexus-linux-gui',
        'platform_version': platform.platform(),
        'automation_name': _selected_automation_name(),
        'window_width': width,
        'window_height': height,
        'wm_size': f'{width}x{height}',
        'wm_density': '1',
        'current_package': _active_window_name(),
        'current_activity': _active_window_pid(),
    }


__all__ = [
    'AUTOMATION_BACKEND_ENV',
    'PyAutoGUIBackend',
    'XDoToolBackend',
    'click',
    'close_app',
    'double_click',
    'drag',
    'drag_to',
    'get_device_info',
    'hotkey',
    'key_down',
    'key_up',
    'launch_app',
    'long_click',
    'mouse_down',
    'mouse_up',
    'move_to',
    'press_key',
    'reset_apps',
    'right_click',
    'run_pyautogui',
    'screen_size',
    'screenshot',
    'scroll',
    'start_stack',
    'type_text',
]
