"""Desktop GUI tools for the local Docker GUI runtime."""

from typing import Any, Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field, model_validator

from nexus.core import ToolContext, ToolResponse, ToolType, require_context, tool_registry
from nexus.extensions.tools.desktop.osworld_actions import (
    COMPUTER_13_ACTION_SPACE,
    COMPUTER_13_ACTION_TYPES,
    COMPUTER_13_FIXED_MARKER_GUIDANCE,
    COMPUTER_13_PARAMETER_GUIDE,
    PYAUTOGUI_CODE_SHAPE_GUIDANCE,
    PYAUTOGUI_RESTRICTED_METHODS,
    PYAUTOGUI_SAFETY_POLICY,
)
from nexus.runtime.pyautogui_guard import PyAutoGUICodeValidationError, validate_pyautogui_code
from nexus.runtime.utils.gui import session_user_command


DESKTOP_RUNTIME_NAME = 'main'
DESKTOP_SESSION_USER = 'user'
DESKTOP_SESSION_HOME_DIR = '/home/user'
DESKTOP_LAST_TOOL_STATE = 'desktop_last_tool'

COMPUTER_13_TOOL_DESCRIPTION = f"""\
Execute one official OSWorld computer_13 desktop action through the Nexus tool interface. Use exactly one action per call.
{COMPUTER_13_ACTION_SPACE}
{COMPUTER_13_FIXED_MARKER_GUIDANCE}
"""

COMPUTER_13_ACTION_TYPE_DESCRIPTION = (
    f'Official OSWorld computer_13 action type. Choose one of {", ".join(COMPUTER_13_ACTION_TYPES)}.\n'
    + COMPUTER_13_FIXED_MARKER_GUIDANCE
    + '\n'
)

COMPUTER_13_PARAMETERS_DESCRIPTION = COMPUTER_13_PARAMETER_GUIDE + '\n' + COMPUTER_13_FIXED_MARKER_GUIDANCE + '\n'

PYAUTOGUI_TOOL_DESCRIPTION = f"""\
Execute official OSWorld pyautogui code in the desktop session, grounded to the current observation.
Use this when the official OSWorld pyautogui action space would return Python code rather than a computer_13 action.
{PYAUTOGUI_RESTRICTED_METHODS}
{PYAUTOGUI_SAFETY_POLICY}
Return one line or multiple lines of python code, and keep the code focused on the next desktop action.
{PYAUTOGUI_CODE_SHAPE_GUIDANCE}
"""

PYAUTOGUI_CODE_DESCRIPTION = f"""\
Python code for one official OSWorld pyautogui action, grounded to the current observation.
{PYAUTOGUI_RESTRICTED_METHODS}
{PYAUTOGUI_SAFETY_POLICY}
Return one line or multiple lines of python code. For multiple lines, include small intervals such as time.sleep(0.5).
Each call must be complete code; no variables or function can be shared from history.
You must specify coordinates yourself based on the observation.
Examples include pyautogui.click(...), pyautogui.moveTo(...), pyautogui.dragTo(...), pyautogui.press(...),
pyautogui.hotkey(...), or time.sleep(...).
"""


class ScreenshotInput(BaseModel):
    """Input payload for capturing a desktop screenshot."""

    pass


class ScreenSizeInput(BaseModel):
    """Input payload for querying the desktop screen size."""

    pass


class StartInput(BaseModel):
    """Input payload for explicitly starting the desktop GUI stack."""

    desktop_session: Literal['gnome-session', 'gnome-shell'] | None = Field(
        None,
        description='Optional desktop session. Leave unset for the default lightweight X11 stack.',
    )
    session_options: dict[str, str] = Field(
        default_factory=dict,
        description='Optional desktop-session-specific settings, such as {"gnome_session_name": "ubuntu"}.',
    )


class LaunchAppInput(BaseModel):
    """Input payload for launching a desktop application."""

    app: str = Field(..., description='Shell command to launch the desktop app')


class CloseAppInput(BaseModel):
    """Input payload for closing a desktop application."""

    app: str = Field(..., description='Name or command substring of the app to close')


class DesktopComputer13Input(BaseModel):
    """Input payload for OSWorld-style desktop computer_13 actions."""

    model_config = ConfigDict(extra='allow')

    action_type: Literal[
        'MOVE_TO',
        'CLICK',
        'MOUSE_DOWN',
        'MOUSE_UP',
        'RIGHT_CLICK',
        'DOUBLE_CLICK',
        'DRAG_TO',
        'SCROLL',
        'TYPING',
        'PRESS',
        'KEY_DOWN',
        'KEY_UP',
        'HOTKEY',
    ] = Field(..., description=COMPUTER_13_ACTION_TYPE_DESCRIPTION)
    parameters: dict[str, Any] = Field(default_factory=dict, description=COMPUTER_13_PARAMETERS_DESCRIPTION)

    @model_validator(mode='after')
    def merge_top_level_parameters(self) -> 'DesktopComputer13Input':
        extra = self.model_extra or {}
        if extra:
            self.parameters = {**extra, **self.parameters}
        return self


class DesktopPyAutoGUIInput(BaseModel):
    """Input payload for direct PyAutoGUI controller code."""

    code: str = Field(..., description=PYAUTOGUI_CODE_DESCRIPTION)
    # 桌面 GUI 动作常含长 sleep / 大段 typewrite，单次执行 30s 易触 TimeoutError 误杀整条 agent 轨迹，
    # 故默认放宽到 120s；模型仍可按需 per-call 覆盖。
    timeout: float = Field(120.0, gt=0, description='Execution timeout in seconds')


class DesktopBashInput(BaseModel):
    """Input payload for shell commands in the desktop runtime."""

    command: str = Field(..., description='Shell command to execute in the runtime')
    timeout: float = Field(180.0, gt=0, description='Execution timeout in seconds')
    cwd: str | None = Field(None, description='Optional working directory')


async def _runtime():
    """Resolve the shared desktop runtime from the active tool context."""
    ctx = require_context()
    return await ctx.get_runtime(DESKTOP_RUNTIME_NAME)


def _desktop_tool_context_update(tool_name: str) -> dict[str, str]:
    return {DESKTOP_LAST_TOOL_STATE: tool_name}


def _tool_config(context: ToolContext | None, tool_name: str) -> dict[str, Any]:
    if context is None:
        return {}
    config = context.tool_configs.get(tool_name, {})
    return dict(config) if isinstance(config, dict) else {}


class _SessionConfig(TypedDict):
    user: str
    home_dir: str


def _session_config(context: ToolContext | None, tool_name: str) -> _SessionConfig:
    config = _tool_config(context, tool_name)
    user = str(config.get('user') or config.get('run_as_user') or DESKTOP_SESSION_USER)
    home_dir = str(config.get('home_dir') or f'/home/{user}')
    return {'user': user, 'home_dir': home_dir}


@tool_registry.register(
    'desktop/screenshot',
    meta={
        'name': 'screenshot',
        'description': 'Capture the current Linux desktop screenshot and return it as a PNG image.',
        'tool_type': ToolType.LOCAL,
    },
)
async def screenshot_tool(params: ScreenshotInput, context: ToolContext) -> ToolResponse:
    """Capture the current desktop and return an inline image response."""
    runtime = await _runtime()
    response = await runtime.gui_screenshot()
    if not response.success or not response.image:
        return ToolResponse(
            output=[{'type': 'text', 'text': f'Screenshot failed: {response.error or "Unknown error"}'}],
            meta={'success': False, 'error': response.error},
            context_updates=_desktop_tool_context_update('desktop/screenshot'),
        )
    return ToolResponse(
        output=[
            {'type': 'image_url', 'image_url': {'url': f'data:image/{response.format};base64,{response.image}'}},
            {'type': 'text', 'text': f'Desktop screenshot captured at {response.width}x{response.height}.'},
        ],
        meta={'success': True, 'width': response.width, 'height': response.height},
        context_updates=_desktop_tool_context_update('desktop/screenshot'),
    )


@tool_registry.register(
    'desktop/screen_size',
    meta={
        'name': 'screen_size',
        'description': 'Return the current Linux desktop screen size.',
        'tool_type': ToolType.LOCAL,
    },
)
async def screen_size_tool(params: ScreenSizeInput, context: ToolContext) -> ToolResponse:
    """Return the current desktop screen dimensions."""
    runtime = await _runtime()
    response = await runtime.gui_get_screen_size()
    if not response.success:
        return ToolResponse(
            output=[{'type': 'text', 'text': f'Screen size failed: {response.error or "Unknown error"}'}],
            meta={'success': False, 'error': response.error},
        )
    return ToolResponse(
        output=[{'type': 'text', 'text': f'Screen size is {response.width}x{response.height}.'}],
        meta={'success': True, 'width': response.width, 'height': response.height},
    )


@tool_registry.register(
    'desktop/start',
    meta={
        'name': 'start',
        'description': 'Start the Linux desktop GUI stack, including Xvfb, VNC, noVNC, and an optional desktop session.',
        'tool_type': ToolType.LOCAL,
    },
)
async def start_tool(params: StartInput, context: ToolContext) -> ToolResponse:
    """Start the runtime desktop GUI stack before using other desktop tools."""
    runtime = await _runtime()
    session_options = dict(params.session_options)
    if params.desktop_session == 'gnome-session':
        session_options.setdefault('gnome_session_name', 'ubuntu')
    response = await runtime.gui_start(
        desktop_session=params.desktop_session,
        **session_options,
    )
    if not response.success:
        return ToolResponse(
            output=[{'type': 'text', 'text': f'Desktop start failed: {response.error or "Unknown error"}'}],
            meta={'success': False, 'error': response.error, 'stderr': response.stderr},
        )
    return ToolResponse(
        output=[
            {
                'type': 'text',
                'text': f'Desktop GUI started on {response.display}; noVNC port {response.novnc_port}.',
            }
        ],
        meta={
            'success': True,
            'display': response.display,
            'vnc_port': response.vnc_port,
            'novnc_port': response.novnc_port,
            'desktop_session': response.desktop_session,
        },
    )


@tool_registry.register(
    'desktop/launch_app',
    meta={
        'name': 'launch_app',
        'description': 'Launch a Linux desktop application using a shell command.',
        'tool_type': ToolType.LOCAL,
    },
)
async def launch_app_tool(params: LaunchAppInput, context: ToolContext) -> ToolResponse:
    """Launch a desktop application through the runtime GUI backend."""
    runtime = await _runtime()
    response = await runtime.gui_launch_app(params.app, **_session_config(context, 'desktop/launch_app'))
    if not response.success:
        return ToolResponse(
            output=[{'type': 'text', 'text': f'Launch app failed: {response.error or "Unknown error"}'}],
            meta={'success': False, 'error': response.error},
        )
    return ToolResponse(
        output=[{'type': 'text', 'text': f'Launched app: {params.app}'}],
        meta={'success': True},
    )


@tool_registry.register(
    'desktop/close_app',
    meta={
        'name': 'close_app',
        'description': 'Close a Linux desktop application by name or command substring.',
        'tool_type': ToolType.LOCAL,
    },
)
async def close_app_tool(params: CloseAppInput, context: ToolContext) -> ToolResponse:
    """Close a desktop application through the runtime GUI backend."""
    runtime = await _runtime()
    response = await runtime.gui_close_app(params.app)
    if not response.success:
        return ToolResponse(
            output=[{'type': 'text', 'text': f'Close app failed: {response.error or "Unknown error"}'}],
            meta={'success': False, 'error': response.error},
        )
    return ToolResponse(
        output=[{'type': 'text', 'text': f'Closed app: {params.app}'}],
        meta={'success': True},
    )


@tool_registry.register(
    'desktop/computer_13',
    meta={
        'name': 'computer_13',
        'description': COMPUTER_13_TOOL_DESCRIPTION,
        'tool_type': ToolType.LOCAL,
    },
)
async def computer_13_tool(params: DesktopComputer13Input, context: ToolContext) -> ToolResponse:
    """Execute one OSWorld-style desktop action through the runtime GUI backend."""
    runtime = await _runtime()
    response = await runtime.gui_computer_13(params.action_type, params.parameters)
    text = f'computer_13 {params.action_type}: {"success" if response.success else "failed"}'
    return ToolResponse(
        output=[{'type': 'text', 'text': text}],
        meta={
            'success': response.success,
            'error': response.error,
        },
        context_updates=_desktop_tool_context_update('desktop/computer_13'),
    )


@tool_registry.register(
    'desktop/pyautogui',
    meta={
        'name': 'pyautogui',
        'description': PYAUTOGUI_TOOL_DESCRIPTION,
        'tool_type': ToolType.LOCAL,
    },
)
async def pyautogui_tool(params: DesktopPyAutoGUIInput, context: ToolContext) -> ToolResponse:
    """Execute direct PyAutoGUI code through the runtime GUI backend."""
    try:
        validate_pyautogui_code(params.code)
    except PyAutoGUICodeValidationError as exc:
        error = str(exc)
        return ToolResponse(
            output=[{'type': 'text', 'text': error}],
            meta={'success': False, 'stdout': '', 'stderr': '', 'error': error},
            context_updates=_desktop_tool_context_update('desktop/pyautogui'),
        )
    runtime = await _runtime()
    call_kwargs: dict[str, object] = {'timeout': params.timeout, **_session_config(context, 'desktop/pyautogui')}
    response = await runtime.gui_pyautogui(params.code, **call_kwargs)
    stdout = response.stdout or ''
    stderr = response.stderr or ''
    error = response.error
    success = response.success
    text = stdout or stderr or error or ''
    return ToolResponse(
        output=[{'type': 'text', 'text': text}],
        meta={
            'success': success,
            'stdout': stdout,
            'stderr': stderr,
            'error': error,
        },
        context_updates=_desktop_tool_context_update('desktop/pyautogui'),
    )


@tool_registry.register(
    'desktop/bash',
    meta={
        'name': 'bash',
        'description': 'Execute a shell command in the desktop runtime.',
        'tool_type': ToolType.LOCAL,
    },
)
async def bash_tool(params: DesktopBashInput, context: ToolContext) -> ToolResponse:
    """Execute a shell command through the runtime command executor."""
    runtime = await _runtime()
    session_config = _session_config(context, 'desktop/bash')
    command = session_user_command(
        params.command,
        user=session_config['user'],
        home_dir=session_config['home_dir'],
        cwd=params.cwd,
    )
    response = await runtime.run_command(command=command, timeout=params.timeout)
    stdout = getattr(response, 'stdout', '') or ''
    stderr = getattr(response, 'stderr', '') or ''
    text = stdout if stdout else stderr
    return ToolResponse(
        output=[{'type': 'text', 'text': text}],
        meta={
            'success': getattr(response, 'return_code', None) == 0,
            'return_code': getattr(response, 'return_code', None),
            'stderr': stderr,
        },
    )
