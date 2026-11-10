"""GUI executor request/response models.

This module defines Pydantic models for GUI operations across different platforms
(Android, iOS, Windows, macOS). These models are used by ExecutorMixin to provide
typed convenience methods for GUI automation.

Note: Models are prefixed with 'Gui' to distinguish from browser executor models
which have similar names (e.g., browser.ClickRequest vs gui.GuiClickRequest).
"""

from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field

from nexus.runtime.executors.models.base import ExecutorRequest


# ==================== Start GUI Stack ====================


class GuiStartResponse(BaseModel):
    success: bool
    display: str | None = None
    vnc_port: int | None = None
    novnc_port: int | None = None
    desktop_session: str | None = None
    stdout: str | None = None
    stderr: str | None = None
    error: str | None = None


class GuiStartRequest(ExecutorRequest[GuiStartResponse]):
    executor_name: ClassVar[str] = 'gui_start'
    response_type: ClassVar[type[BaseModel]] = GuiStartResponse

    desktop_session: str | None = Field(None, description='Optional desktop session, such as gnome-session')
    session_user: str | None = Field('user', description='Linux user used to start the desktop session')
    session_home_dir: str | None = Field('/home/user', description='HOME directory for the desktop session user')
    window_manager: str | None = Field(None, description='Optional X11 window manager command')
    screen_width: int | None = Field(None, description='Virtual desktop width')
    screen_height: int | None = Field(None, description='Virtual desktop height')
    session_options: dict[str, object] = Field(
        default_factory=dict,
        description='Optional desktop-session-specific settings, such as {"gnome_session_name": "ubuntu"}.',
    )


# ==================== Click ====================


class GuiClickResponse(BaseModel):
    success: bool
    error: str | None = None


class GuiClickRequest(ExecutorRequest[GuiClickResponse]):
    executor_name: ClassVar[str] = 'gui_click'
    response_type: ClassVar[type[BaseModel]] = GuiClickResponse

    x: int | None = Field(None, description='X coordinate; omit to click at the current cursor position')
    y: int | None = Field(None, description='Y coordinate; omit to click at the current cursor position')
    button: Literal['left', 'right', 'middle'] = Field('left', description='Mouse button')
    num_clicks: int = Field(1, ge=1, le=3, description='Number of clicks')


# ==================== Move To ====================


class GuiMoveToResponse(BaseModel):
    success: bool
    error: str | None = None


class GuiMoveToRequest(ExecutorRequest[GuiMoveToResponse]):
    executor_name: ClassVar[str] = 'gui_move_to'
    response_type: ClassVar[type[BaseModel]] = GuiMoveToResponse

    x: int = Field(..., description='X coordinate')
    y: int = Field(..., description='Y coordinate')


# ==================== Mouse Down/Up ====================


class GuiMouseDownResponse(BaseModel):
    success: bool
    error: str | None = None


class GuiMouseDownRequest(ExecutorRequest[GuiMouseDownResponse]):
    executor_name: ClassVar[str] = 'gui_mouse_down'
    response_type: ClassVar[type[BaseModel]] = GuiMouseDownResponse

    button: Literal['left', 'right', 'middle'] = Field('left', description='Mouse button')


class GuiMouseUpResponse(BaseModel):
    success: bool
    error: str | None = None


class GuiMouseUpRequest(ExecutorRequest[GuiMouseUpResponse]):
    executor_name: ClassVar[str] = 'gui_mouse_up'
    response_type: ClassVar[type[BaseModel]] = GuiMouseUpResponse

    button: Literal['left', 'right', 'middle'] = Field('left', description='Mouse button')


# ==================== Right Click ====================


class GuiRightClickResponse(BaseModel):
    success: bool
    error: str | None = None


class GuiRightClickRequest(ExecutorRequest[GuiRightClickResponse]):
    executor_name: ClassVar[str] = 'gui_right_click'
    response_type: ClassVar[type[BaseModel]] = GuiRightClickResponse

    x: int | None = Field(None, description='X coordinate; omit to right-click at the current cursor position')
    y: int | None = Field(None, description='Y coordinate; omit to right-click at the current cursor position')


# ==================== Double Click ====================


class GuiDoubleClickResponse(BaseModel):
    success: bool
    error: str | None = None


class GuiDoubleClickRequest(ExecutorRequest[GuiDoubleClickResponse]):
    executor_name: ClassVar[str] = 'gui_double_click'
    response_type: ClassVar[type[BaseModel]] = GuiDoubleClickResponse

    x: int | None = Field(None, description='X coordinate; omit to double-click at the current cursor position')
    y: int | None = Field(None, description='Y coordinate; omit to double-click at the current cursor position')


# ==================== Long Click ====================


class GuiLongClickResponse(BaseModel):
    success: bool
    error: str | None = None


class GuiLongClickRequest(ExecutorRequest[GuiLongClickResponse]):
    executor_name: ClassVar[str] = 'gui_long_click'
    response_type: ClassVar[type[BaseModel]] = GuiLongClickResponse

    x: int = Field(..., description='X coordinate')
    y: int = Field(..., description='Y coordinate')
    duration: int = Field(1000, description='Hold duration in milliseconds')


# ==================== Drag ====================


class GuiDragResponse(BaseModel):
    success: bool
    error: str | None = None


class GuiDragRequest(ExecutorRequest[GuiDragResponse]):
    executor_name: ClassVar[str] = 'gui_drag'
    response_type: ClassVar[type[BaseModel]] = GuiDragResponse

    x1: int = Field(..., description='Start X coordinate')
    y1: int = Field(..., description='Start Y coordinate')
    x2: int = Field(..., description='End X coordinate')
    y2: int = Field(..., description='End Y coordinate')
    duration: int = Field(300, description='Drag duration in milliseconds')


class GuiDragToRequest(ExecutorRequest[GuiDragResponse]):
    executor_name: ClassVar[str] = 'gui_drag_to'
    response_type: ClassVar[type[BaseModel]] = GuiDragResponse

    x: int = Field(..., description='End X coordinate')
    y: int = Field(..., description='End Y coordinate')
    duration: int = Field(300, description='Drag duration in milliseconds')


# ==================== Scroll ====================


class GuiScrollResponse(BaseModel):
    success: bool
    error: str | None = None


class GuiScrollRequest(ExecutorRequest[GuiScrollResponse]):
    executor_name: ClassVar[str] = 'gui_scroll'
    response_type: ClassVar[type[BaseModel]] = GuiScrollResponse

    dx: int = Field(0, description='Horizontal scroll amount')
    dy: int = Field(0, description='Vertical scroll amount')


# ==================== Type ====================


class GuiTypeResponse(BaseModel):
    success: bool
    error: str | None = None


class GuiTypeRequest(ExecutorRequest[GuiTypeResponse]):
    executor_name: ClassVar[str] = 'gui_type'
    response_type: ClassVar[type[BaseModel]] = GuiTypeResponse

    text: str = Field(..., description='Text to type')


# ==================== Press Key ====================


class GuiPressKeyResponse(BaseModel):
    success: bool
    error: str | None = None


class GuiPressKeyRequest(ExecutorRequest[GuiPressKeyResponse]):
    executor_name: ClassVar[str] = 'gui_press_key'
    response_type: ClassVar[type[BaseModel]] = GuiPressKeyResponse

    key: str = Field(..., description='Key name: Back/Home/Menu/Enter/Delete/Tab/Escape...')


class GuiKeyDownResponse(BaseModel):
    success: bool
    error: str | None = None


class GuiKeyDownRequest(ExecutorRequest[GuiKeyDownResponse]):
    executor_name: ClassVar[str] = 'gui_key_down'
    response_type: ClassVar[type[BaseModel]] = GuiKeyDownResponse

    key: str = Field(..., description='Key name')


class GuiKeyUpResponse(BaseModel):
    success: bool
    error: str | None = None


class GuiKeyUpRequest(ExecutorRequest[GuiKeyUpResponse]):
    executor_name: ClassVar[str] = 'gui_key_up'
    response_type: ClassVar[type[BaseModel]] = GuiKeyUpResponse

    key: str = Field(..., description='Key name')


class GuiHotkeyResponse(BaseModel):
    success: bool
    error: str | None = None


class GuiHotkeyRequest(ExecutorRequest[GuiHotkeyResponse]):
    executor_name: ClassVar[str] = 'gui_hotkey'
    response_type: ClassVar[type[BaseModel]] = GuiHotkeyResponse

    keys: list[str] = Field(..., min_length=1, description='Keys to press as a combination')


class GuiPyAutoGUIResponse(BaseModel):
    success: bool
    stdout: str | None = None
    stderr: str | None = None
    error: str | None = None


class GuiPyAutoGUIRequest(ExecutorRequest[GuiPyAutoGUIResponse]):
    executor_name: ClassVar[str] = 'gui_pyautogui'
    response_type: ClassVar[type[BaseModel]] = GuiPyAutoGUIResponse

    code: str = Field(
        ...,
        description='Python code limited to direct pyautogui.* calls, time.sleep(...), and read-only platform helpers',
    )
    timeout: float = Field(30.0, gt=0, description='Execution timeout in seconds')
    user: str | None = Field('user', description='Optional Linux user used to execute the PyAutoGUI code')
    home_dir: str | None = Field('/home/user', description='Optional HOME directory when executing as a Linux user')


class GuiComputer13Response(BaseModel):
    success: bool
    done: bool = False
    fail: bool = False
    error: str | None = None


class GuiComputer13Request(ExecutorRequest[GuiComputer13Response]):
    executor_name: ClassVar[str] = 'gui_computer_13'
    response_type: ClassVar[type[BaseModel]] = GuiComputer13Response

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
        'WAIT',
        'FAIL',
        'DONE',
    ] = Field(..., description='OSWorld computer_13 action type')
    parameters: dict[str, Any] = Field(default_factory=dict, description='OSWorld computer_13 action parameters')


# ==================== Screenshot ====================


class GuiScreenshotResponse(BaseModel):
    success: bool
    image: str | None = Field(None, description='Base64 encoded image bytes')
    width: int | None = None
    height: int | None = None
    format: str = 'png'
    error: str | None = None


class GuiScreenshotRequest(ExecutorRequest[GuiScreenshotResponse]):
    executor_name: ClassVar[str] = 'gui_screenshot'
    response_type: ClassVar[type[BaseModel]] = GuiScreenshotResponse


# ==================== Screen Size ====================


class GuiScreenSizeResponse(BaseModel):
    success: bool
    width: int | None = None
    height: int | None = None
    error: str | None = None


class GuiScreenSizeRequest(ExecutorRequest[GuiScreenSizeResponse]):
    executor_name: ClassVar[str] = 'gui_get_screen_size'
    response_type: ClassVar[type[BaseModel]] = GuiScreenSizeResponse


# ==================== Launch App ====================


class GuiLaunchAppResponse(BaseModel):
    success: bool
    error: str | None = None


class GuiLaunchAppRequest(ExecutorRequest[GuiLaunchAppResponse]):
    executor_name: ClassVar[str] = 'gui_launch_app'
    response_type: ClassVar[type[BaseModel]] = GuiLaunchAppResponse

    app: str = Field(..., description='App name or package name')
    user: str | None = Field('user', description='Optional Linux user used to launch the app')
    home_dir: str | None = Field('/home/user', description='Optional HOME directory when launching the app')


# ==================== Close App ====================


class GuiCloseAppResponse(BaseModel):
    success: bool
    error: str | None = None


class GuiCloseAppRequest(ExecutorRequest[GuiCloseAppResponse]):
    executor_name: ClassVar[str] = 'gui_close_app'
    response_type: ClassVar[type[BaseModel]] = GuiCloseAppResponse

    app: str = Field(..., description='App name or package name')


# ==================== Reset Apps ====================


class GuiResetAppsResponse(BaseModel):
    success: bool
    stopped_packages: list[str] = Field(default_factory=list)
    error: str | None = None


class GuiResetAppsRequest(ExecutorRequest[GuiResetAppsResponse]):
    executor_name: ClassVar[str] = 'gui_reset_apps'
    response_type: ClassVar[type[BaseModel]] = GuiResetAppsResponse


# ==================== Device Info ====================


class GuiGetDeviceInfoResponse(BaseModel):
    success: bool
    device_name: str | None = None
    platform_version: str | None = None
    automation_name: str | None = None
    window_width: int | None = None
    window_height: int | None = None
    wm_size: str | None = None
    wm_density: str | None = None
    current_package: str | None = None
    current_activity: str | None = None
    error: str | None = None


class GuiGetDeviceInfoRequest(ExecutorRequest[GuiGetDeviceInfoResponse]):
    executor_name: ClassVar[str] = 'gui_get_device_info'
    response_type: ClassVar[type[BaseModel]] = GuiGetDeviceInfoResponse


__all__ = [
    'GuiClickRequest',
    'GuiClickResponse',
    'GuiCloseAppRequest',
    'GuiCloseAppResponse',
    'GuiComputer13Request',
    'GuiComputer13Response',
    'GuiDoubleClickRequest',
    'GuiDoubleClickResponse',
    'GuiDragRequest',
    'GuiDragResponse',
    'GuiDragToRequest',
    'GuiHotkeyRequest',
    'GuiHotkeyResponse',
    'GuiKeyDownRequest',
    'GuiKeyDownResponse',
    'GuiKeyUpRequest',
    'GuiKeyUpResponse',
    'GuiGetDeviceInfoRequest',
    'GuiGetDeviceInfoResponse',
    'GuiLaunchAppRequest',
    'GuiLaunchAppResponse',
    'GuiLongClickRequest',
    'GuiLongClickResponse',
    'GuiMouseDownRequest',
    'GuiMouseDownResponse',
    'GuiMouseUpRequest',
    'GuiMouseUpResponse',
    'GuiMoveToRequest',
    'GuiMoveToResponse',
    'GuiPressKeyRequest',
    'GuiPressKeyResponse',
    'GuiPyAutoGUIRequest',
    'GuiPyAutoGUIResponse',
    'GuiResetAppsRequest',
    'GuiResetAppsResponse',
    'GuiRightClickRequest',
    'GuiRightClickResponse',
    'GuiScreenSizeRequest',
    'GuiScreenSizeResponse',
    'GuiScrollRequest',
    'GuiScrollResponse',
    'GuiScreenshotRequest',
    'GuiScreenshotResponse',
    'GuiStartRequest',
    'GuiStartResponse',
    'GuiTypeRequest',
    'GuiTypeResponse',
]
