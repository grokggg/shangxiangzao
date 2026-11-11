# /// script
# dependencies = []
#
# [tool.nexus.executors]
# gui_click = "nexus.runtime.executors.gui"
# gui_computer_13 = "nexus.runtime.executors.gui"
# gui_double_click = "nexus.runtime.executors.gui"
# gui_long_click = "nexus.runtime.executors.gui"
# gui_drag = "nexus.runtime.executors.gui"
# gui_drag_to = "nexus.runtime.executors.gui"
# gui_hotkey = "nexus.runtime.executors.gui"
# gui_key_down = "nexus.runtime.executors.gui"
# gui_key_up = "nexus.runtime.executors.gui"
# gui_mouse_down = "nexus.runtime.executors.gui"
# gui_mouse_up = "nexus.runtime.executors.gui"
# gui_move_to = "nexus.runtime.executors.gui"
# gui_type = "nexus.runtime.executors.gui"
# gui_press_key = "nexus.runtime.executors.gui"
# gui_pyautogui = "nexus.runtime.executors.gui"
# gui_right_click = "nexus.runtime.executors.gui"
# gui_scroll = "nexus.runtime.executors.gui"
# gui_start = "nexus.runtime.executors.gui"
# gui_screenshot = "nexus.runtime.executors.gui"
# gui_get_screen_size = "nexus.runtime.executors.gui"
# gui_launch_app = "nexus.runtime.executors.gui"
# gui_close_app = "nexus.runtime.executors.gui"
# gui_reset_apps = "nexus.runtime.executors.gui"
# gui_get_device_info = "nexus.runtime.executors.gui"
# ///

"""GUI executor implementations backed by the local runtime GUI backend."""

import asyncio
import base64

from nexus.core import executor_registry
from nexus.runtime.executors.models.gui import *  # noqa: F401,F403
from nexus.runtime.executors.models.gui import (
    GuiClickRequest,
    GuiClickResponse,
    GuiCloseAppRequest,
    GuiCloseAppResponse,
    GuiComputer13Request,
    GuiComputer13Response,
    GuiDoubleClickRequest,
    GuiDoubleClickResponse,
    GuiDragRequest,
    GuiDragResponse,
    GuiDragToRequest,
    GuiGetDeviceInfoRequest,
    GuiGetDeviceInfoResponse,
    GuiHotkeyRequest,
    GuiHotkeyResponse,
    GuiKeyDownRequest,
    GuiKeyDownResponse,
    GuiKeyUpRequest,
    GuiKeyUpResponse,
    GuiLaunchAppRequest,
    GuiLaunchAppResponse,
    GuiLongClickRequest,
    GuiLongClickResponse,
    GuiMouseDownRequest,
    GuiMouseDownResponse,
    GuiMouseUpRequest,
    GuiMouseUpResponse,
    GuiMoveToRequest,
    GuiMoveToResponse,
    GuiPressKeyRequest,
    GuiPressKeyResponse,
    GuiPyAutoGUIRequest,
    GuiPyAutoGUIResponse,
    GuiResetAppsRequest,
    GuiResetAppsResponse,
    GuiRightClickRequest,
    GuiRightClickResponse,
    GuiScreenSizeRequest,
    GuiScreenSizeResponse,
    GuiScreenshotRequest,
    GuiScreenshotResponse,
    GuiScrollRequest,
    GuiScrollResponse,
    GuiStartRequest,
    GuiStartResponse,
    GuiTypeRequest,
    GuiTypeResponse,
)
from nexus.runtime.server import gui


def _success_response(payload: dict[str, object] | None = None) -> tuple[bool, str | None]:
    if payload is None:
        return True, None
    success = bool(payload.get('success', True))
    error = payload.get('error')
    return success, str(error) if error else None


def _optional_str(value: object) -> str | None:
    return str(value) if value is not None else None


def _optional_int(value: object) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value)
    return None


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def _int_param(parameters: dict[str, object], name: str, default: int | None = None) -> int:
    value = parameters.get(name, default)
    if isinstance(value, bool) or value is None:
        raise ValueError(f'Missing integer parameter: {name}')
    if isinstance(value, int | float | str):
        return int(value)
    raise ValueError(f'Invalid integer parameter: {name}')


def _optional_int_param(parameters: dict[str, object], name: str) -> int | None:
    value = parameters.get(name)
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int | float | str):
        return int(value)
    return None


def _str_param(parameters: dict[str, object], name: str, default: str | None = None) -> str:
    value = parameters.get(name, default)
    if value is None:
        raise ValueError(f'Missing string parameter: {name}')
    return str(value)


@executor_registry.register('gui_click')
async def gui_click(args: GuiClickRequest) -> GuiClickResponse:
    payload = await gui.click(args.x, args.y, button=args.button, num_clicks=args.num_clicks)
    success, error = _success_response(payload)
    return GuiClickResponse(success=success, error=error)


@executor_registry.register('gui_move_to')
async def gui_move_to(args: GuiMoveToRequest) -> GuiMoveToResponse:
    payload = await gui.move_to(args.x, args.y)
    success, error = _success_response(payload)
    return GuiMoveToResponse(success=success, error=error)


@executor_registry.register('gui_mouse_down')
async def gui_mouse_down(args: GuiMouseDownRequest) -> GuiMouseDownResponse:
    payload = await gui.mouse_down(button=args.button)
    success, error = _success_response(payload)
    return GuiMouseDownResponse(success=success, error=error)


@executor_registry.register('gui_mouse_up')
async def gui_mouse_up(args: GuiMouseUpRequest) -> GuiMouseUpResponse:
    payload = await gui.mouse_up(button=args.button)
    success, error = _success_response(payload)
    return GuiMouseUpResponse(success=success, error=error)


@executor_registry.register('gui_right_click')
async def gui_right_click(args: GuiRightClickRequest) -> GuiRightClickResponse:
    payload = await gui.right_click(args.x, args.y)
    success, error = _success_response(payload)
    return GuiRightClickResponse(success=success, error=error)


@executor_registry.register('gui_double_click')
async def gui_double_click(args: GuiDoubleClickRequest) -> GuiDoubleClickResponse:
    payload = await gui.double_click(args.x, args.y)
    success, error = _success_response(payload)
    return GuiDoubleClickResponse(success=success, error=error)


@executor_registry.register('gui_long_click')
async def gui_long_click(args: GuiLongClickRequest) -> GuiLongClickResponse:
    payload = await gui.long_click(args.x, args.y, duration_ms=args.duration)
    success, error = _success_response(payload)
    return GuiLongClickResponse(success=success, error=error)


@executor_registry.register('gui_drag')
async def gui_drag(args: GuiDragRequest) -> GuiDragResponse:
    payload = await gui.drag(args.x1, args.y1, args.x2, args.y2, duration_ms=args.duration)
    success, error = _success_response(payload)
    return GuiDragResponse(success=success, error=error)


@executor_registry.register('gui_drag_to')
async def gui_drag_to(args: GuiDragToRequest) -> GuiDragResponse:
    payload = await gui.drag_to(args.x, args.y, duration_ms=args.duration)
    success, error = _success_response(payload)
    return GuiDragResponse(success=success, error=error)


@executor_registry.register('gui_scroll')
async def gui_scroll(args: GuiScrollRequest) -> GuiScrollResponse:
    payload = await gui.scroll(dx=args.dx, dy=args.dy)
    success, error = _success_response(payload)
    return GuiScrollResponse(success=success, error=error)


@executor_registry.register('gui_type')
async def gui_type(args: GuiTypeRequest) -> GuiTypeResponse:
    payload = await gui.type_text(args.text)
    success, error = _success_response(payload)
    return GuiTypeResponse(success=success, error=error)


@executor_registry.register('gui_press_key')
async def gui_press_key(args: GuiPressKeyRequest) -> GuiPressKeyResponse:
    payload = await gui.press_key(args.key)
    success, error = _success_response(payload)
    return GuiPressKeyResponse(success=success, error=error)


@executor_registry.register('gui_key_down')
async def gui_key_down(args: GuiKeyDownRequest) -> GuiKeyDownResponse:
    payload = await gui.key_down(args.key)
    success, error = _success_response(payload)
    return GuiKeyDownResponse(success=success, error=error)


@executor_registry.register('gui_key_up')
async def gui_key_up(args: GuiKeyUpRequest) -> GuiKeyUpResponse:
    payload = await gui.key_up(args.key)
    success, error = _success_response(payload)
    return GuiKeyUpResponse(success=success, error=error)


@executor_registry.register('gui_hotkey')
async def gui_hotkey(args: GuiHotkeyRequest) -> GuiHotkeyResponse:
    payload = await gui.hotkey(args.keys)
    success, error = _success_response(payload)
    return GuiHotkeyResponse(success=success, error=error)


@executor_registry.register('gui_pyautogui')
async def gui_pyautogui(args: GuiPyAutoGUIRequest) -> GuiPyAutoGUIResponse:
    payload = await gui.run_pyautogui(args.code, timeout=args.timeout, user=args.user, home_dir=args.home_dir)
    return GuiPyAutoGUIResponse(
        success=bool(payload.get('success', True)),
        stdout=_optional_str(payload.get('stdout')),
        stderr=_optional_str(payload.get('stderr')),
        error=_optional_str(payload.get('error')),
    )


@executor_registry.register('gui_computer_13')
async def gui_computer_13(args: GuiComputer13Request) -> GuiComputer13Response:
    parameters = dict(args.parameters)
    try:
        match args.action_type:
            case 'MOVE_TO':
                payload = await gui.move_to(_int_param(parameters, 'x'), _int_param(parameters, 'y'))
            case 'CLICK':
                payload = await gui.click(
                    _optional_int_param(parameters, 'x'),
                    _optional_int_param(parameters, 'y'),
                    button=_str_param(parameters, 'button', 'left'),
                    num_clicks=_int_param(parameters, 'num_clicks', 1),
                )
            case 'MOUSE_DOWN':
                payload = await gui.mouse_down(button=_str_param(parameters, 'button', 'left'))
            case 'MOUSE_UP':
                payload = await gui.mouse_up(button=_str_param(parameters, 'button', 'left'))
            case 'RIGHT_CLICK':
                payload = await gui.right_click(
                    _optional_int_param(parameters, 'x'), _optional_int_param(parameters, 'y')
                )
            case 'DOUBLE_CLICK':
                payload = await gui.double_click(
                    _optional_int_param(parameters, 'x'), _optional_int_param(parameters, 'y')
                )
            case 'DRAG_TO':
                payload = await gui.drag_to(
                    _int_param(parameters, 'x'),
                    _int_param(parameters, 'y'),
                    duration_ms=_int_param(parameters, 'duration', 300),
                )
            case 'SCROLL':
                payload = await gui.scroll(dx=_int_param(parameters, 'dx', 0), dy=_int_param(parameters, 'dy', 0))
            case 'TYPING':
                payload = await gui.type_text(_str_param(parameters, 'text'))
            case 'PRESS':
                payload = await gui.press_key(_str_param(parameters, 'key'))
            case 'KEY_DOWN':
                payload = await gui.key_down(_str_param(parameters, 'key'))
            case 'KEY_UP':
                payload = await gui.key_up(_str_param(parameters, 'key'))
            case 'HOTKEY':
                keys = parameters.get('keys')
                if not isinstance(keys, list) or not all(isinstance(key, str) for key in keys):
                    raise ValueError('HOTKEY requires a string list parameter: keys')
                payload = await gui.hotkey(keys)
            case 'WAIT':
                await asyncio.sleep(float(parameters.get('duration', 2.0)))
                return GuiComputer13Response(success=True)
            case 'DONE':
                return GuiComputer13Response(success=True, done=True)
            case 'FAIL':
                return GuiComputer13Response(success=True, fail=True)
    except Exception as exc:
        return GuiComputer13Response(success=False, error=str(exc))

    success, error = _success_response(payload)
    return GuiComputer13Response(success=success, error=error)


@executor_registry.register('gui_start')
async def gui_start(args: GuiStartRequest) -> GuiStartResponse:
    payload = await gui.start_stack(
        desktop_session=args.desktop_session,
        session_user=args.session_user,
        session_home_dir=args.session_home_dir,
        window_manager=args.window_manager,
        screen_width=args.screen_width,
        screen_height=args.screen_height,
        **args.session_options,
    )
    return GuiStartResponse(
        success=bool(payload.get('success', True)),
        display=_optional_str(payload.get('display')),
        vnc_port=_optional_int(payload.get('vnc_port')),
        novnc_port=_optional_int(payload.get('novnc_port')),
        desktop_session=_optional_str(payload.get('desktop_session')),
        stdout=_optional_str(payload.get('stdout')),
        stderr=_optional_str(payload.get('stderr')),
        error=_optional_str(payload.get('error')),
    )


@executor_registry.register('gui_screenshot')
async def gui_screenshot(_args: GuiScreenshotRequest) -> GuiScreenshotResponse:
    image_bytes, width, height = await gui.screenshot()
    return GuiScreenshotResponse(
        success=True,
        image=base64.b64encode(image_bytes).decode('ascii'),
        width=width,
        height=height,
        format='png',
    )


@executor_registry.register('gui_get_screen_size')
async def gui_get_screen_size(_args: GuiScreenSizeRequest) -> GuiScreenSizeResponse:
    width, height = await gui.screen_size()
    return GuiScreenSizeResponse(success=True, width=width, height=height)


@executor_registry.register('gui_launch_app')
async def gui_launch_app(args: GuiLaunchAppRequest) -> GuiLaunchAppResponse:
    payload = await gui.launch_app(args.app, user=args.user, home_dir=args.home_dir)
    success, error = _success_response(payload)
    return GuiLaunchAppResponse(success=success, error=error)


@executor_registry.register('gui_close_app')
async def gui_close_app(args: GuiCloseAppRequest) -> GuiCloseAppResponse:
    payload = await gui.close_app(args.app)
    success, error = _success_response(payload)
    return GuiCloseAppResponse(success=success, error=error)


@executor_registry.register('gui_reset_apps')
async def gui_reset_apps(_args: GuiResetAppsRequest) -> GuiResetAppsResponse:
    payload = await gui.reset_apps()
    return GuiResetAppsResponse(
        success=bool(payload.get('success', True)),
        stopped_packages=_string_list(payload.get('stopped_packages')),
        error=_optional_str(payload.get('error')),
    )


@executor_registry.register('gui_get_device_info')
async def gui_get_device_info(_args: GuiGetDeviceInfoRequest) -> GuiGetDeviceInfoResponse:
    body = await gui.get_device_info()
    return GuiGetDeviceInfoResponse(
        success=bool(body.get('success', True)),
        device_name=_optional_str(body.get('device_name')),
        platform_version=_optional_str(body.get('platform_version')),
        automation_name=_optional_str(body.get('automation_name')),
        window_width=_optional_int(body.get('window_width')),
        window_height=_optional_int(body.get('window_height')),
        wm_size=_optional_str(body.get('wm_size')),
        wm_density=_optional_str(body.get('wm_density')),
        current_package=_optional_str(body.get('current_package')),
        current_activity=_optional_str(body.get('current_activity')),
        error=_optional_str(body.get('error')),
    )


__all__ = [
    'gui_click',
    'gui_close_app',
    'gui_computer_13',
    'gui_double_click',
    'gui_drag',
    'gui_drag_to',
    'gui_get_device_info',
    'gui_get_screen_size',
    'gui_hotkey',
    'gui_key_down',
    'gui_key_up',
    'gui_launch_app',
    'gui_long_click',
    'gui_mouse_down',
    'gui_mouse_up',
    'gui_move_to',
    'gui_press_key',
    'gui_pyautogui',
    'gui_reset_apps',
    'gui_right_click',
    'gui_screenshot',
    'gui_scroll',
    'gui_start',
    'gui_type',
]
