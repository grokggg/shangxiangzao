"""Runtime abstract base class and ExecutorMixin.

Provides the core abstractions for runtime client implementations:
- ``ExecutorMixin``: Typed convenience methods for executor calls
- ``SupportsCallExecutor``: Protocol for call_executor dispatch
- ``Runtime``: Abstract base class combining ExecutorMixin with transport
"""

from __future__ import annotations

import asyncio
import logging

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, Literal, Protocol, overload, runtime_checkable

from nexus.core.tool import ToolContext
from nexus.runtime.client.models import VenvHandle
from nexus.runtime.executors.models.base import ExecutorRequest, ResponseT
from nexus.utils.otel.transport import transport_span


logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from nexus.core.tool import ToolResponse
    from nexus.runtime.executors.models.browser import (
        BidLocator,
        ClickResponse,
        CloseTabResponse,
        CreateBrowserSessionResponse,
        DestroyBrowserSessionResponse,
        ElementLocator,
        EvalJsResponse,
        FillResponse,
        FocusTabResponse,
        GetContentResponse,
        GetObservationResponse,
        GetUrlResponse,
        GoBackResponse,
        GoForwardResponse,
        GotoResponse,
        HoverResponse,
        KeyboardTypeResponse,
        ListTabsResponse,
        NewTabResponse,
        PressKeyResponse,
        ScreenshotResponse,
        ScrollResponse,
        SelectOptionResponse,
        SelectorLocator,
        SetGeolocationResponse,
        WaitForLoadStateResponse,
        WaitForSelectorResponse,
    )
    from nexus.runtime.executors.models.command import RunCommandResponse
    from nexus.runtime.executors.models.file import DownloadFileResponse, FileItem, UploadFileResponse
    from nexus.runtime.executors.models.gui import (
        GuiClickResponse,
        GuiCloseAppResponse,
        GuiComputer13Response,
        GuiDoubleClickResponse,
        GuiDragResponse,
        GuiGetDeviceInfoResponse,
        GuiHotkeyResponse,
        GuiKeyDownResponse,
        GuiKeyUpResponse,
        GuiLaunchAppResponse,
        GuiLongClickResponse,
        GuiMouseDownResponse,
        GuiMouseUpResponse,
        GuiMoveToResponse,
        GuiPressKeyResponse,
        GuiPyAutoGUIResponse,
        GuiResetAppsResponse,
        GuiRightClickResponse,
        GuiScreenSizeResponse,
        GuiScreenshotResponse,
        GuiScrollResponse,
        GuiStartResponse,
        GuiTypeResponse,
    )
    from nexus.runtime.executors.models.jupyter import (
        CreateIPythonSessionResponse,
        DestroyIPythonSessionResponse,
        ExecuteIPythonResponse,
    )
    from nexus.runtime.executors.models.terminal import (
        CapturePaneResponse,
        CommandInfo,
        CreateTerminalSessionResponse,
        DestroyTerminalSessionResponse,
        SendKeysToSessionResponse,
        TerminateSessionProcessesResponse,
    )
    from nexus.runtime.models import InvokeRequest


# ============== Exceptions ==============


class RuntimeUnhealthyError(RuntimeError):
    """Raised when an API call is made against a runtime that has been marked unhealthy.

    A runtime becomes unhealthy when an external watchdog (e.g. GongfengRuntimeProvider's
    status poller) observes that the remote sandbox is no longer in a usable state
    (stopped, OOMKilled, SandboxExit, etc.).
    """


# ============== Executor Mixin ==============


@runtime_checkable
class SupportsCallExecutor(Protocol):
    """Protocol for classes that support call_executor method."""

    async def call_executor(self, executor_name: str, arg: Any) -> Any: ...


class ExecutorMixin:
    """Mixin providing typed convenience methods for executor calls.

    Two usage styles:

    1. **Generic call** (power users)::

           response = await runtime.call(RunCommandRequest(command='ls'))
           # response is RunCommandResponse — fully typed

    2. **Thin wrappers** (convenience)::

           response = await runtime.run_command(command='ls')

    Subclasses **must** implement ``call_executor(executor_name, arg) -> Any``.
    """

    async def call_executor(self, executor_name: str, arg: Any) -> Any:
        """Call an executor by name. Subclasses must override."""
        raise NotImplementedError

    async def call(self, request: ExecutorRequest[ResponseT]) -> ResponseT:
        """Dispatch any ExecutorRequest and return the typed response."""
        req_type = type(request)
        result = await self.call_executor(req_type.executor_name, request.model_dump())
        return req_type.response_type.model_validate(result)  # type: ignore[return-value]

    # ==================== Command ====================

    async def run_command(
        self,
        command: str,
        *,
        cwd: str | None = None,
        timeout: float = 180,
        stdin: str | None = None,
        extra_env: dict[str, str] | None = None,
    ) -> RunCommandResponse:
        from nexus.runtime.executors.models.command import RunCommandRequest

        return await self.call(
            RunCommandRequest(command=command, cwd=cwd, timeout=timeout, stdin=stdin, extra_env=extra_env or {})
        )

    # ==================== File ====================

    async def upload_file(
        self,
        files: list[FileItem],
        *,
        overwrite: bool = True,
    ) -> UploadFileResponse:
        from nexus.runtime.executors.models.file import UploadFileRequest

        return await self.call(UploadFileRequest(files=files, overwrite=overwrite))

    async def download_file(
        self,
        files: list[FileItem],
    ) -> DownloadFileResponse:
        from nexus.runtime.executors.models.file import DownloadFileRequest

        return await self.call(DownloadFileRequest(files=files))

    # ==================== IPython/Jupyter ====================

    async def create_ipython_session(
        self,
        *,
        python_executable: str | None = None,
    ) -> CreateIPythonSessionResponse:
        from nexus.runtime.executors.models.jupyter import CreateIPythonSessionRequest

        return await self.call(CreateIPythonSessionRequest(python_executable=python_executable))

    async def destroy_ipython_session(
        self,
        session_id: str,
    ) -> DestroyIPythonSessionResponse:
        from nexus.runtime.executors.models.jupyter import DestroyIPythonSessionRequest

        return await self.call(DestroyIPythonSessionRequest(session_id=session_id))

    async def execute_ipython(
        self,
        session_id: str,
        code: str,
        *,
        timeout: float = 180,
    ) -> ExecuteIPythonResponse:
        from nexus.runtime.executors.models.jupyter import ExecuteIPythonRequest

        return await self.call(ExecuteIPythonRequest(session_id=session_id, code=code, timeout=timeout))

    # ==================== Terminal ====================

    async def create_terminal_session(
        self,
        *,
        session_id: str | None = None,
        height: int = 40,
        width: int = 120,
    ) -> CreateTerminalSessionResponse:
        from nexus.runtime.executors.models.terminal import CreateTerminalSessionRequest

        return await self.call(CreateTerminalSessionRequest(session_id=session_id, height=height, width=width))

    async def destroy_terminal_session(
        self,
        session_id: str,
    ) -> DestroyTerminalSessionResponse:
        from nexus.runtime.executors.models.terminal import DestroyTerminalSessionRequest

        return await self.call(DestroyTerminalSessionRequest(session_id=session_id))

    async def run_command_in_terminal_session(
        self,
        session_id: str,
        command: str,
        *,
        timeout: float | None = None,
    ) -> CommandInfo:
        from nexus.runtime.executors.models.terminal import RunCommandInSessionRequest

        return await self.call(RunCommandInSessionRequest(session_id=session_id, command=command, timeout=timeout))

    async def start_command_in_terminal_session(
        self,
        session_id: str,
        command: str,
    ) -> CommandInfo:
        from nexus.runtime.executors.models.terminal import StartCommandInSessionRequest

        return await self.call(StartCommandInSessionRequest(session_id=session_id, command=command))

    async def query_terminal_command_status(
        self,
        session_id: str,
        command_id: int,
    ) -> CommandInfo:
        from nexus.runtime.executors.models.terminal import QueryCommandStatusRequest

        return await self.call(QueryCommandStatusRequest(session_id=session_id, command_id=command_id))

    async def send_keys_to_terminal_session(
        self,
        session_id: str,
        keys: str | list[str],
    ) -> SendKeysToSessionResponse:
        from nexus.runtime.executors.models.terminal import SendKeysToSessionRequest

        return await self.call(SendKeysToSessionRequest(session_id=session_id, keys=keys))

    async def capture_terminal_pane(
        self,
        session_id: str,
        *,
        capture_entire: bool = False,
    ) -> CapturePaneResponse:
        from nexus.runtime.executors.models.terminal import CapturePaneRequest

        return await self.call(CapturePaneRequest(session_id=session_id, capture_entire=capture_entire))

    async def terminate_terminal_session_processes(
        self,
        session_id: str,
    ) -> TerminateSessionProcessesResponse:
        from nexus.runtime.executors.models.terminal import TerminateSessionProcessesRequest

        return await self.call(TerminateSessionProcessesRequest(session_id=session_id))

    # ==================== Browser ====================

    async def create_browser_session(
        self,
        *,
        headless: bool = True,
        viewport_width: int = 1280,
        viewport_height: int = 720,
        timeout: int = 10000,
        executable_path: str = '/nix/runtime/nix-env/bin/chromium',
        user_agent: str | None = None,
        demo_mode: str = 'off',
        record_har: bool = False,
        har_path: str | None = None,
        har_content: str = 'embed',
        har_url_filter: str | None = None,
    ) -> CreateBrowserSessionResponse:
        from nexus.runtime.executors.models.browser import CreateBrowserSessionRequest

        return await self.call(
            CreateBrowserSessionRequest(
                headless=headless,
                viewport_width=viewport_width,
                viewport_height=viewport_height,
                timeout=timeout,
                executable_path=executable_path,
                user_agent=user_agent,
                demo_mode=demo_mode,
                record_har=record_har,
                har_path=har_path,
                har_content=har_content,
                har_url_filter=har_url_filter,
            )
        )

    async def destroy_browser_session(
        self,
        session_id: str,
    ) -> DestroyBrowserSessionResponse:
        from nexus.runtime.executors.models.browser import DestroyBrowserSessionRequest

        return await self.call(DestroyBrowserSessionRequest(session_id=session_id))

    async def browser_goto(
        self,
        session_id: str,
        url: str,
        *,
        timeout: int | None = None,
    ) -> GotoResponse:
        from nexus.runtime.executors.models.browser import GotoRequest

        return await self.call(GotoRequest(session_id=session_id, url=url, timeout=timeout))

    async def browser_click(
        self,
        session_id: str,
        bid: str | None = None,
        *,
        locator: ElementLocator | dict[str, Any] | None = None,
        timeout: int | None = None,
        button: Literal['left', 'right', 'middle'] = 'left',
    ) -> ClickResponse:
        from nexus.runtime.executors.models.browser import ClickRequest

        request = ClickRequest.model_validate({
            'session_id': session_id,
            'bid': bid,
            'locator': locator,
            'timeout': timeout,
            'button': button,
        })
        return await self.call(request)

    async def browser_fill(
        self,
        session_id: str,
        bid: str | None = None,
        value: str = '',
        *,
        locator: BidLocator | SelectorLocator | dict[str, Any] | None = None,
        timeout: int | None = None,
    ) -> FillResponse:
        from nexus.runtime.executors.models.browser import FillRequest

        return await self.call(
            FillRequest(session_id=session_id, bid=bid, locator=locator, value=value, timeout=timeout)  # ty: ignore[invalid-argument-type]
        )

    async def browser_scroll(
        self,
        session_id: str,
        direction: str,
        *,
        amount: int = 500,
    ) -> ScrollResponse:
        from nexus.runtime.executors.models.browser import ScrollRequest

        return await self.call(ScrollRequest(session_id=session_id, direction=direction, amount=amount))

    async def browser_press_key(
        self,
        session_id: str,
        key: str,
        *,
        bid: str | None = None,
        locator: BidLocator | SelectorLocator | dict[str, Any] | None = None,
    ) -> PressKeyResponse:
        from nexus.runtime.executors.models.browser import PressKeyRequest

        return await self.call(PressKeyRequest(session_id=session_id, key=key, bid=bid, locator=locator))  # ty: ignore[invalid-argument-type]

    async def browser_keyboard_type(
        self,
        session_id: str,
        text: str,
        *,
        delay: int = 0,
    ) -> KeyboardTypeResponse:
        from nexus.runtime.executors.models.browser import KeyboardTypeRequest

        return await self.call(KeyboardTypeRequest(session_id=session_id, text=text, delay=delay))

    async def browser_hover(
        self,
        session_id: str,
        bid: str | None = None,
        *,
        locator: ElementLocator | dict[str, Any] | None = None,
        timeout: int | None = None,
    ) -> HoverResponse:
        from nexus.runtime.executors.models.browser import HoverRequest

        return await self.call(HoverRequest(session_id=session_id, bid=bid, locator=locator, timeout=timeout))  # ty: ignore[invalid-argument-type]

    async def browser_select_option(
        self,
        session_id: str,
        bid: str | None = None,
        value: str = '',
        *,
        locator: BidLocator | SelectorLocator | dict[str, Any] | None = None,
        timeout: int | None = None,
    ) -> SelectOptionResponse:
        from nexus.runtime.executors.models.browser import SelectOptionRequest

        return await self.call(
            SelectOptionRequest(session_id=session_id, bid=bid, locator=locator, value=value, timeout=timeout)  # ty: ignore[invalid-argument-type]
        )

    async def browser_new_tab(
        self,
        session_id: str,
        *,
        url: str | None = None,
    ) -> NewTabResponse:
        from nexus.runtime.executors.models.browser import NewTabRequest

        return await self.call(NewTabRequest(session_id=session_id, url=url))

    async def browser_close_tab(
        self,
        session_id: str,
        *,
        tab_index: int | None = None,
    ) -> CloseTabResponse:
        from nexus.runtime.executors.models.browser import CloseTabRequest

        return await self.call(CloseTabRequest(session_id=session_id, tab_index=tab_index))

    async def browser_focus_tab(
        self,
        session_id: str,
        tab_index: int,
    ) -> FocusTabResponse:
        from nexus.runtime.executors.models.browser import FocusTabRequest

        return await self.call(FocusTabRequest(session_id=session_id, tab_index=tab_index))

    async def browser_list_tabs(
        self,
        session_id: str,
    ) -> ListTabsResponse:
        from nexus.runtime.executors.models.browser import ListTabsRequest

        return await self.call(ListTabsRequest(session_id=session_id))

    async def browser_go_back(
        self,
        session_id: str,
    ) -> GoBackResponse:
        from nexus.runtime.executors.models.browser import GoBackRequest

        return await self.call(GoBackRequest(session_id=session_id))

    async def browser_go_forward(
        self,
        session_id: str,
    ) -> GoForwardResponse:
        from nexus.runtime.executors.models.browser import GoForwardRequest

        return await self.call(GoForwardRequest(session_id=session_id))

    async def browser_get_observation(
        self,
        session_id: str,
        *,
        include_screenshot: bool = False,
        screenshot_scale_factor: float = 1.0,
        viewport_only: bool = False,
    ) -> GetObservationResponse:
        from nexus.runtime.executors.models.browser import GetObservationRequest

        return await self.call(
            GetObservationRequest(
                session_id=session_id,
                include_screenshot=include_screenshot,
                screenshot_scale_factor=screenshot_scale_factor,
                viewport_only=viewport_only,
            )
        )

    async def browser_screenshot(
        self,
        session_id: str,
        *,
        options: dict[str, Any] | None = None,
        image_type: Literal['png', 'jpeg'] | None = None,
        quality: int | None = None,
        full_page: bool | None = None,
        clip: dict[str, Any] | None = None,
        omit_background: bool | None = None,
        timeout: int | None = None,
        animations: Literal['allow', 'disabled'] | None = None,
        caret: Literal['hide', 'initial'] | None = None,
        scale: Literal['css', 'device'] | None = None,
        mask_color: str | None = None,
        style: str | None = None,
        path: str | None = None,
        **kwargs: Any,
    ) -> ScreenshotResponse:
        from nexus.runtime.executors.models.browser import ScreenshotRequest

        request_kwargs = {
            'session_id': session_id,
            'options': options or {},
            'quality': quality,
            'full_page': full_page,
            'clip': clip,
            'omit_background': omit_background,
            'timeout': timeout,
            'animations': animations,
            'caret': caret,
            'scale': scale,
            'mask_color': mask_color,
            'style': style,
            'path': path,
            **kwargs,
        }
        if image_type is not None:
            request_kwargs['type'] = image_type

        return await self.call(ScreenshotRequest.model_validate(request_kwargs))

    async def browser_get_url(
        self,
        session_id: str,
    ) -> GetUrlResponse:
        from nexus.runtime.executors.models.browser import GetUrlRequest

        return await self.call(GetUrlRequest(session_id=session_id))

    async def browser_eval_js(
        self,
        session_id: str,
        expression: str,
    ) -> EvalJsResponse:
        from nexus.runtime.executors.models.browser import EvalJsRequest

        return await self.call(EvalJsRequest(session_id=session_id, expression=expression))

    async def browser_wait_for_selector(
        self,
        session_id: str,
        selector: str,
        *,
        timeout: int = 30000,
        state: str = 'visible',
    ) -> WaitForSelectorResponse:
        from nexus.runtime.executors.models.browser import WaitForSelectorRequest

        return await self.call(
            WaitForSelectorRequest(session_id=session_id, selector=selector, timeout=timeout, state=state)  # ty: ignore[invalid-argument-type]
        )

    async def browser_set_geolocation(
        self,
        session_id: str,
        *,
        latitude: float,
        longitude: float,
        accuracy: float = 0,
    ) -> SetGeolocationResponse:
        from nexus.runtime.executors.models.browser import SetGeolocationRequest

        return await self.call(
            SetGeolocationRequest(session_id=session_id, latitude=latitude, longitude=longitude, accuracy=accuracy)
        )

    async def browser_wait_for_load_state(
        self,
        session_id: str,
        *,
        state: str = 'load',
        timeout: int = 30000,
    ) -> WaitForLoadStateResponse:
        from nexus.runtime.executors.models.browser import WaitForLoadStateRequest

        return await self.call(
            WaitForLoadStateRequest(session_id=session_id, state=state, timeout=timeout)  # ty: ignore[invalid-argument-type]
        )

    async def browser_get_content(
        self,
        session_id: str,
        *,
        url: str | None = None,
    ) -> GetContentResponse:
        from nexus.runtime.executors.models.browser import GetContentRequest

        return await self.call(GetContentRequest(session_id=session_id, url=url))

    # ==================== GUI ====================

    async def gui_click(
        self,
        x: int | None = None,
        y: int | None = None,
        *,
        button: Literal['left', 'right', 'middle'] = 'left',
        num_clicks: int = 1,
    ) -> GuiClickResponse:
        from nexus.runtime.executors.models.gui import GuiClickRequest

        return await self.call(GuiClickRequest(x=x, y=y, button=button, num_clicks=num_clicks))

    async def gui_move_to(self, x: int, y: int) -> GuiMoveToResponse:
        from nexus.runtime.executors.models.gui import GuiMoveToRequest

        return await self.call(GuiMoveToRequest(x=x, y=y))

    async def gui_mouse_down(self, *, button: Literal['left', 'right', 'middle'] = 'left') -> GuiMouseDownResponse:
        from nexus.runtime.executors.models.gui import GuiMouseDownRequest

        return await self.call(GuiMouseDownRequest(button=button))

    async def gui_mouse_up(self, *, button: Literal['left', 'right', 'middle'] = 'left') -> GuiMouseUpResponse:
        from nexus.runtime.executors.models.gui import GuiMouseUpRequest

        return await self.call(GuiMouseUpRequest(button=button))

    async def gui_right_click(self, x: int | None = None, y: int | None = None) -> GuiRightClickResponse:
        from nexus.runtime.executors.models.gui import GuiRightClickRequest

        return await self.call(GuiRightClickRequest(x=x, y=y))

    async def gui_double_click(self, x: int | None = None, y: int | None = None) -> GuiDoubleClickResponse:
        from nexus.runtime.executors.models.gui import GuiDoubleClickRequest

        return await self.call(GuiDoubleClickRequest(x=x, y=y))

    async def gui_long_click(self, x: int, y: int, *, duration: int = 1000) -> GuiLongClickResponse:
        from nexus.runtime.executors.models.gui import GuiLongClickRequest

        return await self.call(GuiLongClickRequest(x=x, y=y, duration=duration))

    async def gui_drag(self, x1: int, y1: int, x2: int, y2: int, *, duration: int = 300) -> GuiDragResponse:
        from nexus.runtime.executors.models.gui import GuiDragRequest

        return await self.call(GuiDragRequest(x1=x1, y1=y1, x2=x2, y2=y2, duration=duration))

    async def gui_drag_to(self, x: int, y: int, *, duration: int = 300) -> GuiDragResponse:
        from nexus.runtime.executors.models.gui import GuiDragToRequest

        return await self.call(GuiDragToRequest(x=x, y=y, duration=duration))

    async def gui_scroll(self, *, dx: int = 0, dy: int = 0) -> GuiScrollResponse:
        from nexus.runtime.executors.models.gui import GuiScrollRequest

        return await self.call(GuiScrollRequest(dx=dx, dy=dy))

    async def gui_type(self, text: str) -> GuiTypeResponse:
        from nexus.runtime.executors.models.gui import GuiTypeRequest

        return await self.call(GuiTypeRequest(text=text))

    async def gui_press_key(self, key: str) -> GuiPressKeyResponse:
        from nexus.runtime.executors.models.gui import GuiPressKeyRequest

        return await self.call(GuiPressKeyRequest(key=key))

    async def gui_key_down(self, key: str) -> GuiKeyDownResponse:
        from nexus.runtime.executors.models.gui import GuiKeyDownRequest

        return await self.call(GuiKeyDownRequest(key=key))

    async def gui_key_up(self, key: str) -> GuiKeyUpResponse:
        from nexus.runtime.executors.models.gui import GuiKeyUpRequest

        return await self.call(GuiKeyUpRequest(key=key))

    async def gui_hotkey(self, keys: list[str]) -> GuiHotkeyResponse:
        from nexus.runtime.executors.models.gui import GuiHotkeyRequest

        return await self.call(GuiHotkeyRequest(keys=keys))

    async def gui_pyautogui(
        self,
        code: str,
        *,
        timeout: float = 30.0,
        user: str | None = None,
        home_dir: str | None = None,
    ) -> GuiPyAutoGUIResponse:
        from nexus.runtime.executors.models.gui import GuiPyAutoGUIRequest

        return await self.call(GuiPyAutoGUIRequest(code=code, timeout=timeout, user=user, home_dir=home_dir))

    async def gui_computer_13(
        self, action_type: str, parameters: dict[str, Any] | None = None
    ) -> GuiComputer13Response:
        from nexus.runtime.executors.models.gui import GuiComputer13Request

        return await self.call(GuiComputer13Request(action_type=action_type, parameters=parameters or {}))  # type: ignore[arg-type]

    async def gui_start(
        self,
        *,
        desktop_session: str | None = None,
        session_user: str | None = 'user',
        session_home_dir: str | None = '/home/user',
        window_manager: str | None = None,
        screen_width: int | None = None,
        screen_height: int | None = None,
        **session_options: object,
    ) -> GuiStartResponse:
        from nexus.runtime.executors.models.gui import GuiStartRequest

        return await self.call(
            GuiStartRequest(
                desktop_session=desktop_session,
                session_user=session_user,
                session_home_dir=session_home_dir,
                window_manager=window_manager,
                screen_width=screen_width,
                screen_height=screen_height,
                session_options=session_options,
            )
        )

    async def gui_screenshot(self) -> GuiScreenshotResponse:
        from nexus.runtime.executors.models.gui import GuiScreenshotRequest

        return await self.call(GuiScreenshotRequest())

    async def gui_get_screen_size(self) -> GuiScreenSizeResponse:
        from nexus.runtime.executors.models.gui import GuiScreenSizeRequest

        return await self.call(GuiScreenSizeRequest())

    async def gui_launch_app(
        self,
        app: str,
        *,
        user: str | None = 'user',
        home_dir: str | None = '/home/user',
    ) -> GuiLaunchAppResponse:
        from nexus.runtime.executors.models.gui import GuiLaunchAppRequest

        return await self.call(GuiLaunchAppRequest(app=app, user=user, home_dir=home_dir))

    async def gui_close_app(self, app: str) -> GuiCloseAppResponse:
        from nexus.runtime.executors.models.gui import GuiCloseAppRequest

        return await self.call(GuiCloseAppRequest(app=app))

    async def gui_get_device_info(self) -> GuiGetDeviceInfoResponse:
        from nexus.runtime.executors.models.gui import GuiGetDeviceInfoRequest

        return await self.call(GuiGetDeviceInfoRequest())

    async def gui_reset_apps(self) -> GuiResetAppsResponse:
        from nexus.runtime.executors.models.gui import GuiResetAppsRequest

        return await self.call(GuiResetAppsRequest())

    # Mobile-style aliases
    gui_tap = gui_click
    gui_double_tap = gui_double_click
    gui_long_press = gui_long_click
    gui_swipe = gui_drag

    # ==================== MCP ====================

    async def mcp_start_proxy(
        self,
        client: Any,
        *,
        server_name: str = 'mcp',
        nexus_managed: bool = False,
    ) -> Any:
        from nexus.core.tool.mcp import MCPClientConfig, MCPStartProxyRequest, MCPStartProxyResponse

        request = MCPStartProxyRequest(
            client=MCPClientConfig.model_validate(client),
            server_name=server_name,
            nexus_managed=nexus_managed,
        )
        result = await self.call_executor('mcp_start_proxy', request.model_dump())
        return MCPStartProxyResponse.model_validate(result)

    async def mcp_stop_proxy(self, proxy_id: str) -> Any:
        from nexus.runtime.executors.mcp import MCPStopProxyRequest, MCPStopProxyResponse

        request = MCPStopProxyRequest(proxy_id=proxy_id)
        result = await self.call_executor('mcp_stop_proxy', request.model_dump())
        return MCPStopProxyResponse.model_validate(result)


# ============== Runtime Base Class ==============


class Runtime(ExecutorMixin, ABC):
    """Base class for runtime implementations.

    Runtime provides a unified interface for agents/tasks to interact with
    execution environments (sandboxes, containers, etc.).

    Attributes:
        port_mappings: Dictionary mapping internal port numbers to external URLs.
                      Example: {8023: 'http://localhost:32768', 8080: 'http://localhost:32769'}
    """

    def __init__(self) -> None:
        """Initialize runtime with port mappings."""
        self.port_mappings: dict[int, str] = {}
        self.default_venv_id: str | None = None
        self._unhealthy_reason: str | None = None
        self._unhealthy_event: asyncio.Event = asyncio.Event()

    @property
    def base_url(self) -> str | None:
        """HTTP base URL of the runtime server. Used to build additional service endpoints (e.g. MCP proxy)."""
        return None

    async def drain_trace(self, timeout: float | None = None) -> None:  # noqa: ARG002
        """Wait until runtime trace events accepted so far reach the local trace controller."""

    # ==================== Health tracking ====================

    @property
    def is_unhealthy(self) -> bool:
        """Whether this runtime has been marked as unhealthy by an external watchdog."""
        return self._unhealthy_reason is not None

    @property
    def unhealthy_reason(self) -> str | None:
        """Human-readable reason the runtime was marked unhealthy, or None."""
        return self._unhealthy_reason

    def mark_unhealthy(self, reason: str) -> None:
        """Mark this runtime as unhealthy. Subsequent API calls will raise immediately.

        In-flight polls watching ``_unhealthy_event`` will wake up and abort.
        Calling this repeatedly is a no-op after the first call.
        """
        if self._unhealthy_reason is not None:
            return
        self._unhealthy_reason = reason
        self._unhealthy_event.set()
        logger.warning('Runtime marked unhealthy: %s', reason)

    def _raise_if_unhealthy(self) -> None:
        """Raise :class:`RuntimeUnhealthyError` if this runtime has been marked unhealthy."""
        if self._unhealthy_reason is not None:
            raise RuntimeUnhealthyError(self._unhealthy_reason)

    # ==================== Unified Invoke Primitive ====================

    @abstractmethod
    async def _invoke(self, request: InvokeRequest) -> Any:
        """Subclass transport implementation.

        Accepts a fully-formed :class:`InvokeRequest` and returns the
        deserialized return value, or raises the remote exception.
        """
        ...

    @overload
    async def invoke(self, request: InvokeRequest, /) -> Any: ...

    @overload
    async def invoke(self, target: str, /, *args: Any, venv_id: str | None = None, **kwargs: Any) -> Any: ...

    @overload
    async def invoke(
        self, target: Callable[..., Any], /, *args: Any, venv_id: str | None = None, **kwargs: Any
    ) -> Any: ...

    async def invoke(
        self,
        first: InvokeRequest | str | Callable[..., Any],
        /,
        *args: Any,
        venv_id: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """The single RPC primitive for remote execution.

        *first* can be:

        - A ready-made :class:`InvokeRequest` (full control)::

              await runtime.invoke(InvokeRequest.build(my_func, x, y))

        - An import-path string::

              await runtime.invoke('mod.sub::func', x, y)

        - A callable (import path derived automatically)::

              await runtime.invoke(my_func, x, y)
        """
        from nexus.runtime.models import InvokeRequest as _InvokeRequest

        if isinstance(first, _InvokeRequest):
            request = first
        else:
            # Apply default_venv_id when caller doesn't specify one
            effective_venv_id = venv_id if venv_id is not None else self.default_venv_id

            # str or callable — InvokeRequest.build handles both
            request = _InvokeRequest.build(first, *args, venv_id=effective_venv_id, **kwargs)

        with transport_span(
            'runtime.invoke',
            kind='client',
            attributes={
                'rpc.system': 'nexus.runtime',
                'rpc.method': 'invoke',
                'nexus.runtime.target': request.import_path,
            },
        ):
            return await self._invoke(request)

    # ==================== exec (convenience) ======================================

    async def exec(self, code: str, *, venv_id: str | None = None) -> Any:
        """Execute a Python code string in the runtime.

        Wraps the code in an ``async def`` so ``return`` and ``await``
        work naturally::

            result = await runtime.exec('''
                import os
                return os.listdir('/')
            ''')
        """
        return await self.invoke('nexus.utils.exec::exec_code', code, venv_id=venv_id)

    # ==================== Concrete call_* (delegate to invoke) ====================

    async def call_tool(
        self,
        import_path: str,
        params: dict[str, Any],
        context: ToolContext,
    ) -> ToolResponse:
        """Invoke a tool on the runtime.

        Delegates to :meth:`invoke` — the server auto-coerces *params*
        (dict) into the tool's input model.

        Args:
            import_path: Tool import path (e.g., "nexus.extensions.tools.sandbox::bash_tool")
            params: Tool parameters
            context: Tool context

        Returns:
            ToolResponse containing output, meta, and context_updates
        """
        return await self.invoke(import_path, params, context)

    async def call_executor(
        self,
        executor_name: str,
        arg: Any,
    ) -> Any:
        """Invoke a registered executor in the runtime environment.

        Delegates to :meth:`invoke` with the executor name. The server
        resolves it from the executor registry and auto-coerces the arg.
        Returns a dict (model_dump) for backward compatibility with
        :class:`ExecutorMixin` methods.
        """
        result = await self.invoke(executor_name, arg)
        if hasattr(result, 'model_dump'):
            return result.model_dump()
        return result

    # ==================== Venv Management ====================

    async def create_venv(
        self,
        spec: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> VenvHandle:
        """Create a worker venv in the runtime environment.

        Args:
            spec: VenvSpec as a dict (or pass fields as kwargs).

        Returns:
            VenvHandle with venv_id and status info.
        """
        raise NotImplementedError(f'{self.__class__.__name__} does not support create_venv()')

    async def destroy_venv(self, venv_id: str) -> None:
        """Destroy a worker venv."""
        raise NotImplementedError(f'{self.__class__.__name__} does not support destroy_venv()')

    # ==================== WebSocket Connection Management ====================

    async def ws_connect(self) -> None:
        """Establish WebSocket/Socket.IO connection to the runtime environment.

        Raises:
            NotImplementedError: If the runtime does not support WebSocket connections.
        """
        raise NotImplementedError(f'{self.__class__.__name__} does not support WebSocket connections')

    async def ws_disconnect(self) -> None:
        """Disconnect WebSocket/Socket.IO connection from the runtime environment.

        Raises:
            NotImplementedError: If the runtime does not support WebSocket connections.
        """
        raise NotImplementedError(f'{self.__class__.__name__} does not support WebSocket connections')

    @property
    def ws_connected(self) -> bool:
        """Check if the WebSocket/Socket.IO connection is currently established.

        Returns:
            True if connected, False otherwise.

        Raises:
            NotImplementedError: If the runtime does not support WebSocket connections.
        """
        raise NotImplementedError(f'{self.__class__.__name__} does not support WebSocket connections')

    # ==================== Event Management ====================

    def on(
        self,
        event: str,
        listener: Callable[..., None] | Callable[..., Awaitable[None]],
    ) -> Callable[[], None]:
        """Subscribe to runtime events with a callback listener.

        Args:
            event: The event name to listen for.
            listener: Callback function to invoke when the event is received.
                     Can be sync or async.

        Returns:
            A function that, when called, unsubscribes the listener.

        Raises:
            NotImplementedError: If the runtime does not support event subscription.
        """
        raise NotImplementedError(f'{self.__class__.__name__} does not support event subscription')

    def once(
        self,
        event: str,
        listener: Callable[..., None] | Callable[..., Awaitable[None]],
    ) -> Callable[[], None]:
        """Subscribe to a runtime event for a single invocation.

        The listener will be automatically unsubscribed after being called once.

        Args:
            event: The event name to listen for.
            listener: Callback function to invoke when the event is received.

        Returns:
            A function that, when called, unsubscribes the listener (if not yet triggered).

        Raises:
            NotImplementedError: If the runtime does not support event subscription.
        """
        raise NotImplementedError(f'{self.__class__.__name__} does not support event subscription')

    def off(
        self,
        event: str,
        listener: Callable[..., None] | Callable[..., Awaitable[None]] | None = None,
    ) -> None:
        """Unsubscribe from runtime events.

        Args:
            event: The event name to unsubscribe from.
            listener: Specific listener to remove. If None, removes all listeners for the event.

        Raises:
            NotImplementedError: If the runtime does not support event subscription.
        """
        raise NotImplementedError(f'{self.__class__.__name__} does not support event subscription')

    async def emit(
        self,
        event: str,
        *args: Any,
        ack_callback: Callable[..., None] | Callable[..., Awaitable[None]] | None = None,
    ) -> None:
        """Emit an event to the runtime environment.

        Args:
            event: The event name to emit.
            *args: Arguments to pass with the event.
            ack_callback: Optional callback for acknowledgement from the server.

        Raises:
            NotImplementedError: If the runtime does not support event emission.
        """
        raise NotImplementedError(f'{self.__class__.__name__} does not support event emission')


__all__ = [
    'ExecutorMixin',
    'Runtime',
    'RuntimeUnhealthyError',
    'SupportsCallExecutor',
]
