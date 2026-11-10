# ruff: noqa: F405
"""Lightweight executor models — pure Pydantic, no heavy dependencies.

Import from here when you only need the data shapes (request/response models)
without pulling in Playwright, jupyter_client, tmux, etc.

Each executor's implementation module (e.g., ``executors/browser.py``) re-exports
its models via ``from .models.browser import *``, so existing import paths
continue to work unchanged.
"""

from nexus.runtime.executors.models.base import *  # noqa: F401,F403
from nexus.runtime.executors.models.browser import *  # noqa: F401,F403
from nexus.runtime.executors.models.command import *  # noqa: F401,F403
from nexus.runtime.executors.models.file import *  # noqa: F401,F403
from nexus.runtime.executors.models.gui import *  # noqa: F401,F403
from nexus.runtime.executors.models.jupyter import *  # noqa: F401,F403
from nexus.runtime.executors.models.terminal import *  # noqa: F401,F403


# Explicit __all__ to prevent sub-module names (browser, command, …) and
# third-party symbols (BaseModel, Field, …) from leaking into the parent
# package namespace via star-import.
__all__ = [
    # base
    'ExecutorRequest',
    'ResponseT',
    # browser
    'BidLocator',
    'ClickRequest',
    'ClickResponse',
    'CloseTabRequest',
    'CloseTabResponse',
    'CreateBrowserSessionRequest',
    'CreateBrowserSessionResponse',
    'DestroyBrowserSessionRequest',
    'DestroyBrowserSessionResponse',
    'ElementLocator',
    'EvalJsRequest',
    'EvalJsResponse',
    'FillRequest',
    'FillResponse',
    'FocusTabRequest',
    'FocusTabResponse',
    'GetContentRequest',
    'GetContentResponse',
    'GetObservationRequest',
    'GetObservationResponse',
    'GetUrlRequest',
    'GetUrlResponse',
    'GoBackRequest',
    'GoBackResponse',
    'GoForwardRequest',
    'GoForwardResponse',
    'GotoRequest',
    'GotoResponse',
    'HoverRequest',
    'HoverResponse',
    'KeyboardTypeRequest',
    'KeyboardTypeResponse',
    'ListTabsRequest',
    'ListTabsResponse',
    'MouseDragRequest',
    'MouseDragResponse',
    'NewTabRequest',
    'NewTabResponse',
    'PressKeyRequest',
    'PressKeyResponse',
    'ScrollRequest',
    'ScrollResponse',
    'SelectOptionRequest',
    'SelectOptionResponse',
    'SelectorLocator',
    'SetGeolocationRequest',
    'SetGeolocationResponse',
    'ScreenshotClip',
    'ScreenshotRequest',
    'ScreenshotResponse',
    'TabInfo',
    'WaitForLoadStateRequest',
    'WaitForLoadStateResponse',
    'WaitForSelectorRequest',
    'WaitForSelectorResponse',
    'XYLocator',
    # command
    'CommandRunStatus',
    'RunCommandRequest',
    'RunCommandResponse',
    # file
    'DownloadFileRequest',
    'DownloadFileResponse',
    'FileItem',
    'FileOperationResult',
    'UploadFileRequest',
    'UploadFileResponse',
    # gui
    'GuiClickRequest',
    'GuiClickResponse',
    'GuiCloseAppRequest',
    'GuiCloseAppResponse',
    'GuiDoubleClickRequest',
    'GuiDoubleClickResponse',
    'GuiDragRequest',
    'GuiDragResponse',
    'GuiLaunchAppRequest',
    'GuiLaunchAppResponse',
    'GuiLongClickRequest',
    'GuiLongClickResponse',
    'GuiPressKeyRequest',
    'GuiPressKeyResponse',
    'GuiScreenSizeRequest',
    'GuiScreenSizeResponse',
    'GuiScreenshotRequest',
    'GuiScreenshotResponse',
    'GuiStartRequest',
    'GuiStartResponse',
    'GuiTypeRequest',
    'GuiTypeResponse',
    # jupyter
    'CreateIPythonSessionRequest',
    'CreateIPythonSessionResponse',
    'DestroyIPythonSessionRequest',
    'DestroyIPythonSessionResponse',
    'ExecuteIPythonRequest',
    'ExecuteIPythonResponse',
    # terminal
    'CapturePaneRequest',
    'CapturePaneResponse',
    'CommandInfo',
    'CreateTerminalSessionRequest',
    'CreateTerminalSessionResponse',
    'DestroyTerminalSessionRequest',
    'DestroyTerminalSessionResponse',
    'QueryCommandStatusRequest',
    'RunCommandInSessionRequest',
    'SendKeysToSessionRequest',
    'SendKeysToSessionResponse',
    'StartCommandInSessionRequest',
    'TerminateSessionProcessesRequest',
    'TerminateSessionProcessesResponse',
]
