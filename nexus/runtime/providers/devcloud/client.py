"""DevCloud OSWorld runtime client."""

from __future__ import annotations

import asyncio
import base64
import binascii
import shlex

from typing import Any

import aiohttp

from aiohttp import FormData

from nexus.runtime.bridge.runtime_kind import NativeBridgeRuntimeKind
from nexus.runtime.client.base import Runtime
from nexus.utils.async_http import http_client


class DevCloudRuntime(Runtime):
    """Runtime client for an OSWorld HTTP server running in a DevCloud macOS sandbox."""

    native_bridge_kind = NativeBridgeRuntimeKind.DEVCLOUD_MACOS

    def __init__(
        self, *, base_url: str, request_timeout: float = 1800.0, metadata: dict[str, Any] | None = None
    ) -> None:
        super().__init__()
        self._base_url = base_url.rstrip('/')
        self.request_timeout = float(request_timeout)
        self.metadata = metadata or {}
        self._timeout = aiohttp.ClientTimeout(total=self.request_timeout)

    @property
    def base_url(self) -> str | None:
        return self._base_url

    async def _post_json(self, path: str, payload: dict[str, Any]) -> tuple[Any, str, int]:
        resp = await http_client.post(f'{self._base_url}{path}', json=payload, timeout=self._timeout)
        try:
            body = resp.json()
        except Exception:
            body = None
        return body, resp.text, resp.status_code

    async def _run_python_controller_code(self, code: str) -> dict[str, Any]:
        payload = {
            'command': ['python3', '-c', f'import pyautogui; import time; pyautogui.FAILSAFE = False; {code}'],
            'shell': False,
        }
        body, text, status_code = await self._post_json('/execute', payload)
        if status_code >= 400 or not isinstance(body, dict):
            raise RuntimeError(f'OSWorld execute failed: HTTP {status_code} {text[:500]}')
        return body

    @staticmethod
    def _build_osworld_execute_payload(
        *,
        command: str,
        cwd: str | None,
        timeout: float,
        stdin: str | None,
        extra_env: dict[str, str] | None,
    ) -> dict[str, Any]:
        script_lines: list[str] = []
        for key, value in (extra_env or {}).items():
            script_lines.append(f'export {key}={shlex.quote(str(value))}')
        if cwd:
            script_lines.append(f'cd {shlex.quote(cwd)}')
        runner = command
        if stdin is not None:
            runner = '\n'.join(["cat <<'__NEXUS_STDIN__' | (", command, ')', stdin, '__NEXUS_STDIN__'])
        script_lines.append(runner)
        timeout_value = max(float(timeout), 0.001)
        wrapped_script = '\n'.join(script_lines)
        shell_command = f'timeout --kill-after=5s {timeout_value:g}s bash -lc {shlex.quote(wrapped_script)}'
        return {'command': ['bash', '-lc', shell_command], 'shell': False}

    async def call_executor(self, executor_name: str, arg: Any) -> Any:
        if executor_name == 'run_command':
            return await self._run_command(arg)
        if executor_name in {
            'gui_click',
            'gui_double_click',
            'gui_long_click',
            'gui_drag',
            'gui_type',
            'gui_press_key',
        }:
            return await self._run_gui_pyautogui(executor_name, arg)
        if executor_name == 'gui_screenshot':
            return await self._gui_screenshot()
        if executor_name == 'gui_get_screen_size':
            return await self._gui_get_screen_size()
        if executor_name == 'gui_launch_app':
            return await self._setup_route('/setup/launch', {'command': ['open', arg['app']], 'shell': False})
        if executor_name == 'gui_close_app':
            return await self._setup_route('/setup/close_window', {'window_name': arg['app']})
        if executor_name == 'download_file':
            return await self._download_file(arg)
        if executor_name == 'upload_file':
            return await self._upload_file(arg)
        raise NotImplementedError(f'Unsupported DevCloud executor: {executor_name!r}')

    async def _run_command(self, arg: Any) -> dict[str, Any]:
        from nexus.runtime.executors.command import CommandRunStatus, RunCommandRequest, RunCommandResponse

        request = RunCommandRequest.model_validate(arg)
        payload = self._build_osworld_execute_payload(
            command=request.command,
            cwd=request.cwd,
            timeout=request.timeout,
            stdin=request.stdin,
            extra_env=request.extra_env,
        )
        started = asyncio.get_running_loop().time()
        body, text, status_code = await self._post_json('/execute', payload)
        if status_code >= 400 and not isinstance(body, dict):
            raise RuntimeError(f'OSWorld execute failed: HTTP {status_code} {text[:500]}')
        data = body if isinstance(body, dict) else {}
        return_code = data.get('returncode')
        if return_code == 124:
            status = CommandRunStatus.TimeLimitExceeded
        elif return_code == 0:
            status = CommandRunStatus.Finished
        else:
            status = CommandRunStatus.Error
        return RunCommandResponse(
            status=status,
            execution_time=asyncio.get_running_loop().time() - started,
            return_code=return_code,
            stdout=data.get('output'),
            stderr=data.get('error'),
        ).model_dump()

    async def _gui_screenshot(self) -> dict[str, Any]:
        from nexus.runtime.executors.gui import GuiScreenshotResponse

        resp = await http_client.get(f'{self._base_url}/screenshot', timeout=self._timeout)
        if resp.status_code >= 400:
            return GuiScreenshotResponse(success=False, error=resp.text[:500]).model_dump()
        screen = await self._gui_get_screen_size()
        return GuiScreenshotResponse(
            success=True,
            image=base64.b64encode(resp.content).decode('utf-8'),
            width=screen.get('width'),
            height=screen.get('height'),
            format='png',
        ).model_dump()

    async def _gui_get_screen_size(self) -> dict[str, Any]:
        from nexus.runtime.executors.gui import GuiScreenSizeResponse

        body, text, status_code = await self._post_json('/screen_size', {})
        if status_code >= 400 or not isinstance(body, dict):
            return GuiScreenSizeResponse(success=False, error=text[:500]).model_dump()
        return GuiScreenSizeResponse(success=True, width=body.get('width'), height=body.get('height')).model_dump()

    async def _run_gui_pyautogui(self, executor_name: str, arg: Any) -> dict[str, Any]:
        from nexus.runtime.executors.gui import (
            GuiClickRequest,
            GuiClickResponse,
            GuiDoubleClickRequest,
            GuiDoubleClickResponse,
            GuiDragRequest,
            GuiDragResponse,
            GuiLongClickRequest,
            GuiLongClickResponse,
            GuiPressKeyRequest,
            GuiPressKeyResponse,
            GuiTypeRequest,
            GuiTypeResponse,
        )

        if executor_name == 'gui_click':
            request = GuiClickRequest.model_validate(arg)
            code = f'pyautogui.click(x={request.x}, y={request.y})'
            result = await self._run_python_controller_code(code)
            return GuiClickResponse(
                success=result.get('returncode', 1) == 0, error=result.get('error') or None
            ).model_dump()
        if executor_name == 'gui_double_click':
            request = GuiDoubleClickRequest.model_validate(arg)
            code = f'pyautogui.doubleClick(x={request.x}, y={request.y})'
            result = await self._run_python_controller_code(code)
            return GuiDoubleClickResponse(
                success=result.get('returncode', 1) == 0, error=result.get('error') or None
            ).model_dump()
        if executor_name == 'gui_long_click':
            request = GuiLongClickRequest.model_validate(arg)
            code = (
                f'pyautogui.mouseDown(x={request.x}, y={request.y}); '
                f'time.sleep({request.duration / 1000.0}); pyautogui.mouseUp()'
            )
            result = await self._run_python_controller_code(code)
            return GuiLongClickResponse(
                success=result.get('returncode', 1) == 0, error=result.get('error') or None
            ).model_dump()
        if executor_name == 'gui_drag':
            request = GuiDragRequest.model_validate(arg)
            duration = max(request.duration / 1000.0, 0.0)
            code = (
                f'pyautogui.moveTo({request.x1}, {request.y1}); '
                f'pyautogui.dragTo({request.x2}, {request.y2}, duration={duration}, '
                'button="left", mouseDownUp=True)'
            )
            result = await self._run_python_controller_code(code)
            return GuiDragResponse(
                success=result.get('returncode', 1) == 0, error=result.get('error') or None
            ).model_dump()
        if executor_name == 'gui_type':
            request = GuiTypeRequest.model_validate(arg)
            result = await self._run_python_controller_code(f'pyautogui.typewrite({request.text!r})')
            return GuiTypeResponse(
                success=result.get('returncode', 1) == 0, error=result.get('error') or None
            ).model_dump()
        request = GuiPressKeyRequest.model_validate(arg)
        result = await self._run_python_controller_code(f'pyautogui.press({request.key.lower()!r})')
        return GuiPressKeyResponse(
            success=result.get('returncode', 1) == 0, error=result.get('error') or None
        ).model_dump()

    async def _setup_route(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        from nexus.runtime.executors.gui import GuiCloseAppResponse, GuiLaunchAppResponse

        _, text, status_code = await self._post_json(path, payload)
        response_cls = GuiLaunchAppResponse if path.endswith('/setup/launch') else GuiCloseAppResponse
        return response_cls(success=status_code == 200, error=None if status_code == 200 else text[:500]).model_dump()

    async def _download_file(self, arg: Any) -> dict[str, Any]:
        from nexus.runtime.executors.models.file import (
            DownloadFileRequest,
            DownloadFileResponse,
            FileItem,
            FileOperationResult,
        )

        request = DownloadFileRequest.model_validate(arg)
        files: list[FileItem] = []
        errors: list[FileOperationResult] = []
        for item in request.files:
            try:
                if request.max_bytes is not None:
                    async with aiohttp.ClientSession(timeout=self._timeout) as session:
                        async with session.post(f'{self._base_url}/file', data={'file_path': item.path}) as response:
                            response.raise_for_status()
                            raw = bytearray()
                            async for chunk in response.content.iter_chunked(64 * 1024):
                                if len(raw) + len(chunk) > request.max_bytes:
                                    raise ValueError(f'File exceeds {request.max_bytes} byte limit')
                                raw.extend(chunk)
                    content = (
                        base64.b64encode(raw).decode('utf-8') if item.encoding == 'base64' else raw.decode('utf-8')
                    )
                    files.append(FileItem(path=item.path, content=content, encoding=item.encoding))
                    continue
                resp = await http_client.post(
                    f'{self._base_url}/file',
                    data={'file_path': item.path},
                    timeout=self._timeout,
                )
                if resp.status_code >= 400:
                    errors.append(FileOperationResult(path=item.path, success=False, message=resp.text[:500]))
                    continue
                if item.encoding == 'base64':
                    content = base64.b64encode(resp.content).decode('utf-8')
                else:
                    content = resp.content.decode('utf-8')
                files.append(FileItem(path=item.path, content=content, encoding=item.encoding))
            except Exception as exc:
                errors.append(FileOperationResult(path=item.path, success=False, message=f'Error: {exc}'))
        return DownloadFileResponse(files=files, errors=errors if errors else None).model_dump()

    async def _upload_file(self, arg: Any) -> dict[str, Any]:
        from nexus.runtime.executors.models.file import FileOperationResult, UploadFileRequest, UploadFileResponse

        request = UploadFileRequest.model_validate(arg)
        results: list[FileOperationResult] = []
        for item in request.files:
            if item.content is None:
                results.append(FileOperationResult(path=item.path, success=False, message='Content is required'))
                continue
            try:
                if item.encoding == 'base64':
                    file_bytes = base64.b64decode(item.content)
                else:
                    file_bytes = item.content.encode('utf-8')
                form = FormData()
                form.add_field('file_path', item.path)
                form.add_field('file_data', file_bytes, filename=item.path.rsplit('/', 1)[-1] or 'upload.bin')
                resp = await http_client.post(f'{self._base_url}/setup/upload', data=form, timeout=self._timeout)
                if resp.status_code >= 400:
                    results.append(FileOperationResult(path=item.path, success=False, message=resp.text[:500]))
                else:
                    results.append(FileOperationResult(path=item.path, success=True, message=resp.text))
            except (binascii.Error, ValueError) as exc:
                results.append(
                    FileOperationResult(path=item.path, success=False, message=f'Invalid base64 content: {exc}')
                )
            except Exception as exc:
                results.append(FileOperationResult(path=item.path, success=False, message=f'Error: {exc}'))
        return UploadFileResponse(results=results).model_dump()

    async def _invoke(self, request: Any) -> Any:
        raise NotImplementedError('DevCloudRuntime does not support invoke')

    async def call_tool(self, import_path: str, params: dict[str, Any], context: Any) -> Any:
        raise NotImplementedError('DevCloudRuntime does not support call_tool')


__all__ = ['DevCloudRuntime']
