"""AGS WindowsAgentArena runtime implementations."""

from __future__ import annotations

import asyncio
import base64
import logging

from typing import Any

import aiohttp

from nexus.runtime.bridge.runtime_kind import NativeBridgeRuntimeKind
from nexus.runtime.providers.ags.base import AGSRawRuntime, _BaseAGSRuntime
from nexus.runtime.providers.ags.handle import AGSInstanceHandle
from nexus.utils.async_http import http_client


logger = logging.getLogger(__name__)


_SETUP_OPEN_FILE_TRANSIENT_MARKERS = (
    'open_file_timeout',
    'GUI open helper failed',
    'UPSTREAM_DIAL_ERROR',
    'timeout awaiting response headers',
)
_SETUP_OPEN_FILE_TRANSIENT_STATUSES = {502, 503, 504}
# WAA /setup/open_file 经 AGS 网关时会遇到短暂 502/504 或 GUI helper 冷启动超时；
# 这些复跑通常可恢复，但 HTTP 404 File not found 代表 agent 产物缺失，不能重试成基础设施成功。
_SETUP_OPEN_FILE_RETRY_DELAYS = (2.0,) * 30


def _validate_waa_png_response(response: Any, *, endpoint: str) -> bytes:
    content_type = response.headers.get('content-type', '')
    content = response.content
    if response.status_code >= 400:
        raise RuntimeError(
            f'WAA {endpoint} failed: HTTP {response.status_code} '
            f'content-type={content_type!r} body={response.text[:500]!r}'
        )
    if not content.startswith(b'\x89PNG\r\n\x1a\n'):
        raise RuntimeError(
            f'WAA {endpoint} returned invalid PNG bytes: HTTP {response.status_code} '
            f'content-type={content_type!r} len={len(content)} prefix={content[:32]!r}'
        )
    return content


def _is_setup_open_file_transient(response: Any) -> bool:
    if response.status_code not in _SETUP_OPEN_FILE_TRANSIENT_STATUSES:
        return False
    return any(marker in response.text for marker in _SETUP_OPEN_FILE_TRANSIENT_MARKERS)


class AGSWindowsAgentArenaRuntime(_BaseAGSRuntime):
    """Nexus-style runtime backed by a WindowsAgentArena-compatible server in AGS.

    Implements the standard executor dispatch interface (gui_click, gui_screenshot,
    run_command, etc.) by mapping to WAA's HTTP API. This enables agents built on
    the ExecutorMixin interface to interact with WAA VMs through the same API they'd
    use for OSWorld, Android, etc.
    """

    native_bridge_kind = NativeBridgeRuntimeKind.AGS_WAA

    def __init__(
        self,
        handle: AGSInstanceHandle,
        *,
        server_port: int = 5000,
        chromium_port: int = 9222,
        vnc_port: int = 8006,
        request_timeout: float = 1800.0,
        resource_spec: dict[str, int] | None = None,
    ) -> None:
        super().__init__(handle, request_timeout=request_timeout, resource_spec=resource_spec)
        self.server_port = server_port
        self.chromium_port = chromium_port
        self.vnc_port = vnc_port

        self.port_mappings[server_port] = self.handle.url(server_port)
        self.port_mappings[chromium_port] = self.handle.url(chromium_port)
        self.port_mappings[vnc_port] = self.handle.url(vnc_port)

    @property
    def base_url(self) -> str | None:
        return self.handle.url(self.server_port)

    @property
    def vnc_url(self) -> str:
        """VNC web client URL."""
        return self.handle.url(self.vnc_port)

    async def _post_waa_json(
        self,
        path: str,
        payload: dict[str, Any],
        *,
        timeout: float | None = None,
    ) -> tuple[Any, str, int]:
        """POST JSON to the WAA server, return (parsed_body, raw_text, status_code)."""
        request_timeout = aiohttp.ClientTimeout(total=max(float(timeout), 0.001)) if timeout is not None else None
        kwargs: dict[str, Any] = {}
        if request_timeout is not None:
            kwargs['timeout'] = request_timeout
        response = await self._http_request('POST', self.handle.url(self.server_port, path), json=payload, **kwargs)
        return response.json(), response.text, response.status_code

    async def _run_python_on_vm(self, code: str) -> dict[str, Any]:
        """Execute Python code on the WAA VM via the /execute endpoint.

        Wraps the code with pyautogui imports (FAILSAFE disabled) for GUI operations.
        """
        payload = {
            'command': [
                'python',
                '-c',
                f'import pyautogui; import time; pyautogui.FAILSAFE = False; {code}',
            ],
            'shell': False,
        }
        body, text, status_code = await self._post_waa_json('/execute', payload)
        if status_code >= 400:
            raise RuntimeError(f'WAA execute failed: HTTP {status_code} {text[:500]}')
        if not isinstance(body, dict):
            raise RuntimeError(f'WAA execute returned unexpected payload: {text[:500]}')
        return body

    @staticmethod
    def _build_waa_execute_payload(
        *,
        command: str,
        cwd: str | None,
        extra_env: dict[str, str] | None,
    ) -> dict[str, Any]:
        """Build the shell command payload for WAA's /execute endpoint (Windows)."""
        parts: list[str] = []

        # Set environment variables
        for key, value in (extra_env or {}).items():
            parts.append(f'set "{key}={value}"')

        # Change directory
        if cwd:
            parts.append(f'cd /d "{cwd}"')

        # Append the actual command
        parts.append(command)

        # Join with && for sequential execution
        full_command = ' && '.join(parts) if len(parts) > 1 else command
        return {'command': full_command, 'shell': True}

    async def call_executor(self, executor_name: str, arg: Any) -> Any:  # noqa: C901, PLR0911, PLR0912
        if executor_name == 'run_command':
            from nexus.runtime.executors.command import CommandRunStatus, RunCommandRequest, RunCommandResponse

            request = RunCommandRequest.model_validate(arg)
            payload = self._build_waa_execute_payload(
                command=request.command,
                cwd=request.cwd,
                extra_env=request.extra_env,
            )

            started = asyncio.get_running_loop().time()
            body, text, status_code = await self._post_waa_json('/execute', payload, timeout=request.timeout)
            if status_code >= 400 and not isinstance(body, dict):
                raise RuntimeError(f'WAA execute failed: HTTP {status_code} {text[:500]}')

            data = body if isinstance(body, dict) else {}
            return_code = data.get('returncode')
            status = CommandRunStatus.Finished if return_code == 0 else CommandRunStatus.Error
            response = RunCommandResponse(
                status=status,
                execution_time=asyncio.get_running_loop().time() - started,
                return_code=return_code,
                stdout=data.get('output'),
                stderr=data.get('error'),
            )
            return response.model_dump()

        if executor_name == 'gui_click':
            from nexus.runtime.executors.gui import GuiClickRequest, GuiClickResponse

            request = GuiClickRequest.model_validate(arg)
            result = await self._run_python_on_vm(f'pyautogui.click(x={request.x}, y={request.y})')
            return GuiClickResponse(success=result.get('returncode', 1) == 0).model_dump()

        if executor_name == 'gui_double_click':
            from nexus.runtime.executors.gui import GuiDoubleClickRequest, GuiDoubleClickResponse

            request = GuiDoubleClickRequest.model_validate(arg)
            result = await self._run_python_on_vm(f'pyautogui.doubleClick(x={request.x}, y={request.y})')
            return GuiDoubleClickResponse(success=result.get('returncode', 1) == 0).model_dump()

        if executor_name == 'gui_long_click':
            from nexus.runtime.executors.gui import GuiLongClickRequest, GuiLongClickResponse

            request = GuiLongClickRequest.model_validate(arg)
            code = (
                f'pyautogui.mouseDown(x={request.x}, y={request.y}); '
                f'time.sleep({request.duration / 1000.0}); '
                'pyautogui.mouseUp()'
            )
            result = await self._run_python_on_vm(code)
            return GuiLongClickResponse(success=result.get('returncode', 1) == 0).model_dump()

        if executor_name == 'gui_drag':
            from nexus.runtime.executors.gui import GuiDragRequest, GuiDragResponse

            request = GuiDragRequest.model_validate(arg)
            duration = max(request.duration / 1000.0, 0.0)
            code = (
                f'pyautogui.moveTo({request.x1}, {request.y1}); '
                f'pyautogui.dragTo({request.x2}, {request.y2}, duration={duration}, button="left", mouseDownUp=True)'
            )
            result = await self._run_python_on_vm(code)
            return GuiDragResponse(success=result.get('returncode', 1) == 0).model_dump()

        if executor_name == 'gui_type':
            from nexus.runtime.executors.gui import GuiTypeRequest, GuiTypeResponse

            request = GuiTypeRequest.model_validate(arg)
            result = await self._run_python_on_vm(f'pyautogui.typewrite({request.text!r})')
            return GuiTypeResponse(success=result.get('returncode', 1) == 0).model_dump()

        if executor_name == 'gui_press_key':
            from nexus.runtime.executors.gui import GuiPressKeyRequest, GuiPressKeyResponse

            request = GuiPressKeyRequest.model_validate(arg)
            result = await self._run_python_on_vm(f'pyautogui.press({request.key.lower()!r})')
            return GuiPressKeyResponse(success=result.get('returncode', 1) == 0).model_dump()

        if executor_name == 'gui_screenshot':
            from nexus.runtime.executors.gui import GuiScreenshotResponse

            response = await http_client.get(
                self.handle.url(self.server_port, '/screenshot'),
                timeout=self._timeout,
                headers=self.handle.headers,
            )
            try:
                image = _validate_waa_png_response(response, endpoint='/screenshot')
            except RuntimeError as exc:
                return GuiScreenshotResponse(success=False, error=str(exc)).model_dump()

            screen_size = await self.call_executor('gui_get_screen_size', {})
            return GuiScreenshotResponse(
                success=True,
                image=base64.b64encode(image).decode('utf-8'),
                width=screen_size.get('width'),
                height=screen_size.get('height'),
                format='png',
            ).model_dump()

        if executor_name == 'gui_get_screen_size':
            from nexus.runtime.executors.gui import GuiScreenSizeResponse

            body, text, status_code = await self._post_waa_json('/screen_size', {})
            if status_code >= 400 or not isinstance(body, dict):
                return GuiScreenSizeResponse(success=False, error=text[:500]).model_dump()
            return GuiScreenSizeResponse(success=True, width=body.get('width'), height=body.get('height')).model_dump()

        if executor_name == 'gui_launch_app':
            from nexus.runtime.executors.gui import GuiLaunchAppRequest, GuiLaunchAppResponse

            request = GuiLaunchAppRequest.model_validate(arg)
            _, text, status_code = await self._post_waa_json('/setup/open_file', {'path': request.app})
            return GuiLaunchAppResponse(
                success=status_code == 200, error=None if status_code == 200 else text[:500]
            ).model_dump()

        if executor_name == 'gui_close_app':
            from nexus.runtime.executors.gui import GuiCloseAppRequest, GuiCloseAppResponse

            request = GuiCloseAppRequest.model_validate(arg)
            _, text, status_code = await self._post_waa_json('/setup/close_window', {'window_name': request.app})
            return GuiCloseAppResponse(
                success=status_code == 200, error=None if status_code == 200 else text[:500]
            ).model_dump()

        if executor_name == 'gui_get_device_info':
            from nexus.runtime.executors.gui import GuiGetDeviceInfoResponse

            body, text, status_code = await self._post_waa_json('/screen_size', {})
            if status_code >= 400 or not isinstance(body, dict):
                return GuiGetDeviceInfoResponse(success=False, error=text[:500]).model_dump()
            return GuiGetDeviceInfoResponse(
                success=True,
                device_name='WindowsAgentArena',
                window_width=body.get('width'),
                window_height=body.get('height'),
            ).model_dump()

        if executor_name == 'gui_reset_apps':
            from nexus.runtime.executors.gui import GuiResetAppsResponse

            _, text, status_code = await self._post_waa_json('/setup/close_all', {})
            return GuiResetAppsResponse(
                success=status_code == 200,
                error=None if status_code == 200 else text[:500],
            ).model_dump()

        raise NotImplementedError(f'Unsupported AGS WAA executor: {executor_name!r}')

    def to_raw(self) -> AGSRawWindowsAgentArenaRuntime:
        """Convert to AGSRawWindowsAgentArenaRuntime (raw HTTP surface)."""
        return AGSRawWindowsAgentArenaRuntime(
            self.handle,
            server_port=self.server_port,
            chromium_port=self.chromium_port,
            vnc_port=self.vnc_port,
            request_timeout=self.request_timeout,
            resource_spec=self.resource_spec,
        )


class AGSRawWindowsAgentArenaRuntime(AGSRawRuntime):
    """Raw WindowsAgentArena-compatible runtime that preserves the original HTTP surface."""

    native_bridge_kind = NativeBridgeRuntimeKind.AGS_WAA

    def __init__(
        self,
        handle: AGSInstanceHandle,
        *,
        server_port: int = 5000,
        chromium_port: int = 9222,
        vnc_port: int = 8006,
        request_timeout: float = 1800.0,
        resource_spec: dict[str, int] | None = None,
    ) -> None:
        super().__init__(
            handle,
            kind='waa',
            default_port=server_port,
            request_timeout=request_timeout,
            resource_spec=resource_spec,
        )
        self.server_port = server_port
        self.chromium_port = chromium_port
        self.vnc_port = vnc_port

        self.port_mappings[server_port] = self.handle.url(server_port)
        self.port_mappings[chromium_port] = self.handle.url(chromium_port)
        self.port_mappings[vnc_port] = self.handle.url(vnc_port)
        logger.info(
            f'AGSRawWindowsAgentArenaRuntime initialized with vnc_url: {self.vnc_url} token: {self.handle.access_token}'
        )

    @property
    def server_url(self) -> str:
        return self.handle.url(self.server_port)

    @property
    def chromium_url(self) -> str:
        """Base URL for Chrome DevTools Protocol HTTP endpoints."""
        return self.handle.url(self.chromium_port)

    @property
    def chromium_cdp_url(self) -> str:
        """CDP version endpoint URL (e.g. http://host:9222/json/version)."""
        return self.handle.url(self.chromium_port, '/json/version')

    @property
    def vnc_url(self) -> str:
        """VNC web client URL."""
        return self.handle.url(self.vnc_port)

    async def probe(self) -> Any:
        return await self.request_json('GET', '/probe', port=self.server_port)

    async def platform(self) -> str:
        response = await self.request('GET', '/platform', port=self.server_port)
        return response.text

    async def update_computer(
        self,
        *,
        screenshot: str,
        rects: list[Any] | None = None,
        window_rect: list[Any] | None = None,
        scale: tuple[float, float] | None = None,
        clipboard_content: str | None = None,
        swap_ctrl_alt: bool = False,
    ) -> Any:
        payload: dict[str, Any] = {
            'screenshot': screenshot,
            'swap_ctrl_alt': swap_ctrl_alt,
        }
        if rects is not None:
            payload['rects'] = rects
        if window_rect is not None:
            payload['window_rect'] = window_rect
        if scale is not None:
            payload['scale'] = scale
        if clipboard_content is not None:
            payload['clipboard_content'] = clipboard_content
        return await self.request_json('POST', '/update_computer', port=self.server_port, json=payload)

    async def execute_raw(self, payload: dict[str, Any]) -> Any:
        return await self.request_json('POST', '/execute', port=self.server_port, json=payload)

    async def setup_execute_raw(self, payload: dict[str, Any]) -> Any:
        return await self.request_json('POST', '/setup/execute', port=self.server_port, json=payload)

    async def execute(self, command: str | list[str], shell: bool = False) -> Any:
        return await self.execute_raw({'command': command, 'shell': shell})

    async def setup_execute(self, command: str | list[str], shell: bool = False) -> Any:
        return await self.setup_execute_raw({'command': command, 'shell': shell})

    async def execute_windows(self, command: str) -> Any:
        return await self.request_json('POST', '/execute_windows', port=self.server_port, json={'command': command})

    async def setup_launch(self, command: str | list[str], shell: bool = True) -> str:
        response = await self.request(
            'POST', '/setup/launch', port=self.server_port, json={'command': command, 'shell': shell}
        )
        if response.status_code >= 400:
            raise RuntimeError(f'WAA setup launch failed: HTTP {response.status_code} {response.text[:500]}')
        return response.text

    async def shutdown(self) -> Any:
        return await self.request_json('POST', '/shutdown', port=self.server_port)

    async def screenshot(self) -> bytes:
        # LibreOffice/浏览器恢复窗口期间，/screenshot 偶发返回 UPSTREAM_DIAL_ERROR 或短暂连接失败；
        # 全量测试中这类错误低并发复跑可过，因此只对代理/连接瞬时错误做有限重试。
        last_error: Exception | None = None
        max_attempts = 6
        for attempt in range(1, max_attempts + 1):
            try:
                response = await self.request('GET', '/screenshot', port=self.server_port)
                try:
                    return _validate_waa_png_response(response, endpoint='/screenshot')
                except RuntimeError as exc:
                    if not self._is_transient_proxy_response(response) or attempt >= max_attempts:
                        raise
                    last_error = exc
                    logger.warning(
                        'Transient WAA screenshot failure on attempt %s/%s: HTTP %s %s',
                        attempt,
                        max_attempts,
                        response.status_code,
                        response.text[:200],
                    )
            except Exception as exc:
                if not self._is_transient_runtime_exception(exc) or attempt >= max_attempts:
                    raise
                last_error = exc
                logger.warning('Transient WAA screenshot failure on attempt %s/%s: %s', attempt, max_attempts, exc)
            await asyncio.sleep(min(5.0, float(attempt)))

        if last_error is not None:
            raise last_error
        raise RuntimeError('WAA /screenshot failed after transient retries')

    async def obs_winagent(self) -> Any:
        return await self.request_json('GET', '/obs_winagent', port=self.server_port)

    async def screen_size(self) -> Any:
        return await self.request_json('POST', '/screen_size', port=self.server_port, json={})

    async def window_size(self, app_class_name: str) -> Any:
        return await self.request_json(
            'POST',
            '/window_size',
            port=self.server_port,
            data={'app_class_name': app_class_name},
        )

    async def accessibility_tree(self, backend: str = 'win32') -> Any:
        params: dict[str, Any] = {'backend': backend}
        if backend != 'uia':
            return await self.request_json('GET', '/accessibility', port=self.server_port, params=params)

        # Bound UIA work below the front proxy response-header timeout, then
        # retry transient upstream failures instead of returning a missing tree.
        # UIA 树生成可能接近 AGS 网关响应头超时；服务端硬超时低于网关超时，客户端再重试短暂失败。
        params.update({'hard_timeout': '28', 'attempts': '1'})
        last_response: Any = None
        for attempt in range(1, 5):
            response = await self.request_json('GET', '/accessibility', port=self.server_port, params=params)
            if not self._is_transient_uia_accessibility_failure(response):
                return response
            last_response = response
            logger.warning('Transient UIA accessibility failure on attempt %s/4: %s', attempt, response)
            await asyncio.sleep(min(4.0, 1.0 + attempt))
        return last_response

    @staticmethod
    def _is_transient_uia_accessibility_failure(response: Any) -> bool:
        if not isinstance(response, dict):
            return False
        if response.get('AT'):
            return False
        code = response.get('code')
        message = str(response.get('message') or '')
        error = response.get('error')
        return (
            code == 'UPSTREAM_DIAL_ERROR'
            or 'timeout awaiting response headers' in message
            or error in {'uia_timeout', 'uia_error'}
        )

    @staticmethod
    def _is_transient_proxy_response(response: Any) -> bool:
        if getattr(response, 'status_code', 0) not in {502, 503, 504}:
            return False
        text = getattr(response, 'text', '') or ''
        return 'UPSTREAM_DIAL_ERROR' in text or 'timeout awaiting response headers' in text

    @staticmethod
    def _is_transient_runtime_exception(exc: Exception) -> bool:
        text = str(exc)
        return (
            'UPSTREAM_DIAL_ERROR' in text
            or 'timeout awaiting response headers' in text
            or 'connection refused' in text.lower()
            or 'cannot connect' in text.lower()
        )

    async def terminal_output(self) -> Any:
        return await self.request_json('GET', '/terminal', port=self.server_port)

    async def cursor_position(self) -> str:
        response = await self.request('GET', '/cursor_position', port=self.server_port)
        return response.text

    async def desktop_path(self) -> Any:
        return await self.request_json('POST', '/desktop_path', port=self.server_port, json={})

    async def documents_path(self) -> Any:
        return await self.request_json('POST', '/documents_path', port=self.server_port, json={})

    async def setup_create_folder(self, path: str) -> Any:
        return await self.request_json('POST', '/setup/create_folder', port=self.server_port, json={'path': path})

    async def setup_create_file(self, path: str, content: str = '') -> Any:
        return await self.request_json(
            'POST',
            '/setup/create_file',
            port=self.server_port,
            json={'path': path, 'content': content},
        )

    async def setup_recycle(self, path: str) -> Any:
        return await self.request_json('POST', '/setup/recycle', port=self.server_port, json={'path': path})

    async def folder_exists(self, folder_name: str, path: str) -> Any:
        return await self.request_json(
            'POST',
            '/folder_exists',
            port=self.server_port,
            json={'folder_name': folder_name, 'path': path},
        )

    async def file_exists(self, file_name: str, path: str) -> Any:
        return await self.request_json(
            'POST',
            '/file_exists',
            port=self.server_port,
            json={'file_name': file_name, 'path': path},
        )

    async def is_details_view(self, path: str) -> Any:
        return await self.request_json('POST', '/is_details_view', port=self.server_port, json={'path': path})

    async def wallpaper(self) -> bytes:
        return await self.request_bytes('POST', '/wallpaper', port=self.server_port, json={})

    async def list_directory(self, path: str) -> Any:
        return await self.request_json('POST', '/list_directory', port=self.server_port, json={'path': path})

    async def get_file(self, file_path: str) -> bytes:
        return await self.request_bytes('POST', '/file', port=self.server_port, data={'file_path': file_path})

    async def setup_upload(self, file_path: str, file_data: bytes) -> str:
        response = await self.request(
            'POST',
            '/setup/upload',
            port=self.server_port,
            data={'file_path': file_path, 'file_data': file_data},
        )
        return response.text

    async def setup_change_wallpaper(self, path: str) -> str:
        response = await self.request(
            'POST',
            '/setup/change_wallpaper',
            port=self.server_port,
            json={'path': path},
        )
        return response.text

    async def setup_download_file(self, url: str, path: str) -> str:
        response = await self.request(
            'POST',
            '/setup/download_file',
            port=self.server_port,
            json={'url': url, 'path': path},
        )
        return response.text

    async def setup_open_file(self, path: str) -> str:
        # setup 阶段打开文件是任务前置条件。对 GUI helper/网关瞬时失败重试，避免全量批次因
        # 可恢复冷启动失败直接 runner failed；但非瞬时错误仍保留为失败，交给 evaluator 或日志定位。
        max_attempts = len(_SETUP_OPEN_FILE_RETRY_DELAYS) + 1
        for attempt in range(max_attempts):
            response = await self.request('POST', '/setup/open_file', port=self.server_port, json={'path': path})
            if response.status_code < 400:
                return response.text

            retryable = _is_setup_open_file_transient(response) and attempt < len(_SETUP_OPEN_FILE_RETRY_DELAYS)
            if not retryable:
                raise RuntimeError(f'WAA setup open_file failed: HTTP {response.status_code} {response.text[:500]}')

            delay = _SETUP_OPEN_FILE_RETRY_DELAYS[attempt]
            logger.warning(
                'WAA setup open_file transient failure; retrying: attempt=%s/%s path=%s http_status=%s body=%r',
                attempt + 1,
                max_attempts,
                path,
                response.status_code,
                response.text[:200],
            )
            try:
                await self.setup_close_all()
            except Exception as exc:
                logger.warning('WAA setup open_file cleanup before retry failed: path=%s error=%r', path, exc)
            if delay > 0:
                await asyncio.sleep(delay)

        raise AssertionError('unreachable')

    async def setup_activate_window(
        self,
        window_name: str,
        *,
        strict: bool = False,
        by_class: bool = False,
    ) -> str:
        response = await self.request(
            'POST',
            '/setup/activate_window',
            port=self.server_port,
            json={'window_name': window_name, 'strict': strict, 'by_class': by_class},
        )
        return response.text

    async def setup_clear_task_files(self) -> str:
        response = await self.request('POST', '/setup/clear_task_files', port=self.server_port, json={})
        return response.text

    async def setup_close_all(self) -> str:
        response = await self.request('POST', '/setup/close_all', port=self.server_port, json={})
        return response.text

    async def setup_close_window(
        self,
        window_name: str,
        *,
        strict: bool = False,
        by_class: bool = False,
    ) -> str:
        response = await self.request(
            'POST',
            '/setup/close_window',
            port=self.server_port,
            json={'window_name': window_name, 'strict': strict, 'by_class': by_class},
        )
        return response.text

    async def start_recording(self) -> Any:
        return await self.request_json('POST', '/start_recording', port=self.server_port, json={})

    async def end_recording(self) -> Any:
        return await self.request_json('POST', '/end_recording', port=self.server_port, json={})

    async def save_state(self, snapshot_name: str) -> Any:
        return await self.request_json(
            'POST',
            '/save_state',
            port=self.server_port,
            data={'snapshot_name': snapshot_name, 'file_path': snapshot_name},
        )

    async def revert_to_snapshot(self) -> Any:
        return await self.request_json('POST', '/revert_to_snapshot', port=self.server_port, json={})

    async def are_files_sorted_by_modified_time(self, directory: str) -> Any:
        return await self.request_json(
            'POST',
            '/are_files_sorted_by_modified_time',
            port=self.server_port,
            json={'directory': directory},
        )

    async def is_directory_read_only_for_user(self, directory: str, user: str) -> Any:
        return await self.request_json(
            'POST',
            '/is_directory_read_only_for_user',
            port=self.server_port,
            json={'directory': directory, 'user': user},
        )

    async def are_all_images_tagged(self, directory: str, tag: str) -> Any:
        return await self.request_json(
            'POST',
            '/are_all_images_tagged',
            port=self.server_port,
            json={'directory': directory, 'tag': tag},
        )

    async def library_folders(self, library_name: str) -> Any:
        return await self.request_json(
            'POST',
            '/library_folders',
            port=self.server_port,
            json={'library_name': library_name},
        )

    async def check_if_timer_started(self, hours: int | float, minutes: int | float, seconds: int | float) -> Any:
        return await self.request_json(
            'POST',
            '/check_if_timer_started',
            port=self.server_port,
            json={'hours': hours, 'minutes': minutes, 'seconds': seconds},
        )

    async def check_if_world_clock_exists(self, city: str, country: str) -> Any:
        return await self.request_json(
            'POST',
            '/check_if_world_clock_exists',
            port=self.server_port,
            json={'city': city, 'country': country},
        )

    def to_nexus(self) -> AGSWindowsAgentArenaRuntime:
        """Convert to AGSWindowsAgentArenaRuntime (executor-based interface)."""
        return AGSWindowsAgentArenaRuntime(
            self.handle,
            server_port=self.server_port,
            chromium_port=self.chromium_port,
            vnc_port=self.vnc_port,
            request_timeout=self.request_timeout,
            resource_spec=self.resource_spec,
        )


def require_raw_waa_runtime(runtime: Any) -> AGSRawWindowsAgentArenaRuntime:
    """Ensure we have an AGSRawWindowsAgentArenaRuntime, converting if needed.

    Accepts either AGSRawWindowsAgentArenaRuntime or AGSWindowsAgentArenaRuntime.
    """
    if isinstance(runtime, AGSRawWindowsAgentArenaRuntime):
        return runtime
    if isinstance(runtime, AGSWindowsAgentArenaRuntime):
        return runtime.to_raw()
    raise TypeError(
        f'Expected AGSRawWindowsAgentArenaRuntime or AGSWindowsAgentArenaRuntime, got {type(runtime).__name__}'
    )


WAAExecutorRuntime = AGSWindowsAgentArenaRuntime | AGSRawWindowsAgentArenaRuntime


def require_waa_runtime(runtime: Any) -> AGSWindowsAgentArenaRuntime:
    """Ensure we have an AGSWindowsAgentArenaRuntime, converting raw WAA runtimes if needed."""
    if isinstance(runtime, AGSWindowsAgentArenaRuntime):
        return runtime
    if isinstance(runtime, AGSRawWindowsAgentArenaRuntime):
        return runtime.to_nexus()
    raise TypeError(
        f'Expected AGSWindowsAgentArenaRuntime or AGSRawWindowsAgentArenaRuntime, got {type(runtime).__name__}'
    )
