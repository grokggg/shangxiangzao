"""AGR Mobile sandbox runtime using Appium WebDriver.

Connects to an existing E2B sandbox and controls the Android device inside
it via Appium WebDriver on port 4723.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import re
import time

from typing import Any

from nexus.runtime.client.base import Runtime


logger = logging.getLogger(__name__)


def _check_appium_dependency():
    """Check if Appium-Python-Client is installed and return the webdriver module."""
    try:
        import appium.webdriver as appium_webdriver

        return appium_webdriver
    except ImportError as e:
        raise ImportError(
            'Appium-Python-Client is not installed. Install it manually: pip install Appium-Python-Client>=3.0.0'
        ) from e


def _create_appium_connection_class(access_token: str):
    """Create an AppiumConnection subclass with sandbox-local auth headers."""
    from appium.webdriver.appium_connection import AppiumConnection

    class IsolatedConnection(AppiumConnection):
        extra_headers = {'X-Access-Token': access_token}

    return IsolatedConnection


def _check_e2b_dependency():
    """Check if e2b is installed and return the Sandbox class."""
    try:
        from e2b import Sandbox

        return Sandbox
    except ImportError as e:
        raise ImportError('e2b is not installed. Install it manually: pip install e2b>=0.17.0') from e


class AGRMobileRuntime(Runtime):
    """AGR Mobile sandbox runtime using Appium WebDriver.

    Connects to an existing E2B sandbox and controls the Android device inside
    it via Appium WebDriver on port 4723. Screenshot bytes are returned
    directly from Appium with no external object storage hop.

    Example:
        runtime = AGRMobileRuntime(
            sandbox_id='abc123',
            api_key=***REDACTED***',
            domain='ap-guangzhou.tencentags.com',
        )
        await runtime.connect()
        await runtime.gui_click(500, 800)
        screenshot = await runtime.gui_screenshot()
        await runtime.disconnect()
    """

    APPIUM_PORT = 4723
    APPIUM_READY_TIMEOUT = 120  # seconds
    APPIUM_POLL_INTERVAL = 2  # seconds
    APK_CHUNK_SIZE = 20 * 1024 * 1024

    KEY_CODE_MAP = {
        'Back': 4,
        'Home': 3,
        'Menu': 82,
        'Enter': 66,
        'Delete': 67,
        'Tab': 61,
        'Escape': 111,
    }

    RESET_APP_EXCLUDE_PACKAGES = {
        'android',
        'com.android.launcher3',
        'com.android.systemui',
        'io.appium.settings',
        'io.appium.uiautomator2.server',
        'io.appium.uiautomator2.server.test',
    }

    def __init__(
        self,
        sandbox_id: str,
        *,
        api_key: str | None = None,
        domain: str | None = None,
        appium_port: int = APPIUM_PORT,
        ready_timeout: int = APPIUM_READY_TIMEOUT,
    ) -> None:
        super().__init__()
        self.sandbox_id = sandbox_id
        self.api_key = api_key
        self.domain = domain
        self.appium_port = appium_port
        self.ready_timeout = ready_timeout

        self._sandbox: Any = None
        self._driver: Any = None  # appium.webdriver.Remote
        self._connected = False
        self._screen_size: tuple[int, int] | None = None

    def _get_appium_url(self) -> str:
        """Build Appium server URL from connected sandbox metadata."""
        if self._sandbox is not None:
            return f'https://{self._sandbox.get_host(self.appium_port)}'
        if self.domain and self.sandbox_id:
            host = f'{self.appium_port}-{self.sandbox_id}.{self.domain}'
            return f'https://{host}'
        return f'http://localhost:{self.appium_port}'

    def _get_healthcheck_url(self) -> str | None:
        if self._sandbox is None:
            return None
        return f'https://{self._sandbox.get_host(8080)}/healthz'

    def _get_access_token(self) -> str | None:
        return getattr(self._sandbox, '_envd_access_token', None)

    async def _wait_for_appium_ready(self, appium_url: str) -> None:
        """Poll sandbox health and Appium status until ready or timeout."""
        import aiohttp

        health_url = self._get_healthcheck_url()
        status_url = f'{appium_url}/status'
        timeout = aiohttp.ClientTimeout(total=5)
        deadline = asyncio.get_running_loop().time() + self.ready_timeout
        last_error = ''
        access_token = self._get_access_token()
        headers = {'X-Access-Token': access_token} if access_token else None

        logger.info(f'AGRMobileRuntime: waiting for Appium at {status_url}')
        while True:
            try:
                async with aiohttp.ClientSession() as session:
                    if health_url:
                        async with session.get(health_url, timeout=timeout, headers=headers) as health_resp:
                            if health_resp.status != 200:
                                last_error = f'healthz HTTP {health_resp.status}'
                                raise RuntimeError(last_error)

                    async with session.get(status_url, timeout=timeout, headers=headers) as resp:
                        if resp.status == 200:
                            logger.info('AGRMobileRuntime: Appium is ready')
                            return
                        last_error = f'status HTTP {resp.status}'
            except Exception as exc:
                last_error = f'{type(exc).__name__}: {exc}'

            if asyncio.get_running_loop().time() >= deadline:
                raise TimeoutError(
                    f'AGRMobileRuntime: Appium not ready at {status_url} after {self.ready_timeout}s: {last_error}'
                )
            await asyncio.sleep(self.APPIUM_POLL_INTERVAL)

    async def connect(self) -> bool:
        """Connect to an existing E2B sandbox and initialise Appium WebDriver."""
        sandbox_cls = _check_e2b_dependency()
        appium_webdriver = _check_appium_dependency()

        connect_kwargs: dict[str, Any] = {}
        if self.api_key:
            connect_kwargs['api_key'] = self.api_key
        if self.domain:
            connect_kwargs['domain'] = self.domain

        logger.info(f'AGRMobileRuntime: connecting to sandbox {self.sandbox_id}')
        self._sandbox = await asyncio.to_thread(sandbox_cls.connect, self.sandbox_id, **connect_kwargs)

        appium_url = self._get_appium_url()
        await self._wait_for_appium_ready(appium_url)

        from appium.options.android import UiAutomator2Options
        from appium.webdriver.client_config import AppiumClientConfig

        options = UiAutomator2Options()
        options.platform_name = 'Android'
        options.automation_name = 'UiAutomator2'
        options.new_command_timeout = 0

        client_config = AppiumClientConfig(
            remote_server_addr=appium_url,
            timeout=300,
        )

        access_token = self._get_access_token()
        executor = None
        if access_token:
            connection_class = _create_appium_connection_class(access_token)
            executor = connection_class(client_config=client_config)

        self._driver = await asyncio.to_thread(
            appium_webdriver.Remote,
            command_executor=executor,
            options=options,
            client_config=client_config,
        )

        self._connected = True
        logger.info(f'AGRMobileRuntime: connected to sandbox {self.sandbox_id}')
        return True

    async def disconnect(self) -> bool:
        """Quit the Appium driver session."""
        if self._driver is not None:
            try:
                await asyncio.to_thread(self._driver.quit)
            except Exception as exc:
                logger.warning(f'AGRMobileRuntime: error quitting Appium driver: {exc}')
            finally:
                self._driver = None
        self._sandbox = None
        self._connected = False
        logger.info(f'AGRMobileRuntime: disconnected from sandbox {self.sandbox_id}')
        return True

    @property
    def connected(self) -> bool:
        return self._connected

    def _assert_connected(self) -> None:
        if not self._connected or self._driver is None:
            raise RuntimeError('AGRMobileRuntime not connected. Call connect() first.')

    # ------------------------------------------------------------------
    # Low-level Appium helpers (all run in thread pool to avoid blocking)
    # ------------------------------------------------------------------

    def _tap(self, x: int, y: int) -> None:
        self._driver.execute_script('mobile: clickGesture', {'x': x, 'y': y})

    def _double_tap(self, x: int, y: int) -> None:
        self._driver.execute_script('mobile: doubleClickGesture', {'x': x, 'y': y})

    def _long_press(self, x: int, y: int, duration_ms: int) -> None:
        self._driver.execute_script('mobile: longClickGesture', {'x': x, 'y': y, 'duration': duration_ms})

    def _swipe(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int) -> None:
        self._driver.swipe(x1, y1, x2, y2, duration_ms)

    def _type_text(self, text: str) -> None:
        try:
            el = self._driver.switch_to.active_element
            el.send_keys(text)
        except Exception:
            from appium.webdriver.common.appiumby import AppiumBy

            self._driver.find_element(AppiumBy.XPATH, '//*').send_keys(text)

    def _press_key(self, key: str) -> None:
        key_code = self.KEY_CODE_MAP.get(key, self.KEY_CODE_MAP.get(key.capitalize()))
        if key_code is None:
            raise ValueError(f'AGRMobileRuntime: unsupported key {key!r}. Supported: {list(self.KEY_CODE_MAP)}')
        self._driver.press_keycode(key_code)

    def _get_screenshot_png(self) -> bytes:
        return self._driver.get_screenshot_as_png()

    def _push_file(self, remote_path: str, content_b64: str) -> None:
        self._driver.push_file(remote_path, content_b64)

    def _execute_shell(
        self,
        command: str,
        args: list[str],
        *,
        timeout_ms: int | None = None,
    ) -> Any:
        payload: dict[str, Any] = {
            'command': command,
            'args': args,
        }
        if timeout_ms is not None:
            payload['timeout'] = timeout_ms
        return self._driver.execute_script('mobile: shell', payload)

    def _get_screen_size(self) -> tuple[int, int]:
        size = self._driver.get_window_size()
        return int(size['width']), int(size['height'])

    def _get_device_info(self) -> dict[str, Any]:
        capabilities = self._driver.capabilities or {}
        window_width, window_height = self._get_screen_size()

        try:
            wm_size = self._execute_shell('wm', ['size'])
        except Exception:
            wm_size = None

        try:
            wm_density = self._execute_shell('wm', ['density'])
        except Exception:
            wm_density = None

        current_package = None
        try:
            current_package = self._driver.current_package
        except Exception:
            current_package = None

        current_activity = None
        try:
            current_activity = self._driver.current_activity
        except Exception:
            current_activity = None

        return {
            'success': True,
            'device_name': capabilities.get('deviceName'),
            'platform_version': capabilities.get('platformVersion'),
            'automation_name': capabilities.get('automationName'),
            'window_width': window_width,
            'window_height': window_height,
            'wm_size': wm_size.strip() if isinstance(wm_size, str) else wm_size,
            'wm_density': wm_density.strip() if isinstance(wm_density, str) else wm_density,
            'current_package': current_package,
            'current_activity': current_activity,
        }

    def _launch_app(self, package: str) -> None:
        self._driver.activate_app(package)

    def _close_app(self, package: str) -> None:
        self._driver.terminate_app(package)

    def _query_app_state(self, package: str) -> int:
        return int(self._driver.query_app_state(package))

    def _is_app_installed(self, package: str) -> bool:
        try:
            return self._query_app_state(package) != 0
        except Exception:
            result = self._execute_shell('pm', ['list', 'packages', package])
            return package in str(result or '')

    def _upload_apk(self, local_path: str, remote_path: str, *, chunk_size: int = APK_CHUNK_SIZE) -> dict[str, Any]:
        from pathlib import Path

        apk_path = Path(local_path)
        if not apk_path.is_file():
            raise FileNotFoundError(f'APK file not found: {apk_path}')

        file_size = apk_path.stat().st_size
        total_chunks = (file_size + chunk_size - 1) // chunk_size
        temp_dir = '/data/local/tmp/chunks'

        self._execute_shell('rm', ['-rf', temp_dir])
        self._execute_shell('mkdir', ['-p', temp_dir])
        self._execute_shell('rm', ['-f', remote_path])

        with apk_path.open('rb') as f:
            for index in range(total_chunks):
                chunk_data = f.read(chunk_size)
                chunk_b64 = base64.b64encode(chunk_data).decode('utf-8')
                chunk_path = f'{temp_dir}/chunk_{index:04d}'
                self._push_file(chunk_path, chunk_b64)

        for index in range(total_chunks):
            chunk_path = f'{temp_dir}/chunk_{index:04d}'
            if index == 0:
                self._execute_shell('cp', [chunk_path, remote_path])
            else:
                self._execute_shell('cat', [chunk_path, '>>', remote_path])
            self._execute_shell('rm', ['-f', chunk_path])

        self._execute_shell('rm', ['-rf', temp_dir])
        remote_listing = self._execute_shell('ls', ['-la', remote_path])

        return {
            'success': 'No such file' not in str(remote_listing or ''),
            'local_path': str(apk_path),
            'remote_path': remote_path,
            'file_size': file_size,
            'total_chunks': total_chunks,
            'remote_listing': str(remote_listing or '').strip(),
        }

    def _install_apk(self, package: str, remote_path: str) -> dict[str, Any]:
        if self._is_app_installed(package):
            return {
                'success': True,
                'already_installed': True,
                'install_output': 'already installed',
            }

        result = self._execute_shell('pm', ['install', '-r', '-g', remote_path], timeout_ms=120000)
        install_output = str(result or '').strip()
        success = 'Success' in install_output or self._is_app_installed(package)
        return {
            'success': success,
            'already_installed': False,
            'install_output': install_output,
        }

    def _grant_permissions(self, package: str, permissions: list[str]) -> dict[str, Any]:
        granted: list[str] = []
        failed: list[str] = []
        for permission in permissions:
            try:
                self._execute_shell('pm', ['grant', package, permission])
                granted.append(permission)
            except Exception:
                failed.append(permission)

        return {
            'success': len(granted) > 0 or not permissions,
            'granted_permissions': granted,
            'failed_permissions': failed,
        }

    def _launch_app_with_fallback(self, package: str, activity: str | None = None) -> dict[str, Any]:
        self._driver.activate_app(package)
        time.sleep(3)

        state = self._query_app_state(package)
        if state >= 3:
            return {
                'success': True,
                'app_state': state,
                'used_fallback': False,
            }

        if activity:
            component = f'{package}/{activity}'
            self._execute_shell('am', ['start', '-n', component])
            time.sleep(5)
            state = self._query_app_state(package)
            return {
                'success': state >= 3,
                'app_state': state,
                'used_fallback': True,
                'component': component,
            }

        return {
            'success': False,
            'app_state': state,
            'used_fallback': False,
        }

    def _dump_logcat(self) -> str:
        result = self._execute_shell('logcat', ['-d'])
        return str(result or '')

    def _extract_started_packages(self, raw_text: str) -> list[str]:
        packages: list[str] = []
        seen: set[str] = set()
        patterns = [
            r'mActivityComponent=([A-Za-z0-9._]+)/',
            r'packageName=([A-Za-z0-9._]+)',
            r'cmp=([A-Za-z0-9._]+)/',
            r'A=\d+:([A-Za-z0-9._]+)',
        ]

        for pattern in patterns:
            for package in re.findall(pattern, raw_text):
                if package in seen or package in self.RESET_APP_EXCLUDE_PACKAGES:
                    continue
                seen.add(package)
                packages.append(package)

        return packages

    def _reset_apps(self) -> list[str]:
        activities_dump = self._execute_shell('dumpsys', ['activity', 'activities'])
        recents_dump = self._execute_shell('dumpsys', ['activity', 'recents'])

        packages = self._extract_started_packages(f'{activities_dump}\n{recents_dump}')
        stopped_packages: list[str] = []
        for package in packages:
            self._execute_shell('am', ['force-stop', package])
            stopped_packages.append(package)
        return stopped_packages

    async def upload_apk(self, local_path: str, remote_path: str) -> dict[str, Any]:
        self._assert_connected()
        return await asyncio.to_thread(self._upload_apk, local_path, remote_path)

    async def install_apk(self, package: str, remote_path: str) -> dict[str, Any]:
        self._assert_connected()
        return await asyncio.to_thread(self._install_apk, package, remote_path)

    async def grant_permissions(self, package: str, permissions: list[str]) -> dict[str, Any]:
        self._assert_connected()
        return await asyncio.to_thread(self._grant_permissions, package, permissions)

    async def launch_app_with_fallback(self, package: str, activity: str | None = None) -> dict[str, Any]:
        self._assert_connected()
        return await asyncio.to_thread(self._launch_app_with_fallback, package, activity)

    async def dump_logcat(self) -> str:
        self._assert_connected()
        return await asyncio.to_thread(self._dump_logcat)

    # ------------------------------------------------------------------
    # call_executor dispatch
    # ------------------------------------------------------------------

    async def call_executor(self, executor_name: str, arg: Any) -> Any:
        self._assert_connected()

        try:
            if executor_name == 'gui_click':
                await asyncio.to_thread(self._tap, arg['x'], arg['y'])
                return {'success': True}

            if executor_name == 'gui_double_click':
                await asyncio.to_thread(self._double_tap, arg['x'], arg['y'])
                return {'success': True}

            if executor_name == 'gui_long_click':
                await asyncio.to_thread(self._long_press, arg['x'], arg['y'], arg.get('duration', 1000))
                return {'success': True}

            if executor_name == 'gui_drag':
                await asyncio.to_thread(
                    self._swipe,
                    arg['x1'],
                    arg['y1'],
                    arg['x2'],
                    arg['y2'],
                    arg.get('duration', 300),
                )
                return {'success': True}

            if executor_name == 'gui_type':
                await asyncio.to_thread(self._type_text, arg['text'])
                return {'success': True}

            if executor_name == 'gui_press_key':
                await asyncio.to_thread(self._press_key, arg['key'])
                return {'success': True}

            if executor_name == 'gui_screenshot':
                png_bytes = await asyncio.to_thread(self._get_screenshot_png)
                image_b64 = base64.b64encode(png_bytes).decode('utf-8')
                if self._screen_size is None:
                    self._screen_size = await asyncio.to_thread(self._get_screen_size)
                return {
                    'success': True,
                    'image': image_b64,
                    'width': self._screen_size[0],
                    'height': self._screen_size[1],
                    'format': 'png',
                }

            if executor_name == 'gui_get_screen_size':
                self._screen_size = await asyncio.to_thread(self._get_screen_size)
                return {
                    'success': True,
                    'width': self._screen_size[0],
                    'height': self._screen_size[1],
                }

            if executor_name == 'gui_get_device_info':
                return await asyncio.to_thread(self._get_device_info)

            if executor_name == 'gui_launch_app':
                await asyncio.to_thread(self._launch_app, arg['app'])
                return {'success': True}

            if executor_name == 'gui_close_app':
                await asyncio.to_thread(self._close_app, arg['app'])
                return {'success': True}

            if executor_name == 'gui_reset_apps':
                stopped_packages = await asyncio.to_thread(self._reset_apps)
                return {'success': True, 'stopped_packages': stopped_packages}

            return {'success': False, 'error': f'Unsupported executor: {executor_name}'}

        except Exception as e:
            logger.error(f'AGRMobileRuntime executor {executor_name} failed: {e}')
            return {'success': False, 'error': str(e)}

    async def _invoke(self, request: Any) -> Any:
        raise NotImplementedError('AGRMobileRuntime does not support invoke')

    async def call_function(self, func, /, *args, recurse=False, **kwargs):
        raise NotImplementedError('AGRMobileRuntime does not support call_function')

    async def call_tool(self, import_path: str, params: dict, context: Any) -> Any:
        raise NotImplementedError('AGRMobileRuntime does not support call_tool')
