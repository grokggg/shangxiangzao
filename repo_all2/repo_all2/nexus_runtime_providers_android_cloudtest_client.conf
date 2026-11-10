"""CloudTest-based Android runtime implementation."""

from __future__ import annotations

import asyncio
import base64
import logging

from typing import TYPE_CHECKING, Any

from cloudtestapi import CloudTestAPI

from nexus.runtime.client.base import Runtime


if TYPE_CHECKING:
    from nexus.runtime.models import InvokeRequest

logger = logging.getLogger(__name__)


class CloudTestAndroidRuntime(Runtime):
    """CloudTest-based Android runtime for Tencent cloud devices.

    Uses CloudTest API to control real Android devices.
    Coordinates are normalized to 0-1000 range.

    Example:
        runtime = CloudTestAndroidRuntime(
            device_id=28093,
            user_id='your_user_id',
            user_key='your_user_key',
        )
        await runtime.connect()
        await runtime.gui_click(720, 1600)
        await runtime.disconnect()
    """

    KEY_CODES = {
        'Back': 4,
        'Home': 3,
        'Menu': 82,
        'Enter': 66,
        'Delete': 67,
        'Tab': 61,
        'Escape': 111,
        'Space': 62,
        'VolumeUp': 24,
        'VolumeDown': 25,
        'Power': 26,
    }

    APP_PACKAGES = {
        '微信': 'com.tencent.mm',
        'WeChat': 'com.tencent.mm',
        'QQ': 'com.tencent.mobileqq',
        '淘宝': 'com.taobao.taobao',
        'Taobao': 'com.taobao.taobao',
        '抖音': 'com.ss.android.ugc.aweme',
        'TikTok': 'com.ss.android.ugc.aweme',
        '设置': 'com.android.settings',
        'Settings': 'com.android.settings',
        '浏览器': 'com.android.browser',
        'Browser': 'com.android.browser',
        'Chrome': 'com.android.chrome',
        '相机': 'com.android.camera',
        'Camera': 'com.android.camera',
    }

    def __init__(
        self,
        device_id: int,
        *,
        host: str = 'http://api.cloudtest.woa.com/',
        user_id: str | None = None,
        user_key: str | None = None,
        is_public_cloud: bool = False,
        project: str | None = None,
    ) -> None:
        super().__init__()
        self.device_id = device_id
        self.host = host
        self.user_id = user_id
        self.user_key = user_key
        self.is_public_cloud = is_public_cloud
        self.project = project

        self._client: Any = None
        self._device: Any = None
        self._screen_size: tuple[int, int] | None = None
        self._connected = False

    async def connect(self) -> bool:
        """Connect to CloudTest device."""
        self._client = CloudTestAPI(self.host, self.user_id, self.user_key)

        if self.is_public_cloud:
            self._device = self._client.new_cloud_device(self.device_id, project=self.project)
        else:
            self._device = self._client.new_cloud_device(self.device_id)

        window_size = await asyncio.to_thread(self._device.get_window_size)
        if isinstance(window_size, tuple):
            self._screen_size = window_size
        elif isinstance(window_size, dict) and (
            window_size.get('result') == 0 or ('width' in window_size and 'height' in window_size)
        ):
            self._screen_size = (window_size['width'], window_size['height'])

        self._connected = True
        logger.info(f'CloudTestAndroidRuntime connected to device {self.device_id}, screen_size={self._screen_size}')
        return True

    async def disconnect(self) -> bool:
        """Disconnect from device."""
        self._device = None
        self._connected = False
        logger.info(f'CloudTestAndroidRuntime disconnected from device {self.device_id}')
        return True

    @property
    def connected(self) -> bool:
        return self._connected

    def _normalize_coordinate(self, x: int, y: int) -> tuple[int, int]:
        if not self._screen_size:
            return x, y
        screen_w, screen_h = self._screen_size
        norm_x = int(x * 1000 / screen_w)
        norm_y = int(y * 1000 / screen_h)
        return max(0, min(1000, norm_x)), max(0, min(1000, norm_y))

    @staticmethod
    def _check_result(result: Any) -> bool:
        if isinstance(result, dict):
            if 'success' in result:
                return result.get('success') is True
            if 'result' in result:
                return result.get('result') == 0
            return False
        return bool(result)

    def _get_package_name(self, app: str) -> str:
        if '.' in app:
            return app
        return self.APP_PACKAGES.get(app, app)

    async def call_executor(self, executor_name: str, arg: Any) -> Any:
        """Execute GUI action via CloudTest API."""
        if not self._connected:
            raise RuntimeError('CloudTest device not connected. Call connect() first.')

        try:
            if executor_name == 'gui_click':
                norm_x, norm_y = self._normalize_coordinate(arg['x'], arg['y'])
                result = await asyncio.to_thread(self._device.tap, x=norm_x, y=norm_y)
                return {'success': self._check_result(result)}

            if executor_name == 'gui_double_click':
                norm_x, norm_y = self._normalize_coordinate(arg['x'], arg['y'])
                result1 = await asyncio.to_thread(self._device.tap, x=norm_x, y=norm_y)
                await asyncio.sleep(0.1)
                result2 = await asyncio.to_thread(self._device.tap, x=norm_x, y=norm_y)
                return {'success': self._check_result(result1) and self._check_result(result2)}

            if executor_name == 'gui_long_click':
                norm_x, norm_y = self._normalize_coordinate(arg['x'], arg['y'])
                duration_sec = arg.get('duration', 1000) / 1000.0
                result = await asyncio.to_thread(self._device.long_press, x=norm_x, y=norm_y, duration=duration_sec)
                return {'success': self._check_result(result)}

            if executor_name == 'gui_drag':
                norm_x1, norm_y1 = self._normalize_coordinate(arg['x1'], arg['y1'])
                norm_x2, norm_y2 = self._normalize_coordinate(arg['x2'], arg['y2'])
                result = await asyncio.to_thread(
                    self._device.swipe,
                    start_x=norm_x1,
                    start_y=norm_y1,
                    end_x=norm_x2,
                    end_y=norm_y2,
                    duration=arg.get('duration', 300),
                )
                return {'success': self._check_result(result)}

            if executor_name == 'gui_type':
                result = await asyncio.to_thread(self._device.input_text, text=arg['text'])
                return {'success': self._check_result(result)}

            if executor_name == 'gui_press_key':
                key = arg['key']
                if key == 'Home':
                    result = await asyncio.to_thread(self._device.home)
                elif key == 'Back':
                    result = await asyncio.to_thread(self._device.back)
                else:
                    keycode = self.KEY_CODES.get(key)
                    if keycode is None:
                        return {'success': False, 'error': f'Unknown key: {key}'}
                    result = await asyncio.to_thread(self._device.press_key, keycode=keycode)
                return {'success': self._check_result(result)}

            if executor_name == 'gui_screenshot':
                import os
                import tempfile

                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                    save_path = f.name

                await asyncio.to_thread(self._device.screenshot, save_path=save_path)

                with open(save_path, 'rb') as f:
                    image_bytes = f.read()

                os.unlink(save_path)

                return {
                    'success': True,
                    'image': base64.b64encode(image_bytes).decode('utf-8'),
                    'width': self._screen_size[0] if self._screen_size else None,
                    'height': self._screen_size[1] if self._screen_size else None,
                    'format': 'png',
                }

            if executor_name == 'gui_get_screen_size':
                return {
                    'success': True,
                    'width': self._screen_size[0] if self._screen_size else None,
                    'height': self._screen_size[1] if self._screen_size else None,
                }

            if executor_name == 'gui_launch_app':
                package = self._get_package_name(arg['app'])
                result = await asyncio.to_thread(self._device.start_app, package=package)
                return {'success': self._check_result(result)}

            if executor_name == 'gui_close_app':
                package = self._get_package_name(arg['app'])
                await asyncio.to_thread(self._device.shell, command=f'am force-stop {package}')
                return {'success': True}

            return {'success': False, 'error': f'Unsupported executor: {executor_name}'}

        except Exception as e:
            logger.error(f'CloudTestAndroidRuntime executor {executor_name} failed: {e}')
            return {'success': False, 'error': str(e)}

    async def _invoke(self, request: InvokeRequest) -> Any:
        raise NotImplementedError('CloudTestAndroidRuntime does not support invoke')

    async def call_tool(self, import_path: str, params: dict, context: Any) -> Any:
        raise NotImplementedError('CloudTestAndroidRuntime does not support call_tool')
