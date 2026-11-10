"""SmartRun cloud real-device runtime implementation."""

from __future__ import annotations

import base64
import logging

from typing import TYPE_CHECKING, Any

from nexus.runtime.client.base import Runtime
from nexus.utils.async_http import http_client


if TYPE_CHECKING:
    from nexus.runtime.models import InvokeRequest

logger = logging.getLogger(__name__)


class SmartRunRuntime(Runtime):
    """SmartRun-based runtime for iOS/Android cloud real devices.

    Translates nexus GUI executor calls into SmartRun MCP HTTP requests.
    Supports both iOS and Android; platform differences are handled internally.

    Example:
        runtime = SmartRunRuntime(
            device_id='00008020-000E6D863C01002E',
            session_id='session-abc',
            screen_size=(1125, 2436),
            platform='ios',
            api_base_url='https://smartrun.woa.com',
            token='mcp_xxx',
        )
        await runtime.connect()
        result = await runtime.gui_screenshot()
        await runtime.disconnect()
    """

    def __init__(
        self,
        device_id: str,
        session_id: str,
        screen_size: tuple[int, int],
        platform: str,
        *,
        api_base_url: str = 'https://smartrun.woa.com',
        token: str,
    ) -> None:
        super().__init__()
        self.device_id = device_id
        self.session_id = session_id
        self.screen_size = screen_size
        self.platform = platform
        self._api_base_url = api_base_url.rstrip('/')
        self._token = token
        self._connected = False

    @property
    def connected(self) -> bool:
        return self._connected

    async def connect(self) -> bool:
        """Verify session is valid and mark connected."""
        data = await self._call('get_session_status', {'session_id': self.session_id, 'json_output': True})
        if not data.get('success'):
            raise RuntimeError(f'SmartRun session {self.session_id} is not valid: {data}')
        self._connected = True
        logger.info(
            'SmartRunRuntime connected: device=%s session=%s platform=%s screen=%s',
            self.device_id,
            self.session_id,
            self.platform,
            self.screen_size,
        )
        return True

    async def disconnect(self) -> bool:
        """Mark disconnected. Actual release is handled by the provider."""
        self._connected = False
        return True

    async def _call(self, tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Send a JSON-RPC request to SmartRun MCP and return the parsed result dict.

        Raises:
            RuntimeError: when SmartRun returns isError=true or HTTP error.
        """
        import json

        url = f'{self._api_base_url}/mcp'
        headers = {
            'Authorization': f'Bearer {self._token}',
            'Content-Type': 'application/json',
        }
        payload = {
            'jsonrpc': '2.0',
            'method': 'tools/call',
            'params': {'name': tool, 'arguments': arguments},
            'id': 1,
        }

        resp = await http_client.post(url, json=payload, headers=headers)
        if resp.status_code != 200:
            raise RuntimeError(f'SmartRun HTTP error {resp.status_code} for tool={tool}: {resp.text[:512]}')

        # SmartRun returns SSE lines; find the data: line
        data_line = ''
        for line in resp.text.splitlines():
            if line.startswith('data:'):
                data_line = line[5:].strip()
                break

        if not data_line:
            raise RuntimeError(f'SmartRun returned no data line for tool={tool}: {resp.text[:512]}')

        rpc = json.loads(data_line)

        result = rpc.get('result', {})
        if result.get('isError'):
            content = result.get('content', [{}])
            msg = content[0].get('text', 'unknown error') if content else 'unknown error'
            raise RuntimeError(f'SmartRun tool={tool} error: {msg}')

        content = result.get('content', [{}])
        text = content[0].get('text', '{}') if content else '{}'
        return json.loads(text)

    async def call_executor(self, executor_name: str, arg: Any) -> Any:
        """Execute a GUI action via SmartRun API."""
        if not self._connected:
            raise RuntimeError('SmartRunRuntime is not connected. Call connect() first.')
        try:
            return await self._dispatch(executor_name, arg)
        except RuntimeError:
            raise
        except Exception as e:
            logger.error('SmartRunRuntime executor %s failed: %s', executor_name, e)
            return {'success': False, 'error': str(e)}

    async def _dispatch(self, executor_name: str, arg: Any) -> dict[str, Any]:
        """Route executor_name to the appropriate SmartRun tool call."""
        base = {'device_id': self.device_id, 'session_id': self.session_id, 'json_output': True}

        if executor_name == 'gui_screenshot':
            result = await self._call('screenshot', base)
            url = result.get('screenshot_url', '')
            if not url:
                return {'success': False, 'error': 'no screenshot_url in response'}
            image_resp = await http_client.get(url)
            image_b64 = base64.b64encode(image_resp.content).decode('utf-8')
            return {
                'success': True,
                'image': image_b64,
                'width': self.screen_size[0],
                'height': self.screen_size[1],
                'format': 'jpg',
            }

        if executor_name == 'gui_get_screen_size':
            return {'success': True, 'width': self.screen_size[0], 'height': self.screen_size[1]}

        if executor_name == 'gui_click':
            result = await self._call('tap', {**base, 'x': arg['x'], 'y': arg['y']})
            return {'success': result.get('success', False)}

        if executor_name == 'gui_double_click':
            result = await self._call('double_tap', {**base, 'x': arg['x'], 'y': arg['y']})
            return {'success': result.get('success', False)}

        if executor_name == 'gui_long_click':
            result = await self._call(
                'long_press',
                {**base, 'x': arg['x'], 'y': arg['y'], 'duration': arg.get('duration', 1000)},
            )
            return {'success': result.get('success', False)}

        if executor_name == 'gui_drag':
            result = await self._call(
                'swipe',
                {
                    **base,
                    'start_x': arg['x1'],
                    'start_y': arg['y1'],
                    'end_x': arg['x2'],
                    'end_y': arg['y2'],
                    'duration': arg.get('duration', 300),
                },
            )
            return {'success': result.get('success', False)}

        if executor_name == 'gui_type':
            result = await self._call('input_text', {**base, 'text': arg['text']})
            return {'success': result.get('success', False)}

        if executor_name == 'gui_press_key':
            key = arg.get('key', '')
            if key == 'Home':
                result = await self._call('press_home', base)
            elif key == 'Back':
                result = await self._call('press_back', base)
            else:
                return {'success': False, 'error': f'unsupported key: {key}'}
            return {'success': result.get('success', False)}

        if executor_name == 'gui_launch_app':
            result = await self._call('launch_app', {**base, 'package_name': arg['app']})
            return {'success': result.get('success', False)}

        if executor_name == 'gui_close_app':
            result = await self._call('stop_app', {**base, 'package_name': arg['app']})
            return {'success': result.get('success', False)}

        return {'success': False, 'error': f'unsupported executor: {executor_name}'}

    async def _invoke(self, request: InvokeRequest) -> Any:
        raise NotImplementedError('SmartRunRuntime does not support invoke')

    async def call_tool(self, import_path: str, params: dict, context: Any) -> Any:
        raise NotImplementedError('SmartRunRuntime does not support call_tool')


__all__ = ['SmartRunRuntime']
