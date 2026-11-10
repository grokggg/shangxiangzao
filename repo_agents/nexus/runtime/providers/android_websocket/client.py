"""WebSocket-based Android runtime implementation."""

from __future__ import annotations

import base64
import logging

from typing import TYPE_CHECKING, Any

import websockets

from nexus.runtime.client.base import Runtime


if TYPE_CHECKING:
    from nexus.runtime.models import InvokeRequest

logger = logging.getLogger(__name__)


class WebSocketAndroidRuntime(Runtime):
    """WebSocket-based Android runtime for sandbox environments.

    Communicates with a WebSocket sandbox server to control Android devices.
    Uses fixed 999x999 resolution for coordinate normalization.

    Example:
        runtime = WebSocketAndroidRuntime(
            ws_url='ws://9.151.13.55:8788',
            instance_id='cai-1383819211-xxxxx',
        )
        await runtime.connect()
        await runtime.gui_click(500, 300)
        await runtime.disconnect()
    """

    SCREEN_WIDTH = 999
    SCREEN_HEIGHT = 999

    ACTION_MAPPING = {
        'gui_click': 'click',
        'gui_double_click': 'double_click',
        'gui_long_click': 'long_press',
        'gui_drag': 'scroll',
        'gui_type': 'type',
        'gui_press_key': 'button_press',
        'gui_screenshot': 'screenshot',
        'gui_get_screen_size': 'get_screen_size',
        'gui_launch_app': 'open_app',
        'gui_close_app': 'close_app',
    }

    def __init__(
        self,
        ws_url: str,
        instance_id: str,
        *,
        instructions: list[str] | None = None,
        with_history: bool = True,
        model_version: str = 'no',
        manual_vlm: bool = True,
    ) -> None:
        super().__init__()
        self.ws_url = ws_url
        self.instance_id = instance_id
        self.instructions = instructions or ['自动化测试']
        self.with_history = with_history
        self.model_version = model_version
        self.manual_vlm = manual_vlm

        self._ws: Any = None
        self._last_response: dict | None = None
        self._connected = False

    async def connect(self) -> bool:
        """Connect to WebSocket and send initialization data."""
        import json

        self._ws = await websockets.connect(self.ws_url, open_timeout=30)

        init_data = {
            'instance_id': self.instance_id,
            'instructions': self.instructions,
            'with_history': self.with_history,
            'model_version': self.model_version,
            'manual_vlm': self.manual_vlm,
        }
        await self._ws.send(json.dumps(init_data))

        response = await self._ws.recv()
        self._last_response = json.loads(response)
        self._connected = True
        logger.info(f'WebSocketAndroidRuntime connected to {self.ws_url}')
        return True

    async def disconnect(self) -> bool:
        """Disconnect from WebSocket."""
        if self._ws:
            await self._ws.close()
            self._ws = None
        self._connected = False
        logger.info(f'WebSocketAndroidRuntime disconnected from {self.ws_url}')
        return True

    @property
    def connected(self) -> bool:
        return self._connected

    async def call_executor(self, executor_name: str, arg: Any) -> Any:
        """Execute GUI action via WebSocket."""
        import json

        if not self._connected:
            raise RuntimeError('WebSocket not connected. Call connect() first.')

        action = self.ACTION_MAPPING.get(executor_name)
        if not action:
            raise ValueError(f'Unsupported executor: {executor_name}')

        action_data = self._build_toolcall_request(action, executor_name, arg)

        await self._ws.send(json.dumps(action_data, ensure_ascii=False))
        response = await self._ws.recv()
        self._last_response = json.loads(response)

        return await self._transform_response(executor_name, self._last_response)

    def _build_toolcall_request(self, action: str, executor_name: str, arg: dict) -> dict:
        if action == 'screenshot':
            return {'action': action}

        arguments: dict = {
            'points': [[0, 0]],
            'text': '',
            'time': 100,
            'type': '',
            'app': '',
        }

        if action == 'click' or action == 'double_click':
            arguments['points'] = [[arg.get('x', 0), arg.get('y', 0)]]
        elif action == 'long_press':
            arguments['points'] = [[arg.get('x', 0), arg.get('y', 0)]]
            arguments['time'] = arg.get('duration', 1000)
        elif action == 'scroll':
            arguments['points'] = [
                [arg.get('x1', 0), arg.get('y1', 0)],
                [arg.get('x2', 0), arg.get('y2', 0)],
            ]
            arguments['time'] = arg.get('duration', 300)
        elif action == 'type':
            arguments['text'] = arg.get('text', '')
        elif action == 'button_press':
            key = arg.get('key', 'back')
            key_mapping = {
                'Back': 'back',
                'Home': 'home',
                'Menu': 'menu',
                'Enter': 'enter',
                'Delete': 'delete',
                'Tab': 'tab',
                'Escape': 'escape',
            }
            arguments['type'] = key_mapping.get(key, key.lower())
        elif action == 'open_app' or action == 'close_app':
            arguments['app'] = arg.get('app', '')

        return {
            'Thought': '',
            'Description': '',
            'ScreenshotWaitingTime': 2000,
            'ToolCall': {
                'name': 'mobile',
                'action': action,
                'arguments': arguments,
            },
        }

    async def _transform_response(self, executor_name: str, response: dict) -> dict:
        if executor_name == 'gui_screenshot':
            output = response.get('output', [])
            if output:
                img_data = output[-1] if isinstance(output, list) else output

                if isinstance(img_data, str) and (img_data.startswith('http://') or img_data.startswith('https://')):
                    try:
                        import aiohttp

                        async with aiohttp.ClientSession() as session:
                            async with session.get(img_data, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                                if resp.status == 200:
                                    image_bytes = await resp.read()
                                    img_data = base64.b64encode(image_bytes).decode('utf-8')
                                else:
                                    logger.warning(f'Failed to download screenshot: HTTP {resp.status}')
                                    img_data = None
                    except Exception as e:
                        logger.warning(f'Failed to download screenshot from URL: {e}')
                        img_data = None

                return {
                    'success': True,
                    'image': img_data,
                    'width': self.SCREEN_WIDTH,
                    'height': self.SCREEN_HEIGHT,
                    'format': 'png',
                }
            return {'success': True, 'image': None}
        if executor_name == 'gui_get_screen_size':
            return {
                'success': True,
                'width': self.SCREEN_WIDTH,
                'height': self.SCREEN_HEIGHT,
            }
        return {'success': True}

    async def _invoke(self, request: InvokeRequest) -> Any:
        raise NotImplementedError('WebSocketAndroidRuntime does not support invoke')

    async def call_tool(self, import_path: str, params: dict, context: Any) -> Any:
        raise NotImplementedError('WebSocketAndroidRuntime does not support call_tool')
