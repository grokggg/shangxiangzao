"""Android World HTTP runtime.

Talks to an Android World backend over its REST surface (the same one the
CloudTest ``android_world`` tool exposes). Unlike the Docker / Gongfeng
runtimes there is no Nexus runtime server in the loop: this client translates
Nexus ``gui_*`` executor calls directly into Android World HTTP requests, the
same way :class:`~nexus.runtime.providers.smartrun.client.SmartRunRuntime`
does for SmartRun devices.

Coordinates are passed through as actual screen pixels (no 0-1000
normalization); the task's system prompt tells the model the real resolution.

API conventions:
- Auth: ``Authorization: Bearer <api_key>`` header.
- GET requests do NOT send a Content-Type header.
- Screenshot: ``GET /screenshot?fmt=jpg`` -> ``{image_base64, image_format}``.
- Actions: ``POST /execute_action`` with a JSON body.
- Task lifecycle: query parameters (``task_type`` / ``task_idx``), not a body.
"""

from __future__ import annotations

import base64
import logging
import struct

from typing import TYPE_CHECKING, Any

from nexus.runtime.client.base import Runtime
from nexus.utils.async_http import http_client


if TYPE_CHECKING:
    from nexus.runtime.models import InvokeRequest

logger = logging.getLogger(__name__)


class AndroidWorldRuntime(Runtime):
    """HTTP-based runtime for Android World devices.

    Translates Nexus GUI executor calls into Android World REST requests and
    exposes the Android World task lifecycle (initialize / score / tear_down)
    used by the ``android_world`` task adapter.

    Example:
        runtime = AndroidWorldRuntime(
            device_id=39810,
            host='http://api.cloudtest.woa.com',
            api_key='your_api_key',
        )
        await runtime.connect()
        goal = await runtime.task_initialize('ContactsAddContact', seed=42)
        shot = await runtime.gui_screenshot()
        score = await runtime.task_score()
        await runtime.task_tear_down()
        await runtime.disconnect()
    """

    def __init__(
        self,
        device_id: int,
        *,
        host: str = 'http://api.cloudtest.woa.com',
        api_key: str | None = None,
    ) -> None:
        super().__init__()
        self.device_id = device_id
        self.host = host.rstrip('/')
        self.api_key = api_key

        self._base_url = f'{self.host}/v1/agent/devices/{self.device_id}/tool/android_world'
        self._screen_size: tuple[int, int] | None = None
        self._connected = False
        # Track the current task so score/tear_down default to the right task.
        self._current_task_type: str | None = None
        self._current_task_idx: int = 0

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    def _auth_headers(self) -> dict[str, str]:
        """Headers for GET requests (no Content-Type)."""
        return {'Authorization': f'Bearer {self.api_key}'} if self.api_key else {}

    def _json_headers(self) -> dict[str, str]:
        """Headers for JSON POST requests."""
        headers = {'Content-Type': 'application/json'}
        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'
        return headers

    async def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        return await http_client.get(f'{self._base_url}{path}', headers=self._auth_headers(), params=params)

    async def _post(
        self, path: str, *, json: dict[str, Any] | None = None, params: dict[str, Any] | None = None
    ) -> Any:
        if json is not None:
            return await http_client.post(
                f'{self._base_url}{path}', json=json, headers=self._json_headers(), params=params
            )
        return await http_client.post(f'{self._base_url}{path}', headers=self._auth_headers(), params=params)

    @staticmethod
    def _jpeg_size(data: bytes) -> tuple[int, int] | None:
        """Extract (width, height) from JPEG bytes without an image library."""
        try:
            i = 0
            while i < len(data) - 9:
                if data[i] != 0xFF:
                    i += 1
                    continue
                marker = data[i + 1]
                if marker in (0xC0, 0xC1, 0xC2, 0xC3):
                    height = struct.unpack('>H', data[i + 5 : i + 7])[0]
                    width = struct.unpack('>H', data[i + 7 : i + 9])[0]
                    return width, height
                if marker in (0xD8, 0xD9) or marker == 0x00 or 0xD0 <= marker <= 0xD7:
                    i += 2
                else:
                    block_len = struct.unpack('>H', data[i + 2 : i + 4])[0]
                    i += 2 + block_len
        except (struct.error, IndexError):
            return None
        return None

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    async def connect(self) -> bool:
        """Health-check the device and cache its screen size."""
        health = await self._get('/health')
        if not health.ok:
            raise RuntimeError(f'Android World health check failed for device {self.device_id}: HTTP {health.status}')

        # Best-effort screen size probe; not fatal if it fails.
        try:
            shot = await self._do_screenshot()
            if shot.get('success') and shot.get('width') and shot.get('height'):
                self._screen_size = (shot['width'], shot['height'])
        except Exception as e:
            logger.warning('Android World screen size probe failed (device %s): %s', self.device_id, e)

        self._connected = True
        logger.info(
            'AndroidWorldRuntime connected: device=%s host=%s screen=%s',
            self.device_id,
            self.host,
            self._screen_size,
        )
        return True

    async def disconnect(self) -> bool:
        self._connected = False
        logger.info('AndroidWorldRuntime disconnected from device %s', self.device_id)
        return True

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def screen_size(self) -> tuple[int, int] | None:
        return self._screen_size

    # ------------------------------------------------------------------
    # GUI executor dispatch
    # ------------------------------------------------------------------

    async def call_executor(self, executor_name: str, arg: Any) -> Any:
        """Translate a Nexus GUI executor call into an Android World request."""
        if not self._connected:
            raise RuntimeError('AndroidWorldRuntime is not connected. Call connect() first.')

        try:
            if executor_name == 'gui_screenshot':
                return await self._do_screenshot()

            if executor_name == 'gui_get_screen_size':
                return await self._do_screen_size()

            if executor_name == 'gui_click':
                return await self._do_action({'action_type': 'click', 'x': arg['x'], 'y': arg['y']})

            if executor_name == 'gui_double_click':
                return await self._do_action({'action_type': 'double_tap', 'x': arg['x'], 'y': arg['y']})

            if executor_name == 'gui_long_click':
                return await self._do_action({'action_type': 'long_press', 'x': arg['x'], 'y': arg['y']})

            if executor_name == 'gui_drag':
                return await self._do_action({
                    'action_type': 'swipe',
                    'x1': arg['x1'],
                    'y1': arg['y1'],
                    'x2': arg['x2'],
                    'y2': arg['y2'],
                })

            if executor_name == 'gui_type':
                return await self._do_action({'action_type': 'input_text', 'text': arg['text']})

            if executor_name == 'gui_press_key':
                return await self._do_press_key(arg['key'])

            if executor_name == 'gui_launch_app':
                return await self._do_action({'action_type': 'open_app', 'app_name': arg['app']})

            if executor_name == 'gui_close_app':
                # Android World has no explicit close; going home is the closest no-op.
                return await self._do_action({'action_type': 'navigate_home'})

            # ---- Android World specific executors (used by mobile/computer) ----
            if executor_name == 'aw_scroll':
                return await self._do_action({'action_type': 'scroll', 'direction': arg['direction']})

            if executor_name == 'aw_swipe':
                # Coordinate-packed swipe: "x1,y1,x2,y2" in the direction field.
                return await self._do_action({'action_type': 'swipe', 'direction': arg['direction']})

            if executor_name == 'aw_answer':
                return await self._do_action({'action_type': 'answer', 'text': arg.get('answer', '')})

            logger.warning('AndroidWorldRuntime unsupported executor: %s', executor_name)
            return {'success': False, 'error': f'unsupported executor: {executor_name}'}

        except Exception as e:
            logger.error('AndroidWorldRuntime executor %s failed: %s', executor_name, e)
            return {'success': False, 'error': str(e)}

    async def _do_screenshot(self) -> dict[str, Any]:
        resp = await self._get(
            '/screenshot',
            params={'wait_to_stabilize': 'false', 'fmt': 'jpg', 'include_ui_elements': 'false'},
        )
        if not resp.ok:
            return {'success': False, 'error': f'screenshot failed: HTTP {resp.status}'}

        data = resp.json()
        if not isinstance(data, dict):
            return {'success': False, 'error': 'unexpected screenshot response format'}

        image_b64 = data.get('image_base64', '') or data.get('image', '')
        if not image_b64:
            return {'success': False, 'error': 'no image in screenshot response'}

        width = data.get('width')
        height = data.get('height')
        if not (width and height):
            size = self._jpeg_size(base64.b64decode(image_b64))
            if size:
                width, height = size
        if width and height:
            self._screen_size = (width, height)

        return {
            'success': True,
            'image': image_b64,
            'width': self._screen_size[0] if self._screen_size else None,
            'height': self._screen_size[1] if self._screen_size else None,
            'format': 'jpeg',
        }

    async def _do_screen_size(self) -> dict[str, Any]:
        if self._screen_size:
            return {'success': True, 'width': self._screen_size[0], 'height': self._screen_size[1]}
        result = await self._do_screenshot()
        if result.get('success') and result.get('width') and result.get('height'):
            return {'success': True, 'width': result['width'], 'height': result['height']}
        return {'success': False, 'error': 'cannot determine screen size'}

    async def _do_action(self, action: dict[str, Any]) -> dict[str, Any]:
        resp = await self._post('/execute_action', json=action)
        if not resp.ok:
            body = resp.text[:200] if resp.text else ''
            return {'success': False, 'error': f'action {action.get("action_type")} failed: HTTP {resp.status} {body}'}
        return {'success': True}

    async def _do_press_key(self, key: str) -> dict[str, Any]:
        key_action_map = {
            'home': 'navigate_home',
            'back': 'navigate_back',
            'enter': 'keyboard_enter',
        }
        action_type = key_action_map.get(key.lower(), f'press_{key.lower()}')
        return await self._do_action({'action_type': action_type})

    # ------------------------------------------------------------------
    # Task lifecycle (Android World REST)
    # ------------------------------------------------------------------

    async def task_initialize(self, task_type: str, seed: int = 0, task_idx: int = 0) -> str:
        """Initialize an Android World task and return its goal description.

        Runs the Android World initialization sequence:
        1. ``POST /reset?go_home=true`` — reset the environment.
        2. ``GET /suite/reinitialize`` — reinitialize the test suite for ``seed``.
        3. ``POST /task/tear_down`` — tear down any previous task (errors ignored).
        4. ``POST /task/initialize`` — initialize the requested task.
        5. ``GET /task/goal`` — fetch the goal description.
        """
        self._current_task_type = task_type
        self._current_task_idx = task_idx
        task_params = {'task_type': task_type, 'task_idx': str(task_idx)}

        resp = await self._post('/reset', params={'go_home': 'true'})
        if not resp.ok:
            logger.warning('Android World reset failed: HTTP %s, continuing', resp.status)

        resp = await self._get(
            '/suite/reinitialize',
            params={'n_task_combinations': '1', 'seed': str(seed), 'task_family': 'android_world'},
        )
        if not resp.ok:
            logger.warning('Android World suite reinitialize failed: HTTP %s, continuing', resp.status)

        try:
            resp = await self._post('/task/tear_down', params=task_params)
            if not resp.ok:
                logger.debug('Android World pre-init tear_down failed (ignored): HTTP %s', resp.status)
        except Exception as e:
            logger.debug('Android World pre-init tear_down exception (ignored): %s', e)

        resp = await self._post('/task/initialize', params=task_params)
        if not resp.ok:
            body = resp.text[:300] if resp.text else ''
            raise RuntimeError(f'Android World task initialize failed: HTTP {resp.status} {body}')

        resp = await self._get('/task/goal', params=task_params)
        if not resp.ok:
            body = resp.text[:300] if resp.text else ''
            raise RuntimeError(f'Android World get task goal failed: HTTP {resp.status} {body}')

        data = resp.json()
        goal = data.get('goal', data.get('text', str(data))) if isinstance(data, dict) else str(data)
        logger.info('Android World task initialized: type=%s idx=%s goal=%.100s', task_type, task_idx, goal)
        return goal

    async def task_score(self, task_type: str | None = None, task_idx: int | None = None) -> dict[str, Any]:
        """Get the native Android World score for the current task."""
        params = {
            'task_type': task_type or self._current_task_type,
            'task_idx': str(task_idx if task_idx is not None else self._current_task_idx),
        }
        resp = await self._get('/task/score', params=params)
        if not resp.ok:
            logger.error('Android World task score failed: HTTP %s', resp.status)
            return {'score': 0.0, 'error': f'HTTP {resp.status}'}
        data = resp.json()
        if isinstance(data, dict):
            return data
        return {'score': float(data) if data else 0.0}

    async def task_tear_down(self, task_type: str | None = None, task_idx: int | None = None) -> bool:
        """Tear down the current task."""
        params = {
            'task_type': task_type or self._current_task_type,
            'task_idx': str(task_idx if task_idx is not None else self._current_task_idx),
        }
        resp = await self._post('/task/tear_down', params=params)
        if not resp.ok:
            logger.warning('Android World task tear_down failed: HTTP %s', resp.status)
            return False
        return True

    # ------------------------------------------------------------------
    # Unsupported transport methods
    # ------------------------------------------------------------------

    async def _invoke(self, request: InvokeRequest) -> Any:
        raise NotImplementedError('AndroidWorldRuntime does not support invoke')

    async def call_tool(self, import_path: str, params: dict, context: Any) -> Any:
        raise NotImplementedError('AndroidWorldRuntime does not support call_tool')


__all__ = ['AndroidWorldRuntime']
