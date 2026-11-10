"""AGS RuntimeProvider for Tencent Cloud Agent Sandbox."""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import json
import logging
import uuid

from collections.abc import Mapping
from typing import Any

import aiohttp

from e2b_code_interpreter import Sandbox
from tencentcloud.ags.v20250920 import ags_client, models
from tencentcloud.common import credential
from tencentcloud.common.profile.client_profile import ClientProfile
from tencentcloud.common.profile.http_profile import HttpProfile

from nexus.runtime.client.base import Runtime
from nexus.runtime.client.remote import HttpRuntime, SocketIoRuntime
from nexus.runtime.providers.ags.base import AGSRawRuntime
from nexus.runtime.providers.ags.browser import AGSBrowserRuntime, AGSRawBrowserRuntime
from nexus.runtime.providers.ags.code_interpreter import AGSCodeInterpreterRuntime, AGSRawCodeInterpreterRuntime
from nexus.runtime.providers.ags.handle import AGSInstanceHandle
from nexus.runtime.providers.ags.osworld import AGSOSWorldRuntime, AGSRawOSWorldRuntime
from nexus.runtime.providers.ags.waa import AGSRawWindowsAgentArenaRuntime, AGSWindowsAgentArenaRuntime
from nexus.runtime.providers.base import InvalidPersistentRuntimeError, RuntimeProvider
from nexus.runtime.providers.registry import runtime_provider_registry
from nexus.utils.async_http import http_client
from nexus.utils.common import get_current_python_version


logger = logging.getLogger(__name__)


def _is_sensitive_log_key(key: str) -> bool:
    normalized = ''.join(ch for ch in key.lower() if ch.isalnum())
    return any(token in normalized for token in ('token', 'secret', 'apikey', 'password'))


def _redact_for_log(value: Any, *, key: str | None = None) -> Any:
    if key is not None and _is_sensitive_log_key(key):
        return '***REDACTED***'
    if isinstance(value, dict):
        return {item_key: _redact_for_log(item_value, key=item_key) for item_key, item_value in value.items()}
    if isinstance(value, list):
        return [_redact_for_log(item) for item in value]
    return value


def _exception_code(exc: Exception) -> str:
    for attr_name in ('code', 'Code'):
        value = getattr(exc, attr_name, None)
        if value:
            return str(value)
    getter = getattr(exc, 'get_code', None)
    if callable(getter):
        try:
            value = getter()
            if value:
                return str(value)
        except Exception:
            pass
    text = str(exc)
    marker = 'code:'
    if marker in text:
        return text.split(marker, 1)[1].split(None, 1)[0].strip()
    return ''


def _is_retryable_ags_api_error(action: str, exc: Exception) -> bool:
    # WAA 全量跑时观测到 StartSandboxInstance 偶发 InternalError，单用例复跑可成功；
    # 这里只把启动实例的 AGS 控制面瞬时错误纳入重试，配额/鉴权/参数错误必须快速失败。
    if action != 'StartSandboxInstance':
        return False
    code = _exception_code(exc)
    text = str(exc)
    if code.startswith('LimitExceeded') or code.startswith('InvalidParameter') or code.startswith('AuthFailure'):
        return False
    return code.startswith('InternalError') or 'Internal server error' in text


def _normalize_tags(tags: dict[str, str] | list[dict[str, str]] | None) -> list[dict[str, str]] | None:
    if not tags:
        return None
    if isinstance(tags, dict):
        return [{'Key': str(key), 'Value': str(value)} for key, value in tags.items()]

    normalized = []
    for item in tags:
        key = item.get('Key') or item.get('key')
        value = item.get('Value') or item.get('value')
        if key is None:
            raise ValueError(f'Invalid AGS tag item: {item!r}')
        normalized.append({'Key': str(key), 'Value': '' if value is None else str(value)})
    return normalized


def _normalize_env(env: dict[str, str] | list[dict[str, str]] | None) -> list[dict[str, str]] | None:
    if not env:
        return None
    if isinstance(env, dict):
        return [{'Name': str(key), 'Value': str(value)} for key, value in env.items()]

    normalized = []
    for item in env:
        name = item.get('Name') or item.get('name')
        value = item.get('Value') or item.get('value')
        if name is None:
            raise ValueError(f'Invalid AGS env item: {item!r}')
        normalized.append({'Name': str(name), 'Value': '' if value is None else str(value)})
    return normalized


def _normalize_ports(ports: list[int] | list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
    if not ports:
        return None

    normalized = []
    for item in ports:
        if isinstance(item, int):
            normalized.append({'Name': f'port-{item}', 'Port': item, 'Protocol': 'TCP'})
            continue

        port = item.get('Port') if 'Port' in item else item.get('port')
        if port is None:
            raise ValueError(f'Invalid AGS port item: {item!r}')

        normalized.append({
            'Name': item.get('Name') or item.get('name') or f'port-{port}',
            'Port': int(port),
            'Protocol': item.get('Protocol') or item.get('protocol') or 'TCP',
        })
    return normalized


def _normalize_resources(resources: dict[str, Any] | None) -> dict[str, str] | None:
    if not resources:
        return None

    cpu = resources.get('CPU') if 'CPU' in resources else resources.get('cpu')
    memory = resources.get('Memory') if 'Memory' in resources else resources.get('memory')

    if cpu is None and memory is None:
        return None

    normalized: dict[str, str] = {}
    if cpu is not None:
        normalized['CPU'] = str(cpu)
    if memory is not None:
        if isinstance(memory, (int, float)):
            normalized['Memory'] = f'{memory}Gi'
        else:
            normalized['Memory'] = str(memory)
    return normalized


def _normalize_probe(probe: dict[str, Any] | None) -> dict[str, Any] | None:
    if not probe:
        return None

    if 'HttpGet' in probe:
        http_get = probe.get('HttpGet') or {}
        normalized = {'HttpGet': http_get}
        for key in (
            'ReadyTimeoutMs',
            'ProbeTimeoutMs',
            'ProbePeriodMs',
            'SuccessThreshold',
            'FailureThreshold',
        ):
            if key in probe:
                normalized[key] = probe[key]
        return normalized

    port = probe.get('port') if 'port' in probe else probe.get('Port')
    if port is None:
        raise ValueError(f'AGS probe requires a port, got {probe!r}')

    return {
        'HttpGet': {
            'Path': probe.get('path') or probe.get('Path') or '/health',
            'Port': int(port),
            'Scheme': probe.get('scheme') or probe.get('Scheme') or 'HTTP',
        },
        'ReadyTimeoutMs': probe.get('ready_timeout_ms', probe.get('ReadyTimeoutMs', 30000)),
        'ProbeTimeoutMs': probe.get('probe_timeout_ms', probe.get('ProbeTimeoutMs', 2000)),
        'ProbePeriodMs': probe.get('probe_period_ms', probe.get('ProbePeriodMs', 1000)),
        'SuccessThreshold': probe.get('success_threshold', probe.get('SuccessThreshold', 1)),
        'FailureThreshold': probe.get('failure_threshold', probe.get('FailureThreshold', 100)),
    }


def _infer_runtime_port(config: dict[str, Any], default_port: int) -> int:
    if config.get('runtime_port') is not None:
        return int(config['runtime_port'])

    probe = config.get('probe')
    if isinstance(probe, dict):
        if 'HttpGet' in probe and isinstance(probe['HttpGet'], dict):
            port = probe['HttpGet'].get('Port')
            if port is not None:
                return int(port)
        if probe.get('port') is not None:
            return int(probe['port'])
        if probe.get('Port') is not None:
            return int(probe['Port'])

    ports = config.get('ports')
    if isinstance(ports, list) and ports:
        first = ports[0]
        if isinstance(first, int):
            return int(first)
        if isinstance(first, dict):
            port = first.get('Port') if 'Port' in first else first.get('port')
            if port is not None:
                return int(port)

    return default_port


@runtime_provider_registry.register('ags')
class AGSRuntimeProvider(RuntimeProvider):
    """RuntimeProvider backed by Tencent Cloud Agent Sandbox."""

    BUILTIN_BROWSER_PORT = 9000
    BUILTIN_CODE_INTERPRETER_PORT = 49999

    def __init__(
        self,
        *,
        secret_id: str | None = None,
        secret_key: str | None = None,
        api_key: str | None = None,
        e2b_api_key: str | None = None,
        region: str = 'ap-guangzhou',
        domain: str | None = None,
        e2b_domain: str | None = None,
        template: str | None = None,
        template_timeout: int = 1800,
        role_arn: str | None = None,
        image: str | None = None,
        image_registry_type: str = 'personal',
        runtime_port: int = 8000,
        python_version: str | None = None,
        network_mode: str = 'PUBLIC',
        resource_spec: dict[str, Any] | None = None,
        request_timeout: float = 1800.0,
        auto_connect_ws: bool = True,
        socketio_request_timeout: float = 60.0,
        rpc_protocol: str = 'http',
        interface_mode: str = 'nexus',
        auth_mode: str | None = None,
        enable_runtime_logs: bool = True,
        tool_ready_timeout: float = 300.0,
        instance_ready_timeout: float = 300.0,
        poll_interval: float = 2.0,
        **config: Any,
    ) -> None:
        super().__init__(**config)

        resolved_api_key = api_key or e2b_api_key
        if bool(secret_id) ^ bool(secret_key):
            raise ValueError('secret_id and secret_key must be provided together')
        if not ((secret_id and secret_key) or resolved_api_key):
            raise ValueError('Either secret_id/secret_key or api_key is required for AGSRuntimeProvider')

        self.secret_id = secret_id
        self.secret_key = secret_key
        self.api_key = resolved_api_key
        self.region = region
        self.domain = domain or e2b_domain or f'{region}.tencentags.com'
        self.template = template
        self.template_timeout = template_timeout
        self.role_arn = role_arn
        self.image = image
        self.image_registry_type = image_registry_type
        self.runtime_port = runtime_port
        self.python_version = python_version or get_current_python_version()
        self.network_mode = network_mode
        self.resource_spec = resource_spec or {}
        self.request_timeout = request_timeout
        self.auto_connect_ws = auto_connect_ws
        self.socketio_request_timeout = socketio_request_timeout
        self.rpc_protocol = rpc_protocol
        self.interface_mode = self._normalize_interface_mode(interface_mode)
        self.auth_mode = auth_mode
        self.enable_runtime_logs = enable_runtime_logs
        self.tool_ready_timeout = tool_ready_timeout
        self.instance_ready_timeout = instance_ready_timeout
        self.poll_interval = poll_interval

        self._client: Any = None
        self._models: Any = None
        self._instance_ids: dict[str, str] = {}
        self._external_instance_names: set[str] = set()
        self._tool_ids: dict[str, str] = {}
        self._tools_to_delete: dict[str, str] = {}
        self._template_sandboxes: dict[str, Any] = {}
        self._acquire_locks: dict[str, asyncio.Lock] = {}

    def get_runtime_id(self, name: str) -> str | None:
        return self._instance_ids.get(name)

    def _get_sdk_client(self):
        if self._client is not None and self._models is not None:
            return self._client, self._models

        if not self.secret_id or not self.secret_key:
            raise ValueError('secret_id and secret_key are required for AGS control-plane operations')

        cred = credential.Credential(self.secret_id, self.secret_key)
        http_profile = HttpProfile(endpoint='ags.tencentcloudapi.com', reqTimeout=max(int(self.request_timeout), 60))
        client_profile = ClientProfile(httpProfile=http_profile)
        self._client = ags_client.AgsClient(cred, self.region, client_profile)
        self._models = models
        return self._client, self._models

    @staticmethod
    def _normalize_provision_mode(value: Any) -> str:
        mode = str(value).strip().lower()
        if mode not in {'control_plane', 'template'}:
            raise ValueError(f'Unsupported AGS provision_mode: {value!r}')
        return mode

    @staticmethod
    def _normalize_interface_mode(value: Any) -> str:
        mode = str(value).strip().lower()
        if mode not in {'nexus', 'raw'}:
            raise ValueError(f'Unsupported AGS interface_mode: {value!r}')
        return mode

    def _get_template_api_key(self, merged: dict[str, Any]) -> str | None:
        return merged.get('api_key') or merged.get('e2b_api_key') or self.api_key

    def _get_runtime_domain(self, merged: dict[str, Any]) -> str:
        return str(merged.get('domain') or merged.get('e2b_domain') or self.domain)

    def _infer_interface_mode(self, merged: dict[str, Any]) -> str:
        explicit = merged.get('interface_mode')
        if explicit is None:
            return self.interface_mode
        return self._normalize_interface_mode(explicit)

    def _get_template_name(self, merged: dict[str, Any]) -> str | None:
        return merged.get('template') or merged.get('sandbox_template') or self.template

    def _get_template_timeout(self, merged: dict[str, Any]) -> int:
        timeout = merged.get('template_timeout')
        if timeout is None:
            timeout = merged.get('timeout')
        if timeout is None:
            timeout = self.template_timeout
        return int(timeout)

    def _infer_provision_mode(self, merged: dict[str, Any]) -> str:
        explicit = merged.get('provision_mode')
        if explicit is None:
            explicit = merged.get('instance_mode')
        if explicit is not None:
            return self._normalize_provision_mode(explicit)

        if self._get_template_name(merged) and self._get_template_api_key(merged):
            return 'template'
        return 'control_plane'

    @staticmethod
    def _extract_template_access_token(sandbox: Any) -> str:
        for attr_name in ('_envd_access_token', '_SandboxBase__envd_access_token'):
            access_token = getattr(sandbox, attr_name, None)
            if access_token:
                return access_token

        raise RuntimeError('AGS template sandbox did not expose an access token')

    async def _create_template_sandbox(self, merged: dict[str, Any]) -> Any:
        template = self._get_template_name(merged)
        if not template:
            raise ValueError('template is required for AGS provision_mode="template"')

        api_key = self._get_template_api_key(merged)
        if not api_key:
            raise ValueError('api_key is required for AGS provision_mode="template"')

        create_payload: dict[str, Any] = {
            'template': template,
            'timeout': self._get_template_timeout(merged),
            'api_key': api_key,
            'domain': self._get_runtime_domain(merged),
            'request_timeout': self.request_timeout,
        }
        logger.info('AGS template create payload=%s', _redact_for_log(create_payload))
        return await asyncio.to_thread(Sandbox.create, **create_payload)

    async def _call_api(self, action: str, payload: dict[str, Any]) -> dict[str, Any]:
        client, models = self._get_sdk_client()
        request_cls = getattr(models, f'{action}Request')
        request = request_cls()
        request.from_json_string(json.dumps(payload))

        logger.info('AGS API request: %s payload=%s', action, _redact_for_log(payload))

        # 仅实例启动做窄范围重试，避免把非幂等或明确失败的 AGS API 调用隐藏掉。
        max_attempts = 3 if action == 'StartSandboxInstance' else 1
        for attempt in range(1, max_attempts + 1):
            try:
                response = await asyncio.to_thread(getattr(client, action), request)
                break
            except Exception as exc:
                if attempt < max_attempts and _is_retryable_ags_api_error(action, exc):
                    delay = min(5 * attempt, 15)
                    logger.warning(
                        'AGS %s transient failure on attempt %s/%s; retrying in %ss: %s: %s',
                        action,
                        attempt,
                        max_attempts,
                        delay,
                        type(exc).__name__,
                        exc,
                    )
                    await asyncio.sleep(delay)
                    continue
                raise RuntimeError(f'AGS {action} failed: {type(exc).__name__}: {exc}') from exc

        data = json.loads(response.to_json_string())
        logger.info('AGS API response: %s body=%s', action, _redact_for_log(data))
        return data

    async def _list_tools(self) -> list[dict[str, Any]]:
        tools: list[dict[str, Any]] = []
        offset = 0
        limit = 100

        while True:
            response = await self._call_api('DescribeSandboxToolList', {'Offset': offset, 'Limit': limit})
            batch = response.get('SandboxToolSet') or []
            tools.extend(batch)
            total = response.get('TotalCount') or len(tools)
            if len(batch) < limit or len(tools) >= total:
                return tools
            offset += len(batch)

    async def _find_tool(self, *, tool_id: str | None = None, tool_name: str | None = None) -> dict[str, Any] | None:
        tools = await self._list_tools()
        for tool in tools:
            if tool_id is not None and tool.get('ToolId') == tool_id:
                return tool
            if tool_name is not None and tool.get('ToolName') == tool_name:
                return tool
        return None

    async def _wait_for_tool_active(self, tool_id: str) -> dict[str, Any]:
        deadline = asyncio.get_running_loop().time() + self.tool_ready_timeout
        last_status = None

        while True:
            tool = await self._find_tool(tool_id=tool_id)
            if tool is not None:
                status = tool.get('Status')
                last_status = status
                if status == 'ACTIVE':
                    return tool
                if status and ('FAIL' in status or status in {'ERROR', 'INACTIVE'}):
                    raise RuntimeError(
                        f'AGS tool {tool.get("ToolName") or tool_id} entered status {status}: '
                        f'{tool.get("StatusReason") or ""}'
                    )

            if asyncio.get_running_loop().time() >= deadline:
                raise TimeoutError(f'AGS tool {tool_id} did not become ACTIVE, last status={last_status}')
            await asyncio.sleep(self.poll_interval)

    def _build_tool_name(self, name: str, merged: dict[str, Any]) -> str:
        if merged.get('tool_name'):
            return str(merged['tool_name'])

        runtime_kind = merged.get('runtime_kind') or 'code'
        base = f'ags-{runtime_kind}-{name}'
        if merged.get('delete_tool_on_release'):
            return f'{base}-{uuid.uuid4().hex[:8]}'
        return base

    def _infer_launch_mode(self, merged: dict[str, Any]) -> str:
        if merged.get('launch_mode'):
            return str(merged['launch_mode'])

        interface_mode = self._infer_interface_mode(merged)
        tool_type = merged.get('tool_type')
        if tool_type == 'custom':
            return 'native' if interface_mode == 'raw' else 'nexus_server'

        runtime_kind = merged.get('runtime_kind')
        if runtime_kind in {'browser', 'osworld', 'waa'}:
            return 'native'
        if interface_mode == 'raw' and runtime_kind in {'code', 'custom'}:
            return 'native'
        # A code-interpreter template is a pre-built native service (/execute on
        # 49999) with no Nexus server, so template provisioning must use native.
        # control_plane keeps defaulting to nexus_server, where runtime_kind=code
        # legitimately runs a full Nexus runtime server (-> RemoteRuntime).
        if runtime_kind == 'code' and self._infer_provision_mode(merged) == 'template':
            return 'native'

        return 'native' if tool_type in {'browser', 'code-interpreter'} else 'nexus_server'

    def _infer_tool_type(self, merged: dict[str, Any], launch_mode: str) -> str:
        if merged.get('tool_type'):
            return str(merged['tool_type'])

        if launch_mode == 'nexus_server':
            return 'custom'

        runtime_kind = merged.get('runtime_kind') or 'code'
        if runtime_kind == 'browser':
            return 'browser'
        if runtime_kind == 'code':
            return 'code-interpreter'
        if runtime_kind == 'osworld':
            return 'osworld'
        if runtime_kind == 'waa':
            return 'waa'
        return 'custom'

    def _infer_runtime_kind(self, merged: dict[str, Any], tool_type: str) -> str:
        if merged.get('runtime_kind'):
            return str(merged['runtime_kind'])
        if tool_type == 'custom' and self._infer_interface_mode(merged) == 'raw':
            return 'custom'
        if tool_type == 'browser':
            return 'browser'
        if tool_type == 'code-interpreter':
            return 'code'
        if tool_type == 'osworld':
            return 'osworld'
        if tool_type == 'waa':
            return 'waa'
        return 'code'

    def _default_tool_strategy(self, merged: dict[str, Any], tool_type: str) -> str:
        if merged.get('tool_strategy'):
            return str(merged['tool_strategy'])
        if tool_type in {'browser', 'code-interpreter', 'osworld', 'waa'}:
            return 'ensure_builtin'
        if merged.get('image') or self.image:
            return 'ensure_custom'
        return 'existing'

    def _build_nexus_server_custom_configuration(self, merged: dict[str, Any]) -> dict[str, Any]:
        runtime_port = _infer_runtime_port(merged, self.runtime_port)
        image = merged.get('image') or self.image
        if not image:
            raise ValueError('image is required for AGS custom nexus runtime')

        ports = _normalize_ports(merged.get('ports'))
        if not ports:
            ports = [{'Name': 'runtime', 'Port': runtime_port, 'Protocol': 'TCP'}]

        probe = _normalize_probe(merged.get('probe'))
        if probe is None:
            probe = _normalize_probe({'path': '/health', 'port': runtime_port})

        resources = _normalize_resources(merged.get('resource_spec') or self.resource_spec)
        if resources is None:
            resources = {'CPU': '1', 'Memory': '4Gi'}

        command = merged.get('command') or ['/nix/runtime/nexus/deployment/runtime/start.sh']
        args = merged.get('args')
        if args is None:
            args = ['-p', str(runtime_port), '-H', '0.0.0.0', '-v', merged.get('python_version') or self.python_version]

        return {
            'Image': image,
            'ImageRegistryType': merged.get('image_registry_type') or self.image_registry_type,
            'Command': command,
            'Args': args,
            'Env': _normalize_env(merged.get('env')),
            'Ports': ports,
            'Resources': resources,
            'Probe': probe,
        }

    def _build_custom_configuration(self, merged: dict[str, Any], launch_mode: str) -> dict[str, Any]:
        explicit = merged.get('custom_configuration')
        if explicit:
            normalized = dict(explicit)
            if 'Env' not in normalized and 'env' in merged:
                normalized['Env'] = _normalize_env(merged.get('env'))
            if 'Ports' not in normalized and 'ports' in merged:
                normalized['Ports'] = _normalize_ports(merged.get('ports'))
            if 'Resources' not in normalized and (merged.get('resource_spec') or self.resource_spec):
                normalized['Resources'] = _normalize_resources(merged.get('resource_spec') or self.resource_spec)
            if 'Probe' not in normalized and merged.get('probe'):
                normalized['Probe'] = _normalize_probe(merged.get('probe'))
            return normalized

        if launch_mode == 'nexus_server':
            return self._build_nexus_server_custom_configuration(merged)

        image = merged.get('image') or self.image
        if not image:
            raise ValueError('image is required for AGS custom tool creation')

        return {
            'Image': image,
            'ImageRegistryType': merged.get('image_registry_type') or self.image_registry_type,
            'Command': merged.get('command'),
            'Args': merged.get('args'),
            'Env': _normalize_env(merged.get('env')),
            'Ports': _normalize_ports(merged.get('ports')),
            'Resources': _normalize_resources(merged.get('resource_spec') or self.resource_spec),
            'Probe': _normalize_probe(merged.get('probe')),
        }

    def _build_tool_create_payload(
        self,
        *,
        tool_name: str,
        tool_type: str,
        merged: dict[str, Any],
        launch_mode: str,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            'ToolName': tool_name,
            'ToolType': tool_type,
            'NetworkConfiguration': {'NetworkMode': merged.get('network_mode') or self.network_mode},
            'ClientToken': uuid.uuid4().hex,
        }

        description = merged.get('description')
        if description is not None:
            payload['Description'] = description

        default_timeout = merged.get('default_timeout')
        if default_timeout is not None:
            payload['DefaultTimeout'] = int(default_timeout)

        tags = _normalize_tags(merged.get('tags'))
        if tags:
            payload['Tags'] = tags

        if tool_type == 'custom':
            role_arn = merged.get('role_arn') or self.role_arn
            if not role_arn:
                raise ValueError('role_arn is required for AGS custom tool creation')
            payload['RoleArn'] = role_arn
            custom_configuration = self._build_custom_configuration(merged, launch_mode)
            self._validate_custom_configuration(custom_configuration, launch_mode=launch_mode)
            payload['CustomConfiguration'] = custom_configuration

        return payload

    def _validate_tool_selection(self, *, launch_mode: str, tool_type: str, strategy: str) -> None:
        if tool_type not in {'browser', 'code-interpreter', 'custom', 'osworld', 'waa'}:
            raise ValueError(f'Unsupported AGS tool_type: {tool_type!r}')

        if launch_mode == 'nexus_server' and tool_type != 'custom':
            raise ValueError('AGS launch_mode="nexus_server" requires tool_type="custom"')

        if strategy == 'ensure_builtin' and tool_type not in {'browser', 'code-interpreter', 'osworld', 'waa'}:
            raise ValueError(
                'AGS tool_strategy="ensure_builtin" only supports browser/code-interpreter/osworld/waa tools'
            )

        if strategy == 'ensure_custom' and tool_type != 'custom':
            raise ValueError('AGS tool_strategy="ensure_custom" only supports custom tools')

    def _validate_interface_selection(self, *, launch_mode: str, interface_mode: str) -> None:
        if interface_mode == 'raw' and launch_mode != 'native':
            raise ValueError('AGS interface_mode="raw" requires launch_mode="native"')

    def _validate_custom_configuration(self, custom_configuration: dict[str, Any], *, launch_mode: str) -> None:
        if not custom_configuration.get('Image'):
            raise ValueError('AGS custom tool creation requires CustomConfiguration.Image')

        probe = custom_configuration.get('Probe')
        ports = custom_configuration.get('Ports')

        if launch_mode != 'nexus_server':
            if not custom_configuration.get('Command'):
                raise ValueError('AGS native custom runtime requires Command before calling CreateSandboxTool')
            if not probe:
                raise ValueError('AGS native custom runtime requires Probe before calling CreateSandboxTool')
            if not ports:
                raise ValueError('AGS native custom runtime requires Ports before calling CreateSandboxTool')

        if not isinstance(probe, dict):
            return

        ready_timeout_ms = probe.get('ReadyTimeoutMs')
        if ready_timeout_ms is not None and int(ready_timeout_ms) > 30000:
            raise ValueError('AGS Probe.ReadyTimeoutMs must be <= 30000')

        failure_threshold = probe.get('FailureThreshold')
        if failure_threshold is not None and int(failure_threshold) > 100:
            raise ValueError('AGS Probe.FailureThreshold must be <= 100')

    async def _ensure_tool(self, name: str, merged: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        launch_mode = self._infer_launch_mode(merged)
        interface_mode = self._infer_interface_mode(merged)
        tool_type = self._infer_tool_type(merged, launch_mode)
        tool_name = self._build_tool_name(name, merged)
        strategy = self._default_tool_strategy(merged, tool_type)
        self._validate_interface_selection(launch_mode=launch_mode, interface_mode=interface_mode)
        self._validate_tool_selection(launch_mode=launch_mode, tool_type=tool_type, strategy=strategy)

        existing = None
        if merged.get('tool_id'):
            existing = await self._find_tool(tool_id=str(merged['tool_id']))
        if existing is None:
            existing = await self._find_tool(tool_name=tool_name)

        if existing is not None:
            if tool_type and existing.get('ToolType') != tool_type:
                raise ValueError(
                    f'AGS tool {tool_name!r} already exists with type {existing.get("ToolType")!r}, '
                    f'expected {tool_type!r}'
                )
            tool = await self._wait_for_tool_active(existing['ToolId'])
            self._tool_ids[name] = tool['ToolId']
            return tool, False

        if strategy == 'existing':
            raise ValueError(f'AGS tool {tool_name!r} not found. Set tool_strategy to ensure_* or create it first.')

        if strategy not in {'ensure_builtin', 'ensure_custom'}:
            raise ValueError(f'Unsupported AGS tool_strategy: {strategy!r}')

        payload = self._build_tool_create_payload(
            tool_name=tool_name,
            tool_type=tool_type,
            merged=merged,
            launch_mode=launch_mode,
        )
        response = await self._call_api('CreateSandboxTool', payload)
        tool_id = response.get('ToolId')
        if not tool_id:
            raise RuntimeError(f'AGS CreateSandboxTool did not return ToolId: {response}')

        tool = await self._wait_for_tool_active(tool_id)
        self._tool_ids[name] = tool_id
        return tool, True

    async def _start_instance(self, tool: dict[str, Any], merged: dict[str, Any]) -> dict[str, Any]:
        payload: dict[str, Any] = {
            'ToolId': tool['ToolId'],
            'ClientToken': uuid.uuid4().hex,
        }

        auth_mode = merged.get('auth_mode', self.auth_mode)
        if auth_mode:
            payload['AuthMode'] = auth_mode

        timeout = merged.get('timeout')
        if timeout is not None:
            if isinstance(timeout, (int, float)):
                payload['Timeout'] = f'{int(timeout)}s'
            else:
                payload['Timeout'] = str(timeout)

        start_custom_configuration = merged.get('start_custom_configuration')
        if start_custom_configuration:
            payload['CustomConfiguration'] = start_custom_configuration

        response = await self._call_api('StartSandboxInstance', payload)
        instance = response.get('Instance')
        if not isinstance(instance, dict) or not instance.get('InstanceId'):
            raise RuntimeError(f'AGS StartSandboxInstance did not return Instance: {response}')
        return instance

    async def _acquire_token(self, instance_id: str) -> str:
        response = await self._call_api('AcquireSandboxInstanceToken', {'InstanceId': instance_id})
        token = response.get('Token')
        if not token:
            raise RuntimeError(f'AGS AcquireSandboxInstanceToken did not return Token: {response}')
        return token

    async def _describe_running_instance(self, instance_id: str) -> dict[str, Any]:
        response = await self._call_api(
            'DescribeSandboxInstanceList',
            {'InstanceIds': [instance_id], 'Offset': 0, 'Limit': 1},
        )
        instances = response.get('InstanceSet') or []
        instance = next((item for item in instances if item.get('InstanceId') == instance_id), None)
        if instance is None:
            raise InvalidPersistentRuntimeError(f'AGS instance {instance_id} was not found')
        status = str(instance.get('Status') or '').upper()
        if status != 'RUNNING':
            raise InvalidPersistentRuntimeError(
                f'AGS instance {instance_id} is not running: status={status or "UNKNOWN"}'
            )
        return instance

    async def _acquire_existing_runtime(self, name: str, merged: dict[str, Any], instance_id: str) -> Runtime:
        instance = await self._describe_running_instance(instance_id)
        auth_mode = merged.get('auth_mode', self.auth_mode)
        token = '' if auth_mode in {'PUBLIC', 'NONE'} else await self._acquire_token(instance_id)
        tool = {
            'ToolId': instance.get('ToolId'),
            'ToolName': instance.get('ToolName'),
            'ToolType': merged.get('tool_type'),
        }
        handle = AGSInstanceHandle(
            instance_id=instance_id,
            access_token=token,
            domain=self._get_runtime_domain(merged),
            tool_id=instance.get('ToolId'),
            tool_name=instance.get('ToolName'),
            tool_type=merged.get('tool_type'),
            metadata={'instance': instance, 'reused': True},
        )
        runtime = await self._create_runtime(handle, merged, tool)
        self._instance_ids[name] = instance_id
        self._external_instance_names.add(name)
        logger.info('Reused AGS runtime %s: instance_id=%s', name, instance_id)
        return runtime

    async def _wait_for_custom_runtime_ready(self, handle: AGSInstanceHandle, merged: dict[str, Any]) -> None:
        runtime_port = _infer_runtime_port(merged, self.runtime_port)
        health_url = handle.url(runtime_port, '/health')
        timeout = aiohttp.ClientTimeout(total=5)
        deadline = asyncio.get_running_loop().time() + self.instance_ready_timeout
        last_error = ''

        while True:
            try:
                response = await http_client.get(health_url, timeout=timeout, headers=handle.headers)
                if response.status_code == 200:
                    return
                last_error = f'HTTP {response.status_code}: {response.text[:200]}'
            except Exception as exc:
                last_error = f'{type(exc).__name__}: {exc}'

            if asyncio.get_running_loop().time() >= deadline:
                raise TimeoutError(f'AGS custom runtime not ready at {health_url}: {last_error}')
            await asyncio.sleep(self.poll_interval)

    async def _wait_for_http_endpoint_ready(self, handle: AGSInstanceHandle, port: int, path: str, label: str) -> None:
        ready_url = handle.url(port, path)
        timeout = aiohttp.ClientTimeout(total=5)
        deadline = asyncio.get_running_loop().time() + self.instance_ready_timeout
        last_error = ''

        while True:
            try:
                response = await http_client.get(ready_url, timeout=timeout, headers=handle.headers)
                if response.status_code == 200:
                    return
                last_error = f'HTTP {response.status_code}: {response.text[:200]}'
            except Exception as exc:
                last_error = f'{type(exc).__name__}: {exc}'

            if asyncio.get_running_loop().time() >= deadline:
                raise TimeoutError(f'AGS {label} runtime not ready at {ready_url}: {last_error}')
            await asyncio.sleep(self.poll_interval)

    async def _wait_for_raw_runtime_ready(self, handle: AGSInstanceHandle, merged: dict[str, Any]) -> None:
        probe = _normalize_probe(merged.get('probe'))
        if not isinstance(probe, dict):
            return

        http_get = probe.get('HttpGet')
        if not isinstance(http_get, dict):
            return

        port = http_get.get('Port')
        if port is None:
            return

        path = str(http_get.get('Path') or '/')
        await self._wait_for_http_endpoint_ready(handle, int(port), path, 'raw')

    def _populate_raw_port_mappings(self, runtime: AGSRawRuntime, merged: dict[str, Any]) -> None:
        ports = _normalize_ports(merged.get('ports')) or []
        for item in ports:
            runtime.port_mappings[int(item['Port'])] = runtime.handle.url(int(item['Port']))

    async def _wait_for_browser_ready(self, handle: AGSInstanceHandle, port: int) -> None:
        status_url = handle.url(port, '/cdp/json/version')
        timeout = aiohttp.ClientTimeout(total=5)
        deadline = asyncio.get_running_loop().time() + self.instance_ready_timeout
        last_error = ''

        while True:
            try:
                response = await http_client.get(status_url, timeout=timeout, headers=handle.headers)
                if response.status_code == 200:
                    return
                last_error = f'HTTP {response.status_code}: {response.text[:200]}'
            except Exception as exc:
                last_error = f'{type(exc).__name__}: {exc}'

            if asyncio.get_running_loop().time() >= deadline:
                raise TimeoutError(f'AGS browser runtime not ready at {status_url}: {last_error}')
            await asyncio.sleep(self.poll_interval)

    async def _wait_for_osworld_ready(self, handle: AGSInstanceHandle, server_port: int) -> None:
        screenshot_url = handle.url(server_port, '/screenshot')
        timeout = aiohttp.ClientTimeout(total=10)
        deadline = asyncio.get_running_loop().time() + self.instance_ready_timeout
        last_error = ''

        while True:
            try:
                response = await http_client.get(screenshot_url, timeout=timeout, headers=handle.headers)
                if response.status_code == 200 and response.content:
                    return
                last_error = f'HTTP {response.status_code}: {response.text[:200]}'
            except Exception as exc:
                last_error = f'{type(exc).__name__}: {exc}'

            if asyncio.get_running_loop().time() >= deadline:
                raise TimeoutError(f'AGS OSWorld runtime not ready at {screenshot_url}: {last_error}')
            await asyncio.sleep(self.poll_interval)

    async def _wait_for_code_interpreter_ready(self, handle: AGSInstanceHandle, port: int) -> None:
        execute_url = handle.url(port, '/execute')
        timeout = aiohttp.ClientTimeout(total=10)
        deadline = asyncio.get_running_loop().time() + self.instance_ready_timeout
        last_error = ''

        while True:
            try:
                response = await http_client.post(
                    execute_url,
                    json={'code': 'print("ready")', 'language': 'python'},
                    timeout=timeout,
                    headers=handle.headers,
                )
                if response.status_code == 200:
                    return
                last_error = f'HTTP {response.status_code}: {response.text[:200]}'
            except Exception as exc:
                last_error = f'{type(exc).__name__}: {exc}'

            if asyncio.get_running_loop().time() >= deadline:
                raise TimeoutError(f'AGS code runtime not ready at {execute_url}: {last_error}')
            await asyncio.sleep(self.poll_interval)

    async def _wait_for_waa_ready(self, handle: AGSInstanceHandle, server_port: int) -> None:
        probe_url = handle.url(server_port, '/probe')
        timeout = aiohttp.ClientTimeout(total=10)
        deadline = asyncio.get_running_loop().time() + self.instance_ready_timeout
        last_error = ''

        while True:
            try:
                response = await http_client.get(probe_url, timeout=timeout, headers=handle.headers)
                if response.status_code == 200 and response.content:
                    return
                last_error = f'HTTP {response.status_code}: {response.text[:200]}'
            except Exception as exc:
                last_error = f'{type(exc).__name__}: {exc}'

            if asyncio.get_running_loop().time() >= deadline:
                raise TimeoutError(f'AGS WAA runtime not ready at {probe_url}: {last_error}')
            await asyncio.sleep(self.poll_interval)

    def _create_nexus_runtime(self, handle: AGSInstanceHandle, merged: dict[str, Any]) -> Runtime:
        base_url = handle.url(_infer_runtime_port(merged, self.runtime_port))
        runtime_cls = SocketIoRuntime if self.rpc_protocol == 'socketio' else HttpRuntime
        runtime = runtime_cls(
            base_url=base_url,
            request_timeout=self.request_timeout,
            resource_spec=merged.get('resource_spec') or self.resource_spec,
            enable_runtime_logs=merged.get('enable_runtime_logs', self.enable_runtime_logs),
            auto_connect_ws=merged.get('auto_connect_ws', self.auto_connect_ws),
            socketio_request_timeout=merged.get('socketio_request_timeout', self.socketio_request_timeout),
            default_headers=handle.headers,
        )

        ports = _normalize_ports(merged.get('ports')) or []
        for item in ports:
            runtime.port_mappings[int(item['Port'])] = handle.url(int(item['Port']))
        return runtime

    async def _create_runtime(self, handle: AGSInstanceHandle, merged: dict[str, Any], tool: dict[str, Any]) -> Runtime:
        launch_mode = self._infer_launch_mode(merged)
        interface_mode = self._infer_interface_mode(merged)
        runtime_kind = self._infer_runtime_kind(merged, tool.get('ToolType') or '')
        self._validate_interface_selection(launch_mode=launch_mode, interface_mode=interface_mode)

        if launch_mode == 'nexus_server':
            await self._wait_for_custom_runtime_ready(handle, merged)
            return self._create_nexus_runtime(handle, merged)

        if interface_mode == 'raw':
            if runtime_kind == 'browser':
                browser_port = int(merged.get('browser_port') or self.BUILTIN_BROWSER_PORT)
                await self._wait_for_browser_ready(handle, browser_port)
                return AGSRawBrowserRuntime(
                    handle,
                    browser_port=browser_port,
                    request_timeout=self.request_timeout,
                    resource_spec=merged.get('resource_spec') or self.resource_spec,
                )

            if runtime_kind == 'osworld':
                server_port = int(merged.get('server_port') or 5000)
                await self._wait_for_osworld_ready(handle, server_port)
                return AGSRawOSWorldRuntime(
                    handle,
                    server_port=server_port,
                    chromium_port=int(merged.get('chromium_port') or 9222),
                    vnc_port=int(merged.get('vnc_port') or 5910),
                    vlc_port=int(merged.get('vlc_port') or 8080),
                    request_timeout=self.request_timeout,
                    resource_spec=merged.get('resource_spec') or self.resource_spec,
                )

            if runtime_kind == 'code':
                code_port = int(merged.get('code_interpreter_port') or self.BUILTIN_CODE_INTERPRETER_PORT)
                await self._wait_for_code_interpreter_ready(handle, code_port)
                return AGSRawCodeInterpreterRuntime(
                    handle,
                    execute_port=code_port,
                    request_timeout=self.request_timeout,
                    resource_spec=merged.get('resource_spec') or self.resource_spec,
                )

            if runtime_kind == 'waa':
                server_port = int(merged.get('server_port') or 5000)
                await self._wait_for_waa_ready(handle, server_port)
                return AGSRawWindowsAgentArenaRuntime(
                    handle,
                    server_port=server_port,
                    chromium_port=int(merged.get('chromium_port') or 9222),
                    vnc_port=int(merged.get('vnc_port') or 8006),
                    request_timeout=self.request_timeout,
                    resource_spec=merged.get('resource_spec') or self.resource_spec,
                )

            await self._wait_for_raw_runtime_ready(handle, merged)
            raw_runtime = AGSRawRuntime(
                handle,
                kind='custom',
                default_port=_infer_runtime_port(merged, self.runtime_port),
                request_timeout=self.request_timeout,
                resource_spec=merged.get('resource_spec') or self.resource_spec,
            )
            self._populate_raw_port_mappings(raw_runtime, merged)
            return raw_runtime

        if runtime_kind == 'browser':
            browser_port = int(merged.get('browser_port') or self.BUILTIN_BROWSER_PORT)
            await self._wait_for_browser_ready(handle, browser_port)
            runtime = AGSBrowserRuntime(
                handle,
                cdp_url=handle.url(browser_port, '/cdp'),
                request_timeout=self.request_timeout,
                resource_spec=merged.get('resource_spec') or self.resource_spec,
            )
            runtime.port_mappings[browser_port] = handle.url(browser_port)
            return runtime

        if runtime_kind == 'osworld':
            server_port = int(merged.get('server_port') or 5000)
            await self._wait_for_osworld_ready(handle, server_port)
            return AGSOSWorldRuntime(
                handle,
                server_port=server_port,
                chromium_port=int(merged.get('chromium_port') or 9222),
                vnc_port=int(merged.get('vnc_port') or 5910),
                vlc_port=int(merged.get('vlc_port') or 8080),
                request_timeout=self.request_timeout,
                resource_spec=merged.get('resource_spec') or self.resource_spec,
            )

        if runtime_kind == 'waa':
            server_port = int(merged.get('server_port') or 5000)
            await self._wait_for_waa_ready(handle, server_port)
            return AGSWindowsAgentArenaRuntime(
                handle,
                server_port=server_port,
                chromium_port=int(merged.get('chromium_port') or 9222),
                vnc_port=int(merged.get('vnc_port') or 8006),
                request_timeout=self.request_timeout,
                resource_spec=merged.get('resource_spec') or self.resource_spec,
            )

        code_port = int(merged.get('code_interpreter_port') or self.BUILTIN_CODE_INTERPRETER_PORT)
        await self._wait_for_code_interpreter_ready(handle, code_port)
        runtime = AGSCodeInterpreterRuntime(
            handle,
            execute_port=code_port,
            request_timeout=self.request_timeout,
            resource_spec=merged.get('resource_spec') or self.resource_spec,
        )
        runtime.port_mappings[code_port] = handle.url(code_port)
        return runtime

    async def _acquire_template_runtime(self, name: str, merged: dict[str, Any]) -> Runtime:
        sandbox = await self._create_template_sandbox(merged)
        instance_id = getattr(sandbox, 'sandbox_id', None)
        if not instance_id:
            raise RuntimeError('AGS template sandbox did not expose sandbox_id')

        access_token = self._extract_template_access_token(sandbox)
        launch_mode = self._infer_launch_mode(merged)
        interface_mode = self._infer_interface_mode(merged)
        tool_type = self._infer_tool_type(merged, launch_mode)
        template_name = self._get_template_name(merged)
        self._validate_interface_selection(launch_mode=launch_mode, interface_mode=interface_mode)
        handle = AGSInstanceHandle(
            instance_id=str(instance_id),
            access_token=access_token,
            domain=self._get_runtime_domain(merged),
            tool_name=template_name,
            tool_type=tool_type,
            metadata={'provision_mode': 'template', 'template': template_name},
        )

        try:
            runtime = await self._create_runtime(
                handle,
                merged,
                {'ToolName': template_name, 'ToolType': tool_type},
            )
        except Exception:
            with contextlib.suppress(Exception):
                await asyncio.to_thread(sandbox.kill)
            raise

        self._template_sandboxes[name] = sandbox
        self._instance_ids[name] = str(instance_id)
        logger.info('AGS template runtime %s ready: instance_id=%s template=%s', name, instance_id, template_name)
        return runtime

    async def acquire(self, name: str, **params: Any) -> Runtime:
        if name not in self._acquire_locks:
            self._acquire_locks[name] = asyncio.Lock()

        async with self._acquire_locks[name]:
            if name in self._active_runtimes:
                return self._active_runtimes[name]

            merged = self.get_params(name, **params)
            runtime_credential = merged.get('runtime_credential')
            instance_id = ''
            if runtime_credential is not None:
                if not isinstance(runtime_credential, Mapping):
                    raise InvalidPersistentRuntimeError('AGS runtime_credential must be a mapping')
                instance_id = str(runtime_credential.get('instance_id') or '').strip()
                if not instance_id:
                    raise InvalidPersistentRuntimeError('AGS runtime_credential requires instance_id')
            elif merged.get('instance_id') is not None:
                raise ValueError('AGS instance_id is not supported; use runtime_credential.instance_id')
            if instance_id:
                runtime = await self._acquire_existing_runtime(name, merged, instance_id)
                self._active_runtimes[name] = runtime
                return runtime

            provision_mode = self._infer_provision_mode(merged)
            if provision_mode == 'template':
                runtime = await self._acquire_template_runtime(name, merged)
                self._active_runtimes[name] = runtime
                return runtime

            tool, created_tool = await self._ensure_tool(name, merged)
            instance = await self._start_instance(tool, merged)
            self._instance_ids[name] = instance['InstanceId']
            self._tool_ids[name] = tool['ToolId']

            if created_tool and merged.get('delete_tool_on_release'):
                self._tools_to_delete[name] = tool['ToolId']

            try:
                auth_mode = merged.get('auth_mode', self.auth_mode)
                token = '' if auth_mode in {'PUBLIC', 'NONE'} else await self._acquire_token(instance['InstanceId'])

                handle = AGSInstanceHandle(
                    instance_id=instance['InstanceId'],
                    access_token=token,
                    domain=self.domain,
                    tool_id=tool.get('ToolId'),
                    tool_name=tool.get('ToolName'),
                    tool_type=tool.get('ToolType'),
                    metadata={'instance': instance, 'tool': tool},
                )
                runtime = await self._create_runtime(handle, merged, tool)
            except BaseException:
                with contextlib.suppress(BaseException):
                    await asyncio.shield(self._release_impl(name))
                raise

            self._active_runtimes[name] = runtime

            logger.info('AGS runtime %s ready: instance_id=%s tool=%s', name, instance['InstanceId'], tool['ToolName'])
            return runtime

    async def _close_runtime(self, runtime: Runtime) -> None:
        close_method = getattr(runtime, 'close', None)
        if callable(close_method):
            result = close_method()
            if inspect.isawaitable(result):
                await result

        with contextlib.suppress(Exception):
            await runtime.ws_disconnect()

    async def _stop_instance_with_retries(self, name: str, instance_id: str) -> None:
        # 停实例失败会直接造成 AGS 残留和后续容量噪声，因此释放阶段也给 StopSandboxInstance
        # 一个很短的重试窗口；仍失败则抛出，方便外层/日志明确暴露清理问题。
        last_error: BaseException | None = None
        for attempt in range(1, 4):
            try:
                await self._call_api('StopSandboxInstance', {'InstanceId': instance_id})
                return
            except Exception as exc:
                last_error = exc
                if attempt < 3:
                    logger.warning(
                        'Failed to stop AGS instance for runtime %s on attempt %s/3; retrying: instance_id=%s error=%s: %s',
                        name,
                        attempt,
                        instance_id,
                        type(exc).__name__,
                        exc,
                    )
                    await asyncio.sleep(min(2.0 * attempt, 5.0))
                else:
                    logger.warning(
                        'Failed to stop AGS instance for runtime %s on final attempt 3/3: instance_id=%s',
                        name,
                        instance_id,
                        exc_info=True,
                    )

        raise RuntimeError(f'Failed to stop AGS instance {instance_id} for runtime {name}') from last_error

    async def _release_impl(self, name: str) -> None:
        runtime = self._active_runtimes.pop(name, None)
        if runtime is not None:
            await self._close_runtime(runtime)

        if name in self._external_instance_names:
            self._external_instance_names.discard(name)
            self._instance_ids.pop(name, None)
            self._tool_ids.pop(name, None)
            self._tools_to_delete.pop(name, None)
            logger.info('Detached reused AGS runtime %s without stopping its instance', name)
            return

        sandbox = self._template_sandboxes.pop(name, None)
        if sandbox is not None:
            with contextlib.suppress(Exception):
                await asyncio.to_thread(sandbox.kill)
            self._instance_ids.pop(name, None)
            self._tool_ids.pop(name, None)
            self._tools_to_delete.pop(name, None)
            logger.info('Released AGS template runtime %s', name)
            return

        instance_id = self._instance_ids.pop(name, None)
        if instance_id:
            await self._stop_instance_with_retries(name, instance_id)

        tool_id = self._tools_to_delete.pop(name, None)
        if tool_id:
            with contextlib.suppress(Exception):
                await self._call_api('DeleteSandboxTool', {'ToolId': tool_id})

        self._tool_ids.pop(name, None)
        logger.info('Released AGS runtime %s', name)


__all__ = ['AGSRuntimeProvider']
