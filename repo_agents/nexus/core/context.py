"""Execution context for nexus framework.

ExecutionContext provides a unified way to access runtime dependencies,
shared state, and task metadata throughout the Task → Agent → Tool chain.

IMPORTANT: ExecutionContext MUST be used with `async with` to ensure
proper cleanup of runtime resources.
"""

from __future__ import annotations

import asyncio
import inspect
import logging

from contextvars import ContextVar, Token
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from nexus.core.openai_client import OpenAICompatibleClient
from nexus.core.runtime_shared import RuntimeSharedStore
from nexus.core.workflow import WorkflowState, _active_workflow_node
from nexus.dataset.task_files import TaskFiles
from nexus.runtime import InvalidPersistentRuntimeError, Runtime, RuntimeProvider
from nexus.runtime.accounts import AccountPoolManager, AcquiredAccount
from nexus.utils.event_loop import get_current_task_cancellation_count


logger = logging.getLogger(__name__)

# Global ContextVar for coroutine-safe context propagation
_current_context: ContextVar[ExecutionContext | None] = ContextVar('execution_context', default=None)


class RuntimePoolConfig(BaseModel):
    """Account-backed allocation settings for a persistent runtime."""

    model_config = ConfigDict(extra='forbid')

    name: str = Field(min_length=1)
    lease_ttl_seconds: int = Field(default=3600, gt=0)
    tags: list[str] | None = None


class ExecutionContext(BaseModel):
    """Unified execution context for Task/Agent/Tool.

    Provides access to:
    - Runtime instances (lazy-loaded via RuntimeProvider)
    - Account pool (acquire/release accounts with heartbeat)
    - OpenAI-compatible clients
    - Task metadata and parameters
    - Shared state between tools

    MUST be used with `async with` to ensure runtime cleanup:

        # In runner
        ctx = ExecutionContext(
            runtime_provider=provider,
            openai_clients={'main': client, 'grm': grm_client},
            task_name='swe_bench',
            task_config={'instance_id': '123'},
            run_id='run-001',  # Host-unique identifier
            account_pool_endpoint='http://nexus-pool.production.polaris:8080',
        )

        # Pre-configure runtime params (optional)
        ctx.runtime_provider.configure('main', image='python:3.11')

        async with ctx:
            await task_fn(task_input)
        # All runtimes and accounts are automatically released here

        # In task/agent/tool
        ctx = require_context()
        runtime = await ctx.get_runtime()  # lazy load 'main' runtime
        runtime = await ctx.get_runtime('main', memory='4G')  # with extra params
        client = ctx.get_openai_client()  # get 'main' client

        # Account pool（管理后台 https://nexus-pool.woa.com/ ）
        account = await ctx.get_account('github')
        account = await ctx.get_account('gui-sandbox', tags=['chrome', 'vscode'])
        token = account.tokens['access_token']
    """

    model_config = {'arbitrary_types_allowed': True}

    # === Runtime Management ===
    runtime_provider: RuntimeProvider | None = None
    runtime_configs: dict[str, dict[str, Any]] = {}  # Per-runtime configs (may contain worker_venv)

    # === OpenAI Clients ===
    openai_clients: dict[str, OpenAICompatibleClient] = {}

    # === Task Metadata ===
    task_name: str | None = None
    task_config: dict[str, Any] = {}
    run_id: str | None = None  # Host-unique identifier for this run

    # === Agent Metadata ===
    agent_name: str | None = None

    # === Tool Configuration ===
    tool_configs: dict[str, Any] = {}  # Tool-specific configurations (e.g., API keys)

    # === Dataset Task Files ===
    task_files: TaskFiles | None = None

    # === Shared State (for tool coordination) ===
    tool_state: dict[str, Any] = {}

    # === Resumable Workflow State ===
    workflow_states: dict[str, WorkflowState] = {}

    # === Context Management ===
    compaction_manager: Any = None  # CompactionManager | None (Any to avoid core→extensions dep)

    # === Dynamic MCP Tools ===
    mcp_tools: dict[str, Any] = {}

    # === Tracing ===
    trace_id: str | None = None  # Globally unique identifier (UUID-like) for distributed tracing

    # === Timeout Control ===
    timeout_handle: Any = None  # asyncio.Timeout object for reschedule support

    # === Account Pool ===
    account_pool_endpoint: str | None = None  # Account pool server URL; None = env var or polaris default
    account_pool_ttl: int = 60  # Default lease TTL for acquired accounts

    # === Private State ===
    _runtimes: dict[str, Runtime] = PrivateAttr(default_factory=dict)
    _runtime_accounts: dict[str, AcquiredAccount] = PrivateAttr(default_factory=dict)
    _mcp_manager: Any = PrivateAttr(default=None)
    _account_pool_manager: AccountPoolManager | None = PrivateAttr(default=None)
    _context_token: Token | None = PrivateAttr(default=None)
    _runtime_shared: RuntimeSharedStore = PrivateAttr(default_factory=RuntimeSharedStore)
    _runtime_shared_unsubs: dict[int, list[Any]] = PrivateAttr(default_factory=dict)
    _task_files_runtime_unsubs: dict[int, list[Any]] = PrivateAttr(default_factory=dict)

    # === Runtime Access ===

    @property
    def workflow(self) -> WorkflowState:
        """Return the workflow state active in the current coroutine."""
        return self._get_active_workflow_state()

    def _get_active_workflow_state(self) -> WorkflowState:
        active = _active_workflow_node.get()
        if active is None:
            raise RuntimeError('No workflow node is active')
        return self.workflow_states[active.scope]

    @property
    def runtime_shared(self) -> RuntimeSharedStore:
        """Return values shared with Runtime processes for this execution."""

        return self._runtime_shared

    async def get_runtime(self, name: str = 'main', **params: Any) -> Runtime:
        """Get a runtime instance by name, creating it lazily if needed.

        On first call for a name, acquires a new runtime with merged params:
        - Pre-configured params via runtime_provider.configure()
        - Params passed to this method
        - Automatically includes run_id if available

        If ``runtime_configs[name]`` contains a ``worker_venv`` key, an artifact
        wheel is built (with caching) and a worker venv is created automatically.
        All subsequent ``invoke()`` / ``exec()`` calls on the returned runtime
        will be routed to that venv via ``runtime.default_venv_id``.

        Subsequent calls return the cached runtime (params ignored).

        Args:
            name: Runtime name. Defaults to 'main'.
            **params: Additional params for runtime acquisition (first call only)

        Returns:
            Runtime instance

        Raises:
            RuntimeError: If no runtime provider is configured
        """
        if name in self._runtimes:
            return self._runtimes[name]

        if self.runtime_provider is None:
            raise RuntimeError('No runtime provider configured')

        # Automatically inject run_id if available
        if self.run_id is not None and 'run_id' not in params:
            params['run_id'] = self.run_id

        runtime_pool = self._get_runtime_pool_config(name, params)
        account: AcquiredAccount | None = None
        if runtime_pool is not None:
            account = await self.get_account(
                runtime_pool.name,
                ttl_seconds=runtime_pool.lease_ttl_seconds,
                tags=runtime_pool.tags,
            )
            params['runtime_credential'] = account.credentials

        logger.info(f"Acquiring runtime '{name}'...")
        try:
            runtime = await self.runtime_provider.acquire(name, **params)

            # Auto-create worker venv if configured (true/{dict} → create, false/None → skip)
            venv_raw = self.runtime_configs.get(name, {}).get('worker_venv')
            if venv_raw is not None and venv_raw is not False:
                runtime = await self._setup_worker_venv(runtime, venv_raw)

            self._attach_runtime_shared(runtime)
            self._attach_task_files_runtime(runtime)
            self._runtimes[name] = runtime
            if account is not None:
                self._runtime_accounts[name] = account
        except InvalidPersistentRuntimeError as exc:
            if account is not None:
                try:
                    await account.invalidate(str(exc))
                except Exception:
                    logger.warning(
                        "Failed to invalidate persistent runtime account for '%s' after acquisition failed",
                        name,
                        exc_info=True,
                    )
            raise
        except BaseException:
            if account is not None:
                await account.release()
            raise

        logger.info(f"Runtime '{name}' acquired successfully")
        return runtime

    def _get_runtime_pool_config(self, name: str, params: dict[str, Any]) -> RuntimePoolConfig | None:
        raw = params.pop('runtime_pool', None)
        if raw is None:
            raw = self.runtime_configs.get(name, {}).get('runtime_pool')
        if raw is None:
            return None
        return RuntimePoolConfig.model_validate(raw)

    async def release_runtime(self, name: str = 'main', *, force: bool = False) -> bool:
        """Release one named runtime and remove it from this context's cache.

        If release is suppressed or provider cleanup fails, the cached runtime
        is retained so it can be reused or retried during context shutdown.
        """
        if self.runtime_provider is None:
            raise RuntimeError('No runtime provider configured')
        if name not in self._runtimes:
            return False

        logger.info(f"Releasing runtime '{name}'...")
        released = await self.runtime_provider.release(name, force=force)
        if not released:
            logger.info(f"Runtime '{name}' release suppressed; retaining context cache")
            return False
        self._runtimes.pop(name, None)
        account = self._runtime_accounts.pop(name, None)
        if account is not None:
            await account.release()
        logger.info(f"Runtime '{name}' released successfully")
        return True

    async def replace_runtime(self, name: str = 'main', **params: Any) -> Runtime:
        """Release existing runtime and acquire a new one with different parameters.

        Used for switching runtime base image or other configurations mid-execution.
        """
        if self.runtime_provider is None:
            raise RuntimeError('No runtime provider configured')

        if name in self._runtimes:
            logger.info(f"Releasing existing runtime '{name}' before replacement")
            await self.runtime_provider.release(name)
            del self._runtimes[name]

        if self.run_id is not None and 'run_id' not in params:
            params['run_id'] = self.run_id

        from nexus.utils.otel.redaction import redact_exporter_credentials

        logger.info("Acquiring new runtime '%s' with params: %s", name, redact_exporter_credentials(params))
        runtime = await self.runtime_provider.acquire(name, **params)
        venv_raw = self.runtime_configs.get(name, {}).get('worker_venv')
        if venv_raw is not None and venv_raw is not False:
            runtime = await self._setup_worker_venv(runtime, venv_raw)

        self._attach_runtime_shared(runtime)
        self._attach_task_files_runtime(runtime)
        self._runtimes[name] = runtime
        logger.info(
            f"Runtime '{name}' replaced successfully, base_url: {getattr(getattr(runtime, 'config', None), 'base_url', 'N/A')}"
        )
        return runtime

    def get_runtime_id(self, target: str | Runtime = 'main') -> str | None:
        """Return the provider-native runtime ID for an acquired runtime.

        Looks up the runtime through :attr:`_runtimes` and delegates to
        ``runtime_provider.get_runtime_id(name)`` so callers don't have to
        scrape ``runtime.config.base_url`` (which is provider-specific and
        couples call sites to a particular URL format).

        Args:
            target: Either the runtime ``name`` (str) used with
                :meth:`get_runtime` (defaults to ``'main'``), or a
                :class:`Runtime` instance previously returned by
                :meth:`get_runtime`. When a ``Runtime`` is given the matching
                name is recovered from the cached ``_runtimes`` map.

        Returns:
            The native runtime ID (e.g. Gongfeng ``nexus-XXXX``), or ``None``
            when no runtime is registered for ``target``, no runtime provider
            is configured, or the provider does not track an ID.
        """
        if self.runtime_provider is None:
            return None

        if isinstance(target, str):
            name: str | None = target
        else:
            name = next((n for n, rt in self._runtimes.items() if rt is target), None)

        if name is None:
            return None

        return self.runtime_provider.get_runtime_id(name)

    async def _setup_worker_venv(self, runtime: Runtime, venv_raw: dict[str, Any] | bool) -> Runtime:
        """Build artifact, create worker venv, and set default_venv_id on the runtime."""
        import base64

        from nexus.runtime.client.models import WorkerVenvConfig
        from nexus.runtime.client.packaging import build_artifact_cached

        cfg = WorkerVenvConfig.model_validate(venv_raw if isinstance(venv_raw, dict) else {})
        source = Path(cfg.source).resolve()

        logger.info(f'Building artifact from {source} ...')
        whl_path = await build_artifact_cached(source, exclude_patterns=cfg.exclude_patterns or None)
        logger.info(f'Artifact ready: {whl_path.name}')

        whl_data = base64.b64encode(whl_path.read_bytes()).decode()
        artifact_extras = cfg.extras
        venv_spec: dict[str, Any] = {
            'python_version': cfg.python_version,
            'artifacts': [{'name': whl_path.stem, 'whl_data': whl_data, 'extras': artifact_extras}],
            'extra_requirements': cfg.extra_requirements,
            'modules': cfg.modules,
        }
        from nexus.utils.otel.auto import is_enabled
        from nexus.utils.otel.environment import managed_otel_env

        if is_enabled():
            artifact_extras = list(dict.fromkeys((*cfg.extras, 'otel')))
            venv_spec['artifacts'][0]['extras'] = artifact_extras
            venv_spec['env_vars'] = managed_otel_env(
                process_role='runtime-worker',
                service_name='nexus-runtime-worker',
            )
        venv = await runtime.create_venv(venv_spec)

        if venv.state != 'ready':
            raise RuntimeError(f'Worker venv creation failed: {venv.error}')

        logger.info(f'Worker venv ready: {venv.venv_id}')
        runtime.default_venv_id = venv.venv_id
        return runtime

    def _attach_runtime_shared(self, runtime: Runtime) -> None:
        """Serve lazily requested shared values over the existing SIO bridge."""

        if not hasattr(runtime, 'on') or not hasattr(runtime, 'emit'):
            return
        key = id(runtime)
        if key in self._runtime_shared_unsubs:
            return

        async def on_shared_value_request(data: dict[str, Any]) -> None:
            request_id = str(data.get('request_id') or '')
            name = str(data.get('name') or '')
            if not request_id:
                logger.warning('Runtime shared value request missing request_id')
                return
            try:
                value = self._runtime_shared_payload(name)
            except Exception as exc:
                logger.warning('Failed to provide Runtime shared value %r: %s', name, exc)
                await runtime.invoke(
                    'nexus.runtime.context::fail_runtime_shared_value',
                    request_id,
                    f'{exc.__class__.__name__}: {exc}',
                )
                return
            await runtime.invoke('nexus.runtime.context::provide_runtime_shared_value', request_id, value)

        try:
            self._runtime_shared_unsubs[key] = [runtime.on('shared-value:request', on_shared_value_request)]
        except NotImplementedError:
            # In-process runtimes (e.g. LocalRuntime) run in the same process and do not
            # support the SIO event bridge; shared values are served directly, so skip.
            logger.debug(
                'Runtime %s does not support event subscription; skipping shared-value bridge', type(runtime).__name__
            )

    def _runtime_shared_payload(self, name: str) -> dict[str, Any]:
        from nexus.utils.artifact.auth import (
            ARTIFACT_SHARED_VALUE_NAME,
            ArtifactCredentials,
            load_artifact_credentials,
        )
        from nexus.utils.container_registry_auth import (
            RegistryCredentials,
            load_registry_credentials,
            registry_from_shared_value_name,
        )
        from nexus.utils.cos import COS_SHARED_VALUE_NAME, CosCredentials, load_cos_credentials

        if name == ARTIFACT_SHARED_VALUE_NAME:
            credentials = self.runtime_shared.get_optional(ARTIFACT_SHARED_VALUE_NAME, ArtifactCredentials)
            if credentials is None:
                credentials = load_artifact_credentials()
                self.runtime_shared.set(ARTIFACT_SHARED_VALUE_NAME, credentials)
            return credentials.to_shared_payload()
        if name == COS_SHARED_VALUE_NAME:
            credentials = self.runtime_shared.get_optional(COS_SHARED_VALUE_NAME, CosCredentials)
            if credentials is None:
                credentials = load_cos_credentials()
                self.runtime_shared.set(COS_SHARED_VALUE_NAME, credentials)
            return credentials.to_shared_payload()
        registry = registry_from_shared_value_name(name)
        if registry is not None:
            credentials = self.runtime_shared.get_optional(name, RegistryCredentials)
            if credentials is None:
                credentials = load_registry_credentials(registry)
                self.runtime_shared.set(name, credentials)
            return credentials.to_shared_payload()
        raise KeyError(f'Unsupported Runtime shared value: {name}')

    def _attach_task_files_runtime(self, runtime: Runtime) -> None:
        """Attach host-side task_files SIO responders to a runtime client."""

        task_files = self.task_files
        if task_files is None:
            return
        if not hasattr(runtime, 'on') or not hasattr(runtime, 'emit'):
            return
        key = id(runtime)
        if key in self._task_files_runtime_unsubs:
            return

        async def on_session_request(data: dict[str, Any]) -> None:
            request_id = str(data.get('request_id') or '')
            if not request_id:
                logger.warning('task_files session request missing request_id')
                return
            try:
                session = await task_files.create_session(run_id=self.run_id)
                payload = {'ok': True, 'session': session.model_dump(mode='json', exclude_none=True)}
            except Exception as exc:
                logger.warning('Failed to build task_files session: %s', exc)
                payload = {'ok': False, 'error': f'{exc.__class__.__name__}: {exc}'}
            await runtime.emit(f'task_files:session_response:{request_id}', payload)

        async def on_local_file_request(data: dict[str, Any]) -> None:
            request_id = str(data.get('request_id') or '')
            if not request_id:
                logger.warning('task_files local file request missing request_id')
                return
            try:
                op = data.get('op')
                if op == 'list':
                    pattern = str(data.get('pattern') or '**/*')
                    payload = {'ok': True, 'paths': await task_files.list(pattern)}
                elif op in {'materialize', 'download'}:
                    path = str(data.get('path') or '')
                    target = str(data.get('target') or '')
                    if not target:
                        raise ValueError(f'task_files {op} request missing target')
                    content = await task_files.read_bytes(path)
                    result = await runtime.invoke(
                        'nexus.dataset.task_files::runtime_receive_task_file',
                        content,
                        target,
                    )
                    response: dict[str, Any] = result if isinstance(result, dict) else {}
                    payload = {
                        'ok': True,
                        'path': str(response.get('path') or target),
                        'size': response.get('size'),
                    }
                else:
                    raise ValueError(f'Unsupported task_files local file op: {op}')
            except Exception as exc:
                logger.warning('Failed to serve task_files local file request: %s', exc)
                payload = {'ok': False, 'error': f'{exc.__class__.__name__}: {exc}'}
            await runtime.emit(f'task_files:local_file_response:{request_id}', payload)

        try:
            self._task_files_runtime_unsubs[key] = [
                runtime.on('task_files:session_request', on_session_request),
                runtime.on('task_files:local_file_request', on_local_file_request),
            ]
        except NotImplementedError:
            # In-process runtimes (e.g. LocalRuntime) do not support the SIO event bridge;
            # task_files are accessible directly, so skip attaching the responders.
            logger.debug(
                'Runtime %s does not support event subscription; skipping task_files bridge', type(runtime).__name__
            )

    # === OpenAI Client Access ===

    def get_openai_client(self, name: str = 'main') -> OpenAICompatibleClient:
        """Get an OpenAI-compatible client by name.

        Args:
            name: Client name. Defaults to 'main'.

        Returns:
            OpenAICompatibleClient instance

        Raises:
            KeyError: If client name not found
        """
        if name not in self.openai_clients:
            raise KeyError(f"OpenAI client '{name}' not found. Available: {list(self.openai_clients.keys())}")
        return self.openai_clients[name]

    @property
    def openai_client(self) -> OpenAICompatibleClient:
        """Convenience property for 'main' OpenAI client."""
        return self.get_openai_client('main')

    # === Account Pool Access ===

    async def get_account(
        self,
        pool: str,
        *,
        ttl_seconds: int | None = None,
        reservation_token: str = '',
        tags: list[str] | None = None,
        **params: Any,
    ) -> AcquiredAccount:
        """Get an account from the account pool, with automatic lifecycle management.

        All accounts acquired through this method are automatically released
        (and heartbeats stopped) when the ExecutionContext exits.

        管理后台（建池 / 加号 / 打 tag）：https://nexus-pool.woa.com/

        建新池还是打 tag：
        - 凭证形态、环境、配额、TTL、启停要独立 → 建新池（如 ``github`` vs ``gui-sandbox``）。
        - 同一池内可组合的能力，且一台号要能被多种请求领走 → 给账号打 tag（如 ``chrome``、``vscode``）。
        - 不要为每种 tag 组合新建池。任务里每次都写死同一组 ``tags=`` 且池内其它号
          永不该被领到 → 那是假 tag，应拆成独立池。

        Calling examples::

            # 1. 不筛 tag：池里任意一条 idle 账号都可以领
            account = await ctx.get_account('github')
            token = account.tokens['access_token']

            # 2. 按 tag 筛选（AND）：账号必须**同时**带上列表里每一个 tag。
            #    关键字参数 tags= 传 Python list[str]；账号上多出来的 tag 不影响匹配。
            #    不传 / tags=None / tags=[] 等价于不按 tag 筛。
            account = await ctx.get_account('gui-sandbox', tags=['chrome', 'vscode'])

            # 3. 从已预留批次里领（benchmark 流程）
            rsv = await mgr.reserve('github', count=10)
            for task in tasks:
                account = await ctx.get_account('github', reservation_token=rsv.reservation_token)

        The returned :class:`AcquiredAccount` provides:
        - ``.credentials`` — parsed credentials dict
        - ``.tokens`` — parsed tokens dict
        - ``.account_id`` / ``.account_key`` / ``.pool`` — metadata
        - ``await account.invalidate(reason)`` — mark as invalid (stops heartbeat, skips release)
        - ``await account.release()`` — manually release early

        Args:
            pool: Pool name (e.g. ``'github'``).
            ttl_seconds: Lease TTL in seconds (defaults to ``account_pool_ttl``).
            reservation_token: Token from a prior reservation to acquire from a reserved batch.
            tags: 可选关键字参数，类型 ``list[str] | None``。传参示例
                ``tags=['chrome', 'vscode']``。匹配规则是 AND：账号必须同时具备列表里
                每一个非空 tag。省略该参数、传 ``None`` 或 ``[]`` 时请求体不含 ``tags``，
                服务端不按 tag 过滤。账号上多出来的 tag 不妨碍匹配。
            **params: Additional params (reserved for future extensions).

        Returns:
            :class:`AcquiredAccount` with parsed credentials and tokens.

        Raises:
            NoAvailableAccountError: No available account in the pool.
            AccountPoolError: Other server/client errors.
            RuntimeError: If no account pool endpoint is configured and
                the env var ``NEXUS_ACCOUNT_POOL_ENDPOINT`` is not set.
        """
        if self._account_pool_manager is None:
            from nexus.runtime.accounts import AccountPoolManager

            endpoint = self.account_pool_endpoint  # None = use default
            ttl = self.account_pool_ttl
            self._account_pool_manager = AccountPoolManager(
                endpoint=endpoint,
                ttl_seconds=ttl,
            )

        mgr = self._account_pool_manager

        return await mgr.acquire(
            pool,
            ttl_seconds=ttl_seconds,
            reservation_token=reservation_token,
            tags=tags,
        )

    # === Async Context Manager ===

    async def __aenter__(self) -> ExecutionContext:
        """Enter async context, setting this as the current active context."""
        self._context_token = _current_context.set(self)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit async context - releases all runtimes, accounts, and restores previous context."""
        cleanup_error: BaseException | None = None
        cleanup_stage: str | None = None
        try:
            try:
                await self.close_mcp_tools()
            except BaseException as exc:
                cleanup_error = exc
                cleanup_stage = 'close_mcp_tools'
            for name, runtime in self._runtimes.items():
                try:
                    await runtime.drain_trace()
                except asyncio.CancelledError as exc:
                    cancellation_count = get_current_task_cancellation_count()
                    if cancellation_count == 0:
                        logger.warning(
                            '[nexus-context-trace-drain-child-cancellation] primary_error_type=%s '
                            'cleanup_error_type=%s cleanup_stage=drain_trace:%s task_name=%s '
                            'agent_name=%s run_id=%s task_cancelling=%s',
                            exc_type.__name__ if exc_type is not None else None,
                            type(exc).__name__,
                            name,
                            self.task_name,
                            self.agent_name,
                            self.run_id,
                            cancellation_count,
                            exc_info=(type(exc), exc, exc.__traceback__),
                        )
                    elif cleanup_error is None:
                        cleanup_error = exc
                        cleanup_stage = f'drain_trace:{name}'
                except Exception:
                    logger.warning('Failed to drain runtime trace for %s', name, exc_info=True)
                except BaseException as exc:
                    if cleanup_error is None:
                        cleanup_error = exc
                        cleanup_stage = f'drain_trace:{name}'
            for unsubs in self._runtime_shared_unsubs.values():
                for unsub in unsubs:
                    try:
                        unsub()
                    except Exception:
                        logger.debug('Failed to unsubscribe Runtime shared value listener', exc_info=True)
            self._runtime_shared_unsubs.clear()
            for unsubs in self._task_files_runtime_unsubs.values():
                for unsub in unsubs:
                    try:
                        unsub()
                    except Exception:
                        logger.debug('Failed to unsubscribe task_files runtime listener', exc_info=True)
            self._task_files_runtime_unsubs.clear()
            # Release all account pool resources
            if self._account_pool_manager is not None:
                try:
                    await self._account_pool_manager.release_all()
                except BaseException as exc:
                    logger.warning('Failed to release account pool accounts: %s', exc)
                finally:
                    await self._account_pool_manager.aclose()
                    self._account_pool_manager = None
            if self.runtime_provider is not None:
                try:
                    await self.runtime_provider.release_all()
                except BaseException as exc:
                    if cleanup_error is None:
                        cleanup_error = exc
                else:
                    self._runtimes.clear()
        finally:
            if self._context_token is not None:
                _current_context.reset(self._context_token)
                self._context_token = None
        if cleanup_error is not None:
            logger.error(
                '[nexus-context-cleanup-error] primary_error_type=%s cleanup_error_type=%s '
                'cleanup_stage=%s task_name=%s agent_name=%s run_id=%s task_cancelling=%s',
                exc_type.__name__ if exc_type is not None else None,
                type(cleanup_error).__name__,
                cleanup_stage,
                self.task_name,
                self.agent_name,
                self.run_id,
                get_current_task_cancellation_count(),
                exc_info=(type(cleanup_error), cleanup_error, cleanup_error.__traceback__),
            )
        if cleanup_error is not None and exc_val is None:
            raise cleanup_error
        if cleanup_error is not None:
            logger.error(
                'Context cleanup failed while preserving primary %s: %s',
                type(exc_val).__name__,
                cleanup_error,
            )

    async def load_mcp_tools(
        self,
        extra_servers: list[dict[str, Any] | Any] | None = None,
        *,
        clear_existing: bool = False,
    ) -> dict[str, Any]:
        """Load MCP tools into this execution context.

        Loads default specs from ``tool_configs.mcp_servers`` and appends any
        ``extra_servers`` passed in for the current run.
        """
        from nexus.core.tool.mcp.manager import MCPToolManager

        default_servers = self.tool_configs.get('mcp_servers', [])
        specs = [*(default_servers or []), *(extra_servers or [])]

        if self._mcp_manager is None:
            self._mcp_manager = MCPToolManager(self)

        if not specs and not clear_existing:
            return dict(self.mcp_tools)

        self.mcp_tools = await self._mcp_manager.load(specs, clear_existing=clear_existing)
        return dict(self.mcp_tools)

    async def close_mcp_tools(self) -> None:
        """Close all live MCP tools attached to this execution context."""
        if self._mcp_manager is not None:
            try:
                await self._mcp_manager.close()
            finally:
                self._mcp_manager = None
                self.mcp_tools.clear()
            return
        self.mcp_tools.clear()


def _is_inside_runtime_tool_function() -> bool:
    """Check if we're currently inside a RUNTIME tool function by inspecting the call stack.

    Looks for functions with nexus_meta attribute where:
    - nexus_meta['type'] == 'tool'
    - nexus_meta['tool_type'] == ToolType.RUNTIME

    LOCAL tools are allowed to use require_context() to access runtime,
    but RUNTIME tools should use ToolContext instead.

    Returns:
        True if we're inside a registered RUNTIME tool function
    """
    from nexus.core.tool import ToolType

    for frame_info in inspect.stack():
        frame = frame_info.frame
        # Check if this frame's function has nexus_meta indicating it's a tool
        func_name = frame_info.function
        # Try to get the function object from locals or globals
        func = frame.f_locals.get(func_name) or frame.f_globals.get(func_name)
        if func is not None:
            nexus_meta = getattr(func, 'nexus_meta', None)
            if nexus_meta and nexus_meta.get('type') == 'tool':
                # Only block RUNTIME tools, allow LOCAL tools to use require_context()
                tool_type = nexus_meta.get('tool_type')
                if tool_type == ToolType.RUNTIME:
                    return True
    return False


def require_context() -> ExecutionContext:
    """Get the current execution context, raising if not available.

    Returns:
        The current ExecutionContext

    Raises:
        RuntimeError: If no context is active or if called from within a RUNTIME tool
    """
    if _is_inside_runtime_tool_function():
        raise RuntimeError(
            'require_context() should not be called from within a RUNTIME tool function. '
            'RUNTIME tools should use ToolContext parameter instead. '
            'LOCAL tools can use require_context() to access runtime.'
        )

    ctx = _current_context.get()
    if ctx is None:
        raise RuntimeError("No active ExecutionContext. Use 'async with ExecutionContext(...):' to create one.")
    return ctx


__all__ = [
    'ExecutionContext',
    'RuntimePoolConfig',
    'require_context',
    'AcquiredAccount',
    'AccountPoolManager',
]
