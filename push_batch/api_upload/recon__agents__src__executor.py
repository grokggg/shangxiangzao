"""LocalToolExecutor — ToolExecutor with permission engine and sandbox integration.

All tools are executed locally (bypassing Docker runtime), with permission
checks and optional bubblewrap sandboxing for bash commands.
"""

from __future__ import annotations

import fnmatch
import logging

from typing import Any

from nexus.core.context import require_context
from nexus.core.tool.executor import ToolExecutor, get_tool_meta
from nexus.core.tool.types import ToolContext, ToolFunc, ToolResponse
from nexus.extensions.agents.local_one_agent.context import LocalContext
from nexus.extensions.agents.local_one_agent.error_codes import LocalToolErrorCode
from nexus.extensions.agents.local_one_agent.permission_engine import DecisionAction, PermissionEngine
from nexus.extensions.agents.local_one_agent.sandbox import LocalSandbox
from nexus.extensions.agents.local_one_agent.waa_backend import WaaBackend
from nexus.extensions.agents.local_one_agent.windows_sandbox import WindowsSandbox


logger = logging.getLogger(__name__)


def _make_output(text: str) -> list[dict]:
    return [{'type': 'text', 'text': text}]


def _permission_denied_text(category: str | None) -> str:
    """对齐线上 ``tool_proxy/for_model.go``：``Permission denied: {category}``。

    K2：模型看到的 for_model 文本由 error code + category 生成（英文），而非中文 reason，
    以保证训推一致。 (category 取值词表最终以端侧线上为准。)
    """
    return f'Permission denied: {category}' if category else 'Permission denied'


def _user_rejected_text() -> str:
    """对齐线上 for_model.go 的 ``USER_REJECTED`` 文案。"""
    return 'User rejected the operation.'


class LocalToolExecutor(ToolExecutor):
    """ToolExecutor that intercepts tool calls with permission checks.

    All tools (including those marked RUNTIME) are executed locally — we never
    go through the Docker runtime HTTP path. This is intentional: local_one_agent
    runs on the user's machine, not inside a sandbox container.
    """

    def __init__(
        self,
        tools: list[ToolFunc],
        *,
        local_context: LocalContext,
        allowed_tool_names: list[str] | None = None,
    ) -> None:
        super().__init__(tools, allowed_tool_names=allowed_tool_names)
        self._local_context = local_context
        self._permission_engine = PermissionEngine(local_context)
        self._sandbox: LocalSandbox | None = None
        # os == 'windows'：执行后端转发 WAA（见 waa_backend.py）。 懒获取、缓存一次。
        self._waa_backend: WaaBackend | None = None

    def get_model_tool_schemas(
        self,
        *,
        exclude_canonical_names: frozenset[str] | set[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Schemas for the LLM ``tools`` parameter.

        Like yb_one_agent ``build_tool_schemas``: excluded tools stay registered and
        callable (e.g. after ``load_skill``), but are not advertised up-front.
        """
        exclude = frozenset(exclude_canonical_names or ())
        self._assert_no_duplicate_schema_names()
        return [
            self._build_schema_for_tool(name) for name in self._visible_canonical_tool_names() if name not in exclude
        ]

    @property
    def sandbox(self) -> LocalSandbox:
        """Lazy-initialize sandbox on first access."""
        if self._sandbox is None:
            self._sandbox = LocalSandbox(
                workspace=self._local_context.workspace,
                runtime=self._local_context.runtime,
                permission=self._local_context.permission,
            )
        return self._sandbox

    def _build_tool_context(self, **kwargs: object) -> ToolContext:
        """Override to inject local-specific state into tool_state."""
        ctx = require_context()
        tool_state = ctx.tool_state.copy()

        if self._local_context.workspace:
            tool_state['workspace_path'] = self._local_context.workspace.path
            # v3 bash 用 working_dir 决定执行目录与落盘位置（<cwd>/.tool_output）。
            # 本地模式下命令应在用户工作区执行，故对齐到 workspace.path。
            tool_state['working_dir'] = self._local_context.workspace.path

        if self._local_context.skill_path:
            tool_state['skill_path'] = self._local_context.skill_path

        # Windows：注入 WAA 执行后端，且**不**注入 bwrap 沙箱（Windows 无 bwrap，隔离交给 VM 边界）。
        # macos/linux：维持进程内 + bwrap 沙箱包装（老路一字不变）。
        if self._local_context.runtime.os == 'windows':
            if self._waa_backend is not None:
                tool_state['waa_backend'] = self._waa_backend
        elif self.sandbox.is_available():
            tool_state['sandbox_wrap_fn'] = self._sandbox_wrap

        return ToolContext(
            run_id=ctx.run_id,
            tool_configs=ctx.tool_configs.copy(),
            tool_state=tool_state,
            trace_id=ctx.trace_id,
        )

    def _sandbox_wrap(self, command: str) -> str:
        """bash 命令的沙箱包装入口，叠加 M3 提权 mock（预判式）。

        分层安全：权限引擎是第一道闸（DENY 的命令根本到不了这里）；沙箱是第二道
        内核闸。 M3：命中**显式** confirmed 决策的命令视为用户已授权跳出沙箱，直接
        不加 bwrap 执行（模拟「被沙箱拦住 → 提权 → 跳出」）。 无显式规则则照常隔离，
        避免未配规则时沙箱整体失效。
        """
        if self._lookup_decision(command) == 'confirmed':
            logger.info('sandbox-escape: 命令命中 confirmed 决策，跳出沙箱执行: %s', command)
            return command
        return self.sandbox.wrap_command(command)

    @staticmethod
    def _short_tool_name(canonical_name: str) -> str:
        """Extract the short tool name from canonical name: 'local_one_agent/bash' → 'bash'."""
        return canonical_name.rsplit('/', 1)[-1] if '/' in canonical_name else canonical_name

    async def _call_resolved_tool(self, tool_name: str, params: dict[str, Any]) -> ToolResponse:
        """Override to inject permission check and force local execution."""
        short_name = self._short_tool_name(tool_name)
        permission_response = await self._check_permission_and_call(short_name, params)
        if permission_response is not None:
            return permission_response

        if tool_name in self._tools:
            tool_func = self._tools[tool_name]
            meta = get_tool_meta(tool_func)
            try:
                parsed_params = meta.input_model.model_validate(params)
            except Exception as exc:
                raise ValueError(f"Invalid params for tool '{tool_name}': {exc}") from exc

            await self._ensure_waa_backend()
            tool_context = self._build_tool_context()
            return await tool_func(parsed_params, tool_context)

        return await super()._call_resolved_tool(tool_name, params)

    async def _ensure_waa_backend(self) -> None:
        """os == 'windows' 时懒获取 WAA runtime 并构造执行后端（缓存一次）。

        非 windows 直接返回，绝不触碰 runtime，保证 Linux 老路零改动。
        命令在 VM 内恒走 PowerShell（见 WaaBackend），与 ``runtime.shell`` 的 SP 展示解耦。

        构造后探测 ybsbr（OS 级沙箱,对标 Linux bwrap）:探测须在 set_sandbox 之前,否则探测命令
        会被自身包装。 探测通过则注入 backend,后续命令自动被 ybsbr 包装;不可用则降级为仅权限引擎。
        """
        if self._local_context.runtime.os != 'windows':
            return
        if self._waa_backend is not None:
            return
        ctx = require_context()
        runtime = await ctx.get_runtime()
        backend = WaaBackend(runtime)
        sandbox = WindowsSandbox(
            self._local_context.workspace,
            self._local_context.permission,
            enabled=self._local_context.windows_sandbox,
        )
        await sandbox.probe(backend)
        backend.set_sandbox(sandbox)
        self._waa_backend = backend

    async def _check_permission_and_call(self, tool_name: str, tool_args: dict[str, Any]) -> ToolResponse | None:
        """Check permission for a tool call. Returns a ToolResponse if blocked, None if allowed."""
        decision = self._permission_engine.check_permission(tool_name, tool_args)

        if decision.action == DecisionAction.DENY:
            logger.warning(
                'Permission DENY: tool=%s, reason=%s, category=%s',
                tool_name,
                decision.reason,
                decision.redline_category,
            )
            return ToolResponse(
                output=_make_output(_permission_denied_text(decision.redline_category)),  # type: ignore[invalid-argument-type]
                meta={
                    'ok': False,
                    'error': {
                        'code': LocalToolErrorCode.PERMISSION_DENIED.value,
                        'category': decision.redline_category,
                    },
                },
            )

        if decision.action == DecisionAction.ASK:
            call_repr = self._build_call_repr(tool_name, tool_args)
            user_decision = await self._resolve_ask_decision(call_repr)
            if user_decision == 'rejected':
                logger.info('User rejected: tool=%s', tool_name)
                return ToolResponse(
                    output=_make_output(_user_rejected_text()),  # type: ignore[invalid-argument-type]
                    meta={
                        'ok': False,
                        'error': {'code': LocalToolErrorCode.USER_REJECTED.value},
                    },
                )
            logger.info('User confirmed: tool=%s', tool_name)

        return None

    def _lookup_decision(self, call_repr: str) -> str | None:
        """查 user_decisions 命中的**显式**决策。

        返回 ``'confirmed'`` / ``'rejected'``；无配置或无规则命中返回 ``None``（无显式决策）。
        """
        config = self._local_context.permission_engine
        if config is None:
            return None

        for rule in config.user_decisions:
            if rule.pattern == '*' or fnmatch.fnmatch(call_repr, f'*{rule.pattern}*'):
                return rule.decision

        return None

    async def _resolve_ask_decision(self, call_repr: str) -> str:
        """Resolve ASK decision using the configured user_decisions rules.

        Returns 'confirmed' or 'rejected'. Defaults to 'confirmed' if no rule matches.
        """
        return self._lookup_decision(call_repr) or 'confirmed'

    def _build_call_repr(self, tool_name: str, tool_args: dict[str, Any]) -> str:
        """Build a human-readable representation of a tool call for pattern matching."""
        if tool_name == 'bash':
            return tool_args.get('command', '')
        if tool_name in ('read', 'write', 'edit'):
            return tool_args.get('file_path', '')
        return f'{tool_name}({tool_args})'
