"""local_one_agent bash tool.

继承 harness v3 的 ``bash_tool_v2``（已含输出过大存盘逻辑），在 local 层叠加：
- sandbox 包装（``sandbox_wrap_fn``，仅模拟 macOS + bwrap 可用时由 executor 注入）
- Windows 分支（``waa_backend`` 由 executor 在 ``os == 'windows'`` 时注入）：命令转发 WAA VM 执行，
  输出形状与 Linux 一致（复用 v3 截断 / 预览 helper），大输出经 WAA 写回 VM。
- ``description`` 字段（v3 BashInputV2 已有，本层放宽为可选）
"""

from __future__ import annotations

import logging
import ntpath
import uuid

from nexus.core import ToolContext, ToolResponse, ToolType, tool_registry
from nexus.extensions.agents.local_one_agent.waa_backend import WaaBackend
from nexus.extensions.agents.yb_one_agent.tools import (
    MAX_TIMEOUT_MS,
    OUTPUT_THRESHOLD,
    BashInputV2,
    _format_large_output,
    _make_output,
    bash_tool_v2,
    maybe_truncate,
)
from nexus.runtime.executors.models.command import CommandRunStatus


logger = logging.getLogger(__name__)


class LocalBashInput(BashInputV2):
    pass


async def _run_via_waa(
    backend: WaaBackend,
    *,
    command: str,
    cwd: str,
    timeout_seconds: float,
) -> ToolResponse:
    """Windows 分支：命令在 WAA VM 内执行，输出形状对齐 v3 bash。"""
    response = await backend.run_command(command, cwd=cwd, timeout=timeout_seconds)

    if response.status == CommandRunStatus.TimeLimitExceeded:
        parts = []
        if response.stdout:
            parts.append(response.stdout)
        if response.stderr:
            parts.append(f'[stderr]\n{response.stderr}')
        parts.append(f'[Command timed out after {timeout_seconds}s]')
        output_text = '\n'.join(parts) if parts else f'Command timed out after {timeout_seconds}s'
        return ToolResponse(
            output=_make_output(maybe_truncate(output_text)),
            meta={'command': command, 'exit_code': -1, 'timeout': True, 'execution_time': response.execution_time},
        )

    if response.status == CommandRunStatus.Error:
        error_text = response.stderr or 'Command execution failed'
        return ToolResponse(
            output=_make_output(maybe_truncate(f'Error: {error_text}')),
            meta={'command': command, 'exit_code': -1, 'error': True},
        )

    exit_code = response.return_code or 0
    output_text = response.stdout or ''
    if response.stderr:
        output_text += f'\n[stderr]\n{response.stderr}'
    if not output_text.strip():
        output_text = '(no output)'

    # 大输出：完整内容写回 VM（<cwd>\.tool_output\<id>.txt），for_model 给「路径 + 预览」。
    # 写回失败不阻断，退化为字符截断（对齐 v3 的 OSError 兜底）。
    if len(output_text) > OUTPUT_THRESHOLD:
        saved_path = ntpath.join(cwd, '.tool_output', f'{uuid.uuid4().hex[:8]}.txt')
        try:
            await backend.write_file(saved_path, output_text)
        except Exception as exc:  # noqa: BLE001  写回失败仅退化，不影响命令结果
            logger.warning('local bash(waa): failed to persist large output to %s: %s', saved_path, exc)
        else:
            return ToolResponse(
                output=_make_output(_format_large_output(output_text, saved_path)),
                meta={
                    'command': command,
                    'exit_code': exit_code,
                    'output_file': saved_path,
                    'output_size': len(output_text),
                    'execution_time': response.execution_time,
                },
            )
        output_text = maybe_truncate(output_text)

    output_text += f'\n[Command finished with exit code {exit_code}]'
    return ToolResponse(
        output=_make_output(output_text),
        meta={'command': command, 'exit_code': exit_code, 'execution_time': response.execution_time},
    )


@tool_registry.register(
    'local_one_agent/bash',
    meta={
        'name': 'bash',
        'description': (
            'Executes a bash command with optional timeout.\n\n'
            'IMPORTANT: This tool is for terminal operations like git, npm, running scripts, etc. '
            'DO NOT use it for file operations — use the specialized read, edit, write tools instead.\n\n'
            'Usage:\n'
            '- Always quote file paths that contain spaces with double quotes.\n'
            '- For independent commands, prefer chaining with && or calling multiple times.\n'
            '- Use the description field to explain your intent for the command.'
        ),
        'tool_type': ToolType.RUNTIME,
    },
)
async def bash_tool(params: LocalBashInput, context: ToolContext) -> ToolResponse:
    """Local bash：Windows 走 WAA，其余按沙箱策略包装后委托 v3 bash_tool_v2。"""
    waa_backend = context.tool_state.get('waa_backend')
    if isinstance(waa_backend, WaaBackend):
        cwd = context.tool_state.get('working_dir', '')
        timeout_seconds = min(params.timeout, MAX_TIMEOUT_MS) / 1000.0
        logger.debug('local bash: forwarding to WAA, cwd=%s', cwd)
        return await _run_via_waa(waa_backend, command=params.command, cwd=cwd, timeout_seconds=timeout_seconds)

    command = params.command
    sandbox_wrap_fn = context.tool_state.get('sandbox_wrap_fn')
    if sandbox_wrap_fn is not None:
        command = sandbox_wrap_fn(command)
        logger.debug('local bash: command wrapped by sandbox')

    v3_params = BashInputV2(
        description=params.description or 'run command',
        command=command,
        timeout=params.timeout,
    )
    return await bash_tool_v2(v3_params, context)
