"""local_one_agent write tool.

继承 harness v3 的 ``write_tool_v2``，local 层叠加：
- 自动创建父目录（``os.makedirs``）
"""

from __future__ import annotations

import logging
import os

from nexus.core import ToolContext, ToolResponse, ToolType, tool_registry
from nexus.extensions.agents.local_one_agent.waa_backend import WaaBackend
from nexus.extensions.agents.yb_one_agent.tools import WriteInputV2, _make_output, write_tool_v2


logger = logging.getLogger(__name__)


async def _write_via_waa(backend: WaaBackend, params: LocalWriteInput) -> ToolResponse:
    """Windows 分支：写入 WAA VM 内文件（覆盖），for_model 文本与 v3 write 一致。"""
    try:
        await backend.write_file(params.file_path, params.content)
    except Exception as e:  # noqa: BLE001  写失败转成 for_model 错误文本
        logger.warning('local write(waa): failed to write %s: %s', params.file_path, e)
        return ToolResponse(output=_make_output(f'Error: {e}'))

    lines = params.content.count('\n') + 1
    return ToolResponse(output=_make_output(f'Successfully wrote {lines} lines to {params.file_path}.'))


class LocalWriteInput(WriteInputV2):
    pass


@tool_registry.register(
    'local_one_agent/write',
    meta={
        'name': 'write',
        'description': (
            'Writes content to a file, creating it or overwriting if it already exists.\n\n'
            'Usage:\n'
            '- file_path must be an absolute path.\n'
            '- Parent directories are created automatically if they do not exist.\n'
            '- Prefer edit for modifying existing files — only use write for new files or full rewrites.\n'
            '- If modifying an existing file, use read first to review current content.'
        ),
        'tool_type': ToolType.RUNTIME,
    },
)
async def write_tool(params: LocalWriteInput, context: ToolContext) -> ToolResponse:
    """Local write：Windows 走 WAA；其余先创建父目录，再委托 v3 write_tool_v2。"""
    waa_backend = context.tool_state.get('waa_backend')
    if isinstance(waa_backend, WaaBackend):
        logger.debug('local write: forwarding to WAA, path=%s', params.file_path)
        return await _write_via_waa(waa_backend, params)

    parent = os.path.dirname(params.file_path)
    if parent:
        try:
            os.makedirs(parent, exist_ok=True)
        except OSError as exc:
            logger.warning('write: failed to create parent dir %s: %s', parent, exc)

    v3_params = WriteInputV2(
        description=params.description or 'write file',
        file_path=params.file_path,
        content=params.content,
    )
    return await write_tool_v2(v3_params, context)
