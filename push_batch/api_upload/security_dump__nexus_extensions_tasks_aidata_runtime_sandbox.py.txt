"""沙箱 I/O 辅助工具：文件上传 + case bootstrap 执行。

本模块所有函数与 nexus 运行时（ ``ctx.get_runtime('main')`` 返回的实例）
交互，操作路径位于 gongfeng 沙箱内。 主机侧的对应实现位于
:mod:`engine.context` + :mod:`engine.case_spec` 。

三大关注点：

1. **通用上传** —— ``upload_text_file`` / ``upload_dir`` 是对
   ``runtime.upload_file`` 的轻量封装，附带 base64 编码 + 分批。

2. **Case 资产上传** —— ``upload_case_assets`` 会把 ``cases/<id>/`` 下
   除清单文件与仅主机使用的辅助脚本以外的所有内容上传至沙箱的每个 case 的
   scratch 目录。 过滤逻辑有明确文档，方便 case 作者了解哪些名称会被跳过。

3. **Bootstrap** —— ``run_bootstrap`` 会在 MCP 工具加载前于沙箱中执行 case
   的 ``bootstrap_script`` ，以便启动后台服务 / 灌入种子数据 / 准备目录。
"""

from __future__ import annotations

import base64
import logging

from pathlib import Path
from typing import Any

from nexus.runtime.executors.file import FileItem


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 通用上传辅助
# ---------------------------------------------------------------------------


async def upload_text_file(runtime: Any, remote_path: str, content: str) -> None:
    """上传单个 utf-8 字符串到沙箱内的 ``remote_path`` 。"""
    b64 = base64.b64encode(content.encode('utf-8')).decode('ascii')
    await runtime.upload_file(
        files=[FileItem(path=remote_path, content=b64, encoding='base64')],
        overwrite=True,
    )


async def upload_dir(runtime: Any, local_dir: Path, remote_dir: str) -> None:
    """递归上传主机目录到 ``remote_dir`` 。 仅上传文件。"""
    if not local_dir.is_dir():
        raise FileNotFoundError(f'upload source directory does not exist: {local_dir}')

    files: list[FileItem] = []
    for p in local_dir.rglob('*'):
        if not p.is_file():
            continue
        rel = p.relative_to(local_dir).as_posix()
        files.append(
            FileItem(
                path=f'{remote_dir}/{rel}',
                content=base64.b64encode(p.read_bytes()).decode('ascii'),
                encoding='base64',
            )
        )
    if not files:
        raise ValueError(f'upload source directory contains no files: {local_dir}')
    batch = 20
    for i in range(0, len(files), batch):
        await runtime.upload_file(files=files[i : i + batch], overwrite=True)
    logger.info('[aidata] uploaded %d files to %s', len(files), remote_dir)


def ensure_command_succeeded(result: Any, description: str) -> None:
    """Raise a diagnostic error when a sandbox command did not finish successfully."""
    stdout = (getattr(result, 'stdout', None) or '').strip()
    stderr = (getattr(result, 'stderr', None) or '').strip()
    status = getattr(getattr(result, 'status', None), 'value', getattr(result, 'status', None))
    return_code = getattr(result, 'return_code', None)
    if return_code is None:
        return_code = getattr(result, 'exit_code', None)
    if status not in (None, 'Finished') or return_code not in (0, None):
        raise RuntimeError(
            f'{description} failed status={status!r} return_code={return_code!r}; '
            f'stderr head: {stderr[:400]!r}; stdout head: {stdout[:400]!r}'
        )


# ---------------------------------------------------------------------------
# Case 资产上传
# ---------------------------------------------------------------------------


# case 目录下永远不应上传到沙箱的主机侧文件/目录：它们要么是声明本身
# （ agent 通过独立通道获取 prompt ），要么是有独立上传路径的上下文 fixture ，
# 要么是仅主机使用的辅助脚本（例如 xlsx fixture 生成器 —— 已由下方下划线前缀
# 规则过滤掉）。
#
# 注意： ``solution/`` / ``tests/`` / ``run-test.sh`` / 每个 case 的
# ``assets/`` 在此处 **不** 被跳过 —— 它们 **会** 被上传，但落在
# :data:`CASE_DIR_IN` （ ``/workspace/.aidata/case`` ）下，位于 shell MCP 的
# ``ALLOWED_DIR`` （ ``/workspace/desktop`` ）之外。 因此 agent 无法
# ``cat`` 到它们；引擎侧仍可读取 ``run-test.sh`` / 检查
# ``solution/golden_rollout.json`` 。
_CASE_SKIP_NAMES = {
    # 清单 —— 引擎从主机副本加载
    'case.yaml',
    'task.yaml',
    'rubric.yaml',
    # 提示 —— 已被内联到 agent 的 system/user 消息
    'instruction.md',
    # 面向人类的 README —— 仅供审阅者查阅
    'README.md',
    # 上下文 fixture —— 通过 ``context_files`` 上传到 AGENT_INPUTS_DIR
    'files',
    'data',
    # python 缓存残留物
    '__pycache__',
}


def is_case_asset_shippable(relative_path: str | Path) -> bool:
    """当 case 相对路径会被 :func:`upload_case_assets` 上传时返回 True。"""
    parts = Path(relative_path).parts
    if not parts:
        return False
    top_level_name = parts[0]
    if top_level_name in _CASE_SKIP_NAMES:
        return False
    return not top_level_name.startswith(('_', '.'))


def _should_ship_case_entry(p: Path) -> bool:
    """当 ``p`` （ case 目录的直接子项）可以被上传到沙箱时返回 True 。
    仅主机使用的辅助脚本通过名称前缀（下划线 / 点号）过滤，这样 case 就可以
    把生成器脚本与资产放在一起，而不必担心它们被泄漏到沙箱状态中。
    """
    return is_case_asset_shippable(p.name)


async def upload_case_assets(runtime: Any, case_dir: Path, remote_dir: str) -> int:
    """把 ``case_dir`` 下所有未被过滤的条目上传到沙箱的 ``remote_dir``
    （保留子目录结构）。 仅文件、递归。

    跳过 ``case.yaml`` / ``rubric.yaml`` / ``README.md`` / ``files/`` 以及任何
    以 ``_`` 或 ``.`` 起始的名称 —— 它们要么是冗余的（上下文文件有独立的
    上传路径），要么是仅主机使用的辅助脚本。

    返回上传文件数（若 case 无可上传资产则为 0 ；调用方在这种情况下可以
    安全地跳过 bootstrap ）。
    """
    shippable: list[Path] = []
    for entry in case_dir.iterdir():
        if not _should_ship_case_entry(entry):
            continue
        if entry.is_file():
            shippable.append(entry)
        elif entry.is_dir():
            shippable.extend(p for p in entry.rglob('*') if p.is_file())

    if not shippable:
        return 0

    items: list[FileItem] = []
    for p in shippable:
        rel = p.relative_to(case_dir).as_posix()
        items.append(
            FileItem(
                path=f'{remote_dir}/{rel}',
                content=base64.b64encode(p.read_bytes()).decode('ascii'),
                encoding='base64',
            )
        )
    batch = 20
    for i in range(0, len(items), batch):
        await runtime.upload_file(files=items[i : i + batch], overwrite=True)
    logger.info('[aidata] uploaded %d case assets to %s', len(items), remote_dir)
    return len(items)


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------


async def run_bootstrap(
    runtime: Any,
    script_rel: str,
    case_remote_dir: str,
    timeout: int = 180,
    extra_env: dict[str, str] | None = None,
) -> None:
    """在沙箱内执行 ``{case_remote_dir}/{script_rel}`` 。

    脚本会先被 chmod +x （防止 case 目录来自不保留可执行位的文件系统）。
    非零退出会中止任务 —— 缺失种子数据 / 服务启动失败应显式报错，而不是
    静默降级 rubric 。

    ``extra_env`` 会被转发到 runtime 的 ``run_command`` ，从而让服务
    （参见 :mod:`aidata.engine.services` ）可以把解析出的 ``host:port``
    句柄通过环境变量暴露给 bootstrap 脚本。
    """
    script_path = f'{case_remote_dir}/{script_rel}'
    logger.info('[aidata] running bootstrap: %s', script_path)
    cmd = f'chmod +x {script_path} 2>/dev/null; cd {case_remote_dir} && bash {script_path}'
    result = await runtime.run_command(cmd, timeout=timeout, extra_env=extra_env or {})
    stdout = (getattr(result, 'stdout', None) or '').strip()
    stderr = (getattr(result, 'stderr', None) or '').strip()
    ensure_command_succeeded(result, f'bootstrap script {script_rel!r}')
    if stdout:
        logger.info('[aidata] bootstrap stdout: %s', stdout[:400])
    if stderr:
        logger.info('[aidata] bootstrap stderr: %s', stderr[:400])


__all__ = [
    'ensure_command_succeeded',
    'run_bootstrap',
    'upload_case_assets',
    'upload_dir',
    'upload_text_file',
]
