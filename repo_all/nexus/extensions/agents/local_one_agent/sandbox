"""Bubblewrap (bwrap) sandbox wrapper for bash tool commands.

训练在 Linux 容器里用 bwrap **模拟 macOS 沙箱**：``runtime.os == 'macos'`` 时开
bwrap（B1）。 ``runtime.os == 'windows'`` 走 WAA Runtime / VM，本期沙箱留 stub、
不开 bwrap（B2），优雅降级为仅权限引擎。
"""

from __future__ import annotations

import logging
import re
import shlex
import shutil
import subprocess

from nexus.extensions.agents.local_one_agent.context import LocalPermission, LocalRuntime, LocalWorkspace


logger = logging.getLogger(__name__)

# install-escape（Part Ⅰ §6）：训练镜像不预装 Node 包，运行时安装需放网。
# 命中以下安装类命令前缀时，沙箱不套外层网络隔离 namespace（其余命令维持网络隔离）。
# 保守白名单匹配，避免成为网络逃逸通道。
_INSTALL_ESCAPE_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r'^\s*npm\s+(install|i|ci|add)\b',
        r'^\s*pnpm\s+(install|i|add)\b',
        r'^\s*yarn\s+(add|install)\b',
        r'^\s*npx\s+',
        r'^\s*pip3?\s+install\b',
        # `python -m pip install` / `py -m pip install`：极常见,裸 pip 不在 PATH 时的标准写法。
        r'^\s*(python3?|py)\s+-m\s+pip\s+install\b',
        r'^\s*uv\s+(pip\s+install|add|sync)\b',
        r'^\s*cargo\s+(install|add|fetch)\b',
        r'^\s*go\s+(install|get|mod\s+download)\b',
    ]
]


def _is_install_command(command: str) -> bool:
    """Best-effort detection of dependency-install commands that require network."""
    return any(pattern.search(command) for pattern in _INSTALL_ESCAPE_PATTERNS)


class LocalSandbox:
    """Wraps bash commands in bubblewrap for process-level isolation.

    Lazy-initialized: the first call to wrap_command() checks for bwrap availability.
    Falls back gracefully to unwrapped execution when bwrap is unavailable.
    """

    def __init__(
        self,
        workspace: LocalWorkspace,
        runtime: LocalRuntime,
        permission: LocalPermission,
    ) -> None:
        self._workspace = workspace
        self._runtime = runtime
        self._permission = permission
        self._bwrap_path: str | None = None
        # 网络隔离外层用（见 _build_bwrap_command）：gVisor 下 bwrap --unshare-net 拉 lo 失败，
        # 改用外层 unshare 建空 netns。 缺 unshare 则降级为「仅 FS 隔离、不隔离网络」。
        self._unshare_path: str | None = None
        self._checked = False

    def is_available(self) -> bool:
        """bwrap 是否可用。 仅在模拟 macOS（``os == 'macos'``）且容器内有 bwrap 时为真。"""
        if self._runtime.os != 'macos':
            return False
        self._ensure_checked()
        return self._bwrap_path is not None

    def wrap_command(self, command: str) -> str:
        """Wrap a command in bwrap sandbox. Returns the original command if sandboxing is unavailable."""
        # B2：windows（及非 macos）不开 bwrap，走 WAA/仅权限引擎。
        if self._runtime.os != 'macos':
            return command

        self._ensure_checked()
        if self._bwrap_path is None:
            return command

        return self._build_bwrap_command(command)

    def _ensure_checked(self) -> None:
        if self._checked:
            return
        self._checked = True

        path = shutil.which('bwrap')
        if path is None:
            # 缺 bwrap 不阻断：优雅降级为「仅权限引擎」，bash 直接执行（不影响非沙箱功能）。
            logger.info('bwrap 不在 PATH，本地沙箱关闭，bash 直接执行（不影响非沙箱功能）')
            return

        # 只装了 bwrap 还不够：容器若未放开 unprivileged user namespace（非 privileged），
        # bwrap 会在创建 namespace 时失败。 若直接 wrap，则每条命令都失败，比不开沙箱更糟。
        # 故先做一次可用性探测：跑不起来就当作不可用，干净降级，绝不阻断流程。
        if not self._bwrap_usable(path):
            logger.warning(
                'bwrap 已安装但无法创建 namespace（通常是容器未放开 unprivileged user '
                'namespace / 非 privileged）；本地沙箱关闭，bash 直接执行，不阻断流程。'
            )
            return

        self._bwrap_path = path
        self._unshare_path = shutil.which('unshare')
        logger.info('bwrap found at %s (unshare=%s)', path, self._unshare_path)

    @staticmethod
    def _bwrap_usable(path: str) -> bool:
        """探测 bwrap 能否真正创建 namespace 并挂载，跑不起来则视为不可用。

        关键：必须带 ``--unshare-user``。 训练容器里进程是 root 但 cap 集为 Docker 默认
        （无 CAP_SYS_ADMIN），bwrap 见 euid==0 会默认不建 user namespace，直接在原 ns 里
        ``mount --make-rslave /`` → EPERM。 加 ``--unshare-user`` 先进 userns 拿全 caps 才行。
        不带 ``--unshare-net``：gVisor 下它会因拉起 lo 失败（RTM_NEWADDR ECHILD），且我们
        的网络隔离改由外层 unshare 负责，探测只需覆盖 FS 隔离这条真实路径。
        """
        try:
            result = subprocess.run(  # noqa: S603
                [
                    path,
                    '--unshare-user',
                    '--ro-bind',
                    '/',
                    '/',
                    '--dev-bind',
                    '/dev',
                    '/dev',
                    '--tmpfs',
                    '/tmp',
                    '--die-with-parent',
                    '--',
                    '/bin/true',
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5,
            )
            return result.returncode == 0
        except (OSError, subprocess.SubprocessError):
            return False

    def _build_bwrap_command(self, command: str) -> str:
        level = self._permission.level
        ws_path = self._workspace.path

        # --unshare-user：root + Docker 默认 caps 场景的必需项（见 _bwrap_usable 说明）。
        parts = [self._bwrap_path, '--unshare-user']

        parts.extend(['--ro-bind', '/', '/'])
        # --dev-bind 必须在 --ro-bind / / 之后：bwrap 中「后写的挂载覆盖先写的」，
        # 否则只读根会盖掉 /dev，导致 /dev/null 等设备节点只读、命令报错。
        parts.extend(['--dev-bind', '/dev', '/dev'])
        parts.extend(['--proc', '/proc'])

        if level != 'readonly':
            parts.extend(['--bind', ws_path, ws_path])

        parts.extend(['--tmpfs', '/tmp'])
        parts.extend(['--die-with-parent', '--new-session'])

        # command 作为 `sh -c` 的单个参数整体传入。
        parts.extend(['--', '/bin/sh', '-c', command])

        # M4：对**所有** parts 做 shlex.quote（含 ws_path / command），避免含空格/特殊
        # 字符的路径或命令被 ' '.join 错误拆分。
        bwrap_cmd = ' '.join(shlex.quote(str(part)) for part in parts if part is not None)

        # 网络隔离（Part Ⅰ §1）：readonly / basic 默认都隔离网络。
        # install-escape（Part Ⅰ §6）：安装类命令放网 → 不套外层隔离。
        if _is_install_command(command):
            logger.info('install-escape: 放网执行安装命令: %s', command)
            return bwrap_cmd

        # gVisor 下 bwrap --unshare-net 会因拉起 lo 失败（RTM_NEWADDR ECHILD），
        # 故网络隔离改用外层 unshare 建空 netns（仅含 down 的 lo，无外部网卡）。
        # 缺 unshare 时降级为「仅 FS 隔离、不隔离网络」，不阻断命令。
        if self._unshare_path is None:
            logger.warning('unshare 不可用，本命令网络不隔离（文件系统仍隔离）')
            return bwrap_cmd

        outer = [self._unshare_path, '--user', '--map-root-user', '--net', '--']
        return ' '.join(shlex.quote(part) for part in outer) + ' ' + bwrap_cmd
