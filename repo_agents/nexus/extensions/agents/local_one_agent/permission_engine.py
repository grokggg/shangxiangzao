"""Permission engine for local_one_agent.

Pure-function based permission checking. No side effects.
See reference/permission-engine.md for the full specification.
"""

from __future__ import annotations

import fnmatch
import ntpath
import os
import re
import shlex

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Literal


if TYPE_CHECKING:
    from nexus.extensions.agents.local_one_agent.context import LocalContext


# os_type 用于把纯函数按目标系统分流。 训练进程跑在 Linux，但 task 的目标系统可能是
# macos（bwrap 模拟，走 posix 语义）或 windows（WAA，走 ntpath 语义 + PowerShell/cmd 命令表）。
OsType = Literal['posix', 'windows']


class DecisionAction(str, Enum):
    ALLOW = 'ALLOW'
    DENY = 'DENY'
    ASK = 'ASK'


class RiskLevel(str, Enum):
    NONE = 'none'
    LOW = 'low'
    MEDIUM_HIGH = 'medium_high'


@dataclass
class Decision:
    action: DecisionAction
    reason: str = ''
    risk_level: RiskLevel | None = None
    redline_category: str | None = None


# ═══════════════ Red Line Patterns ═══════════════

SENSITIVE_FILE_PATTERNS = [
    '~/.ssh/id_*',
    '~/.ssh/authorized_keys',
    '/etc/shadow',
    '/etc/sudoers',
    '/etc/passwd',
    '~/Library/Keychains/*',
    '/Library/Keychains/*',
    '~/Library/Application Support/Google/Chrome/*/Login Data*',
    '~/Library/Application Support/Firefox/Profiles/*/logins.json',
    '~/Library/Application Support/Firefox/Profiles/*/key*.db',
    '~/.config/chromium/*/Login Data*',
    '~/.mozilla/firefox/*/logins.json',
    '~/.aws/credentials',
    '~/.aws/config',
    '~/.azure/*',
    '~/.config/gcloud/credentials.db',
    '~/.config/gcloud/application_default_credentials.json',
    '~/.kube/config',
    '~/.docker/config.json',
    '~/.npmrc',
    '~/.pypirc',
    '~/.gem/credentials',
    '~/.netrc',
    '~/.gnupg/private-keys-v1.d/*',
    '~/.gnupg/secring.gpg',
    '**/token',
    '**/secrets.yaml',
    '**/secrets.json',
]

SYSTEM_DIRS = [
    '/System/',
    '/Library/',
    '/usr/',
    '/bin/',
    '/sbin/',
    '/etc/',
    '/var/',
    '/private/',
    '/Applications/',
    '/boot/',
    '/proc/',
    '/sys/',
    'C:\\Windows\\',
    'C:\\Program Files\\',
    'C:\\Program Files (x86)\\',
]

EXFIL_PATTERNS = [
    (r'scp\s+(?!.*\S+:\S+\s+\.)\S+\s+\S+:', 'scp 外发'),
    (r'rsync\s+.*\s+\S+:', 'rsync 外发'),
    (r'curl\s+.*(-d\s|--data|--data-binary|-F\s|--form|--upload-file)', 'curl 外发数据'),
    (r'wget\s+.*--post-(data|file)', 'wget 外发'),
    (r'git\s+push', 'git push'),
    (r'\bnc\b|\bnetcat\b|\bncat\b', 'netcat'),
    (r'\btelnet\b', 'telnet'),
    (r'\bsocat\b', 'socat'),
    (r'\|\s*(curl|wget|nc|ncat|ssh)', '管道到网络工具'),
]

_COMPILED_EXFIL = [(re.compile(p, re.IGNORECASE), desc) for p, desc in EXFIL_PATTERNS]

# ═══════════════ Windows Red Line Patterns ═══════════════

# 全部小写 + 反斜杠，用户目录用 * 通配（Linux 上拿不到 windows 家目录，无法展开 ~）。
# 与 _normalize_win 归一后的路径做 fnmatchcase 比较。
WINDOWS_SENSITIVE_FILE_PATTERNS = [
    r'c:\users\*\.ssh\id_*',
    r'c:\users\*\.ssh\authorized_keys',
    r'c:\users\*\.aws\credentials',
    r'c:\users\*\.aws\config',
    r'c:\users\*\.azure\*',
    r'c:\users\*\.kube\config',
    r'c:\users\*\.docker\config.json',
    r'c:\users\*\.npmrc',
    r'c:\users\*\.pypirc',
    r'c:\users\*\.netrc',
    r'c:\users\*\appdata\*\google\chrome\*\login data*',
    r'c:\users\*\appdata\*\mozilla\firefox\profiles\*\logins.json',
    r'c:\users\*\appdata\*\mozilla\firefox\profiles\*\key*.db',
    r'c:\users\*\appdata\roaming\microsoft\credentials\*',
    r'c:\windows\system32\config\sam',
    r'c:\windows\system32\config\security',
    r'*\secrets.yaml',
    r'*\secrets.json',
    r'*\token',
]

WINDOWS_SYSTEM_DIRS = [
    r'c:\windows',
    r'c:\program files',
    r'c:\program files (x86)',
    r'c:\programdata',
]

WINDOWS_EXFIL_PATTERNS = [
    (r'(invoke-webrequest|iwr|invoke-restmethod|irm)\b.*(-method\s+post|-body|-infile)', 'PowerShell 外发'),
    (r'\bcurl(\.exe)?\b.*(-d\s|--data|--data-binary|-F\s|--form|--upload-file|-t\s|-T\s)', 'curl 外发数据'),
    (r'git\s+push', 'git push'),
    (r'\bscp\b', 'scp 外发'),
    (r'\bnc(\.exe)?\b|\bncat\b', 'netcat'),
    (r'net\s+use\s+\\\\', 'net use 挂载远程'),
    (r'\|\s*(curl|iwr|invoke-webrequest|nc|ncat)', '管道到网络工具'),
]

_COMPILED_WINDOWS_EXFIL = [(re.compile(p, re.IGNORECASE), desc) for p, desc in WINDOWS_EXFIL_PATTERNS]

# ═══════════════ Readonly Bash Whitelist ═══════════════

READONLY_COMMANDS: set[str] = {
    'cat',
    'head',
    'tail',
    'less',
    'more',
    'wc',
    'nl',
    'file',
    'stat',
    'md5sum',
    'sha1sum',
    'sha256sum',
    'cksum',
    # 十六进制 / 二进制查看，纯只读（对齐端上 kDefaultAllowRules）。
    'od',
    'xxd',
    'hexdump',
    # 路径解析工具，纯只读（对齐端上 kDefaultAllowRules）。
    'basename',
    'dirname',
    'realpath',
    'readlink',
    'ls',
    'tree',
    'pwd',
    # 目录导航 builtin：不改文件、不起进程，与 WINDOWS_READONLY_COMMANDS 的 cd/chdir 对齐。
    'cd',
    'chdir',
    'pushd',
    'popd',
    'dirs',
    # 无副作用 builtin：无操作 / 条件判断 / 查询 / 终端控制，端上 basic 靠默认放行，此处显式列入只读。
    'true',
    'false',
    ':',
    'test',
    '[',
    'type',
    'hash',
    'alias',
    'sleep',
    'clear',
    'tty',
    'tput',
    'find',
    'du',
    'df',
    'grep',
    'rg',
    'ag',
    'fd',
    'fdfind',
    'sort',
    'uniq',
    'cut',
    'paste',
    'column',
    'awk',
    'tr',
    'jq',
    'yq',
    'diff',
    'comm',
    'echo',
    'printf',
    'date',
    'hostname',
    'uname',
    'whoami',
    'id',
    'env',
    'printenv',
    'ps',
    'lsof',
    'netstat',
    'ss',
    'which',
}

READONLY_COMMAND_PREFIXES: list[str] = [
    'git status',
    'git diff',
    'git log',
    'git show',
    'git blame',
    'git branch',
    'git tag',
    'git remote',
    'git ls-files',
    'git stash list',
    'git rev-parse',
    'git describe',
    'git shortlog',
    'git reflog',
    'git cat-file',
    'git grep',
    'python --version',
    'python3 --version',
    'node --version',
    'npm --version',
    'pip list',
    'pip show',
    'pip3 list',
    'pip3 show',
    'npm list',
    'cargo --version',
    'rustc --version',
    'go version',
    'java -version',
    'top -l 1',
    'sed -n',
]

DESTRUCTIVE_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r'git\s+push\s+.*(-f|--force)',
        r'git\s+reset\s+--hard',
        r'git\s+clean\s+.*-[a-zA-Z]*f',
        r'git\s+(checkout|restore)\s+(--\s+)?\.(\s|$)',
        r'git\s+stash\s+(drop|clear)',
        r'git\s+branch\s+.*-D',
        r'git\s+rebase',
        r'rm\s+.*-[a-zA-Z]*r',
        r'rm\s+.*-[a-zA-Z]*f',
        r'rmdir',
        r'chmod',
        r'chown',
        r'npm\s+install',
        r'yarn\s+(add|install)',
        r'pnpm\s+(add|install)',
        r'pip3?\s+install',
        r'pip3?\s+uninstall',
        r'brew\s+(install|uninstall|upgrade)',
        r'(apt|apt-get)\s+(install|remove|purge)',
        r'cargo\s+install',
        r'go\s+install',
        r'(DROP|TRUNCATE)\s+(TABLE|DATABASE|SCHEMA)',
        r'DELETE\s+FROM\s+\w+\s*(;|$)',
        r'docker\s+(rm|rmi|system\s+prune)',
        r'kubectl\s+delete',
        r'terraform\s+destroy',
        # L2（P0 补漏）：磁盘 / 提权 / 远程代码执行，此前漏判会在 basic 档静默执行。
        r'\bsudo\b',
        r'\bmkfs(\.\w+)?\b',
        r'\bdd\b\s+.*\bof=',
        r'\bshred\b',
        r'diskutil\s+.*\berase',
        r'(curl|wget)\s+[^|]*\|\s*(sudo\s+)?(bash|sh|zsh|ksh)\b',  # curl|bash 型 RCE
    ]
]

# ═══════════════ Windows Readonly / Destructive ═══════════════

# PowerShell / cmd 只读命令与别名（全小写；PS/cmd 大小写不敏感）。
WINDOWS_READONLY_COMMANDS: set[str] = {
    # cmd 内建 / 工具
    'dir',
    'type',
    'more',
    'where',
    'findstr',
    'find',
    'fc',
    'tree',
    'ver',
    'vol',
    'hostname',
    'whoami',
    'echo',
    'cd',
    'chdir',
    'date',
    'time',
    # PowerShell 只读 cmdlet + 别名
    'get-content',
    'gc',
    'cat',
    'get-childitem',
    'gci',
    'ls',
    'get-item',
    'gi',
    'get-itemproperty',
    'gp',
    'select-string',
    'sls',
    'get-location',
    'gl',
    'pwd',
    'test-path',
    'resolve-path',
    'split-path',
    'measure-object',
    'measure',
    'get-process',
    'gps',
    'get-service',
    'gsv',
    'get-date',
    'get-filehash',
    'compare-object',
    'compare',
    'sort-object',
    'sort',
    'select-object',
    'select',
    'where-object',
    'group-object',
    'format-table',
    'ft',
    'format-list',
    'fl',
    'out-string',
    'write-output',
    'write-host',
    'get-command',
    'gcm',
    'get-help',
    'help',
    'get-member',
    'gm',
}

# Windows 破坏性命令（含 cmd/PowerShell 删改、提权、包安装、下载执行、以及跨平台的 git/sql/infra）。
WINDOWS_DESTRUCTIVE_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        # 删除 / 改名 / 移动 / 写入
        r'\bremove-item\b',
        r'\bdel\b',
        r'\berase\b',
        r'\brmdir\b',
        r'\brd\b',
        r'\bmove-item\b',
        r'\brename-item\b',
        r'\bren\b',
        r'\bnew-item\b',
        r'\bset-content\b',
        r'\badd-content\b',
        r'\bout-file\b',
        r'\bclear-content\b',
        r'\bset-itemproperty\b',
        # 进程 / 服务 / 电源
        r'\bstop-process\b',
        r'\bstop-service\b',
        r'\bstop-computer\b',
        r'\brestart-computer\b',
        r'\bshutdown\b',
        # 提权 / 权限 / 注册表
        r'\bset-executionpolicy\b',
        r'\btakeown\b',
        r'\bicacls\b',
        r'\bcacls\b',
        r'\battrib\b',
        r'\breg\s+(delete|add)\b',
        r'\bregedit\b',
        r'\bnet\s+(user|localgroup)\b',
        r'\bsc(\.exe)?\s+delete\b',
        # 磁盘
        r'\bformat\b',
        r'\bformat-volume\b',
        r'\bclear-disk\b',
        # 下载执行 / 任意代码
        r'\binvoke-expression\b',
        r'\biex\b',
        r'(iwr|invoke-webrequest|curl|wget)\s+[^|]*\|\s*(iex|invoke-expression)',
        r'\bpowershell(\.exe)?\b.*\s-enc(odedcommand)?\b',
        r'\bstart-process\b',
        # 包安装
        r'\bpip3?\s+install\b',
        r'\bpip3?\s+uninstall\b',
        r'\bnpm\s+(install|i)\b',
        r'\byarn\s+(add|install)\b',
        r'\bpnpm\s+(add|install)\b',
        r'\bchoco\s+(install|uninstall|upgrade)\b',
        r'\bwinget\s+(install|uninstall|upgrade)\b',
        r'\binstall-module\b',
        r'\binstall-package\b',
        # 跨平台 git 破坏性
        r'git\s+push\s+.*(-f|--force)',
        r'git\s+reset\s+--hard',
        r'git\s+clean\s+.*-[a-zA-Z]*f',
        r'git\s+rebase',
        # SQL / 基础设施（与 posix 一致）
        r'(DROP|TRUNCATE)\s+(TABLE|DATABASE|SCHEMA)',
        r'DELETE\s+FROM\s+\w+\s*(;|$)',
        r'docker\s+(rm|rmi|system\s+prune)',
        r'kubectl\s+delete',
        r'terraform\s+destroy',
    ]
]

# PowerShell 里 backtick 是转义/续行符（非命令替换），故 windows 的 unsafe 只看子表达式与 iex。
_WINDOWS_UNSAFE_PATTERNS = re.compile(r'\$\(|\$\{|\biex\b|\binvoke-expression\b', re.IGNORECASE)

_UNSAFE_PATTERNS = re.compile(r'\$\(|`|\$\{')

# 无副作用重定向：丢弃到 /dev/null / $null，或 fd 复制（如 ``2>&1``），均不算写文件。
_HARMLESS_REDIRECT = re.compile(r'\d*>>?\s*(?:/dev/null|\$null)\b|\d*>>?&\s*\d+', re.IGNORECASE)
_WRITE_REDIRECT = re.compile(r'>>?')


def _has_write_redirect(segment: str) -> bool:
    """片段是否含写文件 / append 重定向。

    先剔除无副作用重定向（``2>/dev/null`` / ``2>&1`` / ``>$null``），残留的
    ``>`` / ``>>`` 才算写文件。 修此前把 ``2>/dev/null`` 误判为写而误杀只读命令的 bug。
    """
    stripped = _HARMLESS_REDIRECT.sub('', segment)
    return bool(_WRITE_REDIRECT.search(stripped))


def _split_command_segments(command: str) -> list[str]:
    """把复合命令按顶层 shell 操作符（``&&`` / ``||`` / ``;`` / ``|``）拆段。

    引号（``'`` ``"``）与反引号内的分隔符不拆，避免 ``echo "a;b"`` 被误拆；
    用于逐段分级，避免 ``cat a && rm -rf b`` 因首段只读而整体漏判。
    """
    segments: list[str] = []
    buf: list[str] = []
    quote: str | None = None
    i = 0
    n = len(command)
    while i < n:
        ch = command[i]
        if quote:
            buf.append(ch)
            if ch == quote:
                quote = None
            i += 1
            continue
        if ch in ('"', "'", '`'):
            quote = ch
            buf.append(ch)
            i += 1
            continue
        if command[i : i + 2] in ('&&', '||'):
            segments.append(''.join(buf))
            buf = []
            i += 2
            continue
        if ch in (';', '|'):
            segments.append(''.join(buf))
            buf = []
            i += 1
            continue
        buf.append(ch)
        i += 1
    segments.append(''.join(buf))
    return [s.strip() for s in segments if s.strip()]


def _first_token(segment: str) -> str:
    """取片段首个命令 token（跳过 ``sudo`` / ``doas`` 提权前缀），best-effort。"""
    try:
        tokens = shlex.split(segment, posix=True)
    except ValueError:
        tokens = segment.split()
    idx = 0
    while idx < len(tokens) and tokens[idx] in ('sudo', 'doas'):
        idx += 1
    return tokens[idx] if idx < len(tokens) else ''


# ═══════════════ Disaster Commands (hard deny) ═══════════════
# 灾难级命令：任何档位都硬拒（对齐端上 kDefaultForbidRules）。 此前归为 LOW/MEDIUM_HIGH，
# basic 档可被 mock confirmed 静默执行，属安全洞（见文档 4.4）。

_DISASTER_FIRST_TOKENS = {'mkfs', 'format', 'shred', 'srm', 'mkswap', 'fdisk'}


def _segment_is_disaster(segment: str) -> bool:
    first = _first_token(segment)
    if not first:
        return False
    if first in _DISASTER_FIRST_TOKENS or first.startswith('mkfs.'):
        return True
    if first == 'dd' and re.search(r'\bof=', segment):
        return True
    return first == 'diskutil' and 'erase' in segment.lower()


def is_disaster_command(command: str) -> bool:
    """命令是否含灾难级子命令（逐段判断）。 任何档位都应硬拒。"""
    return any(_segment_is_disaster(seg) for seg in (_split_command_segments(command) or [command]))


# ═══════════════ No-risk Tool Names ═══════════════

NO_RISK_TOOLS: set[str] = {
    'read',
    'load_skill',
    'ask_user_question',
    'web_search',
    'web_fetch',
    'media_generation',
    'task_schedule',
    'tencent_doc',
}


# ═══════════════ Path Helpers ═══════════════


def _expand_home(path: str) -> str:
    """Expand ~ to the home directory for pattern matching."""
    home = os.path.expanduser('~')
    if path.startswith('~/'):
        return home + path[1:]
    return path


def _normalize_win(path: str) -> str:
    """Windows 路径归一：正斜杠→反斜杠、折叠 ..、统一小写（盘符大小写不敏感）。"""
    normalized = ntpath.normpath(path.replace('/', '\\'))
    return normalized.lower()


def _matches_sensitive_file(path: str, os_type: OsType = 'posix') -> bool:
    if os_type == 'windows':
        norm = _normalize_win(path)
        return any(fnmatch.fnmatchcase(norm, _normalize_win(p)) for p in WINDOWS_SENSITIVE_FILE_PATTERNS)
    expanded = _expand_home(path)
    for pattern in SENSITIVE_FILE_PATTERNS:
        expanded_pattern = _expand_home(pattern)
        if fnmatch.fnmatch(expanded, expanded_pattern):
            return True
    return False


def _is_system_dir(path: str, os_type: OsType = 'posix') -> bool:
    if os_type == 'windows':
        norm = _normalize_win(path)
        for prefix in WINDOWS_SYSTEM_DIRS:
            nprefix = _normalize_win(prefix)
            if norm == nprefix or norm.startswith(nprefix + '\\'):
                return True
        return False
    expanded = _expand_home(path)
    return any(expanded.startswith(prefix) for prefix in SYSTEM_DIRS)


def _is_write_tool(tool_name: str) -> bool:
    return tool_name in ('write', 'edit')


def _extract_path_from_args(tool_name: str, tool_args: dict) -> str | None:
    if tool_name in ('read', 'write', 'edit'):
        return tool_args.get('file_path')
    return None


def _extract_paths_from_bash(command: str, os_type: OsType = 'posix') -> list[str]:
    """Best-effort extraction of file paths from a command. Not comprehensive."""
    paths = []
    for token in command.split():
        token = token.strip('"\'')
        if os_type == 'windows':
            # 盘符路径 X:\ 或 X:/、UNC \\host、相对 .\ 或 ./
            if re.match(r'^[a-zA-Z]:[\\/]', token) or token.startswith('\\\\') or token.startswith(('.\\', './')):
                paths.append(token)
        elif token.startswith('/') or token.startswith('~/') or token.startswith('./'):
            paths.append(token)
    return paths


def _is_under_root(path: str, root: str, os_type: OsType = 'posix') -> bool:
    """Check if path is inside a root directory.

    L4：相对路径先按 ``root`` 解析成绝对路径再判，避免相对路径被误判为区外。
    """
    if os_type == 'windows':
        norm_root = _normalize_win(root)
        raw = path.replace('/', '\\')
        if not ntpath.isabs(raw):
            raw = ntpath.join(root.replace('/', '\\'), raw)
        norm_path = _normalize_win(raw)
        return norm_path == norm_root or norm_path.startswith(norm_root + '\\')
    expanded = _expand_home(path)
    expanded_root = os.path.normpath(_expand_home(root))
    if not os.path.isabs(expanded):
        expanded = os.path.join(expanded_root, expanded)
    expanded_path = os.path.normpath(expanded)
    return expanded_path.startswith(expanded_root + os.sep) or expanded_path == expanded_root


def _is_in_workspace(
    path: str,
    workspace_path: str,
    skill_path: str | None = None,
    os_type: OsType = 'posix',
) -> bool:
    """Path 是否在工作区内。 L3：``skill_path`` 视同白名单（读/写 skill 目录判区内）。"""
    return _is_under_root(path, workspace_path, os_type) or bool(
        skill_path and _is_under_root(path, skill_path, os_type)
    )


# ═══════════════ Core Logic ═══════════════


def is_redline(tool_name: str, tool_args: dict, os_type: OsType = 'posix') -> bool:
    """Check if this tool call hits a red line (unconditional DENY)."""
    file_path = _extract_path_from_args(tool_name, tool_args)

    if file_path is not None:
        if _matches_sensitive_file(file_path, os_type):
            return True
        if _is_write_tool(tool_name) and _is_system_dir(file_path, os_type):
            return True

    if tool_name == 'bash':
        command = tool_args.get('command', '')
        for path in _extract_paths_from_bash(command, os_type):
            if _matches_sensitive_file(path, os_type):
                return True
        exfil = _COMPILED_WINDOWS_EXFIL if os_type == 'windows' else _COMPILED_EXFIL
        for pattern, _desc in exfil:
            if pattern.search(command):
                return True

    return False


# PowerShell 只读 cmdlet（小写匹配，PS 大小写不敏感）。 训练轨迹含 windows/powershell，
# 这些 cmdlet 名与 unix 命令不冲突，纳入同一只读判定即可降低 PS 只读命令被误判为需确认。
POWERSHELL_READONLY_CMDLETS: set[str] = {
    'get-childitem',
    'gci',
    'dir',
    'get-content',
    'gc',
    'type',
    'get-location',
    'gl',
    'get-item',
    'gi',
    'get-itemproperty',
    'select-string',
    'sls',
    'select-object',
    'where-object',
    'measure-object',
    'sort-object',
    'format-table',
    'format-list',
    'out-string',
    'write-output',
    'write-host',
    'test-path',
    'resolve-path',
    'get-command',
    'get-date',
    'compare-object',
    'get-process',
    'get-service',
    'get-help',
    'get-member',
    'get-alias',
    'get-variable',
    'get-psdrive',
    'split-path',
    'join-path',
    'convertto-json',
    'convertfrom-json',
    'group-object',
    'foreach-object',
    'findstr',
    'where',
}


def _is_readonly_bash(segment: str, os_type: OsType = 'posix') -> bool:
    """单个命令片段是否只读（白名单命令、无副作用）。

    ``segment`` 应为已按管道/操作符拆开的单段（见 ``_split_command_segments``）。
    """
    stripped = segment.strip()
    if not stripped:
        return False

    if os_type == 'windows':
        if _WINDOWS_UNSAFE_PATTERNS.search(stripped):
            return False
        # 复用无副作用重定向剔除逻辑：``2>$null`` / ``2>&1`` 不算写文件。
        if _has_write_redirect(stripped):
            return False
        for prefix in READONLY_COMMAND_PREFIXES:
            if stripped.lower().startswith(prefix):
                return True
        tokens = stripped.split()
        first_token = tokens[0].lower() if tokens else ''
        return first_token in WINDOWS_READONLY_COMMANDS

    if _UNSAFE_PATTERNS.search(stripped):  # $() / `` / ${}：无法静态确认 → 非只读
        return False
    if _has_write_redirect(stripped):  # 写文件 / append 重定向 → 非只读
        return False

    for prefix in READONLY_COMMAND_PREFIXES:
        if stripped.startswith(prefix):
            return True

    first_token = _first_token(stripped)
    if not first_token:
        return False
    if first_token in READONLY_COMMANDS:
        return True
    return first_token.lower() in POWERSHELL_READONLY_CMDLETS


def _is_destructive_bash(command: str, os_type: OsType = 'posix') -> bool:
    """Check if a command matches destructive patterns."""
    patterns = WINDOWS_DESTRUCTIVE_PATTERNS if os_type == 'windows' else DESTRUCTIVE_PATTERNS
    return any(pattern.search(command) for pattern in patterns)


def classify_risk(
    tool_name: str,
    tool_args: dict,
    workspace_path: str | None = None,
    skill_path: str | None = None,
    os_type: OsType = 'posix',
) -> RiskLevel:
    """Classify risk level for a tool call. Does NOT check red lines."""
    if tool_name in NO_RISK_TOOLS:
        return RiskLevel.NONE

    if tool_name == 'bash':
        command = tool_args.get('command', '')
        # 跨段模式（如 curl|bash 管道到 shell）在整条命令上判。
        if _is_destructive_bash(command, os_type):
            return RiskLevel.MEDIUM_HIGH
        # L1：复合命令逐段分级，取最严档（任一段破坏性 → MH；全只读 → NONE；否则 LOW）。
        segments = _split_command_segments(command) or [command]
        risk = RiskLevel.NONE
        for seg in segments:
            if _is_destructive_bash(seg, os_type):
                return RiskLevel.MEDIUM_HIGH
            if not _is_readonly_bash(seg, os_type):
                risk = RiskLevel.LOW
        return risk

    if tool_name in ('write', 'edit'):
        file_path = tool_args.get('file_path', '')
        if workspace_path and _is_in_workspace(file_path, workspace_path, skill_path, os_type):
            return RiskLevel.LOW
        return RiskLevel.MEDIUM_HIGH

    # L5：未知工具默认无风险（已知有副作用的工具都显式处理；工具集由配置管控）。
    return RiskLevel.NONE


class PermissionEngine:
    """Permission engine that checks tool calls against permission rules."""

    def __init__(self, local_context: LocalContext) -> None:
        self._ctx = local_context

    @property
    def _permission_level(self) -> str:
        return self._ctx.permission.level

    @property
    def _workspace_path(self) -> str | None:
        return self._ctx.workspace.path if self._ctx.workspace else None

    @property
    def _skill_path(self) -> str | None:
        return self._ctx.skill_path

    @property
    def _os_type(self) -> OsType:
        return 'windows' if self._ctx.runtime.os == 'windows' else 'posix'

    def check_permission(self, tool_name: str, tool_args: dict) -> Decision:
        if is_redline(tool_name, tool_args, self._os_type):
            return self._make_deny_decision(tool_name, tool_args)

        # 灾难级命令（磁盘擦除 / 格式化等）：任何档位硬拒，不进确权（对齐端上 forbid → readonly_denied）。
        if tool_name == 'bash' and is_disaster_command(tool_args.get('command', '')):
            return Decision(
                action=DecisionAction.DENY,
                reason='Permission denied: 检测到灾难级命令（磁盘擦除 / 格式化等），已被安全策略禁止。',
                redline_category='readonly_denied',
            )

        risk = classify_risk(
            tool_name,
            tool_args,
            workspace_path=self._workspace_path,
            skill_path=self._skill_path,
            os_type=self._os_type,
        )
        level = self._permission_level

        if risk == RiskLevel.NONE:
            return Decision(action=DecisionAction.ALLOW, risk_level=risk)

        if level == 'readonly':
            # 只读模式：非白名单命令 / 写改类一律拒绝，不弹确权、不给二次机会
            # （对齐最新端上 tool_registry_pre_hooks.cc：SDK Prompt/Forbidden 在
            # ReadOnly 档 → readonly_denied，no escalation；write/edit RO=Reject）。
            # 只读白名单命令 risk==NONE，已在上方直接 ALLOW，不会到这里。
            if tool_name in ('write', 'edit'):
                reason = 'Permission denied: 当前为只读模式，该操作需要用户切换到编辑权限后才能执行。'
            else:
                reason = 'Permission denied: 只读模式下一律拒绝写 / 改 / 执行类操作，需用户切换到编辑权限。'
            return Decision(
                action=DecisionAction.DENY,
                reason=reason,
                risk_level=risk,
                redline_category='readonly_denied',
            )

        # basic：低风险直接执行，中高风险确认。
        if risk == RiskLevel.LOW:
            return Decision(action=DecisionAction.ALLOW, risk_level=risk)

        return Decision(
            action=DecisionAction.ASK,
            reason='该操作被判定为中高风险，需要用户确认。',
            risk_level=risk,
        )

    def _make_deny_decision(self, tool_name: str, tool_args: dict) -> Decision:
        os_type = self._os_type
        file_path = _extract_path_from_args(tool_name, tool_args)

        if file_path and _matches_sensitive_file(file_path, os_type):
            return Decision(
                action=DecisionAction.DENY,
                reason='Permission denied: 该操作涉及系统敏感文件，已被安全策略禁止。',
                redline_category='redline_sensitive_file',
            )

        if file_path and _is_write_tool(tool_name) and _is_system_dir(file_path, os_type):
            return Decision(
                action=DecisionAction.DENY,
                reason='Permission denied: 不允许写入系统目录。',
                redline_category='redline_system_dir',
            )

        if tool_name == 'bash':
            command = tool_args.get('command', '')
            for path in _extract_paths_from_bash(command, os_type):
                if _matches_sensitive_file(path, os_type):
                    return Decision(
                        action=DecisionAction.DENY,
                        reason='Permission denied: 该操作涉及系统敏感文件，已被安全策略禁止。',
                        redline_category='redline_sensitive_file',
                    )
            exfil = _COMPILED_WINDOWS_EXFIL if os_type == 'windows' else _COMPILED_EXFIL
            for pattern, _desc in exfil:
                if pattern.search(command):
                    return Decision(
                        action=DecisionAction.DENY,
                        reason='Permission denied: 检测到数据外发行为，目标地址未在白名单中。',
                        redline_category='redline_exfiltration',
                    )

        return Decision(
            action=DecisionAction.DENY,
            reason='Permission denied: 红线操作，禁止执行。',
            redline_category='redline_unknown',
        )
