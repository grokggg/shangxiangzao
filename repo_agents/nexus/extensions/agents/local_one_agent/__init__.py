"""local_one_agent — permission-aware agent for local workspace scenarios."""

# ── nexus.core compat shim ────────────────────────────────────────────────────
# local_one_agent 复用 main_agent（``nexus.extensions.agents.main.agent``）的推理循环，
# 而 main_agent 在模块顶层执行 ``from nexus.core import ToolArgumentValidationError,
# ToolNameResolutionError``。本分支的 nexus.core 未再 re-export 这两个符号（它们只保留在
# nexus.core.tool 下），为避免改动 nexus.core / main 等共享模块，这里在导入 main_agent 之前
# 把它们补挂回 nexus.core 的命名空间。
#
# 该 __init__ 会在 local_one_agent 任一子模块（如 .agent）被 import 前先执行，因此 main_agent
# 的顶层 from-import 能取到这两个名字；由于操作的是同一个 nexus.core 模块对象，同样受影响的
# tool_search.tools 等既有消费者也一并得到修复。
import nexus.core as _core

from nexus.core.tool import (
    ToolArgumentValidationError as _ToolArgumentValidationError,
)
from nexus.core.tool import (
    ToolNameResolutionError as _ToolNameResolutionError,
)


if not hasattr(_core, 'ToolArgumentValidationError'):
    _core.ToolArgumentValidationError = _ToolArgumentValidationError  # type: ignore[unresolved-attribute]
if not hasattr(_core, 'ToolNameResolutionError'):
    _core.ToolNameResolutionError = _ToolNameResolutionError  # type: ignore[unresolved-attribute]

del _core, _ToolArgumentValidationError, _ToolNameResolutionError
