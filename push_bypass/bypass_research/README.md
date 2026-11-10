# pyautogui_guard 绕过研究

## 测试目标
nexus/runtime/pyautogui_guard.py —— OSWorld PyAutoGUI 动作子集校验器

## 测试结论
**该 guard 对桌面动作子集的校验是完备的**，测试的全部 15 种绕过向量全部被拦截：

| 向量类别 | 测试数 | 绕过 |
|---|---|---|
| 字符串拼接 getattr | 4 | 0 |
| 动态属性 __dict__/__getattribute__ | 3 | 0 |
| hex/base64 exec 编码 | 2 | 0 |
| import 语句（含 import/as/from） | 3 | 0 |
| time.sleep 通道后续系统调用 | 1 | 0 |
| platform 方法通道后续系统调用 | 2 | 0 |
| if 控制流内嵌禁用方法 | 2 | 0 |

## 关键发现：guard 的作用域边界

pyautogui_guard 是**桌面动作子集校验器**，不是通用 shell 沙箱。
真实的命令执行链路是另一条：

```
nexus/runtime/executors/command.py
  → run_command() → asyncio.create_subprocess_shell()
  → env={**get_os_env_without_ld_library_path(), **(args.extra_env or {})}
  → BASH_ENV='~/.bashrc'
```

权限控制在 `nexus/extensions/agents/local_one_agent/`：
- permission_engine.py：纯函数式文件/命令/外发检查
- sandbox.py：bwrap 包装器，但仅 os=='macos' 生效；缺 bwrap/unshare 时降级为"仅权限引擎"
