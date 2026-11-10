"""Nexus Agent 注册表与 MCP 工具链侦察结果"""
from nexus.core import tool_registry, agent_registry

# 触发延迟加载
try:
    tools = tool_registry.get_all()
    print(f"已注册工具总数: {len(tools)}")
    for name, func in sorted(tools.items()):
        meta = getattr(func, '__tool_meta', None)
        if meta:
            print(f"  {meta.name:40s} | {meta.tool_type.value:10s} | {meta.description[:80]}")
        else:
            print(f"  {name:40s} | (no meta)")
except Exception as e:
    print(f"[!] 加载失败: {e}")

try:
    agents = agent_registry.get_all()
    print(f"\n已注册 Agent 总数: {len(agents)}")
    for name, func in sorted(agents.items()):
        print(f"  {name}")
except Exception as e:
    print(f"[!] Agent 加载失败: {e}")
