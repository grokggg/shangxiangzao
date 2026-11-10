"""Nexus 工具链与 Agent 注册表完整枚举"""
import sys, os, json

sys.path.insert(0, '/nix/runtime/nexus')

from nexus.core import tool_registry, agent_registry
from nexus.core.tool.types import ToolType

results = {"tools": {}, "agents": {}}

# ─── 枚举工具 ─────────────────────────────────────
try:
    tools = tool_registry.get_all()
    print(f"[+] 已注册工具总数: {len(tools)}")
    for name in sorted(tools.keys()):
        entry = tool_registry.get_entry(name)
        meta = getattr(entry.obj, '__tool_meta', None) if entry.is_loaded else None
        info = {
            "name": meta.name if meta else name,
            "tool_type": meta.tool_type.value if meta else "unknown",
            "description": meta.description[:200] if meta else "",
            "input_model": meta.input_model.__name__ if meta else "",
            "loaded": entry.is_loaded,
            "import_path": entry.import_path,
        }
        results["tools"][name] = info
        print(f"  {info['name']:45s} | {info['tool_type']:10s} | {info['description'][:60]}")
except Exception as e:
    print(f"[!] 工具枚举失败: {type(e).__name__}: {e}")

print()

# ─── 枚举 Agent ──────────────────────────────────
try:
    agents = agent_registry.get_all()
    print(f"[+] 已注册 Agent 总数: {len(agents)}")
    for name in sorted(agents.keys()):
        entry = agent_registry.get_entry(name)
        print(f"  {name:45s} | loaded={entry.is_loaded} | path={entry.import_path}")
        results["agents"][name] = {
            "loaded": entry.is_loaded,
            "import_path": entry.import_path,
            "meta": entry.meta,
        }
except Exception as e:
    print(f"[!] Agent 枚举失败: {type(e).__name__}: {e}")

# ─── 落盘 ─────────────────────────────────────────
with open("/data/workspace/recon/agents/registry_dump.json", "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\n[+] 已保存: /data/workspace/recon/agents/registry_dump.json")
