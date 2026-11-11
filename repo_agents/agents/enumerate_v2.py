"""直接读取 registry 内部状态，逐条懒加载"""
import sys, json
sys.path.insert(0, '/nix/runtime/nexus')

from nexus.core import tool_registry, agent_registry

results = {"tools": {}, "agents": {}}

# ─── 工具：先列出所有 key，再逐条 get ─────────────
print("=== Tool Registry Keys ===")
tool_names = list(tool_registry._registry.keys())
print(f"Total keys: {len(tool_names)}")

for name in sorted(tool_names):
    entry = tool_registry.get_entry(name)
    info = {
        "loaded": entry.is_loaded,
        "import_path": entry.import_path,
        "meta_keys": list(entry.meta.keys()) if entry.meta else [],
    }
    if entry.is_loaded:
        meta = getattr(entry.obj, '__tool_meta', None)
        if meta:
            info["tool_name"] = meta.name
            info["tool_type"] = meta.tool_type.value
            info["description"] = meta.description[:150]
            info["input_model"] = meta.input_model.__name__
    results["tools"][name] = info
    status = "✓" if entry.is_loaded else "○"
    print(f"  {status} {name}")

print(f"\n=== Agent Registry Keys ===")
agent_names = list(agent_registry._registry.keys())
print(f"Total keys: {len(agent_names)}")
for name in sorted(agent_names):
    entry = agent_registry.get_entry(name)
    results["agents"][name] = {
        "loaded": entry.is_loaded,
        "import_path": entry.import_path,
        "meta": {k: str(v)[:100] for k, v in (entry.meta or {}).items()},
    }
    status = "✓" if entry.is_loaded else "○"
    print(f"  {status} {name}")

with open("/data/workspace/recon/agents/registry_dump.json", "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\n[+] 已保存: registry_dump.json ({len(results['tools'])} tools, {len(results['agents'])} agents)")
