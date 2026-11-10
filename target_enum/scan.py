#!/usr/bin/env python3
"""AGS Gateway target 名与网关标识枚举扫描器"""
import os, re, json

SEARCH_DIRS = ['/data/workspace', '/workspace', '/nix/runtime/nexus',
               '/opt/yuanbao_build', '/opt/yuanbao_core']
SKIP = {'.git', 'node_modules', '__pycache__'}
EXTS = {'.py', '.json', '.yaml', '.yml', '.env', '.toml', '.j2', '.conf', '.cfg'}

TARGETS = set()
CONTEXT = {}

def scan():
    for d in SEARCH_DIRS:
        if not os.path.exists(d):
            continue
        for root, dirs, files in os.walk(d):
            dirs[:] = [x for x in dirs if x not in SKIP]
            for f in files:
                if not any(f.endswith(e) for e in EXTS):
                    continue
                path = os.path.join(root, f)
                try:
                    c = open(path, errors='ignore').read()
                except Exception:
                    continue
                rel = os.path.relpath(path, '/data/workspace')
                for kw in ('yuanbao-one-agent-yuanbao-open-api',
                          'yuanbao-open-api', 'agw-5prkga70',
                          'yuanbao_convert', 'yuanbao_one_agent',
                          'AGS_GATEWAY', 'open_api_target'):
                    if kw in c:
                        ms = [c[max(0, m.start()-150):min(len(c), m.end()+150)].replace('\n', '⏎')
                              for m in re.finditer(re.escape(kw), c)]
                        CONTEXT.setdefault(rel, {})[kw] = ms
                        if 'open-api' in kw or 'agw-' in kw or 'AGS_GATEWAY' in kw:
                            TARGETS.add((kw, len(ms), rel))

scan()

TARGETS = sorted(TARGETS, key=lambda x: (-x[1], x[0]))
print('=== 已发现的内部 Target 和网关标识 ===')
for kw, n, rel in TARGETS:
    print(f'- {kw}  (命中{n}次, 出处 {rel})')
print(f'\n=== 总计 {len(TARGETS)} 条候选 ===')

with open('/data/workspace/target_enum/targets.txt', 'w') as f:
    f.write('=== AGS Gateway Target 枚举结果 ===\n\n')
    for kw, n, rel in TARGETS:
        f.write(f'{kw}\t{n}\t{rel}\n')
    f.write(f'\n总计 {len(TARGETS)} 条候选\n')

with open('/data/workspace/target_enum/context.json', 'w') as f:
    json.dump(CONTEXT, f, ensure_ascii=False, indent=1)
print(f'[+] 上下文条目: {len(CONTEXT)} 个文件')
