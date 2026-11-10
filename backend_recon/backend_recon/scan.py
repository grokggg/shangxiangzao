"""
backend.py 与推理后端、JWT/签名签发点 全量定位扫描
"""
import os, re, sys, json, hashlib, base64

SCAN = [
    '/usr/local/lib/python3.10/dist-packages/nexus',
    '/usr/local/lib/python3.10/dist-packages/yuanbao_convert',
    '/usr/local/lib/python3.10/dist-packages/ags_runtime',
    '/usr/local/lib/python3.10/dist-packages/yuanbao_one_agent',
    '/data/workspace/nexus',
    '/data/workspace/repo_all/nexus',
    '/data/workspace/repo_agents/nexus',
    '/nix/runtime/nexus/nexus',
    '/tmp/nexus',
    '/data/workspace/push_bypass/nexus',
]
if os.environ.get('NEXUS_PATH'):
    SCAN += [p for p in os.environ['NEXUS_PATH'].split(':') if p.strip()]

SKIP = re.compile(r'(node_modules|\.git|__pycache__|\.pyc$)')
URL_RE = re.compile(rb'(?:https?://|wss?://|grpc(?:s)?://)[^\s\'"<>\\]{4,200}')
REALNET = re.compile(r'(woa\.com|tencent|gateway\.ap-|polaris|qcloud|yuanbao|hunyuan|taiji|cloud\.tencent|cos\.ap-|litellm|myqcloud|miitang)')

results = {
    'scan_dirs': SCAN,
    'real_urls': [],
    'url_hits': [],
    'backend_classes': [],
    'jwt_lines': [],
    'header_lines': [],
    'hmac_sha256': [],
    'csrf': [],
    'tai_identity': [],
    'env_vars': [],
}

for base in SCAN:
    if not os.path.exists(base):
        continue
    for root, ds, fs in os.walk(base):
        if SKIP.search(root):
            continue
        for fn in fs:
            if not fn.endswith(('.py', '.json', '.yaml', '.yml', '.toml', '.conf', '.ini', '.env', '.j2', '.jinja')):
                continue
            fp = os.path.join(root, fn)
            try:
                with open(fp, 'rb') as fh:
                    data = fh.read()
            except Exception:
                continue
            if len(data) > 3000000:
                continue
            txt = data.decode('utf-8', 'replace')

            for m in URL_RE.findall(data):
                try:
                    u = m.decode()
                except Exception:
                    u = m.decode('latin-1', 'replace')
                results['url_hits'].append({'file': fp, 'url': u})
                if REALNET.search(u) and not re.search(r'[`"\'}{$]', u):
                    results['real_urls'].append(u)

            if 'backend.py' in fn and 'gateway' in root:
                for i, line in enumerate(txt.split('\n'), 1):
                    if re.search(r'class\s+\w*Backend|def\s+\w*(backend|infer|generate|chat)\w*|HyTuner|VLLM|OpenAI|Vllm', line):
                        results['backend_classes'].append({
                            'file': fp, 'ln': i, 'line': line.strip()[:260]})

            if re.search(r'\b(jwt|HyTuner|ServerBackend|sign\(|encode.*token|jwt\.encode|Bearer|Authorization)\b', txt, re.I):
                for i, line in enumerate(txt.split('\n'), 1):
                    if re.search(r'\b(jwt|HyTuner|ServerBackend|jwt\.encode|secret|sign\(|authorization|bearer)\b', line, re.I):
                        results['jwt_lines'].append({'file': fp, 'ln': i, 'line': line.strip()[:280]})
                    if re.search(r'headers\s*\[|header=|\.headers\s*=|Header\(|X-(?:Forwarded|Real|Target|Gateway|Source|Host|Request-Id)', line, re.I):
                        results['header_lines'].append({'file': fp, 'ln': i, 'line': line.strip()[:280]})

            if 'hmac' in txt or 'sha256' in txt:
                for i, line in enumerate(txt.split('\n'), 1):
                    if re.search(r'hmac\.|sha256|HMAC', line):
                        results['hmac_sha256'].append({'file': fp, 'ln': i, 'line': line.strip()[:280]})

            if 'csrf' in txt.lower() or 'tai' in txt.lower():
                for i, line in enumerate(txt.split('\n'), 1):
                    if re.search(r'csrf|tai_identity|application_token', line, re.I):
                        results['csrf'].append({'file': fp, 'ln': i, 'line': line.strip()[:280]})
                    if re.search(r'x-tai-identity|x-rio-|decrypt_tai_identity|verify_sgw', line, re.I):
                        results['tai_identity'].append({'file': fp, 'ln': i, 'line': line.strip()[:280]})

results['real_urls'] = sorted({u for u in results['real_urls'] if len(u) < 300})
for k in ('url_hits', 'jwt_lines', 'header_lines', 'hmac_sha256', 'csrf', 'tai_identity'):
    seen = set(); uniq = []
    for r in results[k]:
        key = (r.get('file'), r.get('ln'))
        if key in seen:
            continue
        seen.add(key); uniq.append(r)
    results[k] = uniq

print('SCAN_DIRS=', results['scan_dirs'])
print('URL_HITS=', len(results['url_hits']))
print('REAL_URLS=', len(results['real_urls']))
print('BACKEND_CLASSES=', len(results['backend_classes']))
print('JWT_LINES=', len(results['jwt_lines']))
print('HDR_LINES=', len(results['header_lines']))
print('HMAC_LINES=', len(results['hmac_sha256']))
print('CSRF=', len(results['csrf']))
print('TAI_IDENTITY=', len(results['tai_identity']))

with open('/data/workspace/backend_recon/raw.json', 'w') as fh:
    json.dump(results, fh, ensure_ascii=False, indent=1)
print('[+] raw.json written')
