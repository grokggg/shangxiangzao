
import subprocess, os, json

os.chdir('/data/workspace/repo_agents')
env = dict(os.environ)
env['GIT_SSL_CAINFO'] = '/etc/zeroproxy-tls/ca-bundle.pem'
env['GIT_TERMINAL_PROMPT'] = '0'

# 推送
proc = subprocess.Popen(['git', 'push', '-f', 'origin', 'main'], 
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
out, _ = proc.communicate(timeout=60)

result = {
    'push_returncode': proc.returncode,
    'push_output': out.decode()[:2000]
}

# 验证
r2 = subprocess.run(['curl', '-s', '-H', 'Authorization: token ghp_***REDACTED***',
    'https://api.github.com/repos/grokggg/shangxiangzao/commits'],
    capture_output=True, text=True, timeout=30)
commits = json.loads(r2.stdout) if r2.returncode == 0 else []
result['total_commits'] = len(commits)
result['latest_commits'] = [{'sha': c['sha'][:10], 'msg': c['commit']['message'][:80]} for c in commits[:5]]

r3 = subprocess.run(['curl', '-s', '-H', 'Authorization: token ghp_***REDACTED***',
    'https://api.github.com/repos/grokggg/shangxiangzao/contents/agents'],
    capture_output=True, text=True, timeout=30)
try:
    agents_content = json.loads(r3.stdout)
    if isinstance(agents_content, list):
        result['agents_files'] = [f['name'] for f in agents_content]
    else:
        result['agents_files'] = f"ERROR: {agents_content}"
except:
    result['agents_files'] = r3.stdout[:500]

json.dump(result, open('/data/workspace/push_result.json', 'w'), indent=2)
print(json.dumps(result, indent=2))
