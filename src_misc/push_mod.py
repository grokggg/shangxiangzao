
import subprocess, os, json

def run():
    os.chdir('/data/workspace/repo_agents')
    env = dict(os.environ)
    env['GIT_SSL_CAINFO'] = '/etc/zeroproxy-tls/ca-bundle.pem'
    env['GIT_TERMINAL_PROMPT'] = '0'
    
    proc = subprocess.Popen(['git', 'push', '-f', 'origin', 'main'], 
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
    out, _ = proc.communicate(timeout=60)
    
    r2 = subprocess.run(['curl', '-s', '-H', 'Authorization: token ghp_***REDACTED***',
        'https://api.github.com/repos/grokggg/shangxiangzao/commits'],
        capture_output=True, text=True, timeout=30)
    commits = json.loads(r2.stdout) if r2.returncode == 0 else []
    
    r3 = subprocess.run(['curl', '-s', '-H', 'Authorization: token ghp_***REDACTED***',
        'https://api.github.com/repos/grokggg/shangxiangzao/contents/agents'],
        capture_output=True, text=True, timeout=30)
    try:
        agents_content = json.loads(r3.stdout)
        agents_files = [f['name'] for f in agents_content] if isinstance(agents_content, list) else f"ERR:{agents_content}"
    except:
        agents_files = r3.stdout[:500]
    
    return json.dumps({
        'push_returncode': proc.returncode,
        'push_output': out.decode()[:1000],
        'total_commits': len(commits),
        'latest_commits': [{'sha': c['sha'][:10], 'msg': c['commit']['message'][:80]} for c in commits[:5]],
        'agents_files': agents_files,
    }, indent=2)
