#!/usr/bin/env python3
import os, sys, base64, json, time, requests

TOKEN = "ghp_***REDACTED***"
OWNER, REPO, BRANCH = "grokggg", "shangxiangzao", "main"
API = "https://api.github.com"
DNAME = "/data/workspace/push_batch/api_upload"
MAX_BYTES = 900 * 1024
BATCH = 80

s = requests.Session()
s.headers.update({"Authorization": f"token {TOKEN}",
                  "Accept": "application/vnd.github.v3+json",
                  "User-Agent": "Mozilla/5.0"})

def gh(method, path, **kw):
    for a in range(4):
        r = getattr(s, method)(f"{API}{path}", **kw)
        if r.status_code in (502, 503, 429) or r.status_code >= 500:
            time.sleep(2 ** a); continue
        return r
    return r

def head():
    r = gh("get", f"/repos/{OWNER}/{REPO}/commits/{BRANCH}")
    if r.status_code != 200:
        print("FATAL ref:", r.status_code, r.text[:300]); sys.exit(1)
    cs = r.json()["sha"]
    cr = gh("get", f"/repos/{OWNER}/{REPO}/git/commits/{cs}")
    return cs, cr.json()["tree"]["sha"]

files = []
for fn in sorted(os.listdir(DNAME)):
    fp = os.path.join(DNAME, fn)
    if not os.path.isfile(fp): continue
    sz = os.path.getsize(fp)
    if sz > MAX_BYTES:
        print(f"  [SKIP] {fn} {sz}B"); continue
    files.append((fn, fp, sz))
print(f"files: {len(files)}")

batches = [files[i:i+BATCH] for i in range(0, len(files), BATCH)]
print(f"batches: {len(batches)}")

def one(batch, csha, tsha, msg):
    blobs = []
    for fn, fp, sz in batch:
        data = open(fp, "rb").read()
        try:
            data.decode("utf-8"); enc = "utf-8"; payload = data.decode("utf-8")
        except:
            enc = "base64"; payload = base64.b64encode(data).decode("ascii")
        r = gh("post", f"/repos/{OWNER}/{REPO}/git/blobs",
               json={"content": payload, "encoding": enc})
        if r.status_code != 201:
            print(f"  [BLOB FAIL] {fn}: {r.status_code} {r.text[:150]}"); return None
        blobs.append({"path": fn, "mode": "100644", "type": "blob", "sha": r.json()["sha"]})
    r = gh("post", f"/repos/{OWNER}/{REPO}/git/trees",
           json={"base_tree": tsha, "tree": blobs})
    if r.status_code != 201:
        print(f"  [TREE FAIL] {r.status_code} {r.text[:150]}"); return None
    tsha = r.json()["sha"]
    r = gh("post", f"/repos/{OWNER}/{REPO}/git/commits",
           json={"message": msg, "tree": tsha, "parents": [csha]})
    if r.status_code != 201:
        print(f"  [COMMIT FAIL] {r.status_code} {r.text[:150]}"); return None
    nsha = r.json()["sha"]
    r = gh("patch", f"/repos/{OWNER}/{REPO}/git/refs/heads/{BRANCH}",
           json={"sha": nsha, "force": False})
    if r.status_code != 200:
        print(f"  [REF FAIL] {r.status_code} {r.text[:150]}"); return None
    return (nsha, tsha)

log = []
for i, batch in enumerate(batches):
    csha, tsha = head()
    res = one(batch, csha, tsha, f"添加: 全量推送 {i+1}/{len(batches)} ({len(batch)} files)")
    if not res:
        print(f"  [FAIL] batch {i+1}"); break
    log.append({"batch": i+1, "commit": res[0], "files": len(batch)})
    print(f"  [OK] {i+1}/{len(batches)} -> {res[0][:10]} ({len(batch)} files)")
    time.sleep(0.4)

json.dump(log, open("/data/workspace/push_batch/_upload2_log.json","w"), indent=2)
print(f"\nDone: {len(log)}/{len(batches)} batches")
r = gh("get", f"/repos/{OWNER}/{REPO}/git/trees/HEAD?recursive=1")
print("Remote blobs:", len([t for t in r.json().get("tree",[]) if t["type"]=="blob"]))
