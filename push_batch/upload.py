#!/usr/bin/env python3
"""Batch upload all files in api_upload/ to GitHub repo via REST API."""
import os, sys, base64, json, time, hmac, hashlib

TOKEN = "ghp_REDACTED_PLACEHOLDER"
OWNER = "grokggg"
REPO = "shangxiangzao"
BRANCH = "main"
API = "https://api.github.com"

DNAME = "/data/workspace/push_batch/api_upload"
MAX_BYTES = 900 * 1024  # GitHub blob limit ~1MB, stay safe
MAX_FILES_PER_COMMIT = 80
SLEEP = 0.3  # rate limit friendliness

s = __import__("requests").Session()
s.headers.update({
    "Authorization": f"token {TOKEN}",
    "Accept": "application/vnd.github.v3+json",
    "User-Agent": "Mozilla/5.0",
})

def gh(method, path, **kw):
    for attempt in range(3):
        r = getattr(s, method)(f"{API}{path}", **kw)
        if r.status_code == 404 and method == "get":
            return r
        if r.status_code in (502, 503, 429) or (r.status_code >= 500):
            wait = 2 ** attempt
            print(f"  [retry {attempt+1}] {r.status_code} -> sleep {wait}s")
            time.sleep(wait)
            continue
        return r
    return r

# 1. Get latest commit + tree
r = gh("get", f"/repos/{OWNER}/{REPO}/git/refs/heads/{BRANCH}")
if r.status_code != 200:
    print("FATAL: cannot fetch ref:", r.status_code, r.text[:500])
    sys.exit(1)
base_sha = r.json()["object"]["sha"]

r = gh("get", f"/repos/{OWNER}/{REPO}/git/commits/{base_sha}")
if r.status_code != 200:
    print("FATAL: cannot fetch commit:", r.status_code)
    sys.exit(1)
base_tree = r.json()["tree"]["sha"]

# 2. Collect files, skip oversized
all_files = []
for fn in sorted(os.listdir(DNAME)):
    fp = os.path.join(DNAME, fn)
    if not os.path.isfile(fp):
        continue
    sz = os.path.getsize(fp)
    if sz > MAX_BYTES:
        print(f"  [SKIP too large] {fn} ({sz} B)")
        continue
    all_files.append((fn, fp, sz))

print(f"Total files to upload: {len(all_files)}")

# 3. Batch upload
def upload_batch(batch, base_tree_sha, base_commit_sha, msg):
    """base_tree_sha = tree SHA (for building new tree),
    base_commit_sha = parent commit SHA (for creating commit)."""
    blobs = []
    for fn, fp, sz in batch:
        with open(fp, "rb") as f:
            content = f.read()
        # Determine encoding
        try:
            content.decode("utf-8")
            encoding = "utf-8"
            payload = content.decode("utf-8")
        except:
            encoding = "base64"
            payload = base64.b64encode(content).decode("ascii")
        r = gh("post", f"/repos/{OWNER}/{REPO}/git/blobs",
               json={"content": payload, "encoding": encoding})
        if r.status_code != 201:
            print(f"  [BLOB FAIL] {fn}: {r.status_code} {r.text[:200]}")
            return None, None
        blobs.append({"path": fn, "mode": "100644", "type": "blob", "sha": r.json()["sha"]})
    r = gh("post", f"/repos/{OWNER}/{REPO}/git/trees",
           json={"base_tree": base_tree_sha, "tree": blobs})
    if r.status_code != 201:
        print(f"  [TREE FAIL] {r.status_code} {r.text[:200]}")
        return None, None
    tree_sha = r.json()["sha"]
    r = gh("post", f"/repos/{OWNER}/{REPO}/git/commits",
           json={"message": msg, "tree": tree_sha, "parents": [base_commit_sha]})
    if r.status_code != 201:
        print(f"  [COMMIT FAIL] {r.status_code} {r.text[:200]}")
        return None, None
    commit_sha = r.json()["sha"]
    r = gh("patch", f"/repos/{OWNER}/{REPO}/git/refs/heads/{BRANCH}",
           json={"sha": commit_sha, "force": False})
    if r.status_code != 200:
        print(f"  [REF FAIL] {r.status_code} {r.text[:200]}")
        return None, None
    return commit_sha, tree_sha, commit_sha

batches = [all_files[i:i+MAX_FILES_PER_COMMIT]
           for i in range(0, len(all_files), MAX_FILES_PER_COMMIT)]
print(f"Batches: {len(batches)}")

commit_log = []
for idx, batch in enumerate(batches):
    msg = f"添加: 批量推送 第{idx+1}/{len(batches)}批，共{len(batch)}个文件"
    res = upload_batch(batch, base_tree, base_sha, msg)
    if res is None or res[0] is None:
        print(f"  [BATCH FAIL] batch {idx+1}")
        break
    cs, ts, new_commit = res
    base_tree = ts
    base_sha = new_commit
    commit_log.append({"batch": idx+1, "commit": cs, "files": len(batch)})
    print(f"  [OK] batch {idx+1}/{len(batches)} -> {cs[:10]} ({len(batch)} files)")
    time.sleep(SLEEP)

with open("/data/workspace/push_batch/_upload_log.json", "w") as f:
    json.dump(commit_log, f, indent=2)

print(f"\nDone. {len(commit_log)} batches pushed.")
r = gh("get", f"/repos/{OWNER}/{REPO}/git/trees/HEAD?recursive=1")
print("Remote file count:", len([t for t in r.json().get("tree", []) if t["type"] == "blob"]))
