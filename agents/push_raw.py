#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
原样推送本地缺失文件到 grokggg/shangxiangzao。
规则：
1. 不脱敏、不修改、不重写源码；按字节原文推。
2. 文件内容若含 token 敏感串也一并原样推送（用户明确要求保留原始源码）。
3. 先拉远程 tree 比对，仅推缺失文件，避免重复与误覆盖。
4. 每批 80 文件；每个 blob 建树后单次 commit；强制 fast-forward。
"""
import os, json, base64, urllib.request, urllib.error

TOKEN = "ghp_REDACTED_PLACEHOLDER"
OWNER = "grokggg"
REPO = "shangxiangzao"
ROOT = "/data/workspace"
BATCH = 80

HEADERS_JSON = {
    "Authorization": f"token {TOKEN}",
    "Accept": "application/vnd.github+json",
    "Content-Type": "application/json",
}


def gh(method, path, body=None, parse=True):
    url = f"https://api.github.com{path}"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers=HEADERS_JSON)
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            return (json.loads(r.read().decode()) if parse else None), r.status
    except urllib.error.HTTPError as e:
        return {"__http_error__": e.code, "body": e.read().decode(errors="replace")}, e.code


def list_remote():
    obj, _ = gh("GET", f"/repos/{OWNER}/{REPO}/git/trees/HEAD?recursive=1")
    return set(x["path"] for x in obj.get("tree", []) if x["type"] == "blob")


def collect():
    skip_ext = {".pkl", ".pyc"}
    out = []
    for cur, ds, fs in os.walk(ROOT):
        ds[:] = [d for d in ds if not d.startswith(".")]
        for fn in fs:
            if fn.startswith("."):
                continue
            full = os.path.join(cur, fn)
            if os.path.splitext(fn)[1].lower() in skip_ext:
                continue
            rel = os.path.relpath(full, ROOT).replace("\\", "/")
            if "/.git/" in rel:
                continue
            out.append((rel, full))
    return out


def push_batch(items, base_sha, msg):
    blobs = []
    for rel, full in items:
        with open(full, "rb") as f:
            raw = f.read()
        b64 = base64.b64encode(raw).decode()
        obj, st = gh("POST", f"/repos/{OWNER}/{REPO}/git/blobs",
                     {"content": b64, "encoding": "base64"})
        if st != 201:
            print(f"  [BLOB FAIL] {st} {rel} {obj}")
            continue
        blobs.append({"path": rel, "mode": "100644", "type": "blob", "sha": obj["sha"]})

    if not blobs:
        return base_sha, 0

    tree, st = gh("POST", f"/repos/{OWNER}/{REPO}/git/trees",
                  {"base_tree": base_sha, "tree": blobs})
    if st != 201:
        print(f"  [TREE FAIL] {st} {tree}")
        return base_sha, 0

    cmt, st = gh("POST", f"/repos/{OWNER}/{REPO}/git/commits",
                 {"message": msg, "tree": tree["sha"], "parents": [base_sha]})
    if st != 201:
        print(f"  [COMMIT FAIL] {st} {cmt}")
        return base_sha, 0

    ref_body = {"sha": cmt["sha"], "force": True}
    _, st = gh("PATCH", f"/repos/{OWNER}/{REPO}/git/refs/heads/main", ref_body, parse=False)
    if st not in (200, 202):
        print(f"  [REF FAIL] {st}")
        return base_sha, 0
    return cmt["sha"], len(blobs)


def main():
    print("[*] 拉取远程 tree ...")
    remote = list_remote()
    print(f"[*] 远程现有 blob: {len(remote)}")

    all_files = collect()
    missing = [(r, f) for r, f in all_files if r not in remote]
    print(f"[*] 本地文件 {len(all_files)}, 需推送 {len(missing)}")

    ref, _ = gh("GET", f"/repos/{OWNER}/{REPO}/git/refs/heads/main")
    sha = ref["object"]["sha"]
    print(f"[*] base sha: {sha}")

    pushed = 0
    i = 0
    for start in range(0, len(missing), BATCH):
        batch = missing[start:start + BATCH]
        i += 1
        sha, n = push_batch(batch, sha, f"raw push batch {i}: {len(batch)} files (no sanitize)")
        pushed += n
        print(f"[+] batch {i}: {n} files pushed, new sha {sha[:10]}")

    print(f"\n[DONE] 累计推送 {pushed} 文件")

    remote2 = list_remote()
    still = [r for r, _ in all_files if r not in remote2]
    print(f"[*] 本地 {len(all_files)} / 远程命中 {len(all_files) - len(still)} / 仍缺 {len(still)}")
    if still:
        with open(os.path.join(ROOT, "agents", "still_missing.json"), "w") as f:
            json.dump(still, f, ensure_ascii=False, indent=2)
        print("[*] 缺口已写入 agents/still_missing.json")


if __name__ == "__main__":
    main()
