#!/usr/bin/env bash
# dind-runc — runc wrapper for the inner dockerd inside the DinD sandbox.
#
# At `create` it rewrites the OCI spec:
#   1) bind-mounts the merged CA bundle (public roots + Squid CA) and the
#      squid-only CA into fixed in-container paths;
#   2) injects proxy / mirror / CA env into .process.env (respecting user-set
#      values via `docker run -e`);
#   3) when the image ships a shell, appends a `startContainer` OCI hook that
#      installs the squid CA into the container's own trust stores
#      (system store / certifi / java) — same effect as suanlicluster.py,
#      executed in the container mount ns after pivot_root so it never touches
#      the host.
# Every other runc subcommand is passed through untouched. Any injection
# failure is swallowed so it can never abort container creation.
#
# Runtime state files produced by docker-bootstrap.sh:
#   /var/run/dind-ca-bundle.crt  — public roots + Squid CA (CA env target src)
#   /var/run/dind-inject.env     — proxy + mirror env whitelist snapshot
#   /usr/local/share/ca-certificates/squid-proxy-ca.crt — squid-only CA (host)
#
# This script is packaged by the nix flake into
# /nix/runtime/nix-env/bin/dind-runc — it does NOT need runtime code
# generation. The real runc is resolved from non-wrapper nix store entries to
# avoid recursion after PATH and moby libexec/docker/runc are repointed here;
# jq still uses PATH with a fixed nix-env fallback.
set -euo pipefail

# Resolve the real runc binary. dind-runc is installed as "runc" in both the
# nix profile and moby's libexec/docker/runc, so we must not use
# `command -v runc` — it may resolve back to this wrapper. Instead, select a
# non-wrapper runc from the nix store. When BuildKit reaches us through moby's
# libexec/docker/runc path, prefer docker-runc; otherwise prefer standalone runc
# and fall back to docker-runc if standalone runc is not present in the closure.
RUNC_BIN=""
case "$0" in
    *moby*libexec/docker/runc*|*libexec/docker/runc*)
        for c in /nix/store/*-docker-runc-*/bin/runc; do
            [[ -x "$c" ]] && { RUNC_BIN="$c"; break; }
        done
        ;;
    *)
        for c in /nix/store/*-runc-*/bin/runc; do
            case "$c" in *docker-runc*|*dind-runc*) continue ;; esac
            [[ -x "$c" ]] && { RUNC_BIN="$c"; break; }
        done
        if [[ ! -x "$RUNC_BIN" ]]; then
            for c in /nix/store/*-docker-runc-*/bin/runc; do
                [[ -x "$c" ]] && { RUNC_BIN="$c"; break; }
            done
        fi
        ;;
esac
if [[ ! -x "$RUNC_BIN" ]]; then
    echo "dind-runc: FATAL: cannot locate real runc binary (called as $0)" >&2
    exit 1
fi

JQ_BIN="$(command -v jq 2>/dev/null || true)"
[[ -x "$JQ_BIN"   ]] || JQ_BIN=/nix/runtime/nix-env/bin/jq

CA=/var/run/dind-ca-bundle.crt        # public roots + Squid CA (from docker-bootstrap.sh)
DST=/etc/ssl/dind-proxy-ca.crt        # in-container mount of the FULL bundle (CA env target)
SQUID_SRC=/etc/ssl/dind-squid-ca.crt  # in-cont