#!/usr/bin/env bash

set -euo pipefail

LOG=/var/log/dockerd-bootstrap.log
# /run must be created before /var/run: in some images /var/run is a symlink to
# a missing /run, and `mkdir -p /var/run` then fails on the dangling symlink.
mkdir -p /var/log /run /var/run

if [ -S /var/run/docker.sock ] && docker info >/dev/null 2>&1; then
    echo "dockerd already running"
    exit 0
fi

if ! iptables --version 2>/dev/null | grep -q legacy; then
    echo "FATAL: iptables must use the legacy backend for Docker-in-Docker under gVisor" >&2
    iptables --version 2>&1 | head -1 >&2 || true
    exit 1
fi

dev="$(ip route show default | awk '/default/ {print $5; exit}')"
if [[ -z "$dev" ]]; then
    echo "FATAL: cannot determine default network device for Docker-in-Docker SNAT" >&2
    exit 1
fi

addr="$(ip -4 addr show dev "$dev" | awk '/inet / {print $2; exit}' | cut -d/ -f1)"
if [[ -z "$addr" ]]; then
    echo "FATAL: cannot determine IPv4 address for device $dev" >&2
    exit 1
fi

echo 1 > /proc/sys/net/ipv4/ip_forward
iptables -t nat -A POSTROUTING -o "$dev" -j SNAT --to-source "$addr" -p tcp 2>>"$LOG" || true
iptables -t nat -A POSTROUTING -o "$dev" -j SNAT --to-source "$addr" -p udp 2>>"$LOG" || true

# Snapshot the outer sandbox's env whitelist for dind-runc to inject into inner
# containers. MUST run before the `export SSL_CERT_FILE` below — otherwise the
# snapshot would capture docker-bootstrap.sh's dockerd-specific CA path
# (/var/run/dind-ca-bundle.crt) instead of the original outer value
# (/etc/ssl/certs/ca-certificates.crt), and inner containers would get a
# SSL_CERT_FILE pointing to a file that doesn't exist inside them.
INJECT_ENV=/var/run/dind-inject.env
: >"$INJECT_ENV"
for k in \
    HTTP_PROXY HTTPS_PROXY FTP_PROXY NO_PROXY \
    http_proxy https_proxy ftp_proxy no_proxy \
    PIP_INDEX_URL PIP_EXTRA_INDEX_URL UV_DEFAULT_INDEX UV_INDEX_URL \
    UV_NATIVE_TLS CARGO_HTTP_MULTIPLEXING GRADLE_OPTS \
    SSL_CERT_FILE REQUESTS_CA_BUNDLE NODE_EXTRA_CA_CERTS; do
    v="${!k:-}"; [[ -n "$v" ]] && printf '%s=%s\n' "$k" "$v" >>"$INJECT_ENV"
done

# Build a CA bundle for dockerd's TLS verification when pulling images.
# Outbound traffic goes through a squid proxy that does TLS interception, so the
# registry cert dockerd sees is re-signed by a custom squid CA — without it
# dockerd fails with "x509: certificate signed by unknown authority". The task
# image rootfs (e.g. ubuntu:24.04) may also lack ca-certificates entirely.
# We therefore concatenate a public-root bundle (best effort) with the custom
# squid CA into one file and point dockerd at it via SSL_CERT_FILE. This is
# registry-agnostic: it applies to every registry dockerd talks to.
DIND_CA_BUNDLE=/var/run/dind-ca-bundle.crt
SQUID_CA=/usr/local/share/ca-certificates/squid-proxy-ca.crt
: >"$DIND_CA_BUNDLE"
for base_ca in \
    /nix/runtime/nix-env/etc/ssl/certs/ca-bundle.crt \
    /etc/ssl/certs/ca-certificates.crt \
    /etc/ssl/certs/ca-bundle.crt \
    /nix/store/*-nss-cacert-*/etc/ssl