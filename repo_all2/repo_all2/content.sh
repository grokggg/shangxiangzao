#!/nix/runtime/nix-env/bin/bash
set -Eeuo pipefail

NEXUS_ENVD_BIN="${NEXUS_ENVD_BIN:-/usr/bin/envd}"
NEXUS_ENVD_PORT="${NEXUS_ENVD_PORT:-49983}"
NEXUS_RUNTIME_BASH="${NEXUS_RUNTIME_BASH:-/nix/runtime/nix-env/bin/bash}"
NEXUS_RUNTIME_START_SCRIPT="${NEXUS_RUNTIME_START_SCRIPT:-/nix/runtime/nexus/deployment/runtime/start.sh}"
NEXUS_RUNTIME_HOST="${NEXUS_RUNTIME_HOST:-0.0.0.0}"
NEXUS_RUNTIME_PORT="${NEXUS_RUNTIME_PORT:-32000}"
NEXUS_PYTHON_VERSION="${NEXUS_PYTHON_VERSION:-3.10}"

# Per-file size limit (in 512-byte blocks) for all commands spawned by the bash tool.
# Default: 204800 blocks = 100 MiB. Override by setting NEXUS_ULIMIT_FSIZE_BLOCKS at runtime.
# Set to "unlimited" to disable.
NEXUS_ULIMIT_FSIZE_BLOCKS="${NEXUS_ULIMIT_FSIZE_BLOCKS:-204800}"
if [[ "$NEXUS_ULIMIT_FSIZE_BLOCKS" != "unlimited" ]]; then
  ulimit -f "$NEXUS_ULIMIT_FSIZE_BLOCKS" || true
fi

ENVD_PID=""
RUNTIME_PID=""

log() {
  echo "$(date '+%Y-%m-%d %H:%M:%S') | $*"
}

terminate_pid() {
  local pid="$1"
  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    kill "$pid" 2>/dev/null || true
  fi
}

cleanup() {
  local exit_code=$?
  set +e
  terminate_pid "$RUNTIME_PID"
  terminate_pid "$ENVD_PID"
  wait "$RUNTIME_PID" 2>/dev/null || true
  wait "$ENVD_PID" 2>/dev/null || true
  exit "$exit_code"
}

forward_signal() {
  log "received stop signal, shutting down"
  terminate_pid "$RUNTIME_PID"
  terminate_pid "$ENVD_PID"
}

trap cleanup EXIT
trap forward_signal TERM INT

if [[ ! -x "$NEXUS_ENVD_BIN" ]]; then
  log "envd binary not found: ${NEXUS_ENVD_BIN}"
  exit 1
fi

if [[ ! -x "$NEXUS_RUNTIME_BASH" ]]; then
  log "bash interpreter not found: ${NEXUS_RUNTIME_BASH}"
  exit 1
fi

log "starting envd on port ${NEXUS_ENVD_PORT}"
"$NEXUS_ENVD_BIN" -port "$NEXUS_ENVD_PORT" &
ENVD_PID=$!

sleep 1
if ! kill -0 "$ENVD_PID" 2>/dev/null; then
  log "envd exited unexpectedly during startup"
  exit 1
fi

log "starting nexus runtime via ${NEXUS_RUNTIME_START_SCRIPT}"
"$NEXUS_RUNTIME_BASH" "$NEXUS_RUNTIME_START_SCRIPT" \
  --host "$NEXUS_RUNTIME_HOST" \
  --port "$NEXUS_RUNTIME_PORT" \
  --python-version "$NEXUS_PYTHON_VERSION" &
RUNTIME_PID=$!

wait "$RUNTIME_PID"