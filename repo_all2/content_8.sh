#!/usr/bin/env bash

set -euo pipefail

# Set default port and host
DEFAULT_PORT=8000
DEFAULT_HOST="::"
DEFAULT_PY_VER="3.13"
PORT=$DEFAULT_PORT
HOST=$DEFAULT_HOST
PY_VER=$DEFAULT_PY_VER

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -p|--port)
            PORT="$2"
            shift 2
            ;;
        -H|--host)
            HOST="$2"
            shift 2
            ;;
        -v|--python-version)
            PY_VER="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  -p, --port PORT              Specify service port (default: $DEFAULT_PORT)"
            echo "  -H, --host HOST              Specify service host (default: $DEFAULT_HOST)"
            echo "  -v, --python-version VERSION Specify Python version (default: $DEFAULT_PY_VER)"
            echo "  -h, --help                   Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown parameter: $1"
            echo "Use -h or --help for help"
            exit 1
            ;;
    esac
done

export PATH="/nix/runtime/nix-env/bin:$PATH"

NEXUS_ENABLE_DIND_VALUE="${NEXUS_ENABLE_DIND:-}"
if [[ "${NEXUS_ENABLE_DIND_VALUE,,}" == "true" ]]; then
    DIND_BOOTSTRAP_SCRIPT="${NEXUS_DIND_BOOTSTRAP_SCRIPT:-/nix/runtime/nix-env/bin/docker-bootstrap.sh}"
    if ! command -v dockerd >/dev/null 2>&1; then
        echo "[start.sh] FATAL: NEXUS_ENABLE_DIND=true but dockerd not found in PATH" >&2
        exit 1
    fi
    if [[ ! -x "$DIND_BOOTSTRAP_SCRIPT" ]]; then
        echo "[start.sh] FATAL: docker bootstrap script not executable: $DIND_BOOTSTRAP_SCRIPT" >&2
        exit 1
    fi
    if ! "$DIND_BOOTSTRAP_SCRIPT"; then
        echo "[start.sh] FATAL: docker bootstrap failed, aborting runtime server startup." >&2
        exit 1
    fi
fi

GUI_STACK_SCRIPT="${NEXUS_GUI_STACK_SCRIPT:-/nix/runtime/nexus/deployment/runtime/gui_stack.sh}"
if [[ ! -f "$GUI_STACK_SCRIPT" ]]; then
    GUI_STACK_SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/gui_stack.sh"
fi
source "$GUI_STACK_SCRIPT"

echo "Starting service on host $HOST port $PORT with Python $PY_VER..."
if [[ "${NEXUS_GUI_AUTO_START:-0}" == "1" ]]; then
    if ! start_gui_stack; then
        echo "GUI stack startup failed; aborting runtime server startup." >&2
        exit 1
    fi
else
    echo "GUI stack auto-start disabled; call gui_start executor to enable desktop/VNC."
fi

# 在调用 uvicorn.run 之前清理 sys.path，移除 CWD，防止主镜像包覆盖 venv 包
# 必须在 import uvicorn 之前做，否则 Config.__init__ 会先污染 sys.path
/nix/runtime/nexus/.venv-${PY_VER}/bin/python -c "
import os, sys
_cwd = os.getcwd()
# 移除所有可能的 CWD 表示形式
sys.path = [p for p in sys.path if p not in ('', '.', _cwd)]
# 先加载 app 模块并配置 server 地址，worker 进程启动时通过 CLI 参数获取该地址
from nexus.runtime.server.app import configure, asgi_app
configure(port=$PORT)
import uvicorn
uvicorn.run(asgi_app, host='$HOST', port=$PORT, ws_max_size=100 * 1024 * 1024)
"
