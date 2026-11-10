#!/usr/bin/env bash
set -euo pipefail

# Standard STEM-MAS benchmark runner using Nexus Dataset and RunPart compose.

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../../.." && pwd)}"
DEFAULT_ENV_FILE="nexus/extensions/tasks/stem_mas/benchmark/runners/benchmark.env"

resolve_path() {
  case "$1" in
    /*) printf '%s\n' "$1" ;;
    *) printf '%s\n' "$ROOT_DIR/$1" ;;
  esac
}

source_env_file() {
  if [ "${SKIP_ENV_FILE:-0}" = "1" ]; then
    return
  fi

  local env_file="${ENV_FILE:-${CONF_PATH:-$DEFAULT_ENV_FILE}}"

  env_file="$(resolve_path "$env_file")"
  if [ ! -f "$env_file" ]; then
    echo "ERROR: ENV_FILE not found: $env_file" >&2
    exit 2
  fi

  set -a
  # shellcheck source=/dev/null
  source "$env_file"
  set +a
}

source_env_file

# A Ray working directory starts from a clean environment. Let Nexus install
# host dependencies declared by lazily loaded agent, tool, and client modules.
export NEXUS_AUTO_INSTALL="${NEXUS_AUTO_INSTALL:-1}"

if [ "${STEM_MAS_CLEAR_MODEL_API_PROXY:-0}" = "1" ]; then
  export HTTP_PROXY=
  export HTTPS_PROXY=
  export http_proxy=
  export https_proxy=
  export NO_PROXY="localhost,127.0.0.1,.woa.com,woa.com,llm-api.model-eval.woa.com,api.model-eval.woa.com"
  export no_proxy="$NO_PROXY"
fi

require_env() {
  local missing=0
  for name in "$@"; do
    local value="${!name:-}"
    case "$value" in
      ""|REPLACE_ME_*|*REPLACE_ME_*)
        echo "ERROR: $name is not set or still contains a placeholder." >&2
        missing=1
        ;;
    esac
  done
  if [ "$missing" -ne 0 ]; then
    exit 2
  fi
}

require_file() {
  for path in "$@"; do
    if [ ! -f "$path" ]; then
      echo "ERROR: required file not found: $path" >&2
      exit 2
    fi
  done
}

require_env CONFIG RUN_DIR MAX_CONCURRENT DATABASE EXPORT_PATH CODE_INTERPRETER

CONFIG="$(resolve_path "$CONFIG")"
RUN_DIR="$(resolve_path "$RUN_DIR")"
DATABASE="$(resolve_path "$DATABASE")"
EXPORT_PATH="$(resolve_path "$EXPORT_PATH")"
MATERIALIZER="$ROOT_DIR/nexus/extensions/tasks/stem_mas/benchmark/runners/tools/materialize_nexus_dataset.py"
STEM_MAS_PROJECT="$ROOT_DIR/nexus/extensions/tasks/stem_mas/pyproject.toml"
CONFIG_ROOT="$ROOT_DIR/nexus/extensions/tasks/stem_mas/benchmark/eval/config"
DATASET_DIR="$RUN_DIR/nexus_dataset"

require_file "$CONFIG" "$MATERIALIZER" "$STEM_MAS_PROJECT"

PROFILE="$(python - "$CONFIG" "$CONFIG_ROOT" <<'PY'
import sys
from pathlib import Path

config_path = Path(sys.argv[1]).resolve()
config_root = Path(sys.argv[2]).resolve()
try:
    profile = config_path.relative_to(config_root)
except ValueError as exc:
    raise SystemExit(f'Config must stay under {config_root}: {config_path}') from exc
print(profile.with_suffix('').as_posix())
PY
)"

mkdir -p "$RUN_DIR"

echo "=========================================="
echo "  stem_mas benchmark"
echo "=========================================="
echo "  Config:         $CONFIG"
echo "  Profile:        $PROFILE"
echo "  Dataset:        $DATASET_DIR"
echo "  Data