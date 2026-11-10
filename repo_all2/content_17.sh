#!/usr/bin/env bash
set -euo pipefail

# Submit a stem_mas benchmark run to a Ray Jobs platform.
#
# This script uploads the current checkout as the Ray working directory and
# runs runners/run_batch_template.sh inside the job. Keep secrets in env vars or
# an untracked conf file; do not write them into benchmark configs.

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../../.." && pwd)}"
DEFAULT_ENV_FILE="nexus/extensions/tasks/stem_mas/benchmark/runners/benchmark.env"

resolve_local_path() {
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

  env_file="$(resolve_local_path "$env_file")"
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

require_env \
  RAY_ADDRESS \
  JOB_ID \
  CONFIG \
  RUNNER_SCRIPT \
  RUN_DIR \
  MAX_CONCURRENT \
  DATABASE \
  EXPORT_PATH \
  CODE_INTERPRETER
resolved_config="$(resolve_local_path "$CONFIG")"
require_file "$resolved_config" "$(resolve_local_path "$RUNNER_SCRIPT")"
export_if_set() {
  for name in "$@"; do
    if [ -n "${!name:-}" ]; then
      export "$name"
    fi
  done
}

RUNTIME_ENV_NAMES=(
  http_proxy \
  https_proxy \
  HTTP_PROXY \
  HTTPS_PROXY \
  NO_PROXY \
  no_proxy \
  UV_DEFAULT_INDEX \
  NEXUS_AUTO_INSTALL \
  SEARCH_USER \
  SEARCH_API_KEY \
  SEARCH_STATEFUL_JINA_PROVIDER \
  STEM_MAS_DATASET_NAME \
  STEM_MAS_DATASET_PATH \
  STEM_MAS_DATASET_LIMIT \
  SCIENCE_API_KEY \
  SCIENCE_BASE_URL \
  STEM_MAS_CLEAR_MODEL_API_PROXY \
  GENERATION_MODEL \
  GENERATION_MAX_TOKENS \
  GENERATION_MAX_RETRIES \
  GENERATION_TIMEOUT \
  GENERATION_THINKING_TYPE \
  GENERATION_ENABLE_THINKING \
  GENERATION_REASONING_EFFORT \
  GENERATION_TAIJI_POLARIS_SERVICE \
  GONGFENG_API_BASE_URL \
  GONGFENG_RUNTIME_TOKEN \
  STEM_MAS_IMAGE \
  STEM_MAS_RUNTIME_IMAGE \
  JUDGE_MODEL \
  TAIJI_AUTH \
  TAIJI_NS \
  TAIJI_POLARIS_SERVICE \
  TAIJI_URL_PATTERN \
  TAIJI_WSID \
  STEM_MAS_EVALUATOR_TYPE \
  NEXUS_REWARD_SYSTEM_TOKEN \
  NEXUS_REWARD_SYSTEM_EXP_ID \
  NEXUS_REWARD_SYSTEM_ROUTER_ID \
  NEXUS_REWARD_SYSTEM_CI_TEST \
  NEXUS_REWARD_SYSTEM_EXTRA_JSON \
  CODEX_VERSION \
  CODEX_MODEL \
  CODEX_NPM_REGISTRY \
  LEAN_LSP_MCP_URL \
  LEAN_LSP_MCP_TIMEOUT_S \
  LEAN_LSP_MCP_TOOL_TIMEOUT_S \
  LEAN4_VERI