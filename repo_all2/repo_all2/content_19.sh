#!/usr/bin/env bash
# Example: evaluate ONE SWE-Cosmos instance from an instances.jsonl.
#
#   bash nexus/extensions/tasks/swe/swe_cosmos/run_single.swecosmos.sh \
#       /data/swecosmos/instances.jsonl  swe-env.v1.aws__s2n-tls-3982.function_rewrite_16r3qwsq
#
# Each row of the jsonl MUST carry dataset_type: "swe-cosmos" plus repo_dict / patch / test_patch / FAIL_TO_PASS / PASS_TO_PASS / image / workdir (see config.swecosmos.yaml for the full data contract).
set -euo pipefail

cd "$(git rev-parse --show-toplevel 2>/dev/null || echo .)"

INSTANCES="${1:?usage: run_single.example.sh <instances.jsonl> <instance_id>}"
INSTANCE_ID="${2:?usage: run_single.example.sh <instances.jsonl> <instance_id>}"

# --- Sandbox (Gongfeng) credentials ------------------------------------------
# Sandbox API base URL: prefer SWE_RUNTIME_BASE_URL, fall back to the legacy GONGFENG_API_BASE_URL.
export SWE_RUNTIME_BASE_URL="${SWE_RUNTIME_BASE_URL:-${GONGFENG_API_BASE_URL:-http://hyrl-sandbox.prod.woa.com:8052}}"
# Sandbox API token: prefer GONGFENG_RUNTIME_TOKEN, fall back to the legacy GONGFENG_TOKEN.
export GONGFENG_RUNTIME_TOKEN="${GONGFENG_RUNTIME_TOKEN:-${GONGFENG_TOKEN:-}}"
: "${GONGFENG_RUNTIME_TOKEN:?set GONGFENG_RUNTIME_TOKEN (or legacy GONGFENG_TOKEN) (sandbox API token)}"
export GONGFENG_PURPOSE="${GONGFENG_PURPOSE:-test}"
# --- Host-side LLM the agent talks to ----------------------------------------
export OPENAI_CLIENT_NAME="${OPENAI_CLIENT_NAME:-litellm}"
export OPENAI_MODEL="${OPENAI_MODEL:-bedrock/anthropic.claude-opus-4-7}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://trpc-gpt-eval.production.polaris:8080}"
export OPENAI_API_KEY="${OPENAI_API_KEY:?set OPENAI_API_KEY}"

export SWE_AGENT_NAME="${SWE_AGENT_NAME:-swe_agent}"
export SWE_MAX_ITERATIONS="${SWE_MAX_ITERATIONS:-100}"
# Whether to run the oracle (only for debugging)
export SWE_RUN_ORACLE="${SWE_RUN_ORACLE:-0}"
export SWE_EVAL_TIMEOUT_SECONDS="${SWE_EVAL_TIMEOUT_SECONDS:-1800}"
# Set SWE_WORKER_VENV=false to instead use the runtime image's baked nexus.
export SWE_WORKER_VENV="${SWE_WORKER_VENV:-true}"
export SWE_TOOLS_MODE="${SWE_TOOLS_MODE:-branch_last}"

OUTPUT_PATH="${OUTPUT_PATH:-./outputs/swecosmos_eval/${INSTANCE_ID}.json}"
mkdir -p "$(dirname "${OUTPUT_PATH}")"

# Render the example config with this instance, then run it. We keep the YAML as the single source of truth and only override dataset_path / instance_id.
TMP_CFG="$(mktemp --suffix=.yaml)"
trap 'rm -f "${TMP_CFG}"' EXIT
sed \
    -e "s#^\(  dataset_path:\).*#\1 \"${INSTANCES}\"#" \
    -e "s#^\(  instance_id:\).*#\1 \"${INSTANCE_ID}\"#" \
    nexus/extensions/tasks/swe/swe_cosmos/config.swecosmos.yaml > "${TMP_CFG}"

uv run nexus run-single "${TMP_CFG}" -o "${OUTPUT_PATH}"

echo "Result written to: ${OUTPUT_PATH}"
