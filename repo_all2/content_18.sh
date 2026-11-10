#!/usr/bin/env bash
# Example: evaluate MANY SWE-Cosmos instances in parallel from an instances.jsonl.
#
#   bash nexus/extensions/tasks/swe/swe_cosmos/run_batch.swecosmos.sh /data/swecosmos/instances.jsonl
#
# This reuses the shared SWE batch generator (nexus/extensions/tasks/swe/swe_cosmos/generator.py), which reads the local jsonl and propagates each row's dataset_type. Every row MUST carry dataset_type: "swe-cosmos" so swe_task routes to the SWE-Cosmos handler, plus repo_dict / patch / test_patch / FAIL_TO_PASS / PASS_TO_PASS / image / workdir.
set -euo pipefail

cd "$(git rev-parse --show-toplevel 2>/dev/null || echo .)"

# --- Data source -------------------------------------------------------------
# Debugging data (101 instances): /apdcephfs_qy4/share_302593112/petrichorqi/data/swe-cosmos/swe-env/debug/swe-env.v1.debug_instances.jsonl
export SWE_DATASET_PATH="${1:-${SWE_DATASET_PATH:-/apdcephfs_qy4/share_302593112/petrichorqi/data/swe-cosmos/swe-env/debug/swe-env.v1.debug_instances.jsonl}}"
# Optional subset / paging:
# Include = whitelist (empty = all); Exclude = blacklist (applied after include).
# Each is a comma-separated list of literal instance_ids and/or .json/.jsonl file paths.
# export SWE_INCLUDE_INSTANCE_IDS="swe-env.v1.aws__s2n-tls-3982.function_rewrite_16r3qwsq"
# export SWE_EXCLUDE_INSTANCE_IDS="/data/swecosmos/skip.jsonl"
# export SWE_SKIP=0
# export SWE_LIMIT=0   # unset/0 = no limit

# --- Sandbox (Gongfeng) ------------------------------------------------------
# Sandbox API token: prefer GONGFENG_RUNTIME_TOKEN, fall back to the legacy GONGFENG_TOKEN.
export GONGFENG_RUNTIME_TOKEN="${GONGFENG_RUNTIME_TOKEN:-${GONGFENG_TOKEN:-}}"
: "${GONGFENG_RUNTIME_TOKEN:?set GONGFENG_RUNTIME_TOKEN (or legacy GONGFENG_TOKEN) (sandbox API token)}"
export GONGFENG_PURPOSE="${GONGFENG_PURPOSE:-test}"

# --- Agent + host-side LLM ---------------------------------------------------
export SWE_AGENT_NAME="${SWE_AGENT_NAME:-swe_agent}"
export SWE_MAX_ITERATIONS="${SWE_MAX_ITERATIONS:-300}"
# Host LLM backend: "litellm" (AWS/bedrock, default) or "vertex_anthropic" (GCP).
# A single generator.py handles both; only the OPENAI_MODEL default differs (picked below).
export OPENAI_CLIENT_NAME="${OPENAI_CLIENT_NAME:-litellm}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://trpc-gpt-eval.production.polaris:8080}"
export OPENAI_API_KEY="${OPENAI_API_KEY:?set OPENAI_API_KEY}"

# Re-raise infra errors so failed instances can be retried by the batch layer.
export SWE_RAISE_ON_RUNTIME_ERROR="${SWE_RAISE_ON_RUNTIME_ERROR:-1}"
# Whether to run the oracle (only for debugging)
export SWE_RUN_ORACLE="${SWE_RUN_ORACLE:-0}"

MAX_CONCURRENT="${MAX_CONCURRENT:-20}"
WORKER_PROCESSES="${WORKER_PROCESSES:-8}"
DB="${DB:-./outputs/swecosmos_eval_batch.db}"
EXPORT_PATH="${EXPORT_PATH:-./outputs/swecosmos_eval/batch_results.jsonl}"
mkdir -p "$(dirname "${DB}")" "$(dirname "${EXPORT_PATH}")"

# Single generator handles both backends, switching on OPENAI_CLIENT_NAME. Only