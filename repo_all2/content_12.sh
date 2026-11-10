#!/bin/bash

# NL2RepoBench 测试脚本 — 用于跑单道题验证流程

# ========== 路径配置 ==========
# 获取脚本所在目录的绝对路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 获取项目根目录 (脚本位于 nexus/extensions/tasks/nl2repo/tests/ 下)
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"

# 切换到项目根目录执行
cd "${PROJECT_ROOT}"

# ========== 数据集配置 ==========
export NL2REPO_DATASET_PATH="${SCRIPT_DIR}/test.jsonl"

# ========== Runtime 配置 ==========
export NL2REPO_RUNTIME_PROVIDER_NAME="gongfeng"
export NL2REPO_RUNTIME_BASE_URL="http://hyrl-sandbox.prod.woa.com:8052"
export NL2REPO_RUNTIME_TAG="{runtime_tag}"
export NL2REPO_RUNTIME_TIMEOUT=1800
export GONGFENG_RUNTIME_TOKEN="{gongfeng_runtime_token}"

# ========== Model / OpenAI 客户端配置 ==========
export OPENAI_CLIENT_NAME="litellm"
export OPENAI_API_KEY="{user_name}:{token}?provider=aws_third&timeout=300"
export OPENAI_BASE_URL="http://trpc-gpt-eval.production.polaris:8080"
export OPENAI_MODEL="bedrock/anthropic.claude-opus-4-6-v1"

# ========== Agent 配置 ==========
export NL2REPO_AGENT_NAME="codeact"
export NL2REPO_MAX_ITERATIONS=50

# ========== 可选：分批运行配置 ==========
# 跳过前 N 条
# export NL2REPO_SKIP=0

# 最多运行 N 条
# export NL2REPO_LIMIT=1

# 指定运行特定 instance_ids (逗号分隔)
# export NL2REPO_INSTANCE_IDS="aiofiles"

# ========== 执行批量任务 ==========
uv run nexus run-batch \
  -g "${PROJECT_ROOT}/nexus/extensions/tasks/nl2repo/generator.py" \
  -d "${PROJECT_ROOT}/workspace/nl2repo_batch_test.db" \
  --max-concurrent 1
