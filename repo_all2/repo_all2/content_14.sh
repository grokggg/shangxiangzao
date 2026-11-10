#!/bin/bash
# one_agent_infer — 多模型推理/评测脚本（原 YB OneAgentBench runner）
#
# 依次使用多个模型评测，每个模型的结果存在各自的路径下：
#   outputs/{bench_name}_{ts}/{model_name}/
#     ├── results.jsonl
#     ├── results.db
#     ├── files/
#     └── html/
#
# 用法:
#   bash run_infer.sh                          # 默认参数
#   bash run_infer.sh --max-concurrent 6       # 自定义并发数
#   bash run_infer.sh --csig-api-key sk-xxx    # 注入 CSIG 私有 key（也可放 config/.env）
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# 从 SCRIPT_DIR 向上查找包含 pyproject.toml 的目录作为 NEXUS_ROOT
_dir="$SCRIPT_DIR"
while [ "$_dir" != "/" ]; do
    if [ -f "$_dir/pyproject.toml" ]; then
        NEXUS_ROOT="$_dir"
        break
    fi
    _dir="$(dirname "$_dir")"
done
if [ -z "${NEXUS_ROOT:-}" ]; then
    echo "ERROR: Cannot find nexus project root (no pyproject.toml found above $SCRIPT_DIR)"
    exit 1
fi

# 默认参数
BENCH_NAME="YB_OneAgentBench_V0.2"
BENCH_ROOT="${NEXUS_ROOT}/yb-agent-openbench/benchmarks/releases/${BENCH_NAME}"
CONFIG="${SCRIPT_DIR}/config/base.yaml"
OUTPUT_BASE="${NEXUS_ROOT}/outputs"
MAX_CONCURRENT=5
RESUME_DIR=""  # 如果指定，复用已有目录（断点续跑）
LIMIT=""       # 只跑前 N 条
NUDGE_ON_EMPTY=""  # 空响应时 prefill 引导次数（0=关闭）
ENABLE_AUQ="${ENABLE_AUQ:-true}"  # 是否启用 ask_user_question 工具（true/false，默认 true）
HISTORY_MODE="${HISTORY_MODE:-qa_only}"  # 多轮 session 中 history 拼接方式：
                                         # "qa_only"=只保留 user query + assistant answer（默认）
                                         # "full"=完整保留所有 message（含 tool_calls 和 tool response）

# 待评测的模型列表
MODELS=(
    # "openai/HY3.0-dev-0405-OneAgent-SFT-v0.2"
    # "openai/HY3.0-dev-0405-OneAgent-SFT-v0.3"
    # "openai/HY3.0-dev-0405-OneAgent-SFT-v0.4"
    # "openai/HY3.0-dev-0405-OneAgent-SFT-v0.5"
    # "openai/HY3.0-dev-0405-OneAgent-SFT-v0.4.1"
    # "openai/glm-5"
    # "openai/HY3.0-dev-0405-OneAgent-SFT-v0.2.8-step600"
    # "openai/HY3.0-dev-0405-OneAgent-SFT-v0.2.4-step600_HF_w8a8c8"
    # "openai/HY3.0-dev-0405-OneAgent-SFT-v0.3.3-mtp-step845"
    # "openai/HY3.0-dev-0405-OneAgent-SFT-v0.3.2-mtp-step845_w8a8c8"
    # "openai/HY3.0-dev-0405-OneAgent-SFT-v0.2.4-step600-2"
    # "openai/HY3.0-dev-0405-OneAgent-SFT-v0.2.3-step600"
    # "openai/HY3.0-dev-0405-OneAgent-SFT-v0.3.4-mtp-step614"
    # "openai/aws-claude-sonnet-4-6"
    # "openai/HY3.0-dev-0424-OneAgent-SFT-v1.3.2-mtp-HF-step465"
    # "openai/HY3.0-dev-0424-OneAgent-SFT-v1.3.3-mtp-HF-step465"
    # "openai/HY3.0-dev-0424-OneAgent-SFT-v1.2.2-mtp-step381-smooth-2"
    # "openai/HY3.0-dev-0424-OneAgent-SFT-v1.2.5-mtp-HF-step66"
    # "openai/deepops-claude-opus-4-6"
    # "openai/hy3-dev0405-jim"
    # "openai/dev-0424"
    # "openai/claude-sonnet-4.6"
    # "openai/claude-opus-4-6"
    "openai/taiji/hy3-a3b-sft-v2.3.1.1-step310_HF"
    # "openai/aws-claude-opus-4-6"
    # "openai/claude-sonnet-4-5-20250929"
)

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case "$1" in
        --max-concurrent) MAX_CONCURRENT="$2"; shift 2 ;;
        --bench-root)     BENCH_ROOT="$2";     shift 2 ;;
   