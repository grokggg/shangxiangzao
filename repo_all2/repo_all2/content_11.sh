#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "${SCRIPT_DIR}/../../../../.." && pwd)
BUILD_ROOT="${SCRIPT_DIR}/.build"
SYNCED_DATASETS_DIR="${BUILD_ROOT}/datasets"
IMAGE_NAME="${MCP_BENCHMARK_MAIN_IMAGE:-nexus-mcp-benchmark-main:latest}"
UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-/tmp/nexus-mcp-benchmark-build-venv}"
read -r -a DOCKER_BIN <<< "${NEXUS_DOCKER_CMD:-docker}"

mkdir -p "${BUILD_ROOT}"
env UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT}" uv run python "${SCRIPT_DIR}/helpers/sync_dataset_configs.py" \
  --source-root "${PROJECT_ROOT}/workspace/mcp/datasets" \
  --output-root "${SYNCED_DATASETS_DIR}"

"${DOCKER_BIN[@]}" build \
  -f "${SCRIPT_DIR}/Dockerfile" \
  -t "${IMAGE_NAME}" \
  "${SCRIPT_DIR}"
