#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "${SCRIPT_DIR}/../../../../.." && pwd)
SERVICE_DIR="${SCRIPT_DIR}/services/atlas-mongodb"
BUILD_ROOT="${SERVICE_DIR}/.build"
IMAGE_NAME="${MCP_BENCHMARK_ATLAS_MONGODB_IMAGE:-nexus-mcp-benchmark-atlas-mongodb:latest}"
SOURCE_ZIP="${PROJECT_ROOT}/workspace/mcp/datasets/atlas/data_exports/mongo_dump_video_game_store-UNZIP-FIRST.zip"
read -r -a DOCKER_BIN <<< "${NEXUS_DOCKER_CMD:-docker}"

if [[ ! -f "${SOURCE_ZIP}" ]]; then
  echo "MongoDB dump zip not found: ${SOURCE_ZIP}" >&2
  exit 1
fi

rm -rf "${BUILD_ROOT}"
mkdir -p "${BUILD_ROOT}/dump"

env SOURCE_ZIP="${SOURCE_ZIP}" BUILD_ROOT="${BUILD_ROOT}" python3 - <<'PY'
from pathlib import Path
import zipfile
import os

source_zip = Path(os.environ['SOURCE_ZIP'])
build_root = Path(os.environ['BUILD_ROOT'])
target_root = build_root / 'dump'

with zipfile.ZipFile(source_zip) as archive:
    for member in archive.infolist():
        name = member.filename
        if not name or member.is_dir():
            continue
        if name.startswith('__MACOSX/') or '/.DS_Store' in name or name.endswith('/.DS_Store'):
            continue
        relative = Path(name)
        destination = target_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        with archive.open(member) as src, destination.open('wb') as dst:
            dst.write(src.read())
PY

"${DOCKER_BIN[@]}" build \
  -f "${SERVICE_DIR}/Dockerfile" \
  -t "${IMAGE_NAME}" \
  "${SERVICE_DIR}"
