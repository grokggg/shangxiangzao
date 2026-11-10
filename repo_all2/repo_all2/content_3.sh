#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_VERSION="${NEXUS_BINARY_PYTHON_VERSION:-3.10}"
IMAGE="${NEXUS_BINARY_MANYLINUX_IMAGE:-quay.io/pypa/manylinux_2_28_x86_64}"
PLAT_TAG="${NEXUS_BINARY_PLAT_TAG:-manylinux_2_28_x86_64}"
OUT_ROOT="${NEXUS_BINARY_BUILD_ROOT:-$ROOT/build}"
WHEEL_DIR="${NEXUS_BINARY_WHEEL_DIR:-$ROOT/dist}"
WORKDIR_IN_CONTAINER="${NEXUS_BINARY_BUILD_WORKDIR:-/io}"
USE_SJTU_MIRRORS="${NEXUS_BINARY_USE_SJTU_MIRRORS:-1}"
PIP_INDEX_URL="${NEXUS_BINARY_PIP_INDEX_URL:-https://mirror.sjtu.edu.cn/pypi/web/simple}"
ALMALINUX_MIRROR_URL="${NEXUS_BINARY_ALMALINUX_MIRROR_URL:-https://mirrors.sjtug.sjtu.edu.cn/almalinux}"
EPEL_MIRROR_URL="${NEXUS_BINARY_EPEL_MIRROR_URL:-https://ftp.sjtu.edu.cn/epel}"
PACKAGE_NAME="${NEXUS_BINARY_PACKAGE_NAME:-}"
NUITKA_JOBS="${NUITKA_JOBS:-32}"
CCACHE_DIR_HOST="${NEXUS_BINARY_CCACHE_DIR:-$ROOT/.cache/nuitka-ccache}"
CCACHE_MAX_SIZE="${NEXUS_BINARY_CCACHE_MAX_SIZE:-5G}"
PIP_CACHE_DIR_HOST="${NEXUS_BINARY_PIP_CACHE_DIR:-$ROOT/.cache/pip}"

usage() {
  cat <<'EOF'
Usage: deployment/pypi/build_nuitka_binary_wheel.sh [OPTIONS]

Build a compiled Nexus wheel with Nuitka inside a manylinux container.

Options:
  --python VERSION_OR_TAG  Python version/tag, e.g. 3.10, 3.11, cp310-cp310.
  --image IMAGE           manylinux image. Default: quay.io/pypa/manylinux_2_28_x86_64.
  --plat TAG              wheel platform tag. Default: manylinux_2_28_x86_64.
  --out-dir DIR           output wheel directory. Default: dist.
  -h, --help              show this help.

Examples:
  deployment/pypi/build_nuitka_binary_wheel.sh --python 3.10
  deployment/pypi/build_nuitka_binary_wheel.sh --python cp311-cp311 --plat manylinux_2_28_x86_64

Environment:
  NEXUS_BINARY_USE_SJTU_MIRRORS=0   Disable SJTU yum/pip mirror setup.
  NEXUS_BINARY_PACKAGE_NAME=NAME     Override wheel distribution name.
  NUITKA_JOBS=8                     Override Nuitka parallel compile jobs.
  NEXUS_BINARY_CCACHE_DIR=PATH      Persistent ccache directory.
  NEXUS_BINARY_CCACHE_MAX_SIZE=SIZE  Persistent ccache max size. Default: 5G.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python)
      PYTHON_VERSION="${2:?missing value for --python}"
      shift 2
      ;;
    --image)
      IMAGE="${2:?missing value for --image}"
      shift 2
      ;;
    --plat)
      PLAT_TAG="${2:?missing value for --plat}"
      shift 2
      ;;
    --out-dir)
      WHEEL_DIR="${2:?missing value for --out-dir}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

python_tag_from_version() {
  case "$1" in
    cp*-cp*) printf '%s\n' "$1" ;;
    3.10|310|cp310) printf 'cp310-cp310\n' ;;
    3.11|311|cp311) printf 'cp311-cp311\n' ;;
    3.12|312|cp312) printf 'cp312-cp312\n' ;;
    3.13|313|cp313) printf 'cp313-cp313\n' ;;
    *)
      echo "Unsupported Python version/tag: $1" >&2
      echo "Use one of 3.