#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-}"

BASE_IMAGE="${BASE_IMAGE:-mirrors.tencent.com/hunyuan_yanguan/nexus-runtime:latest}"
UV_INDEX_URL="${UV_INDEX_URL:-https://mirrors.cloud.tencent.com/pypi/simple/}"
UV_EXTRA_INDEX_URL="${UV_EXTRA_INDEX_URL:-https://mirrors.cloud.tencent.com/repository/pypi/tencent_pypi/simple/}"
LOCAL_IMAGE="${IMAGE_NAME:-nexus-benchmark:local}"
REGISTRY_IMAGE="${REGISTRY_IMAGE:-mirrors.tencent.com/asi/nexus-eval}"
REGISTRY_SCHEME="${REGISTRY_SCHEME:-https}"
PUSH="${PUSH:-true}"

get_local_ip() {
  if [[ -n "${LOCAL_IP:-}" ]]; then
    echo "${LOCAL_IP}"
    return
  fi
  if command -v ip >/dev/null 2>&1; then
    ip route get 1 2>/dev/null | awk '{for (i=1; i<=NF; i++) if ($i=="src") {print $(i+1); exit}}'
    return
  fi
  if command -v hostname >/dev/null 2>&1; then
    hostname -I 2>/dev/null | awk '{print $1}'
    return
  fi
}

find_repo_root() {
  if [[ -n "${REPO_ROOT}" ]]; then
    echo "${REPO_ROOT}"
    return
  fi
  local dir="${SCRIPT_DIR}"
  while [[ "${dir}" != "/" ]]; do
    if [[ -f "${dir}/pyproject.toml" ]]; then
      echo "${dir}"
      return
    fi
    dir="$(dirname "${dir}")"
  done
  echo "Failed to locate repo root (pyproject.toml). Set REPO_ROOT." >&2
  exit 1
}

TAG="${IMAGE_TAG:-tmp-$(get_local_ip)}"
if [[ -z "${TAG}" ]]; then
  echo "Failed to detect local IP. Set LOCAL_IP or IMAGE_TAG." >&2
  exit 1
fi
TAG="${TAG//:/-}"

REMOTE_IMAGE="${REGISTRY_IMAGE}:${TAG}"

get_registry_parts() {
  local image="$1"
  REGISTRY_HOST="${image%%/*}"
  REGISTRY_REPO="${image#*/}"
}

delete_remote_tag() {
  local image="$1"
  local tag="$2"
  get_registry_parts "${image}"

  local auth=()
  if [[ -n "${REGISTRY_USER:-}" && -n "${REGISTRY_PASSWORD:-}" ]]; then
    auth=(-u "${REGISTRY_USER}:${REGISTRY_PASSWORD}")
  fi

  local manifest_url="${REGISTRY_SCHEME}://${REGISTRY_HOST}/v2/${REGISTRY_REPO}/manifests/${tag}"
  local digest
  digest=$(curl -fsI "${auth[@]}" \
    -H "Accept: application/vnd.docker.distribution.manifest.v2+json" \
    "${manifest_url}" \
    | awk 'tolower($1)=="docker-content-digest:" {print $2}' \
    | tr -d '\r' \
    || true)

  if [[ -z "${digest}" ]]; then
    return 0
  fi

  local delete_url="${REGISTRY_SCHEME}://${REGISTRY_HOST}/v2/${REGISTRY_REPO}/manifests/${digest}"
  if ! curl -fsS -X DELETE "${auth[@]}" "${delete_url}" >/dev/null; then
    echo "Failed to delete existing tag: ${image}:${tag}" >&2
    exit 1
  fi
}

if [[ "${PUSH}" == "true" ]]; then
  delete_remote_tag "${REGISTRY_IMAGE}" "${TAG}"
fi

# Build local image.
BUILD_CONTEXT="$(find_repo_root)"

docker build \
  -f "${SCRIPT_DIR}/Dockerfile" \
  -t "${LOCAL_IMAGE}" \
  --build-arg BASE_IMAGE="${BASE_IMAGE}" \
  --build-arg UV_INDEX_URL="${UV_INDEX_URL}" \
  --build-arg UV_EXTRA_INDEX_URL="${UV_EXTRA_INDEX_URL}" \
  "${BUILD_CONTEXT}"

# Tag for registry.
docker tag "${LOCAL_IMAGE}" "${REMOTE_IMAGE}"

if [[ "${PUSH}" == "true" ]]; t