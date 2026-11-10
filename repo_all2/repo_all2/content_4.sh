#!/bin/bash
# Publish to Tencent internal PyPI as 'tencent-nexus'.
# Optional compiled wheels are published as 'tencent-nexus-binary'.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
PYPROJECT="$PROJECT_DIR/pyproject.toml"

# Tencent internal PyPI config
PUBLISH_URL="https://mirrors.tencent.com/repository/pypi/tencent_pypi/simple"
CRED_FILE="$HOME/.nexus-pypi-tencent"

ORIGINAL_NAME="nexus-agent"
PUBLISH_NAME="tencent-nexus"
BINARY_PUBLISH_NAME="${TENCENT_PYPI_BINARY_NAME:-tencent-nexus-binary}"

# Set TENCENT_PYPI_BUILD_COMPILED=1 to build Nuitka compiled wheel(s) into
# the same dist/ directory before publishing. Separate Python targets by spaces,
# for example: TENCENT_PYPI_COMPILED_PYTHONS="3.10 3.11".
BUILD_COMPILED="${TENCENT_PYPI_BUILD_COMPILED:-0}"
COMPILED_PYTHONS="${TENCENT_PYPI_COMPILED_PYTHONS:-3.10}"

# Load credentials (skip file if env vars already set)
if [ -n "$TENCENT_PYPI_USER" ] && [ -n "$TENCENT_PYPI_TOKEN" ]; then
    echo "Using credentials from environment variables"
else
    if [ ! -f "$CRED_FILE" ]; then
        echo "Error: Credentials file not found at $CRED_FILE"
        echo "Set TENCENT_PYPI_USER and TENCENT_PYPI_TOKEN env vars, or create $CRED_FILE"
        exit 1
    fi
    source "$CRED_FILE"
fi

# Cleanup function to restore original name
cleanup() {
    echo "Restoring package name to '$ORIGINAL_NAME'..."
    sed -i "s/^name = \"$PUBLISH_NAME\"/name = \"$ORIGINAL_NAME\"/" "$PYPROJECT"
    rm -rf "$PROJECT_DIR/dist/" "$PROJECT_DIR/build/"
}

# Set trap to ensure cleanup runs on exit
trap cleanup EXIT

# Main
main() {
    echo "=== Publishing to Tencent Internal PyPI ==="
    echo ""

    cd "$PROJECT_DIR"

    # Extract version from pyproject.toml
    VERSION=$(grep -m1 '^version = ' "$PYPROJECT" | sed 's/version = "\(.*\)"/\1/')
    echo "Package: $PUBLISH_NAME  Version: $VERSION"

    # Check if this version already exists on the registry
    HTTP_CODE=$(curl -s -o /dev/null -w '%{http_code}' "$PUBLISH_URL/$PUBLISH_NAME/$VERSION/")
    if [ "$HTTP_CODE" = "200" ]; then
        echo "Version $VERSION already exists on $PUBLISH_URL, skipping publish."
        trap - EXIT  # cancel cleanup since we haven't modified anything
        exit 0
    fi

    # Clean previous builds
    rm -rf dist/ build/

    # Temporarily change package name
    echo "Changing package name to '$PUBLISH_NAME'..."
    sed -i "s/^name = \"$ORIGINAL_NAME\"/name = \"$PUBLISH_NAME\"/" "$PYPROJECT"

    # Build
    echo "Building package..."
    uv build

    if [ "$BUILD_COMPILED" = "1" ]; then
        echo ""
        echo "Building compiled wheel(s) into dist/..."
        for PYTHON_VERSION in $COMPILED_PYTHONS; do
            echo "Building compiled wheel for Python $PYTHON_VERSION..."
            NEXUS_BINARY_PACKAGE_NAME="$BINARY_PUBLISH_NAME" \
                "$PROJECT_DIR/deployment/pypi/build_nuitka_binary_wheel.sh" \
                --python "$PYTHON_VERSION" \
