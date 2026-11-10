#!/bin/bash
#
# Deploy MkDocs documentation to oa-pages branch
#
# This script uses temporary directories to avoid modifying the current working repository.
# It clones a fresh copy of the repo for deployment operations.
#
# Usage: ./scripts/deploy-docs.sh
#

set -e

# Configuration
PAGES_BRANCH="oa-pages"
CNAME_DOMAIN="nexus.pages.woa.com"
SOURCE_BRANCH="${SOURCE_BRANCH:-main}"
REMOTE_URL="git@git.woa.com:nexus-agent/nexus.git"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Get the root directory of the git repository
REPO_ROOT=$(git rev-parse --show-toplevel)
cd "$REPO_ROOT"

# Create temporary directories
TMP_CLONE=$(mktemp -d)
TMP_BUILD=$(mktemp -d)
log_info "Created temp clone directory: $TMP_CLONE"
log_info "Created temp build directory: $TMP_BUILD"

# Cleanup function
cleanup() {
    log_info "Cleaning up temporary directories..."
    rm -rf "$TMP_CLONE" "$TMP_BUILD"
}
trap cleanup EXIT

# Find mkdocs executable
if [ -f ".venv/bin/mkdocs" ]; then
    MKDOCS="$REPO_ROOT/.venv/bin/mkdocs"
elif command -v mkdocs &> /dev/null; then
    MKDOCS="mkdocs"
else
    log_error "mkdocs not found. Please install it first:"
    echo "  pip install mkdocs mkdocs-material mkdocs-minify-plugin"
    exit 1
fi

log_info "Using mkdocs: $MKDOCS"

# Build the documentation in the current repository (read-only operation)
log_info "Building documentation..."
$MKDOCS build --clean --site-dir "$TMP_BUILD/site"

if [ ! -d "$TMP_BUILD/site" ]; then
    log_error "Build failed: site directory not created"
    exit 1
fi

log_info "Build successful!"

# Clone the repository to temp directory
log_info "Cloning repository to temp directory..."
git clone --depth 1 "$REMOTE_URL" "$TMP_CLONE/repo"

cd "$TMP_CLONE/repo"

# Inherit git identity from the source repo
GIT_USER_NAME=$(git -C "$REPO_ROOT" config user.name)
GIT_USER_EMAIL=$(git -C "$REPO_ROOT" config user.email)
git config user.name "$GIT_USER_NAME"
git config user.email "$GIT_USER_EMAIL"
log_info "Using git identity: $GIT_USER_NAME <$GIT_USER_EMAIL>"

# Check if oa-pages branch exists on remote
if git ls-remote --exit-code --heads origin "$PAGES_BRANCH" &>/dev/null; then
    log_info "Fetching existing $PAGES_BRANCH branch..."
    git fetch origin "$PAGES_BRANCH":"$PAGES_BRANCH"
    git checkout "$PAGES_BRANCH"
else
    log_info "Creating orphan branch $PAGES_BRANCH..."
    git checkout --orphan "$PAGES_BRANCH"
    git rm -rf --cached . 2>/dev/null || true
fi

# Remove all files (except .git)
log_info "Cleaning $PAGES_BRANCH branch..."
find . -maxdepth 1 ! -name '.git' ! -name '.' -exec rm -rf {} +

# Copy the built site to the repository root
log_info "Copying built site..."
cp -r "$TMP_BUILD/site/"* .

# Add CNAME file
log_info "Adding CNAME file..."
echo "$CNAME_DOMAIN" > CNAME

# Add all files to g