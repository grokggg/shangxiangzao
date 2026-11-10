#!/bin/bash
set -Eeuo pipefail
exec "${NEXUS_ENVD_BIN:-/usr/bin/envd}" -port "${NEXUS_ENVD_PORT:-49983}"
