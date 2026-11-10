#!/bin/bash
set -Eeuo pipefail
exec /usr/bin/python3 /opt/yb/bin/nexus-isolation.py
