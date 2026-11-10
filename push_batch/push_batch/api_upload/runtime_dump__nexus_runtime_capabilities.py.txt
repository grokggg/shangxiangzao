from __future__ import annotations

from collections.abc import Mapping
from typing import Any


TRACE_FLUSH_CAPABILITY = 'trace_flush'
TRACE_FLUSH_CAPABILITY_VERSION = 1


def runtime_capabilities() -> dict[str, int]:
    return {TRACE_FLUSH_CAPABILITY: TRACE_FLUSH_CAPABILITY_VERSION}


def supports_trace_flush(capabilities: Any) -> bool:
    if not isinstance(capabilities, Mapping):
        return False
    version = capabilities.get(TRACE_FLUSH_CAPABILITY)
    return isinstance(version, int) and not isinstance(version, bool) and version >= TRACE_FLUSH_CAPABILITY_VERSION


__all__ = [
    'TRACE_FLUSH_CAPABILITY',
    'TRACE_FLUSH_CAPABILITY_VERSION',
    'runtime_capabilities',
    'supports_trace_flush',
]
