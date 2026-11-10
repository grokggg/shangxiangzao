"""Native bridge runtime classification.

Runtimes self-declare their native bridge kind via a ``native_bridge_kind``
class attribute, so this module stays free of concrete provider imports.
"""

from __future__ import annotations

from enum import Enum
from typing import Any


class NativeBridgeRuntimeKind(str, Enum):
    AGS_OSWORLD = 'ags-osworld'
    AGS_WAA = 'ags-waa'
    DEVCLOUD_MACOS = 'devcloud-macos'


class NativeBridgeBundleTarget(str, Enum):
    LINUX_X86_64_PY310 = 'linux-x86_64-py310'
    WINDOWS_AMD64_PY310 = 'windows-amd64-py310'
    MACOS_X86_64_PY311 = 'macos-x86_64-py311'


_BUNDLE_TARGET_BY_KIND: dict[NativeBridgeRuntimeKind, NativeBridgeBundleTarget] = {
    NativeBridgeRuntimeKind.AGS_OSWORLD: NativeBridgeBundleTarget.LINUX_X86_64_PY310,
    NativeBridgeRuntimeKind.AGS_WAA: NativeBridgeBundleTarget.WINDOWS_AMD64_PY310,
    NativeBridgeRuntimeKind.DEVCLOUD_MACOS: NativeBridgeBundleTarget.MACOS_X86_64_PY311,
}


def classify_native_bridge_runtime(runtime: Any) -> NativeBridgeRuntimeKind:
    """Classify a native bridge runtime from its self-declared kind."""
    kind = getattr(runtime, 'native_bridge_kind', None)
    if isinstance(kind, NativeBridgeRuntimeKind):
        return kind
    raise TypeError(f'Unsupported native bridge runtime: {type(runtime).__name__}')


def native_bridge_bundle_target_for_kind(kind: NativeBridgeRuntimeKind) -> NativeBridgeBundleTarget:
    """Return the bridge bundle target for a classified native runtime kind."""
    target = _BUNDLE_TARGET_BY_KIND.get(kind)
    if target is None:
        raise TypeError(f'Unsupported native bridge runtime kind: {kind.value}')
    return target


def native_bridge_bundle_target_for_runtime(runtime: Any) -> NativeBridgeBundleTarget:
    """Classify a runtime and return the bridge bundle target enum."""
    return native_bridge_bundle_target_for_kind(classify_native_bridge_runtime(runtime))


__all__ = [
    'NativeBridgeBundleTarget',
    'NativeBridgeRuntimeKind',
    'classify_native_bridge_runtime',
    'native_bridge_bundle_target_for_kind',
    'native_bridge_bundle_target_for_runtime',
]
