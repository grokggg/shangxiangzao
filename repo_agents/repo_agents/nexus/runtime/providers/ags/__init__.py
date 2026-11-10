"""AGS runtime provider package."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from nexus.runtime.providers.ags.base import AGSRawRuntime as AGSRawRuntime
from nexus.runtime.providers.ags.browser import AGSBrowserRuntime as AGSBrowserRuntime
from nexus.runtime.providers.ags.browser import AGSRawBrowserRuntime as AGSRawBrowserRuntime
from nexus.runtime.providers.ags.code_interpreter import AGSCodeInterpreterRuntime as AGSCodeInterpreterRuntime
from nexus.runtime.providers.ags.code_interpreter import AGSRawCodeInterpreterRuntime as AGSRawCodeInterpreterRuntime
from nexus.runtime.providers.ags.handle import AGSInstanceHandle as AGSInstanceHandle
from nexus.runtime.providers.ags.osworld import AGSOSWorldRuntime as AGSOSWorldRuntime
from nexus.runtime.providers.ags.osworld import AGSRawOSWorldRuntime as AGSRawOSWorldRuntime


if TYPE_CHECKING:
    from nexus.runtime.providers.ags.provider import AGSRuntimeProvider as AGSRuntimeProvider


def __getattr__(name: str) -> Any:
    if name == 'AGSRuntimeProvider':
        from nexus.runtime.providers.ags.provider import AGSRuntimeProvider

        return AGSRuntimeProvider
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')


__all__ = [
    'AGSBrowserRuntime',
    'AGSCodeInterpreterRuntime',
    'AGSInstanceHandle',
    'AGSOSWorldRuntime',
    'AGSRawBrowserRuntime',
    'AGSRawCodeInterpreterRuntime',
    'AGSRawOSWorldRuntime',
    'AGSRawRuntime',
    'AGSRuntimeProvider',
]
