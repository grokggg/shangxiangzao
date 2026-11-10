"""Gateway: LLM session management, model I/O, and trajectory collection.

Gateway disaggregates model interaction (chat template, tokenize, generate, decode,
tool parse, segment assembly) from agent control flow, providing a unified
session-based interface for both training and pure inference paths.
"""

from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from nexus.gateway.client import GatewayOpenAIClient
    from nexus.gateway.session_manager import session_manager
    from nexus.gateway.trajectory import Trajectory


__all__ = [
    'GatewayOpenAIClient',
    'Trajectory',
    'session_manager',
]


def __getattr__(name: str) -> Any:
    if name == 'GatewayOpenAIClient':
        from nexus.gateway.client import GatewayOpenAIClient

        return GatewayOpenAIClient
    if name == 'Trajectory':
        from nexus.gateway.trajectory import Trajectory

        return Trajectory
    if name == 'session_manager':
        from nexus.gateway.session_manager import session_manager

        return session_manager
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
