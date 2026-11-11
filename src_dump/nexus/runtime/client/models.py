"""Request/Response models for runtime client RPC calls."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field

from nexus.core.tool import ToolContext
from nexus.trace import RemoteTraceContext
from nexus.utils.common import get_current_python_version, get_running_nexus_source


# ============== Request/Response Models ==============


class CallExecutorRequest(BaseModel):
    """Request model for executor invocation."""

    request_id: str | None = None  # Optional id for replay-safe deduplication on retries
    executor_name: str
    arg: dict
    trace_context: RemoteTraceContext | None = None  # Optional trace context for distributed tracing


class CallToolRequest(BaseModel):
    """Request model for tool invocation.

    Uses import_path to load the tool function on the runtime server,
    ensuring both sides execute the same code.
    """

    request_id: str | None = None  # Optional id for replay-safe deduplication on retries
    import_path: str  # e.g., "nexus.extensions.tools.sandbox::bash_tool"
    params: dict[str, Any]
    context: ToolContext
    trace_context: RemoteTraceContext | None = None  # Optional trace context for distributed tracing


@dataclass
class VenvHandle:
    """Handle returned by create_venv, holds venv_id and status info."""

    venv_id: str
    state: str = 'unknown'
    python_version: str = ''
    artifacts: list[str] = field(default_factory=list)
    error: str | None = None


class WorkerVenvConfig(BaseModel):
    """Configuration for building an artifact and creating a worker venv.

    Used as ``runtime_configs[name].worker_venv`` in RunInput.

    Accepts ``true`` as shorthand for all defaults, ``false``/``null`` to disable.

    When present, ``ExecutionContext.get_runtime()`` will:
    1. Build a wheel from *source* (with caching)
    2. Create a worker venv with that wheel installed
    3. Set ``runtime.default_venv_id`` so all subsequent calls use the venv

    Example YAML::

        # Shorthand — all defaults
        worker_venv: true

        # Customised
        worker_venv:
          modules: [swe_agent, file_editor]
          python_version: '3.11'
    """

    source: str = Field(default_factory=get_running_nexus_source)
    python_version: str = Field(default_factory=get_current_python_version)
    extras: list[str] = []
    extra_requirements: list[str] = []
    exclude_patterns: list[str] = []
    modules: list[str] | None = Field(
        default=None,
        description="Which nexus modules' runtime deps to install. None=all, []=none, [...]=specific.",
    )


__all__ = [
    'CallExecutorRequest',
    'CallToolRequest',
    'VenvHandle',
    'WorkerVenvConfig',
]
