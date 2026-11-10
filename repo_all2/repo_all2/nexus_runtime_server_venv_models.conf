"""Data models for venv and artifact management."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class ArtifactSpec(BaseModel):
    """Describes an artifact to install into a worker venv."""

    name: str
    whl_data: str | None = Field(None, description='Base64-encoded wheel bytes (push mode)')
    whl_url: str | None = Field(None, description='URL to download the wheel from (pull mode)')
    whl_sha256: str | None = Field(None, description='SHA-256 checksum for integrity verification')
    extras: list[str] = Field(default_factory=list, description='Extras to activate')


class VenvSpec(BaseModel):
    """Request payload for creating a worker venv."""

    python_version: str = '3.11'
    artifacts: list[ArtifactSpec] = Field(default_factory=list)
    extra_requirements: list[str] = Field(default_factory=list, description='Additional pip requirements not in base')
    modules: list[str] | None = Field(
        default=None,
        description='Nexus modules whose runtime deps to install. None=all, []=none, [...]=specific.',
    )
    env_vars: dict[str, str] = Field(default_factory=dict, description='Environment variables for the worker process')


class VenvStatus(BaseModel):
    """Status of a managed venv."""

    venv_id: str
    state: Literal['creating', 'ready', 'failed', 'destroying']
    python_version: str
    artifacts: list[str] = Field(default_factory=list, description='Installed artifact names')
    extra_requirements: list[str] = Field(default_factory=list)
    created_at: datetime
    error: str | None = None


class WorkerStatus(BaseModel):
    """Status of a worker process attached to a venv."""

    venv_id: str
    worker_id: str
    state: Literal['starting', 'connected', 'running', 'stopped']
    pid: int | None = None
    connected_at: datetime | None = None


__all__ = [
    'ArtifactSpec',
    'VenvSpec',
    'VenvStatus',
    'WorkerStatus',
]
