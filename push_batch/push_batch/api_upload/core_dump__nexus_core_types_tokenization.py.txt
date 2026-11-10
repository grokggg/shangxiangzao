"""Token counting result type for OpenAI-compatible clients."""

from __future__ import annotations

from pydantic import BaseModel


class TokenCountResult(BaseModel):
    input_tokens: int
