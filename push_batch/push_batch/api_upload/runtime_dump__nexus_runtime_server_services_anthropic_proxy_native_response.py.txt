"""Native response envelope handling for the Anthropic proxy."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def extract_native_anthropic_response(response: dict[str, Any] | Any) -> dict[str, Any] | None:
    """Return a validated native Anthropic response body when one is present.

    Older OpenAI-compatible responses have no native envelope and must continue
    through the compatibility converter. Once an envelope is present, however,
    its protocol and body are treated as a strict contract rather than silently
    falling back to a lossy conversion.
    """
    model_dump = getattr(response, 'model_dump', None)
    response_dict = model_dump() if callable(model_dump) else response
    if not isinstance(response_dict, Mapping):
        return None

    provider_fields = response_dict.get('provider_specific_fields')
    if not isinstance(provider_fields, Mapping) or 'native_response' not in provider_fields:
        return None

    envelope = provider_fields['native_response']
    if not isinstance(envelope, Mapping):
        raise ValueError('native_response must be a dict')
    if envelope.get('protocol') != 'anthropic':
        raise ValueError("native_response.protocol must be 'anthropic'")

    body = envelope.get('body')
    if not isinstance(body, dict):
        raise ValueError('native_response.body must be a dict')
    return body


__all__ = ['extract_native_anthropic_response']
