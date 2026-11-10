"""Runtime services.

This module contains services that run inside the runtime environment,
such as Anthropic, Gemini, OpenAI, and Responses API proxies for installed agents.

Submodules are imported directly to avoid pulling fastapi (a runtime-only
optional dependency) into the host-side import graph.
"""
