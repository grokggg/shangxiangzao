"""Anthropic API Proxy Server.

This FastAPI server provides an Anthropic Messages API compatible endpoint
that proxies requests to a host-side OpenAI client via Socket.IO.

The proxy runs inside the runtime container and allows Anthropic clients
(like Claude Code) to use any OpenAI-compatible backend.

Architecture:
    - Proxy runs inside the runtime container alongside the runtime server
    - Uses the runtime server's Socket.IO server to broadcast events to host
    - Host's HttpRuntime client listens for llm_request events and responds
"""

from __future__ import annotations

import asyncio
import logging
import os

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager, suppress
from typing import TYPE_CHECKING, Any

import uvicorn

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from nexus.runtime.server.services.rsio_llm_bridge import RSIOLLMBridge, bridge_error_payload
from nexus.utils.event_loop import create_detached_task

from .converter import anthropic_to_openai, client_request_meta, openai_to_anthropic, set_conversion_log_level
from .models import ErrorDetail, ErrorResponse, MessagesRequest, ThinkingConfig, get_thinking_config
from .native_response import extract_native_anthropic_response
from .streaming import _sse_event, simulate_native_streaming, simulate_streaming


if TYPE_CHECKING:
    from nexus.runtime.context import SIOBridge


logger = logging.getLogger(__name__)

_STREAM_KEEPALIVE_ENABLED_ENV = 'NEXUS_LLM_STREAM_KEEPALIVE'
_STREAM_KEEPALIVE_INTERVAL_ENV = 'NEXUS_LLM_STREAM_KEEPALIVE_INTERVAL'


def _stream_keepalive_interval() -> float:
    enabled = os.getenv(_STREAM_KEEPALIVE_ENABLED_ENV)
    if enabled is not None and enabled.strip().lower() in {'0', 'false', 'no', 'off', 'disabled'}:
        return 0.0

    raw = os.getenv(_STREAM_KEEPALIVE_INTERVAL_ENV)
    if not raw:
        return 15.0
    try:
        return float(raw)
    except ValueError:
        logger.warning(
            'Invalid %s=%r; using default stream keepalive interval',
            _STREAM_KEEPALIVE_INTERVAL_ENV,
            raw,
        )
        return 15.0


class LLMBridge(RSIOLLMBridge):
    """Bridge for LLM requests via Socket.IO.

    Thin alias over the shared :class:`RSIOLLMBridge`; kept as a named
    subclass so existing imports (including responses_proxy) stay valid.
    """


class AnthropicProxy:
    """Anthropic API Proxy Server.

    Provides an Anthropic Messages API compatible endpoint that proxies
    requests to a host-side OpenAI client via Socket.IO.

    This version uses the runtime server's Socket.IO server directly,
    broadcasting events to connected host clients.

    Usage:
        from nexus.runtime.server.app import sio

        proxy = AnthropicProxy(
            sio=sio,  # Runtime server's Socket.IO server
            listen_port=8082
        )
        await proxy.start()
        # ... run Claude Code with ANTHROPIC_BASE_URL=http://localhost:8082
        await proxy.stop()
    """

    def __init__(
        self,
        sio: SIOBridge,
        listen_port: int = 8082,
        listen_host: str = '127.0.0.1',
        request_timeout: float = 300,
        conversion_log_level: str | int = 'DEBUG',
        thinking_config: str | dict[str, Any] | ThinkingConfig | None = None,
    ):
        """Initialize the proxy server.

        Args:
            sio: The runtime server's Socket.IO AsyncServer instance
            listen_port: Port to listen on for Anthropic API requests
            listen_host: Host to bind to (default: localhost only)
            request_timeout: Timeout for LLM requests in seconds
            conversion_log_level: Level used for anthropic<->openai conversion
                dumps. Accepts a name (e.g. ``'DEBUG'``) or numeric level.
                Set ``NEXUS_CONVERSION_LOG_LEVEL`` in the environment to
                override any value passed in — the env var wins so callers
                threading a hard-coded ``'INFO'`` all the way down the stack
                can still be silenced from the outside.
            thinking_config: Preset name (e.g. ``'hunyuan'``), raw dict, or
                ThinkingConfig instance. Defaults to ``'hunyuan'``.
        """
        self.sio = sio
        self.port = listen_port
        self.host = listen_host
        self.request_timeout = request_timeout
        self.thinking_config = get_thinking_config(thinking_config)
        env_level = os.getenv('NEXUS_CONVERSION_LOG_LEVEL', '').strip()
        set_conversion_log_level(env_level or conversion_log_level)

        self.bridge: LLMBridge | None = None
        self._server: uvicorn.Server | None = None
        self._server_task: asyncio.Task | None = None

    def _on_request_body(self, body: dict[str, Any]) -> None:
        """Hook for subclasses to observe the raw request body before validation."""

    def _on_response(self, response: Any) -> None:
        """Hook for subclasses to observe the final response (native dict or MessagesResponse)."""

    def create_app(self) -> FastAPI:
        """Create the FastAPI application and initialize its LLM bridge."""
        if self.bridge is None:
            self.bridge = LLMBridge(self.sio)

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            logger.info(f'Anthropic Proxy starting on {self.host}:{self.port}')
            yield
            logger.info('Anthropic Proxy shutting down')

        app = FastAPI(
            title='Anthropic API Proxy',
            description='Proxy Anthropic Messages API to OpenAI-compatible backends',
            lifespan=lifespan,
        )

        @app.get('/')
        async def health():
            """Health check endpoint."""
            return {
                'status': 'ok',
                'bridge_ready': self.bridge is not None,
            }

        @app.post('/v1/messages')
        async def messages(raw_request: Request):
            """Handle Anthropic Messages API requests."""
            if self.bridge is None:
                raise HTTPException(status_code=503, detail='LLM bridge not initialized')

            body = await raw_request.json()
            self._on_request_body(body)

            try:
                request = MessagesRequest.model_validate(body)
            except Exception as e:
                logger.error('Request validation failed: %s', e)
                raise HTTPException(status_code=422, detail=str(e)) from e

            try:
                # Convert Anthropic request to OpenAI format (always non-streaming to backend).
                # The converter is a synchronous, CPU-bound function; large payloads (e.g. a
                # multi-thousand-character thinking block) can hold the event loop for hundreds
                # of milliseconds. Offload to a worker thread so the proxy loop stays responsive.
                openai_params = await asyncio.to_thread(
                    anthropic_to_openai, request, thinking_config=self.thinking_config
                )
                request_meta = client_request_meta(raw_request.headers)
                if request_meta:
                    openai_params['request_meta'] = request_meta
                logger.debug(f'Converted request to OpenAI format: {openai_params.get("model")}')

                if request.stream:
                    # Reverse-streaming with keep-alive. The backend (host LLM) is
                    # non-streaming, so we can only emit real SSE events once the full
                    # response is back. But we open the SSE response immediately and emit
                    # `: keepalive` comments while waiting, so the client's underlying HTTP
                    # stack (Node/undici defaults headersTimeout/bodyTimeout to 300s) never
                    # sees a long silence and drops the socket. Comments are ignored by the
                    # Anthropic SDK decoder (lines starting with ':'), so they don't pollute
                    # the business event stream.
                    bridge = self.bridge
                    request_timeout = self.request_timeout
                    model = request.model
                    thinking_config = self.thinking_config
                    on_response = self._on_response

                    async def stream_with_keepalive() -> AsyncGenerator[str, None]:
                        bridge_task = asyncio.create_task(bridge.request(openai_params, timeout=request_timeout))
                        keepalive_interval = _stream_keepalive_interval()
                        try:
                            if keepalive_interval > 0:
                                # First byte immediately so response headers are flushed and the
                                # client's headersTimeout window is satisfied right away.
                                yield ': keepalive\n\n'
                                while not bridge_task.done():
                                    done, _ = await asyncio.wait({bridge_task}, timeout=keepalive_interval)
                                    if not done:
                                        # Host still working — refresh the body idle timer.
                                        yield ': keepalive\n\n'
                            else:
                                logger.info('Anthropic streaming keepalive disabled by environment')
                                if not bridge_task.done():
                                    await bridge_task

                            try:
                                openai_response = bridge_task.result()
                                native_response = extract_native_anthropic_response(openai_response)
                                if native_response is not None:
                                    on_response(native_response)
                                    stream = simulate_native_streaming(native_response)
                                else:
                                    anthropic_response = await asyncio.to_thread(
                                        openai_to_anthropic,
                                        openai_response,
                                        model,
                                        thinking_config=thinking_config,
                                    )
                                    on_response(anthropic_response)
                                    stream = simulate_streaming(anthropic_response)
                                async for chunk in stream:
                                    yield chunk
                            except Exception as exc:
                                # Headers (200) already sent — cannot raise HTTP 500 now;
                                # signal bridge, native-envelope, validation, and encoding failures
                                # through the established Anthropic-style SSE error event.
                                logger.error(f'Error processing streaming request: {exc}')
                                coded = bridge_error_payload(exc)
                                if coded is not None:
                                    _, content = coded
                                    error = content['error']
                                    yield _sse_event(
                                        'error',
                                        {
                                            'type': 'error',
                                            'error': {'type': error['type'], 'message': error['message']},
                                        },
                                    )
                                    return
                                yield _sse_event(
                                    'error',
                                    {'type': 'error', 'error': {'type': 'api_error', 'message': str(exc)}},
                                )
                                return
                        except asyncio.CancelledError:
                            # Client disconnected — cancel the in-flight bridge request so
                            # LLMBridge.request()'s finally clears pending_requests.
                            bridge_task.cancel()
                            with suppress(asyncio.CancelledError, Exception):
                                await bridge_task
                            raise

                    return StreamingResponse(
                        stream_with_keepalive(),
                        media_type='text/event-stream',
                        headers={
                            'Cache-Control': 'no-cache',
                            'Connection': 'keep-alive',
                            'X-Accel-Buffering': 'no',
                        },
                    )

                # Non-streaming: block until the host returns the full response, then return
                # it directly. Errors here still surface as HTTP 500 (headers not yet sent).
                openai_response = await self.bridge.request(openai_params, timeout=self.request_timeout)
                native_response = extract_native_anthropic_response(openai_response)
                if native_response is not None:
                    self._on_response(native_response)
                    return native_response
                anthropic_response = await asyncio.to_thread(
                    openai_to_anthropic,
                    openai_response,
                    request.model,
                    thinking_config=self.thinking_config,
                )
                self._on_response(anthropic_response)
                return JSONResponse(content=anthropic_response.model_dump(mode='json', exclude_none=True))

            except HTTPException:
                raise
            except Exception as e:
                logger.error(f'Error processing request: {e}')
                coded = bridge_error_payload(e)
                if coded is not None:
                    status_code, content = coded
                    return JSONResponse(status_code=status_code, content=content)
                raise HTTPException(status_code=500, detail=str(e)) from e

        @app.post('/v1/messages/count_tokens')
        async def count_tokens(request: Request):
            """Token counting endpoint (stub implementation)."""
            body = await request.json()
            # Return a rough estimate - actual counting would need a tokenizer
            messages = body.get('messages', [])
            total_chars = sum(len(str(m.get('content', ''))) for m in messages)
            # Rough estimate: ~4 chars per token
            estimated_tokens = total_chars // 4

            return {'input_tokens': estimated_tokens}

        @app.exception_handler(Exception)
        async def exception_handler(request: Request, exc: Exception):
            """Global exception handler."""
            logger.error(f'Unhandled exception: {exc}')
            return JSONResponse(
                status_code=500,
                content=ErrorResponse(
                    error=ErrorDetail(
                        type='internal_error',
                        message=str(exc),
                    )
                ).model_dump(),
            )

        return app

    async def start(self):
        """Start the proxy server."""
        # Create and start FastAPI server
        app = self.create_app()
        config = uvicorn.Config(
            app,
            host=self.host,
            port=self.port,
            log_level='warning',  # Reduce uvicorn logging noise
        )
        self._server = uvicorn.Server(config)

        # Run server in background task
        self._server_task = create_detached_task(self._server.serve(), name='nexus-anthropic-proxy')

        # Wait for server to be ready
        await asyncio.sleep(0.5)
        logger.info(f'Anthropic Proxy ready at http://{self.host}:{self.port}')

    async def stop(self):
        """Stop the proxy server."""
        # Stop HTTP server
        if self._server is not None:
            self._server.should_exit = True
            if self._server_task is not None:
                try:
                    await asyncio.wait_for(self._server_task, timeout=5)
                except asyncio.TimeoutError:
                    self._server_task.cancel()
                    with suppress(asyncio.CancelledError):
                        await self._server_task

        logger.info('Anthropic Proxy stopped')

    @property
    def is_running(self) -> bool:
        """Check if the proxy is running."""
        return self._server is not None and self.bridge is not None


async def run_proxy(
    sio: SIOBridge,
    listen_port: int = 8082,
    listen_host: str = '127.0.0.1',
) -> AnthropicProxy:
    """Convenience function to create and start a proxy.

    Args:
        sio: The runtime server's Socket.IO AsyncServer instance
        listen_port: Port to listen on
        listen_host: Host to bind to

    Returns:
        Running AnthropicProxy instance
    """
    proxy = AnthropicProxy(
        sio=sio,
        listen_port=listen_port,
        listen_host=listen_host,
    )
    await proxy.start()
    return proxy


__all__ = [
    'AnthropicProxy',
    'LLMBridge',
    'run_proxy',
]
