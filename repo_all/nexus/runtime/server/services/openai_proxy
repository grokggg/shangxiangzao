"""OpenAI API proxy server.

This proxy exposes a minimal OpenAI-compatible HTTP surface inside the runtime
and forwards chat completion requests back to the host-side Nexus OpenAI client
over the existing Socket.IO bridge.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid

from contextlib import asynccontextmanager, suppress
from typing import TYPE_CHECKING, Any

import uvicorn

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from nexus.runtime.server.services.rsio_llm_bridge import RSIOLLMBridge, bridge_error_payload
from nexus.utils.event_loop import create_detached_task


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
    """Bridge OpenAI-format requests to the host-side OpenAI client.

    Thin alias over the shared :class:`RSIOLLMBridge`; kept as a named
    subclass so existing imports (including responses_proxy) stay valid.
    """


async def simulate_chat_completions_stream(
    response: dict[str, Any] | Any,
    *,
    include_usage: bool = False,
):
    """Simulate an OpenAI chat completions SSE stream from a full response payload."""
    model_dump = getattr(response, 'model_dump', None)
    payload: dict[str, Any] = model_dump() if callable(model_dump) else response

    response_id = payload.get('id', f'chatcmpl-{uuid.uuid4().hex}')
    created = payload.get('created', int(time.time()))
    model = payload.get('model', 'unknown')
    choices = payload.get('choices', [])

    for index, choice in enumerate(choices):
        message = choice.get('message', {}) or {}
        delta: dict[str, Any] = {}
        if message.get('role'):
            delta['role'] = message['role']
        if message.get('reasoning_content'):
            delta['reasoning_content'] = message['reasoning_content']
        if message.get('content'):
            delta['content'] = message['content']
        if message.get('tool_calls'):
            tool_calls = []
            for tool_call_index, tool_call in enumerate(message['tool_calls']):
                streamed_tool_call = dict(tool_call)
                if streamed_tool_call.get('index') is None:
                    streamed_tool_call['index'] = tool_call_index
                tool_calls.append(streamed_tool_call)
            delta['tool_calls'] = tool_calls

        if delta:
            chunk = {
                'id': response_id,
                'object': 'chat.completion.chunk',
                'created': created,
                'model': model,
                'choices': [
                    {
                        'index': choice.get('index', index),
                        'delta': delta,
                        'finish_reason': None,
                    }
                ],
            }
            yield f'data: {json.dumps(chunk)}\n\n'
            await asyncio.sleep(0)

        final_chunk = {
            'id': response_id,
            'object': 'chat.completion.chunk',
            'created': created,
            'model': model,
            'choices': [
                {
                    'index': choice.get('index', index),
                    'delta': {},
                    'finish_reason': choice.get('finish_reason', 'stop'),
                }
            ],
        }
        yield f'data: {json.dumps(final_chunk)}\n\n'
        await asyncio.sleep(0)

    usage = payload.get('usage')
    if include_usage and usage is not None:
        usage_chunk = {
            'id': response_id,
            'object': 'chat.completion.chunk',
            'created': created,
            'model': model,
            'choices': [],
            'usage': usage,
        }
        yield f'data: {json.dumps(usage_chunk)}\n\n'
        await asyncio.sleep(0)

    yield 'data: [DONE]\n\n'


class OpenAIProxy:
    """Minimal OpenAI-compatible proxy for chat completions."""

    def __init__(
        self,
        sio: SIOBridge,
        listen_port: int = 8083,
        listen_host: str = '127.0.0.1',
        request_timeout: float = 300,
        models: list[str] | None = None,
    ):
        self.sio = sio
        self.port = listen_port
        self.host = listen_host
        self.request_timeout = request_timeout
        self.models = models or []

        self.bridge: LLMBridge | None = None
        self._server: uvicorn.Server | None = None
        self._server_task: asyncio.Task | None = None

    def _inspect_response_payload(self, payload: dict[str, Any]) -> None:
        """Override in subclasses to peek at the LLM response payload.

        Called for every /chat/completions response, regardless of stream mode.
        Used e.g. by Kilo to detect 'suggest' tool_calls and signal early termination.
        Default: no-op.
        """

    def _prepare_request_body(self, body: dict[str, Any]) -> dict[str, Any]:
        """Override in subclasses to normalize client-specific request payloads."""
        return body

    def _create_app(self) -> FastAPI:
        @asynccontextmanager
        async def lifespan(_app: FastAPI):
            logger.info('OpenAI Proxy starting on %s:%s', self.host, self.port)
            yield
            logger.info('OpenAI Proxy shutting down')

        app = FastAPI(
            title='OpenAI API Proxy',
            description='Proxy OpenAI chat completions to the host-side OpenAI client',
            lifespan=lifespan,
        )

        @app.get('/')
        @app.get('/health')
        async def health():
            return {
                'status': 'ok',
                'bridge_ready': self.bridge is not None,
            }

        @app.get('/v1/models')
        @app.get('/models')
        async def list_models():
            return {
                'object': 'list',
                'data': [{'id': model, 'object': 'model'} for model in self.models],
            }

        @app.post('/v1/chat/completions')
        @app.post('/chat/completions')
        async def chat_completions(request: Request):
            bridge = self.bridge
            if bridge is None:
                raise HTTPException(status_code=503, detail='LLM bridge not initialized')

            body = await request.json()
            if not isinstance(body, dict):
                raise HTTPException(status_code=400, detail='Request body must be a JSON object')
            body = self._prepare_request_body(body)

            stream = bool(body.get('stream'))
            stream_options = body.get('stream_options')
            include_usage = isinstance(stream_options, dict) and stream_options.get('include_usage') is True
            if stream:
                # Host-side Nexus clients currently return full completions, so we
                # replay them as SSE chunks for OpenAI-compatible consumers.
                body = {**body, 'stream': False}

            inspect_hook = self._inspect_response_payload

            if stream:
                # Reverse-streaming: send SSE response headers + keep-alive comments
                # immediately so the client (e.g. cline's bun fetch with hard 5min
                # headers timeout) sees first byte right away, regardless of how
                # long Bedrock takes to return the full payload (xhigh thinking
                # can exceed 5min). Once bridge returns, emit real chunks.
                request_timeout = self.request_timeout

                async def stream_with_keepalive():
                    bridge_task = asyncio.create_task(bridge.request(body, timeout=request_timeout))
                    keepalive_interval = _stream_keepalive_interval()
                    try:
                        if keepalive_interval > 0:
                            while not bridge_task.done():
                                try:
                                    await asyncio.wait_for(asyncio.shield(bridge_task), timeout=keepalive_interval)
                                except asyncio.TimeoutError:
                                    # Bedrock still working — emit SSE comment to keep
                                    # the connection (and bun fetch headers timer) alive.
                                    yield ': keepalive\n\n'
                                    continue
                        else:
                            logger.info('OpenAI streaming keepalive disabled by environment')
                            if not bridge_task.done():
                                await bridge_task
                        try:
                            response = bridge_task.result()
                        except Exception as exc:
                            logger.error('Error processing OpenAI streaming request: %s', exc)
                            err_chunk = {'error': {'message': str(exc), 'type': 'proxy_error'}}
                            yield f'data: {json.dumps(err_chunk)}\n\n'
                            yield 'data: [DONE]\n\n'
                            return

                        model_dump = getattr(response, 'model_dump', None)
                        payload = model_dump() if callable(model_dump) else response
                        try:
                            inspect_hook(payload)
                        except Exception:
                            logger.exception('inspect_response_payload hook failed; ignoring')
                        async for chunk in simulate_chat_completions_stream(payload, include_usage=include_usage):
                            yield chunk
                    except asyncio.CancelledError:
                        bridge_task.cancel()
                        raise

                return StreamingResponse(
                    stream_with_keepalive(),
                    media_type='text/event-stream',
                )

            try:
                response = await bridge.request(body, timeout=self.request_timeout)
            except Exception as exc:
                logger.error('Error processing OpenAI request: %s', exc)
                coded = bridge_error_payload(exc)
                if coded is not None:
                    status_code, content = coded
                    return JSONResponse(status_code=status_code, content=content)
                raise HTTPException(status_code=500, detail=str(exc)) from exc

            model_dump = getattr(response, 'model_dump', None)
            payload = model_dump() if callable(model_dump) else response

            try:
                inspect_hook(payload)
            except Exception:
                logger.exception('inspect_response_payload hook failed; ignoring')

            return JSONResponse(content=payload)

        @app.exception_handler(Exception)
        async def exception_handler(_request: Request, exc: Exception):
            logger.error('Unhandled exception: %s', exc)
            return JSONResponse(
                status_code=500,
                content={
                    'error': {
                        'message': str(exc),
                        'type': 'internal_error',
                    }
                },
            )

        return app

    async def start(self) -> None:
        self.bridge = LLMBridge(self.sio)

        app = self._create_app()
        config = uvicorn.Config(
            app,
            host=self.host,
            port=self.port,
            log_level='warning',
        )
        self._server = uvicorn.Server(config)
        self._server_task = create_detached_task(self._server.serve(), name='nexus-openai-proxy')

        await asyncio.sleep(0.5)
        logger.info('OpenAI Proxy ready at http://%s:%s', self.host, self.port)

    async def stop(self) -> None:
        if self._server:
            self._server.should_exit = True

        if self._server_task:
            self._server_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._server_task

        self._server = None
        self._server_task = None
        self.bridge = None
        logger.info('OpenAI Proxy stopped')


__all__ = [
    'LLMBridge',
    'OpenAIProxy',
    'simulate_chat_completions_stream',
]
