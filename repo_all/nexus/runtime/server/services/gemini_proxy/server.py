"""Gemini API proxy server for runtime-local Gemini CLI."""

from __future__ import annotations

import asyncio
import json
import logging

from contextlib import asynccontextmanager, suppress
from typing import TYPE_CHECKING, Any

import uvicorn

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from nexus.runtime.server.services.rsio_llm_bridge import RSIOLLMBridge
from nexus.utils.event_loop import create_detached_task

from .converter import (
    gemini_request_to_openai,
    openai_response_to_gemini,
    simulate_gemini_streaming,
)


if TYPE_CHECKING:
    from nexus.runtime.context import SIOBridge


logger = logging.getLogger(__name__)


class GeminiProxy:
    """Proxy Gemini CLI HTTP requests from runtime back to the host."""

    def __init__(
        self,
        sio: SIOBridge,
        listen_port: int = 8084,
        listen_host: str = '127.0.0.1',
        request_timeout: float = 300,
    ):
        self.sio = sio
        self.port = listen_port
        self.host = listen_host
        self.request_timeout = request_timeout

        self.bridge: RSIOLLMBridge | None = None
        self._server: uvicorn.Server | None = None
        self._server_task: asyncio.Task | None = None

    def _create_app(self) -> FastAPI:
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            logger.info('Gemini Proxy starting on %s:%s', self.host, self.port)
            yield
            logger.info('Gemini Proxy shutting down')

        app = FastAPI(
            title='Gemini API Proxy',
            description='Proxy Gemini Vertex-style requests to the host-side handler',
            lifespan=lifespan,
        )

        @app.get('/')
        async def health():
            return {
                'status': 'ok',
                'bridge_ready': self.bridge is not None,
            }

        @app.api_route('/{full_path:path}', methods=['GET', 'POST', 'PUT', 'PATCH', 'DELETE', 'OPTIONS'])
        async def proxy(full_path: str, request: Request):
            if self.bridge is None:
                raise HTTPException(status_code=503, detail='Gemini bridge not initialized')

            try:
                request_path = '/' + full_path if full_path else '/'
                rewritten_path = self._rewrite_path(request_path)
                if request.method == 'POST' and self._is_generation_path(rewritten_path):
                    body = await request.json()
                    if not isinstance(body, dict):
                        raise HTTPException(status_code=400, detail='Gemini request body must be a JSON object')

                    openai_params = await asyncio.to_thread(gemini_request_to_openai, rewritten_path, body)
                    openai_response = await self.bridge.request(openai_params, timeout=self.request_timeout)
                    gemini_response = await asyncio.to_thread(
                        openai_response_to_gemini,
                        openai_response,
                        model=openai_params.get('model'),
                    )

                    if self._is_stream_request(rewritten_path, request):
                        return StreamingResponse(
                            simulate_gemini_streaming(gemini_response),
                            media_type='text/event-stream',
                        )

                    return gemini_response

                if request.method == 'POST' and rewritten_path.endswith(':countTokens'):
                    body = await request.json()
                    return self._count_tokens_response(body if isinstance(body, dict) else {})

                raise HTTPException(status_code=404, detail=f'Unsupported Gemini proxy path: {request_path}')
            except HTTPException:
                raise
            except Exception as exc:
                logger.error('Error processing Gemini proxy request: %s', exc)
                raise HTTPException(status_code=500, detail=str(exc)) from exc

        @app.exception_handler(Exception)
        async def exception_handler(request: Request, exc: Exception):
            logger.error('Unhandled Gemini proxy exception: %s', exc)
            return JSONResponse(
                status_code=500,
                content={
                    'error': {
                        'type': 'internal_error',
                        'message': str(exc),
                    }
                },
            )

        return app

    def _rewrite_path(self, path: str) -> str:
        if path == '/v1beta1':
            return '/v1'
        if path.startswith('/v1beta1/'):
            return '/v1/' + path.removeprefix('/v1beta1/')
        return path

    def _is_generation_path(self, path: str) -> bool:
        return path.endswith(':generateContent') or path.endswith(':streamGenerateContent')

    def _is_stream_request(self, path: str, request: Request) -> bool:
        if path.endswith(':streamGenerateContent'):
            return True
        return request.query_params.get('alt') == 'sse'

    def _count_tokens_response(self, body: dict[str, Any]) -> dict[str, Any]:
        messages = gemini_request_to_openai('/v1/publishers/google/models/dummy:generateContent', body).get(
            'messages', []
        )
        total_chars = sum(len(_message_to_text(message)) for message in messages if isinstance(message, dict))
        estimated_tokens = max(total_chars // 4, 1) if total_chars else 0
        return {
            'totalTokens': estimated_tokens,
        }

    async def start(self) -> None:
        self.bridge = RSIOLLMBridge(self.sio)

        app = self._create_app()
        config = uvicorn.Config(
            app,
            host=self.host,
            port=self.port,
            log_level='warning',
        )
        self._server = uvicorn.Server(config)
        self._server_task = create_detached_task(self._server.serve(), name='nexus-gemini-proxy')

        await asyncio.sleep(0.5)
        logger.info('Gemini Proxy ready at http://%s:%s', self.host, self.port)

    async def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
            if self._server_task is not None:
                try:
                    await asyncio.wait_for(self._server_task, timeout=5)
                except asyncio.TimeoutError:
                    self._server_task.cancel()
                    with suppress(asyncio.CancelledError):
                        await self._server_task

        logger.info('Gemini Proxy stopped')

    @property
    def is_running(self) -> bool:
        return self._server is not None and self.bridge is not None


def _message_to_text(message: dict[str, Any]) -> str:
    content = message.get('content')
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return json.dumps(content, ensure_ascii=False) if content is not None else ''
    parts: list[str] = []
    for item in content:
        if isinstance(item, dict):
            if isinstance(item.get('text'), str) or item.get('type') == 'text' and isinstance(item.get('text'), str):
                parts.append(item['text'])
            else:
                parts.append(json.dumps(item, ensure_ascii=False))
        else:
            parts.append(str(item))
    return '\n'.join(parts)


async def run_proxy(
    sio: SIOBridge,
    listen_port: int = 8084,
    listen_host: str = '127.0.0.1',
) -> GeminiProxy:
    proxy = GeminiProxy(
        sio=sio,
        listen_port=listen_port,
        listen_host=listen_host,
    )
    await proxy.start()
    return proxy


__all__ = [
    'GeminiProxy',
    'run_proxy',
]
