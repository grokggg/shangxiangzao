"""Responses API proxy server for Codex."""

from __future__ import annotations

import asyncio
import json
import logging

from collections.abc import Callable
from contextlib import asynccontextmanager, suppress
from typing import TYPE_CHECKING, Any

import uvicorn

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from nexus.runtime.server.services.anthropic_proxy.server import LLMBridge
from nexus.runtime.server.services.rsio_llm_bridge import bridge_error_payload
from nexus.utils.event_loop import create_detached_task

from .converter import openai_to_responses_events, responses_to_openai


if TYPE_CHECKING:
    from nexus.runtime.context import SIOBridge


logger = logging.getLogger(__name__)


ResponsesToOpenAIConverter = Callable[..., dict[str, Any]]
OpenAIToResponsesEventsConverter = Callable[..., list[dict[str, Any]]]


def _format_sse_event(payload: dict[str, Any]) -> str:
    event_type = payload.get('type', 'message')
    body = json.dumps(payload, ensure_ascii=False)
    return f'event: {event_type}\ndata: {body}\n\n'


class ResponsesProxy:
    """Proxy Codex Responses API calls to the host-side OpenAI chat client."""

    def __init__(
        self,
        sio: SIOBridge,
        listen_port: int = 8083,
        listen_host: str = '127.0.0.1',
        request_timeout: float = 300,
        responses_to_openai_converter: ResponsesToOpenAIConverter = responses_to_openai,
        openai_to_responses_events_converter: OpenAIToResponsesEventsConverter = openai_to_responses_events,
    ):
        self.sio = sio
        self.port = listen_port
        self.host = listen_host
        self.request_timeout = request_timeout
        self.responses_to_openai_converter = responses_to_openai_converter
        self.openai_to_responses_events_converter = openai_to_responses_events_converter

        self.bridge: LLMBridge | None = None
        self._server: uvicorn.Server | None = None
        self._server_task: asyncio.Task | None = None

    def _create_app(self) -> FastAPI:
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            logger.info(f'Responses Proxy starting on {self.host}:{self.port}')
            yield
            logger.info('Responses Proxy shutting down')

        app = FastAPI(
            title='Responses API Proxy',
            description='Proxy Codex Responses API calls to OpenAI chat completions',
            lifespan=lifespan,
        )

        @app.get('/')
        async def health():
            return {
                'status': 'ok',
                'bridge_ready': self.bridge is not None,
            }

        @app.get('/v1/models')
        async def models():
            return {
                'data': [],
                'object': 'list',
            }

        @app.post('/v1/responses')
        async def responses(request: Request):
            if self.bridge is None:
                raise HTTPException(status_code=503, detail='LLM bridge not initialized')

            try:
                request_body = await request.json()
                # Collect names of Responses `custom` tools (e.g. Codex freeform apply_patch) so
                # the matching function_call can be restored as a custom_tool_call on the return
                # trip. Local to this request -> per-request isolation (ResponsesProxy is shared).
                custom_tool_names: set[str] = set()
                # Converters are synchronous and CPU-bound; large payloads can hold the proxy
                # event loop. Offload to a worker thread to keep the loop responsive.
                openai_params = await asyncio.to_thread(
                    self.responses_to_openai_converter,
                    request_body,
                    collect_custom_tools=custom_tool_names,
                )
                logger.debug(
                    'Converted Responses request: model=%s messages=%s tools=%s custom_tools=%s',
                    openai_params.get('model'),
                    len(openai_params.get('messages') or []),
                    len(openai_params.get('tools') or []),
                    sorted(custom_tool_names),
                )

                openai_response = await self.bridge.request(openai_params, timeout=self.request_timeout)
                events = await asyncio.to_thread(
                    self.openai_to_responses_events_converter,
                    openai_response,
                    custom_tool_names=custom_tool_names,
                )

                async def stream():
                    for event in events:
                        yield _format_sse_event(event)
                        await asyncio.sleep(0)

                return StreamingResponse(stream(), media_type='text/event-stream')

            except Exception as e:
                logger.error(f'Error processing Responses request: {e}')
                coded = bridge_error_payload(e)
                if coded is not None:
                    status_code, content = coded
                    return JSONResponse(status_code=status_code, content=content)
                raise HTTPException(status_code=500, detail=str(e)) from e

        @app.exception_handler(Exception)
        async def exception_handler(request: Request, exc: Exception):
            logger.error(f'Unhandled exception: {exc}')
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

    async def start(self):
        self.bridge = LLMBridge(self.sio)

        app = self._create_app()
        config = uvicorn.Config(
            app,
            host=self.host,
            port=self.port,
            log_level='warning',
        )
        self._server = uvicorn.Server(config)
        self._server_task = create_detached_task(self._server.serve(), name='nexus-responses-proxy')

        await asyncio.sleep(0.5)
        logger.info(f'Responses Proxy ready at http://{self.host}:{self.port}')

    async def stop(self):
        if self._server is not None:
            self._server.should_exit = True
            if self._server_task is not None:
                try:
                    await asyncio.wait_for(self._server_task, timeout=5)
                except asyncio.TimeoutError:
                    self._server_task.cancel()
                    with suppress(asyncio.CancelledError):
                        await self._server_task

        logger.info('Responses Proxy stopped')

    @property
    def is_running(self) -> bool:
        return self._server is not None and self.bridge is not None


async def run_proxy(
    sio: SIOBridge,
    listen_port: int = 8083,
    listen_host: str = '127.0.0.1',
    responses_to_openai_converter: ResponsesToOpenAIConverter = responses_to_openai,
    openai_to_responses_events_converter: OpenAIToResponsesEventsConverter = openai_to_responses_events,
) -> ResponsesProxy:
    proxy = ResponsesProxy(
        sio=sio,
        listen_port=listen_port,
        listen_host=listen_host,
        responses_to_openai_converter=responses_to_openai_converter,
        openai_to_responses_events_converter=openai_to_responses_events_converter,
    )
    await proxy.start()
    return proxy


__all__ = [
    'OpenAIToResponsesEventsConverter',
    'ResponsesProxy',
    'ResponsesToOpenAIConverter',
    'run_proxy',
]
