"""Worker-side HTTP server for exposing ASGI endpoints.

Provides a singleton HTTP server (FastAPI + uvicorn) per Worker process.
Any code running inside a Worker can mount ASGI sub-applications on it and
register routes with the Base server for reverse-proxy forwarding.

The server is lazy-started on first use and listens on ``127.0.0.1:0``
(OS-assigned random port).  Since Workers run in the same machine/container
as the Base server, localhost networking is sufficient.
"""

from __future__ import annotations

import asyncio
import logging

from contextlib import suppress
from typing import Any

from nexus.utils.event_loop import create_detached_task


logger = logging.getLogger(__name__)


class WorkerHTTPServer:
    """Singleton HTTP server per Worker process.

    Usage::

        from nexus.runtime.server.worker.http import get_or_create

        http_server = await get_or_create()
        http_server.mount('/mcp/proxy', mcp_proxy_asgi)
        # http_server.base_url == 'http://127.0.0.1:<port>'
    """

    def __init__(self) -> None:
        from fastapi import FastAPI

        self.app = FastAPI()
        self._server: Any = None
        self._task: asyncio.Task | None = None
        self.port: int | None = None
        self.base_url: str | None = None
        self._mounts: set[str] = set()

    def mount(self, path: str, asgi_app: Any) -> None:
        """Mount an ASGI sub-application at *path* (idempotent)."""
        if path not in self._mounts:
            self.app.mount(path, asgi_app)
            self._mounts.add(path)
            logger.info('Worker HTTP: mounted %s', path)

    async def start(self) -> None:
        """Start uvicorn on a random port."""
        import uvicorn

        config = uvicorn.Config(
            self.app,
            host='127.0.0.1',
            port=0,
            log_level='error',
        )
        self._server = uvicorn.Server(config)
        self._task = create_detached_task(self._server.serve(), name='nexus-worker-http')

        # Wait for the server to be ready
        while not self._server.started:
            await asyncio.sleep(0.05)

        self.port = self._server.servers[0].sockets[0].getsockname()[1]
        self.base_url = f'http://127.0.0.1:{self.port}'
        logger.info('Worker HTTP server started at %s', self.base_url)

    async def stop(self) -> None:
        """Gracefully stop the HTTP server."""
        if self._server is not None:
            self._server.should_exit = True
            if self._task is not None:
                try:
                    await asyncio.wait_for(self._task, timeout=5)
                except asyncio.TimeoutError:
                    self._task.cancel()
                    with suppress(asyncio.CancelledError):
                        await self._task
            logger.info('Worker HTTP server stopped')


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_instance: WorkerHTTPServer | None = None


async def get_or_create() -> WorkerHTTPServer:
    """Return the singleton :class:`WorkerHTTPServer`, starting it if needed."""
    global _instance
    if _instance is None:
        _instance = WorkerHTTPServer()
        await _instance.start()
    return _instance


async def shutdown() -> None:
    """Stop the singleton server if running.  Called on Worker exit."""
    global _instance
    if _instance is not None:
        await _instance.stop()
        _instance = None
