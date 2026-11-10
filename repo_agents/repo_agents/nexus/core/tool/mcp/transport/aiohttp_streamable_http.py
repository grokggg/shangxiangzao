"""Aiohttp-based Streamable HTTP transport for MCP without background GET SSE.

This transport keeps the official MCP `ClientSession` and only replaces the
HTTP transport layer. It intentionally does not open the optional background
GET/SSE channel for server-initiated messages.
"""

from __future__ import annotations

import codecs
import contextlib
import logging

from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass

import aiohttp
import anyio

from anyio.abc import TaskGroup
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from mcp.shared.message import SessionMessage
from mcp.types import ErrorData, JSONRPCError, JSONRPCMessage, JSONRPCNotification, JSONRPCRequest, JSONRPCResponse

from nexus.utils.async_http import http_client


logger = logging.getLogger(__name__)

JSON_CONTENT_TYPE = 'application/json'
SSE_CONTENT_TYPE = 'text/event-stream'
MCP_SESSION_ID = 'mcp-session-id'
MCP_PROTOCOL_VERSION = 'mcp-protocol-version'
ACCEPT = 'Accept'
CONTENT_TYPE = 'Content-Type'


@dataclass(slots=True)
class _SSEEvent:
    event: str = 'message'
    data: str = ''
    id: str | None = None
    retry: int | None = None


class AiohttpStreamableHTTPTransport:
    """Minimal Streamable HTTP client transport using aiohttp."""

    def __init__(
        self,
        url: str,
        session: aiohttp.ClientSession,
        *,
        headers: dict[str, str] | None = None,
        timeout_s: float | None = None,
    ):
        self.url = url
        self.session = session
        self.headers = headers or {}
        self.timeout_s = timeout_s
        self.session_id: str | None = None
        self.protocol_version: str | None = None

    def _prepare_headers(self) -> dict[str, str]:
        headers = {
            ACCEPT: f'{JSON_CONTENT_TYPE}, {SSE_CONTENT_TYPE}',
            CONTENT_TYPE: JSON_CONTENT_TYPE,
        }
        headers.update(self.headers)
        if self.session_id:
            headers[MCP_SESSION_ID] = self.session_id
        if self.protocol_version:
            headers[MCP_PROTOCOL_VERSION] = self.protocol_version
        return headers

    @staticmethod
    def _is_initialization_request(message: JSONRPCMessage) -> bool:
        return isinstance(message.root, JSONRPCRequest) and message.root.method == 'initialize'

    @staticmethod
    def _is_initialized_notification(message: JSONRPCMessage) -> bool:
        return isinstance(message.root, JSONRPCNotification) and message.root.method == 'notifications/initialized'

    def _maybe_extract_session_id(self, response: aiohttp.ClientResponse) -> None:
        session_id = response.headers.get(MCP_SESSION_ID)
        if session_id:
            self.session_id = session_id

    async def _send_message(self, message: JSONRPCMessage, read_stream_writer) -> None:
        await read_stream_writer.send(SessionMessage(message=message))

    async def _handle_json_response(
        self,
        response: aiohttp.ClientResponse,
        read_stream_writer,
        *,
        is_initialization: bool,
    ) -> None:
        payload = await response.read()
        message = JSONRPCMessage.model_validate_json(payload)
        if is_initialization and isinstance(message.root, JSONRPCResponse) and message.root.result:
            protocol_version = message.root.result.get('protocolVersion')
            if protocol_version:
                self.protocol_version = str(protocol_version)
        await self._send_message(message, read_stream_writer)

    async def _handle_sse_response(
        self,
        response: aiohttp.ClientResponse,
        read_stream_writer,
        *,
        is_initialization: bool,
    ) -> None:
        async for event in self._iter_sse_events(response):
            if event.event != 'message' or not event.data:
                continue
            message = JSONRPCMessage.model_validate_json(event.data)
            if is_initialization and isinstance(message.root, JSONRPCResponse) and message.root.result:
                protocol_version = message.root.result.get('protocolVersion')
                if protocol_version:
                    self.protocol_version = str(protocol_version)
            await self._send_message(message, read_stream_writer)
            if isinstance(message.root, (JSONRPCResponse, JSONRPCError)):
                return

    async def _handle_post_request(self, session_message: SessionMessage, read_stream_writer) -> None:
        message = session_message.message
        is_initialization = self._is_initialization_request(message)

        headers = self._prepare_headers()
        payload = message.model_dump(by_alias=True, mode='json', exclude_none=True)

        async with self.session.post(
            self.url,
            json=payload,
            headers=headers,
            timeout=self.timeout_s,
        ) as response:
            if response.status == 202:
                return

            if response.status == 404 and isinstance(message.root, JSONRPCRequest):
                error = JSONRPCError(
                    jsonrpc='2.0',
                    id=message.root.id,
                    error=ErrorData(code=32600, message='Session terminated'),
                )
                await self._send_message(JSONRPCMessage(error), read_stream_writer)
                return

            response.raise_for_status()

            if is_initialization:
                self._maybe_extract_session_id(response)

            if not isinstance(message.root, JSONRPCRequest):
                return

            content_type = response.headers.get(CONTENT_TYPE, '').lower()
            if content_type.startswith(JSON_CONTENT_TYPE):
                await self._handle_json_response(
                    response,
                    read_stream_writer,
                    is_initialization=is_initialization,
                )
                return
            if content_type.startswith(SSE_CONTENT_TYPE):
                await self._handle_sse_response(
                    response,
                    read_stream_writer,
                    is_initialization=is_initialization,
                )
                return

            raise RuntimeError(f'Unexpected MCP content type: {content_type}')

    async def post_writer(
        self,
        write_stream_reader: MemoryObjectReceiveStream[SessionMessage],
        read_stream_writer,
        _tg: TaskGroup,
    ) -> None:
        """Consume outgoing MCP messages and POST them to the server."""
        try:
            async with write_stream_reader:
                async for session_message in write_stream_reader:
                    message = session_message.message

                    # Keep the standard MCP handshake but skip the optional GET SSE channel.
                    if self._is_initialized_notification(message):
                        logger.debug('Skipping optional MCP background GET SSE stream')

                    await self._handle_post_request(session_message, read_stream_writer)
        except Exception as exc:
            logger.exception('Error in aiohttp MCP post_writer')
            await read_stream_writer.send(exc)
        finally:
            await read_stream_writer.aclose()

    async def terminate_session(self) -> None:
        if not self.session_id:
            return
        try:
            async with self.session.delete(
                self.url,
                headers=self._prepare_headers(),
                timeout=self.timeout_s,
            ) as response:
                if response.status not in (200, 204, 405):
                    logger.warning('MCP session termination returned HTTP %s', response.status)
        except Exception as exc:
            logger.warning('MCP session termination failed: %s', exc)

    def get_session_id(self) -> str | None:
        return self.session_id

    async def _iter_sse_events(self, response: aiohttp.ClientResponse) -> AsyncGenerator[_SSEEvent, None]:
        event = _SSEEvent()
        data_lines: list[str] = []
        line_buffer = ''
        decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')

        def parse_line(stripped: str) -> None:
            nonlocal event
            if stripped.startswith(':'):
                return
            field, _, value = stripped.partition(':')
            if value.startswith(' '):
                value = value[1:]
            if field == 'event':
                event.event = value or 'message'
            elif field == 'data':
                data_lines.append(value)
            elif field == 'id':
                event.id = value
            elif field == 'retry':
                with contextlib.suppress(ValueError):
                    event.retry = int(value)

        async for raw_chunk in response.content.iter_any():
            text = decoder.decode(raw_chunk)
            line_buffer += text

            while True:
                newline_index = line_buffer.find('\n')
                if newline_index == -1:
                    break

                stripped = line_buffer[:newline_index].rstrip('\r')
                line_buffer = line_buffer[newline_index + 1 :]
                if stripped == '':
                    if data_lines:
                        event.data = '\n'.join(data_lines)
                        yield event
                    event = _SSEEvent()
                    data_lines = []
                    continue
                parse_line(stripped)

        line_buffer += decoder.decode(b'', final=True)

        if line_buffer:
            stripped = line_buffer.rstrip('\r')
            if stripped:
                parse_line(stripped)

        if data_lines:
            event.data = '\n'.join(data_lines)
            yield event


@asynccontextmanager
async def aiohttp_streamable_http_client(
    url: str,
    *,
    session: aiohttp.ClientSession | None = None,
    headers: dict[str, str] | None = None,
    timeout_s: float | None = None,
    terminate_on_close: bool = True,
) -> AsyncGenerator[
    tuple[
        MemoryObjectReceiveStream[SessionMessage | Exception],
        MemoryObjectSendStream[SessionMessage],
        Callable[[], str | None],
    ],
    None,
]:
    """Create an aiohttp-based MCP streamable HTTP transport without GET SSE."""
    read_stream_writer, read_stream = anyio.create_memory_object_stream[SessionMessage | Exception](0)
    write_stream, write_stream_reader = anyio.create_memory_object_stream[SessionMessage](0)

    own_session = False
    aiohttp_session = session or http_client._get_session()  # Reuse nexus global aiohttp session when available.
    transport = AiohttpStreamableHTTPTransport(
        url,
        aiohttp_session,
        headers=headers,
        timeout_s=timeout_s,
    )

    async with anyio.create_task_group() as tg:
        try:
            tg.start_soon(transport.post_writer, write_stream_reader, read_stream_writer, tg)
            yield (read_stream, write_stream, transport.get_session_id)
        finally:
            if terminate_on_close:
                await transport.terminate_session()
            tg.cancel_scope.cancel()
            await write_stream.aclose()
            if own_session:
                with contextlib.suppress(Exception):
                    await aiohttp_session.close()
