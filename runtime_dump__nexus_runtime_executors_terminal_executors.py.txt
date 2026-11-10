"""Terminal session executors: thin registration layer over TerminalSession.

8 executor names mirror the legacy implementation 1:1, so client/base.py and
all upstream agents need zero changes.
"""

import logging

from nexus.core import executor_registry
from nexus.runtime.executors.models.terminal import (
    CapturePaneRequest,
    CapturePaneResponse,
    CommandInfo,
    CreateTerminalSessionRequest,
    CreateTerminalSessionResponse,
    DestroyTerminalSessionRequest,
    DestroyTerminalSessionResponse,
    QueryCommandStatusRequest,
    RunCommandInSessionRequest,
    SendKeysToSessionRequest,
    SendKeysToSessionResponse,
    StartCommandInSessionRequest,
    TerminateSessionProcessesRequest,
    TerminateSessionProcessesResponse,
)
from nexus.runtime.executors.terminal.session import TerminalSession


logger = logging.getLogger(__name__)

# Registry of live sessions, keyed by session_id. Owned by the executor layer;
# TerminalSession itself has no knowledge of it.
TERMINAL_SESSIONS: dict[str, TerminalSession] = {}


def _get(session_id: str) -> TerminalSession:
    session = TERMINAL_SESSIONS.get(session_id)
    if session is None:
        raise ValueError(f'Session {session_id} not found')
    return session


@executor_registry.register('create_terminal_session')
async def create_terminal_session(request: CreateTerminalSessionRequest) -> CreateTerminalSessionResponse:
    # Duplicate-id guard: reject a reused live id BEFORE spawning, so we never
    # overwrite the registry entry and orphan the first shell (code-review #10).
    if request.session_id is not None and request.session_id in TERMINAL_SESSIONS:
        raise ValueError(f'Session {request.session_id} already exists')
    session = TerminalSession(request.session_id, request.height, request.width)
    await session.start()
    # Register only after a successful start so a failed start leaves no entry.
    TERMINAL_SESSIONS[session.session_id] = session
    return CreateTerminalSessionResponse(session_id=session.session_id)


@executor_registry.register('destroy_terminal_session')
async def destroy_terminal_session(request: DestroyTerminalSessionRequest) -> DestroyTerminalSessionResponse:
    session = TERMINAL_SESSIONS.pop(request.session_id, None)
    if session:
        await session.close()
    return DestroyTerminalSessionResponse(success=True)


@executor_registry.register('run_command_in_terminal_session')
async def run_command_in_session(request: RunCommandInSessionRequest) -> CommandInfo:
    session = _get(request.session_id)
    logger.debug(
        'run_command_in_terminal_session: session=%s command=%s',
        request.session_id,
        request.command,
    )
    return await session.run_command(request.command, timeout=request.timeout)


@executor_registry.register('start_command_in_terminal_session')
async def start_command_in_session(request: StartCommandInSessionRequest) -> CommandInfo:
    session = _get(request.session_id)
    logger.debug(
        'start_command_in_terminal_session: session=%s command=%s',
        request.session_id,
        request.command,
    )
    command_id = await session.start_command(request.command)
    cmd_info = await session.get_command_info(command_id)
    if cmd_info is None:
        raise RuntimeError(f'Failed to get command info for command {command_id}')
    return cmd_info


@executor_registry.register('query_terminal_command_status')
async def query_command_status(request: QueryCommandStatusRequest) -> CommandInfo:
    session = _get(request.session_id)
    cmd_info = await session.get_command_info(request.command_id)
    if cmd_info is None:
        raise ValueError(f'Command {request.command_id} not found')
    return cmd_info


@executor_registry.register('send_keys_to_terminal_session')
async def send_keys_to_session(request: SendKeysToSessionRequest) -> SendKeysToSessionResponse:
    session = _get(request.session_id)
    success = await session.send_keys(request.keys)
    return SendKeysToSessionResponse(success=success)


@executor_registry.register('capture_terminal_pane')
async def capture_pane(request: CapturePaneRequest) -> CapturePaneResponse:
    session = _get(request.session_id)
    return CapturePaneResponse(content=await session.capture_pane(request.capture_entire))


@executor_registry.register('terminate_terminal_session_processes')
async def terminate_session_processes(
    request: TerminateSessionProcessesRequest,
) -> TerminateSessionProcessesResponse:
    session = _get(request.session_id)
    try:
        await session.terminate_session_processes()
        return TerminateSessionProcessesResponse(success=True, error=None)
    except Exception as e:  # noqa: BLE001
        return TerminateSessionProcessesResponse(success=False, error=str(e))
