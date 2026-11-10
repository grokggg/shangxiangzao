"""Terminal executor request/response models.

Pure Pydantic — no tmux/subprocess dependencies.
"""

from typing import ClassVar

from pydantic import BaseModel, Field

from nexus.runtime.executors.models.base import ExecutorRequest


class CommandInfo(BaseModel):
    command_id: int
    command: str | None = None
    exit_code: int | None = None
    start_time: float | None = None
    end_time: float | None = None
    stdout: str | None = None
    stderr: str | None = None
    output: str | None = None


class CreateTerminalSessionResponse(BaseModel):
    session_id: str = Field(..., description='The session ID')


class CreateTerminalSessionRequest(ExecutorRequest[CreateTerminalSessionResponse]):
    executor_name: ClassVar[str] = 'create_terminal_session'
    response_type: ClassVar[type[BaseModel]] = CreateTerminalSessionResponse

    session_id: str | None = Field(None, description='Optional session ID. If not provided, a UUID will be generated.')
    height: int = Field(40, gt=0, le=65535, description='Terminal height in lines')
    width: int = Field(120, gt=0, le=65535, description='Terminal width in characters')


class DestroyTerminalSessionResponse(BaseModel):
    success: bool = Field(..., description='Whether the session was destroyed successfully')


class DestroyTerminalSessionRequest(ExecutorRequest[DestroyTerminalSessionResponse]):
    executor_name: ClassVar[str] = 'destroy_terminal_session'
    response_type: ClassVar[type[BaseModel]] = DestroyTerminalSessionResponse

    session_id: str


class RunCommandInSessionRequest(ExecutorRequest[CommandInfo]):
    executor_name: ClassVar[str] = 'run_command_in_terminal_session'
    response_type: ClassVar[type[BaseModel]] = CommandInfo

    session_id: str
    command: str = Field(..., description='The command to execute')
    timeout: float | None = Field(None, description='Timeout in seconds')


class StartCommandInSessionRequest(ExecutorRequest[CommandInfo]):
    executor_name: ClassVar[str] = 'start_command_in_terminal_session'
    response_type: ClassVar[type[BaseModel]] = CommandInfo

    session_id: str
    command: str = Field(..., description='The command to execute')


class QueryCommandStatusRequest(ExecutorRequest[CommandInfo]):
    executor_name: ClassVar[str] = 'query_terminal_command_status'
    response_type: ClassVar[type[BaseModel]] = CommandInfo

    session_id: str
    command_id: int = Field(..., description='The command ID to query')


class SendKeysToSessionResponse(BaseModel):
    success: bool


class SendKeysToSessionRequest(ExecutorRequest[SendKeysToSessionResponse]):
    executor_name: ClassVar[str] = 'send_keys_to_terminal_session'
    response_type: ClassVar[type[BaseModel]] = SendKeysToSessionResponse

    session_id: str
    keys: str | list[str] = Field(..., description='Keys to send (tmux format, e.g., "C-c", "Enter")')


class CapturePaneResponse(BaseModel):
    content: str


class CapturePaneRequest(ExecutorRequest[CapturePaneResponse]):
    executor_name: ClassVar[str] = 'capture_terminal_pane'
    response_type: ClassVar[type[BaseModel]] = CapturePaneResponse

    session_id: str
    capture_entire: bool = Field(False, description='Whether to capture all history')


class TerminateSessionProcessesResponse(BaseModel):
    success: bool
    error: str | None = None


class TerminateSessionProcessesRequest(ExecutorRequest[TerminateSessionProcessesResponse]):
    executor_name: ClassVar[str] = 'terminate_terminal_session_processes'
    response_type: ClassVar[type[BaseModel]] = TerminateSessionProcessesResponse

    session_id: str
