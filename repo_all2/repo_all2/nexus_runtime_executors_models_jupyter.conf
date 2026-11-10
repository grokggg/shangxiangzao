"""Jupyter executor request/response models.

Pure Pydantic — does NOT import jupyter_client.
"""

from typing import Any, ClassVar

from pydantic import BaseModel, Field

from nexus.runtime.executors.models.base import ExecutorRequest


class CreateIPythonSessionResponse(BaseModel):
    session_id: str = Field(..., examples=['d5b87024-c646-4d1b-849c-80a81d10368f'], description='the session id')


class CreateIPythonSessionRequest(ExecutorRequest[CreateIPythonSessionResponse]):
    executor_name: ClassVar[str] = 'create_ipython_session'
    response_type: ClassVar[type[BaseModel]] = CreateIPythonSessionResponse

    python_executable: str | None = Field(
        None,
        examples=['/usr/bin/python3'],
        description='The python executable to use, must have ipykernel installed. If not specified, use the built-in python of portal service',
    )


class ExecuteIPythonResponse(BaseModel):
    status: str
    stdout: str
    stderr: str
    display: list[dict[str, Any]]
    error: list[dict[str, Any]]
    execution_time: float | None = None


class ExecuteIPythonRequest(ExecutorRequest[ExecuteIPythonResponse]):
    executor_name: ClassVar[str] = 'execute_ipython'
    response_type: ClassVar[type[BaseModel]] = ExecuteIPythonResponse

    session_id: str
    code: str = Field(..., examples=['print(123)'], description='the code to run')
    timeout: float = Field(180, description='code run timeout')


class DestroyIPythonSessionResponse(BaseModel):
    success: bool = Field(..., examples=[True], description='whether the session is destroyed successfully')


class DestroyIPythonSessionRequest(ExecutorRequest[DestroyIPythonSessionResponse]):
    executor_name: ClassVar[str] = 'destroy_ipython_session'
    response_type: ClassVar[type[BaseModel]] = DestroyIPythonSessionResponse

    session_id: str
