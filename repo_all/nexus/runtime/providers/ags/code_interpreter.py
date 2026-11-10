"""AGS code interpreter runtime implementations."""

from __future__ import annotations

import base64
import json
import textwrap

from typing import Any

from nexus.runtime.providers.ags.base import (
    _JSON_RESULT_END,
    _JSON_RESULT_START,
    AGSRawRuntime,
    _BaseAGSRuntime,
    _extract_tagged_json,
)
from nexus.runtime.providers.ags.handle import AGSInstanceHandle


class AGSCodeInterpreterRuntime(_BaseAGSRuntime):
    """Limited runtime backed by AGS native code interpreter."""

    def __init__(
        self,
        handle: AGSInstanceHandle,
        *,
        execute_port: int = 49999,
        request_timeout: float = 1800.0,
        resource_spec: dict[str, int] | None = None,
    ) -> None:
        super().__init__(handle, request_timeout=request_timeout, resource_spec=resource_spec)
        self.execute_port = execute_port

    @property
    def base_url(self) -> str | None:
        return self.handle.url(self.execute_port)

    async def _run_execute(self, code: str, language: str = 'python') -> Any:
        url = self.handle.url(self.execute_port, '/execute')
        body, text, status_code = await self._http_post_json(url, {'code': code, 'language': language})
        if status_code >= 400:
            raise RuntimeError(f'AGS code-interpreter execute failed: HTTP {status_code} {text[:500]}')
        return body if body is not None else text

    @staticmethod
    def _build_run_command_script(
        command: str,
        cwd: str | None,
        timeout: float,
        stdin: str | None,
        extra_env: dict[str, str] | None,
    ) -> str:
        stdin_b64 = base64.b64encode((stdin or '').encode('utf-8')).decode('utf-8') if stdin is not None else None
        script = f"""
import base64
import json
import os
import subprocess
import time

from pathlib import Path

command = {command!r}
cwd = {cwd!r}
timeout = {timeout!r}
stdin_b64 = {stdin_b64!r}
extra_env = {extra_env or {}!r}

env = os.environ.copy()
env.update({{str(k): str(v) for k, v in extra_env.items()}})
stdin_text = base64.b64decode(stdin_b64).decode('utf-8') if stdin_b64 is not None else None
bash_path = '/bin/bash' if Path('/bin/bash').exists() else None
start = time.time()

try:
    result = subprocess.run(
        command,
        shell=True,
        cwd=cwd,
        env=env,
        input=stdin_text,
        capture_output=True,
        text=True,
        timeout=timeout,
        executable=bash_path,
    )
    payload = {{
        'status': 'Finished',
        'execution_time': time.time() - start,
        'return_code': result.returncode,
        'stdout': result.stdout,
        'stderr': result.stderr,
    }}
except subprocess.TimeoutExpired as exc:
    payload = {{
        'status': 'TimeLimitExceeded',
        'execution_time': time.time() - start,
        'return_code': None,
        'stdout': exc.stdout,
        'stderr': exc.stderr,
    }}
except Exception as exc:
    payload = {{
        'status': 'Error',
        'execution_time': time.time() - start,
        'return_code': None,
        'stdout': None,
        'stderr': str(exc),
    }}

print({_JSON_RESULT_START!r})
print(json.dumps(payload))
print({_JSON_RESULT_END!r})
"""
        return textwrap.dedent(script)

    async def call_executor(self, executor_name: str, arg: Any) -> Any:
        if executor_name != 'run_command':
            raise NotImplementedError(
                f'AGS native code runtime only supports run_command, got executor={executor_name!r}'
            )

        from nexus.runtime.executors.command import CommandRunStatus, RunCommandRequest, RunCommandResponse

        request = RunCommandRequest.model_validate(arg)
        script = self._build_run_command_script(
            request.command,
            request.cwd,
            request.timeout,
            request.stdin,
            request.extra_env,
        )
        execute_result = await self._run_execute(script, 'python')
        payload = _extract_tagged_json(execute_result)

        if payload is None:
            response = RunCommandResponse(
                status=CommandRunStatus.Error,
                stderr=f'Unexpected AGS execute response: {json.dumps(execute_result, ensure_ascii=False)[:1000]}',
            )
        else:
            response = RunCommandResponse.model_validate(payload)

        return response.model_dump()


class AGSRawCodeInterpreterRuntime(AGSRawRuntime):
    """Raw AGS code-interpreter runtime that exposes the native execute endpoint."""

    def __init__(
        self,
        handle: AGSInstanceHandle,
        *,
        execute_port: int = 49999,
        request_timeout: float = 1800.0,
        resource_spec: dict[str, int] | None = None,
    ) -> None:
        super().__init__(
            handle,
            kind='code',
            default_port=execute_port,
            request_timeout=request_timeout,
            resource_spec=resource_spec,
        )
        self.execute_port = execute_port

    async def execute_raw(self, payload: dict[str, Any]) -> Any:
        response = await self.request('POST', '/execute', json=payload, port=self.execute_port)
        try:
            return response.json()
        except Exception:
            return response.text

    async def execute(self, code: str, language: str = 'python') -> Any:
        return await self.execute_raw({'code': code, 'language': language})
