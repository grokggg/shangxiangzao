# /// script
# dependencies = []
#
# [project]
# name = "command"
# authors = [
#     { name = "tarschen" },
# ]
#
# [tool.nexus]
# executors = { run_command = "nexus.runtime.executors.command" }
# ///

import asyncio
import contextlib
import logging
import subprocess
import time
import traceback

from asyncio import StreamReader

from nexus.core import executor_registry
from nexus.runtime.executors.models.command import *  # noqa: F401,F403
from nexus.runtime.executors.models.command import CommandRunStatus, RunCommandRequest, RunCommandResponse
from nexus.utils.shell import get_default_pwd, get_os_env_without_ld_library_path, resolve_first_existing_path


logger = logging.getLogger(__name__)


# ==================== Implementation ====================


EXECUTABLE_BASH = resolve_first_existing_path(['/nix/runtime/nix-env/bin/bash', '/bin/bash', 'bash'])


@executor_registry.register('run_command')
async def run_command(args: RunCommandRequest) -> RunCommandResponse:
    """Execute a shell command. Importable standalone function for backward compat."""
    try:
        logger.debug(f'running command {args.command}')
        os_envs = get_os_env_without_ld_library_path()

        # simulate login shell to load user envs
        os_envs['BASH_ENV'] = '~/.bashrc'

        cwd = args.cwd or get_default_pwd()

        p = await asyncio.create_subprocess_shell(
            args.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=cwd,
            executable=EXECUTABLE_BASH,
            env={**os_envs, **(args.extra_env or {})},
        )

        # stdin/stdout/stderr are guaranteed to be non-None because we passed subprocess.PIPE
        assert p.stdin is not None
        assert p.stdout is not None
        assert p.stderr is not None

        if args.stdin is not None:
            p.stdin.write(args.stdin.encode())
        p.stdin.close()

        start_time = time.time()
        stdout_reader = await get_output_reader(p.stdout)
        stderr_reader = await get_output_reader(p.stderr)

        try:
            await asyncio.wait_for(p.wait(), timeout=args.timeout)
            execution_time = time.time() - start_time
            logger.debug(f'stop running command {args.command}')
        except TimeoutError:
            # Kill the process tree to prevent zombie/orphan processes
            import signal

            with contextlib.suppress(ProcessLookupError):
                p.send_signal(signal.SIGTERM)
            try:
                await asyncio.wait_for(p.wait(), timeout=5.0)
            except TimeoutError:
                with contextlib.suppress(ProcessLookupError):
                    p.kill()  # SIGKILL as last resort
                try:
                    await asyncio.wait_for(p.wait(), timeout=3.0)
                except TimeoutError:
                    logger.warning('Process %s did not exit after SIGKILL', p.pid)
            return RunCommandResponse(
                status=CommandRunStatus.TimeLimitExceeded,
                execution_time=time.time() - start_time,
                stdout=await stdout_reader(),
                stderr=await stderr_reader(),
            )
        return RunCommandResponse(
            status=CommandRunStatus.Finished,
            execution_time=execution_time,
            return_code=p.returncode,
            stdout=await stdout_reader(),
            stderr=await stderr_reader(),
        )
    except Exception as e:
        tb = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
        message = f'exception on running command {args.command}: {type(e).__name__}: {e}\n{tb}'
        logger.warning(message)
        return RunCommandResponse(status=CommandRunStatus.Error, stderr=message)


# ==================== Utils for stdio processing ====================


def try_decode(s: bytes) -> str:
    try:
        r = s.decode(encoding='utf-8', errors='replace')
    except Exception as e:
        r = f'[DecodeError] {e}'
    return r


async def get_output_reader(fd: StreamReader, max_out_bytes: int | None = None):
    res = b''

    async def reader():
        nonlocal res
        while True:
            chunk = await fd.read(1024 * 1024)
            if not chunk:
                break
            if not max_out_bytes or len(res) <= max_out_bytes:
                res += chunk
            del chunk

    task = asyncio.create_task(reader())

    async def read():
        try:
            await asyncio.wait_for(task, timeout=0.001)
        except TimeoutError:
            task.cancel()
        return try_decode(res)

    return read
