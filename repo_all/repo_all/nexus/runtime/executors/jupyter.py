# /// script
# dependencies = []
#
# [project]
# name = "jupyter"
# authors = [
#     { name = "tarschen" },
#     { name = "felixjjiang" },
# ]
#
# [optional-dependencies]
# runtime = ["ipykernel>=6.29.0", "jupyter-client>=8.6.0"]
#
# [tool.nexus]
# executors = { create_ipython_session = "nexus.runtime.executors.jupyter", destroy_ipython_session = "nexus.runtime.executors.jupyter", execute_ipython = "nexus.runtime.executors.jupyter" }
# ///

import asyncio
import logging
import os
import sys
import time
import uuid

from typing import TYPE_CHECKING, Any

from jupyter_client.manager import AsyncKernelManager

from nexus.core import executor_registry
from nexus.runtime.executors.models.jupyter import *  # noqa: F401,F403
from nexus.runtime.executors.models.jupyter import (
    CreateIPythonSessionRequest,
    CreateIPythonSessionResponse,
    DestroyIPythonSessionRequest,
    DestroyIPythonSessionResponse,
    ExecuteIPythonRequest,
    ExecuteIPythonResponse,
)
from nexus.utils.shell import get_default_pwd, get_os_env_without_ld_library_path


if TYPE_CHECKING:
    from jupyter_client.asynchronous.client import AsyncKernelClient

logger = logging.getLogger(__name__)
IPY_SESSIONS: dict[str, 'AsyncDirectPythonKernel'] = {}

# Environment variable to override the default Python path for Jupyter kernels
# This allows using the base image's Python (with scientific libraries) instead of
# the nexus runtime's Python
JUPYTER_PYTHON_PATH_ENV = 'JUPYTER_PYTHON_PATH'

# Track which Python paths have had ipykernel installed
_IPYKERNEL_INSTALLED: set[str] = set()

# ==================== Registered Executors ====================


@executor_registry.register('create_ipython_session')
async def create_ipython_session(request: CreateIPythonSessionRequest) -> CreateIPythonSessionResponse:
    session_id = str(uuid.uuid4())
    kernel = AsyncDirectPythonKernel(request.python_executable)
    await kernel.start()
    IPY_SESSIONS[session_id] = kernel
    return CreateIPythonSessionResponse(session_id=session_id)


@executor_registry.register('destroy_ipython_session')
async def destroy_ipython_session(request: DestroyIPythonSessionRequest) -> DestroyIPythonSessionResponse:
    session_id = request.session_id
    kernel = IPY_SESSIONS.pop(session_id, None)
    if kernel:
        await kernel.close()
    return DestroyIPythonSessionResponse(success=True)


@executor_registry.register('execute_ipython')
async def execute_ipython(request: ExecuteIPythonRequest) -> ExecuteIPythonResponse:
    session_id = request.session_id
    kernel = IPY_SESSIONS.get(session_id)
    if not kernel:
        raise ValueError(f'Session {session_id} not found')
    return await kernel.execute(request.code, timeout=request.timeout)


# ==================== Helper Functions ====================


async def _ensure_ipykernel_installed(python_path: str) -> None:
    """Ensure ipykernel is installed for the given Python interpreter.

    This is needed when using an external Python (e.g., from the base image)
    that may not have ipykernel pre-installed.

    This function only validates and raises on missing dependency; it does not
    attempt to auto-install ipykernel.
    """
    if python_path in _IPYKERNEL_INSTALLED:
        logger.info('ipykernel already verified for %s (cached)', python_path)
        return

    logger.info('Checking ipykernel installation for %s...', python_path)
    try:
        process = await asyncio.create_subprocess_exec(
            python_path,
            '-m',
            'ipykernel',
            '--version',
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr_bytes = await asyncio.wait_for(process.communicate(), timeout=30)
    except FileNotFoundError as e:
        raise FileNotFoundError(f'Python path does not exist: {python_path}') from e
    except TimeoutError as e:
        raise TimeoutError(f'Timeout while checking ipykernel for interpreter {python_path}') from e

    if process.returncode != 0:
        stderr_msg = stderr_bytes.decode('utf-8', errors='replace').strip() if stderr_bytes else ''
        raise RuntimeError(
            f'ipykernel is not available for interpreter {python_path}: {stderr_msg or f"exit code {process.returncode}"}. '
            f'Please install it manually with "{python_path} -m pip install ipykernel".'
        )

    logger.info('ipykernel already installed for %s', python_path)
    _IPYKERNEL_INSTALLED.add(python_path)


# ==================== AsyncDirectPythonKernel Implementation ====================


class _ExecuteResult:
    """Mutable container for execution result to avoid dict type inference issues."""

    def __init__(self):
        self.stdout: str = ''
        self.stderr: str = ''
        self.display: list[dict[str, Any]] = []
        self.error: list[dict[str, Any]] = []
        self.status: str = ''
        self.execution_time: float | None = None


class AsyncDirectPythonKernel:
    def __init__(self, python_path: str | None):
        # Priority: explicit parameter > environment variable > sys.executable
        # The environment variable allows users to specify a Python from the base image
        # (e.g., /root/miniconda3/bin/python with numpy installed) instead of the
        # nexus runtime's Python (/nix/runtime/nexus/.venv-3.11/bin/python)
        if python_path:
            self.python_path = python_path
        else:
            env_python_path = os.environ.get(JUPYTER_PYTHON_PATH_ENV)
            if env_python_path:
                logger.info('Using JUPYTER_PYTHON_PATH from environment: %s', env_python_path)
                self.python_path = env_python_path
            else:
                self.python_path = sys.executable
        self.km: AsyncKernelManager | None = None
        self.kc: AsyncKernelClient | None = None

    async def start(self, max_tries: int = 5):
        # Ensure ipykernel is installed for external Python interpreters
        logger.info(f'Starting kernel with python_path: {self.python_path} (sys.executable: {sys.executable})')
        if self.python_path != sys.executable:
            await _ensure_ipykernel_installed(self.python_path)

        num_tries = 0
        last_error: Exception | None = None
        while num_tries < max_tries:
            try:
                km: AsyncKernelManager = AsyncKernelManager()
                # HACK: set python interpreter path
                logger.debug('original kernel spec: ' + str(km.kernel_spec.__dict__))
                if km._kernel_spec is not None:
                    km._kernel_spec.argv[0] = self.python_path
                logger.info(f'kernel spec argv: {km._kernel_spec.argv if km._kernel_spec else "None"}')
                kwargs: dict[str, Any] = {'cwd': get_default_pwd()}
                if self.python_path != sys.executable:
                    # running external python interpreter, avoid messing it up with nix env
                    kwargs['env'] = get_os_env_without_ld_library_path()
                logger.info(f'Starting kernel (attempt {num_tries + 1}/{max_tries})...')
                await km.start_kernel(**kwargs)
                logger.info('Kernel process started, creating client...')
                kc = km.client()
                kc.start_channels()
                logger.info('Waiting for kernel ready...')
                await kc.wait_for_ready()
                logger.info('Kernel ready, testing with print(123)...')

                stdout = ''

                def hook(msg):
                    nonlocal stdout
                    msg_type = msg['header']['msg_type']
                    content = msg['content']
                    if msg_type == 'stream' and content['name'] == 'stdout':
                        stdout += content['text']

                await kc.execute_interactive('print(123)', output_hook=hook, timeout=2)
                if stdout != '123\n':
                    raise Exception(f'incorrect kernel output: expected "123\\n", got "{stdout}"')
                logger.info('Kernel started successfully')
                self.km, self.kc = km, kc
                return self
            except Exception as e:
                import traceback

                logger.warning(f'kernel failed to start (attempt {num_tries + 1}/{max_tries}): {e}')
                logger.warning(f'traceback: {traceback.format_exc()}')
                last_error = e
                try:
                    await km.shutdown_kernel()
                except Exception as err:
                    logger.debug(f'failed to shutdown kernel, ignored: {err}')
                num_tries += 1
                if num_tries >= max_tries:
                    raise Exception(
                        f'failed to start kernel after {max_tries} tries. last error: {last_error}'
                    ) from last_error
        return None

    async def close(self):
        if self.km:
            await self.km.shutdown_kernel()
        if self.kc:
            self.kc.stop_channels()

    async def execute(self, code: str, timeout: float | None = None) -> ExecuteIPythonResponse:
        if not self.kc or not self.km:
            raise RuntimeError("Kernel not started. Call 'await kernel.start()' first.")

        result = _ExecuteResult()

        def hook(msg: dict[str, Any]) -> None:
            msg_type = msg['header']['msg_type']
            content = msg['content']
            if msg_type == 'stream':
                if content['name'] == 'stdout':
                    result.stdout += content['text']
                elif content['name'] == 'stderr':
                    result.stderr += content['text']
            elif msg_type in ('display_data', 'execute_result'):
                result.display.append(content['data'])
            elif msg_type == 'error':
                result.error.append(content)

        try:
            start_time = time.time()
            exec_result = await self.kc.execute_interactive(code, timeout=timeout, output_hook=hook)
            result.status = exec_result['content']['status']
            result.execution_time = time.time() - start_time
        except TimeoutError:
            result.status = 'TimeLimitExceeded'
            result.execution_time = time.time() - start_time
            await self.km.interrupt_kernel()
            num_tries, max_tries = 0, 60
            while num_tries < max_tries:
                try:
                    msg = await asyncio.wait_for(self.kc.get_iopub_msg(), timeout=1)
                    if msg['header']['msg_type'] == 'status' and msg['content']['execution_state'] == 'idle':
                        break
                except TimeoutError:
                    await asyncio.sleep(0.1)
                except Exception as e:
                    logger.debug(f'Error while clearing messages: {e}')
                    break
                num_tries += 1
            if num_tries >= max_tries:
                logger.warning('Kernel did not respond to interrupt, restarting...')
                try:
                    await self.close()
                    await self.start()
                except Exception as e:
                    logger.error(f'Failed to restart kernel: {e}')

        return ExecuteIPythonResponse(
            status=result.status,
            stdout=result.stdout,
            stderr=result.stderr,
            display=result.display,
            error=result.error,
            execution_time=result.execution_time,
        )
