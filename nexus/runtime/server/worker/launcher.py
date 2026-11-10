"""WorkerLauncher — spawn and manage worker subprocess lifecycle."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import signal

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from nexus.runtime.server.venv.models import WorkerStatus
from nexus.utils.otel.environment import is_managed_otel_env_key, managed_otel_env


logger = logging.getLogger(__name__)

_STARTUP_OUTPUT_LIMIT = 4096


@dataclass
class _WorkerProcess:
    """Internal record for a running worker subprocess."""

    venv_id: str
    worker_id: str
    process: asyncio.subprocess.Process
    output_tasks: tuple[asyncio.Task[None], ...] = field(default_factory=tuple)
    output_tails: dict[str, bytearray] = field(default_factory=dict)
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class WorkerLauncher:
    """Spawn worker processes that connect back to the Base server via SIO.

    Each worker runs the WorkerSDK in a subprocess using the venv's Python
    interpreter.
    """

    def __init__(self) -> None:
        self._workers: dict[str, _WorkerProcess] = {}

    async def start_worker(
        self,
        venv_id: str,
        venv_dir: Path,
        *,
        base_url: str,
        worker_id: str | None = None,
        env_vars: dict[str, str] | None = None,
    ) -> WorkerStatus:
        """Launch a worker subprocess for the given venv.

        *base_url* is the Base server's address that the worker connects back to
        (e.g. ``http://localhost:8000``).  Passed directly as a CLI argument.
        """
        wid = worker_id or f'worker-{venv_id}'

        python = str(venv_dir / 'bin' / 'python')
        if not Path(python).exists():
            raise FileNotFoundError(f'Python not found at {python}')

        from nexus.utils.shell import get_clean_sandbox_env

        env = get_clean_sandbox_env()
        # Worker Python is a nix binary — it needs nix runtime libraries to import C extensions
        # (e.g., playwright). get_clean_sandbox_env() strips nix paths for tool isolation,
        # but the worker process itself must retain them.
        nix_ld = os.environ.get('LD_LIBRARY_PATH', '')
        if nix_ld:
            env['LD_LIBRARY_PATH'] = nix_ld
        env['PYTHONSAFEPATH'] = '1'
        if env_vars:
            env.update(env_vars)
        overlay = managed_otel_env(
            process_role='runtime-worker',
            service_name='nexus-runtime-worker',
            identifiers={'nexus.worker.id': wid},
        )
        if overlay:
            env = {key: value for key, value in env.items() if not is_managed_otel_env_key(key)}
            env.update(overlay)

        if overlay:
            from nexus.utils.otel.subprocess import preserve_managed_otel_environment

            environment_scope = preserve_managed_otel_environment()
        else:
            environment_scope = contextlib.nullcontext()
        with environment_scope:
            process = await asyncio.create_subprocess_exec(
                python,
                '-m',
                'nexus.runtime.server.worker.sdk',
                base_url,
                venv_id,
                wid,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )

        output_tails = {'stdout': bytearray(), 'stderr': bytearray()}
        output_tasks = (
            asyncio.create_task(
                self._drain_worker_stream(
                    process.stdout,
                    worker_id=wid,
                    stream_name='stdout',
                    output_tail=output_tails['stdout'],
                )
            ),
            asyncio.create_task(
                self._drain_worker_stream(
                    process.stderr,
                    worker_id=wid,
                    stream_name='stderr',
                    output_tail=output_tails['stderr'],
                )
            ),
        )
        wp = _WorkerProcess(
            venv_id=venv_id,
            worker_id=wid,
            process=process,
            output_tasks=output_tasks,
            output_tails=output_tails,
        )
        self._workers[wid] = wp

        logger.info('Launched worker %s (pid=%s) for venv %s', wid, process.pid, venv_id)

        return WorkerStatus(
            venv_id=venv_id,
            worker_id=wid,
            state='starting',
            pid=process.pid,
        )

    @staticmethod
    async def _drain_worker_stream(
        stream: asyncio.StreamReader | None,
        *,
        worker_id: str,
        stream_name: str,
        output_tail: bytearray,
    ) -> None:
        """Drain a worker pipe while retaining a bounded diagnostic tail."""
        if stream is None:
            return

        try:
            while chunk := await stream.read(64 * 1024):
                output_tail.extend(chunk)
                if len(output_tail) > _STARTUP_OUTPUT_LIMIT:
                    del output_tail[:-_STARTUP_OUTPUT_LIMIT]
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception('Failed to drain %s for worker %s', stream_name, worker_id)

    async def read_startup_output(self, worker_id: str, *, limit: int = 4096) -> str:
        """Return the bounded stdout/stderr tails retained by the stream drainers."""
        worker = self._workers.get(worker_id)
        if worker is None:
            return ''

        returncode = worker.process.returncode
        parts = [f'worker exited rc={returncode}' if returncode is not None else 'worker still running (rc=None)']

        for name in ('stderr', 'stdout'):
            data = bytes(worker.output_tails.get(name, b''))[-limit:]
            text = data.decode('utf-8', errors='replace').strip()
            if text:
                parts.append(f'{name}: {text}')

        return ' | '.join(parts)

    async def stop_worker(self, worker_id: str, timeout: float = 10.0) -> None:
        """Stop a worker subprocess gracefully."""
        wp = self._workers.pop(worker_id, None)
        if wp is None:
            logger.warning('Worker %s not found', worker_id)
            return

        try:
            if wp.process.returncode is None:
                # Try graceful SIGTERM first
                wp.process.send_signal(signal.SIGTERM)
                try:
                    await asyncio.wait_for(wp.process.wait(), timeout=timeout)
                except asyncio.TimeoutError:
                    logger.warning('Worker %s did not exit after SIGTERM, sending SIGKILL', worker_id)
                    wp.process.kill()
                    await wp.process.wait()
        finally:
            for task in wp.output_tasks:
                task.cancel()
            if wp.output_tasks:
                await asyncio.gather(*wp.output_tasks, return_exceptions=True)

        logger.info('Worker %s stopped (exit=%s)', worker_id, wp.process.returncode)

    async def stop_all(self) -> None:
        """Stop all running workers."""
        worker_ids = list(self._workers.keys())
        await asyncio.gather(*(self.stop_worker(wid) for wid in worker_ids), return_exceptions=True)

    def get_status(self, worker_id: str) -> WorkerStatus | None:
        """Get the status of a worker."""
        wp = self._workers.get(worker_id)
        if wp is None:
            return None
        state = 'running' if wp.process.returncode is None else 'stopped'
        return WorkerStatus(
            venv_id=wp.venv_id,
            worker_id=wp.worker_id,
            state=state,
            pid=wp.process.pid,
            connected_at=wp.started_at,
        )

    def list_workers(self) -> list[WorkerStatus]:
        return [s for wid in self._workers if (s := self.get_status(wid)) is not None]


__all__ = ['WorkerLauncher']
