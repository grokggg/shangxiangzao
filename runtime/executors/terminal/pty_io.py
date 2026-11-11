"""Single-PTY low-level I/O: spawn shell, read the master, resize, kill.

stdin/stdout/stderr all share one PTY (the controlling terminal). A single
shared PTY keeps shell job control working, which is required for interactive
programs (python/vim REPLs) and Ctrl-C to function — routing stderr to a second
PTY breaks job control and suspends interactive children with SIGTTIN.

The cost is that stdout and stderr are merged on the PTY screen (capture_pane);
per-command stdout/stderr are separated by bash-capture files.
"""

import asyncio
import contextlib
import fcntl
import os
import pty
import signal
import struct
import termios

from collections.abc import Callable

import psutil


def set_winsize(fd: int, rows: int, cols: int) -> None:
    winsize = struct.pack('HHHH', rows, cols, 0, 0)
    fcntl.ioctl(fd, termios.TIOCSWINSZ, winsize)


class PtyProcess:
    def __init__(
        self,
        proc: asyncio.subprocess.Process,
        master: int,
        on_data: Callable[[bytes], None],
        on_exit: Callable[[], None],
    ) -> None:
        self.proc = proc
        self.master = master
        # The child is a session leader (os.setsid() in _preexec), so its pgid
        # equals its pid. If the process already exited and was reaped before we
        # read it, getpgid raises ProcessLookupError — fall back to proc.pid,
        # which is exactly what the pgid was.
        try:
            self.shell_pgid = os.getpgid(proc.pid)
        except ProcessLookupError:
            self.shell_pgid = proc.pid
        self._on_data = on_data
        self._on_exit = on_exit
        self._closed = False
        self._write_lock = asyncio.Lock()
        self._write_fut: asyncio.Future[None] | None = None
        self._loop = asyncio.get_running_loop()
        self._loop.add_reader(master, self._read)
        self._exit_task = asyncio.create_task(self._watch_exit())

    @classmethod
    async def spawn(
        cls,
        argv: list[str],
        env: dict[str, str],
        rows: int,
        cols: int,
        on_data: Callable[[bytes], None],
        on_exit: Callable[[], None],
    ) -> 'PtyProcess':
        if not 1 <= rows <= 65535 or not 1 <= cols <= 65535:
            raise ValueError('terminal rows and columns must be between 1 and 65535')

        master, slave = pty.openpty()
        proc: asyncio.subprocess.Process | None = None
        try:
            set_winsize(master, rows, cols)

            flags = fcntl.fcntl(master, fcntl.F_GETFL)
            fcntl.fcntl(master, fcntl.F_SETFL, flags | os.O_NONBLOCK)

            def _preexec() -> None:
                # Become session leader, THEN claim the slave as controlling tty.
                # We call setsid() ourselves rather than relying on
                # start_new_session=True: under uvloop (the runtime server's loop),
                # start_new_session's setsid() does not run before preexec_fn, so the
                # child is not yet a session leader and TIOCSCTTY fails with EPERM.
                # Calling setsid() here makes the ordering explicit and works under
                # both the default asyncio loop and uvloop.
                os.setsid()
                fcntl.ioctl(slave, termios.TIOCSCTTY, 0)

            proc = await asyncio.create_subprocess_exec(
                *argv,
                stdin=slave,
                stdout=slave,
                stderr=slave,
                preexec_fn=_preexec,
                env=env,
            )
            os.close(slave)
            slave = -1
            result = cls(proc, master, on_data, on_exit)
            master = -1
            return result
        finally:
            if slave >= 0:
                with contextlib.suppress(OSError):
                    os.close(slave)
            if master >= 0:
                with contextlib.suppress(OSError):
                    os.close(master)
                if proc is not None and proc.returncode is None:
                    with contextlib.suppress(ProcessLookupError):
                        proc.kill()
                    with contextlib.suppress(Exception):
                        await proc.wait()

    def _read(self) -> None:
        try:
            data = os.read(self.master, 65536)
        except (BlockingIOError, InterruptedError):
            return
        except OSError:
            data = b''
        if data:
            self._on_data(data)
        else:
            with contextlib.suppress(ValueError, OSError):
                self._loop.remove_reader(self.master)

    async def _watch_exit(self) -> None:
        await self.proc.wait()
        self._on_exit()

    async def write(self, data: bytes) -> None:  # noqa: C901
        """Write all of *data* to the master without blocking the event loop."""
        if not data:
            return
        loop = self._loop
        async with self._write_lock:
            fut: asyncio.Future[None] = loop.create_future()
            self._write_fut = fut
            mv = memoryview(data)
            state = {'off': 0}

            def _on_writable() -> None:
                try:
                    while state['off'] < len(mv):
                        try:
                            n = os.write(self.master, mv[state['off'] :])
                        except (BlockingIOError, InterruptedError):
                            return
                        except OSError as exc:
                            loop.remove_writer(self.master)
                            if not fut.done():
                                fut.set_exception(exc)
                            return
                        state['off'] += n
                    loop.remove_writer(self.master)
                    if not fut.done():
                        fut.set_result(None)
                except Exception as exc:  # noqa: BLE001
                    loop.remove_writer(self.master)
                    if not fut.done():
                        fut.set_exception(exc)

            loop.add_writer(self.master, _on_writable)
            try:
                await fut
            finally:
                with contextlib.suppress(ValueError, OSError):
                    loop.remove_writer(self.master)
                self._write_fut = None

    def resize(self, rows: int, cols: int) -> None:
        set_winsize(self.master, rows, cols)

    def killpg(self, sig: int = signal.SIGKILL) -> None:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(self.shell_pgid, sig)

    def _descendant_signal_targets(self) -> tuple[set[int], set[int]]:
        """Return (independent process groups, PIDs sharing the shell group)."""
        try:
            children = psutil.Process(self.proc.pid).children(recursive=True)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return set(), set()

        groups: set[int] = set()
        pids: set[int] = set()
        for child in children:
            try:
                pgid = os.getpgid(child.pid)
            except (ProcessLookupError, PermissionError):
                continue
            if pgid > 0 and pgid != self.shell_pgid:
                groups.add(pgid)
            elif pgid == self.shell_pgid:
                pids.add(child.pid)
        return groups, pids

    @staticmethod
    def _signal_targets(groups: set[int], pids: set[int], sig: int) -> None:
        for pgid in groups:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(pgid, sig)
        for pid in pids:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.kill(pid, sig)

    async def terminate_descendants(self, grace_seconds: float = 3.0) -> list[int]:
        """Terminate every shell job, including background and stopped jobs.

        Process groups are collected from the shell's descendant tree, so the
        shell itself survives. SIGCONT lets stopped jobs process termination;
        SIGKILL removes anything still present after the grace period.
        """
        groups, pids = self._descendant_signal_targets()
        if not groups and not pids:
            return []

        all_groups = set(groups)
        all_pids = set(pids)
        self._signal_targets(groups, pids, signal.SIGCONT)
        self._signal_targets(groups, pids, signal.SIGTERM)

        loop = asyncio.get_running_loop()
        deadline = loop.time() + grace_seconds
        while loop.time() < deadline:
            current_groups, current_pids = self._descendant_signal_targets()
            new_groups = current_groups - all_groups
            new_pids = current_pids - all_pids
            if new_groups or new_pids:
                all_groups.update(new_groups)
                all_pids.update(new_pids)
                self._signal_targets(new_groups, new_pids, signal.SIGCONT)
                self._signal_targets(new_groups, new_pids, signal.SIGTERM)
            if not current_groups and not current_pids:
                return sorted(all_groups)
            await asyncio.sleep(min(0.05, max(0.0, deadline - loop.time())))

        remaining_groups, remaining_pids = self._descendant_signal_targets()
        all_groups.update(remaining_groups)
        all_pids.update(remaining_pids)
        self._signal_targets(remaining_groups, remaining_pids, signal.SIGKILL)
        return sorted(all_groups)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        with contextlib.suppress(ValueError, OSError):
            self._loop.remove_reader(self.master)
        with contextlib.suppress(ValueError, OSError):
            self._loop.remove_writer(self.master)
        if self._write_fut is not None and not self._write_fut.done():
            self._write_fut.set_exception(OSError('pty closed during write'))
        with contextlib.suppress(OSError):
            os.close(self.master)
        if not self._exit_task.done():
            self._exit_task.cancel()
