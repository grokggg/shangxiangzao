"""Virtual terminal screen built on pyte for capture_pane.

pyte's ByteStream is a pure-Python, byte-by-byte state machine: feeding a single
64KB PTY chunk can take 300-500ms of CPU. Running that inline on the asyncio
event loop (from PtyProcess._read's add_reader callback) would freeze every other
session and coroutine for that long. pyte is also NOT thread-safe and requires
bytes fed in arrival order.

So all pyte work runs on a single dedicated worker thread (ThreadPoolExecutor
with max_workers=1): submitting to it is FIFO, so feeds stay ordered, and feed
never runs concurrently with display/resize. feed() is sync (the add_reader
callback can't await) and returns immediately after submitting; display()/resize()
are async and await the same executor, so they observe all feeds submitted before
them.
"""

import asyncio
import logging

from concurrent.futures import ThreadPoolExecutor

import pyte

from wcwidth import wcswidth


logger = logging.getLogger(__name__)


def _render_line(line: dict, columns: int) -> str:
    output: list[str] = []
    skip_stub = False
    for x in range(columns):
        if skip_stub:
            skip_stub = False
            continue

        data = line[x].data
        if not data:
            # A stub without a preceding wide character is an orphan left by
            # pyte when the leading cell is overwritten. Preserve its column.
            output.append(' ')
            continue

        output.append(data)
        skip_stub = wcswidth(data) == 2
    return ''.join(output).rstrip()


class VirtualScreen:
    """Wrap a pyte HistoryScreen + ByteStream, driven on a single worker thread."""

    def __init__(self, rows: int = 40, cols: int = 120) -> None:
        self._screen = pyte.HistoryScreen(cols, rows, history=2000, ratio=0.5)
        self._stream = pyte.ByteStream(self._screen)
        # max_workers=1: the single thread serializes all pyte access (feed/
        # display/resize) and preserves feed ordering via the FIFO submit queue.
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='pyte')

    def feed(self, data: bytes) -> None:
        """Submit a chunk for parsing off the event loop. Returns immediately.

        Called from the PtyProcess add_reader callback (sync), so it cannot await;
        it hands the chunk to the worker thread and returns. Ordering is preserved
        because submits to a single-worker executor run FIFO.
        """
        future = self._executor.submit(self._stream.feed, data)
        future.add_done_callback(self._log_feed_error)

    @staticmethod
    def _log_feed_error(future) -> None:
        exc = future.exception()
        if exc is not None:
            logger.warning('pyte feed failed: %r', exc)

    async def resize(self, rows: int, cols: int) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._executor, self._screen.resize, rows, cols)

    async def display(self, entire: bool = False) -> str:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self._display_sync, entire)

    def _display_sync(self, entire: bool) -> str:
        columns = self._screen.columns
        visible_lines = [_render_line(self._screen.buffer[y], columns) for y in range(self._screen.lines)]
        visible = '\n'.join(visible_lines).rstrip('\n')
        if not entire:
            return self._rstrip_blank_lines(visible)
        history_lines = [_render_line(line, columns) for line in self._screen.history.top]
        combined = '\n'.join(history_lines + visible.splitlines())
        return self._rstrip_blank_lines(combined)

    def close(self) -> None:
        """Shut down the worker thread. Idempotent."""
        self._executor.shutdown(wait=False)

    @staticmethod
    def _rstrip_blank_lines(text: str) -> str:
        lines = text.split('\n')
        while lines and lines[-1].strip() == '':
            lines.pop()
        return '\n'.join(line.rstrip() for line in lines)
