"""Read and validate bash-capture files for terminal sessions.

The bash-capture runtime writes one directory per TerminalSession:

- status.json: current capture status and the latest command id;
- cmd_<id>.json: command metadata, including batch and lifecycle timestamps;
- cmd_content_<id>: command text;
- stdout_<id>, stderr_<id>, output_<id>: captured byte streams.

CaptureReader is the only parser for this file contract. It performs async file
IO, Pydantic validation, CR normalization, and batch collection.
"""

import os

from pydantic import BaseModel

from nexus.utils.aiofiles import read_file_or_empty, read_json_model_or_none


class CaptureStatus(BaseModel):
    """Parsed status.json."""

    status: str
    timestamp: float | None = None
    command_id: int = 0


class CommandMeta(BaseModel):
    """Parsed cmd_<id>.json metadata."""

    command_id: int
    batch: int | None = None
    start_time: float | None = None
    end_time: float | None = None
    exit_code: int | None = None


class CommandRecord(BaseModel):
    """Fully assembled command record from metadata, command text, and streams."""

    command_id: int
    batch: int | None = None
    command: str | None = None
    exit_code: int | None = None
    start_time: float | None = None
    end_time: float | None = None
    stdout: str = ''
    stderr: str = ''
    output: str = ''


class BatchCapture(BaseModel):
    """Contiguous command records belonging to one capture batch."""

    records: list[CommandRecord]
    all_finished: bool


def normalize_cr(text: str) -> str:
    """Collapse CR framing to newlines: '\r\n' -> '\n', lone '\r' -> '\n'.

    No ANSI/OSC stripping: the bash-capture layer writes command output from
    pipes, not from the echoed PTY screen.
    """
    return text.replace('\r\n', '\n').replace('\r', '\n')


class CaptureReader:
    """Async reader for one TerminalSession capture directory."""

    def __init__(self, output_dir: str) -> None:
        self.output_dir = output_dir

    def _path(self, name: str) -> str:
        return os.path.join(self.output_dir, name)

    async def read_status(self) -> CaptureStatus | None:
        """Parse status.json, or return None if it is absent or invalid."""
        return await read_json_model_or_none(self._path('status.json'), CaptureStatus)

    async def read_command(self, command_id: int) -> CommandRecord | None:
        """Assemble one command record, or return None if metadata is missing.

        Missing stream files are treated as empty strings. This is both useful for
        partially written foreground commands and explicit for skip-capture
        background commands, which materialize empty data files.
        """
        meta = await read_json_model_or_none(self._path(f'cmd_{command_id}.json'), CommandMeta)
        if meta is None:
            return None

        command = await read_file_or_empty(self._path(f'cmd_content_{command_id}'))
        stdout = await read_file_or_empty(self._path(f'stdout_{command_id}'))
        stderr = await read_file_or_empty(self._path(f'stderr_{command_id}'))
        output = await read_file_or_empty(self._path(f'output_{command_id}'))
        return CommandRecord(
            command_id=meta.command_id,
            batch=meta.batch,
            command=command or None,
            exit_code=meta.exit_code,
            start_time=meta.start_time,
            end_time=meta.end_time,
            stdout=normalize_cr(stdout),
            stderr=normalize_cr(stderr),
            output=normalize_cr(output),
        )

    async def read_batch(self, first_id: int) -> BatchCapture:
        """Collect contiguous command records that share the first record's batch.

        command_id values are contiguous in one serial Bash process, so reading
        stops at the first missing metadata file or first different batch id. If
        the first record has no batch field, return only that record to avoid
        merging unrelated legacy/regression records where None == None.
        """
        records: list[CommandRecord] = []
        batch: int | None = None
        command_id = first_id
        while True:
            record = await self.read_command(command_id)
            if record is None:
                break
            if not records:
                batch = record.batch
                records.append(record)
                if batch is None:
                    break
            else:
                if record.batch != batch:
                    break
                records.append(record)
            command_id += 1
        all_finished = bool(records) and all(record.end_time is not None for record in records)
        return BatchCapture(records=records, all_finished=all_finished)
