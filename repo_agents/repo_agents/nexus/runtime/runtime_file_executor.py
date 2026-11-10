# /// script
# dependencies = []
#
# [project]
# name = "file"
# authors = [
#     { name = "tarschen" },
# ]
#
# [tool.nexus]
# executors = { upload_file = "nexus.runtime.executors.file", download_file = "nexus.runtime.executors.file" }
# ///

import base64
import os

from pathlib import Path

import aiofiles
import aiofiles.os

from nexus.core import executor_registry
from nexus.runtime.executors.models.file import *  # noqa: F401,F403
from nexus.runtime.executors.models.file import (
    DownloadFileRequest,
    DownloadFileResponse,
    FileItem,
    FileOperationResult,
    UploadFileRequest,
    UploadFileResponse,
)


def _resolve_path(path: str) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p

    workdir = os.getenv('NEXUS_WORKDIR')
    if workdir:
        return Path(workdir) / p

    return Path.cwd() / p


@executor_registry.register('upload_file')
async def upload_file(request: UploadFileRequest) -> UploadFileResponse:
    results = []

    for file_item in request.files:
        try:
            if file_item.content is None:
                results.append(FileOperationResult(path=file_item.path, success=False, message='Content is required'))
                continue

            file_path = _resolve_path(file_item.path)

            if await aiofiles.os.path.exists(file_path) and not request.overwrite:
                results.append(
                    FileOperationResult(
                        path=str(file_path), success=False, message='File exists and overwrite is disabled'
                    )
                )
                continue

            await aiofiles.os.makedirs(file_path.parent, exist_ok=True)

            if file_item.encoding == 'base64':
                content_bytes = base64.b64decode(file_item.content)
                async with aiofiles.open(file_path, 'wb') as f:
                    await f.write(content_bytes)
            else:
                async with aiofiles.open(file_path, 'w', encoding='utf-8') as f:
                    await f.write(file_item.content)

            results.append(FileOperationResult(path=str(file_path), success=True, message='File uploaded successfully'))
        except Exception as e:
            results.append(FileOperationResult(path=file_item.path, success=False, message=f'Error: {str(e)}'))

    return UploadFileResponse(results=results)


@executor_registry.register('download_file')
async def download_file(request: DownloadFileRequest) -> DownloadFileResponse:
    files = []
    errors = []

    for file_item in request.files:
        try:
            file_path = _resolve_path(file_item.path)

            if not await aiofiles.os.path.exists(file_path):
                errors.append(FileOperationResult(path=str(file_path), success=False, message='File does not exist'))
                continue

            if not await aiofiles.os.path.isfile(file_path):
                errors.append(FileOperationResult(path=str(file_path), success=False, message='Path is not a file'))
                continue

            async with aiofiles.open(file_path, 'rb') as f:
                content_bytes = await f.read(-1 if request.max_bytes is None else request.max_bytes + 1)
            if request.max_bytes is not None and len(content_bytes) > request.max_bytes:
                raise ValueError(f'File exceeds {request.max_bytes} byte limit')
            if file_item.encoding == 'base64':
                content = base64.b64encode(content_bytes).decode('utf-8')
            else:
                content = content_bytes.decode('utf-8').replace('\r\n', '\n').replace('\r', '\n')

            files.append(FileItem(path=str(file_path), content=content, encoding=file_item.encoding))
        except Exception as e:
            errors.append(FileOperationResult(path=file_item.path, success=False, message=f'Error: {str(e)}'))

    return DownloadFileResponse(files=files, errors=errors if errors else None)
