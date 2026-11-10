"""File executor request/response models.

Pure Pydantic — no aiofiles dependency.
"""

from typing import ClassVar, Literal

from pydantic import BaseModel, Field

from nexus.runtime.executors.models.base import ExecutorRequest


class FileItem(BaseModel):
    path: str = Field(
        ...,
        description="File absolute path, e.g., 'folder/file.txt'. If relative path is provided, it will be resolved based on NEXUS_WORKDIR environment variable first, then current working directory.",
    )
    content: str | None = Field(default=None, description='File content, plain text or base64 encoded')
    encoding: Literal['utf8', 'base64'] = Field(
        default='utf8', description='Encoding method: utf8 (plain text) or base64 (binary)'
    )


class FileOperationResult(BaseModel):
    path: str = Field(..., description='File path')
    success: bool = Field(..., description='Whether the operation succeeded')
    message: str | None = Field(default=None, description='Error or success message')


class UploadFileResponse(BaseModel):
    results: list[FileOperationResult] = Field(..., description='Operation result for each file')


class UploadFileRequest(ExecutorRequest[UploadFileResponse]):
    executor_name: ClassVar[str] = 'upload_file'
    response_type: ClassVar[type[BaseModel]] = UploadFileResponse

    files: list[FileItem] = Field(..., description='List of files to upload')
    overwrite: bool = Field(default=True, description='Whether to overwrite existing files')

    model_config = {
        'json_schema_extra': {
            'example': {
                'files': [
                    {'path': '/tmp/nexus/docs/readme.txt', 'content': 'Hello World', 'encoding': 'utf8'},
                    {
                        'path': '/tmp/nexus/images/logo.png',
                        'content': 'iVBORw0KGgoAAAANSUhEUgAAAAoAAAAKCAYAAACNMs+9AAAAFUlEQVR42mP8z8BQz0AEYBxVSF+FABJADveWkH6oAAAAAElFTkSuQmCC',
                        'encoding': 'base64',
                    },
                ],
                'overwrite': True,
            }
        }
    }


class DownloadFileResponse(BaseModel):
    files: list[FileItem] = Field(..., description='Successfully downloaded files with content')
    errors: list[FileOperationResult] | None = Field(default=None, description='Failed file operations (if any)')


class DownloadFileRequest(ExecutorRequest[DownloadFileResponse]):
    executor_name: ClassVar[str] = 'download_file'
    response_type: ClassVar[type[BaseModel]] = DownloadFileResponse

    files: list[FileItem] = Field(
        ..., min_length=1, description='List of files to download (only path and encoding need to be specified)'
    )

    max_bytes: int | None = Field(default=None, ge=0, description='Maximum bytes per file; reject larger files')

    model_config = {
        'json_schema_extra': {
            'example': {
                'files': [
                    {'path': '/tmp/nexus/docs/readme.txt', 'encoding': 'utf8'},
                    {'path': '/tmp/nexus/images/logo.png', 'encoding': 'base64'},
                ]
            }
        }
    }
