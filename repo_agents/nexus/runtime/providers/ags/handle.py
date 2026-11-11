"""AGS instance handle — connection details for an AGS sandbox instance."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AGSInstanceHandle:
    """Connection details for an AGS sandbox instance."""

    instance_id: str
    access_token: str = field(repr=False)
    domain: str
    tool_id: str | None = None
    tool_name: str | None = None
    tool_type: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def headers(self) -> dict[str, str]:
        return {'X-Access-Token': self.access_token}

    def host(self, port: int) -> str:
        return f'{port}-{self.instance_id}.{self.domain}'

    def url(self, port: int, path: str = '') -> str:
        normalized_path = path if path.startswith('/') or not path else f'/{path}'
        return f'https://{self.host(port)}{normalized_path}'
