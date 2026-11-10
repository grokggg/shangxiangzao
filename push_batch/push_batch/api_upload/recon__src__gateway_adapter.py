"""
AGS Gateway Transport for Python SDK

This module provides a custom HTTP adapter for routing SDK requests
through the AGS Gateway.
"""

import os
from pathlib import Path
from urllib.parse import urlparse
from enum import Enum
from typing import Optional, Union
from requests.adapters import HTTPAdapter
from requests import PreparedRequest, Response

from yuanbao_convert.etc import conf as CONF


class AGSGatewayType(Enum):
    """Enum for AGS Gateway protocol conversion types."""
    COS = "cos"
    TENCENTCLOUD = "tencentcloud"
    YUANBAO_OPEN_API = "yuanbao-open-api"


class AGSGatewayAdapter(HTTPAdapter):
    """
    Custom HTTP adapter that routes requests through AGS Gateway.

    This adapter intercepts SDK requests and modifies them to:
    1. Route through the gateway endpoint
    2. Add JWT authentication headers
    3. Preserve original host information
    """

    def __init__(
        self,
        gateway_endpoint: str,
        gateway_type: Union[AGSGatewayType, str],
        *args,
        target_name: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize the AGSGatewayAdapter.

        Args:
            gateway_endpoint: The AGS Gateway endpoint URL
                (e.g., "https://ags-gw-sample.gateway.ap-guangzhou.tencentags.com")
            gateway_type: The protocol conversion type
                (AGSGatewayType.COS, AGSGatewayType.TENCENTCLOUD, AGSGatewayType.YUANBAO_OPEN_API)
                or a raw target name string
            target_name: Optional explicit gateway target name under /targets/<name>/
        """
        super().__init__(*args, **kwargs)
        self.gateway_endpoint = gateway_endpoint.rstrip('/')
        self.gateway_type = gateway_type
        self.target_name = target_name or self._resolve_target_name(gateway_type)

    @staticmethod
    def _resolve_target_name(gateway_type: Union[AGSGatewayType, str]) -> str:
        if isinstance(gateway_type, str):
            return gateway_type
        if gateway_type == AGSGatewayType.YUANBAO_OPEN_API:
            return (
                CONF.AGS_GATEWAY_YUANBAO_OPEN_API_TARGET
                or AGSGatewayType.YUANBAO_OPEN_API.value
            )
        return gateway_type.value

    def send(self, request: PreparedRequest, **kwargs) -> Response:
        """
        Send the request through the gateway.

        Args:
            request: The prepared request to send
            **kwargs: Additional arguments passed to the parent send method

        Returns:
            Response object from the gateway
        """
        # Save original host
        original_url = urlparse(request.url)
        original_host = original_url.netloc

        # Parse gateway URL
        gateway_url = urlparse(
            f"{self.gateway_endpoint}/targets/{self.target_name}"
        )

        # Modify request URL to point to gateway
        # Keep the original path but add gateway path prefix
        new_path = gateway_url.path + original_url.path
        if original_url.query:
            new_path += f"?{original_url.query}"

        request.url = f"{gateway_url.scheme}://{gateway_url.netloc}{new_path}"

        # Set headers
        request.headers['X-AGS-Origin-Host'] = original_host
        request.headers["Host"] = gateway_url.netloc

        # Set JWT Token authentication header
        jwt_token = self._get_jwt_token()
        if jwt_token:
            request.headers['X-AGS-Authorization'] = f"Bearer {jwt_token}"

        # Send the modified request
        return super().send(request, **kwargs)

    def _get_jwt_token(self) -> str:
        """
        Read JWT token from file.

        Default path is ~/.ags/jwt, can be customized via AGS_JWT_PATH environment variable.

        Returns:
            JWT token string, or empty string if file doesn't exist or can't be read
        """
        # Get JWT file path
        jwt_path = os.environ.get('AGS_JWT_PATH')
        if not jwt_path:
            # Use default path ~/.ags/jwt
            home_dir = Path.home()
            jwt_path = home_dir / '.ags' / 'jwt'
        else:
            jwt_path = Path(jwt_path)

        try:
            # Read JWT token file
            with open(jwt_path, 'r', encoding='utf-8') as f:
                token = f.read().strip()
            return token
        except (FileNotFoundError, IOError):
            # File doesn't exist or can't be read, return empty string
            return ""
