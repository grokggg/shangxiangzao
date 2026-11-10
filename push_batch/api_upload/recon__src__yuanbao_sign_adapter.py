# -*- coding: utf-8 -*-
"""HTTP adapter that signs Yuanbao Open API requests (V2)."""

from urllib.parse import urlparse

from requests import PreparedRequest, Response
from requests.adapters import HTTPAdapter

from yuanbao_convert.adapter.yuanbao_sign import build_sign_headers


class YuanbaoSignAdapter(HTTPAdapter):
    def __init__(self, app_key: str, secret: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.app_key = app_key
        self.secret = secret

    def send(self, request: PreparedRequest, **kwargs) -> Response:
        parsed = urlparse(request.url)
        method = request.method or "GET"
        sign_headers = build_sign_headers(
            method,
            parsed.path,
            parsed.query or "",
            self.app_key,
            self.secret,
        )
        for key, value in sign_headers.items():
            request.headers[key] = value

        if "Accept" not in request.headers:
            request.headers["Accept"] = "application/json"

        return super().send(request, **kwargs)
