from __future__ import annotations

import logging

from pathlib import Path
from typing import Any

import aiohttp

from aiohttp import web
from sqlalchemy.ext.asyncio import async_sessionmaker

from nexus.distributed.access.auth import (
    AuthorizationPolicy,
    create_auth_middleware,
)
from nexus.distributed.artifacts.admission import TransferAdmission
from nexus.distributed.artifacts.spool import ArtifactSpool
from nexus.distributed.artifacts.store import ArtifactUploadCoordinator, create_artifact_store
from nexus.distributed.artifacts.tickets import (
    ArtifactContentTicketSigner,
    UploadTicketSigner,
)
from nexus.distributed.config import DistributedConfig
from nexus.distributed.gateways.application.routes._cors import add_cors_headers as _add_cors_headers
from nexus.distributed.gateways.application.routes.live_compatibility import create_live_compatibility_app
from nexus.distributed.gateways.common.errors import handle_gateway_errors
from nexus.distributed.gateways.common.keys import (
    ARTIFACT_CONTENT_TICKET_SIGNER_KEY,
    ARTIFACT_DOWNLOAD_ADMISSION_KEY,
    ARTIFACT_SPOOL_KEY,
    ARTIFACT_STORE_KEY,
    ARTIFACT_UPLOAD_COORDINATOR_KEY,
    AUTHORIZATION_KEY,
    CONFIG_KEY,
    HTTP_CLIENT_KEY,
    SESSION_FACTORY_KEY,
    UPLOAD_TICKET_SIGNER_KEY,
)


logger = logging.getLogger(__name__)


@web.middleware
async def cors_middleware(request: web.Request, handler: Any) -> web.StreamResponse:
    if request.method == 'OPTIONS':
        return _add_cors_headers(web.Response(status=204), request)
    response = await handler(request)
    if not response.prepared:
        _add_cors_headers(response, request)
    return response


@web.middleware
async def error_middleware(request: web.Request, handler: Any) -> web.StreamResponse:
    return await handle_gateway_errors(
        request,
        handler,
        logger=logger,
        component='application gateway',
    )


async def health(_request: web.Request) -> web.Response:
    return web.json_response({'status': 'ok'})


def create_application_app(
    *,
    session_factory: async_sessionmaker,
    config: DistributedConfig,
    live_compatibility_control_base_url: str | None = None,
    live_compatibility_application_base_url: str | None = None,
    live_compatibility_default_placement: dict[str, Any] | None = None,
) -> web.Application:
    from nexus.distributed.gateways.application.routes import register_application_routes

    app = web.Application(
        middlewares=[
            cors_middleware,
            error_middleware,
            create_auth_middleware(session_factory=session_factory, config=config),
        ],
        client_max_size=config.gateway_json_body_max_size_bytes,
    )
    app[SESSION_FACTORY_KEY] = session_factory
    app[CONFIG_KEY] = config
    app[AUTHORIZATION_KEY] = AuthorizationPolicy()
    app[UPLOAD_TICKET_SIGNER_KEY] = UploadTicketSigner(
        keys=config.artifact_ticket_keyring(),
        current_key_id=config.artifact_upload_t