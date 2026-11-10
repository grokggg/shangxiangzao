from __future__ import annotations

import logging

from typing import Any

from aiohttp import web
from sqlalchemy.ext.asyncio import async_sessionmaker

from nexus.distributed.access.auth import (
    AuthorizationPolicy,
    create_auth_middleware,
)
from nexus.distributed.access.taihu import parse_tai_identity
from nexus.distributed.access.trace_database import TraceDatabaseAccountManager
from nexus.distributed.artifacts.tickets import UploadTicketSigner
from nexus.distributed.config import DistributedConfig
from nexus.distributed.gateways.common.errors import handle_gateway_errors
from nexus.distributed.gateways.common.keys import (
    AUTHORIZATION_KEY,
    CONFIG_KEY,
    SESSION_FACTORY_KEY,
    TRACE_DATABASE_ACCOUNT_MANAGER_KEY,
    UPLOAD_TICKET_SIGNER_KEY,
)
from nexus.distributed.gateways.common.proxy import is_websocket_request as _shared_is_websocket_request
from nexus.distributed.gateways.common.proxy import proxy_request_headers as _shared_proxy_headers
from nexus.distributed.gateways.control.routes._allocations import allocation_payload as _route_allocation_payload
from nexus.distributed.gateways.control.routes.operations import CONSOLE_INDEX as _CONTROL_CONSOLE_INDEX
from nexus.distributed.gateways.control.routes.operations import (
    _database_runtime_metrics as _operation_database_runtime_metrics,
)
from nexus.distributed.persistence.db import run_idempotent_transaction as _db_run_idempotent_transaction
from nexus.distributed.repositories.access import AccessRepository
from nexus.distributed.utils.clock import utcnow
from nexus.utils.otel.propagation import inject_carrier as _otel_inject_carrier


logger = logging.getLogger(__name__)
CONSOLE_INDEX = _CONTROL_CONSOLE_INDEX
_allocation_payload = _route_allocation_payload
_database_runtime_metrics = _operation_database_runtime_metrics
_run_idempotent_transaction = _db_run_idempotent_transaction
_is_websocket_request = _shared_is_websocket_request
_proxy_headers = _shared_proxy_headers


def inject_carrier() -> dict[str, str]:
    """Compatibility hook for callers that patch trace propagation at the app entrypoint."""

    return _otel_inject_carrier()


@web.middleware
async def error_middleware(request: web.Request, handler: Any) -> web.StreamResponse:
    response = await handle_gateway_errors(
        request,
        handler,
        logger=logger,
        component='control gateway',
    )
    if request.path == '/v1/system/config-overrides':
        response.headers['Cache-Control'] = 'no-store'
    return response


def create_control_app(*, session_factory: async_sessionmaker, config: DistributedConfig) -> web.Application:
    from nexus.distributed.gateways.control.routes import register_control_routes

    app = web.Application(
        middlewares=[error_middleware, create_auth_middleware(session_factory=session_factory, config=config)],
        client_max_size=config.gateway_json_body_max_size_bytes,
    )
    app[SESSION_FACTORY_KEY] = session_factory
    app[CON