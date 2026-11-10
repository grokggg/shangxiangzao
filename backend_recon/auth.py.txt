from __future__ import annotations

import hashlib
import hmac
import logging

from dataclasses import dataclass
from typing import Any

from aiohttp import web
from sqlalchemy.ext.asyncio import async_sessionmaker

from nexus.distributed.access.models import SubmitTokenKind
from nexus.distributed.access.taihu import (
    TAI_IDENTITY_HEADER,
    decrypt_tai_identity,
    has_sgw_auth_headers,
    has_sgw_backend_signature_headers,
    sgw_auth_header_counts,
    trusted_sgw_app_id,
    verify_sgw_backend_signature,
)
from nexus.distributed.config import DistributedConfig
from nexus.distributed.repositories.access import AccessRepository, AccessUser, AuthenticatedAppToken
from nexus.distributed.utils.clock import utcnow


@dataclass(frozen=True, slots=True)
class AuthContext:
    principal_id: str
    issuer: str
    tenant: str
    subject: str
    roles: frozenset[str]
    scopes: frozenset[str]
    auth_type: str
    system_token_id: str | None = None
    display_name: str | None = None
    actor_app_id: str | None = None
    credential_token_id: str | None = None
    credential_environment_id: str | None = None
    internal_delegation: bool = False
    accessible_owner_principal_ids: frozenset[str] = frozenset()

    @property
    def is_admin(self) -> bool:
        return 'admin' in self.roles


class AuthorizationPolicy:
    def require_scope(self, context: AuthContext, scope: str) -> None:
        if '*' not in context.scopes and scope not in context.scopes:
            raise web.HTTPForbidden(text='scope_required')

    def require_admin(self, context: AuthContext) -> None:
        if not context.is_admin:
            raise web.HTTPForbidden(text='admin_required')

    def effective_owner(self, context: AuthContext, requested_owner: str | None) -> str:
        if requested_owner is None or requested_owner == context.principal_id:
            return context.principal_id
        self.require_admin(context)
        self.require_scope(context, 'submission:as_user')
        return requested_owner

    def require_owner(self, context: AuthContext, owner_principal_id: str) -> None:
        if context.is_admin or owner_principal_id == context.principal_id:
            return
        raise web.HTTPNotFound(text='resource_not_found')

    def require_shared_owner_access(self, context: AuthContext, owner_principal_id: str) -> None:
        if owner_principal_id in context.accessible_owner_principal_ids:
            return
        self.require_owner(context, owner_principal_id)

    def require_environment(self, context: AuthContext, environment_id: str | None) -> None:
        if context.credential_environment_id is not None and environment_id != context.credential_environment_id:
            raise web.HTTPNotFound(text='resource_not_found')

    def accessible_owner_ids(self, context: AuthContext) -> frozenset[str] | None:
        if context.is_admin:
            return None
        return frozenset({context.principal_id, *context.accessible_owner_principal_ids})

    def require_execution_access(
        self,
        context: AuthContext,
        *,
        owner_principal_id: str,
        environment_id: str | None,
    ) -> None:
        self.require_shared_owner_access(context, owner_principal_id)
        self.require_environment(context, environment_id)


AUTH_CONTEXT_KEY = 'nexus.auth_context'
CSRF_HEADER = 'X-Nexus-CSRF-Token'
DELEGATED_APP_ID_HEADER = 'X-Nexus-Delegated-App-Id'
DELEGATED_PRINCIPAL_ID_HEADER = 'X-Nexus-Delegated-Principal-Id'
INTERNAL_TOKEN_HEADER = 'X-Nexus-Internal-Token'
WEBSOCKET_CSRF_PROTOCOL_PREFIX = 'nexus-csrf.'
logger = logging.getLogger(__name__)


def create_auth_middleware(
    *,
    session_factory: async_sessionmaker,
    config: DistributedConfig,
):
    @web.middleware
    async def auth_middleware(request: web.Request, handler: Any) -> web.StreamResponse:
        if _is_unauthenticated_route(request):
            return await handler(request)
        try:
            context = await _authenticate(request, session_factory=session_factory, config=config)
        except PermissionError as exc:
            raise web.HTTPUnauthorized(text=str(exc)) from exc
        _enforce_browser_origin(request, context=context, config=config)
        request[AUTH_CONTEXT_KEY] = context
        return await handler(request)

    return auth_middleware


def auth_context(request: web.Request) -> AuthContext:
    context = request.get(AUTH_CONTEXT_KEY)
    if not isinstance(context, AuthContext):
        raise web.HTTPUnauthorized(text='authentication_required')
    return context


async def _authenticate(
    request: web.Request,
    *,
    session_factory: async_sessionmaker,
    config: DistributedConfig,
) -> AuthContext:
    if config.auth_mode == 'disabled':
        return AuthContext(
            principal_id='local-dev',
            issuer='local',
            tenant='local',
            subject='local-dev',
            roles=frozenset({'user', 'admin'}),
            scopes=frozenset({'*'}),
            auth_type='disabled',
            display_name='Local Developer',
        )

    if config.auth_mode not in {'identity', 'required'}:
        raise PermissionError('authentication_required')
    delegated_app_id = request.headers.get(DELEGATED_APP_ID_HEADER)
    delegated_principal_id = request.headers.get(DELEGATED_PRINCIPAL_ID_HEADER)
    delegated_internal_token = request.headers.get(INTERNAL_TOKEN_HEADER)
    raw_identity = request.headers.get(TAI_IDENTITY_HEADER)
    has_sgw_identity = has_sgw_auth_headers(request.headers)
    has_internal_headers = any(
        value is not None for value in (delegated_app_id, delegated_principal_id, delegated_internal_token)
    )
    if has_internal_headers:
        if raw_identity is not None or has_sgw_identity:
            raise PermissionError('ambiguous_authentication')
        return await _authenticate_delegated_identity(
            session_factory=session_factory,
            config=config,
            internal_token=delegated_internal_token,
            app_id=delegated_app_id,
            principal_id=delegated_principal_id,
        )
    if raw_identity is not None:
        return await _authenticate_tai_identity(
            raw_identity,
            session_factory=session_factory,
            config=config,
        )
    if has_sgw_identity:
        return await _authenticate_tai_app(request, session_factory=session_factory, config=config)
    raise PermissionError('authentication_required')


async def _authenticate_tai_identity(
    raw_identity: str,
    *,
    session_factory: async_sessionmaker,
    config: DistributedConfig,
) -> AuthContext:
    try:
        application_token = config.read_tai_application_token()
    except ValueError as exc:
        raise PermissionError('tai_identity_unavailable') from exc
    identity = decrypt_tai_identity(
        raw_identity,
        application_token=application_token,
        issuer=config.tai_identity_issuer,
        tenant=config.tai_identity_tenant,
    )
    async with session_factory() as session:
        repository = AccessRepository(session)
        user = await repository.get_user(identity.principal_id)
        if user is None or (user.display_name is None and identity.display_name is not None):
            user = await repository.ensure_user(
                principal_id=identity.principal_id,
                issuer=identity.issuer,
                tenant=identity.tenant,
                subject=identity.subject,
                roles={'user'},
                display_name=identity.display_name,
                actor='system:tai-identity',
                now=utcnow(),
            )
        system_token = await repository.ensure_system_token(
            owner_principal_id=user.principal_id,
            created_by='system:tai-identity',
            now=utcnow(),
        )
        accessible_owner_principal_ids = await repository.list_accessible_owner_principal_ids(user.principal_id)
        await session.commit()
    return _personal_auth_context(
        user,
        system_token_id=system_token.token_id,
        display_name=user.display_name or identity.display_name,
        accessible_owner_principal_ids=accessible_owner_principal_ids,
    )


async def _authenticate_tai_app(
    request: web.Request,
    *,
    session_factory: async_sessionmaker,
    config: DistributedConfig,
) -> AuthContext:
    try:
        application_token = config.read_tai_application_token()
    except ValueError as exc:
        raise PermissionError('tai_app_identity_unavailable') from exc
    backend_signature_present = has_sgw_backend_signature_headers(request.headers)
    try:
        if config.tai_trust_x_rio_app_id and not backend_signature_present:
            app_id = trusted_sgw_app_id(request.headers)
            logger.warning('Accepted trusted x-rio app identity without backend signature: app_id=%s', app_id)
        else:
            app_id = verify_sgw_backend_signature(request.headers, application_token=application_token)
    except PermissionError:
        logger.warning('Rejected SGW app identity: auth_header_counts=%s', sgw_auth_header_counts(request.headers))
        raise
    async with session_factory() as session:
        authenticated = await AccessRepository(session).authenticate_app_id(app_id, now=utcnow())
    if authenticated is None:
        raise PermissionError('invalid_tai_app_identity')
    return _app_auth_context(authenticated, app_id=app_id)


async def _authenticate_delegated_identity(
    *,
    session_factory: async_sessionmaker,
    config: DistributedConfig,
    internal_token: str | None,
    app_id: str | None,
    principal_id: str | None,
) -> AuthContext:
    expected_internal_token = config.read_internal_token()
    if (
        not internal_token
        or not expected_internal_token
        or not hmac.compare_digest(internal_token, expected_internal_token)
        or bool(app_id) == bool(principal_id)
    ):
        raise PermissionError('invalid_delegated_identity')
    async with session_factory() as session:
        repository = AccessRepository(session)
        if app_id:
            authenticated = await repository.authenticate_app_id(app_id, now=utcnow())
            if authenticated is None:
                raise PermissionError('invalid_delegated_identity')
            return _app_auth_context(authenticated, app_id=app_id, internal_delegation=True)
        assert principal_id is not None
        user = await repository.get_user(principal_id)
        system_token = await repository.get_system_token(principal_id)
        accessible_owner_principal_ids = await repository.list_accessible_owner_principal_ids(principal_id)
    if user is None or system_token is None:
        raise PermissionError('invalid_delegated_identity')
    return _personal_auth_context(
        user,
        system_token_id=system_token.token_id,
        internal_delegation=True,
        accessible_owner_principal_ids=accessible_owner_principal_ids,
    )


def _personal_auth_context(
    user: AccessUser,
    *,
    system_token_id: str,
    display_name: str | None = None,
    internal_delegation: bool = False,
    accessible_owner_principal_ids: frozenset[str] = frozenset(),
) -> AuthContext:
    return AuthContext(
        principal_id=user.principal_id,
        issuer=user.issuer,
        tenant=user.tenant,
        subject=user.subject,
        roles=user.roles,
        scopes=frozenset({'*'}),
        auth_type='tai_identity',
        system_token_id=system_token_id,
        display_name=display_name or user.display_name,
        internal_delegation=internal_delegation,
        accessible_owner_principal_ids=accessible_owner_principal_ids,
    )


def _app_auth_context(
    authenticated: AuthenticatedAppToken,
    *,
    app_id: str,
    internal_delegation: bool = False,
) -> AuthContext:
    user = authenticated.user
    return AuthContext(
        principal_id=user.principal_id,
        issuer=user.issuer,
        tenant=user.tenant,
        subject=user.subject,
        roles=frozenset({'user'}),
        scopes=authenticated.scopes,
        auth_type='tai_app',
        system_token_id=(
            authenticated.token_id if authenticated.token_kind == SubmitTokenKind.SYSTEM_DEFAULT else None
        ),
        display_name=user.display_name,
        actor_app_id=app_id.strip().lower(),
        credential_token_id=authenticated.token_id,
        credential_environment_id=authenticated.environment_id,
        internal_delegation=internal_delegation,
    )


def csrf_token(context: AuthContext, config: DistributedConfig) -> str | None:
    if context.auth_type != 'tai_identity' or context.internal_delegation:
        return None
    try:
        key = config.read_tai_application_token()
    except ValueError as exc:
        raise RuntimeError('csrf_key_unavailable') from exc
    return hmac.new(key.encode(), f'nexus-csrf:v1\0{context.principal_id}'.encode(), hashlib.sha256).hexdigest()


def websocket_requested_protocols(request: web.Request) -> tuple[str, ...]:
    raw_protocols = request.headers.get('Sec-WebSocket-Protocol', '')
    return tuple(protocol.strip() for protocol in raw_protocols.split(',') if protocol.strip())


def websocket_application_protocols(request: web.Request) -> tuple[str, ...]:
    return tuple(
        protocol
        for protocol in websocket_requested_protocols(request)
        if not protocol.startswith(WEBSOCKET_CSRF_PROTOCOL_PREFIX)
    )


def _is_unauthenticated_route(request: web.Request) -> bool:
    if request.path == '/v1/health' or request.path.startswith('/v1/console'):
        return True
    if request.path == '/compat/live/health':
        return True
    if request.path.startswith('/v1/internal/'):
        return True
    return (
        request.method == 'PUT'
        and request.path.startswith('/v1/artifact-uploads/')
        and request.path.endswith('/content')
    )


def _enforce_browser_origin(
    request: web.Request,
    *,
    context: AuthContext,
    config: DistributedConfig,
) -> None:
    websocket = request.headers.get('Upgrade', '').lower() == 'websocket'
    if (
        context.auth_type != 'tai_identity'
        or context.internal_delegation
        or (request.method not in {'POST', 'PUT', 'PATCH', 'DELETE'} and not websocket)
    ):
        return
    fetch_site = request.headers.get('Sec-Fetch-Site', '').lower()
    if fetch_site == 'cross-site':
        raise web.HTTPForbidden(text='cross_site_request_rejected')
    origin = request.headers.get('Origin')
    if not origin:
        if websocket or request.headers.get('Sec-Fetch-Mode'):
            raise web.HTTPForbidden(text='browser_origin_required')
        return
    same_origin = f'{request.scheme}://{request.host}'
    if origin != same_origin and origin not in config.cors_origins:
        raise web.HTTPForbidden(text='origin_not_allowed')
    if websocket:
        csrf_protocols = [
            protocol
            for protocol in websocket_requested_protocols(request)
            if protocol.startswith(WEBSOCKET_CSRF_PROTOCOL_PREFIX)
        ]
        if not csrf_protocols:
            raise web.HTTPForbidden(text='csrf_token_required')
        if len(csrf_protocols) != 1:
            raise web.HTTPForbidden(text='invalid_csrf_token')
        supplied_token = csrf_protocols[0][len(WEBSOCKET_CSRF_PROTOCOL_PREFIX) :]
    else:
        supplied_token = request.headers.get(CSRF_HEADER)
    if not supplied_token:
        raise web.HTTPForbidden(text='csrf_token_required')
    expected_token = csrf_token(context, config)
    if expected_token is None or not hmac.compare_digest(supplied_token, expected_token):
        raise web.HTTPForbidden(text='invalid_csrf_token')
