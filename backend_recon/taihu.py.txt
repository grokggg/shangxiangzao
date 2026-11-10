from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import re

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from nexus.distributed.utils.identity import (
    IDENTITY_DISPLAY_NAME_MAX_LENGTH,
    IDENTITY_ISSUER_MAX_LENGTH,
    IDENTITY_SUBJECT_MAX_LENGTH,
    IDENTITY_TENANT_MAX_LENGTH,
)


TAI_IDENTITY_HEADER = 'x-tai-identity'
SGW_CALLER_APP_ID_HEADER = 'x-rio-paasid'
SGW_CALLER_AUTH_HEADERS = frozenset({
    SGW_CALLER_APP_ID_HEADER,
    'x-rio-nonce',
    'x-rio-signature',
    'x-rio-timestamp',
})
SGW_BACKEND_AUTH_HEADERS = frozenset({'timestamp', 'x-rio-seq', 'signature'})
SGW_AUTH_HEADERS = SGW_CALLER_AUTH_HEADERS | SGW_BACKEND_AUTH_HEADERS
_SIGNATURE_PATTERN = re.compile(r'^[0-9A-F]{64}$')
_MAX_CLOCK_SKEW_SECONDS = 180
_MAX_SEQUENCE_LENGTH = 256
_TAI_IDENTITY_EXPIRATION_LEEWAY = timedelta(minutes=3)


@dataclass(frozen=True, slots=True)
class TaihuIdentity:
    issuer: str
    tenant: str
    subject: str
    display_name: str | None = None

    @property
    def principal_id(self) -> str:
        return principal_id(self.issuer, self.tenant, self.subject)


def parse_tai_identity(raw: str, *, default_issuer: str = 'tai') -> TaihuIdentity:
    raw = raw.strip()
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        value = None
    if isinstance(value, dict):
        issuer = value.get('issuer') or default_issuer
        tenant = value.get('tenant') or value.get('tenant_id') or value.get('department') or 'default'
        subject = value.get('subject') or value.get('user_id') or value.get('username') or value.get('name')
        display_name = value.get('display_name') or value.get('displayName')
    else:
        parts = raw.split('/', 2)
        if len(parts) == 3:
            issuer, tenant, subject = parts
        else:
            issuer, tenant, subject = default_issuer, 'default', raw
        display_name = None
    return TaihuIdentity(
        issuer=_identity_component(issuer, 'issuer', IDENTITY_ISSUER_MAX_LENGTH, lowercase=True),
        tenant=_identity_component(tenant, 'tenant', IDENTITY_TENANT_MAX_LENGTH, lowercase=True),
        subject=_identity_component(subject, 'subject', IDENTITY_SUBJECT_MAX_LENGTH, lowercase=True),
        display_name=str(display_name).strip()[:IDENTITY_DISPLAY_NAME_MAX_LENGTH] if display_name else None,
    )


def decrypt_tai_identity(
    raw: str,
    *,
    application_token: str,
    issuer: str = 'tai',
    tenant: str = 'default',
    now: datetime | None = None,
) -> TaihuIdentity:
    key = application_token.encode('utf-8')
    if len(key) != 32:
        raise PermissionError('invalid_tai_application_token')
    parts = raw.strip().split('.')
    if len(parts) != 5 or parts[1]:
        raise PermissionError('invalid_tai_identity')
    protected_encoded, _, iv_encoded, ciphertext_encoded, tag_encoded = parts
    try:
        protected = json.loads(_decode_base64url(protected_encoded).decode('utf-8'))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise PermissionError('invalid_tai_identity') from exc
    if not isinstance(protected, dict) or protected.get('alg') != 'dir' or protected.get('enc') != 'A256GCM':
        raise PermissionError('invalid_tai_identity_algorithm')
    try:
        iv = _decode_base64url(iv_encoded)
        ciphertext = _decode_base64url(ciphertext_encoded)
        tag = _decode_base64url(tag_encoded)
    except ValueError as exc:
        raise PermissionError('invalid_tai_identity') from exc
    if len(iv) != 12 or len(tag) != 16:
        raise PermissionError('invalid_tai_identity')
    try:
        plaintext = AESGCM(key).decrypt(iv, ciphertext + tag, protected_encoded.encode('ascii'))
        payload = json.loads(plaintext.decode('utf-8'))
    except (InvalidTag, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PermissionError('invalid_tai_identity') from exc
    if not isinstance(payload, dict):
        raise PermissionError('invalid_tai_identity')

    expiration = _tai_identity_expiration(payload.get('Expiration'))
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        raise ValueError('now must be timezone-aware')
    if current > expiration + _TAI_IDENTITY_EXPIRATION_LEEWAY:
        raise PermissionError('expired_tai_identity')

    staff_id = payload.get('StaffId')
    login_name = payload.get('LoginName')
    if isinstance(staff_id, bool) or not isinstance(staff_id, int) or staff_id < 1:
        raise PermissionError('invalid_tai_identity_staff_id')
    if not isinstance(login_name, str):
        raise PermissionError('invalid_tai_identity_login_name')
    return TaihuIdentity(
        issuer=_identity_component(issuer, 'issuer', IDENTITY_ISSUER_MAX_LENGTH, lowercase=True),
        tenant=_identity_component(tenant, 'tenant', IDENTITY_TENANT_MAX_LENGTH, lowercase=True),
        subject=_identity_component(str(staff_id), 'subject', IDENTITY_SUBJECT_MAX_LENGTH, lowercase=True),
        display_name=_identity_component(
            login_name,
            'login_name',
            IDENTITY_DISPLAY_NAME_MAX_LENGTH,
            lowercase=False,
        ),
    )


def principal_id(issuer: str, tenant: str, subject: str) -> str:
    return hashlib.sha256(f'{issuer}\0{tenant}\0{subject}'.encode()).hexdigest()


def verify_sgw_backend_signature(
    headers: Mapping[str, str],
    *,
    application_token: str,
    now: datetime | None = None,
) -> str:
    """Verify the signature SGW adds for a called-party backend."""

    app_id = _required_header(headers, SGW_CALLER_APP_ID_HEADER)
    timestamp = _required_header(headers, 'timestamp')
    sequence = _required_header(headers, 'x-rio-seq')
    signature = _required_header(headers, 'signature')
    if not application_token or len(timestamp) > 20 or not timestamp.isascii() or not timestamp.isdecimal():
        raise PermissionError('invalid_tai_app_identity')
    if (
        len(sequence) > _MAX_SEQUENCE_LENGTH
        or any(ord(character) < 0x20 or ord(character) == 0x7F for character in sequence)
        or _SIGNATURE_PATTERN.fullmatch(signature) is None
    ):
        raise PermissionError('invalid_tai_app_identity')

    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        raise ValueError('now must be timezone-aware')
    signed_at = int(timestamp)
    if abs(int(current.timestamp()) - signed_at) > _MAX_CLOCK_SKEW_SECONDS:
        raise PermissionError('invalid_tai_app_identity')

    expected = hashlib.sha256(f'{timestamp}{application_token}{sequence}{timestamp}'.encode()).hexdigest().upper()
    if not hmac.compare_digest(signature, expected):
        raise PermissionError('invalid_tai_app_identity')
    return app_id


def has_sgw_auth_headers(headers: Mapping[str, str]) -> bool:
    return any(_header_values(headers, name) for name in SGW_AUTH_HEADERS)


def has_sgw_backend_signature_headers(headers: Mapping[str, str]) -> bool:
    return any(_header_values(headers, name) for name in ('timestamp', 'signature'))


def trusted_sgw_app_id(headers: Mapping[str, str]) -> str:
    """Read an app id from a deployment-trusted SGW forwarding boundary."""

    return _required_header(headers, SGW_CALLER_APP_ID_HEADER)


def sgw_auth_header_counts(headers: Mapping[str, str]) -> dict[str, int]:
    """Return structural SGW auth diagnostics without exposing header values."""

    return {name: len(_header_values(headers, name)) for name in sorted(SGW_AUTH_HEADERS)}


def _identity_component(value: Any, name: str, limit: int, *, lowercase: bool) -> str:
    normalized = str(value or '').strip()
    if lowercase:
        normalized = normalized.lower()
    if not normalized or len(normalized) > limit or re.search(r'[\x00-\x1f\x7f]', normalized):
        raise PermissionError(f'invalid_identity_{name}')
    return normalized


def _decode_base64url(value: str) -> bytes:
    try:
        encoded = value.encode('ascii')
        decoded = base64.b64decode(encoded + b'=' * (-len(encoded) % 4), altchars=b'-_', validate=True)
        if base64.urlsafe_b64encode(decoded).rstrip(b'=') != encoded:
            raise ValueError('non-canonical base64url value')
        return decoded
    except (UnicodeEncodeError, binascii.Error, ValueError) as exc:
        raise ValueError('invalid base64url value') from exc


def _tai_identity_expiration(value: Any) -> datetime:
    if not isinstance(value, str) or not value:
        raise PermissionError('invalid_tai_identity_expiration')
    try:
        expiration = datetime.fromisoformat(value.replace('Z', '+00:00'))
    except ValueError as exc:
        raise PermissionError('invalid_tai_identity_expiration') from exc
    if expiration.tzinfo is None:
        raise PermissionError('invalid_tai_identity_expiration')
    return expiration.astimezone(timezone.utc)


def _required_header(headers: Mapping[str, str], name: str) -> str:
    values = _header_values(headers, name)
    if len(values) != 1:
        raise PermissionError('invalid_tai_app_identity')
    value = values[0].strip()
    if not value:
        raise PermissionError('invalid_tai_app_identity')
    return value


def _header_values(headers: Mapping[str, str], name: str) -> tuple[str, ...]:
    getall = getattr(headers, 'getall', None)
    if callable(getall):
        try:
            values: Any = getall(name, ())
        except KeyError:
            values = ()
        return tuple(str(value) for value in values)
    value = headers.get(name)
    return () if value is None else (str(value),)


__all__ = [
    'SGW_AUTH_HEADERS',
    'SGW_BACKEND_AUTH_HEADERS',
    'SGW_CALLER_APP_ID_HEADER',
    'SGW_CALLER_AUTH_HEADERS',
    'TAI_IDENTITY_HEADER',
    'TaihuIdentity',
    'decrypt_tai_identity',
    'has_sgw_auth_headers',
    'has_sgw_backend_signature_headers',
    'parse_tai_identity',
    'principal_id',
    'sgw_auth_header_counts',
    'trusted_sgw_app_id',
    'verify_sgw_backend_signature',
]
