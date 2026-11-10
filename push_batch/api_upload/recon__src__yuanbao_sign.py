# -*- coding: utf-8 -*-
"""Yuanbao Open API V2 signing helpers (see SIGNING.md)."""

from __future__ import annotations

import hashlib
import hmac
import uuid
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional


def build_sign_query(method: str, path: str, raw_query: str = "") -> str:
    if method.upper() == "GET" and raw_query:
        return f"{path}?{raw_query}"
    return path


def build_beijing_timestamp_z(now_utc: Optional[datetime] = None) -> str:
    """Match Go: time.Now().UTC().Add(8h).Format(RFC3339) → wall clock +8, suffix Z."""
    if now_utc is None:
        now_utc = datetime.now(timezone.utc)
    elif now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=timezone.utc)
    else:
        now_utc = now_utc.astimezone(timezone.utc)
    beijing_wall = now_utc + timedelta(hours=8)
    return beijing_wall.strftime("%Y-%m-%dT%H:%M:%SZ")


def build_string_to_sign(
    sign_query: str, nonce: str, timestamp: str, app_key: str, secret: str
) -> str:
    return "v2" + sign_query + nonce + timestamp + app_key + secret


def compute_signature(secret: str, string_to_sign: str) -> str:
    return hmac.new(
        secret.encode("utf-8"),
        string_to_sign.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def build_sign_headers(
    method: str,
    path: str,
    raw_query: str,
    app_key: str,
    secret: str,
    *,
    nonce: Optional[str] = None,
    timestamp: Optional[str] = None,
) -> Dict[str, str]:
    nonce = nonce or str(uuid.uuid4())
    timestamp = timestamp or build_beijing_timestamp_z()
    sign_query = build_sign_query(method, path, raw_query)
    string_to_sign = build_string_to_sign(
        sign_query, nonce, timestamp, app_key, secret
    )
    signature = compute_signature(secret, string_to_sign)
    return {
        "X-AppKey": app_key,
        "X-Nonce": nonce,
        "X-Timestamp": timestamp,
        "X-Signature": signature,
        "X-Sign-Version": "v2",
    }
