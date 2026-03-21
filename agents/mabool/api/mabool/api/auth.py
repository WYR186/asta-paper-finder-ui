"""
Optional API-key authentication.

Set ASTA_API_KEY in conf/.env.secret (or as an env var) to enable.
If the key is empty/unset, all requests pass through (local mode).

Usage in routes:
    @router.get("/my-route", dependencies=[Depends(require_auth)])
    ...

Or protect entire router by passing dependencies= to APIRouter().
"""
from __future__ import annotations

import os

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

_bearer_scheme = HTTPBearer(auto_error=False)


def require_auth(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> None:
    """FastAPI dependency: validates Bearer token if ASTA_API_KEY is set."""
    configured_key = os.environ.get("ASTA_API_KEY", "").strip()
    if not configured_key:
        # No key configured → open access (local mode)
        return

    if credentials is None or credentials.credentials != configured_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "Bearer"},
        )


def is_auth_enabled() -> bool:
    return bool(os.environ.get("ASTA_API_KEY", "").strip())
