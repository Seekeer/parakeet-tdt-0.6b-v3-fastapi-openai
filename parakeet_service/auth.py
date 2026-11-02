"""Bearer API-key authentication for the transcription endpoints."""
from __future__ import annotations

import hashlib
import hmac
from typing import Iterable, Optional, Tuple

from fastapi import Depends, HTTPException
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from . import config

PUBLIC_PATHS = frozenset(
    {"/health", "/docs", "/docs/oauth2-redirect", "/redoc", "/openapi.json"}
)
_UNAUTHORIZED_DETAIL = "Invalid or missing API key"
_CHALLENGE = {"WWW-Authenticate": "Bearer"}

_bearer = HTTPBearer(
    auto_error=False,
    description="API key from PARAKEET_API_KEYS / PARAKEET_API_KEYS_FILE",
)


def is_valid_api_key(token: str) -> bool:
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    matched = False
    # Compare against every key so timing does not reveal which one matched.
    for known in config.API_KEY_DIGESTS:
        matched |= hmac.compare_digest(digest, known)
    return matched


def _bearer_token(headers: Iterable[Tuple[bytes, bytes]]) -> Optional[str]:
    for name, value in headers:
        if name.lower() == b"authorization":
            scheme, _, token = value.decode("latin-1").partition(" ")
            token = token.strip()
            if scheme.lower() == "bearer" and token:
                return token
            return None
    return None


def _route_path(scope) -> str:
    # Behind `uvicorn --root-path /prefix` the ASGI path still carries the prefix.
    path = scope["path"]
    root_path = scope.get("root_path", "")
    if root_path and path.startswith(root_path):
        return path[len(root_path):] or "/"
    return path


class ApiKeyMiddleware:
    """Reject unauthenticated requests before the request body is read.

    FastAPI parses multipart forms before resolving route dependencies, so
    checking the key only in a dependency would let anonymous clients upload
    arbitrarily large bodies first.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if (
            scope["type"] != "http"
            or config.AUTH_DISABLED
            or _route_path(scope) in PUBLIC_PATHS
        ):
            await self.app(scope, receive, send)
            return
        token = _bearer_token(scope["headers"])
        if token is None or not is_valid_api_key(token):
            response = JSONResponse(
                {"detail": _UNAUTHORIZED_DETAIL}, status_code=401, headers=_CHALLENGE
            )
            await response(scope, receive, send)
            return
        await self.app(scope, receive, send)


async def require_api_key(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
) -> None:
    """Route-level check; also publishes the Bearer scheme in OpenAPI."""
    if config.AUTH_DISABLED:
        return
    if credentials is None or not is_valid_api_key(credentials.credentials):
        raise HTTPException(
            status_code=401, detail=_UNAUTHORIZED_DETAIL, headers=_CHALLENGE
        )
