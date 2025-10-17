"""Middleware that attaches the current market session to each request."""

from __future__ import annotations

from typing import Any, Dict, Mapping

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from src.app.services import session_now

_FALLBACK_SESSION: Dict[str, str] = {
    "status": "closed",
    "as_of": "",
    "next_open": "",
    "tz": "America/New_York",
    "banner": "Session unavailable",
}
_SESSION_KEYS = ("status", "as_of", "next_open", "tz", "banner")


def _normalized_session(payload: Mapping[str, Any] | None) -> Dict[str, str]:
    session = dict(_FALLBACK_SESSION)
    if payload:
        for key in _SESSION_KEYS:
            value = payload.get(key)
            if value is None:
                continue
            session[key] = str(value)
    return session


def get_session(request: Request | None = None) -> Dict[str, str]:
    """Return the current session snapshot, caching it on the request state."""

    if request is not None:
        cached = getattr(request.state, "session", None)
        if isinstance(cached, dict) and cached:
            return _normalized_session(cached)

    try:
        snapshot = session_now().to_dict()
    except Exception:  # pragma: no cover - defensive
        snapshot = dict(_FALLBACK_SESSION)

    normalized = _normalized_session(snapshot)
    if request is not None:
        request.state.session = dict(normalized)
    return normalized


class SessionMiddleware(BaseHTTPMiddleware):
    """Ensure every request has a session snapshot and surface it via headers."""

    async def dispatch(self, request: Request, call_next) -> Response:
        session_payload = get_session(request)

        response = await call_next(request)

        try:
            response.headers["X-Session-Status"] = session_payload.get("status", "")
            response.headers["X-Session-As-Of"] = session_payload.get("as_of", "")
            response.headers["X-Session-Next-Open"] = session_payload.get("next_open", "")
            response.headers["X-Session-Tz"] = session_payload.get("tz", "")
        except Exception:  # pragma: no cover - defensive
            pass

        return response


__all__ = ["SessionMiddleware", "get_session"]
