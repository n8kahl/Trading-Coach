"""Middleware that attaches the current market session to each request."""

from __future__ import annotations

from typing import Any, Dict

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from src.app.services import session_now

_FALLBACK_SESSION = {
    "status": "closed",
    "as_of": "",
    "next_open": "",
    "tz": "America/New_York",
    "banner": "Session unavailable",
}


class SessionMiddleware(BaseHTTPMiddleware):
    """Ensure every request has a session snapshot and surface it via headers."""

    async def dispatch(self, request: Request, call_next) -> Response:
        session_payload: Dict[str, Any]
        try:
            session_payload = session_now().to_dict()
        except Exception:  # pragma: no cover - defensive
            session_payload = dict(_FALLBACK_SESSION)

        request.state.session = dict(session_payload)

        response = await call_next(request)

        try:
            response.headers["X-Session-Status"] = str(session_payload.get("status", ""))
            response.headers["X-Session-As-Of"] = str(session_payload.get("as_of", ""))
        except Exception:  # pragma: no cover - defensive
            pass

        return response
