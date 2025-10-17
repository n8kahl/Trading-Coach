"""Session router exposing the market session SSOT."""

from __future__ import annotations

from fastapi import APIRouter, Request

from src.app.middleware import get_session

router = APIRouter(prefix="/api/v1", tags=["session"])


@router.get("/session")
def session_snapshot(request: Request) -> dict:
    """Return the current market session snapshot."""
    return get_session(request)
