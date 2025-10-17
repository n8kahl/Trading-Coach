"""Session router exposing the market session SSOT."""

from __future__ import annotations

from fastapi import APIRouter

from src.app.services import session_now

router = APIRouter(prefix="/api/v1", tags=["session"])


@router.get("/session")
def get_session() -> dict:
    """Return the current market session snapshot."""
    return session_now().to_dict()
