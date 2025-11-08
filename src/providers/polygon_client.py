"""Polygon snapshot client with retries/backoff."""

from __future__ import annotations

import asyncio
import random
from typing import Any, Dict, Optional

import httpx

from ..config import get_settings

POLYGON_BASE_URL = "https://api.massive.com"
_MAX_ATTEMPTS = 3
_BASE_DELAY = 0.35
_TIMEOUT = httpx.Timeout(timeout=8.0, connect=3.0)


async def fetch_option_snapshot(
    symbol: str,
    *,
    expiration: str | None = None,
    client: httpx.AsyncClient | None = None,
) -> Optional[Dict[str, Any]]:
    """Fetch Polygon option snapshot payload with bounded retries."""
    settings = get_settings()
    api_key = settings.polygon_api_key
    if not api_key:
        return None

    params: Dict[str, Any] = {"apiKey": api_key}
    if expiration:
        params["expiration_date"] = expiration

    url = f"{POLYGON_BASE_URL}/v3/snapshot/options/{symbol}"
    own_client = client is None
    session = client or httpx.AsyncClient(timeout=_TIMEOUT)
    try:
        for attempt in range(_MAX_ATTEMPTS):
            try:
                response = await session.get(url, params=params)
            except httpx.TimeoutException:
                await asyncio.sleep(_backoff(attempt))
                continue
            except httpx.HTTPError:
                await asyncio.sleep(_backoff(attempt))
                continue

            status = response.status_code
            if status == 200:
                return response.json()

            if status in {400, 401, 403, 404}:
                _log_status(symbol, status)
                return None

            if status in {429, 500, 502, 503}:
                _log_status(symbol, status)
                await asyncio.sleep(_backoff(attempt, status=status))
                continue

            _log_status(symbol, status)
            return None
        _log_status(symbol, 0)
        return None
    finally:
        if own_client:
            await session.aclose()


def _backoff(attempt: int, *, status: int | None = None) -> float:
    jitter = random.uniform(0.1, 0.3)
    scale = 1.0 if status not in {429} else 2.0
    return (_BASE_DELAY * (attempt + 1) * scale) + jitter


def _log_status(symbol: str, status: int) -> None:
    import logging

    logger = logging.getLogger(__name__)
    label = "timeout" if status == 0 else f"status={status}"
    logger.info("Polygon snapshot fallback for %s (%s)", symbol, label)


__all__ = ["fetch_option_snapshot"]
