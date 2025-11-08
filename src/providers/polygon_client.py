"""Polygon snapshot client with retries/backoff."""

from __future__ import annotations

import asyncio
import os
import random
from typing import Any, Dict, Optional

import httpx

from ..config import get_settings, get_massive_api_key

_MASSIVE_BASE = os.getenv("MARKETDATA_BASE_URL", "https://api.massive.com").rstrip("/")
_POLYGON_BASE = os.getenv("POLYGON_BASE_URL", "https://api.polygon.io").rstrip("/")
_BASE_URL = _MASSIVE_BASE or _POLYGON_BASE
_MAX_ATTEMPTS = 3
_BASE_DELAY = 0.35
_TIMEOUT = httpx.Timeout(timeout=8.0, connect=3.0)


def _normalize_underlying(symbol: str) -> str:
    token = (symbol or "").strip().upper()
    if not token:
        return token
    if token in {"SPX", "NDX", "DJX", "VIX"} and not token.startswith("I:"):
        return f"I:{token}"
    return token


async def fetch_option_snapshot(
    symbol: str,
    *,
    expiration: str | None = None,
    client: httpx.AsyncClient | None = None,
) -> Optional[Dict[str, Any]]:
    """Fetch Massive option snapshot payload with bounded retries and pagination."""
    settings = get_settings()
    api_key = get_massive_api_key(settings)
    if not api_key:
        return None

    params: Dict[str, Any] = {"limit": 250}
    if expiration:
        params["expiration_date"] = expiration

    url = f"{_BASE_URL}/v3/snapshot/options/{_normalize_underlying(symbol)}"
    headers = {"Authorization": f"Bearer {api_key}"}
    results: list[Any] = []
    own_client = client is None
    session = client or httpx.AsyncClient(timeout=_TIMEOUT)
    try:
        next_url = url
        query: Dict[str, Any] | None = dict(params)
        while next_url:
            attempt = 0
            while attempt < _MAX_ATTEMPTS:
                attempt += 1
                try:
                    response = await session.get(next_url, params=query, headers=headers)
                    response.raise_for_status()
                    break
                except httpx.TimeoutException:
                    await asyncio.sleep(_backoff(attempt))
                except httpx.HTTPStatusError as exc:
                    status = exc.response.status_code
                    if status in {400, 401, 403, 404}:
                        _log_status(symbol, status)
                        return None
                    if status in {429, 500, 502, 503}:
                        _log_status(symbol, status)
                        await asyncio.sleep(_backoff(attempt, status=status))
                        continue
                    _log_status(symbol, status)
                    return None
                except httpx.HTTPError:
                    await asyncio.sleep(_backoff(attempt))
            else:
                _log_status(symbol, 0)
                return None

            payload = response.json() or {}
            results.extend(payload.get("results") or [])
            raw_next = payload.get("next_url")
            if raw_next:
                next_url = raw_next if str(raw_next).startswith("http") else f"{_BASE_URL}{raw_next}"
                query = None
            else:
                next_url = None
        return {"results": results}
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
