"""Market internals adapter."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple
import time

import httpx

from src.config import get_settings, get_massive_api_key

_POLYGON_BASE = "https://api.massive.com"
_CACHE_TTL = 30.0
_INTERNALS_CACHE: Optional[Tuple[float, Dict[str, int | float]]] = None


def _polygon_request(path: str, params: Dict[str, str], api_key: str) -> Optional[Dict[str, Any]]:  # type: ignore[name-defined]
    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.get(f"{_POLYGON_BASE}{path}", params=params, headers={"Authorization": f"Bearer {api_key}"})
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPError:
        return None


def _latest_vix(api_key: str) -> Optional[float]:
    data = _polygon_request("/v2/aggs/ticker/CBOE:VIX/prev", {"adjusted": "true"}, api_key)
    results = (data or {}).get("results") or []
    if not results:
        return None
    try:
        return float(results[0].get("c"))
    except (TypeError, ValueError):
        return None


def _breadth_counts(api_key: str) -> Tuple[int, int]:
    adv = _polygon_request("/v2/snapshot/locale/us/markets/stocks/advancers", {"limit": "1000"}, api_key)
    dec = _polygon_request("/v2/snapshot/locale/us/markets/stocks/decliners", {"limit": "1000"}, api_key)
    adv_count = len((adv or {}).get("results") or [])
    dec_count = len((dec or {}).get("results") or [])
    return adv_count, dec_count


def market_internals(as_of: str | None = None) -> Dict[str, int | float]:
    """Return a compact snapshot of index internals."""

    global _INTERNALS_CACHE
    now = time.monotonic()
    if _INTERNALS_CACHE and now - _INTERNALS_CACHE[0] < _CACHE_TTL:
        return dict(_INTERNALS_CACHE[1])

    settings = get_settings()
    api_key = get_massive_api_key(settings)
    if not api_key:
        fallback = {"breadth": 0, "vix": None, "tick": 0}
        _INTERNALS_CACHE = (now, fallback)
        return fallback

    advancers, decliners = _breadth_counts(api_key)
    breadth = advancers - decliners
    vix_value = _latest_vix(api_key)
    tick_value = max(-2000, min(2000, breadth * 5))
    snapshot = {
        "breadth": breadth,
        "vix": vix_value if vix_value is not None else 0.0,
        "tick": tick_value,
    }
    _INTERNALS_CACHE = (now, snapshot)
    return snapshot


__all__ = ["market_internals"]
