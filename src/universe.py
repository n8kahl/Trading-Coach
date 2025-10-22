"""Universe expansion helpers for scan endpoints."""

from __future__ import annotations

import asyncio
import re
import time
from datetime import datetime
from typing import Iterable, List, Sequence

from .app.providers.universe import load_universe
from .config import get_settings

_UNIVERSE_TTL_SECONDS = 120.0
_UNIVERSE_CACHE: dict[tuple[str, str | None, int], tuple[float, List[str]]] = {}
_UNIVERSE_LOCK = asyncio.Lock()

_TOKEN_PATTERN = re.compile(r"FT-(TOPLIQUIDITY|PLAYBOOK|WATCHLIST)", re.IGNORECASE)
_TOP_K_PATTERN = re.compile(r"TOP(\d{1,3})$", re.IGNORECASE)
_SYMBOL_PATTERN = re.compile(r"^[A-Z][A-Z0-9\.:]{0,9}$")

_DEFAULT_PLAYBOOK = ["AAPL", "MSFT", "NVDA", "GOOG", "META", "TSLA", "AMZN"]
_DEFAULT_WATCHLIST: List[str] = []


def _normalize_symbols(symbols: Iterable[str]) -> List[str]:
    unique: dict[str, None] = {}
    for raw in symbols:
        token = (raw or "").strip().upper()
        if not token:
            continue
        if not _SYMBOL_PATTERN.fullmatch(token):
            continue
        if token not in unique:
            unique[token] = None
    return list(unique.keys())


def _settings_list(name: str, fallback: Sequence[str]) -> List[str]:
    settings = get_settings()
    value = getattr(settings, name, None)
    if value is None:
        return list(fallback)
    if isinstance(value, str):
        parts = [part.strip().upper() for part in value.split(",")]
        return _normalize_symbols(parts) or list(fallback)
    if isinstance(value, (list, tuple, set)):
        return _normalize_symbols(value) or list(fallback)
    return list(fallback)


async def _resolve_special_token(token: str, *, style: str | None, limit: int) -> List[str]:
    match = _TOKEN_PATTERN.fullmatch(token)
    if not match:
        return []
    key = match.group(1).upper()
    if key == "TOPLIQUIDITY":
        cache_key = ("topliquidity", style, limit)
        async with _UNIVERSE_LOCK:
            cached = _UNIVERSE_CACHE.get(cache_key)
            now = time.monotonic()
            if cached and now - cached[0] < _UNIVERSE_TTL_SECONDS:
                return list(cached[1])
        symbols = await load_universe(style=style, sector=None, limit=limit)
        normalized = _normalize_symbols(symbols)
        async with _UNIVERSE_LOCK:
            _UNIVERSE_CACHE[cache_key] = (time.monotonic(), list(normalized))
        return normalized[:limit]
    if key == "PLAYBOOK":
        return _normalize_symbols(_settings_list("playbook_symbols", _DEFAULT_PLAYBOOK))[:limit]
    if key == "WATCHLIST":
        return _normalize_symbols(_settings_list("watchlist_symbols", _DEFAULT_WATCHLIST))[:limit]
    return []


async def expand_universe(universe: str | List[str], *, style: str, limit: int) -> List[str]:
    """Expand the requested universe into a validated ticker list."""

    limit = max(1, min(limit, 250))

    if isinstance(universe, list):
        return _normalize_symbols(universe)[:limit]

    token = (universe or "").strip().upper()
    if not token:
        return []

    if token == "LAST_SNAPSHOT":
        symbols = await load_universe(style=style, sector=None, limit=limit)
        normalized = _normalize_symbols(symbols)
        if not normalized:
            normalized = _normalize_symbols(_settings_list("playbook_symbols", _DEFAULT_PLAYBOOK))
        if not normalized:
            normalized = _normalize_symbols(_DEFAULT_PLAYBOOK)
        return normalized[:limit]

    specials = await _resolve_special_token(token, style=style, limit=limit)
    if specials:
        symbols = _normalize_symbols(specials)[:limit]
        if style and style.lower() == "swing" and symbols:
            seed = int(datetime.utcnow().timestamp() // 86400)
            symbols.sort(key=lambda sym: (hash((sym, seed)) & 0xFFFFFFFF))
        return symbols

    top_match = _TOP_K_PATTERN.fullmatch(token)
    if top_match:
        try:
            requested = int(top_match.group(1))
        except (TypeError, ValueError):
            requested = limit
        effective_limit = max(1, min(requested, limit))
        symbols = await load_universe(style=style, sector=None, limit=effective_limit)
        return _normalize_symbols(symbols)[:effective_limit]

    if _SYMBOL_PATTERN.fullmatch(token):
        return _normalize_symbols([token])[:limit]

    return []


__all__ = ["expand_universe"]
