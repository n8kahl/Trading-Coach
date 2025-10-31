"""Shared constants and helpers for index-first planning."""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import os

INDEX_BASE_TICKERS: Tuple[str, ...] = ("SPX", "NDX")
POLYGON_INDEX_TICKERS: Dict[str, str] = {base: f"I:{base}" for base in INDEX_BASE_TICKERS}
ETF_PROXIES: Dict[str, str] = {"SPX": "SPY", "NDX": "QQQ"}
CONTRACT_PREF_ORDER: Tuple[str, ...] = ("INDEX_POLYGON", "INDEX_TRADIER", "ETF_PROXY")


def _ratio_from_env(symbol: str, default: float) -> float:
    env_key = f"{symbol.upper()}_PROXY_RATIO"
    raw = os.getenv(env_key)
    if raw is None:
        return default
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


DEFAULT_INDEX_RATIOS: Dict[str, float] = {
    "SPX": _ratio_from_env("SPX", 10.0),
    "NDX": _ratio_from_env("NDX", 40.0),
}

INDEX_SCALE_THRESHOLDS: Dict[str, float] = {"SPX": 1000.0, "NDX": 5000.0}


def resolve_polygon_symbol(symbol: str) -> str | None:
    """Return the Polygon index snapshot symbol (I:SPX) for a base index ticker."""
    return POLYGON_INDEX_TICKERS.get(symbol.upper())


def resolve_proxy_symbol(symbol: str) -> str | None:
    """Return the ETF liquidity proxy for a base index ticker."""
    return ETF_PROXIES.get(symbol.upper())


def base_index_symbols(*extra: Iterable[str]) -> List[str]:
    """Utility returning canonical base tickers plus any override extras."""
    symbols = list(INDEX_BASE_TICKERS)
    for value in extra:
        for token in value:
            upper = token.upper()
            if upper not in symbols:
                symbols.append(upper)
    return symbols


def is_index_symbol(symbol: str) -> bool:
    """Return True when the ticker is one of the supported cash indexes."""
    return symbol.upper() in INDEX_BASE_TICKERS


def resolve_polygon_index_symbol(symbol: str) -> str | None:
    """Alias exposing the Polygon index ticker mapping."""
    return resolve_polygon_symbol(symbol)


def scale_threshold(symbol: str) -> float | None:
    """Return the minimum acceptable close for scale validation."""
    return INDEX_SCALE_THRESHOLDS.get(symbol.upper())


__all__ = [
    "INDEX_BASE_TICKERS",
    "POLYGON_INDEX_TICKERS",
    "ETF_PROXIES",
    "CONTRACT_PREF_ORDER",
    "DEFAULT_INDEX_RATIOS",
    "INDEX_SCALE_THRESHOLDS",
    "resolve_polygon_symbol",
    "resolve_polygon_index_symbol",
    "resolve_proxy_symbol",
    "is_index_symbol",
    "scale_threshold",
    "base_index_symbols",
]
