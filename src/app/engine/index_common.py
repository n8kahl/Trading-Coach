"""Shared constants and helpers for index-first planning."""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

INDEX_BASE_TICKERS: Tuple[str, ...] = ("SPX", "NDX")
POLYGON_INDEX_TICKERS: Dict[str, str] = {base: f"I:{base}" for base in INDEX_BASE_TICKERS}
ETF_PROXIES: Dict[str, str] = {"SPX": "SPY", "NDX": "QQQ"}
CONTRACT_PREF_ORDER: Tuple[str, ...] = ("INDEX_POLYGON", "INDEX_TRADIER", "ETF_PROXY")


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


__all__ = [
    "INDEX_BASE_TICKERS",
    "POLYGON_INDEX_TICKERS",
    "ETF_PROXIES",
    "CONTRACT_PREF_ORDER",
    "resolve_polygon_symbol",
    "resolve_proxy_symbol",
    "base_index_symbols",
]
