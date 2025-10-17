"""Helpers for determining price precision per instrument."""

from __future__ import annotations

DEFAULT_PRECISION_MAP = {
    "ES": 2,
    "NQ": 2,
    "YM": 0,
    "RTY": 2,
    "CL": 3,
    "GC": 1,
    "BTC": 2,
    "ETH": 2,
    "SPX": 2,
    "NDX": 2,
    "DEFAULT": 2,
}


def _normalize_symbol(symbol: str | None) -> str:
    return (symbol or "").upper()


def get_price_precision(symbol: str | None, *, precision_map: dict[str, int] | None = None) -> int:
    """Return decimal precision for formatting price values."""

    sym = _normalize_symbol(symbol)
    mapping = precision_map or DEFAULT_PRECISION_MAP
    if sym in mapping:
        return int(mapping[sym])
    if sym.startswith("I:") and sym[2:] in mapping:
        return int(mapping[sym[2:]])
    if sym.endswith(".X") and sym[:-2] in mapping:
        return int(mapping[sym[:-2]])
    return int(mapping.get("DEFAULT", 2))


__all__ = ["get_price_precision", "DEFAULT_PRECISION_MAP"]
