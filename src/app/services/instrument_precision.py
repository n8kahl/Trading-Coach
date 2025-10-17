"""Instrument-specific precision helpers used for canonical formatting."""

from __future__ import annotations

from typing import Dict

DEFAULT_PRECISION_MAP: Dict[str, int] = {
    # Equity index futures / proxies
    "ES": 2,
    "NQ": 2,
    "YM": 0,
    "RTY": 2,
    # Commodities
    "CL": 3,
    "GC": 1,
    # Crypto majors
    "BTC": 2,
    "ETH": 2,
    # Cash index proxies
    "SPX": 2,
    "NDX": 2,
    "RUT": 2,
    # Default fallback
    "DEFAULT": 2,
}


def _normalize_symbol(symbol: str | None) -> str:
    return (symbol or "").upper()


def get_precision(symbol: str | None, *, precision_map: dict[str, int] | None = None) -> int:
    """Return decimal precision for formatting numeric fields."""

    mapping = precision_map or DEFAULT_PRECISION_MAP
    normalized = _normalize_symbol(symbol)

    if normalized in mapping:
        return int(mapping[normalized])
    if normalized.startswith("I:") and normalized[2:] in mapping:
        return int(mapping[normalized[2:]])
    if normalized.endswith(".X") and normalized[:-2] in mapping:
        return int(mapping[normalized[:-2]])

    return int(mapping.get("DEFAULT", 2))


__all__ = ["get_precision", "DEFAULT_PRECISION_MAP"]
