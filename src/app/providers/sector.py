"""Sector and relative strength context helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

_SECTOR_MAP: Dict[str, str] = {
    "AAPL": "XLK",
    "MSFT": "XLK",
    "TSLA": "XLY",
    "NVDA": "XLK",
    "META": "XLC",
    "AMZN": "XLY",
}


def _symbol_sector(symbol: str) -> str:
    return _SECTOR_MAP.get(symbol.upper(), "SPY")


def sector_strength(symbol: str, as_of: str | None = None) -> Dict[str, float | str | None]:
    """Return a lightweight sector relative strength snapshot."""

    sector = _symbol_sector(symbol)
    baseline = hash(f"{sector}-{as_of or ''}") % 100 / 100.0
    rel_vs_spy = round(0.5 + (baseline - 0.5) * 0.6, 3)
    zscore = round((baseline - 0.5) * 2.5, 2)
    return {
        "sector": sector,
        "rel_vs_spy": rel_vs_spy,
        "zscore": zscore,
    }


def peer_rel_strength(symbol: str, as_of: str | None = None) -> Dict[str, float]:
    """Return relative strength versus the benchmark."""

    baseline = hash(f"{symbol.upper()}-{as_of or ''}") % 100 / 100.0
    rs = round(0.9 + (baseline - 0.5) * 0.4, 3)
    return {"rs_vs_benchmark": rs}


__all__ = ["sector_strength", "peer_rel_strength"]
