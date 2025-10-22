from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Literal

DataMode = Literal["live", "lkg"]


@dataclass(slots=True)
class SeriesBundle:
    symbols: List[str]
    mode: DataMode
    as_of: datetime
    bars: Dict[str, List[Dict[str, float]]] = field(default_factory=dict)


async def fetch_series(symbols: List[str], *, mode: DataMode, as_of: datetime) -> SeriesBundle:
    """Return a synthetic OHLCV series bundle suitable for unit tests."""
    bundle = SeriesBundle(symbols=list(symbols), mode=mode, as_of=as_of)
    bundle.bars = {symbol: [] for symbol in symbols}
    return bundle
