"""Index planning utilities for SPX/NDX support with ETF fallbacks."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

from ...polygon_options import fetch_polygon_option_chain
from ...tradier import TradierNotConfiguredError, fetch_option_chain_cached
from .index_common import (
    CONTRACT_PREF_ORDER,
    ETF_PROXIES,
    POLYGON_INDEX_TICKERS,
    resolve_polygon_symbol,
    resolve_proxy_symbol,
)
from .index_health import FeedStatus, polygon_index_snapshot, polygon_universal_snapshot, tradier_index_greeks
from .ratio_engine import RatioEngine, RatioSnapshot

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class IndexDataHealth:
    polygon: Dict[str, FeedStatus]
    tradier: Dict[str, FeedStatus]
    universal: Optional[FeedStatus] = None
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "polygon": {symbol: status.to_dict() for symbol, status in self.polygon.items()},
            "tradier": {symbol: status.to_dict() for symbol, status in self.tradier.items()},
            "universal": self.universal.to_dict() if self.universal else None,
            "notes": list(self.notes),
        }


GammaSnapshot = RatioSnapshot  # Backwards-compatible alias


class IndexPlanner:
    """Provides index-first planning helpers with ETF fallbacks."""

    def __init__(self, *, ratio_engine: RatioEngine | None = None) -> None:
        self._ratio_engine = ratio_engine or RatioEngine()
        self.contract_preference = CONTRACT_PREF_ORDER

    @staticmethod
    def supports(symbol: str) -> bool:
        return symbol.upper() in POLYGON_INDEX_TICKERS

    async def polygon_index_chain(self, symbol: str) -> pd.DataFrame:
        polygon_symbol = resolve_polygon_symbol(symbol)
        if not polygon_symbol:
            raise ValueError(f"Unsupported index symbol {symbol}")
        return await fetch_polygon_option_chain(polygon_symbol)

    async def tradier_index_chain(self, symbol: str) -> pd.DataFrame:
        try:
            return await fetch_option_chain_cached(symbol)
        except TradierNotConfiguredError:
            logger.info("Tradier not configured for index chain %s", symbol)
            return pd.DataFrame()
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Tradier index chain fetch failed for %s: %s", symbol, exc)
            return pd.DataFrame()

    async def ratio_snapshot(self, index_symbol: str) -> Optional[RatioSnapshot]:
        return await self._ratio_engine.snapshot(index_symbol)

    async def gamma_snapshot(self, index_symbol: str) -> Optional[GammaSnapshot]:
        return await self.ratio_snapshot(index_symbol)

    def translate_level(self, index_price: float, snapshot: RatioSnapshot | None) -> float:
        return self._ratio_engine.translate_level(index_price, snapshot)

    def translate_targets(self, targets: List[float], snapshot: RatioSnapshot | None) -> List[float]:
        return [self.translate_level(target, snapshot) for target in targets]

    def add_execution_note(self, plan: Dict[str, Any], snapshot: RatioSnapshot | None, fallback_source: str) -> None:
        note = plan.setdefault("execution_notes", [])
        if snapshot:
            note.append(
                {
                    "source": fallback_source,
                    "gamma": round(snapshot.gamma_current, 6),
                    "gamma_mean": round(snapshot.gamma_mean, 6),
                    "gamma_drift": round(snapshot.drift, 6),
                    "ratio": round(snapshot.spot_ratio, 6),
                    "samples": snapshot.samples,
                    "proxy": snapshot.proxy_symbol,
                }
            )
        else:
            note.append({"source": fallback_source, "gamma": None})

    async def feed_health(self, symbol: str, *, expiration: str | None = None) -> IndexDataHealth:
        base = symbol.upper()
        polygon_payload, polygon_status = await polygon_index_snapshot(base)
        tradier_payload, tradier_status = await tradier_index_greeks(base, expiration)
        universal_payload, universal_status = await polygon_universal_snapshot()

        notes: List[str] = []
        if polygon_status.status != "healthy":
            notes.append(f"Polygon index feed {base} marked {polygon_status.status}")
        if tradier_status.status != "healthy":
            notes.append(f"Tradier index feed {base} marked {tradier_status.status}")
        if universal_status.status != "healthy":
            notes.append(f"Polygon universal snapshot {universal_status.status}")
        if polygon_payload and tradier_payload:
            notes.append("Both index feeds responded")

        return IndexDataHealth(
            polygon={base: polygon_status},
            tradier={base: tradier_status},
            universal=universal_status,
            notes=notes,
        )

    def proxy_symbol(self, symbol: str) -> Optional[str]:
        return resolve_proxy_symbol(symbol)

    async def synthetic_index_ohlcv(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Construct an index OHLCV frame from the ETF proxy when direct data is unavailable."""
        proxy_symbol = resolve_proxy_symbol(symbol)
        if not proxy_symbol:
            return None
        proxy_frame = await fetch_polygon_ohlcv(proxy_symbol, timeframe)
        if proxy_frame is None or proxy_frame.empty:
            return None
        snapshot = await self.ratio_snapshot(symbol)
        if snapshot is None:
            return None
        converted = proxy_frame.copy()
        for column in ("open", "high", "low", "close"):
            if column in converted.columns:
                converted[column] = converted[column].apply(snapshot.translate_from_proxy)
        if "volume" in converted.columns:
            try:
                converted["volume"] = converted["volume"].astype(float)
            except Exception:  # pragma: no cover - defensive
                converted["volume"] = 0.0
        else:
            converted["volume"] = 0.0
        converted.attrs["source"] = "proxy_gamma"
        return converted


__all__ = [
    "IndexPlanner",
    "IndexDataHealth",
    "GammaSnapshot",
]
