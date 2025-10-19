"""Universe provider used by planning-mode scans."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from ..config import get_settings
from .. import universe as legacy_universe
from .polygon_client import PolygonAggregatesClient

logger = logging.getLogger(__name__)


def _normalize(symbols: Iterable[str]) -> List[str]:
    items: List[str] = []
    seen: set[str] = set()
    for raw in symbols:
        token = (raw or "").strip().upper()
        if not token or token in seen:
            continue
        if not token[0].isalpha():
            continue
        seen.add(token)
        items.append(token)
    return items


def _settings_list(name: str) -> List[str]:
    settings = get_settings()
    value = getattr(settings, name, None)
    if not value:
        return []
    if isinstance(value, str):
        tokens = [part.strip() for part in value.split(",")]
    elif isinstance(value, (list, tuple, set)):
        tokens = list(value)
    else:
        return []
    return _normalize(tokens)


@dataclass
class UniverseSnapshot:
    name: str
    source: str
    as_of_utc: datetime
    symbols: List[str]
    metadata: Dict[str, object]


class UniverseProvider:
    """Fetch and cache symbol universes for planning mode."""

    _CACHE_TTL_SECONDS = 12 * 60 * 60  # 12 hours

    def __init__(
        self,
        polygon_client: PolygonAggregatesClient,
        *,
        persistence: Optional["PlanningPersistence"] = None,
        cache_ttl_seconds: Optional[int] = None,
    ) -> None:
        self._polygon = polygon_client
        self._persist = persistence
        self._cache_ttl = cache_ttl_seconds or self._CACHE_TTL_SECONDS
        self._cache: Dict[Tuple[str, int], Tuple[float, UniverseSnapshot]] = {}
        self._lock = asyncio.Lock()

    async def get_universe(self, universe: str, *, style: str, limit: int) -> UniverseSnapshot:
        key = (universe.lower(), max(1, min(limit, 1000)))
        async with self._lock:
            cached = self._cache.get(key)
            if cached and time.monotonic() - cached[0] <= self._cache_ttl:
                return cached[1]

        snapshot = await self._load_universe(universe, style=style, limit=limit)
        async with self._lock:
            self._cache[key] = (time.monotonic(), snapshot)
        return snapshot

    async def _load_universe(self, universe: str, *, style: str, limit: int) -> UniverseSnapshot:
        name = universe.strip().lower()
        limit = max(1, min(limit, 500))
        as_of = datetime.now(timezone.utc)
        source = "polygon"
        metadata: Dict[str, object] = {"style": style}
        symbols: List[str] = []

        if name in {"sp500", "s&p500", "spx"}:
            symbols = await self._polygon.fetch_index_constituents("I:SPX")
            metadata["index"] = "I:SPX"
        elif name in {"nasdaq100", "ndx", "qqq"}:
            symbols = await self._polygon.fetch_index_constituents("I:NDX")
            metadata["index"] = "I:NDX"
        elif name in {"watchlist", "custom"}:
            watchlist = _settings_list("planning_watchlist_symbols") or _settings_list("watchlist_symbols")
            symbols = watchlist
            source = "settings"
        elif name in {"playbook"}:
            symbols = _settings_list("playbook_symbols")
            source = "settings"
        else:
            # Attempt to interpret as comma-separated list or fallback to top market cap.
            if "," in universe:
                tokens = _normalize(universe.split(","))
                symbols = tokens[:limit]
                source = "inline"
            else:
                symbols = await self._polygon.fetch_index_constituents(universe.upper())
                if not symbols:
                    symbols = await self._polygon.fetch_top_market_cap(limit)
                    metadata["fallback"] = "top_market_cap"

        if not symbols:
            metadata["warning"] = "polygon_unavailable"
            logger.warning("Universe provider could not retrieve symbols for %s", universe)

        trimmed = _normalize(symbols)[:limit]
        snapshot = UniverseSnapshot(
            name=universe,
            source=source,
            as_of_utc=as_of,
            symbols=trimmed,
            metadata=metadata,
        )
        if self._persist is not None:
            try:
                await self._persist.record_universe_snapshot(snapshot)
            except Exception as exc:  # pragma: no cover - persistence errors shouldn't break scans
                logger.warning("Failed to persist universe snapshot for %s: %s", universe, exc)
        return snapshot


from .persist import PlanningPersistence  # noqa: E402

__all__ = ["UniverseProvider", "UniverseSnapshot"]
