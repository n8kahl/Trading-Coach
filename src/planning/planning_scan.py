"""Entry-point utilities for planning-mode scans."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence

from ..services.contract_rules import ContractRuleBook
from ..services.persist import PlanningPersistence
from ..services.polygon_client import PolygonAggregatesClient
from ..services.scan_engine import PlanningCandidate, PlanningScanEngine, PlanningScanResult
from ..services.universe import UniverseProvider, UniverseSnapshot


@dataclass
class PlanningScanOutput:
    as_of_utc: datetime
    universe: UniverseSnapshot
    run_id: Optional[int]
    indices_context: Dict[str, Dict[str, float]]
    candidates: List[PlanningCandidate]


class PlanningScanRunner:
    """Coordinates the planning-mode scan pipeline."""

    def __init__(
        self,
        *,
        polygon_client: Optional[PolygonAggregatesClient] = None,
        persistence: Optional[PlanningPersistence] = None,
        rulebook: Optional[ContractRuleBook] = None,
    ) -> None:
        self._polygon = polygon_client or PolygonAggregatesClient()
        self._persistence = persistence or PlanningPersistence()
        self._universe_provider = UniverseProvider(self._polygon, persistence=self._persistence)
        self._engine = PlanningScanEngine(self._polygon, self._persistence, rulebook=rulebook)

    async def run(self, *, universe: str, style: str, limit: int) -> PlanningScanOutput:
        snapshot = await self._universe_provider.get_universe(universe, style=style, limit=limit)
        result = await self._engine.run(snapshot, style=style)
        return PlanningScanOutput(
            as_of_utc=result.as_of_utc,
            universe=snapshot,
            run_id=result.run_id,
            indices_context=result.indices_context,
            candidates=result.candidates,
        )

    async def close(self) -> None:
        await self._polygon.close()

    @property
    def persistence(self) -> PlanningPersistence:
        return self._persistence

    async def run_direct(
        self,
        *,
        symbols: Sequence[str],
        style: str,
        universe_name: str = "adhoc",
    ) -> PlanningScanOutput:
        snapshot = UniverseSnapshot(
            name=universe_name,
            source="adhoc",
            as_of_utc=datetime.now(timezone.utc),
            symbols=list(symbols),
            metadata={"source": "fallback"},
        )
        result = await self._engine.run(snapshot, style=style)
        return PlanningScanOutput(
            as_of_utc=result.as_of_utc,
            universe=snapshot,
            run_id=result.run_id,
            indices_context=result.indices_context,
            candidates=result.candidates,
        )

    async def load_cached(
        self,
        *,
        style: str,
        target_as_of: Optional[datetime],
    ) -> Optional[PlanningScanOutput]:
        record = await self._persistence.fetch_latest_run_with_candidates(style=style, as_of_utc=target_as_of)
        if record is None:
            record = await self._persistence.fetch_latest_run_with_candidates(style=style, as_of_utc=None)
        if record is None:
            return None
        run_row, candidate_rows = record
        cached = self._engine.replay_cached_run(run_row, candidate_rows)
        return PlanningScanOutput(
            as_of_utc=cached.as_of_utc,
            universe=cached.universe,
            run_id=cached.run_id,
            indices_context=cached.indices_context,
            candidates=cached.candidates,
        )


__all__ = ["PlanningScanRunner", "PlanningScanOutput"]
