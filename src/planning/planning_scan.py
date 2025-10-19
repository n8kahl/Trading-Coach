"""Entry-point utilities for planning-mode scans."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

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


__all__ = ["PlanningScanRunner", "PlanningScanOutput"]
