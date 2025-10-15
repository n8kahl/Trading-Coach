"""Index planning mode wiring helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import pandas as pd

from .index_mode import IndexPlanner, GammaSnapshot
from .index_selector import ContractsDecision, IndexOptionSelector
from .options_select import rules_for_symbol


@dataclass(slots=True)
class IndexPlanningMode:
    planner: IndexPlanner = field(default_factory=IndexPlanner)
    selector: IndexOptionSelector = field(default_factory=IndexOptionSelector)

    def applies(self, symbol: str) -> bool:
        return self.planner.supports(symbol)

    @property
    def contract_preference(self) -> Tuple[str, ...]:
        return self.planner.contract_preference

    def structure_symbol(self, symbol: str) -> str:
        """Return the canonical index symbol for structure (SPX/NDX)."""
        return symbol.upper()

    def liquidity_proxy(self, symbol: str) -> Optional[str]:
        """Return the ETF proxy used for liquidity gating."""
        return self.planner.proxy_symbol(symbol)

    def rules(self, symbol: str) -> Dict[str, object]:
        return rules_for_symbol(symbol)

    async def select_contract(
        self,
        symbol: str,
        *,
        prefer_delta: float,
        style: str | None = None,
    ) -> Tuple[Optional[Dict[str, object]], ContractsDecision]:
        return await self.selector.contract_decision(symbol, prefer_delta=prefer_delta, style=style)

    async def ratio_snapshot(self, symbol: str) -> Optional[GammaSnapshot]:
        return await self.planner.gamma_snapshot(symbol)

    async def synthetic_ohlcv(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        return await self.planner.synthetic_index_ohlcv(symbol, timeframe)


__all__ = ["IndexPlanningMode"]
