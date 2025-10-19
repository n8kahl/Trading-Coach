"""Rules for building option contract templates used in planning mode."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class ContractTemplate:
    style: str
    option_type: str
    delta_range: Tuple[float, float]
    dte_range: Tuple[int, int]
    min_open_interest: int
    max_spread_pct: float

    def as_dict(self) -> Dict[str, object]:
        return {
            "style": self.style,
            "type": self.option_type,
            "delta_range": list(self.delta_range),
            "dte_range": list(self.dte_range),
            "min_oi": self.min_open_interest,
            "max_spread_pct": self.max_spread_pct,
        }


class ContractRuleBook:
    """Factory for generating baseline contract templates."""

    _BASELINE: Dict[str, ContractTemplate] = {
        "scalp": ContractTemplate("scalp", "CALL", (0.35, 0.45), (0, 1), 250, 6.0),
        "intraday": ContractTemplate("intraday", "CALL", (0.30, 0.40), (0, 3), 400, 5.0),
        "swing": ContractTemplate("swing", "CALL", (0.20, 0.35), (5, 15), 750, 4.0),
        "leaps": ContractTemplate("leaps", "CALL", (0.60, 0.75), (180, 365), 500, 8.0),
    }

    def build(self, style: str) -> ContractTemplate:
        key = (style or "intraday").strip().lower()
        template = self._BASELINE.get(key) or self._BASELINE["intraday"]
        return template


__all__ = ["ContractRuleBook", "ContractTemplate"]
