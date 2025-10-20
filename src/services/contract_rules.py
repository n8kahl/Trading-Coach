"""Rules for building option contract templates used in planning mode."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Tuple


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

    @staticmethod
    def from_dict(payload: Mapping[str, object]) -> "ContractTemplate":
        style = str(payload.get("style") or payload.get("style") or "intraday")
        option_type = str(payload.get("type") or payload.get("option_type") or "CALL").upper()
        delta_range_raw = payload.get("delta_range") or payload.get("deltaRange") or [0.3, 0.4]
        dte_range_raw = payload.get("dte_range") or payload.get("dteRange") or [0, 3]
        min_oi = int(float(payload.get("min_oi") or payload.get("min_open_interest") or 300))
        max_spread = float(payload.get("max_spread_pct") or payload.get("maxSpreadPct") or 8.0)
        delta_range = tuple(float(v) for v in delta_range_raw) if isinstance(delta_range_raw, (list, tuple)) else (0.3, 0.4)
        dte_range = tuple(int(float(v)) for v in dte_range_raw) if isinstance(dte_range_raw, (list, tuple)) else (0, 3)
        return ContractTemplate(
            style=style,
            option_type=option_type,
            delta_range=(delta_range[0], delta_range[1]),
            dte_range=(dte_range[0], dte_range[1]),
            min_open_interest=min_oi,
            max_spread_pct=max_spread,
        )


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
