"""Expected-move based target bands."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence


def _clamp(value: float, lower: float, upper: float) -> float:
    if lower > upper:
        lower, upper = upper, lower
    if value < lower:
        return lower
    if value > upper:
        return upper
    return value


def _style_token(style: str | None) -> str:
    token = (style or "").strip().lower()
    if token in {"0dte", "zero_dte"}:
        return "scalp"
    if token == "leap":
        return "leaps"
    return token or "intraday"


@dataclass(frozen=True)
class TPIdeal:
    label: str
    fraction: float
    lower: float
    upper: float
    optional: bool = False

    def bounded_fraction(self) -> float:
        return _clamp(self.fraction, self.lower, self.upper)

    def clamp_distance(self, expected_move: float, *, max_fraction: float | None = None) -> float:
        """Return an ideal TP distance bounded by style rails and optional EM cap."""
        base = self.fraction * expected_move
        min_distance = self.lower * expected_move
        max_distance = self.upper * expected_move
        ideal = _clamp(base, min_distance, max_distance)
        if max_fraction is not None and max_fraction > 0:
            max_cap = max_fraction * expected_move
            ideal = min(ideal, max_cap)
        return max(0.0, ideal)

    def distance_bounds(self, expected_move: float, *, max_fraction: float | None = None) -> tuple[float, float]:
        minimum = self.lower * expected_move
        maximum = self.upper * expected_move
        if max_fraction is not None and max_fraction > 0:
            maximum = min(maximum, max_fraction * expected_move)
        return max(0.0, minimum), max(0.0, maximum)


_STYLE_TARGETS: dict[str, Sequence[TPIdeal]] = {
    "scalp": (
        TPIdeal("TP1", fraction=0.30, lower=0.25, upper=0.35),
        TPIdeal("TP2", fraction=0.60, lower=0.55, upper=0.70),
        TPIdeal("TP3", fraction=0.90, lower=0.80, upper=1.00, optional=True),
    ),
    "intraday": (
        TPIdeal("TP1", fraction=0.45, lower=0.40, upper=0.55),
        TPIdeal("TP2", fraction=0.80, lower=0.70, upper=0.90),
        TPIdeal("TP3", fraction=1.10, lower=1.00, upper=1.20, optional=True),
    ),
    "swing": (
        TPIdeal("TP1", fraction=0.55, lower=0.45, upper=0.65),
        TPIdeal("TP2", fraction=0.90, lower=0.80, upper=1.10),
        TPIdeal("TP3", fraction=1.25, lower=1.10, upper=1.40, optional=True),
    ),
    "leaps": (
        TPIdeal("TP1", fraction=0.65, lower=0.55, upper=0.80),
        TPIdeal("TP2", fraction=1.00, lower=0.90, upper=1.20),
        TPIdeal("TP3", fraction=1.30, lower=1.10, upper=1.50, optional=True),
    ),
}


def tp_ideals(style: str | None, expected_move: float | None) -> List[TPIdeal]:
    """Return the ordered target ideals for the given style."""
    if expected_move is None or not isinstance(expected_move, (int, float)) or expected_move <= 0:
        return []
    style_token = _style_token(style)
    ideals = _STYLE_TARGETS.get(style_token)
    if not ideals:
        ideals = _STYLE_TARGETS["intraday"]
    return list(ideals)


__all__ = ["TPIdeal", "tp_ideals"]
