"""Target probability calibration helpers."""

from __future__ import annotations

from typing import Mapping, Sequence, Tuple, List

from ..engine.calibration import CalibrationStore


def _clamp(value: float, *, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def calibrate_touch_prob(
    symbol: str,
    style: str,
    raw: float,
    ctx: Mapping[str, object],
) -> Tuple[float, Mapping[str, object] | None]:
    """Apply calibration tables when available, otherwise use monotone fallback."""

    raw_clamped = _clamp(float(raw))
    store = ctx.get("calibration_store")
    cohort = ctx.get("calibration_cohort")
    cohort_token = str(cohort) if cohort is not None else symbol.upper()

    if isinstance(store, CalibrationStore):
        calibrated, meta = store.calibrate(style, raw_clamped, cohort=cohort_token)
        return _clamp(calibrated), meta

    # Fallback: gentle shrink towards mean to avoid overconfidence
    if raw_clamped <= 0.5:
        adjusted = raw_clamped * 0.92 + 0.02
    else:
        adjusted = 0.98 - (1.0 - raw_clamped) * 0.92
    return _clamp(adjusted), None


def enforce_monotone(values: Sequence[float]) -> List[float]:
    """Ensure probabilities are non-increasing TP1 -> TPn."""

    out: List[float] = []
    ceiling = 1.0
    for value in values:
        candidate = _clamp(float(value))
        candidate = min(candidate, ceiling)
        out.append(candidate)
        ceiling = candidate
    return out


__all__ = ["calibrate_touch_prob", "enforce_monotone"]
