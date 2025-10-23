from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List

import pandas as pd

from ..lib.data_route import DataRoute
from .geometry import GeometryBundle, GeometryDetail
from .series import SeriesBundle


def _fallback_atr(series: SeriesBundle, symbol: str) -> float | None:
    frame = series.get_frame(symbol, "1d")
    if frame is None or frame.empty:
        return None
    window = frame.tail(10)
    if len(window) < 2:
        return None
    avg_range = (window["high"] - window["low"]).mean()
    return float(avg_range) if pd.notna(avg_range) else None


def _baseline_direction(detail: GeometryDetail, series: SeriesBundle, symbol: str) -> str:
    if detail.bias:
        return detail.bias
    frame = series.get_frame(symbol, "1d")
    if frame is not None and len(frame) >= 2:
        latest = float(frame["close"].iloc[-1])
        prev = float(frame["close"].iloc[-2])
        return "long" if latest >= prev else "short"
    return "long"


def _sanitize_targets(direction: str, targets: List[float]) -> List[float]:
    if direction == "short":
        return [max(target, 0.01) for target in sorted(targets, reverse=True)]
    return [max(target, 0.01) for target in sorted(targets)]


async def plan(
    symbol: str,
    *,
    series: SeriesBundle,
    geometry: GeometryBundle,
    route: DataRoute,
) -> Dict[str, Any]:
    """Produce a plan derived from ATR-based geometry metrics."""
    detail = geometry.get(symbol)
    if detail is None:
        raise RuntimeError(f"geometry detail unavailable for {symbol}")

    price = detail.last_close or series.latest_close.get(symbol)
    if price is None:
        raise RuntimeError(f"latest close unavailable for {symbol}")

    atr = detail.expected_move or _fallback_atr(series, symbol)
    if atr is None:
        atr = max(price * 0.01, 0.1)
        em_used = False
    else:
        em_used = detail.em_used

    direction = _baseline_direction(detail, series, symbol)

    pullback = min(atr * 0.25, price * 0.015)
    risk_unit = max(atr * 0.75, price * 0.01)
    if direction == "long":
        entry = price - pullback
        stop = entry - risk_unit
        targets = _sanitize_targets(
            direction,
            [
                entry + atr,
                entry + atr * 1.6,
            ],
        )
    else:
        entry = price + pullback
        stop = entry + risk_unit
        targets = _sanitize_targets(
            direction,
            [
                entry - atr,
                entry - atr * 1.6,
            ],
        )

    risk = abs(entry - stop)
    reward = abs(targets[0] - entry) if targets else None
    rr_to_t1 = reward / risk if reward and risk > 0 else None

    snap_trace = list(detail.snap_trace)
    snap_trace.append(f"planner:{direction}")
    snap_trace.append(f"pullback={pullback:.4f}")

    chart_targets = ",".join(f"{target:.4f}" for target in targets)
    charts_params = {
        "symbol": symbol,
        "interval": "15m",
        "direction": direction,
        "entry": round(entry, 4),
        "stop": round(stop, 4),
        "tp": chart_targets,
        "plan_id": f"{symbol}-ATR",
        "plan_version": "1",
    }

    return {
        "plan_id": f"{symbol}-{route.mode}-{route.as_of.strftime('%Y%m%d%H%M')}",
        "version": 1,
        "trade_detail": f"{direction.capitalize()} setup derived from ATR geometry",
        "symbol": symbol,
        "bias": direction,
        "entry": round(entry, 4),
        "stop": round(stop, 4),
        "targets": [round(target, 4) for target in targets],
        "rr_to_t1": rr_to_t1,
        "snap_trace": snap_trace,
        "key_levels_used": detail.key_levels_used,
        "data_quality": {
            "expected_move": detail.expected_move or atr,
            "remaining_atr": detail.remaining_atr,
            "em_used": em_used,
        },
        "warnings": [],
        "charts": {"params": charts_params},
        "charts_params": charts_params,
    }


__all__ = ["plan"]
