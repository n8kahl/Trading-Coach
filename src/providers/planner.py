from __future__ import annotations

import math
from datetime import datetime, timezone
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


def _ensure_aware(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _entry_recency(series: SeriesBundle, symbol: str, *, as_of: datetime) -> Dict[str, Any]:
    probe_frames: List[tuple[str, int]] = [("5m", 5), ("15m", 15)]
    as_of_utc = _ensure_aware(as_of)
    for label, fallback_interval in probe_frames:
        frame = series.get_frame(symbol, label)
        if frame is None or frame.empty:
            continue
        ordered = frame.sort_index()
        last_ts = ordered.index[-1].to_pydatetime()
        last_ts = _ensure_aware(last_ts)
        delta_minutes = max((as_of_utc - last_ts).total_seconds() / 60.0, 0.0)
        if len(ordered.index) >= 2:
            interval_seconds = (ordered.index[-1] - ordered.index[-2]).total_seconds()
            interval_minutes = max(int(round(interval_seconds / 60.0)), 1)
        else:
            interval_minutes = fallback_interval
        interval_minutes = max(interval_minutes, 1)
        recency_bars = delta_minutes / interval_minutes if interval_minutes else None
        threshold = max(1, math.ceil(60 / interval_minutes)) if interval_minutes else None
        status = "fresh"
        if recency_bars is not None and threshold is not None and recency_bars > threshold:
            status = "stale"
        return {
            "frame": label,
            "recency_bars": recency_bars,
            "threshold_bars": threshold,
            "interval_minutes": interval_minutes,
            "status": status,
        }
    return {
        "frame": None,
        "recency_bars": None,
        "threshold_bars": None,
        "interval_minutes": None,
        "status": "unknown",
    }


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

    if series.extended or route.extended:
        charts_params["session"] = "extended"
        charts_params["range"] = "5d"

    recency_stats = _entry_recency(series, symbol, as_of=route.as_of)
    status = recency_stats.get("status") or "unknown"
    if status == "fresh":
        entry_actionability = 1.0
    elif status == "stale":
        entry_actionability = 0.45
    else:
        entry_actionability = 0.75

    evaluation: Dict[str, Any] = {"status": status}
    recency_bars = recency_stats.get("recency_bars")
    if recency_bars is not None:
        evaluation["recency_bars"] = round(recency_bars, 2)
    threshold_bars = recency_stats.get("threshold_bars")
    if threshold_bars is not None:
        evaluation["threshold_bars"] = threshold_bars
    if recency_stats.get("frame"):
        evaluation["frame"] = recency_stats["frame"]

    selected_candidate = status != "stale"
    entry_candidates = [
        {
            "anchor": "pullback",
            "entry": round(entry, 4),
            "stop": round(stop, 4),
            "targets": [round(target, 4) for target in targets],
            "selected": selected_candidate,
            "evaluation": evaluation,
        }
    ]

    warnings: List[str] = []
    if status == "stale":
        warnings.append("ENTRY_STALE")

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
        "warnings": warnings,
        "charts": {"params": charts_params},
        "charts_params": charts_params,
        "entry_candidates": entry_candidates,
        "entry_actionability": round(entry_actionability, 2),
        "actionable_soon": selected_candidate,
        "waiting_for": None if selected_candidate else "fresh_print",
        "use_extended_hours": route.extended,
    }


__all__ = ["plan"]
