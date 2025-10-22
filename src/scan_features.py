"""Fast feature extraction helpers for scan ranking."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, time
from typing import Any, Dict, List, Optional

import pandas as pd

from .ranking import Style
from .scanner import Plan, Signal

_ACTIONABILITY_THRESHOLDS = {
    "pct": 1.5,
    "atr": 0.75,
    "bars": 4.0,
}


@dataclass(slots=True)
class Penalties:
    pen_event: float
    pen_dq: float
    pen_spread: float
    pen_chop: float
    pen_cluster: float


@dataclass(slots=True)
class Metrics:
    symbol: str
    sector: str | None
    entry_quality: float
    rr_t1: float
    rr_t2: float
    liquidity: float
    confluence_micro: float
    momentum_micro: float
    vol_ok: float
    struct_d1: float
    conf_d1: float
    mom_htf: float
    conf_htf: float
    struct_w1: float
    vol_regime: float
    opt_eff: float
    rr_multi: float
    macro_fit: float
    context_score: float
    penalties: Penalties
    confidence: float
    actionability: float
    entry_distance_pct: float
    entry_distance_atr: float
    bars_to_trigger: float
    vol_proxy: float


@dataclass(slots=True)
class MetricsContext:
    symbol: str
    style: Style
    as_of: datetime
    is_open: bool
    simulate_open: bool
    history: pd.DataFrame
    signal: Signal
    plan: Plan | None
    indicator_bundle: Dict[str, Any]
    market_meta: Dict[str, Any]
    data_meta: Dict[str, Any]


def compute_metrics_fast(symbol: str, style: Style, context: MetricsContext) -> Metrics:
    plan = context.plan
    signal = context.signal
    features = signal.features or {}
    sector = (features.get("sector") or features.get("industry") or "").upper() or None
    confidence = _clamp(float(plan.confidence) if plan and plan.confidence is not None else float(features.get("plan_confidence") or 0.5))
    entry_quality = _entry_quality(style, features, plan, context.indicator_bundle)
    rr_t1, rr_t2 = _risk_reward(plan, features)
    liquidity = _liquidity_score(context.history, features)
    confluence_micro = _ema_confluence(context.indicator_bundle)
    momentum_micro = _momentum(context.history, context.indicator_bundle)
    vol_ok = _vol_sweet_spot(context.indicator_bundle)
    struct_d1, conf_d1, mom_htf, conf_htf, struct_w1 = _htf_snapshots(context.indicator_bundle, features)
    vol_regime = _vol_regime(features, context.indicator_bundle)
    opt_eff = _option_efficiency(features)
    rr_multi = _clamp((rr_t1 + rr_t2) / 2.0, 0.0, 3.0)
    macro_fit = _macro_alignment(features, context.as_of)
    context_score = _context_score(context, features)
    penalties = _penalties(features, context.data_meta, simulate_open=context.simulate_open)
    last_close = float("nan")
    try:
        if "close" in context.history.columns and not context.history["close"].empty:
            last_close = float(context.history["close"].iloc[-1])
    except Exception:
        last_close = float("nan")
    entry_value = None
    stop_value = None
    if plan and plan.entry is not None:
        try:
            entry_value = float(plan.entry)
        except (TypeError, ValueError):
            entry_value = None
    if plan and plan.stop is not None:
        try:
            stop_value = float(plan.stop)
        except (TypeError, ValueError):
            stop_value = None
    atr_hint = context.indicator_bundle.get("atr") if isinstance(context.indicator_bundle, dict) else None
    try:
        atr_value = float(atr_hint) if atr_hint is not None else float("nan")
    except (TypeError, ValueError):
        atr_value = float("nan")
    distance_points = float("nan")
    if entry_value is not None and math.isfinite(last_close):
        distance_points = abs(entry_value - last_close)
    entry_distance_pct = _safe_ratio(distance_points, last_close) * 100.0
    entry_distance_atr = _safe_ratio(distance_points, atr_value)
    avg_range = float("nan")
    try:
        if {"high", "low"}.issubset(context.history.columns):
            ranges = (context.history["high"] - context.history["low"]).tail(12).dropna()
            if not ranges.empty:
                avg_range = float(ranges.mean())
    except Exception:
        avg_range = float("nan")
    bars_to_trigger = _safe_ratio(distance_points, avg_range)
    actionability = _actionability_score(entry_distance_pct, entry_distance_atr, bars_to_trigger)
    vol_proxy_raw = None
    if isinstance(context.market_meta, dict):
        for key in ("vol_proxy", "vix_proxy", "vix", "vix_value"):
            if key in context.market_meta:
                vol_proxy_raw = context.market_meta.get(key)
                break
    try:
        vol_proxy = float(vol_proxy_raw) if vol_proxy_raw is not None else 0.0
    except (TypeError, ValueError):
        vol_proxy = 0.0

    return Metrics(
        symbol=symbol,
        sector=sector,
        entry_quality=entry_quality,
        rr_t1=rr_t1,
        rr_t2=rr_t2,
        liquidity=liquidity,
        confluence_micro=confluence_micro,
        momentum_micro=momentum_micro,
        vol_ok=vol_ok,
        struct_d1=struct_d1,
        conf_d1=conf_d1,
        mom_htf=mom_htf,
        conf_htf=conf_htf,
        struct_w1=struct_w1,
        vol_regime=vol_regime,
        opt_eff=opt_eff,
        rr_multi=rr_multi,
        macro_fit=macro_fit,
        context_score=context_score,
        penalties=penalties,
        confidence=confidence,
        actionability=_clamp(actionability),
        entry_distance_pct=entry_distance_pct if math.isfinite(entry_distance_pct) else float("nan"),
        entry_distance_atr=entry_distance_atr if math.isfinite(entry_distance_atr) else float("nan"),
        bars_to_trigger=bars_to_trigger if math.isfinite(bars_to_trigger) else float("nan"),
        vol_proxy=vol_proxy,
    )


def _entry_quality(style: Style, features: Dict[str, Any], plan: Plan | None, indicators: Dict[str, Any]) -> float:
    base = 0.45
    entry_type = str(features.get("entry_type") or "").lower()
    if style == "scalp" and entry_type in {"reclaim", "reclaim_continuation", "continuation"}:
        base += 0.2
    elif style in {"intraday", "swing"} and entry_type in {"reclaim", "continuation"}:
        base += 0.1
    time_bucket = str(features.get("time_bucket") or "").lower()
    if style == "scalp" and time_bucket in {"09:45-10:15", "power_hour"}:
        base += 0.05
    if plan and plan.entry is not None and plan.stop is not None:
        risk = abs(float(plan.entry) - float(plan.stop))
        atr_val = float(indicators.get("atr", 0.0) or 0.0)
        if atr_val > 0:
            ratio = max(min(risk / atr_val, 2.5), 0.1)
            base += 0.1 if 0.3 <= ratio <= 1.1 else -0.1
    return _clamp(base)


def _risk_reward(plan: Plan | None, features: Dict[str, Any]) -> tuple[float, float]:
    rr_t1 = float(plan.risk_reward) if plan and plan.risk_reward is not None else float(features.get("rr_t1") or 1.0)
    targets = list(plan.targets) if plan and plan.targets else []
    rr_t2 = float(features.get("rr_t2") or 1.2)
    if plan and plan.entry is not None and plan.stop is not None and len(targets) >= 2:
        risk = abs(float(plan.entry) - float(plan.stop))
        if risk > 0:
            rr_t1 = abs(float(targets[0]) - float(plan.entry)) / risk
            rr_t2 = abs(float(targets[1]) - float(plan.entry)) / risk
    return max(rr_t1, 0.0), max(rr_t2, 0.0)


def _liquidity_score(history: pd.DataFrame, features: Dict[str, Any]) -> float:
    if history.empty:
        return 0.2
    volume = history["volume"].iloc[-1] if "volume" in history.columns else 0.0
    avg_volume = history["volume"].tail(20).mean() if "volume" in history.columns else 0.0
    rvol = 0.0 if avg_volume == 0 else max(min(float(volume) / float(avg_volume), 3.0), 0.1)
    option_oi = float(features.get("option_oi") or 0.0)
    oi_score = 1.0 if option_oi >= 1000 else option_oi / 1000.0
    spread_pct = float(features.get("option_spread_pct") or 0.12)
    spread_score = 1.0 - _clamp(spread_pct / 0.5)
    return _clamp(0.5 * _map_range(rvol, 0.1, 2.5) + 0.3 * spread_score + 0.2 * oi_score)


def _ema_confluence(indicators: Dict[str, Any]) -> float:
    ema_stack = indicators.get("ema_stack")
    if isinstance(ema_stack, dict):
        ordering = ema_stack.get("ordering") or []
        if ordering == ["ema9", "ema21", "ema55"] or ordering == ["ema9", "ema20", "ema50"]:
            return 0.9
        if ordering == ["ema21", "ema55", "ema200"]:
            return 0.6
    return float(indicators.get("ema_alignment", 0.5))


def _momentum(history: pd.DataFrame, indicators: Dict[str, Any]) -> float:
    if history.empty:
        return 0.3
    closes = history["close"]
    if len(closes) < 3:
        return 0.4
    body = closes.iloc[-1] - history["open"].iloc[-1] if "open" in history.columns else 0.0
    range_span = history["high"].iloc[-1] - history["low"].iloc[-1] if "high" in history.columns and "low" in history.columns else 0.0
    body_pct = 0.0 if range_span == 0 else _clamp(abs(body) / range_span)
    impulse = float(indicators.get("impulse") or 0.5)
    return _clamp(0.6 * body_pct + 0.4 * impulse)


def _vol_sweet_spot(indicators: Dict[str, Any]) -> float:
    atr_ratio = float(indicators.get("atr_multiple") or 1.0)
    if atr_ratio <= 0:
        return 0.5
    delta = abs(1.8 - atr_ratio)
    return _clamp(1.0 - min(delta / 2.0, 0.8))


def _htf_snapshots(indicators: Dict[str, Any], features: Dict[str, Any]) -> tuple[float, float, float, float, float]:
    htf = indicators.get("htf") or {}
    struct_d1 = float(htf.get("structure_d1") or features.get("structure_d1") or 0.5)
    conf_d1 = float(htf.get("confluence_d1") or features.get("confluence_d1") or 0.5)
    mom_htf = float(htf.get("momentum") or features.get("momentum_htf") or 0.5)
    conf_htf = float(htf.get("confluence") or features.get("confluence_htf") or 0.5)
    struct_w1 = float(htf.get("structure_w1") or features.get("structure_w1") or 0.5)
    return (_clamp(struct_d1), _clamp(conf_d1), _clamp(mom_htf), _clamp(conf_htf), _clamp(struct_w1))


def _vol_regime(features: Dict[str, Any], indicators: Dict[str, Any]) -> float:
    ivr = float(features.get("ivr") or features.get("iv_rank") or 0.4)
    atr_norm = float(indicators.get("atr_norm") or 0.6)
    return _clamp(0.6 * _clamp(ivr) + 0.4 * _clamp(atr_norm))


def _option_efficiency(features: Dict[str, Any]) -> float:
    theta_per_risk = float(features.get("theta_per_risk") or 0.1)
    delta_avail = float(features.get("delta_availability") or 0.7)
    spread_pct = float(features.get("option_spread_pct") or 0.12)
    spread_penalty = _clamp(spread_pct / 0.6)
    base = 0.6 * _clamp(theta_per_risk) + 0.4 * _clamp(delta_avail)
    return _clamp(base - 0.3 * spread_penalty)


def _macro_alignment(features: Dict[str, Any], as_of: datetime) -> float:
    macro_score = float(features.get("macro_fit") or 0.55)
    window = features.get("macro_window") or []
    if isinstance(window, list) and window:
        if any(str(event).lower() in {"cpi", "fomc"} for event in window):
            center = datetime.combine(as_of.date(), time(8, 30))
            diff = abs((as_of - center).total_seconds()) / 3600
            macro_score -= min(diff / 6.0, 0.2)
    return _clamp(macro_score)


def _context_score(context: MetricsContext, features: Dict[str, Any]) -> float:
    base = float(features.get("context_score") or 0.55)
    if not context.is_open:
        return _clamp(base)
    dq_mode = str(context.data_meta.get("mode") or "").lower()
    if dq_mode == "degraded":
        base -= 0.1
    session_bias = features.get("session_bias")
    if isinstance(session_bias, (int, float)):
        base += _clamp(session_bias) * 0.1
    return _clamp(base)


def _penalties(features: Dict[str, Any], data_meta: Dict[str, Any], *, simulate_open: bool = False) -> Penalties:
    earnings_days = float(features.get("earnings_days") or 10.0)
    pen_event = 0.2 if earnings_days <= 2 else 0.0
    dq_mode = str(data_meta.get("mode") or "").lower()
    dq_pen = 0.0 if simulate_open else (0.1 if dq_mode == "degraded" else 0.0)
    spread_pct = float(features.get("option_spread_pct") or 0.12)
    pen_spread = min(max(spread_pct - 0.12, 0.0) / 0.3, 0.1)
    chop_risk = float(features.get("chop_risk") or 0.0)
    pen_chop = _clamp(chop_risk) * 0.12
    cluster_risk = float(features.get("cluster_risk") or 0.0)
    pen_cluster = _clamp(cluster_risk) * 0.08
    return Penalties(
        pen_event=_clamp(pen_event, 0.0, 0.2),
        pen_dq=_clamp(dq_pen, 0.0, 0.1),
        pen_spread=_clamp(pen_spread, 0.0, 0.1),
        pen_chop=_clamp(pen_chop, 0.0, 0.12),
        pen_cluster=_clamp(pen_cluster, 0.0, 0.08),
    )


def _safe_ratio(numerator: float, denominator: float) -> float:
    try:
        if numerator is None or denominator is None:
            return float("nan")
        num = float(numerator)
        den = float(denominator)
        if not math.isfinite(num) or not math.isfinite(den) or den == 0:
            return float("nan")
        return num / den
    except Exception:
        return float("nan")


def _score_component(value: float, threshold: float) -> float:
    if threshold <= 0:
        return 0.0
    if not math.isfinite(value):
        return 0.0
    ratio = max(value / threshold, 0.0)
    return _clamp(1.0 - ratio)


def _actionability_score(distance_pct: float, distance_atr: float, bars_to_trigger: float) -> float:
    components: List[tuple[float, float]] = [
        (_score_component(distance_pct, _ACTIONABILITY_THRESHOLDS["pct"]), 0.5),
        (_score_component(distance_atr, _ACTIONABILITY_THRESHOLDS["atr"]), 0.3),
        (_score_component(bars_to_trigger, _ACTIONABILITY_THRESHOLDS["bars"]), 0.2),
    ]
    weighted_sum = sum(value * weight for value, weight in components)
    total_weight = sum(weight for _, weight in components)
    if total_weight <= 0:
        return 0.0
    return _clamp(weighted_sum / total_weight)


def _map_range(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    return _clamp((value - low) / (high - low))


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    if math.isnan(value):
        return low
    return max(low, min(high, value))


__all__ = ["Metrics", "MetricsContext", "Penalties", "compute_metrics_fast"]
