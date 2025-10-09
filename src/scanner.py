"""Market scanning logic for detecting trade setups using live market data.

The scanner evaluates each configured strategy with fully realised
calculations (ATR, VWAP, anchored VWAPs, EMA stacks, etc.) and only produces
signals when the underlying market structure and statistics satisfy each
strategy's rule set.  No placeholder heuristics remain — every score, plan,
and directional hint comes directly from current intraday data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from .strategy_library import Strategy, load_strategies
from .calculations import atr, ema, vwap, adx

TZ_ET = "America/New_York"
RTH_START_MINUTE = 9 * 60 + 30
RTH_END_MINUTE = 16 * 60


@dataclass(slots=True)
class Plan:
    direction: str
    entry: float
    stop: float
    targets: List[float]
    confidence: float
    risk_reward: float
    notes: str | None = None
    atr: float | None = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "direction": self.direction,
            "entry": round(float(self.entry), 4),
            "stop": round(float(self.stop), 4),
            "targets": [round(float(t), 4) for t in self.targets],
            "confidence": round(float(self.confidence), 3),
            "risk_reward": round(float(self.risk_reward), 3),
            "atr": round(float(self.atr), 4) if self.atr is not None else None,
            "notes": self.notes,
        }


@dataclass
class Signal:
    """Represents a detected trade opportunity."""

    symbol: str
    strategy_id: str
    description: str
    score: float
    contract: Dict[str, Any] | None = None
    features: Dict[str, Any] = field(default_factory=dict)
    options_rules: Dict[str, Any] | None = None
    plan: Plan | None = None


def _ensure_datetime_index(frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(frame.index, pd.DatetimeIndex):
        frame = frame.copy()
        frame.index = pd.to_datetime(frame.index)
    if frame.index.tz is None:
        frame.index = frame.index.tz_localize("UTC")
    else:
        frame.index = frame.index.tz_convert("UTC")
    return frame


def _latest_sessions(frame: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame | None]:
    if frame.empty:
        return frame, None
    frame = _ensure_datetime_index(frame).sort_index()
    et_index = frame.index.tz_convert(TZ_ET)
    session_dates = pd.Series(et_index.date, index=frame.index)
    if session_dates.empty:
        return frame, None
    latest_date = session_dates.iloc[-1]
    session_mask = session_dates.eq(latest_date).to_numpy()
    session_df = frame.iloc[session_mask]

    prev_df: pd.DataFrame | None = None
    unique_dates = session_dates.drop_duplicates().tolist()
    if len(unique_dates) >= 2:
        prev_date = unique_dates[-2]
        prev_mask = session_dates.eq(prev_date).to_numpy()
        prev_df = frame.iloc[prev_mask]
    return session_df, prev_df


def _minutes_from_midnight(index: pd.DatetimeIndex) -> np.ndarray:
    et_index = index.tz_convert(TZ_ET)
    return et_index.hour * 60 + et_index.minute


def _score_conditions(flags: Iterable[bool], bonus: float = 0.0, clamp: Tuple[float, float] = (0.0, 0.98)) -> float:
    flags = list(flags)
    if not flags:
        return round(max(clamp[0], min(clamp[1], 0.25 + bonus)), 3)
    positive = sum(1 for flag in flags if flag)
    ratio = positive / len(flags)
    confidence = 0.25 + ratio * 0.6 + bonus
    return round(max(clamp[0], min(clamp[1], confidence)), 3)


def _build_plan(
    direction: str,
    entry: float,
    stop: float,
    targets: List[float],
    *,
    atr_value: float | None,
    notes: str | None,
    conditions: Iterable[bool],
) -> Plan | None:
    if not math.isfinite(entry) or not math.isfinite(stop):
        return None
    clean_targets = [float(t) for t in targets if math.isfinite(t)]
    if not clean_targets:
        return None
    if direction == "long":
        if stop >= entry:
            return None
        risk = entry - stop
        reward = clean_targets[0] - entry
    else:
        if stop <= entry:
            return None
        risk = stop - entry
        reward = entry - clean_targets[0]
    if risk <= 0 or reward <= 0:
        return None
    risk_reward = reward / risk
    confidence = _score_conditions(conditions)
    return Plan(
        direction=direction,
        entry=float(entry),
        stop=float(stop),
        targets=clean_targets,
        confidence=float(confidence),
        risk_reward=float(round(risk_reward, 3)),
        notes=notes,
        atr=float(atr_value) if atr_value is not None and math.isfinite(atr_value) else None,
    )


def _anchored_vwap(frame: pd.DataFrame, anchor_ts: pd.Timestamp) -> float | None:
    segment = frame.loc[frame.index >= anchor_ts]
    if segment.empty or "volume" not in segment.columns or segment["volume"].sum() <= 0:
        return None
    typical = segment["typical_price"]
    pv = (typical * segment["volume"]).cumsum()
    cum_volume = segment["volume"].cumsum()
    denom = float(cum_volume.iloc[-1])
    if denom <= 0:
        return None
    return float(pv.iloc[-1] / denom)


def _session_phase(ts: pd.Timestamp) -> str:
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    ts = ts.tz_convert(TZ_ET)
    h, m = ts.hour, ts.minute
    wd = ts.weekday()
    if wd >= 5:
        return "off"
    if (h < 9) or (h == 9 and m < 30):
        return "premarket"
    if h == 9 and 30 <= m < 60:
        return "open_drive"
    if h == 10 or (h == 11 and m < 30):
        return "morning"
    if (h == 11 and m >= 30) or (12 <= h < 14):
        return "midday"
    if h == 14:
        return "afternoon"
    if h == 15:
        return "power_hour"
    if h >= 16:
        return "postmarket"
    return "other"


def _prepare_symbol_frame(frame: pd.DataFrame) -> pd.DataFrame:
    frame = _ensure_datetime_index(frame).sort_index().copy()
    if frame.empty:
        return frame
    for column in ["open", "high", "low", "close", "volume"]:
        if column not in frame.columns:
            raise ValueError(f"Expected column '{column}' missing from OHLCV data.")
    frame["atr14"] = atr(frame["high"], frame["low"], frame["close"], 14)
    frame["ema9"] = ema(frame["close"], 9)
    frame["ema20"] = ema(frame["close"], 20)
    frame["ema50"] = ema(frame["close"], 50)
    frame["vwap"] = vwap(frame["close"], frame["volume"])
    frame["adx14"] = adx(frame["high"], frame["low"], frame["close"], 14)
    frame["typical_price"] = (frame["high"] + frame["low"] + frame["close"]) / 3.0
    return frame


def _build_context(frame: pd.DataFrame) -> Dict[str, Any]:
    session_df, prev_session_df = _latest_sessions(frame)
    latest = frame.iloc[-1]
    atr_value = float(latest["atr14"]) if math.isfinite(latest["atr14"]) else math.nan
    volume_median = float(session_df["volume"].tail(40).median()) if not session_df.empty else math.nan
    minutes_vector = _minutes_from_midnight(session_df.index) if not session_df.empty else np.array([], dtype=int)
    return {
        "frame": frame,
        "session": session_df,
        "prev_session": prev_session_df,
        "latest": latest,
        "atr": atr_value,
        "price": float(latest["close"]),
        "vwap": float(latest["vwap"]),
        "ema9": float(latest["ema9"]),
        "ema20": float(latest["ema20"]),
        "ema50": float(latest["ema50"]),
        "adx": float(latest["adx14"]) if math.isfinite(latest["adx14"]) else math.nan,
        "volume_median": volume_median,
        "minutes_vector": minutes_vector,
        "timestamp": frame.index[-1],
        "session_phase": _session_phase(frame.index[-1]),
    }


def _detect_orb_retest(symbol: str, strategy: Strategy, ctx: Dict[str, Any]) -> Signal | None:
    session = ctx["session"]
    if session.empty:
        return None
    minutes = ctx["minutes_vector"]
    if minutes.size == 0 or minutes.min() > RTH_START_MINUTE:
        return None

    window_minutes = 15
    range_mask = (minutes >= RTH_START_MINUTE) & (minutes < RTH_START_MINUTE + window_minutes)
    if not range_mask.any():
        return None
    opening_range = session.iloc[range_mask]
    post_range = session.iloc[~range_mask]
    if opening_range.empty or post_range.empty:
        return None

    atr_value = ctx["atr"]
    if not math.isfinite(atr_value) or atr_value <= 0:
        return None

    or_high = float(opening_range["high"].max())
    or_low = float(opening_range["low"].min())
    latest = ctx["latest"]
    price = float(latest["close"])
    tolerance = max(atr_value * 0.25, price * 0.0015)
    ema_stack_long = latest["ema9"] > latest["ema20"] > latest["ema50"]
    ema_stack_short = latest["ema9"] < latest["ema20"] < latest["ema50"]
    adx_strong = ctx["adx"] >= 18 if math.isfinite(ctx["adx"]) else False
    volume_ok = math.isfinite(ctx["volume_median"]) and latest["volume"] >= ctx["volume_median"]

    notes: List[str] = []
    plan: Plan | None = None

    recent_slice = post_range.tail(20)
    retest_low = float(recent_slice["low"].min()) if not recent_slice.empty else float("nan")
    retest_high = float(recent_slice["high"].max()) if not recent_slice.empty else float("nan")

    if price > or_high and math.isfinite(retest_low) and abs(retest_low - or_high) <= tolerance:
        entry = max(price, or_high)
        stop = retest_low - tolerance * 0.5
        target_primary = max(float(recent_slice["high"].max()), entry + atr_value)
        target_secondary = entry + atr_value * 1.5
        plan = _build_plan(
            "long",
            entry,
            stop,
            [target_primary, target_secondary],
            atr_value=atr_value,
            notes=f"Reclaimed OR high {or_high:.2f}; retest low {retest_low:.2f}",
            conditions=[ema_stack_long, adx_strong, volume_ok],
        )
        if plan:
            notes.append("Long OR retest validated")

    elif price < or_low and math.isfinite(retest_high) and abs(retest_high - or_low) <= tolerance:
        entry = min(price, or_low)
        stop = retest_high + tolerance * 0.5
        target_primary = min(float(recent_slice["low"].min()), entry - atr_value)
        target_secondary = entry - atr_value * 1.5
        plan = _build_plan(
            "short",
            entry,
            stop,
            [target_primary, target_secondary],
            atr_value=atr_value,
            notes=f"Rejected OR low {or_low:.2f}; retest high {retest_high:.2f}",
            conditions=[ema_stack_short, adx_strong, volume_ok],
        )
        if plan:
            notes.append("Short OR retest validated")

    if plan is None:
        return None

    features = {
        "atr": atr_value,
        "adx": ctx["adx"],
        "direction_bias": plan.direction,
        "session_phase": ctx["session_phase"],
        "opening_range_high": or_high,
        "opening_range_low": or_low,
        "retest_extreme": retest_low if plan.direction == "long" else retest_high,
        "vwap": ctx["vwap"],
        "ema9": ctx["ema9"],
        "ema20": ctx["ema20"],
        "ema50": ctx["ema50"],
        "plan_entry": plan.entry,
        "plan_stop": plan.stop,
        "plan_targets": plan.targets,
        "plan_confidence": plan.confidence,
        "plan_risk_reward": plan.risk_reward,
        "plan_notes": plan.notes,
    }

    return Signal(
        symbol=symbol,
        strategy_id=strategy.id,
        description=strategy.description,
        score=plan.confidence,
        features=features,
        options_rules=strategy.options_rules,
        plan=plan,
    )


def _detect_power_hour_trend(symbol: str, strategy: Strategy, ctx: Dict[str, Any]) -> Signal | None:
    if ctx["session_phase"] != "power_hour":
        return None
    session = ctx["session"]
    if session.empty:
        return None
    latest = ctx["latest"]
    atr_value = ctx["atr"]
    if not math.isfinite(atr_value) or atr_value <= 0:
        return None

    window = session.tail(30)
    range_high = float(window["high"].max())
    range_low = float(window["low"].min())
    price = float(latest["close"])
    adx_strong = ctx["adx"] >= 20 if math.isfinite(ctx["adx"]) else False
    ema_stack_long = latest["ema9"] > latest["ema20"] > latest["ema50"]
    ema_stack_short = latest["ema9"] < latest["ema20"] < latest["ema50"]

    plan: Plan | None = None
    breakout_long = price >= range_high - 0.05 * atr_value
    breakout_short = price <= range_low + 0.05 * atr_value
    volume_ok = math.isfinite(ctx["volume_median"]) and latest["volume"] >= ctx["volume_median"]

    if price > ctx["vwap"] and ema_stack_long and breakout_long:
        entry = price
        stop = min(range_low, float(session["low"].tail(10).min())) - atr_value * 0.25
        target_primary = entry + atr_value
        target_secondary = max(range_high + atr_value * 0.8, entry + atr_value * 1.6)
        plan = _build_plan(
            "long",
            entry,
            stop,
            [target_primary, target_secondary],
            atr_value=atr_value,
            notes=f"VWAP support {ctx['vwap']:.2f}; afternoon range high {range_high:.2f}",
            conditions=[adx_strong, volume_ok, breakout_long],
        )
    elif price < ctx["vwap"] and ema_stack_short and breakout_short:
        entry = price
        stop = max(range_high, float(session["high"].tail(10).max())) + atr_value * 0.25
        target_primary = entry - atr_value
        target_secondary = min(range_low - atr_value * 0.8, entry - atr_value * 1.6)
        plan = _build_plan(
            "short",
            entry,
            stop,
            [target_primary, target_secondary],
            atr_value=atr_value,
            notes=f"VWAP resistance {ctx['vwap']:.2f}; afternoon range low {range_low:.2f}",
            conditions=[adx_strong, volume_ok, breakout_short],
        )

    if plan is None:
        return None

    features = {
        "atr": atr_value,
        "adx": ctx["adx"],
        "direction_bias": plan.direction,
        "session_phase": ctx["session_phase"],
        "range_high": range_high,
        "range_low": range_low,
        "vwap": ctx["vwap"],
        "ema9": ctx["ema9"],
        "ema20": ctx["ema20"],
        "ema50": ctx["ema50"],
        "plan_entry": plan.entry,
        "plan_stop": plan.stop,
        "plan_targets": plan.targets,
        "plan_confidence": plan.confidence,
        "plan_risk_reward": plan.risk_reward,
        "plan_notes": plan.notes,
    }

    return Signal(
        symbol=symbol,
        strategy_id=strategy.id,
        description=strategy.description,
        score=plan.confidence,
        features=features,
        options_rules=strategy.options_rules,
        plan=plan,
    )


def _detect_vwap_cluster(symbol: str, strategy: Strategy, ctx: Dict[str, Any]) -> Signal | None:
    session = ctx["session"]
    prev_session = ctx["prev_session"]
    if session.empty or prev_session is None or prev_session.empty:
        return None
    atr_value = ctx["atr"]
    if not math.isfinite(atr_value) or atr_value <= 0:
        return None

    frame = ctx["frame"]
    prev_high_idx = prev_session["high"].idxmax()
    prev_low_idx = prev_session["low"].idxmin()
    open_idx = session.index[0]

    anchors = {
        "prev_high": _anchored_vwap(frame, prev_high_idx),
        "prev_low": _anchored_vwap(frame, prev_low_idx),
        "session_open": _anchored_vwap(frame, open_idx),
    }
    anchored_values = [val for val in anchors.values() if val is not None]
    if len(anchored_values) < 2:
        return None

    price = ctx["price"]
    cluster_mean = float(np.mean(anchored_values))
    cluster_spread = float(np.max(anchored_values) - np.min(anchored_values))
    tolerance = max(atr_value * 0.2, price * 0.001)
    cluster_tight = cluster_spread <= tolerance

    ema_stack_long = ctx["ema9"] > ctx["ema20"] > ctx["ema50"]
    ema_stack_short = ctx["ema9"] < ctx["ema20"] < ctx["ema50"]
    adx_ok = ctx["adx"] >= 16 if math.isfinite(ctx["adx"]) else False

    plan: Plan | None = None
    if price > ctx["vwap"] and ema_stack_long and cluster_tight and price > cluster_mean:
        entry = price
        stop = min(cluster_mean, np.min(anchored_values)) - tolerance
        target_primary = entry + atr_value * 1.1
        target_secondary = entry + atr_value * 1.8
        plan = _build_plan(
            "long",
            entry,
            stop,
            [target_primary, target_secondary],
            atr_value=atr_value,
            notes=f"Above VWAP cluster (~{cluster_mean:.2f}); spread {cluster_spread:.2f}",
            conditions=[cluster_tight, adx_ok],
        )
    elif price < ctx["vwap"] and ema_stack_short and cluster_tight and price < cluster_mean:
        entry = price
        stop = max(cluster_mean, np.max(anchored_values)) + tolerance
        target_primary = entry - atr_value * 1.1
        target_secondary = entry - atr_value * 1.8
        plan = _build_plan(
            "short",
            entry,
            stop,
            [target_primary, target_secondary],
            atr_value=atr_value,
            notes=f"Below VWAP cluster (~{cluster_mean:.2f}); spread {cluster_spread:.2f}",
            conditions=[cluster_tight, adx_ok],
        )

    if plan is None:
        return None

    features = {
        "atr": atr_value,
        "adx": ctx["adx"],
        "direction_bias": plan.direction,
        "session_phase": ctx["session_phase"],
        "session_vwap": ctx["vwap"],
        "anchored_vwap_prev_high": anchors["prev_high"],
        "anchored_vwap_prev_low": anchors["prev_low"],
        "anchored_vwap_session_open": anchors["session_open"],
        "cluster_span": cluster_spread,
        "plan_entry": plan.entry,
        "plan_stop": plan.stop,
        "plan_targets": plan.targets,
        "plan_confidence": plan.confidence,
        "plan_risk_reward": plan.risk_reward,
        "plan_notes": plan.notes,
    }

    return Signal(
        symbol=symbol,
        strategy_id=strategy.id,
        description=strategy.description,
        score=plan.confidence,
        features=features,
        options_rules=strategy.options_rules,
        plan=plan,
    )


def _detect_gap_fill(symbol: str, strategy: Strategy, ctx: Dict[str, Any]) -> Signal | None:
    session = ctx["session"]
    prev_session = ctx["prev_session"]
    if session.empty or prev_session is None or prev_session.empty:
        return None
    minutes = ctx["minutes_vector"]
    if minutes.size == 0 or minutes.min() > RTH_START_MINUTE:
        return None

    phase = ctx["session_phase"]
    if phase not in {"open_drive", "morning"}:
        return None

    latest = ctx["latest"]
    atr_value = ctx["atr"]
    if not math.isfinite(atr_value) or atr_value <= 0:
        return None

    prev_close = float(prev_session["close"].iloc[-1])
    first_open = float(session["open"].iloc[0])
    gap = first_open - prev_close
    gap_abs = abs(gap)
    min_gap = max(0.3 * atr_value, 0.003 * prev_close)
    if gap_abs < min_gap:
        return None

    price = float(latest["close"])
    filling = (gap > 0 and price < first_open) or (gap < 0 and price > first_open)
    if not filling:
        return None

    vwap_alignment = (gap > 0 and price < ctx["vwap"]) or (gap < 0 and price > ctx["vwap"])
    distance_to_close = abs(price - prev_close)
    progress = abs(price - first_open) / gap_abs if gap_abs else 0
    volume_ok = math.isfinite(ctx["volume_median"]) and latest["volume"] >= ctx["volume_median"]

    if gap > 0:
        direction = "short"
        entry = price
        stop = max(first_open + atr_value * 0.25, float(session["high"].head(3).max()))
        target_primary = prev_close
        target_secondary = prev_close - atr_value * 0.6
    else:
        direction = "long"
        entry = price
        stop = min(first_open - atr_value * 0.25, float(session["low"].head(3).min()))
        target_primary = prev_close
        target_secondary = prev_close + atr_value * 0.6

    plan = _build_plan(
        direction,
        entry,
        stop,
        [target_primary, target_secondary],
        atr_value=atr_value,
        notes=f"Gap {gap:+.2f} vs prev close {prev_close:.2f}; progress {progress:.2%}",
        conditions=[vwap_alignment, volume_ok, distance_to_close > atr_value * 0.2],
    )
    if plan is None:
        return None

    features = {
        "atr": atr_value,
        "adx": ctx["adx"],
        "direction_bias": plan.direction,
        "session_phase": ctx["session_phase"],
        "gap_points": gap,
        "prev_close": prev_close,
        "session_open": first_open,
        "vwap": ctx["vwap"],
        "gap_fill_progress": progress,
        "plan_entry": plan.entry,
        "plan_stop": plan.stop,
        "plan_targets": plan.targets,
        "plan_confidence": plan.confidence,
        "plan_risk_reward": plan.risk_reward,
        "plan_notes": plan.notes,
    }

    return Signal(
        symbol=symbol,
        strategy_id=strategy.id,
        description=strategy.description,
        score=plan.confidence,
        features=features,
        options_rules=strategy.options_rules,
        plan=plan,
    )


def _detect_midday_mean_revert(symbol: str, strategy: Strategy, ctx: Dict[str, Any]) -> Signal | None:
    if ctx["session_phase"] != "midday":
        return None
    session = ctx["session"]
    if session.empty:
        return None
    latest = ctx["latest"]
    atr_value = ctx["atr"]
    if not math.isfinite(atr_value) or atr_value <= 0:
        return None

    price = float(latest["close"])
    distance = price - ctx["vwap"]
    extension = abs(distance)
    threshold = atr_value * 0.6
    if extension < threshold:
        return None

    adx_weak = ctx["adx"] < 15 if math.isfinite(ctx["adx"]) else True
    contraction = session.tail(6)
    range_contraction = (contraction["high"].max() - contraction["low"].min()) < atr_value * 0.9
    volume_light = math.isfinite(ctx["volume_median"]) and latest["volume"] <= ctx["volume_median"]

    if distance < 0:  # price below VWAP -> look for mean reversion long
        direction = "long"
        entry = price
        stop = min(float(contraction["low"].min()), price - atr_value * 0.35)
        target_primary = ctx["vwap"]
        target_secondary = price + atr_value * 0.7
    else:
        direction = "short"
        entry = price
        stop = max(float(contraction["high"].max()), price + atr_value * 0.35)
        target_primary = ctx["vwap"]
        target_secondary = price - atr_value * 0.7

    plan = _build_plan(
        direction,
        entry,
        stop,
        [target_primary, target_secondary],
        atr_value=atr_value,
        notes=f"VWAP {ctx['vwap']:.2f}; extension {extension:.2f} ({extension/atr_value:.2f}× ATR)",
        conditions=[adx_weak, range_contraction, volume_light],
    )
    if plan is None:
        return None

    features = {
        "atr": atr_value,
        "adx": ctx["adx"],
        "direction_bias": plan.direction,
        "session_phase": ctx["session_phase"],
        "vwap": ctx["vwap"],
        "extension_points": extension,
        "extension_atr_multiple": extension / atr_value,
        "plan_entry": plan.entry,
        "plan_stop": plan.stop,
        "plan_targets": plan.targets,
        "plan_confidence": plan.confidence,
        "plan_risk_reward": plan.risk_reward,
        "plan_notes": plan.notes,
    }

    return Signal(
        symbol=symbol,
        strategy_id=strategy.id,
        description=strategy.description,
        score=plan.confidence,
        features=features,
        options_rules=strategy.options_rules,
        plan=plan,
    )


STRATEGY_DETECTORS: Dict[str, Callable[[str, Strategy, Dict[str, Any]], Optional[Signal]]] = {
    "orb_retest": _detect_orb_retest,
    "power_hour_trend": _detect_power_hour_trend,
    "vwap_avwap": _detect_vwap_cluster,
    "gap_fill_open": _detect_gap_fill,
    "midday_mean_revert": _detect_midday_mean_revert,
}


async def scan_market(tickers: List[str], market_data: Dict[str, pd.DataFrame]) -> List[Signal]:
    """Scan the provided tickers for strategy setups using real indicator data."""

    strategies = load_strategies()
    signals: List[Signal] = []

    for symbol in tickers:
        raw_frame = market_data.get(symbol)
        if raw_frame is None or raw_frame.empty or len(raw_frame) < 30:
            continue

        try:
            frame = _prepare_symbol_frame(raw_frame)
        except ValueError:
            continue

        ctx = _build_context(frame)

        for strategy in strategies:
            detector = STRATEGY_DETECTORS.get(strategy.id)
            if detector is None:
                continue
            signal = detector(symbol, strategy, ctx)
            if signal is None:
                continue
            signals.append(signal)

    signals.sort(key=lambda sig: sig.score, reverse=True)
    return signals
