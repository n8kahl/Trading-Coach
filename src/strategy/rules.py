"""Deterministic strategy rules that can leverage multi-timeframe context."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Callable, List, Optional

import pandas as pd
from zoneinfo import ZoneInfo

from ..features.mtf import MTFBundle, TFState

EST = ZoneInfo("America/New_York")


@dataclass
class RuleContext:
    symbol: str
    direction: str
    timestamp: Optional[datetime]
    mtf: Optional[MTFBundle]
    htf_levels: Optional["HTFLevels"]
    price: Optional[float]
    vwap: Optional[float]
    opening_range_high: Optional[float]
    opening_range_low: Optional[float]
    bars_5m: Optional[pd.DataFrame]
    bars_15m: Optional[pd.DataFrame]
    bars_60m: Optional[pd.DataFrame]


@dataclass
class RuleResult:
    id: str
    name: str
    base_score: float
    reasons: List[str]
    waiting_for: str
    badges: List[str]


def _get_state(bundle: Optional[MTFBundle], tf: str) -> Optional[TFState]:
    if bundle is None:
        return None
    return bundle.by_tf.get(tf) if bundle.by_tf else None


def _session_time(timestamp: Optional[datetime]) -> Optional[datetime]:
    if timestamp is None:
        return None
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=ZoneInfo("UTC"))
    return timestamp.astimezone(EST)


def _in_power_hour(timestamp: Optional[datetime]) -> bool:
    est_time = _session_time(timestamp)
    if est_time is None:
        return False
    return 15 <= est_time.hour < 16


def _has_pullback_signal(bars: Optional[pd.DataFrame], direction: str) -> bool:
    if bars is None or bars.empty or len(bars) < 4:
        return False
    frame = bars.tail(5)
    closes = frame["close"]
    highs = frame["high"]
    lows = frame["low"]
    if direction == "long":
        # Require a recent down bar followed by higher low and higher close
        if closes.iloc[-2] >= closes.iloc[-3]:
            return False
        return lows.iloc[-1] > lows.iloc[-2] and closes.iloc[-1] > closes.iloc[-2]
    else:
        if closes.iloc[-2] <= closes.iloc[-3]:
            return False
        return highs.iloc[-1] < highs.iloc[-2] and closes.iloc[-1] < closes.iloc[-2]


def _rule_power_hour_long(ctx: RuleContext) -> Optional[RuleResult]:
    if ctx.direction != "long" or not _in_power_hour(ctx.timestamp):
        return None
    state_5m = _get_state(ctx.mtf, "5m")
    state_15m = _get_state(ctx.mtf, "15m")
    if not state_5m or not state_15m or not (state_5m.ema_up and state_15m.ema_up):
        return None
    base_score = 0.62
    reasons = ["Power hour window", "5m/15m EMAs stacked up"]
    state_60m = _get_state(ctx.mtf, "60m")
    state_d = _get_state(ctx.mtf, "D")
    if state_60m and state_60m.ema_up:
        base_score += 0.05
        reasons.append("60m trend supportive")
    if state_d and state_d.ema_up:
        base_score += 0.05
        reasons.append("Daily trend supportive")
    return RuleResult(
        id="power_hour_trend",
        name="Power Hour Continuation",
        base_score=base_score,
        reasons=reasons,
        waiting_for="5m close above VWAP + acceptance over ORH",
        badges=["Power Hour"],
    )


def _rule_power_hour_short(ctx: RuleContext) -> Optional[RuleResult]:
    if ctx.direction != "short" or not _in_power_hour(ctx.timestamp):
        return None
    state_5m = _get_state(ctx.mtf, "5m")
    state_15m = _get_state(ctx.mtf, "15m")
    if not state_5m or not state_15m or not (state_5m.ema_down and state_15m.ema_down):
        return None
    base_score = 0.62
    reasons = ["Power hour window", "5m/15m EMAs stacked down"]
    state_60m = _get_state(ctx.mtf, "60m")
    state_d = _get_state(ctx.mtf, "D")
    if state_60m and state_60m.ema_down:
        base_score += 0.05
        reasons.append("60m trend supportive")
    if state_d and state_d.ema_down:
        base_score += 0.05
        reasons.append("Daily trend supportive")
    return RuleResult(
        id="power_hour_trend",
        name="Power Hour Continuation",
        base_score=base_score,
        reasons=reasons,
        waiting_for="5m close below VWAP + acceptance under ORL",
        badges=["Power Hour"],
    )


def _rule_vwap_reclaim_long(ctx: RuleContext) -> Optional[RuleResult]:
    if ctx.direction != "long":
        return None
    state_5m = _get_state(ctx.mtf, "5m")
    if not state_5m or not state_5m.ema_up:
        return None
    if state_5m.vwap_rel not in {"above", "near"}:
        return None
    base_score = 0.56
    reasons = ["VWAP reclaim with 5m trend up"]
    if ctx.vwap is not None and ctx.price is not None and ctx.price > ctx.vwap:
        reasons.append("Price already holding above VWAP")
    if ctx.opening_range_high and ctx.price and ctx.price < ctx.opening_range_high:
        reasons.append("Targeting ORH reclaim")
    if ctx.mtf and ctx.mtf.bias_htf == "short":
        base_score -= 0.08
        reasons.append("HTF bias opposing — penalty")
    return RuleResult(
        id="vwap_reclaim_long",
        name="VWAP Reclaim Continuation",
        base_score=base_score,
        reasons=reasons,
        waiting_for="Close above VWAP and hold 2 bars",
        badges=["VWAP"],
    )


def _rule_vwap_reclaim_short(ctx: RuleContext) -> Optional[RuleResult]:
    if ctx.direction != "short":
        return None
    state_5m = _get_state(ctx.mtf, "5m")
    if not state_5m or not state_5m.ema_down:
        return None
    if state_5m.vwap_rel not in {"below", "near"}:
        return None
    base_score = 0.56
    reasons = ["VWAP rejection with 5m trend down"]
    if ctx.vwap is not None and ctx.price is not None and ctx.price < ctx.vwap:
        reasons.append("Price already holding below VWAP")
    if ctx.opening_range_low and ctx.price and ctx.price > ctx.opening_range_low:
        reasons.append("Targeting ORL breakdown")
    if ctx.mtf and ctx.mtf.bias_htf == "long":
        base_score -= 0.08
        reasons.append("HTF bias opposing — penalty")
    return RuleResult(
        id="vwap_reclaim_short",
        name="VWAP Rejection Fade",
        base_score=base_score,
        reasons=reasons,
        waiting_for="Close below VWAP and hold 2 bars",
        badges=["VWAP"],
    )


def _rule_htf_pullback_long(ctx: RuleContext) -> Optional[RuleResult]:
    if ctx.direction != "long":
        return None
    state_60m = _get_state(ctx.mtf, "60m")
    state_d = _get_state(ctx.mtf, "D")
    if not state_60m or not state_d or not (state_60m.ema_up and state_d.ema_up):
        return None
    state_5m = _get_state(ctx.mtf, "5m")
    if not state_5m or state_5m.adx_slope is None or state_5m.adx_slope <= 0:
        return None
    if not _has_pullback_signal(ctx.bars_5m, "long"):
        return None
    base_score = 0.6
    reasons = ["HTF trend up", "ADX slope rising"]
    if ctx.htf_levels and ctx.htf_levels.pdh and ctx.price and ctx.price < ctx.htf_levels.pdh:
        reasons.append("Room back to PDH")
    return RuleResult(
        id="htf_pullback_long",
        name="HTF Pullback Continuation",
        base_score=base_score,
        reasons=reasons,
        waiting_for="5m HL + reclaim 9EMA",
        badges=["Trend"],
    )


def _rule_htf_pullback_short(ctx: RuleContext) -> Optional[RuleResult]:
    if ctx.direction != "short":
        return None
    state_60m = _get_state(ctx.mtf, "60m")
    state_d = _get_state(ctx.mtf, "D")
    if not state_60m or not state_d or not (state_60m.ema_down and state_d.ema_down):
        return None
    state_5m = _get_state(ctx.mtf, "5m")
    if not state_5m or state_5m.adx_slope is None or state_5m.adx_slope >= 0:
        return None
    if not _has_pullback_signal(ctx.bars_5m, "short"):
        return None
    base_score = 0.6
    reasons = ["HTF trend down", "ADX slope weakening"]
    if ctx.htf_levels and ctx.htf_levels.pdl and ctx.price and ctx.price > ctx.htf_levels.pdl:
        reasons.append("Room back to PDL")
    return RuleResult(
        id="htf_pullback_short",
        name="HTF Pullback Breakdown",
        base_score=base_score,
        reasons=reasons,
        waiting_for="5m LH + reject 9EMA",
        badges=["Trend"],
    )


def candidate_rules(direction: str) -> List[Callable[[RuleContext], Optional[RuleResult]]]:
    """Return candidate rule evaluators for the desired trade direction."""
    if direction == "long":
        return [
            _rule_power_hour_long,
            _rule_vwap_reclaim_long,
            _rule_htf_pullback_long,
        ]
    if direction == "short":
        return [
            _rule_power_hour_short,
            _rule_vwap_reclaim_short,
            _rule_htf_pullback_short,
        ]
    return []


__all__ = ["RuleContext", "RuleResult", "candidate_rules"]

