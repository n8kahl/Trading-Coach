"""Execution profile utilities for precision plan refinement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

from ...strategy_library import normalize_style_input
from ...scanner import Plan, rr


@dataclass(slots=True)
class ExecutionProfile:
    style: str
    default_entry: str
    min_stop_atr: float
    max_stop_atr: float
    min_stop_ticks: int
    entry_gap_ticks: int
    reclaim_offset_ticks: int
    stop_level_padding: int
    stop_vwap_padding: int
    tp_fracs: Tuple[float, float, float]
    tp_rr_floor: Tuple[float, float, float]
    confidence_boost: float = 0.04
    rr_confidence_threshold: float = 0.05
    rr_penalty_threshold: float = 0.15
    confidence_penalty: float = 0.03
    atr_floor_factor: float = 0.7


@dataclass(slots=True)
class ExecutionContext:
    symbol: str
    style: str
    direction: str
    price: float
    key_levels: Dict[str, float]
    atr14: Optional[float]
    expected_move: Optional[float]
    vwap: Optional[float]
    ema_stack: Optional[str]
    session_phase: Optional[str]
    minutes_to_close: Optional[int]
    data_mode: Optional[str]


@dataclass(slots=True)
class ExecutionAdjustment:
    entry_type: Optional[str] = None
    entry_level: Optional[Tuple[str, float]] = None
    alt_entry: Optional[Dict[str, float]] = None
    note: Optional[str] = None
    rr_before: Optional[float] = None
    rr_after: Optional[float] = None
    confidence_delta: float = 0.0

    def feature_updates(self) -> Dict[str, object]:
        payload: Dict[str, object] = {}
        if self.entry_type:
            payload["plan_entry_type"] = self.entry_type
        if self.entry_level:
            payload["plan_entry_anchor"] = {"label": self.entry_level[0], "value": self.entry_level[1]}
        if self.alt_entry:
            payload["plan_alt_entry"] = self.alt_entry
        if self.rr_before is not None and self.rr_after is not None:
            payload["plan_rr_before"] = self.rr_before
            payload["plan_rr_after"] = self.rr_after
        if self.confidence_delta:
            payload["plan_confidence_adjustment"] = self.confidence_delta
        return payload


STYLE_PROFILES: Dict[str, ExecutionProfile] = {
    "scalp": ExecutionProfile(
        style="scalp",
        default_entry="reclaim",
        min_stop_atr=0.35,
        max_stop_atr=1.10,
        min_stop_ticks=3,
        entry_gap_ticks=2,
        reclaim_offset_ticks=2,
        stop_level_padding=3,
        stop_vwap_padding=2,
        tp_fracs=(0.25, 0.5, 0.75),
        tp_rr_floor=(1.2, 1.6, 2.1),
    ),
    "intraday": ExecutionProfile(
        style="intraday",
        default_entry="retest",
        min_stop_atr=0.45,
        max_stop_atr=1.30,
        min_stop_ticks=4,
        entry_gap_ticks=3,
        reclaim_offset_ticks=3,
        stop_level_padding=4,
        stop_vwap_padding=3,
        tp_fracs=(0.33, 0.66, 1.0),
        tp_rr_floor=(1.4, 1.8, 2.3),
    ),
    "swing": ExecutionProfile(
        style="swing",
        default_entry="daily_retest",
        min_stop_atr=0.85,
        max_stop_atr=1.80,
        min_stop_ticks=6,
        entry_gap_ticks=4,
        reclaim_offset_ticks=4,
        stop_level_padding=6,
        stop_vwap_padding=4,
        tp_fracs=(0.5, 1.0, 1.5),
        tp_rr_floor=(1.5, 2.0, 2.6),
        confidence_boost=0.03,
        rr_confidence_threshold=0.08,
    ),
    "leap": ExecutionProfile(
        style="leap",
        default_entry="htf_close",
        min_stop_atr=1.20,
        max_stop_atr=2.40,
        min_stop_ticks=8,
        entry_gap_ticks=5,
        reclaim_offset_ticks=5,
        stop_level_padding=8,
        stop_vwap_padding=5,
        tp_fracs=(1.0, 1.6, 2.4),
        tp_rr_floor=(1.7, 2.3, 3.0),
        confidence_boost=0.02,
        rr_confidence_threshold=0.12,
    ),
}


def _normalize_style(style: Optional[str]) -> str:
    return normalize_style_input(style) or "intraday"


def _infer_tick_size(price: float) -> float:
    if price >= 500:
        return 0.1
    if price >= 200:
        return 0.05
    if price >= 50:
        return 0.02
    if price >= 10:
        return 0.01
    if price >= 1:
        return 0.005
    return 0.001


LEVEL_PRIORITY = {
    "opening_range_high": 0,
    "opening_range_low": 0,
    "session_high": 1,
    "session_low": 1,
    "prev_high": 2,
    "prev_low": 2,
    "prev_close": 2,
    "vwap": 3,
}


def _nearest_level(direction: str, entry: float, levels: Dict[str, float]) -> Optional[Tuple[str, float, float]]:
    best_name: Optional[str] = None
    best_value: Optional[float] = None
    best_diff = float("inf")
    best_priority = float("inf")
    for name, value in (levels or {}).items():
        try:
            level = float(value)
        except (TypeError, ValueError):
            continue
        diff = entry - level if direction == "long" else level - entry
        if diff <= 0:
            continue
        priority = LEVEL_PRIORITY.get(name, 5)
        if priority < best_priority or (priority == best_priority and diff < best_diff):
            best_name = name
            best_value = level
            best_diff = diff
            best_priority = priority
    if best_name is None or best_value is None:
        return None
    return best_name, best_value, best_diff


def _expected_move(entry: float, stop: float, ctx: ExecutionContext, profile: ExecutionProfile) -> float:
    reff = ctx.expected_move
    if reff and reff > 0:
        return float(reff)
    atr = ctx.atr14 if ctx.atr14 and ctx.atr14 > 0 else abs(entry - stop)
    if atr and atr > 0:
        return atr * (1.8 if profile.style in {"intraday", "swing"} else 1.5)
    return max(abs(entry - stop) * 2, 0.5)


def _apply_runner_adjustment(plan: Plan, new_entry: float) -> None:
    if not plan.runner or not isinstance(plan.runner, dict):
        return
    runner = dict(plan.runner)
    anchor = runner.get("anchor")
    if isinstance(anchor, (int, float)):
        delta = anchor - plan.entry
        runner["anchor"] = round(new_entry + delta, 4)
    plan.runner = runner


def _clamp_stop(entry: float, candidate: float, direction: str, min_distance: float, max_distance: float) -> float:
    if direction == "long":
        distance = entry - candidate
        if distance < min_distance:
            return entry - min_distance
        if distance > max_distance:
            return entry - max_distance
        return candidate
    distance = candidate - entry
    if distance < min_distance:
        return entry + min_distance
    if distance > max_distance:
        return entry + max_distance
    return candidate


def _tp_from_profile(
    entry: float,
    direction: str,
    profile: ExecutionProfile,
    expected_move: float,
    risk_distance: float,
    tick: float,
) -> List[float]:
    targets: List[float] = []
    for frac, rr_floor in zip(profile.tp_fracs, profile.tp_rr_floor):
        base_offset = frac * expected_move
        floor_offset = rr_floor * risk_distance
        min_offset = tick * profile.min_stop_ticks
        offset = max(base_offset, floor_offset, min_offset)
        if direction == "long":
            targets.append(entry + offset)
        else:
            targets.append(entry - offset)
    # Ensure monotonic ordering
    if direction == "long":
        targets = sorted({round(t, 4) for t in targets})
    else:
        targets = sorted({round(t, 4) for t in targets}, reverse=True)
    return targets


def get_execution_profile(style: Optional[str]) -> ExecutionProfile:
    style_key = _normalize_style(style)
    return STYLE_PROFILES.get(style_key, STYLE_PROFILES["intraday"])


def refine_plan(plan: Plan, ctx: ExecutionContext) -> Tuple[Plan, ExecutionAdjustment]:
    profile = get_execution_profile(ctx.style)
    style_key = profile.style
    if style_key == "scalp":
        return _refine_scalp(plan, ctx, profile)
    if style_key == "intraday":
        return _refine_intraday(plan, ctx, profile)
    if style_key == "swing":
        return _refine_swing(plan, ctx, profile)
    if style_key == "leap":
        return _refine_leap(plan, ctx, profile)
    return plan, ExecutionAdjustment()


def _refine_scalp(plan: Plan, ctx: ExecutionContext, profile: ExecutionProfile) -> Tuple[Plan, ExecutionAdjustment]:
    if not ctx.key_levels:
        return plan, ExecutionAdjustment()
    nearest = _nearest_level(plan.direction, plan.entry, ctx.key_levels)
    if not nearest:
        return plan, ExecutionAdjustment()
    level_name, level_value, diff = nearest
    tick = _infer_tick_size(ctx.price)
    atr = ctx.atr14 or abs(plan.entry - plan.stop)
    atr = atr if atr and atr > 0 else abs(plan.entry - plan.stop)
    atr = atr if atr and atr > 0 else tick * profile.min_stop_ticks
    min_stop = max(profile.min_stop_atr * atr, tick * profile.min_stop_ticks)
    max_stop = max(profile.max_stop_atr * atr, min_stop * 1.4)
    entry_reclaim = level_value + tick * profile.reclaim_offset_ticks if plan.direction == "long" else level_value - tick * profile.reclaim_offset_ticks
    entry_breakout = plan.entry
    use_reclaim = diff >= tick * profile.entry_gap_ticks
    entry_new = entry_reclaim if use_reclaim else plan.entry
    entry_type = "reclaim" if use_reclaim else "breakout"
    stop_candidate = plan.stop
    level_padding = tick * profile.stop_level_padding
    vwap_padding = tick * profile.stop_vwap_padding
    if plan.direction == "long":
        stop_candidate = min(stop_candidate, entry_new - min_stop)
        stop_candidate = min(stop_candidate, level_value - level_padding)
        if ctx.vwap and ctx.vwap < entry_new:
            stop_candidate = min(stop_candidate, ctx.vwap - vwap_padding)
        stop_new = _clamp_stop(entry_new, stop_candidate, "long", min_stop, max_stop)
    else:
        stop_candidate = max(stop_candidate, entry_new + min_stop)
        stop_candidate = max(stop_candidate, level_value + level_padding)
        if ctx.vwap and ctx.vwap > entry_new:
            stop_candidate = max(stop_candidate, ctx.vwap + vwap_padding)
        stop_new = _clamp_stop(entry_new, stop_candidate, "short", min_stop, max_stop)
    expected_move = _expected_move(entry_new, stop_new, ctx, profile)
    risk_distance = (entry_new - stop_new) if plan.direction == "long" else (stop_new - entry_new)
    targets_new = _tp_from_profile(entry_new, plan.direction, profile, expected_move, risk_distance, tick)
    if len(plan.targets) >= len(targets_new):
        # preserve length
        targets_new = targets_new[: len(plan.targets)]
    rr_before = plan.risk_reward
    rr_after = rr(entry_new, stop_new, targets_new[0], plan.direction) if targets_new else rr_before
    confidence_delta = 0.0
    if rr_after > rr_before + profile.rr_confidence_threshold:
        new_conf = min(plan.confidence + profile.confidence_boost, 0.99)
        confidence_delta = new_conf - plan.confidence
        plan.confidence = new_conf
    elif rr_after + profile.rr_penalty_threshold < rr_before:
        new_conf = max(plan.confidence - profile.confidence_penalty, 0.05)
        confidence_delta = new_conf - plan.confidence
        plan.confidence = new_conf
    plan.entry = round(entry_new, 4)
    plan.stop = round(stop_new, 4)
    plan.targets = [round(val, 4) for val in targets_new]
    plan.risk_reward = round(rr_after, 3)
    plan.atr = ctx.atr14
    _apply_runner_adjustment(plan, plan.entry)
    note = f"Sniper entry ({entry_type}) anchored to {level_name.replace('_', ' ')} {level_value:.2f}."
    if plan.notes:
        if note not in plan.notes:
            plan.notes = f"{plan.notes} {note}"
    else:
        plan.notes = note
    alt_entry = None
    if entry_type == "reclaim":
        alt_entry = {"type": "breakout", "level": round(entry_breakout, 4)}
    else:
        alt_entry = {"type": "reclaim", "level": round(entry_reclaim, 4)}
    adjustment = ExecutionAdjustment(
        entry_type=entry_type,
        entry_level=(level_name, round(level_value, 4)),
        alt_entry=alt_entry,
        note=note,
        rr_before=round(rr_before, 3) if rr_before is not None else None,
        rr_after=round(rr_after, 3),
        confidence_delta=confidence_delta,
    )
    return plan, adjustment


def _refine_intraday(plan: Plan, ctx: ExecutionContext, profile: ExecutionProfile) -> Tuple[Plan, ExecutionAdjustment]:
    tick = _infer_tick_size(ctx.price)
    atr = ctx.atr14 or abs(plan.entry - plan.stop)
    atr = atr if atr and atr > 0 else tick * profile.min_stop_ticks
    min_stop = max(profile.min_stop_atr * atr, tick * profile.min_stop_ticks)
    max_stop = max(profile.max_stop_atr * atr, min_stop * 1.5)
    stop_candidate = plan.stop
    if plan.direction == "long":
        stop_candidate = min(stop_candidate, plan.entry - min_stop)
    else:
        stop_candidate = max(stop_candidate, plan.entry + min_stop)
    stop_new = _clamp_stop(plan.entry, stop_candidate, plan.direction, min_stop, max_stop)
    expected_move = _expected_move(plan.entry, stop_new, ctx, profile)
    risk_distance = (plan.entry - stop_new) if plan.direction == "long" else (stop_new - plan.entry)
    targets_new = _tp_from_profile(plan.entry, plan.direction, profile, expected_move, risk_distance, tick)
    if len(plan.targets) >= len(targets_new):
        targets_new = targets_new[: len(plan.targets)]
    rr_before = plan.risk_reward
    rr_after = rr(plan.entry, stop_new, targets_new[0], plan.direction) if targets_new else rr_before
    confidence_delta = 0.0
    if rr_after > rr_before + profile.rr_confidence_threshold:
        new_conf = min(plan.confidence + profile.confidence_boost, 0.98)
        confidence_delta = new_conf - plan.confidence
        plan.confidence = new_conf
    plan.stop = round(stop_new, 4)
    plan.targets = [round(val, 4) for val in targets_new]
    plan.risk_reward = round(rr_after, 3)
    _apply_runner_adjustment(plan, plan.entry)
    note = "Stop and target ladder calibrated to intraday volatility."
    if plan.notes:
        if note not in plan.notes:
            plan.notes = f"{plan.notes} {note}"
    else:
        plan.notes = note
    adjustment = ExecutionAdjustment(
        entry_type="retest",
        entry_level=None,
        alt_entry=None,
        note=note,
        rr_before=round(rr_before, 3) if rr_before is not None else None,
        rr_after=round(rr_after, 3),
        confidence_delta=confidence_delta,
    )
    return plan, adjustment


def _refine_swing(plan: Plan, ctx: ExecutionContext, profile: ExecutionProfile) -> Tuple[Plan, ExecutionAdjustment]:
    tick = _infer_tick_size(ctx.price)
    atr = ctx.atr14 or abs(plan.entry - plan.stop)
    min_stop = max(profile.min_stop_atr * atr, tick * profile.min_stop_ticks)
    max_stop = max(profile.max_stop_atr * atr, min_stop * 1.8)
    stop_new = _clamp_stop(plan.entry, plan.stop, plan.direction, min_stop, max_stop)
    plan.stop = round(stop_new, 4)
    rr_before = plan.risk_reward
    rr_after = rr(plan.entry, plan.stop, plan.targets[0], plan.direction)
    plan.risk_reward = round(rr_after, 3)
    note = "Swing profile applied: stop sized off daily structure."
    if plan.notes:
        if note not in plan.notes:
            plan.notes = f"{plan.notes} {note}"
    else:
        plan.notes = note
    adjustment = ExecutionAdjustment(
        entry_type="daily_retest",
        entry_level=None,
        rr_before=round(rr_before, 3),
        rr_after=round(rr_after, 3),
    )
    return plan, adjustment


def _refine_leap(plan: Plan, ctx: ExecutionContext, profile: ExecutionProfile) -> Tuple[Plan, ExecutionAdjustment]:
    tick = _infer_tick_size(ctx.price)
    atr = ctx.atr14 or abs(plan.entry - plan.stop)
    min_stop = max(profile.min_stop_atr * atr, tick * profile.min_stop_ticks)
    stop_new = _clamp_stop(plan.entry, plan.stop, plan.direction, min_stop, min_stop * 2.5)
    plan.stop = round(stop_new, 4)
    rr_before = plan.risk_reward
    rr_after = rr(plan.entry, plan.stop, plan.targets[0], plan.direction)
    plan.risk_reward = round(rr_after, 3)
    note = "Leap plan emphasises time-in-market; manage risk via size and hedges."
    if plan.notes:
        if note not in plan.notes:
            plan.notes = f"{plan.notes} {note}"
    else:
        plan.notes = note
    adjustment = ExecutionAdjustment(
        entry_type="htf_close",
        note=note,
        rr_before=round(rr_before, 3),
        rr_after=round(rr_after, 3),
    )
    return plan, adjustment


__all__ = [
    "ExecutionProfile",
    "ExecutionContext",
    "ExecutionAdjustment",
    "get_execution_profile",
    "refine_plan",
]
