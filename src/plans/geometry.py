"""Deterministic TP/SL/runner engine for plan construction."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, time
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

from ..levels.snapper import Level, SnapContext, collect_levels, snap_prices


@dataclass(frozen=True)
class GeometryConfig:
    """Parameter grid per style/strategy."""

    k_stop: float
    rr_min: float
    atr_ladder: Tuple[float, ...]
    em_fraction_max: float
    snap_window_atr_mult: float
    snap_window_pct: float
    runner_fraction: Tuple[float, float]
    runner_trail_mult: float
    runner_trail_step: float
    runner_em_fraction: float


@dataclass
class TargetMeta:
    price: float
    distance: float
    rr_multiple: float
    prob_touch: float
    em_fraction: Optional[float]
    mfe_quantile: Optional[float]
    reason: Optional[str] = None
    em_capped: bool = False


@dataclass
class StopResult:
    price: float
    structural: float
    volatility: float
    snapped: Optional[str]
    rr_min: float


@dataclass
class RunnerPolicy:
    fraction: float
    atr_trail_mult: float
    atr_trail_step: float
    em_fraction_cap: float
    notes: List[str]


@dataclass
class PlanGeometry:
    entry: float
    stop: StopResult
    targets: List[TargetMeta]
    runner: RunnerPolicy
    em_day: float
    em_used: bool
    snap_trace: List[str]
    ratr: float


_STYLE_DEFAULTS: Mapping[str, GeometryConfig] = {
    "scalp": GeometryConfig(
        k_stop=0.6,
        rr_min=1.4,
        atr_ladder=(0.8, 1.2, 1.8),
        em_fraction_max=0.35,
        snap_window_atr_mult=0.15,
        snap_window_pct=0.0008,
        runner_fraction=(0.10, 0.15),
        runner_trail_mult=0.8,
        runner_trail_step=0.4,
        runner_em_fraction=0.45,
    ),
    "intraday": GeometryConfig(
        k_stop=0.9,
        rr_min=1.6,
        atr_ladder=(1.0, 1.6, 2.4),
        em_fraction_max=0.60,
        snap_window_atr_mult=0.20,
        snap_window_pct=0.001,
        runner_fraction=(0.15, 0.25),
        runner_trail_mult=1.0,
        runner_trail_step=0.5,
        runner_em_fraction=0.55,
    ),
    "swing": GeometryConfig(
        k_stop=1.2,
        rr_min=2.0,
        atr_ladder=(0.6, 1.0, 1.6),
        em_fraction_max=0.90,
        snap_window_atr_mult=0.25,
        snap_window_pct=0.0015,
        runner_fraction=(0.20, 0.30),
        runner_trail_mult=1.2,
        runner_trail_step=0.6,
        runner_em_fraction=0.75,
    ),
    "leaps": GeometryConfig(
        k_stop=1.5,
        rr_min=2.0,
        atr_ladder=(0.4, 0.8, 1.2),
        em_fraction_max=1.00,
        snap_window_atr_mult=0.30,
        snap_window_pct=0.0020,
        runner_fraction=(0.20, 0.30),
        runner_trail_mult=1.3,
        runner_trail_step=0.7,
        runner_em_fraction=0.85,
    ),
}

_STRATEGY_MODIFIERS: Mapping[str, Mapping[str, float | int | bool]] = {
    "power_hour": {
        "tod_boost": 1.2,
        "k_stop_scale": 0.9,
        "snap_scale": 0.8,
        "target_count": 2,
        "runner_fraction_delta": 0.1,
        "runner_trail_scale": 0.8,
    },
    "gap_fill": {
        "tod_boost": 1.1,
        "snap_scale": 0.9,
        "prefer_gap_fill": True,
    },
}


def _style_key(style: Optional[str]) -> str:
    token = (style or "intraday").strip().lower()
    if token not in _STYLE_DEFAULTS:
        return "intraday"
    return token


def _strategy_key(strategy: Optional[str]) -> Optional[str]:
    token = (strategy or "").strip().lower()
    if not token:
        return None
    for key in _STRATEGY_MODIFIERS:
        if key in token:
            return key
    return None


def _strategy_modifiers(strategy: Optional[str]) -> Mapping[str, float | int | bool]:
    key = _strategy_key(strategy)
    if key is None:
        return {}
    return _STRATEGY_MODIFIERS.get(key, {})


def compute_expected_move(symbol_ctx: Mapping[str, float | None]) -> float:
    """Derive expected move with IV preference."""

    iv_move = symbol_ctx.get("iv_expected_move")
    if isinstance(iv_move, (int, float)) and math.isfinite(iv_move) and iv_move > 0:
        return float(iv_move)
    atr_daily = symbol_ctx.get("atr_daily")
    calibration = symbol_ctx.get("atr_to_em_coeff") or 1.7
    if isinstance(atr_daily, (int, float)) and math.isfinite(atr_daily) and atr_daily > 0:
        return float(atr_daily) * float(calibration)
    alt = symbol_ctx.get("fallback_em")
    if isinstance(alt, (int, float)) and math.isfinite(alt) and alt > 0:
        return float(alt)
    return 0.0


def remaining_atr(em_day: float, realized_range: float) -> float:
    if not isinstance(em_day, (int, float)) or not math.isfinite(em_day):
        return 0.0
    realized = float(realized_range or 0.0)
    if realized < 0:
        realized = 0.0
    return max(em_day - realized, 0.0)


def tod_multiplier(as_of: Optional[datetime]) -> float:
    if as_of is None:
        return 1.0
    tod_curve = {
        time(9, 30): 1.3,
        time(10, 0): 1.15,
        time(11, 0): 0.9,
        time(12, 0): 0.7,
        time(13, 0): 0.75,
        time(14, 0): 0.95,
        time(15, 0): 1.1,
        time(15, 30): 1.2,
    }
    moment = as_of.time()
    match_key = max((key for key in tod_curve if moment >= key), default=None)
    if match_key is None:
        return 1.0
    return tod_curve.get(match_key, 1.0)


def _level_by_priority(levels: Mapping[str, float], side: str) -> List[Tuple[str, float]]:
    priority = [
        "gap_fill",
        "pdh",
        "pdl",
        "pdc",
        "orh",
        "orl",
        "vah",
        "val",
        "poc",
        "swing_high",
        "swing_low",
        "vwap",
        "avwap",
    ]
    ranked = []
    side = (side or "").lower()
    for key in priority:
        value = levels.get(key)
        if isinstance(value, (int, float)) and math.isfinite(value):
            ranked.append((key, float(value)))
    if side == "long":
        ranked.sort(key=lambda item: item[1], reverse=True)
    else:
        ranked.sort(key=lambda item: item[1])
    return ranked


def derive_stop(
    entry: float,
    side: str,
    levels: Mapping[str, float],
    atr_tf: float,
    style: str,
    strategy: Optional[str],
) -> StopResult:
    style_key = _style_key(style)
    config = _STYLE_DEFAULTS[style_key]
    base_k = config.k_stop
    modifiers = _strategy_modifiers(strategy)
    scale = modifiers.get("k_stop_scale")
    if isinstance(scale, (int, float)) and math.isfinite(scale) and scale > 0:
        base_k = max(0.2, base_k * float(scale))
    atr_tf = max(atr_tf or 0.0, 0.01)
    vol_stop = entry - base_k * atr_tf if side == "long" else entry + base_k * atr_tf
    structural_candidate = None
    for key, level in _level_by_priority(levels, side):
        if side == "long" and level < entry:
            structural_candidate = level
            break
        if side == "short" and level > entry:
            structural_candidate = level
            break
    if structural_candidate is None:
        structural_candidate = vol_stop
    if side == "long":
        wider = min(structural_candidate, vol_stop)
    else:
        wider = max(structural_candidate, vol_stop)
    snapped = None
    return StopResult(
        price=round(wider, 2),
        structural=round(structural_candidate, 2),
        volatility=round(vol_stop, 2),
        snapped=snapped,
        rr_min=config.rr_min,
    )


def _probability_from_mfe_quantile(idx: int) -> float:
    quantiles = [0.55, 0.68, 0.82, 0.9]
    return quantiles[min(idx, len(quantiles) - 1)]


def _quantize_stop_toward_entry(price: float, side: str, entry: float) -> float:
    """
    Quantize the stop price to two decimals while moving it toward the entry.
    Ensures we do not cross the entry boundary and bias rounding to preserve rr tightening.
    """
    side_token = (side or "").lower()
    target = float(price)
    if side_token == "long":
        max_allowed = entry - 0.01
        target = min(target, max_allowed)
        cents = target * 100.0
        quantized = math.ceil(cents - 1e-9) / 100.0
        if quantized > max_allowed:
            quantized = max_allowed
    elif side_token == "short":
        min_allowed = entry + 0.01
        target = max(target, min_allowed)
        cents = target * 100.0
        quantized = math.floor(cents + 1e-9) / 100.0
        if quantized < min_allowed:
            quantized = min_allowed
    else:
        quantized = round(target, 2)
    return round(quantized, 2)


def _enforce_monotonic_targets(
    entry: float,
    side: str,
    targets: Sequence[TargetMeta],
    min_step: float = 0.01,
) -> None:
    min_step = max(0.01, float(min_step))
    prev = entry
    side_token = (side or "").lower()
    for meta in targets:
        if side_token == "long":
            threshold = prev + min_step
            if meta.price <= threshold - 1e-9:
                meta.price = round(threshold, 2)
        elif side_token == "short":
            threshold = prev - min_step
            if meta.price >= threshold + 1e-9:
                meta.price = round(threshold, 2)
        prev = meta.price
        meta.distance = round(abs(meta.price - entry), 2)


def build_targets(
    entry: float,
    side: str,
    atr_tf: float,
    em_day: float,
    ratr: float,
    tod_mult: float,
    style: str,
    strategy: Optional[str],
    levels: Optional[Mapping[str, float]],
) -> Tuple[List[TargetMeta], bool, Optional[float]]:
    style_key = _style_key(style)
    config = _STYLE_DEFAULTS[style_key]
    modifiers = _strategy_modifiers(strategy)
    multipliers = config.atr_ladder
    target_count = modifiers.get("target_count")
    if isinstance(target_count, int) and target_count > 0:
        multipliers = multipliers[: target_count]
    em_cap = config.em_fraction_max * em_day if em_day > 0 else float("inf")
    ratr_cap = ratr * max(tod_mult, 0.5) if ratr > 0 else float("inf")
    far_cap = min(em_cap, ratr_cap)
    targets: List[TargetMeta] = []
    em_used = False
    for idx, mult in enumerate(multipliers):
        distance = mult * atr_tf
        if side == "long":
            price = entry + distance
        else:
            price = entry - distance
        em_fraction = None
        if em_day > 0 and distance > 0:
            em_fraction = min(distance / em_day, 1.0)
        capped_price = price
        if far_cap != float("inf"):
            max_price = entry + far_cap if side == "long" else entry - far_cap
            if (side == "long" and price > max_price) or (side == "short" and price < max_price):
                capped_price = max_price
                em_used = True
        prob = _probability_from_mfe_quantile(idx)
        targets.append(
            TargetMeta(
                price=round(capped_price, 2),
                distance=round(abs(capped_price - entry), 2),
                rr_multiple=0.0,
                prob_touch=round(prob, 2),
                em_fraction=round(em_fraction, 2) if em_fraction is not None else None,
                mfe_quantile=None,
                em_capped=price != capped_price,
            )
        )
    mod_key = _strategy_key(strategy)
    if mod_key == "gap_fill" and levels:
        gap_price = levels.get("gap_fill")
        if isinstance(gap_price, (int, float)) and math.isfinite(gap_price) and targets:
            gap_val = float(gap_price)
            if (side == "long" and gap_val > entry) or (side == "short" and gap_val < entry):
                first = targets[0]
                first.price = round(gap_val, 2)
                first.distance = round(abs(gap_val - entry), 2)
                if em_day > 0:
                    first.em_fraction = round(first.distance / em_day, 2)
                first.reason = "gap_fill"
                # ensure subsequent targets remain monotonic
                for other in targets[1:]:
                    if side == "long" and other.price <= first.price:
                        other.price = round(first.price + max(0.1, atr_tf * 0.5), 2)
                    if side == "short" and other.price >= first.price:
                        other.price = round(first.price - max(0.1, atr_tf * 0.5), 2)
                    other.distance = round(abs(other.price - entry), 2)
    _enforce_monotonic_targets(entry, side, targets)
    capped_distance: Optional[float]
    if far_cap == float("inf"):
        capped_distance = None
    else:
        capped_distance = float(far_cap)
    return targets, em_used, capped_distance


def compute_runner_policy(style: str, strategy: Optional[str]) -> RunnerPolicy:
    style_key = _style_key(style)
    config = _STYLE_DEFAULTS[style_key]
    fraction = sum(config.runner_fraction) / 2.0
    notes: List[str] = []
    modifiers = _strategy_modifiers(strategy)
    delta = modifiers.get("runner_fraction_delta")
    if isinstance(delta, (int, float)) and math.isfinite(delta):
        fraction = min(0.6, max(0.05, fraction + float(delta)))
        notes.append("Runner fraction adjusted")
    trail_mult = config.runner_trail_mult
    trail_step = config.runner_trail_step
    trail_scale = modifiers.get("runner_trail_scale")
    if isinstance(trail_scale, (int, float)) and math.isfinite(trail_scale) and trail_scale > 0:
        trail_mult = max(0.1, trail_mult * float(trail_scale))
        notes.append("Runner trail tightened")
    return RunnerPolicy(
        fraction=round(fraction, 3),
        atr_trail_mult=round(trail_mult, 3),
        atr_trail_step=round(trail_step, 3),
        em_fraction_cap=config.runner_em_fraction,
        notes=notes,
    )


def validate_invariants(entry: float, stop: float, targets: Sequence[TargetMeta], side: str, rr_min: float) -> None:
    if not targets:
        raise ValueError("missing_targets")
    side_token = (side or "").lower()
    if side_token not in {"long", "short"}:
        raise ValueError("invalid_direction")
    if side_token == "long" and stop >= entry:
        raise ValueError("stop_not_below_entry")
    if side_token == "short" and stop <= entry:
        raise ValueError("stop_not_above_entry")
    rr_first = targets[0].rr_multiple
    if rr_first < rr_min:
        raise ValueError("rr_too_low")
    previous = entry
    for idx, target in enumerate(targets, start=1):
        price = target.price
        if side_token == "long":
            if price <= previous:
                raise ValueError(f"tp{idx}_not_above_previous")
        else:
            if price >= previous:
                raise ValueError(f"tp{idx}_not_below_previous")
        previous = price


def build_plan_geometry(
    *,
    entry: float,
    side: str,
    style: str,
    strategy: Optional[str],
    atr_tf: float,
    atr_daily: float,
    iv_expected_move: Optional[float],
    realized_range: float,
    levels: Mapping[str, float],
    timestamp: Optional[datetime],
    em_points: Optional[float] = None,
) -> PlanGeometry:
    symbol_ctx = {
        "iv_expected_move": iv_expected_move,
        "atr_daily": atr_daily,
        "atr_to_em_coeff": 1.7,
        "fallback_em": atr_daily * 1.7 if atr_daily else None,
    }
    if em_points is not None and isinstance(em_points, (int, float)) and math.isfinite(em_points) and em_points > 0:
        em_day = float(em_points)
    else:
        em_day = compute_expected_move(symbol_ctx)
    ratr = remaining_atr(em_day, realized_range)
    modifiers = _strategy_modifiers(strategy)
    tod_mult = tod_multiplier(timestamp)
    tod_boost = modifiers.get("tod_boost")
    if isinstance(tod_boost, (int, float)) and math.isfinite(tod_boost) and tod_boost > 0:
        tod_mult *= float(tod_boost)
    stop = derive_stop(entry, side, levels, atr_tf, style, strategy)
    targets, em_used, cap_distance = build_targets(
        entry, side, atr_tf, em_day, ratr, tod_mult, style, strategy, levels
    )
    runner = compute_runner_policy(style, strategy)
    style_key = _style_key(style)
    config = _STYLE_DEFAULTS[style_key]
    snap_window_atr = config.snap_window_atr_mult * atr_tf
    snap_window_pct = config.snap_window_pct
    snap_scale = modifiers.get("snap_scale")
    if isinstance(snap_scale, (int, float)) and math.isfinite(snap_scale) and snap_scale > 0:
        snap_window_atr *= float(snap_scale)
        snap_window_pct *= float(snap_scale)
    level_objs: List[Level] = collect_levels(levels)
    snap_ctx = SnapContext(
        side=side,
        style=style,
        strategy=strategy,
        window_atr=snap_window_atr,
        window_pct=snap_window_pct,
        rr_min=stop.rr_min,
        entry=entry,
    )
    original_stop_price = stop.price
    original_target_prices = [target.price for target in targets]
    target_prices = list(original_target_prices)
    snapped_stop, stop_reason, snapped_targets = snap_prices(
        entry,
        stop.price,
        target_prices,
        levels=level_objs,
        ctx=snap_ctx,
    )
    snap_trace: List[str] = []
    if stop_reason:
        stop.snapped = stop_reason
        snap_trace.append(f"stop:{original_stop_price:.2f}->{snapped_stop:.2f} via {stop_reason}")
    for idx, (meta, original_price, (snapped_price, reason)) in enumerate(
        zip(targets, original_target_prices, snapped_targets),
        start=1,
    ):
        if reason:
            meta.reason = reason
            snap_trace.append(f"tp{idx}:{original_price:.2f}->{snapped_price:.2f} via {reason}")
        meta.price = round(snapped_price, 2)
    stop.price = round(snapped_stop, 2)
    _enforce_monotonic_targets(entry, side, targets)
    risk = entry - stop.price if side == "long" else stop.price - entry
    if risk <= 0:
        risk = 0.001
    required_reward = stop.rr_min * risk
    rr_spacing: Optional[float] = None
    if cap_distance is not None and required_reward > cap_distance + 1e-6:
        rr_spacing = max(0.05, round(atr_tf * 0.5, 2))
    for meta in targets:
        meta.distance = round(abs(meta.price - entry), 2)
        reward = meta.price - entry if side == "long" else entry - meta.price
        meta.rr_multiple = round(max(reward, 0.0) / risk, 2)
        if em_day > 0:
            meta.em_fraction = round(meta.distance / em_day, 2)
    if targets and targets[0].rr_multiple < stop.rr_min:
        if side == "long":
            new_price = round(entry + required_reward, 2)
            targets[0].price = max(targets[0].price, new_price)
        else:
            new_price = round(entry - required_reward, 2)
            targets[0].price = min(targets[0].price, new_price)
        _enforce_monotonic_targets(entry, side, targets, rr_spacing or 0.01)
        for meta in targets:
            meta.distance = round(abs(meta.price - entry), 2)
            reward = meta.price - entry if side == "long" else entry - meta.price
            meta.rr_multiple = round(max(reward, 0.0) / risk, 2)
            if em_day > 0:
                meta.em_fraction = round(meta.distance / em_day, 2)
    if targets and targets[0].rr_multiple < stop.rr_min:
        first = targets[0]
        reward = first.price - entry if side == "long" else entry - first.price
        if reward > 0 and stop.rr_min > 0:
            min_risk = reward / stop.rr_min
            if side == "long":
                tightened = entry - min_risk
                tightened = min(entry - 0.01, max(stop.price, tightened))
            else:
                tightened = entry + min_risk
                tightened = max(entry + 0.01, min(stop.price, tightened))
            tightened = _quantize_stop_toward_entry(tightened, side, entry)
            if tightened != stop.price:
                if stop.snapped:
                    stop.snapped = f"{stop.snapped}|rr_floor_stop_adjust"
                else:
                    stop.snapped = "rr_floor_stop_adjust"
                stop.price = tightened
                risk = entry - stop.price if side == "long" else stop.price - entry
                if risk <= 0:
                    risk = 0.001
                required_reward = stop.rr_min * risk
                spacing = 0.01
                if cap_distance is not None and required_reward > cap_distance + 1e-6:
                    spacing = max(0.05, round(atr_tf * 0.5, 2))
                _enforce_monotonic_targets(entry, side, targets, spacing)
                for meta in targets:
                    meta.distance = round(abs(meta.price - entry), 2)
                    reward = meta.price - entry if side == "long" else entry - meta.price
                    meta.rr_multiple = round(max(reward, 0.0) / risk, 2)
                    if em_day > 0:
                        meta.em_fraction = round(meta.distance / em_day, 2)
    validate_invariants(entry, stop.price, targets, side, stop.rr_min)
    return PlanGeometry(
        entry=round(entry, 2),
        stop=stop,
        targets=targets,
        runner=runner,
        em_day=em_day,
        em_used=em_used,
        snap_trace=snap_trace,
        ratr=ratr,
    )
