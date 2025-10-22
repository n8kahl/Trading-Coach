"""Deterministic TP/SL/runner engine for plan construction."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime, time
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

from .levels import last_higher_low, last_lower_high

MIN_STOP_ATR: Mapping[str, float] = {
    "scalp": 0.6,
    "intraday": 0.9,
    "swing": 1.2,
    "leaps": 1.6,
}

MAX_STOP_ATR: Mapping[str, float] = {
    "scalp": 1.6,
    "intraday": 2.0,
    "swing": 2.5,
    "leaps": 3.0,
}

MIN_TP1_RR: Mapping[str, float] = {
    "scalp": 1.0,
    "intraday": 1.3,
    "swing": 1.6,
    "leaps": 2.0,
}

MIN_TP_SPACING_ATR = 0.40
TP3_RR_AT_LEAST = 2.2
NEAR_STRUCT_TOL_ATR = 0.25

logger = logging.getLogger(__name__)

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


def enforce_monotone_decreasing(values: Sequence[float]) -> List[float]:
    out: List[float] = []
    last = 1.0
    for value in values:
        p = max(0.0, min(1.0, float(value)))
        last = min(last, p)
        out.append(last)
    return out


def _stop_bounds(style: str) -> Tuple[float, float]:
    style_key = _style_key(style)
    return MIN_STOP_ATR.get(style_key, 0.9), MAX_STOP_ATR.get(style_key, 2.0)


def _tp1_rr_min(style: str) -> float:
    style_key = _style_key(style)
    return MIN_TP1_RR.get(style_key, 1.3)


def _atr_ratio(entry: float, stop: float, atr_tf: float) -> float:
    if atr_tf is None or atr_tf <= 0:
        return float("inf")
    return abs(entry - stop) / atr_tf


def _clamp_stop_price(side: str, entry: float, candidate: float) -> float:
    if side == "long":
        return min(candidate, entry - 0.01)
    return max(candidate, entry + 0.01)


def _local_invalidation(side: str, entry: float, atr_tf: float, levels: Mapping[str, float]) -> Tuple[float, str]:
    buffer_primary = max(0.2 * atr_tf, 0.05)
    buffer_secondary = max(0.25 * atr_tf, 0.05)
    if side == "long":
        swing = last_higher_low(levels, None)
        if swing is not None and math.isfinite(swing) and swing < entry:
            price = _clamp_stop_price(side, entry, float(swing) - buffer_primary)
            return round(price, 2), "swing_low"
        vwap = levels.get("vwap") or levels.get("VWAP")
        if vwap is not None and math.isfinite(float(vwap)):
            price = _clamp_stop_price(side, entry, float(vwap) - buffer_primary)
            return round(price, 2), "vwap_band"
        orl = levels.get("orl") or levels.get("ORL")
        if orl is not None and math.isfinite(float(orl)):
            price = _clamp_stop_price(side, entry, float(orl) - buffer_secondary)
            return round(price, 2), "orl_retest"
    else:
        swing = last_lower_high(levels, None)
        if swing is not None and math.isfinite(swing) and swing > entry:
            price = _clamp_stop_price(side, entry, float(swing) + buffer_primary)
            return round(price, 2), "swing_high"
        vwap = levels.get("vwap") or levels.get("VWAP")
        if vwap is not None and math.isfinite(float(vwap)):
            price = _clamp_stop_price(side, entry, float(vwap) + buffer_primary)
            return round(price, 2), "vwap_band"
        orh = levels.get("orh") or levels.get("ORH")
        if orh is not None and math.isfinite(float(orh)):
            price = _clamp_stop_price(side, entry, float(orh) + buffer_secondary)
            return round(price, 2), "orh_retest"
    if side == "long":
        fallback = entry - max(buffer_primary, 0.1)
    else:
        fallback = entry + max(buffer_primary, 0.1)
    return round(_clamp_stop_price(side, entry, fallback), 2), "atr_buffer"


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
    quantiles = [0.9, 0.82, 0.68, 0.55]
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


def _enforce_monotonic_probabilities(targets: Sequence[TargetMeta]) -> None:
    if not targets:
        return
    probs = [float(getattr(meta, "prob_touch", 0.0) or 0.0) for meta in targets]
    monotone = enforce_monotone_decreasing(probs)
    for meta, value in zip(targets, monotone):
        meta.prob_touch = round(value, 2)


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
    stop_price: float,
) -> Tuple[List[TargetMeta], bool, Optional[float]]:
    direction = (side or "").lower()
    level_map = dict(levels or {})
    min_rr = _tp1_rr_min(style)
    min_spacing_points = max(MIN_TP_SPACING_ATR * atr_tf, 0.1)
    tolerance = NEAR_STRUCT_TOL_ATR * atr_tf
    em_cap = em_day if em_day > 0 else float("inf")
    ratr_cap = ratr * max(tod_mult, 0.5) if ratr > 0 else float("inf")
    far_cap = min(em_cap, ratr_cap)
    modifiers_local = _strategy_modifiers(strategy)
    target_limit_raw = modifiers_local.get("target_count") if modifiers_local else None
    target_limit = 3
    if isinstance(target_limit_raw, int) and target_limit_raw > 0:
        target_limit = target_limit_raw
    base_candidates = _structural_candidates(entry, direction, level_map, atr_tf)
    targets: List[TargetMeta] = []
    em_used = False
    mod_key = _strategy_key(strategy)

    def _distance(price: float) -> float:
        return abs(price - entry)

    def _within_cap(price: float) -> bool:
        if far_cap == float("inf"):
            return True
        return _distance(price) <= far_cap + tolerance + 1e-9

    def _snap(price: float) -> float:
        if far_cap == float("inf"):
            return price
        dist = _distance(price)
        if dist <= far_cap:
            return price
        if dist - far_cap <= tolerance:
            return price
        snapped = entry + far_cap if direction == "long" else entry - far_cap
        nonlocal em_used
        em_used = True
        return snapped

    def _rr(price: float) -> float:
        risk = abs(entry - stop_price)
        if risk <= 0:
            return 0.0
        reward = price - entry if direction == "long" else entry - price
        return reward / risk if reward > 0 else 0.0

    for price, tag in base_candidates:
        if not _within_cap(price):
            continue
        snapped_price = _snap(price)
        if direction == "long" and snapped_price <= entry + 1e-6:
            continue
        if direction == "short" and snapped_price >= entry - 1e-6:
            continue
        if targets and abs(snapped_price - targets[-1].price) < min_spacing_points - 1e-9:
            continue
        rr_multiple = _rr(snapped_price)
        if not targets and rr_multiple < min_rr:
            continue
        meta = TargetMeta(
            price=round(snapped_price, 2),
            distance=round(_distance(snapped_price), 2),
            rr_multiple=0.0,
            prob_touch=round(_probability_from_mfe_quantile(len(targets)), 2),
            em_fraction=(round(_distance(snapped_price) / em_day, 3) if em_day > 0 else None),
            mfe_quantile=None,
            reason=tag.upper(),
            em_capped=abs(snapped_price - price) > 1e-6,
        )
        targets.append(meta)
        if len(targets) == target_limit:
            break

    if mod_key == "gap_fill":
        gap_price = level_map.get("gap_fill")
        if isinstance(gap_price, (int, float)) and math.isfinite(float(gap_price)):
            gap_val = float(gap_price)
            if (direction == "long" and gap_val > entry) or (direction == "short" and gap_val < entry):
                snapped_gap = _snap(gap_val)
                if direction == "long" and snapped_gap > entry + 1e-6:
                    if targets:
                        targets[0].price = round(snapped_gap, 2)
                        targets[0].distance = round(_distance(snapped_gap), 2)
                        targets[0].rr_multiple = _rr(targets[0].price)
                        targets[0].reason = "gap_fill"
                        if em_day > 0:
                            targets[0].em_fraction = round(targets[0].distance / em_day, 2)
                    else:
                        rr_multiple = _rr(snapped_gap)
                        if rr_multiple >= min_rr:
                            targets.insert(
                                0,
                                TargetMeta(
                                    price=round(snapped_gap, 2),
                                    distance=round(_distance(snapped_gap), 2),
                                    rr_multiple=rr_multiple,
                                    prob_touch=round(_probability_from_mfe_quantile(0), 2),
                                    em_fraction=(round(_distance(snapped_gap) / em_day, 2) if em_day > 0 else None),
                                    mfe_quantile=None,
                                    reason="gap_fill",
                                    em_capped=abs(snapped_gap - gap_val) > 1e-6,
                                ),
                            )
                        if len(targets) == target_limit:
                            targets = targets[:target_limit]
                elif direction == "short" and snapped_gap < entry - 1e-6:
                    if targets:
                        targets[0].price = round(snapped_gap, 2)
                        targets[0].distance = round(_distance(snapped_gap), 2)
                        targets[0].rr_multiple = _rr(targets[0].price)
                        targets[0].reason = "gap_fill"
                        if em_day > 0:
                            targets[0].em_fraction = round(targets[0].distance / em_day, 2)
                    else:
                        rr_multiple = _rr(snapped_gap)
                        if rr_multiple >= min_rr:
                            targets.insert(
                                0,
                                TargetMeta(
                                    price=round(snapped_gap, 2),
                                    distance=round(_distance(snapped_gap), 2),
                                    rr_multiple=rr_multiple,
                                    prob_touch=round(_probability_from_mfe_quantile(0), 2),
                                    em_fraction=(round(_distance(snapped_gap) / em_day, 2) if em_day > 0 else None),
                                    mfe_quantile=None,
                                    reason="gap_fill",
                                    em_capped=abs(snapped_gap - gap_val) > 1e-6,
                                ),
                            )
                        if len(targets) == target_limit:
                            targets = targets[:target_limit]

    if len(targets) < target_limit:
        ladder_step = max(min_spacing_points, atr_tf * 0.8)
        while len(targets) < target_limit:
            offset = ladder_step * (len(targets) + 1)
            ladder_price = entry + offset if direction == "long" else entry - offset
            if not _within_cap(ladder_price):
                break
            snapped_price = _snap(ladder_price)
            if targets and abs(snapped_price - targets[-1].price) < min_spacing_points - 1e-9:
                snapped_price = targets[-1].price + (min_spacing_points if direction == "long" else -min_spacing_points)
            meta = TargetMeta(
                price=round(snapped_price, 2),
                distance=round(_distance(snapped_price), 2),
                rr_multiple=0.0,
                prob_touch=round(_probability_from_mfe_quantile(len(targets)), 2),
                em_fraction=(round(_distance(snapped_price) / em_day, 3) if em_day > 0 else None),
                mfe_quantile=None,
                reason="ATR_LADDER",
                em_capped=False,
            )
            targets.append(meta)
            if len(targets) >= target_limit:
                break

    _enforce_monotonic_targets(entry, direction, targets)
    prob_sequence = enforce_monotone_decreasing([meta.prob_touch for meta in targets])
    for meta, prob in zip(targets, prob_sequence):
        meta.prob_touch = round(prob, 2)
    capped_distance = None if far_cap == float("inf") else float(far_cap)
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


def _structural_candidates(
    entry: float,
    side: str,
    levels: Mapping[str, float],
    atr_tf: float,
) -> List[Tuple[float, str]]:
    mapping = (
        "orh",
        "orl",
        "pdh",
        "pdl",
        "session_high",
        "session_low",
        "premarket_high",
        "premarket_low",
        "vah",
        "val",
        "poc",
        "gap_fill",
        "gap_top",
        "gap_bottom",
        "avwap",
        "avwap_session_open",
        "avwap_prev_high",
        "avwap_prev_low",
    )
    direction = side.lower()
    candidates: List[Tuple[float, str]] = []
    for key in mapping:
        value = levels.get(key) or levels.get(key.upper())
        if value is None:
            continue
        try:
            price = float(value)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(price):
            continue
        if direction == "long" and price <= entry:
            continue
        if direction == "short" and price >= entry:
            continue
        candidates.append((price, key.lower()))

    increment = max(0.25, round(atr_tf * 0.5, 2))
    for mult in range(1, 8):
        offset = increment * mult
        if direction == "long":
            price = entry + offset
        else:
            price = entry - offset
        label = f"quarter_{mult}"
        candidates.append((round(price, 2), label))

    candidates.sort(key=lambda item: abs(item[0] - entry))
    return candidates


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
    target_limit_raw = modifiers.get("target_count") if modifiers else None
    target_limit = 3
    if isinstance(target_limit_raw, int) and target_limit_raw > 0:
        target_limit = target_limit_raw
    tod_mult = tod_multiplier(timestamp)
    tod_boost = modifiers.get("tod_boost")
    if isinstance(tod_boost, (int, float)) and math.isfinite(tod_boost) and tod_boost > 0:
        tod_mult *= float(tod_boost)
    stop = derive_stop(entry, side, levels, atr_tf, style, strategy)
    min_stop_ratio, max_stop_ratio = _stop_bounds(style)
    stop_ratio = _atr_ratio(entry, stop.price, atr_tf)
    if stop_ratio < min_stop_ratio - 1e-6:
        adjusted_stop, reason = _local_invalidation(side, entry, atr_tf, levels)
        stop.price = adjusted_stop
        stop.structural = adjusted_stop
        stop.snapped = f"{stop.snapped}|{reason}" if stop.snapped else reason
        stop_ratio = _atr_ratio(entry, stop.price, atr_tf)
    if stop_ratio < min_stop_ratio - 1e-6:
        fallback_stop = entry - min_stop_ratio * atr_tf if side == "long" else entry + min_stop_ratio * atr_tf
        fallback_stop = _clamp_stop_price(side, entry, fallback_stop)
        stop.price = round(fallback_stop, 2)
        stop.structural = stop.price
        stop.snapped = f"{stop.snapped}|min_atr_guardrail" if stop.snapped else "min_atr_guardrail"
        stop_ratio = _atr_ratio(entry, stop.price, atr_tf)
    if stop_ratio > max_stop_ratio + 1e-6:
        fallback_stop = entry - max_stop_ratio * atr_tf if side == "long" else entry + max_stop_ratio * atr_tf
        fallback_stop = _clamp_stop_price(side, entry, fallback_stop)
        stop.price = round(fallback_stop, 2)
        stop.structural = stop.price
        stop.snapped = f"{stop.snapped}|max_atr_guardrail" if stop.snapped else "max_atr_guardrail"
        stop_ratio = _atr_ratio(entry, stop.price, atr_tf)
    targets, em_used, cap_distance = build_targets(
        entry,
        side,
        atr_tf,
        em_day,
        ratr,
        tod_mult,
        style,
        strategy,
        levels,
        stop.price,
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
    min_spacing_points = max(MIN_TP_SPACING_ATR * atr_tf, 0.1)
    _enforce_monotonic_targets(entry, side, targets, min_spacing_points)
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
    if len(targets) >= 3 and targets[-1].rr_multiple < TP3_RR_AT_LEAST:
        desired_reward = TP3_RR_AT_LEAST * risk
        if side == "long":
            adjusted_price = entry + desired_reward
            if cap_distance is not None:
                adjusted_price = min(adjusted_price, entry + cap_distance)
            adjusted_price = max(adjusted_price, targets[-2].price + min_spacing_points)
        else:
            adjusted_price = entry - desired_reward
            if cap_distance is not None:
                adjusted_price = max(adjusted_price, entry - cap_distance)
            adjusted_price = min(adjusted_price, targets[-2].price - min_spacing_points)
        targets[-1].price = round(adjusted_price, 2)
        targets[-1].distance = round(abs(targets[-1].price - entry), 2)
        reward = targets[-1].price - entry if side == "long" else entry - targets[-1].price
        targets[-1].rr_multiple = round(max(reward, 0.0) / risk, 2)
        if em_day > 0:
            targets[-1].em_fraction = round(targets[-1].distance / em_day, 2)
    while len(targets) < target_limit:
        step = max(min_spacing_points, 0.05)
        if not targets:
            candidate_price = entry + step if side == "long" else entry - step
        else:
            candidate_price = targets[-1].price + step if side == "long" else targets[-1].price - step
        if cap_distance is not None:
            cap_bound = entry + cap_distance if side == "long" else entry - cap_distance
            if side == "long":
                candidate_price = min(candidate_price, cap_bound)
            else:
                candidate_price = max(candidate_price, cap_bound)
        if targets and abs(candidate_price - targets[-1].price) < 0.01:
            break
        reward_candidate = candidate_price - entry if side == "long" else entry - candidate_price
        rr_candidate = reward_candidate / risk if risk > 0 else 0.0
        meta = TargetMeta(
            price=round(candidate_price, 2),
            distance=round(abs(candidate_price - entry), 2),
            rr_multiple=round(max(rr_candidate, 0.0), 2),
            prob_touch=round(_probability_from_mfe_quantile(len(targets)), 2),
            em_fraction=(round(abs(candidate_price - entry) / em_day, 3) if em_day > 0 else None),
            mfe_quantile=None,
            reason="EXTENSION",
            em_capped=False,
        )
        targets.append(meta)
    if targets and targets[0].rr_multiple < stop.rr_min:
        if side == "long":
            new_price = round(entry + required_reward, 2)
            targets[0].price = max(targets[0].price, new_price)
        else:
            new_price = round(entry - required_reward, 2)
            targets[0].price = min(targets[0].price, new_price)
        _enforce_monotonic_targets(entry, side, targets, max(rr_spacing or min_spacing_points, 0.01))
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
                _enforce_monotonic_targets(entry, side, targets, max(spacing, min_spacing_points))
                for meta in targets:
                    meta.distance = round(abs(meta.price - entry), 2)
                    reward = meta.price - entry if side == "long" else entry - meta.price
                    meta.rr_multiple = round(max(reward, 0.0) / risk, 2)
                    if em_day > 0:
                        meta.em_fraction = round(meta.distance / em_day, 2)
    _enforce_monotonic_targets(entry, side, targets, min_spacing_points)
    _enforce_monotonic_probabilities(targets)
    final_ratio = _atr_ratio(entry, stop.price, atr_tf)
    if final_ratio < min_stop_ratio - 1e-9:
        logger.debug(
            "stop_guardrail_below_min",
            extra={
                "entry": entry,
                "stop": stop.price,
                "atr": atr_tf,
                "style": style,
                "ratio": final_ratio,
            },
        )
    spacing_ok = all(
        abs(targets[idx + 1].price - targets[idx].price) >= min_spacing_points - 1e-6
        for idx in range(len(targets) - 1)
    )
    if not spacing_ok:
        logger.debug(
            "tp_clustered",
            extra={
                "entry": entry,
                "targets": [meta.price for meta in targets],
                "min_spacing": min_spacing_points,
            },
        )
    prob_values = [meta.prob_touch for meta in targets]
    if prob_values != sorted(prob_values, reverse=True):
        logger.debug(
            "prob_non_monotone",
            extra={
                "probabilities": prob_values,
            },
        )
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
