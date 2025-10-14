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

import json

import numpy as np
import pandas as pd

from .strategy_library import (
    Strategy,
    load_strategies,
    normalize_style_input,
    strategy_internal_category,
)
from .context_overlays import _volume_profile
from .calculations import atr, ema, vwap, adx
from .statistics import get_style_stats, estimate_probability
from .app.engine import build_target_profile, build_structured_plan

TZ_ET = "America/New_York"
RTH_START_MINUTE = 9 * 60 + 30
RTH_END_MINUTE = 16 * 60

_STYLE_TARGET_PRESETS: Dict[str, Dict[str, Any]] = {
    "scalp": {
        "measure": "atr",
        "tp1_range": (0.35, 0.50),
        "tp2_range": (0.70, 0.90),
    },
    "intraday": {
        "measure": "atr",
        "tp1_range": (0.75, 1.00),
        "tp2_range": (1.25, 1.50),
    },
    "swing": {
        "measure": "atr",
        "tp1_range": (1.00, 1.50),
        "tp2_range": (2.00, 3.00),
    },
    "leaps": {
        "measure": "em",
        "tp1_range": (1.00, 1.30),
        "tp2_range": (1.60, 2.00),
    },
}

_PRICE_INCREMENT_OVERRIDES: Dict[str, float] = {
    "ES": 0.25,
    "NQ": 0.25,
    "YM": 1.0,
    "RTY": 0.10,
    "CL": 0.01,
    "GC": 0.10,
}


_TARGET_RULES: Dict[str, List[Dict[str, object]]] = {
    "scalp": [
        {"label": "TP1", "em_fraction": 0.33, "quantile_key": "q50", "pot_min": 0.55},
        {"label": "TP2", "em_fraction": 0.62, "quantile_key": "q70", "pot_min": 0.4},
        {"label": "TP3", "em_fraction": 0.85, "quantile_key": "q80", "pot_min": 0.25, "optional": True},
    ],
    "intraday": [
        {"label": "TP1", "em_fraction": 0.45, "quantile_key": "q50", "pot_min": 0.6},
        {"label": "TP2", "em_fraction": 0.75, "quantile_key": "q70", "pot_min": 0.4},
        {"label": "TP3", "em_fraction": 1.05, "quantile_key": "q80", "pot_min": 0.3},
    ],
    "swing": [
        {"label": "TP1", "em_fraction": 0.5, "quantile_key": "q50", "pot_min": 0.65},
        {"label": "TP2", "em_fraction": 0.8, "quantile_key": "q80", "pot_min": 0.45},
        {"label": "TP3", "em_fraction": 1.1, "quantile_key": "q90", "pot_min": 0.3},
    ],
    "leaps": [
        {"label": "TP1", "em_fraction": 0.55, "quantile_key": "q50", "pot_min": 0.65},
        {"label": "TP2", "em_fraction": 0.85, "quantile_key": "q80", "pot_min": 0.4},
        {"label": "TP3", "em_fraction": 1.2, "quantile_key": "q90", "pot_min": 0.25},
    ],
}

_RUNNER_RULES: Dict[str, Dict[str, Any]] = {
    "0dte": {"type": "chandelier", "timeframe": "1m", "length": 10, "multiplier": 1.4, "label": "Runner Trail", "note": "Trail with 1m chandelier"},
    "scalp": {"type": "chandelier", "timeframe": "1m", "length": 10, "multiplier": 1.4, "label": "Runner Trail", "note": "Trail with 1m chandelier"},
    "intraday": {"type": "chandelier", "timeframe": "5m", "length": 14, "multiplier": 1.8, "label": "Runner Trail", "note": "Trail with 5m chandelier / EMA20"},
    "swing": {"type": "chandelier", "timeframe": "4h", "length": 20, "multiplier": 2.3, "label": "Runner Trail", "note": "Trail below 4h swing lows"},
    "leaps": {"type": "chandelier", "timeframe": "1D", "length": 20, "multiplier": 3.0, "label": "Runner Trail", "note": "Trail on daily swing structure"},
}


def _targets_from_stats(
    *,
    style_key: str,
    bias: str,
    entry: float,
    min_distance: float,
    em_limit: Optional[float],
    stats: Dict[str, object] | None,
) -> List[Dict[str, object]]:
    if not stats:
        return []
    rules = _TARGET_RULES.get(style_key)
    if not rules:
        return []

    dir_key = "long" if bias == "long" else "short"
    dir_stats = stats.get(dir_key) if isinstance(stats, dict) else None
    if not isinstance(dir_stats, dict):
        return []
    mfe_values = dir_stats.get("mfe")
    quantiles = dir_stats.get("quantiles") or {}
    if mfe_values is None or not isinstance(mfe_values, np.ndarray):
        mfe_values = np.array([], dtype=float)

    em_value = None
    try:
        em_value = float(stats.get("expected_move")) if stats.get("expected_move") is not None else None
    except (TypeError, ValueError):
        em_value = None

    targets: List[Dict[str, object]] = []
    for idx, rule in enumerate(rules, start=1):
        label = str(rule.get("label") or f"TP{idx}")
        em_frac = rule.get("em_fraction")
        quantile_key = rule.get("quantile_key")
        candidate_distances: List[float] = []
        meta: Dict[str, object] = {"label": label}

        if em_value and em_frac:
            try:
                em_offset = float(em_frac) * float(em_value)
                if math.isfinite(em_offset) and em_offset > 0:
                    candidate_distances.append(em_offset)
                    meta["em_fraction"] = float(em_frac)
                    meta["em_basis"] = float(em_value)
            except (TypeError, ValueError):
                pass

        if quantile_key and quantile_key in quantiles:
            try:
                ratio = float(quantiles[quantile_key])
                if math.isfinite(ratio) and ratio > 0 and entry > 0:
                    candidate_distances.append(entry * ratio)
                    meta["mfe_quantile"] = quantile_key
                    meta["mfe_ratio"] = ratio
            except (TypeError, ValueError):
                pass

        if not candidate_distances:
            continue

        distance = min(candidate_distances)
        if em_limit is not None and em_limit > 0:
            distance = min(distance, em_limit)
        distance = max(distance, min_distance)

        pot = estimate_probability(mfe_values, distance / entry) if entry > 0 else None
        if pot is not None and not math.isnan(pot):
            meta["prob_touch"] = float(pot)
        pot_min = rule.get("pot_min")
        optional = bool(rule.get("optional"))
        if pot_min is not None and pot is not None and pot < float(pot_min):
            if optional:
                continue
            meta["prob_touch_flag"] = "below_threshold"
        meta["distance"] = float(distance)
        meta["optional"] = optional
        targets.append({"distance": distance, "meta": meta})

    return targets


def _runner_config(style_key: str, bias: str, entry: float, stop: float, targets: List[float]) -> Dict[str, Any]:
    rule = dict(_RUNNER_RULES.get(style_key, _RUNNER_RULES["intraday"]))
    rule["bias"] = bias
    rule["anchor"] = float(targets[-1]) if targets else float(entry)
    rule["initial_stop"] = float(stop)
    return rule


@dataclass(slots=True)
class Plan:
    direction: str
    entry: float
    stop: float
    targets: List[float]
    confidence: float
    risk_reward: float
    target_meta: List[Dict[str, Any]] = field(default_factory=list)
    runner: Dict[str, Any] | None = None
    notes: str | None = None
    atr: float | None = None
    warnings: List[str] = field(default_factory=list)
    target_profile: Dict[str, Any] | None = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "direction": self.direction,
            "entry": round(float(self.entry), 4),
            "stop": round(float(self.stop), 4),
            "targets": [round(float(t), 4) for t in self.targets],
            "target_meta": self.target_meta,
            "runner": self.runner,
            "confidence": round(float(self.confidence), 3),
            "risk_reward": round(float(self.risk_reward), 3),
            "atr": round(float(self.atr), 4) if self.atr is not None else None,
            "notes": self.notes,
            "warnings": list(self.warnings),
            "target_profile": self.target_profile,
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


def rr(entry: float, stop: float, tp: float, bias: str) -> float:
    risk = (entry - stop) if bias == "long" else (stop - entry)
    reward = (tp - entry) if bias == "long" else (entry - tp)
    if risk <= 0:
        return 0.0
    return max(0.0, reward / risk)


def _enrich_plan_with_profile(
    plan: Plan,
    *,
    style: Optional[str],
    bias: Optional[str],
    atr_value: Optional[float],
    expected_move: Optional[float],
    debug: Optional[Dict[str, Any]],
) -> TargetEngineResult:
    profile = build_target_profile(
        entry=plan.entry,
        stop=plan.stop,
        targets=plan.targets,
        target_meta=plan.target_meta,
        debug=debug,
        runner=plan.runner,
        warnings=plan.warnings,
        atr_used=atr_value,
        expected_move=expected_move,
        style=style,
        bias=bias,
    )
    plan.target_profile = profile.to_dict()
    return profile


def _normalize_trade_style(style: str | None) -> str:
    return normalize_style_input(style) or "intraday"


def _base_targets_for_style(
    *,
    style: str | None,
    bias: str,
    entry: float,
    stop: float,
    atr: float | None,
    expected_move: float | None,
    min_rr: float,
    prefer_em_cap: bool = True,
) -> List[float]:
    style_key = _normalize_trade_style(style)
    risk = abs(entry - stop)
    if risk <= 0:
        return []
    atr_val = float(atr or 0.0)
    expected_move_val = float(expected_move) if isinstance(expected_move, (int, float)) else None
    preset = _STYLE_TARGET_PRESETS.get(style_key, _STYLE_TARGET_PRESETS["intraday"])
    measure_type = preset.get("measure", "atr")
    measure_value = atr_val if measure_type == "atr" else expected_move_val
    if measure_value is None or not math.isfinite(measure_value) or measure_value <= 0:
        # Fallback ordering: ATR → expected move → risk
        if atr_val > 0:
            measure_value = atr_val
        elif expected_move_val and expected_move_val > 0:
            measure_value = expected_move_val
        else:
            measure_value = risk

    def _offset_from_range(mult_range: Tuple[float, float]) -> float:
        low, high = mult_range
        base_mult = (float(low) + float(high)) / 2.0
        offset = base_mult * measure_value
        min_distance = max(min_rr * risk, 0.0)
        if offset < min_distance:
            offset = float(high) * measure_value
        if offset < min_distance:
            offset = min_distance
        return max(offset, 0.0)

    tp1_offset = _offset_from_range(tuple(preset["tp1_range"]))
    tp2_offset = _offset_from_range(tuple(preset["tp2_range"]))

    em_limit = None
    if expected_move_val and math.isfinite(expected_move_val) and expected_move_val > 0:
        em_limit = float(expected_move_val) * (1.0 if prefer_em_cap else 1.10)

    if em_limit is not None:
        tp1_offset = min(tp1_offset, em_limit)
        tp2_offset = min(tp2_offset, em_limit)

    if tp2_offset < tp1_offset:
        tp2_offset = tp1_offset

    targets: List[float] = []
    if bias == "long":
        targets.append(entry + tp1_offset)
        targets.append(entry + tp2_offset)
    else:
        targets.append(entry - tp1_offset)
        targets.append(entry - tp2_offset)
    return [float(t) for t in targets if math.isfinite(t)]


def _price_increment(symbol: str | None, reference: float) -> float:
    sym = (symbol or "").upper()
    if sym in _PRICE_INCREMENT_OVERRIDES:
        return _PRICE_INCREMENT_OVERRIDES[sym]
    if reference < 5:
        return 0.01
    if reference < 25:
        return 0.02
    return 0.05


def _round_to_increment(value: float, increment: float) -> float:
    if increment <= 0:
        return round(float(value), 4)
    scaled = round(value / increment)
    return round(scaled * increment, 4)


def _apply_limit_with_tick(entry: float, bias: str, value: float, limit: float | None, tick_size: float) -> float:
    if limit is None or tick_size <= 0:
        return value
    bound = entry + limit if bias == "long" else entry - limit
    epsilon = tick_size * 0.5
    if bias == "long":
        if value <= bound + epsilon:
            return value
        steps = math.floor(bound / tick_size)
        capped = steps * tick_size
        if capped < entry:
            capped = entry
        return round(capped, 4)
    else:
        if value >= bound - epsilon:
            return value
        steps = math.ceil(bound / tick_size)
        capped = steps * tick_size
        if capped > entry:
            capped = entry
        return round(capped, 4)


def _expected_move_limit(expected_move: float | None, prefer_em_cap: bool) -> float | None:
    if expected_move is None:
        return None
    try:
        em_val = abs(float(expected_move))
    except (TypeError, ValueError):
        return None
    if not math.isfinite(em_val) or em_val <= 0:
        return None
    return em_val * (1.0 if prefer_em_cap else 1.10)


def _clamp_to_em(entry: float, candidate: float, bias: str, em_limit: float | None) -> float:
    if em_limit is None:
        return candidate
    if bias == "long":
        max_price = entry + em_limit
        return min(candidate, max_price)
    max_price = entry - em_limit
    return max(candidate, max_price)


def _canonical_style_token(style: str | None) -> str:
    token = (_normalize_trade_style(style) or "intraday").lower()
    if token == "leap":
        return "leaps"
    return token


def _atr_for_style(style_key: str, ctx: Dict[str, Any]) -> float | None:
    lookup = {
        "0dte": ctx.get("atr_5m"),
        "scalp": ctx.get("atr_5m"),
        "intraday": ctx.get("atr_15m") or ctx.get("atr_5m"),
        "swing": ctx.get("atr_1d") or ctx.get("atr_15m"),
        "leaps": ctx.get("atr_1w") or ctx.get("atr_1d"),
    }
    value = lookup.get(style_key, ctx.get("atr"))
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric) or numeric <= 0:
        return None
    return numeric


def _minimum_tp_distance(
    *,
    style_key: str,
    entry: float,
    stop: float,
    ctx: Dict[str, Any],
    min_rr: float,
    tick_size: float,
) -> float:
    risk = abs(entry - stop)
    fallback = max(tick_size, 0.0)

    def _finite_candidates(values: Iterable[float | None]) -> List[float]:
        clean: List[float] = []
        for item in values:
            if item is None:
                continue
            try:
                val = float(item)
            except (TypeError, ValueError):
                continue
            if math.isfinite(val) and val > 0:
                clean.append(val)
        return clean

    distance: float | None = None
    if style_key == "0dte":
        atr_5 = _atr_for_style("0dte", ctx)
        if atr_5 is not None:
            distance = atr_5 * 0.30
    elif style_key == "scalp":
        atr_5 = _atr_for_style("scalp", ctx)
        if atr_5 is not None:
            distance = atr_5 * 0.60
    elif style_key == "intraday":
        atr_15 = _atr_for_style("intraday", ctx)
        if atr_15 is not None:
            distance = atr_15 * 0.80
    elif style_key == "swing":
        atr_1d = ctx.get("atr_1d")
        candidates = _finite_candidates([atr_1d * 1.20 if atr_1d else None, abs(entry) * 0.015])
        if candidates:
            distance = max(candidates)
    elif style_key == "leaps":
        atr_1w = ctx.get("atr_1w")
        atr_1d = ctx.get("atr_1d")
        candidates = _finite_candidates(
            [
                atr_1w * 1.00 if atr_1w else None,
                atr_1d * 2.5 if atr_1d else None,
                abs(entry) * 0.05,
            ]
        )
        if candidates:
            distance = max(candidates)

    if distance is None or not math.isfinite(distance) or distance <= 0:
        return fallback
    return max(distance, fallback)


def _intraday_levels_list(ctx: Dict[str, Any]) -> List[Tuple[str, float]]:
    key = ctx.get("key") or {}
    mapping = {
        "SESSION_HIGH": key.get("session_high"),
        "SESSION_LOW": key.get("session_low"),
        "ORB_HIGH": key.get("opening_range_high"),
        "ORB_LOW": key.get("opening_range_low"),
        "PRIOR_HIGH": key.get("prev_high"),
        "PRIOR_LOW": key.get("prev_low"),
        "PRIOR_CLOSE": key.get("prev_close"),
    }
    levels: List[Tuple[str, float]] = []
    for tag, value in mapping.items():
        try:
            price = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(price):
            levels.append((tag, price))

    for ema_tag in ("ema9", "ema20", "ema50"):
        value = ctx.get(ema_tag)
        try:
            price = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(price):
            levels.append((ema_tag.upper(), price))

    vwap_val = ctx.get("vwap")
    try:
        vwap_price = float(vwap_val)
    except (TypeError, ValueError):
        vwap_price = None
    if vwap_price is not None and math.isfinite(vwap_price):
        levels.append(("VWAP", vwap_price))

    for raw in ctx.get("htf_levels") or []:
        try:
            price = float(raw)
        except (TypeError, ValueError):
            continue
        if math.isfinite(price):
            levels.append(("HTF", price))

    dedup: Dict[float, Tuple[str, float]] = {}
    for tag, price in levels:
        rounded = round(price, 4)
        if rounded not in dedup:
            dedup[rounded] = (tag, price)
    return list(dedup.values())


def _dict_to_level_list(data: Dict[str, float] | None, prefix: str) -> List[Tuple[str, float]]:
    if not data:
        return []
    results: List[Tuple[str, float]] = []
    for key, value in data.items():
        try:
            price = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(price):
            results.append((f"{prefix}_{key.upper()}", price))
    results.sort(key=lambda item: item[1])
    return results


def _fib_candidate_list(style_key: str, bias: str, ctx: Dict[str, Any]) -> List[Tuple[str, float]]:
    direction_key = "long" if bias == "long" else "short"
    if style_key == "leaps":
        pack = ctx.get("fib_weekly") or {}
        entries = pack.get(direction_key, [])
        return [(tag, float(price)) for tag, price in entries if math.isfinite(float(price))]
    if style_key == "swing":
        pack = ctx.get("fib_daily") or {}
        entries = pack.get(direction_key, [])
        return [(tag, float(price)) for tag, price in entries if math.isfinite(float(price))]
    fib_map = ctx.get("fib_up") if bias == "long" else ctx.get("fib_down")
    results: List[Tuple[str, float]] = []
    for tag, value in (fib_map or {}).items():
        try:
            price = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(price):
            results.append((tag, price))
    results.sort(key=lambda item: item[1], reverse=(bias == "short"))
    return results


def _collect_priority_levels(entry: float, bias: str, ctx: Dict[str, Any]) -> List[Tuple[int, float, float, str]]:
    levels: List[Tuple[int, float, float, str]] = []
    symbol_levels: List[Dict[str, Any]] = []
    if bias == "long":
        symbol_levels.extend(_build_tp_candidates_long(entry, ctx))
    else:
        symbol_levels.extend(_build_tp_candidates_short(entry, ctx))

    priority_map = {
        "POC": 1,
        "VAH": 1,
        "VAL": 1,
        "PRIOR_HIGH": 2,
        "PRIOR_LOW": 2,
        "ORB_HIGH": 2,
        "ORB_LOW": 2,
        "SESSION_HIGH": 2,
        "SESSION_LOW": 2,
        "WEEK_HIGH": 3,
        "WEEK_LOW": 3,
        "SWING_HIGH": 4,
        "SWING_LOW": 4,
        "VWAP": 4,
        "EMA9": 4,
        "EMA20": 4,
        "EMA50": 4,
    }

    fib_tags = {"FIB0.618", "FIB0.786", "FIB1.0", "FIB1.272", "FIB1.618", "FIB2.0"}

    for item in list(symbol_levels):
        tag = (item.get("tag") or "").upper()
        level_value = item.get("level")
        if not isinstance(level_value, (int, float)):
            continue
        price = float(level_value)
        if bias == "long" and price <= entry:
            continue
        if bias == "short" and price >= entry:
            continue
        priority = priority_map.get(tag, 5 if tag in fib_tags else 6)
        distance = abs(price - entry)
        levels.append((priority, distance, price, tag))

    htf_levels = ctx.get("htf_levels") or []
    for raw in htf_levels:
        try:
            price = float(raw)
        except (TypeError, ValueError):
            continue
        if bias == "long" and price <= entry:
            continue
        if bias == "short" and price >= entry:
            continue
        distance = abs(price - entry)
        levels.append((3, distance, price, "HTF"))

    seen: set[float] = set()
    unique_levels: List[Tuple[int, float, float, str]] = []
    for priority, distance, price, tag in sorted(levels, key=lambda x: (x[0], x[1])):
        key = round(price, 4)
        if key in seen:
            continue
        seen.add(key)
        unique_levels.append((priority, distance, price, tag))
    if bias == "short":
        unique_levels.sort(key=lambda x: (x[0], x[2]), reverse=False)
    return unique_levels


def _choose_snapped_target(
    *,
    base_price: float,
    base_distance: float,
    entry: float,
    stop: float,
    bias: str,
    min_rr: float,
    tick_size: float,
    em_limit: float | None,
    levels: List[Tuple[int, float, float, str]],
    used_prices: List[float],
    previous_price: float | None,
    atr: float | None,
) -> Tuple[float, str, bool] | None:
    tolerance = max(tick_size * 0.6, 0.01)
    strong_tags = {
        "POC",
        "VAH",
        "VAL",
        "PRIOR_HIGH",
        "PRIOR_LOW",
        "ORB_HIGH",
        "ORB_LOW",
        "SESSION_HIGH",
        "SESSION_LOW",
        "WEEK_HIGH",
        "WEEK_LOW",
    }
    atr_cap = float(atr) * 1.75 if atr and atr > 0 else None
    distance_cap = max(base_distance * 1.8, atr_cap or 0.0)
    for _, _, price, tag in levels:
        if any(abs(price - used) <= tolerance for used in used_prices):
            continue
        if bias == "long":
            if price < base_price - tolerance:
                continue
            if previous_price is not None and price <= previous_price + tolerance:
                continue
            distance = price - entry
        else:
            if price > base_price + tolerance:
                continue
            if previous_price is not None and price >= previous_price - tolerance:
                continue
            distance = entry - price
        if distance_cap and distance > distance_cap * 1.001:
            continue
        allow_extension = tag in strong_tags
        if em_limit is not None:
            limit_allowance = em_limit * (1.10 if allow_extension else 1.0)
            if distance > limit_allowance * 1.001:
                continue
        if rr(entry, stop, price, bias) < float(min_rr) - 1e-6:
            continue
        return price, tag, allow_extension
    return None


def _structural_candidates(
    *,
    style_key: str,
    bias: str,
    entry: float,
    ctx: Dict[str, Any],
    structural_hint: float | None = None,
) -> List[Dict[str, Any]]:
    if style_key == "leaps":
        categories: List[Tuple[str, List[Tuple[str, float]]]] = [
            ("htf_weekly", ctx.get("levels_weekly") or []),
            ("volume_profile_weekly", _dict_to_level_list(ctx.get("vol_profile_weekly"), "WVP")),
            ("anchored_weekly", []),
            ("fib_weekly", _fib_candidate_list(style_key, bias, ctx)),
        ]
    elif style_key == "swing":
        categories = [
            ("htf_daily", ctx.get("levels_daily") or []),
            ("volume_profile_daily", _dict_to_level_list(ctx.get("vol_profile_daily"), "DVP")),
            ("anchored_daily", []),
            ("fib_daily", _fib_candidate_list(style_key, bias, ctx)),
        ]
    else:
        categories = [
            ("htf_intraday", _intraday_levels_list(ctx)),
            ("volume_profile_intraday", _dict_to_level_list(ctx.get("vol_profile"), "VP")),
            ("anchored_vwap_intraday", list((ctx.get("anchored_vwaps_intraday") or {}).items())),
            ("fib_intraday", _fib_candidate_list(style_key, bias, ctx)),
        ]

    if structural_hint is not None and math.isfinite(structural_hint):
        categories.insert(0, ("manual_hint", [("STRUCTURAL_HINT", float(structural_hint))]))

    seen: Dict[float, Dict[str, Any]] = {}
    for priority, (category, items) in enumerate(categories):
        for tag, value in items:
            try:
                price = float(value)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(price):
                continue
            if bias == "long" and price <= entry:
                continue
            if bias == "short" and price >= entry:
                continue
            distance = abs(price - entry)
            rounded = round(price, 4)
            payload = {
                "price": price,
                "tag": str(tag),
                "category": category,
                "priority": priority,
                "distance": distance,
            }
            existing = seen.get(rounded)
            if existing is None or (priority, distance) < (existing["priority"], existing["distance"]):
                seen[rounded] = payload
    candidates = list(seen.values())
    candidates.sort(key=lambda item: (item["priority"], item["distance"]))
    return candidates


def _apply_tp_logic(
    *,
    symbol: str,
    style: str | None,
    bias: str,
    entry: float,
    stop: float,
    base_targets: List[float],
    ctx: Dict[str, Any],
    min_rr: float,
    atr: float | None,
    expected_move: float | None,
    prefer_em_cap: bool,
    structural_tp1: float | None = None,
) -> Tuple[List[float], List[str], Dict[str, Any]]:
    tick_size = _price_increment(symbol, entry)
    style_key = _canonical_style_token(style)
    if style_key not in {"0dte", "scalp", "intraday", "swing", "leaps"}:
        style_key = "intraday"

    use_em_cap = prefer_em_cap and style_key in {"0dte", "scalp", "intraday"}
    em_limit = _expected_move_limit(expected_move if use_em_cap else None, use_em_cap)
    min_distance = _minimum_tp_distance(
        style_key=style_key,
        entry=entry,
        stop=stop,
        ctx=ctx,
        min_rr=min_rr,
        tick_size=tick_size,
    )

    stats_bundle = (ctx.get("target_stats") or {}).get(style_key)
    stats_targets = _targets_from_stats(
        style_key=style_key,
        bias=bias,
        entry=entry,
        min_distance=min_distance,
        em_limit=em_limit,
        stats=stats_bundle,
    )

    direction = 1 if bias == "long" else -1

    seeds: List[Dict[str, Any]] = []

    def _add_seed(base_price: float, meta: Dict[str, Any]) -> None:
        if not math.isfinite(base_price):
            return
        meta = dict(meta)
        meta["price_seed"] = float(base_price)
        seeds.append({"base": float(base_price), "meta": meta})

    if stats_targets:
        for idx, info in enumerate(stats_targets, start=1):
            distance_val = float(info["distance"])
            base_price = entry + direction * distance_val
            meta = dict(info["meta"])
            meta.setdefault("label", f"TP{idx}")
            meta.setdefault("source", "stats")
            meta["distance"] = distance_val
            meta["sequence"] = idx
            _add_seed(base_price, meta)

    fallback_targets = [float(t) for t in base_targets if math.isfinite(t)]
    if not seeds:
        fallback_targets = fallback_targets or [entry + direction * max(min_distance, tick_size)]
    if len(seeds) < 2:
        for offset, price in enumerate(fallback_targets, start=len(seeds) + 1):
            meta = {"label": f"TP{offset}", "source": "fallback", "sequence": offset}
            _add_seed(price, meta)
            if len(seeds) >= 3:
                break

    while len(seeds) < 2:
        offset = len(seeds) + 1
        constructed_price = entry + direction * max(min_distance * offset, tick_size)
        meta = {"label": f"TP{offset}", "source": "constructed", "sequence": offset}
        _add_seed(constructed_price, meta)

    candidates = _structural_candidates(
        style_key=style_key,
        bias=bias,
        entry=entry,
        ctx=ctx,
        structural_hint=structural_tp1,
    )

    warnings: List[str] = []
    if not seeds:
        warnings.append("TP1 constructed using minimum distance guardrail (no structural levels).")
        fallback_price = entry + direction * max(min_distance, tick_size)
        fallback_price = _round_to_increment(fallback_price, tick_size)
        fallback_price = _apply_limit_with_tick(entry, bias, fallback_price, em_limit, tick_size)
        _add_seed(fallback_price, {"label": "TP1", "source": "constructed", "sequence": 1})

    used_prices: set[float] = set()
    levels = _collect_priority_levels(entry, bias, ctx)

    def _distance(target_price: float) -> float:
        return abs(target_price - entry)

    def _within_em(distance_value: float, category: str) -> bool:
        if em_limit is None:
            return True
        allowance = 1.05 if category.startswith("htf") or "volume_profile" in category else 1.0
        return distance_value <= em_limit * allowance * 1.001

    final_pairs: List[Tuple[float, Dict[str, Any]]] = []
    target_debug: Dict[str, Any] = {
        "style": style_key,
        "min_distance": min_distance,
        "em_limit": em_limit,
        "stats_used": bool(stats_bundle),
        "seed_count": len(seeds),
    }

    for idx, seed in enumerate(seeds):
        candidate = float(seed["base"])
        meta = dict(seed["meta"])
        meta.setdefault("sequence", idx + 1)
        meta.setdefault("label", f"TP{idx + 1}")
        base_distance = _distance(candidate)
        if base_distance < min_distance:
            base_distance = max(min_distance, tick_size)
            candidate = entry + direction * base_distance
        previous_price = final_pairs[-1][0] if final_pairs else None
        snapped_choice = _choose_snapped_target(
            base_price=candidate,
            base_distance=_distance(candidate),
            entry=entry,
            stop=stop,
            bias=bias,
            min_rr=min_rr,
            tick_size=tick_size,
            em_limit=em_limit,
            levels=levels,
            used_prices=[price for price, _ in final_pairs],
            previous_price=previous_price,
            atr=atr,
        )
        snapped_price = None
        snapped_tag = None
        snapped_strong = False
        if snapped_choice is not None:
            snapped_price, snapped_tag, snapped_strong = snapped_choice

        effective_limit = em_limit
        if snapped_price is not None and em_limit is not None:
            effective_limit = em_limit * (1.10 if snapped_strong else 1.0)
        final_price = snapped_price if snapped_price is not None else candidate
        final_price = _clamp_to_em(entry, final_price, bias, effective_limit)
        final_price = _round_to_increment(final_price, tick_size)
        final_price = _apply_limit_with_tick(entry, bias, final_price, effective_limit, tick_size)

        rounded = round(final_price, 4)
        if rounded in used_prices:
            continue
        used_prices.add(rounded)

        distance_val = _distance(final_price)
        if distance_val < min_distance:
            distance_val = min_distance
            final_price = entry + direction * distance_val
            final_price = _round_to_increment(final_price, tick_size)

        meta["price"] = final_price
        meta["distance"] = distance_val
        if snapped_tag:
            meta["snap_tag"] = snapped_tag
        meta["rr"] = rr(entry, stop, final_price, bias)
        final_pairs.append((final_price, meta))

    if not final_pairs:
        warnings.append("No valid targets computed")
        default_price = entry + direction * max(min_distance, tick_size)
        final_pairs.append((default_price, {"label": "TP1", "price": default_price, "distance": max(min_distance, tick_size)}))

    while len(final_pairs) < 2:
        seq = len(final_pairs) + 1
        extra_price = entry + direction * max(min_distance * seq, tick_size * seq)
        extra_price = _round_to_increment(extra_price, tick_size)
        extra_price = _apply_limit_with_tick(entry, bias, extra_price, em_limit, tick_size)
        final_pairs.append((extra_price, {"label": f"TP{seq}", "price": extra_price, "distance": _distance(extra_price), "source": "constructed"}))

    if bias == "long":
        final_pairs.sort(key=lambda pair: pair[0])
    else:
        final_pairs.sort(key=lambda pair: pair[0], reverse=True)

    results = [price for price, _ in final_pairs]
    meta_list = [meta for _, meta in final_pairs]

    if len(results) >= 2:
        if bias == "long" and results[1] <= results[0]:
            adjusted = _round_to_increment(results[0] + max(tick_size, min_distance * 0.4), tick_size)
            adjusted = _apply_limit_with_tick(entry, bias, adjusted, em_limit, tick_size)
            results[1] = adjusted
            meta_list[1]["price"] = adjusted
            meta_list[1]["distance"] = _distance(adjusted)
        if bias == "short" and results[1] >= results[0]:
            adjusted = _round_to_increment(results[0] - max(tick_size, min_distance * 0.4), tick_size)
            adjusted = _apply_limit_with_tick(entry, bias, adjusted, em_limit, tick_size)
            results[1] = adjusted
            meta_list[1]["price"] = adjusted
            meta_list[1]["distance"] = _distance(adjusted)

    rr_tp1 = rr(entry, stop, results[0], bias)
    if rr_tp1 < float(min_rr) - 1e-6:
        atr_basis = _atr_for_style(style_key, ctx) or (atr if isinstance(atr, (int, float)) else None)
        risk = abs(entry - stop)
        nudge_basis = float(atr_basis) if atr_basis is not None else risk
        nudge = max(tick_size, nudge_basis * 0.15)
        attempt = results[0] + direction * nudge
        attempt = _round_to_increment(attempt, tick_size)
        attempt = _apply_limit_with_tick(entry, bias, attempt, em_limit, tick_size)
        if (bias == "long" and attempt <= entry) or (bias == "short" and attempt >= entry):
            attempt = results[0] + direction * max(tick_size, nudge * 0.5)
            attempt = _round_to_increment(attempt, tick_size)
            attempt = _apply_limit_with_tick(entry, bias, attempt, em_limit, tick_size)
        attempt_rr = rr(entry, stop, attempt, bias)
        target_debug["rr_adjustment"] = {
            "initial_rr": rr_tp1,
            "attempt_price": attempt,
            "attempt_rr": attempt_rr,
            "nudge": nudge,
        }
        if attempt_rr >= float(min_rr) - 1e-6:
            results[0] = attempt
            meta_list[0]["price"] = attempt
            meta_list[0]["distance"] = _distance(attempt)
            meta_list[0]["rr"] = attempt_rr
            if len(results) >= 2:
                if bias == "long" and results[1] <= results[0]:
                    adjusted = _round_to_increment(results[0] + max(tick_size, min_distance * 0.3), tick_size)
                    adjusted = _apply_limit_with_tick(entry, bias, adjusted, em_limit, tick_size)
                    results[1] = adjusted
                    meta_list[1]["price"] = adjusted
                    meta_list[1]["distance"] = _distance(adjusted)
                if bias == "short" and results[1] >= results[0]:
                    adjusted = _round_to_increment(results[0] - max(tick_size, min_distance * 0.3), tick_size)
                    adjusted = _apply_limit_with_tick(entry, bias, adjusted, em_limit, tick_size)
                    results[1] = adjusted
                    meta_list[1]["price"] = adjusted
                    meta_list[1]["distance"] = _distance(adjusted)
        else:
            warnings.append(f"TP1 R:R {attempt_rr:.2f} < {float(min_rr):.2f}; consider marking as watch plan")

    for idx, meta in enumerate(meta_list):
        meta["sequence"] = idx + 1
        meta["rr"] = rr(entry, stop, results[idx], bias)

    target_debug["meta"] = meta_list

    return results, meta_list, warnings, target_debug


def _strategy_min_rr(strategy_id: str) -> float:
    default = 1.2
    mapping = {
        "orb_retest": 1.3,
        "power_hour_trend": 1.4,
        "vwap_avwap": 1.3,
        "gap_fill_open": 1.5,
        "midday_mean_revert": 1.25,
    }
    return mapping.get(strategy_id.lower(), default)


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
    target_meta: List[Dict[str, Any]] | None,
    runner: Dict[str, Any] | None,
    atr_value: float | None,
    notes: str | None,
    conditions: Iterable[bool],
) -> Plan | None:
    if not math.isfinite(entry) or not math.isfinite(stop):
        return None
    # Clean and geometry-guard targets
    original_targets = [float(t) for t in targets if math.isfinite(t)]
    clean_targets = list(original_targets)
    geometry_reordered = False
    if direction == "long":
        ordered = sorted(clean_targets)
        if ordered != clean_targets:
            geometry_reordered = True
        clean_targets = ordered
    else:
        ordered = sorted(clean_targets, reverse=True)
        if ordered != clean_targets:
            geometry_reordered = True
        clean_targets = ordered
    if not clean_targets:
        return None
    # Geometry checks
    geometry_ok = True
    if direction == "long":
        if stop >= entry:
            geometry_ok = False
        risk = entry - stop
        reward = clean_targets[0] - entry
    else:
        if stop <= entry:
            geometry_ok = False
        risk = stop - entry
        reward = entry - clean_targets[0]
    if risk <= 0 or reward <= 0:
        return None
    risk_reward = reward / risk
    confidence = _score_conditions(conditions)
    # Append guardrail warning to notes if geometry was reordered or invalid
    warnings: List[str] = []
    if geometry_reordered or not geometry_ok:
        warnings.append("Geometry check: targets reordered or watch plan recommended")
        guard_note = "Geometry check: targets reordered or watch plan recommended"
    else:
        guard_note = None
    final_notes = notes
    if guard_note:
        final_notes = (notes + " | " + guard_note) if notes else guard_note

    return Plan(
        direction=direction,
        entry=float(entry),
        stop=float(stop),
        targets=clean_targets,
        target_meta=list(target_meta or [])[:len(clean_targets)],
        confidence=float(confidence),
        risk_reward=float(round(risk_reward, 3)),
        runner=runner,
        notes=final_notes,
        atr=float(atr_value) if atr_value is not None and math.isfinite(atr_value) else None,
        warnings=warnings,
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


def _infer_bar_minutes(index: pd.Index) -> float | None:
    if not isinstance(index, pd.DatetimeIndex) or index.size < 2:
        return None
    try:
        delta = (index[-1] - index[-2]).total_seconds() / 60.0
    except Exception:
        return None
    if not math.isfinite(delta) or delta <= 0:
        return None
    return float(delta)


def _resample_ohlcv(frame: pd.DataFrame, rule: str, min_length: int = 20) -> pd.DataFrame | None:
    if frame.empty or not isinstance(frame.index, pd.DatetimeIndex):
        return None
    rule_token = rule
    if isinstance(rule_token, str) and rule_token.endswith("T"):
        rule_token = f"{rule_token[:-1]}min"
    try:
        resampled = (
            frame[["open", "high", "low", "close", "volume"]]
            .resample(rule_token, label="right", closed="right")
            .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        )
    except Exception:
        return None
    resampled = resampled.dropna(subset=["open", "high", "low", "close"])
    if resampled.empty or len(resampled) < min_length:
        return None
    resampled = resampled.copy()
    resampled["typical_price"] = (resampled["high"] + resampled["low"] + resampled["close"]) / 3.0
    return resampled


def _latest_atr_value(frame: pd.DataFrame | None, period: int = 14) -> float | None:
    if frame is None or frame.empty or len(frame) < period:
        return None
    try:
        series = atr(frame["high"], frame["low"], frame["close"], period)
        value = float(series.iloc[-1])
    except Exception:
        return None
    if not math.isfinite(value) or value <= 0:
        return None
    return value


def _levels_from_frame(frame: pd.DataFrame | None, prefix: str) -> List[Tuple[str, float]]:
    if frame is None or frame.empty:
        return []
    levels: List[Tuple[str, float]] = []
    latest = frame.iloc[-1]
    mapping = {
        f"{prefix}_HIGH": latest.get("high"),
        f"{prefix}_LOW": latest.get("low"),
        f"{prefix}_CLOSE": latest.get("close"),
    }
    if len(frame) >= 2:
        prev = frame.iloc[-2]
        mapping[f"{prefix}_PREV_HIGH"] = prev.get("high")
        mapping[f"{prefix}_PREV_LOW"] = prev.get("low")
        mapping[f"{prefix}_PREV_CLOSE"] = prev.get("close")
    for tag, value in mapping.items():
        try:
            price = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(price):
            levels.append((tag, price))
    dedup: Dict[float, Tuple[str, float]] = {}
    for tag, price in levels:
        rounded = round(price, 4)
        if rounded not in dedup:
            dedup[rounded] = (tag, price)
    return list(dedup.values())


def _volume_profile_levels(frame: pd.DataFrame | None) -> Dict[str, float]:
    if frame is None or frame.empty:
        return {}
    try:
        profile = _volume_profile(frame)
    except Exception:
        profile = {}
    clean = {}
    for key, value in (profile or {}).items():
        try:
            price = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(price):
            clean[key] = price
    return clean


def _fib_extensions_from_frame(frame: pd.DataFrame | None, window: int = 50) -> Dict[str, List[Tuple[str, float]]]:
    if frame is None or frame.empty:
        return {"long": [], "short": []}
    window_frame = frame.tail(window)
    if window_frame.empty:
        return {"long": [], "short": []}
    try:
        swing_high = float(window_frame["high"].max())
        swing_low = float(window_frame["low"].min())
    except Exception:
        return {"long": [], "short": []}
    if not math.isfinite(swing_high) or not math.isfinite(swing_low):
        return {"long": [], "short": []}
    span = swing_high - swing_low
    if span <= 0 or not math.isfinite(span):
        return {"long": [], "short": []}
    long_levels = [
        ("FIB1.0", round(swing_high, 4)),
        ("FIB1.272", round(swing_high + span * 0.272, 4)),
        ("FIB1.618", round(swing_high + span * 0.618, 4)),
    ]
    short_levels = [
        ("FIB1.0", round(swing_low, 4)),
        ("FIB1.272", round(swing_low - span * 0.272, 4)),
        ("FIB1.618", round(swing_low - span * 0.618, 4)),
    ]
    return {"long": long_levels, "short": short_levels}


def _build_context(frame: pd.DataFrame) -> Dict[str, Any]:
    session_df, prev_session_df = _latest_sessions(frame)
    latest = frame.iloc[-1]
    atr_value = float(latest["atr14"]) if math.isfinite(latest["atr14"]) else math.nan
    volume_median = float(session_df["volume"].tail(40).median()) if not session_df.empty else math.nan
    minutes_vector = _minutes_from_midnight(session_df.index) if not session_df.empty else np.array([], dtype=int)
    # Expected move horizon (approx): median true range × bars over a short horizon
    expected_move_horizon = None
    try:
        df = frame.copy()
        close = df["close"]
        high = df["high"]
        low = df["low"]
        prev_close = close.shift(1)
        tr = pd.concat([(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
        tr_med = float(tr.tail(20).median()) if not tr.empty else float("nan")
        # infer bar interval in minutes
        try:
            idx = df.index
            if len(idx) >= 2:
                delta_min = (idx[-1] - idx[-2]).total_seconds() / 60.0
            else:
                delta_min = 5.0
        except Exception:
            delta_min = 5.0
        horizon_minutes = 30 if delta_min <= 2.0 else 60
        horizon_bars = max(1, int(horizon_minutes / max(delta_min, 1.0)))
        if math.isfinite(tr_med):
            expected_move_horizon = tr_med * horizon_bars
    except Exception:
        expected_move_horizon = None
    # Key levels (session OR + previous session H/L/C)
    key_levels: Dict[str, float] = {}
    try:
        if not session_df.empty:
            # Opening range (first 15 minutes)
            mins = _minutes_from_midnight(session_df.index)
            mask_or = (mins >= RTH_START_MINUTE) & (mins < RTH_START_MINUTE + 15)
            if mask_or.any():
                or_slice = session_df.iloc[mask_or]
                key_levels["opening_range_high"] = float(or_slice["high"].max())
                key_levels["opening_range_low"] = float(or_slice["low"].min())
            key_levels["session_high"] = float(session_df["high"].max())
            key_levels["session_low"] = float(session_df["low"].min())
        if prev_session_df is not None and not prev_session_df.empty:
            key_levels["prev_high"] = float(prev_session_df["high"].max())
            key_levels["prev_low"] = float(prev_session_df["low"].min())
            key_levels["prev_close"] = float(prev_session_df["close"].iloc[-1])
    except Exception:
        key_levels = {}

    # Volume profile (session)
    vol_profile = {}
    try:
        if not session_df.empty:
            vol_profile = _volume_profile(session_df)
    except Exception:
        vol_profile = {}

    # Fib anchors (up/down) from recent swing window (last 50 bars)
    fib_up: Dict[str, float] = {}
    fib_down: Dict[str, float] = {}
    try:
        window = frame.tail(50)
        rng_low = float(window["low"].min())
        rng_high = float(window["high"].max())
        span = max(0.0, rng_high - rng_low)
        if span > 0:
            # Upward projections from high
            fib_up = {
                "FIB1.0": round(rng_high, 4),
                "FIB1.272": round(rng_high + 0.272 * span, 4),
                "FIB1.618": round(rng_high + 0.618 * span, 4),
            }
            # Downward projections from low
            fib_down = {
                "FIB1.0": round(rng_low, 4),
                "FIB1.272": round(rng_low - 0.272 * span, 4),
                "FIB1.618": round(rng_low - 0.618 * span, 4),
            }
    except Exception:
        fib_up, fib_down = {}, {}

    bar_minutes = _infer_bar_minutes(frame.index)
    resampled_5m = _resample_ohlcv(frame, "5T") if bar_minutes is not None and bar_minutes <= 5.01 else None
    resampled_15m = _resample_ohlcv(frame, "15T") if bar_minutes is not None and bar_minutes <= 15.01 else None
    resampled_daily = _resample_ohlcv(frame, "1D", min_length=10)
    resampled_weekly = _resample_ohlcv(frame, "1W", min_length=10)

    atr_5m = _latest_atr_value(resampled_5m)
    atr_15m = _latest_atr_value(resampled_15m)
    atr_1d = _latest_atr_value(resampled_daily)
    atr_1w = _latest_atr_value(resampled_weekly)

    daily_levels = _levels_from_frame(resampled_daily, "DAILY")
    weekly_levels = _levels_from_frame(resampled_weekly, "WEEKLY")

    daily_profile = _volume_profile_levels(resampled_daily.tail(60) if resampled_daily is not None else None)
    weekly_profile = _volume_profile_levels(resampled_weekly.tail(30) if resampled_weekly is not None else None)

    fib_daily = _fib_extensions_from_frame(resampled_daily)
    fib_weekly = _fib_extensions_from_frame(resampled_weekly, window=26)

    anchored_vwaps_intraday: Dict[str, float] = {}
    try:
        if prev_session_df is not None and not prev_session_df.empty and not session_df.empty:
            prev_high_idx = prev_session_df["high"].idxmax()
            prev_low_idx = prev_session_df["low"].idxmin()
            session_open_idx = session_df.index[0]
            anchors = {
                "AVWAP_PREV_HIGH": _anchored_vwap(frame, prev_high_idx),
                "AVWAP_PREV_LOW": _anchored_vwap(frame, prev_low_idx),
                "AVWAP_SESSION_OPEN": _anchored_vwap(frame, session_open_idx),
            }
            anchored_vwaps_intraday = {
                tag: float(val)
                for tag, val in anchors.items()
                if val is not None and math.isfinite(float(val))
            }
    except Exception:
        anchored_vwaps_intraday = {}

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
        "htf_levels": _collect_htf_levels(session_df, prev_session_df, latest),
        "expected_move_horizon": expected_move_horizon,
        "key": key_levels,
        "vol_profile": vol_profile,
        "fib_up": fib_up,
        "fib_down": fib_down,
        "bar_minutes": bar_minutes,
        "atr_5m": atr_5m,
        "atr_15m": atr_15m,
        "atr_1d": atr_1d,
        "atr_1w": atr_1w,
        "levels_daily": daily_levels,
        "levels_weekly": weekly_levels,
        "vol_profile_daily": daily_profile,
        "vol_profile_weekly": weekly_profile,
        "fib_daily": fib_daily,
        "fib_weekly": fib_weekly,
        "anchored_vwaps_intraday": anchored_vwaps_intraday,
    }


# Trade-style presets for TP1 selection
_TP1_RULES: Dict[str, Dict[str, Any]] = {
    "scalp":    {"minATR": 0.25, "maxHorizon": 0.5, "ratioTP2": (0.25, 0.45), "weights": {"vwap": 0.4,  "orb": 0.3,  "priorHL": 0.2, "volProfile": 0.2, "fib": 0.1, "microPivot": 0.3}},
    "intraday": {"minATR": 0.35, "maxHorizon": 0.8, "ratioTP2": (0.35, 0.60), "weights": {"vwap": 0.35, "orb": 0.35, "priorHL": 0.25, "volProfile": 0.35, "fib": 0.2, "microPivot": 0.15}},
    "swing":    {"minATR": 0.50, "maxHorizon": 1.0, "ratioTP2": (0.40, 0.65), "weights": {"vwap": 0.15, "orb": 0.15, "priorHL": 0.35, "volProfile": 0.45, "fib": 0.35, "microPivot": 0.05}},
    "leaps":    {"minATR": 0.60, "maxHorizon": 1.0, "ratioTP2": (0.40, 0.70), "weights": {"vwap": 0.05, "orb": 0.05, "priorHL": 0.35, "volProfile": 0.45, "fib": 0.45, "microPivot": 0.00}},
}


def _style_for_strategy_id(strategy_id: str) -> str:
    return strategy_internal_category(strategy_id)


def _build_tp_candidates_long(entry: float, ctx: Dict[str, Any]) -> List[Dict[str, Any]]:
    c: List[Dict[str, Any]] = []
    key = ctx.get("key") or {}
    # Session structure
    for name, tag in (("prev_high", "PRIOR_HIGH"), ("opening_range_high", "ORB_HIGH")):
        val = key.get(name)
        if isinstance(val, (int, float)) and val > entry:
            c.append({"level": float(val), "tag": tag})
    # Volume Profile
    vp = ctx.get("vol_profile") or {}
    for lab, key in [("POC", "poc"), ("VAH", "vah"), ("VAL", "val")]:
        val = vp.get(key)
        if isinstance(val, (int, float)) and val > entry:
            c.append({"level": float(val), "tag": lab})
    # VWAP / EMAs
    v = ctx.get("vwap")
    if isinstance(v, (int, float)) and v > entry:
        c.append({"level": float(v), "tag": "VWAP"})
    for k in ("ema9", "ema20", "ema50"):
        val = ctx.get(k)
        if isinstance(val, (int, float)) and val > entry:
            c.append({"level": float(val), "tag": k.upper()})
    # Fib projections up
    fib_up = ctx.get("fib_up") or {}
    for tag, val in fib_up.items():
        if isinstance(val, (int, float)) and val > entry:
            c.append({"level": float(val), "tag": str(tag).upper()})
    # Dedupe by level within 1 cent
    seen: set[float] = set()
    uniq: List[Dict[str, Any]] = []
    for item in sorted(c, key=lambda x: x["level"]):
        keyf = round(item["level"], 2)
        if keyf in seen:
            continue
        seen.add(keyf)
        uniq.append(item)
    return uniq


def _build_tp_candidates_short(entry: float, ctx: Dict[str, Any]) -> List[Dict[str, Any]]:
    c: List[Dict[str, Any]] = []
    key = ctx.get("key") or {}
    # Session structure below entry
    for name, tag in (("prev_low", "PRIOR_LOW"), ("opening_range_low", "ORB_LOW")):
        val = key.get(name)
        if isinstance(val, (int, float)) and val < entry:
            c.append({"level": float(val), "tag": tag})
    # Volume profile
    vp = ctx.get("vol_profile") or {}
    for lab, k in [("POC", "poc"), ("VAL", "val"), ("VAH", "vah")]:
        val = vp.get(k)
        if isinstance(val, (int, float)) and val < entry:
            c.append({"level": float(val), "tag": lab})
    # VWAP/EMAs
    v = ctx.get("vwap")
    if isinstance(v, (int, float)) and v < entry:
        c.append({"level": float(v), "tag": "VWAP"})
    for k in ("ema9", "ema20", "ema50"):
        val = ctx.get(k)
        if isinstance(val, (int, float)) and val < entry:
            c.append({"level": float(val), "tag": k.upper()})
    # Fib projections down
    fib_down = ctx.get("fib_down") or {}
    for tag, val in fib_down.items():
        if isinstance(val, (int, float)) and val < entry:
            c.append({"level": float(val), "tag": str(tag).upper()})
    # Dedupe
    seen: set[float] = set()
    uniq: List[Dict[str, Any]] = []
    for item in sorted(c, key=lambda x: x["level"], reverse=True):
        keyf = round(item["level"], 2)
        if keyf in seen:
            continue
        seen.add(keyf)
        uniq.append(item)
    return uniq


def _score_tp_candidate_long(level: Dict[str, Any], style: str, entry: float, stop: float, tp2: float, ctx: Dict[str, Any], min_rr: float) -> Optional[float]:
    rules = _TP1_RULES.get(style, _TP1_RULES["intraday"])
    w = rules["weights"]
    s = 0.0
    tag = (level.get("tag") or "").upper()
    if tag == "VWAP":
        s += w.get("vwap", 0)
    if tag in {"ORB_HIGH"}:
        s += w.get("orb", 0)
    if tag in {"PRIOR_HIGH"}:
        s += w.get("priorHL", 0)
    if tag in {"EMA9", "EMA20", "EMA50"}:
        s += w.get("microPivot", 0)
    if tag in {"POC", "VAH", "VAL"}:
        s += w.get("volProfile", 0)
    if tag.startswith("FIB"):
        s += w.get("fib", 0)
    # MTF alignment proxy: EMA stack bullish → +0.1
    try:
        latest_bull = ctx.get("ema9") > ctx.get("ema20") > ctx.get("ema50")
        if latest_bull:
            s += 0.1
    except Exception:
        pass
    # Distance checks
    atr = ctx.get("atr") or 0.0
    min_atr = rules["minATR"] * float(atr or 0.0)
    dist = float(level.get("level") or 0.0) - entry
    if dist < max(0.0, min_atr):
        return None
    # EM cap
    em = ctx.get("expected_move_horizon")
    if isinstance(em, (int, float)) and dist > float(em) * float(rules.get("maxHorizon", 1.0)):
        return None
    # Ratio vs TP2
    span_tp2 = tp2 - entry
    if span_tp2 <= 0:
        return None
    ratio = dist / span_tp2
    rmin, rmax = rules["ratioTP2"]
    if not (rmin <= ratio <= rmax):
        return None
    # R:R
    risk = entry - stop
    if risk <= 0:
        return None
    rr_to_tp1 = dist / risk
    if rr_to_tp1 < float(min_rr):
        return None
    return round(float(s), 4)


def _score_tp_candidate_short(level: Dict[str, Any], style: str, entry: float, stop: float, tp2: float, ctx: Dict[str, Any], min_rr: float) -> Optional[float]:
    rules = _TP1_RULES.get(style, _TP1_RULES["intraday"])
    w = rules["weights"]
    s = 0.0
    tag = (level.get("tag") or "").upper()
    if tag == "VWAP":
        s += w.get("vwap", 0)
    if tag in {"ORB_LOW"}:
        s += w.get("orb", 0)
    if tag in {"PRIOR_LOW"}:
        s += w.get("priorHL", 0)
    if tag in {"EMA9", "EMA20", "EMA50"}:
        s += w.get("microPivot", 0)
    if tag in {"POC", "VAH", "VAL"}:
        s += w.get("volProfile", 0)
    if tag.startswith("FIB"):
        s += w.get("fib", 0)
    # EMA stack bearish bonus
    try:
        latest_bear = ctx.get("ema9") < ctx.get("ema20") < ctx.get("ema50")
        if latest_bear:
            s += 0.1
    except Exception:
        pass
    # Distance checks
    atr = ctx.get("atr") or 0.0
    min_atr = rules["minATR"] * float(atr or 0.0)
    dist = entry - float(level.get("level") or 0.0)
    if dist < max(0.0, min_atr):
        return None
    em = ctx.get("expected_move_horizon")
    if isinstance(em, (int, float)) and dist > float(em) * float(rules.get("maxHorizon", 1.0)):
        return None
    span_tp2 = entry - tp2
    if span_tp2 <= 0:
        return None
    ratio = dist / span_tp2
    rmin, rmax = rules["ratioTP2"]
    if not (rmin <= ratio <= rmax):
        return None
    risk = stop - entry
    if risk <= 0:
        return None
    rr_to_tp1 = dist / risk
    if rr_to_tp1 < float(min_rr):
        return None
    return round(float(s), 4)


def _select_tp1_long(entry: float, stop: float, tp2: float, style: str, ctx: Dict[str, Any], min_rr: float) -> Optional[float]:
    cands = _build_tp_candidates_long(entry, ctx)
    scored: List[Tuple[float, float]] = []  # (level, score)
    for L in cands:
        sc = _score_tp_candidate_long(L, style, entry, stop, tp2, ctx, min_rr)
        if sc is not None:
            scored.append((float(L["level"]), sc))
    if not scored:
        return None
    scored.sort(key=lambda x: x[1], reverse=True)
    return float(scored[0][0])


def _select_tp1_short(entry: float, stop: float, tp2: float, style: str, ctx: Dict[str, Any], min_rr: float) -> Optional[float]:
    cands = _build_tp_candidates_short(entry, ctx)
    scored: List[Tuple[float, float]] = []
    for L in cands:
        sc = _score_tp_candidate_short(L, style, entry, stop, tp2, ctx, min_rr)
        if sc is not None:
            scored.append((float(L["level"]), sc))
    if not scored:
        return None
    scored.sort(key=lambda x: x[1], reverse=True)
    return float(scored[0][0])


def _collect_htf_levels(session: pd.DataFrame, prev_session: pd.DataFrame | None, latest: pd.Series) -> List[float]:
    levels: List[float] = []
    try:
        if session is not None and not session.empty:
            levels.extend(
                [
                    float(session["high"].max()),
                    float(session["low"].min()),
                    float(session["close"].iloc[-1]),
                ]
            )
            head_slice = session.head(3)
            if not head_slice.empty:
                levels.extend(
                    [
                        float(head_slice["high"].max()),
                        float(head_slice["low"].min()),
                    ]
                )
        if prev_session is not None and not prev_session.empty:
            levels.extend(
                [
                    float(prev_session["high"].max()),
                    float(prev_session["low"].min()),
                    float(prev_session["close"].iloc[-1]),
                ]
            )
        vwap_val = latest.get("vwap")
        if math.isfinite(vwap_val):
            levels.append(float(vwap_val))
        ema50_val = latest.get("ema50")
        if math.isfinite(ema50_val):
            levels.append(float(ema50_val))
    except Exception:
        pass
    clean = [lvl for lvl in levels if math.isfinite(lvl)]
    return sorted(set(clean))


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
    tp1_dbg: Dict[str, Any] | None = None
    tp_debug_info: Dict[str, Any] | None = None

    recent_slice = post_range.tail(20)
    retest_low = float(recent_slice["low"].min()) if not recent_slice.empty else float("nan")
    retest_high = float(recent_slice["high"].max()) if not recent_slice.empty else float("nan")

    if price > or_high and math.isfinite(retest_low) and abs(retest_low - or_high) <= tolerance:
        entry = max(price, or_high)
        stop = retest_low - tolerance * 0.5
        min_rr = _strategy_min_rr(strategy.id)
        expected_move = ctx.get("expected_move_horizon")
        style = _style_for_strategy_id(strategy.id)
        base_targets = _base_targets_for_style(
            style=style,
            bias="long",
            entry=entry,
            stop=stop,
            atr=atr_value,
            expected_move=expected_move,
            min_rr=min_rr,
        )
        if not base_targets:
            default_span = atr_value if atr_value > 0 else abs(entry - stop)
            default_span = default_span if default_span and default_span > 0 else max(abs(entry - stop), 1.0)
            base_targets = [entry + default_span, entry + default_span * 1.5]
        if len(base_targets) == 1:
            bump = atr_value if atr_value and atr_value > 0 else abs(entry - stop)
            base_targets.append(base_targets[0] + (bump or 1.0))
        tp2_seed = base_targets[1] if len(base_targets) >= 2 else base_targets[0]
        tp1_struct = _select_tp1_long(entry, stop, tp2_seed, style, ctx, min_rr)
        targets, target_meta, tp_warnings, tp_debug = _apply_tp_logic(
            symbol=symbol,
            style=style,
            bias="long",
            entry=entry,
            stop=stop,
            base_targets=base_targets,
            ctx=ctx,
            min_rr=min_rr,
            atr=atr_value,
            expected_move=expected_move,
            prefer_em_cap=True,
            structural_tp1=tp1_struct,
        )
        tp_debug_info = tp_debug
        if tp1_struct is not None:
            tp1_dbg = {"picked": tp1_struct, "base_targets": base_targets[:2]}
        if not targets:
            return None
        runner_cfg = _runner_config(style, "long", entry, stop, targets)
        plan = _build_plan(
            "long",
            entry,
            stop,
            targets,
            target_meta=target_meta,
            runner=runner_cfg,
            atr_value=atr_value,
            notes=f"Reclaimed OR high {or_high:.2f}; retest low {retest_low:.2f}",
            conditions=[ema_stack_long, adx_strong, volume_ok],
        )
        if plan:
            if tp_warnings:
                plan.warnings.extend(tp_warnings)
            plan.target_meta = target_meta
            plan.runner = runner_cfg
            notes.append("Long OR retest validated")

    elif price < or_low and math.isfinite(retest_high) and abs(retest_high - or_low) <= tolerance:
        entry = min(price, or_low)
        stop = retest_high + tolerance * 0.5
        style = _style_for_strategy_id(strategy.id)
        min_rr = _strategy_min_rr(strategy.id)
        expected_move = ctx.get("expected_move_horizon")
        base_targets = _base_targets_for_style(
            style=style,
            bias="short",
            entry=entry,
            stop=stop,
            atr=atr_value,
            expected_move=expected_move,
            min_rr=min_rr,
        )
        if not base_targets:
            default_span = atr_value if atr_value > 0 else abs(entry - stop)
            default_span = default_span if default_span and default_span > 0 else max(abs(entry - stop), 1.0)
            base_targets = [entry - default_span, entry - default_span * 1.5]
        if len(base_targets) == 1:
            bump = atr_value if atr_value and atr_value > 0 else abs(entry - stop)
            base_targets.append(base_targets[0] - (bump or 1.0))
        tp2_seed = base_targets[1] if len(base_targets) >= 2 else base_targets[0]
        tp1_struct = _select_tp1_short(entry, stop, tp2_seed, style, ctx, min_rr)
        targets, target_meta, tp_warnings, tp_debug = _apply_tp_logic(
            symbol=symbol,
            style=style,
            bias="short",
            entry=entry,
            stop=stop,
            base_targets=base_targets,
            ctx=ctx,
            min_rr=min_rr,
            atr=atr_value,
            expected_move=expected_move,
            prefer_em_cap=True,
            structural_tp1=tp1_struct,
        )
        tp_debug_info = tp_debug
        if tp1_struct is not None:
            tp1_dbg = {"picked": tp1_struct, "base_targets": base_targets[:2]}
        if not targets:
            return None
        runner_cfg = _runner_config(style, "short", entry, stop, targets)
        plan = _build_plan(
            "short",
            entry,
            stop,
            targets,
            target_meta=target_meta,
            runner=runner_cfg,
            atr_value=atr_value,
            notes=f"Rejected OR low {or_low:.2f}; retest high {retest_high:.2f}",
            conditions=[ema_stack_short, adx_strong, volume_ok],
        )
        if plan:
            if tp_warnings:
                plan.warnings.extend(tp_warnings)
            plan.target_meta = target_meta
            plan.runner = runner_cfg
            notes.append("Short OR retest validated")

    if plan is None:
        return None

    profile = _enrich_plan_with_profile(
        plan,
        style=style,
        bias=plan.direction,
        atr_value=atr_value,
        expected_move=ctx.get("expected_move_horizon"),
        debug=tp_debug_info,
    )

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
        "plan_target_meta": plan.target_meta,
        "plan_runner": plan.runner,
        "plan_confidence": plan.confidence,
        "plan_risk_reward": plan.risk_reward,
        "plan_notes": plan.notes,
    }
    if plan.warnings:
        features["plan_warnings"] = list(plan.warnings)
    if plan.warnings:
        features["plan_warnings"] = list(plan.warnings)
    if tp1_dbg:
        features["tp1_struct_debug"] = tp1_dbg
    if tp_debug_info:
        features["tp_targets_debug"] = tp_debug_info
    features["target_profile"] = profile.to_dict()
    features["target_probabilities"] = profile.probabilities

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

    tp1_dbg_ph: Dict[str, Any] | None = None
    tp_debug_info_ph: Dict[str, Any] | None = None
    if price > ctx["vwap"] and ema_stack_long and breakout_long:
        entry = price
        stop = min(range_low, float(session["low"].tail(10).min())) - atr_value * 0.25
        style = _style_for_strategy_id(strategy.id)
        min_rr = _strategy_min_rr(strategy.id)
        expected_move = ctx.get("expected_move_horizon")
        base_targets = _base_targets_for_style(
            style=style,
            bias="long",
            entry=entry,
            stop=stop,
            atr=atr_value,
            expected_move=expected_move,
            min_rr=min_rr,
        )
        if not base_targets:
            default_span = atr_value if atr_value > 0 else abs(entry - stop)
            default_span = default_span if default_span and default_span > 0 else max(abs(entry - stop), 1.0)
            base_targets = [entry + default_span, entry + default_span * 1.5]
        if len(base_targets) == 1:
            bump = atr_value if atr_value and atr_value > 0 else abs(entry - stop)
            base_targets.append(base_targets[0] + (bump or 1.0))
        tp2_seed = base_targets[1] if len(base_targets) >= 2 else base_targets[0]
        tp1_struct = _select_tp1_long(entry, stop, tp2_seed, style, ctx, min_rr)
        targets, target_meta, tp_warnings, tp_debug = _apply_tp_logic(
            symbol=symbol,
            style=style,
            bias="long",
            entry=entry,
            stop=stop,
            base_targets=base_targets,
            ctx=ctx,
            min_rr=min_rr,
            atr=atr_value,
            expected_move=expected_move,
            prefer_em_cap=True,
            structural_tp1=tp1_struct,
        )
        tp_debug_info_ph = tp_debug
        if tp1_struct is not None:
            tp1_dbg_ph = {"picked": tp1_struct, "base_targets": base_targets[:2]}
        if not targets:
            return None
        runner_cfg = _runner_config(style, "long", entry, stop, targets)
        plan = _build_plan(
            "long",
            entry,
            stop,
            targets,
            target_meta=target_meta,
            runner=runner_cfg,
            atr_value=atr_value,
            notes=f"VWAP support {ctx['vwap']:.2f}; afternoon range high {range_high:.2f}",
            conditions=[adx_strong, volume_ok, breakout_long],
        )
        if plan and tp_warnings:
            plan.warnings.extend(tp_warnings)
        if plan:
            plan.target_meta = target_meta
            plan.runner = runner_cfg
    elif price < ctx["vwap"] and ema_stack_short and breakout_short:
        entry = price
        stop = max(range_high, float(session["high"].tail(10).max())) + atr_value * 0.25
        style = _style_for_strategy_id(strategy.id)
        min_rr = _strategy_min_rr(strategy.id)
        expected_move = ctx.get("expected_move_horizon")
        base_targets = _base_targets_for_style(
            style=style,
            bias="short",
            entry=entry,
            stop=stop,
            atr=atr_value,
            expected_move=expected_move,
            min_rr=min_rr,
        )
        if not base_targets:
            default_span = atr_value if atr_value > 0 else abs(entry - stop)
            default_span = default_span if default_span and default_span > 0 else max(abs(entry - stop), 1.0)
            base_targets = [entry - default_span, entry - default_span * 1.5]
        if len(base_targets) == 1:
            bump = atr_value if atr_value and atr_value > 0 else abs(entry - stop)
            base_targets.append(base_targets[0] - (bump or 1.0))
        tp2_seed = base_targets[1] if len(base_targets) >= 2 else base_targets[0]
        tp1_struct = _select_tp1_short(entry, stop, tp2_seed, style, ctx, min_rr)
        targets, target_meta, tp_warnings, tp_debug = _apply_tp_logic(
            symbol=symbol,
            style=style,
            bias="short",
            entry=entry,
            stop=stop,
            base_targets=base_targets,
            ctx=ctx,
            min_rr=min_rr,
            atr=atr_value,
            expected_move=expected_move,
            prefer_em_cap=True,
            structural_tp1=tp1_struct,
        )
        tp_debug_info_ph = tp_debug
        if tp1_struct is not None:
            tp1_dbg_ph = {"picked": tp1_struct, "base_targets": base_targets[:2]}
        if not targets:
            return None
        runner_cfg = _runner_config(style, "short", entry, stop, targets)
        plan = _build_plan(
            "short",
            entry,
            stop,
            targets,
            target_meta=target_meta,
            runner=runner_cfg,
            atr_value=atr_value,
            notes=f"VWAP resistance {ctx['vwap']:.2f}; afternoon range low {range_low:.2f}",
            conditions=[adx_strong, volume_ok, breakout_short],
        )
        if plan and tp_warnings:
            plan.warnings.extend(tp_warnings)
        if plan:
            plan.target_meta = target_meta
            plan.runner = runner_cfg

    if plan is None:
        return None

    profile = _enrich_plan_with_profile(
        plan,
        style=style,
        bias=plan.direction,
        atr_value=atr_value,
        expected_move=ctx.get("expected_move_horizon"),
        debug=tp_debug_info_ph,
    )

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
        "plan_target_meta": plan.target_meta,
        "plan_runner": plan.runner,
        "plan_confidence": plan.confidence,
        "plan_risk_reward": plan.risk_reward,
        "plan_notes": plan.notes,
    }
    if plan.warnings:
        features["plan_warnings"] = list(plan.warnings)
    if tp1_dbg_ph:
        features["tp1_struct_debug"] = tp1_dbg_ph
    if tp_debug_info_ph:
        features["tp_targets_debug"] = tp_debug_info_ph
    features["target_profile"] = profile.to_dict()
    features["target_probabilities"] = profile.probabilities

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
        style = _style_for_strategy_id(strategy.id)
        min_rr = _strategy_min_rr(strategy.id)
        expected_move = ctx.get("expected_move_horizon")
        base_targets = _base_targets_for_style(
            style=style,
            bias="long",
            entry=entry,
            stop=stop,
            atr=atr_value,
            expected_move=expected_move,
            min_rr=min_rr,
        )
        if not base_targets:
            default_span = atr_value if atr_value > 0 else abs(entry - stop)
            default_span = default_span if default_span and default_span > 0 else max(abs(entry - stop), 1.0)
            base_targets = [entry + default_span, entry + default_span * 1.6]
        if len(base_targets) == 1:
            bump = atr_value if atr_value and atr_value > 0 else abs(entry - stop)
            base_targets.append(base_targets[0] + (bump or 1.0))
        targets, target_meta, tp_warnings, tp_debug = _apply_tp_logic(
            symbol=symbol,
            style=style,
            bias="long",
            entry=entry,
            stop=stop,
            base_targets=base_targets,
            ctx=ctx,
            min_rr=min_rr,
            atr=atr_value,
            expected_move=expected_move,
            prefer_em_cap=True,
        )
        tp_warnings_result = tp_warnings
        tp_debug_info = tp_debug
        if not targets:
            return None
        runner_cfg = _runner_config(style, "long", entry, stop, targets)
        plan = _build_plan(
            "long",
            entry,
            stop,
            targets,
            target_meta=target_meta,
            runner=runner_cfg,
            atr_value=atr_value,
            notes=f"Above VWAP cluster (~{cluster_mean:.2f}); spread {cluster_spread:.2f}",
            conditions=[cluster_tight, adx_ok],
        )
        if plan and tp_warnings_result:
            plan.warnings.extend(tp_warnings_result)
        if plan:
            plan.target_meta = target_meta
            plan.runner = runner_cfg
    elif price < ctx["vwap"] and ema_stack_short and cluster_tight and price < cluster_mean:
        entry = price
        stop = max(cluster_mean, np.max(anchored_values)) + tolerance
        style = _style_for_strategy_id(strategy.id)
        min_rr = _strategy_min_rr(strategy.id)
        expected_move = ctx.get("expected_move_horizon")
        base_targets = _base_targets_for_style(
            style=style,
            bias="short",
            entry=entry,
            stop=stop,
            atr=atr_value,
            expected_move=expected_move,
            min_rr=min_rr,
        )
        if not base_targets:
            default_span = atr_value if atr_value > 0 else abs(entry - stop)
            default_span = default_span if default_span and default_span > 0 else max(abs(entry - stop), 1.0)
            base_targets = [entry - default_span, entry - default_span * 1.6]
        if len(base_targets) == 1:
            bump = atr_value if atr_value and atr_value > 0 else abs(entry - stop)
            base_targets.append(base_targets[0] - (bump or 1.0))
        targets, target_meta, tp_warnings, tp_debug = _apply_tp_logic(
            symbol=symbol,
            style=style,
            bias="short",
            entry=entry,
            stop=stop,
            base_targets=base_targets,
            ctx=ctx,
            min_rr=min_rr,
            atr=atr_value,
            expected_move=expected_move,
            prefer_em_cap=True,
        )
        tp_warnings_result = tp_warnings
        tp_debug_info = tp_debug
        if not targets:
            return None
        runner_cfg = _runner_config(style, "short", entry, stop, targets)
        plan = _build_plan(
            "short",
            entry,
            stop,
            targets,
            target_meta=target_meta,
            runner=runner_cfg,
            atr_value=atr_value,
            notes=f"Below VWAP cluster (~{cluster_mean:.2f}); spread {cluster_spread:.2f}",
            conditions=[cluster_tight, adx_ok],
        )
        if plan and tp_warnings_result:
            plan.warnings.extend(tp_warnings_result)
        if plan:
            plan.target_meta = target_meta
            plan.runner = runner_cfg

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
        "plan_target_meta": plan.target_meta,
        "plan_runner": plan.runner,
        "plan_confidence": plan.confidence,
        "plan_risk_reward": plan.risk_reward,
        "plan_notes": plan.notes,
    }
    if plan.warnings:
        features["plan_warnings"] = list(plan.warnings)

    if tp_debug_info:
        features["tp_targets_debug"] = tp_debug_info

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

    style_token = _style_for_strategy_id(strategy.id)
    style_key = _canonical_style_token(style_token)
    target_meta = [
        {"label": "TP1", "price": target_primary, "sequence": 1, "source": "gap"},
        {"label": "TP2", "price": target_secondary, "sequence": 2, "source": "gap"},
    ]
    runner_cfg = _runner_config(style_key, direction, entry, stop, [target_primary, target_secondary])

    plan = _build_plan(
        direction,
        entry,
        stop,
        [target_primary, target_secondary],
        target_meta=target_meta,
        runner=runner_cfg,
        atr_value=atr_value,
        notes=f"Gap {gap:+.2f} vs prev close {prev_close:.2f}; progress {progress:.2%}",
        conditions=[vwap_alignment, volume_ok, distance_to_close > atr_value * 0.2],
    )
    if plan is None:
        return None

    profile = _enrich_plan_with_profile(
        plan,
        style=style_token,
        bias=plan.direction,
        atr_value=atr_value,
        expected_move=ctx.get("expected_move_horizon"),
        debug=None,
    )

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
        "plan_target_meta": plan.target_meta,
        "plan_runner": plan.runner,
        "plan_confidence": plan.confidence,
        "plan_risk_reward": plan.risk_reward,
        "plan_notes": plan.notes,
    }
    if plan.warnings:
        features["plan_warnings"] = list(plan.warnings)
    features["target_profile"] = profile.to_dict()
    features["target_probabilities"] = profile.probabilities

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

    style_key = _canonical_style_token(_style_for_strategy_id(strategy.id))
    target_meta = [
        {"label": "TP1", "price": target_primary, "sequence": 1, "source": "vwap_mean"},
        {"label": "TP2", "price": target_secondary, "sequence": 2, "source": "vwap_mean"},
    ]
    runner_cfg = _runner_config(style_key, direction, entry, stop, [target_primary, target_secondary])

    plan = _build_plan(
        direction,
        entry,
        stop,
        [target_primary, target_secondary],
        target_meta=target_meta,
        runner=runner_cfg,
        atr_value=atr_value,
        notes=f"VWAP {ctx['vwap']:.2f}; extension {extension:.2f} ({extension/atr_value:.2f}× ATR)",
        conditions=[adx_weak, range_contraction, volume_light],
    )
    if plan is None:
        return None

    profile = _enrich_plan_with_profile(
        plan,
        style=_style_for_strategy_id(strategy.id),
        bias=plan.direction,
        atr_value=atr_value,
        expected_move=ctx.get("expected_move_horizon"),
        debug=None,
    )

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
        "plan_target_meta": plan.target_meta,
        "plan_runner": plan.runner,
        "plan_confidence": plan.confidence,
        "plan_risk_reward": plan.risk_reward,
        "plan_notes": plan.notes,
    }
    features["target_profile"] = profile.to_dict()
    features["target_probabilities"] = profile.probabilities

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
        ctx.setdefault("target_stats", {})
        style_stats_cache: Dict[str, Dict[str, object] | None] = {}

        for strategy in strategies:
            detector = STRATEGY_DETECTORS.get(strategy.id)
            if detector is None:
                continue
            style_token = _style_for_strategy_id(strategy.id)
            style_key = _canonical_style_token(style_token)
            if style_key and style_key not in style_stats_cache:
                style_stats_cache[style_key] = await get_style_stats(symbol, style_key)
            stats_bundle = style_stats_cache.get(style_key)
            if stats_bundle:
                ctx["target_stats"][style_key] = stats_bundle
            signal = detector(symbol, strategy, ctx)
            if signal is None:
                continue
            signals.append(signal)

    signals.sort(key=lambda sig: sig.score, reverse=True)
    return signals
