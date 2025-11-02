"""Utilities to construct plan-bound chart layers."""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .precision import get_price_precision


def _is_number(value: Any) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def _parse_conditions(waiting_for: str) -> List[str]:
    """Split compact waiting_for strings into individual tokens."""
    token = (waiting_for or "").strip()
    if not token:
        return []
    parts = token.replace(" and ", " + ").split("+")
    return [part.strip() for part in parts if part and part.strip()]


def _pick_level(name: str, key_levels: Mapping[str, Any]) -> Optional[float]:
    if not isinstance(key_levels, Mapping):
        return None
    lookup = {
        "vwap": ("VWAP", "vwap", "prior_vwap", "pd_vwap"),
        "orh": ("ORH", "opening_range_high", "openingHigh"),
        "orl": ("ORL", "opening_range_low", "openingLow"),
        "pdh": ("PDH", "prior_day_high"),
        "pdl": ("PDL", "prior_day_low"),
        "pivot": ("Pivot", "pivot"),
        "r1": ("R1", "r1"),
        "s1": ("S1", "s1"),
    }.get(name.lower(), (name,))
    for candidate in lookup:
        value = key_levels.get(candidate)
        if isinstance(value, (int, float)):
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return None


def _coerce_float(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _normalize_direction(
    direction: str | None,
    *,
    plan: Mapping[str, Any] | None = None,
) -> Optional[str]:
    for candidate in (direction,):
        if isinstance(candidate, str):
            token = candidate.strip().lower()
            if token in {"long", "short"}:
                return token
    if isinstance(plan, Mapping):
        for key in ("direction", "bias"):
            value = plan.get(key)
            if isinstance(value, str):
                token = value.strip().lower()
                if token in {"long", "short"}:
                    return token
    return None


def _resolve_nested(plan: Mapping[str, Any] | None, path: Sequence[str]) -> Any:
    node: Any = plan
    for key in path:
        if not isinstance(node, Mapping):
            return None
        node = node.get(key)
    return node


def _resolve_entry(plan: Mapping[str, Any] | None) -> Optional[float]:
    if not isinstance(plan, Mapping):
        return None
    entry_candidate = plan.get("entry")
    if isinstance(entry_candidate, Mapping):
        entry_candidate = entry_candidate.get("level") or entry_candidate.get("price")
    entry = _coerce_float(entry_candidate)
    if entry is not None:
        return entry
    structured = plan.get("structured_plan")
    if isinstance(structured, Mapping):
        structured_entry = structured.get("entry")
        if isinstance(structured_entry, Mapping):
            level = _coerce_float(structured_entry.get("level"))
            if level is not None:
                return level
    target_profile = plan.get("target_profile")
    if isinstance(target_profile, Mapping):
        candidate = _coerce_float(target_profile.get("entry"))
        if candidate is not None:
            return candidate
    return None


def _extract_atr(plan: Mapping[str, Any] | None) -> Optional[float]:
    if not isinstance(plan, Mapping):
        return None
    candidates: List[Any] = [
        plan.get("atr_used"),
        plan.get("atr"),
        plan.get("atr_value"),
        plan.get("remaining_atr"),
    ]
    structured = plan.get("structured_plan")
    if isinstance(structured, Mapping):
        candidates.extend([structured.get("atr_used"), structured.get("atr"), structured.get("remaining_atr")])
        strategy_profile = structured.get("strategy_profile")
        if isinstance(strategy_profile, Mapping):
            candidates.append(strategy_profile.get("atr_used"))
    target_profile = plan.get("target_profile")
    if isinstance(target_profile, Mapping):
        candidates.append(target_profile.get("atr_used"))
    planning_snapshot = plan.get("planning_snapshot")
    if isinstance(planning_snapshot, Mapping):
        candidates.extend(
            [
                planning_snapshot.get("atr_used"),
                planning_snapshot.get("atr"),
            ]
        )
    for candidate in candidates:
        atr_val = _coerce_float(candidate)
        if atr_val is not None and atr_val > 0:
            return atr_val
    return None


def _collect_numeric_points(plan: Mapping[str, Any] | None) -> List[float]:
    if not isinstance(plan, Mapping):
        return []
    values: List[float] = []

    for key in ("entry", "stop", "invalid"):
        val = plan.get(key)
        if isinstance(val, Mapping):
            val = val.get("level") or val.get("price")
        numeric = _coerce_float(val)
        if numeric is not None:
            values.append(numeric)

    targets = plan.get("targets")
    if isinstance(targets, Sequence):
        for target in targets:
            candidate = target
            if isinstance(target, Mapping):
                candidate = target.get("price") or target.get("level") or target.get("target")
            numeric = _coerce_float(candidate)
            if numeric is not None:
                values.append(numeric)

    key_levels = plan.get("key_levels")
    if isinstance(key_levels, Mapping):
        for value in key_levels.values():
            numeric = _coerce_float(value)
            if numeric is not None:
                values.append(numeric)

    planning_snapshot = plan.get("planning_snapshot")
    if isinstance(planning_snapshot, Mapping):
        for key in ("entry", "stop", "tp1", "tp2"):
            numeric = _coerce_float(planning_snapshot.get(key))
            if numeric is not None:
                values.append(numeric)

    return values


def _collect_overlay_points(overlays: Mapping[str, Any] | None) -> List[float]:
    if not isinstance(overlays, Mapping):
        return []
    values: List[float] = []
    volume_profile = overlays.get("volume_profile")
    if isinstance(volume_profile, Mapping):
        for key in ("vwap", "vah", "val", "poc"):
            numeric = _coerce_float(volume_profile.get(key))
            if numeric is not None:
                values.append(numeric)
    liquidity_pools = overlays.get("liquidity_pools")
    if isinstance(liquidity_pools, Sequence):
        for pool in liquidity_pools:
            level = None
            if isinstance(pool, Mapping):
                level = pool.get("level")
            numeric = _coerce_float(level)
            if numeric is not None:
                values.append(numeric)
    supply_zones = overlays.get("supply_zones")
    demand_zones = overlays.get("demand_zones")
    for collection in (supply_zones, demand_zones):
        if not isinstance(collection, Sequence):
            continue
        for zone in collection:
            if isinstance(zone, Mapping):
                low = _coerce_float(zone.get("low"))
                high = _coerce_float(zone.get("high"))
                if low is not None:
                    values.append(low)
                if high is not None:
                    values.append(high)
    fib_levels = overlays.get("fib_levels")
    if isinstance(fib_levels, Mapping):
        for branch in ("up", "down"):
            branch_map = fib_levels.get(branch)
            if isinstance(branch_map, Mapping):
                for value in branch_map.values():
                    numeric = _coerce_float(value)
                    if numeric is not None:
                        values.append(numeric)
    avwap = overlays.get("avwap")
    if isinstance(avwap, Mapping):
        for value in avwap.values():
            numeric = _coerce_float(value)
            if numeric is not None:
                values.append(numeric)
    return values


def _min_positive_step(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    unique = sorted({round(val, 6) for val in values if math.isfinite(val)})
    if len(unique) < 2:
        return None
    min_step: Optional[float] = None
    for idx in range(1, len(unique)):
        delta = unique[idx] - unique[idx - 1]
        if delta <= 0:
            continue
        if delta < 1e-6:
            continue
        if min_step is None or delta < min_step:
            min_step = delta
    return min_step


def _resolve_tick_size(
    plan: Mapping[str, Any] | None,
    key_levels: Mapping[str, Any] | None,
    overlays: Mapping[str, Any] | None,
    *,
    precision: int,
) -> float:
    candidates: List[float] = []
    if isinstance(plan, Mapping):
        for token in ("tick_size", "tick"):
            val = _coerce_float(plan.get(token))
            if val is not None and val > 0:
                candidates.append(val)
        nested_paths = [
            ("meta", "tick_size"),
            ("planning_snapshot", "tick_size"),
            ("planning_snapshot", "tick"),
            ("planning_snapshot", "runner_policy", "tick_size"),
            ("target_profile", "tick_size"),
            ("target_profile", "meta", "tick_size"),
            ("target_profile", "runner_policy", "tick_size"),
            ("structured_plan", "tick_size"),
            ("structured_plan", "tick"),
            ("structured_plan", "runner", "tick_size"),
        ]
        for path in nested_paths:
            candidate = _resolve_nested(plan, path)
            numeric = _coerce_float(candidate)
            if numeric is not None and numeric > 0:
                candidates.append(numeric)
    if candidates:
        return min(candidates)

    numeric_points: List[float] = []
    numeric_points.extend(_collect_numeric_points(plan))
    if isinstance(key_levels, Mapping):
        for value in key_levels.values():
            numeric = _coerce_float(value)
            if numeric is not None:
                numeric_points.append(numeric)
    numeric_points.extend(_collect_overlay_points(overlays))

    inferred = _min_positive_step(numeric_points)
    if inferred is not None and inferred > 0:
        return inferred

    if precision < 0:
        return 0.01
    tick = 10 ** (-precision)
    # Guard against over-rounding for coarse instruments (precision=0 -> tick=1)
    if precision > 0:
        tick = round(tick, precision + 2)
    return max(tick, 1e-4)


def _normalize_timeframe(interval: str | None) -> str:
    token = (interval or "").strip().lower()
    mapping = {
        "1": "1m",
        "1m": "1m",
        "3": "1m",
        "3m": "1m",
        "5": "5m",
        "5m": "5m",
        "10": "5m",
        "10m": "5m",
        "15": "15m",
        "15m": "15m",
        "30": "15m",
        "30m": "15m",
        "45": "60m",
        "45m": "60m",
        "60": "60m",
        "60m": "60m",
        "65m": "60m",
        "1h": "60m",
        "2h": "60m",
        "240": "60m",
        "240m": "60m",
        "1d": "1D",
        "d": "1D",
        "day": "1D",
    }
    if token in mapping:
        return mapping[token]
    if token.endswith("m") and token[:-1].isdigit():
        minutes = int(token[:-1])
        if minutes <= 2:
            return "1m"
        if minutes <= 7:
            return "5m"
        if minutes <= 30:
            return "15m"
        return "60m"
    if token.endswith("h"):
        return "60m"
    return "5m"


def _progress_for_price(last_price: Optional[float], band_low: float, band_high: float, half_width: float) -> float:
    """Return normalized progress where 1.0 indicates the price is inside the band."""

    if last_price is None or not math.isfinite(last_price) or half_width <= 0:
        return 0.0

    center = (band_low + band_high) / 2.0
    # Measure how far price sits outside the band; inside the band counts as full progress.
    distance_from_center = abs(last_price - center)
    excess = max(0.0, distance_from_center - half_width)
    if excess <= 0:
        return 1.0
    progress = 1.0 - (excess / half_width)
    return max(0.0, min(1.0, progress))


def _state_for_price(direction: Optional[str], last_price: Optional[float], band_low: float, band_high: float) -> str:
    if direction not in {"long", "short"} or last_price is None or not math.isfinite(last_price):
        return "invalid"
    if direction == "long":
        if last_price < band_low:
            return "arming"
        if last_price <= band_high:
            return "ready"
        return "cooldown"
    # direction == "short"
    if last_price > band_high:
        return "arming"
    if last_price >= band_low:
        return "ready"
    return "cooldown"


def _unique_tokens(values: Iterable[Any]) -> List[str]:
    tokens: List[str] = []
    seen: set[str] = set()
    for value in values:
        if not value:
            continue
        text = str(value).strip()
        if not text:
            continue
        if text not in seen:
            seen.add(text)
            tokens.append(text)
    return tokens


def _extract_confluence(
    plan: Mapping[str, Any] | None,
    meta: Mapping[str, Any] | None,
) -> List[str]:
    sources: List[str] = []
    if isinstance(meta, Mapping):
        for key in ("confluence", "confluence_tags"):
            payload = meta.get(key)
            if isinstance(payload, (list, tuple)):
                sources.extend(str(item) for item in payload if item)
    if isinstance(plan, Mapping):
        plan_confluence = plan.get("confluence_tags") or plan.get("confluence")
        if isinstance(plan_confluence, (list, tuple)):
            sources.extend(str(item) for item in plan_confluence if item)
        structured = plan.get("structured_plan")
        if isinstance(structured, Mapping):
            structured_conf = structured.get("confluence_tags") or structured.get("confluence")
            if isinstance(structured_conf, (list, tuple)):
                sources.extend(str(item) for item in structured_conf if item)
    return _unique_tokens(sources)


def _mtf_alignment_info(plan: Mapping[str, Any] | None, direction: Optional[str]) -> Optional[Dict[str, int]]:
    if direction not in {"long", "short"}:
        return None
    mtf_sources = [
        ("mtf_analysis",),
        ("structured_plan", "mtf_analysis"),
        ("structured_plan", "strategy_profile", "mtf_analysis"),
        ("strategy_profile", "mtf_analysis"),
    ]
    mtf_data: Mapping[str, Any] | None = None
    if isinstance(plan, Mapping):
        for path in mtf_sources:
            candidate = _resolve_nested(plan, path)
            if isinstance(candidate, Mapping) and candidate:
                mtf_data = candidate
                break
    if not isinstance(mtf_data, Mapping):
        return None
    aligned = 0
    opposed = 0
    total = 0
    for frame in mtf_data.values():
        if not isinstance(frame, Mapping):
            continue
        trend = str(frame.get("trend") or "").strip().lower()
        if not trend:
            continue
        total += 1
        if direction == "long":
            if any(token in trend for token in ("bull", "up", "long", "positive")):
                aligned += 1
            elif any(token in trend for token in ("bear", "down", "short", "negative")):
                opposed += 1
        else:
            if any(token in trend for token in ("bear", "down", "short", "negative")):
                aligned += 1
            elif any(token in trend for token in ("bull", "up", "long", "positive")):
                opposed += 1
    return {"score": aligned - opposed, "aligned": aligned, "opposed": opposed, "total": total}


def _resolve_vwap(plan: Mapping[str, Any] | None, overlays: Mapping[str, Any] | None) -> Optional[float]:
    if isinstance(overlays, Mapping):
        volume_profile = overlays.get("volume_profile")
        if isinstance(volume_profile, Mapping):
            numeric = _coerce_float(volume_profile.get("vwap"))
            if numeric is not None:
                return numeric
    if isinstance(plan, Mapping):
        for path in (
            ("context_overlays", "volume_profile", "vwap"),
            ("price", "vwap"),
            ("prices", "vwap"),
            ("context", "vwap"),
            ("indicators", "vwap"),
            ("meta", "vwap"),
        ):
            candidate = _resolve_nested(plan, path)
            numeric = _coerce_float(candidate)
            if numeric is not None:
                return numeric
    return None


def _vwap_side(last_price: Optional[float], vwap_value: Optional[float], tick_size: float) -> Optional[str]:
    if last_price is None or vwap_value is None:
        return None
    if not math.isfinite(last_price) or not math.isfinite(vwap_value):
        return None
    threshold = tick_size * 0.5 if tick_size > 0 else 0.0
    if last_price >= vwap_value + threshold:
        return "above"
    if last_price <= vwap_value - threshold:
        return "below"
    return "above" if last_price >= vwap_value else "below"


def compute_next_objective_meta(
    *,
    symbol: str,
    plan: Mapping[str, Any] | None,
    direction: str | None,
    last_price: Optional[float],
    key_levels: Mapping[str, Any] | None,
    overlays: Mapping[str, Any] | None,
    precision: int,
    interval: str | None,
) -> Optional[Dict[str, Any]]:
    entry_price = _resolve_entry(plan)
    direction_token = _normalize_direction(direction, plan=plan)
    if entry_price is None or direction_token is None:
        return None

    tick_size = _resolve_tick_size(plan, key_levels, overlays, precision=precision)
    atr_used = _extract_atr(plan)
    atr_component = (0.05 * atr_used) if isinstance(atr_used, (int, float)) and atr_used and atr_used > 0 else 0.0
    tick_component = tick_size * 2 if tick_size and tick_size > 0 else 0.0
    half_width = max(atr_component, tick_component)
    if half_width <= 0:
        half_width = max(tick_size if tick_size > 0 else 0.01, 0.01)
    band_low = entry_price - half_width
    band_high = entry_price + half_width

    progress = _progress_for_price(last_price, band_low, band_high, half_width)
    state = _state_for_price(direction_token, last_price, band_low, band_high)

    meta_block = plan.get("meta") if isinstance(plan, Mapping) else None
    why_tokens = _extract_confluence(plan, meta_block)

    mtf_info = _mtf_alignment_info(plan, direction_token)
    if mtf_info and mtf_info["total"] > 0:
        why_tokens.append(f"MTF:+{mtf_info['aligned']}/{mtf_info['total']}")

    vwap_value = _resolve_vwap(plan, overlays)
    vwap_relation = _vwap_side(last_price, vwap_value, tick_size)
    if vwap_relation:
        why_tokens.append(f"VWAP_{vwap_relation}")

    why_tokens = _unique_tokens(why_tokens)

    objective_price = round(entry_price, precision) if precision >= 0 else entry_price
    band_low_rounded = round(band_low, precision) if precision >= 0 else band_low
    band_high_rounded = round(band_high, precision) if precision >= 0 else band_high
    progress_value = round(progress, 4)
    timeframe = _normalize_timeframe(interval)

    entry_distance_pct = None
    if last_price is not None and entry_price:
        entry_distance_pct = abs(last_price - entry_price) / entry_price

    payload: Dict[str, Any] = {
        "state": state,
        "why": why_tokens,
        "objective_price": objective_price,
        "band": {
            "low": band_low_rounded,
            "high": band_high_rounded,
        },
        "timeframe": timeframe,
        "progress": progress_value,
    }
    if isinstance(mtf_info, dict):
        payload["_mtf_score"] = mtf_info["score"]
        payload["_mtf_supporting"] = mtf_info["aligned"]
        payload["_mtf_total"] = mtf_info["total"]
    if vwap_relation:
        payload["_vwap_side"] = vwap_relation
    if tick_size:
        payload["_tick_size"] = tick_size
    if atr_used is not None:
        payload["_atr_used"] = atr_used
    if entry_distance_pct is not None and math.isfinite(entry_distance_pct):
        payload["_entry_distance_pct"] = entry_distance_pct
    if last_price is not None and math.isfinite(last_price):
        payload["_last_price"] = last_price

    return payload


def _strategy_template_points(
    strategy_id: str,
    direction: Optional[str],
    waiting_for: str,
    levels: Mapping[str, Any],
    plan: Mapping[str, Any],
    last_time_s: int,
    interval_s: int,
    last_close: float,
) -> Optional[List[Dict[str, float]]]:
    """Map well-known strategies to forecast waypoint templates."""
    if not strategy_id:
        return None
    sid = strategy_id.lower()
    plan_entry = plan.get("entry")
    if isinstance(plan_entry, Mapping):
        plan_entry = plan_entry.get("level")
    entry = float(plan_entry) if isinstance(plan_entry, (int, float)) else None
    targets = plan.get("targets")
    tp1 = None
    if isinstance(targets, (list, tuple)) and targets:
        first = targets[0]
        if isinstance(first, Mapping):
            first = first.get("level") or first.get("target")
        if isinstance(first, (int, float)):
            tp1 = float(first)

    def _time_offset(idx: int) -> int:
        return last_time_s + idx * interval_s

    dir_is_long = (direction or "").lower() == "long"
    vwap_level = _pick_level("vwap", levels)
    orb_level = _pick_level("orh", levels) if dir_is_long else _pick_level("orl", levels)

    # VWAP reclaim / reject style setups.
    if "vwap" in sid or "reclaim" in sid:
        points = [{"time": last_time_s, "value": last_close}]
        if vwap_level is not None:
            points.append({"time": _time_offset(1), "value": vwap_level})
        if orb_level is not None:
            points.append({"time": _time_offset(2), "value": orb_level})
        if tp1 is not None:
            points.append({"time": _time_offset(3), "value": tp1})
        return points if len(points) >= 2 else None

    # Opening range break / retest family.
    if any(token in sid for token in ("orb", "break_and_retest", "range_break_retest")):
        points = [{"time": last_time_s, "value": last_close}]
        if orb_level is not None:
            points.append({"time": _time_offset(1), "value": orb_level})
            points.append({"time": _time_offset(2), "value": orb_level})
        if tp1 is not None:
            points.append({"time": _time_offset(3), "value": tp1})
        return points if len(points) >= 2 else None

    # EMA pullback behaviour.
    if "ema_pullback" in sid:
        ema_level = _pick_level("EMA20", levels) or _pick_level("ema20", levels)
        swing_level = _pick_level("H1H", levels) if dir_is_long else _pick_level("H1L", levels)
        points = [{"time": last_time_s, "value": last_close}]
        if ema_level is not None:
            points.append({"time": _time_offset(1), "value": ema_level})
        if swing_level is not None:
            points.append({"time": _time_offset(2), "value": swing_level})
        if tp1 is not None:
            points.append({"time": _time_offset(3), "value": tp1})
        return points if len(points) >= 2 else None

    # Midday mean reversion patterns.
    if "midday_mean" in sid:
        points = [{"time": last_time_s, "value": last_close}]
        if vwap_level is not None:
            points.append({"time": _time_offset(1), "value": vwap_level})
        if tp1 is not None:
            points.append({"time": _time_offset(2), "value": tp1})
        return points if len(points) >= 2 else None

    # Macro / index-driven gates.
    if any(token in sid for token in ("vix_gate", "gex_flip", "inside_day_rth")):
        pivot = (
            _pick_level("vwap", levels)
            or _pick_level("pdh", levels)
            or _pick_level("pdl", levels)
        )
        points = [{"time": last_time_s, "value": last_close}]
        if pivot is not None:
            points.append({"time": _time_offset(1), "value": pivot})
        if tp1 is not None:
            points.append({"time": _time_offset(2), "value": tp1})
        return points if len(points) >= 2 else None

    return None


def _build_forecast_path_generic(
    strategy_id: Optional[str],
    direction: Optional[str],
    waiting_for: Optional[str],
    key_levels: Mapping[str, Any],
    plan: Optional[Mapping[str, Any]],
    *,
    last_time_s: Optional[int],
    interval_s: Optional[int],
    last_close: Optional[float],
    precision: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    if last_time_s is None or interval_s is None or interval_s <= 0 or last_close is None:
        return None
    plan_map: Mapping[str, Any] = plan or {}
    levels = key_levels or {}
    template_points = _strategy_template_points(
        strategy_id or "",
        direction,
        waiting_for or "",
        levels,
        plan_map,
        last_time_s,
        interval_s,
        float(last_close),
    )

    def _time(idx: int) -> int:
        return last_time_s + idx * interval_s

    points: List[Dict[str, float]]
    if template_points:
        points = template_points
    else:
        plan_entry = plan_map.get("entry")
        if isinstance(plan_entry, Mapping):
            plan_entry = plan_entry.get("level")
        entry = float(plan_entry) if isinstance(plan_entry, (int, float)) else None
        tp1 = None
        targets = plan_map.get("targets")
        if isinstance(targets, (list, tuple)) and targets:
            first = targets[0]
            if isinstance(first, Mapping):
                first = first.get("level") or first.get("target")
            if isinstance(first, (int, float)):
                tp1 = float(first)

        tokens = _parse_conditions(waiting_for or "")
        points = [{"time": last_time_s, "value": float(last_close)}]
        mapping = {
            "vwap": _pick_level("vwap", levels),
            "orh": _pick_level("orh", levels),
            "orl": _pick_level("orl", levels),
            "pivot": _pick_level("pivot", levels),
            "pdh": _pick_level("pdh", levels),
            "pdl": _pick_level("pdl", levels),
        }
        added = 0
        for token in tokens:
            token_lower = token.lower()
            matched_level = None
            for key, value in mapping.items():
                if key in token_lower and value is not None:
                    matched_level = value
                    break
            if matched_level is not None:
                points.append({"time": _time(len(points)), "value": matched_level})
                added += 1
            if added >= 2:
                break
        if entry is not None:
            points.append({"time": _time(len(points)), "value": entry})
        if tp1 is not None:
            points.append({"time": _time(len(points)), "value": tp1})
        if len(points) < 2:
            return None

    if isinstance(precision, int) and precision >= 0:
        for point in points:
            value = point.get("value")
            if isinstance(value, (int, float)):
                point["value"] = round(float(value), precision)

    return {
        "kind": "forecast_path",
        "label": "Forecast path (strategy-aware, cosmetic)",
        "style": {
            "lineStyle": "dashed",
            "opacity": 0.5,
            "width": 2,
            "bias": (direction or "").lower(),
        },
        "points": points,
    }


def _build_zone_items(zones: Iterable[Dict[str, Any]] | None, kind: str, *, precision: int) -> List[Dict[str, Any]]:
    if zones is None:
        return []
    payload: List[Dict[str, Any]] = []
    for zone in zones:
        if not isinstance(zone, dict):
            continue
        low = zone.get("low")
        high = zone.get("high")
        if not (_is_number(low) and _is_number(high)):
            continue
        payload.append(
            {
                "low": round(float(low), precision),
                "high": round(float(high), precision),
                "kind": kind,
                "label": str(zone.get("label") or zone.get("type") or kind),
            }
        )
    return payload


def _level_priority(level: Dict[str, Any]) -> int:
    label = str(level.get("label") or "").lower()
    if not label:
        return 10
    if any(token in label for token in ("opening", "open range", "orh", "orl")):
        return 95
    if "vwap" in label:
        return 90
    if any(token in label for token in ("session high", "session low", "day high", "day low", "intraday high", "intraday low")):
        return 85
    if any(token in label for token in ("vah", "val", "poc")):
        return 80
    if "volume" in label:
        return 70
    if any(token in label for token in ("supply", "demand")):
        return 60
    if any(token in label for token in ("weekly", "monthly")):
        return 55
    return 20


def _split_level_groups(levels: List[Dict[str, Any]], *, max_primary: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not levels:
        return [], []
    scored = [(idx, level, _level_priority(level)) for idx, level in enumerate(levels)]
    scored.sort(key=lambda item: (item[2], -item[0]), reverse=True)
    primary_indices = sorted(idx for idx, _, _ in scored[:max_primary])
    primary = [dict(levels[idx]) for idx in primary_indices]
    supplemental = [dict(level) for idx, level in enumerate(levels) if idx not in primary_indices]
    return primary, supplemental


def build_plan_layers(
    *,
    symbol: str,
    interval: str,
    as_of: str | None,
    planning_context: str | None,
    key_levels: Dict[str, Any] | None,
    overlays: Dict[str, Any] | None,
    precision_map: Dict[str, int] | None = None,
    strategy_id: str | None = None,
    direction: str | None = None,
    waiting_for: str | None = None,
    plan: Dict[str, Any] | None = None,
    last_time_s: int | None = None,
    interval_s: int | None = None,
    last_close: float | None = None,
) -> Dict[str, Any]:
    """Create plan_layers payload for persistence."""

    precision = get_price_precision(symbol, precision_map=precision_map)
    layers: Dict[str, Any] = {
        "plan_id": None,
        "symbol": symbol,
        "interval": interval,
        "as_of": as_of,
        "planning_context": planning_context,
        "precision": precision,
        "levels": [],
        "zones": [],
        "annotations": [],
    }

    levels: List[Dict[str, Any]] = []
    seen: set[tuple[str | None, float]] = set()

    def _append_level(
        label: str | None,
        value: Any,
        *,
        kind: str,
        normalize: bool = False,
    ) -> None:
        if not _is_number(value):
            return
        price = round(float(value), precision)
        text = (str(label).strip() if label is not None else "") or None
        if text and normalize:
            text = text.replace("_", " ").upper()
        key = (text, price)
        if key in seen:
            return
        levels.append(
            {
                "price": price,
                "label": text,
                "kind": kind,
            }
        )
        seen.add(key)

    if isinstance(key_levels, dict):
        for label, value in key_levels.items():
            _append_level(str(label), value, kind="level")

    zones: List[Dict[str, Any]] = []
    if isinstance(overlays, dict):
        zones.extend(_build_zone_items(overlays.get("supply_zones"), kind="supply_zone", precision=precision))
        zones.extend(_build_zone_items(overlays.get("demand_zones"), kind="demand_zone", precision=precision))
        zones.extend(_build_zone_items(overlays.get("fvg"), kind="fair_value_gap", precision=precision))

        volume_profile = overlays.get("volume_profile")
        if isinstance(volume_profile, dict):
            for key, label in (("vah", "VAH"), ("val", "VAL"), ("poc", "POC")):
                _append_level(label, volume_profile.get(key), kind="volume_profile")

        fib_payload = overlays.get("fib_levels")
        if isinstance(fib_payload, dict):
            up_map = fib_payload.get("up") or {}
            down_map = fib_payload.get("down") or {}

            def _friendly_fib_label(token: str, suffix: str) -> str:
                base = token.replace("FIB", "FIB ").strip()
                return f"{base} {suffix}".strip()

            for token, value in up_map.items():
                _append_level(_friendly_fib_label(str(token), "UP"), value, kind="fib", normalize=True)
            for token, value in down_map.items():
                _append_level(_friendly_fib_label(str(token), "DOWN"), value, kind="fib", normalize=True)

        avwap_payload = overlays.get("avwap")
        if isinstance(avwap_payload, dict):
            avwap_labels = {
                "from_open": "AVWAP OPEN",
                "from_prev_close": "AVWAP PREV CLOSE",
                "from_session_low": "AVWAP SESSION LOW",
                "from_session_high": "AVWAP SESSION HIGH",
            }
            for key, friendly in avwap_labels.items():
                _append_level(friendly, avwap_payload.get(key), kind="avwap")

        liquidity_pools = overlays.get("liquidity_pools")
        if isinstance(liquidity_pools, list):
            for pool in liquidity_pools:
                if not isinstance(pool, dict):
                    continue
                label = str(pool.get("type") or "LIQUIDITY POOL").replace("_", " ").upper()
                _append_level(label, pool.get("level"), kind="liquidity_pool")
    layers["levels"] = levels

    primary_levels, supplemental_levels = _split_level_groups(levels, max_primary=6)
    layers["meta"] = {
        "level_groups": {
            "primary": primary_levels,
            "supplemental": supplemental_levels,
        }
    }

    layers["zones"] = zones

    plan_payload: Dict[str, Any] = plan or {}
    strategy_token = (
        strategy_id
        or plan_payload.get("strategy")
        or plan_payload.get("strategy_id")
        or plan_payload.get("setup")
    )
    direction_token = direction or plan_payload.get("direction") or plan_payload.get("bias")
    waiting_token = waiting_for or plan_payload.get("waiting_for")
    if not waiting_token and isinstance(plan_payload.get("meta"), Mapping):
        waiting_token = plan_payload["meta"].get("waiting_for")
    strategy_profile = plan_payload.get("strategy_profile")
    if not waiting_token and isinstance(strategy_profile, Mapping):
        waiting_token = strategy_profile.get("waiting_for")

    forecast_annotation = _build_forecast_path_generic(
        strategy_token,
        direction_token,
        waiting_token,
        key_levels or {},
        plan_payload,
        last_time_s=last_time_s,
        interval_s=interval_s,
        last_close=last_close,
        precision=precision,
    )
    if forecast_annotation:
        layers["annotations"].append(forecast_annotation)

    next_objective_payload = compute_next_objective_meta(
        symbol=symbol,
        plan=plan_payload,
        direction=direction_token,
        last_price=_coerce_float(last_close),
        key_levels=key_levels,
        overlays=overlays,
        precision=precision,
        interval=interval,
    )
    if next_objective_payload:
        meta_block = layers.setdefault("meta", {})
        public_block = {
            key: value for key, value in next_objective_payload.items() if not (isinstance(key, str) and key.startswith("_"))
        }
        if public_block:
            meta_block["next_objective"] = public_block
        internal_block = {
            key[1:]: value
            for key, value in next_objective_payload.items()
            if isinstance(key, str) and key.startswith("_")
        }
        if internal_block:
            meta_block["_next_objective_internal"] = internal_block

    return layers


__all__ = ["build_plan_layers", "compute_next_objective_meta"]
