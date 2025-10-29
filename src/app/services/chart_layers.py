"""Utilities to construct plan-bound chart layers."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

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

    return layers


__all__ = ["build_plan_layers"]
