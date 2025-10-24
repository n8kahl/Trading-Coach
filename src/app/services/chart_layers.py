"""Utilities to construct plan-bound chart layers."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

from .precision import get_price_precision


def _is_number(value: Any) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


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
    return layers


__all__ = ["build_plan_layers"]
