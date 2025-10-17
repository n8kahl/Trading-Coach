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


def _build_level_items(levels: Dict[str, Any] | None, *, precision: int) -> List[Dict[str, Any]]:
    if not isinstance(levels, dict):
        return []
    payload: List[Dict[str, Any]] = []
    for label, value in levels.items():
        if not _is_number(value):
            continue
        payload.append(
            {
                "price": round(float(value), precision),
                "label": str(label),
                "kind": "level",
            }
        )
    return payload


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

    levels = _build_level_items(key_levels, precision=precision)
    layers["levels"] = levels

    primary_levels, supplemental_levels = _split_level_groups(levels, max_primary=6)
    layers["meta"] = {
        "level_groups": {
            "primary": primary_levels,
            "supplemental": supplemental_levels,
        }
    }

    zones: List[Dict[str, Any]] = []
    if isinstance(overlays, dict):
        zones.extend(_build_zone_items(overlays.get("supply_zones"), kind="supply_zone", precision=precision))
        zones.extend(_build_zone_items(overlays.get("demand_zones"), kind="demand_zone", precision=precision))
        zones.extend(_build_zone_items(overlays.get("fvg"), kind="fair_value_gap", precision=precision))

        volume_profile = overlays.get("volume_profile")
        if isinstance(volume_profile, dict):
            for label in ("vah", "val", "poc"):
                value = volume_profile.get(label)
                if _is_number(value):
                    layers["levels"].append(
                        {
                            "price": round(float(value), precision),
                            "label": label.upper(),
                            "kind": "volume_profile",
                        }
                    )

    layers["zones"] = zones
    return layers


__all__ = ["build_plan_layers"]
