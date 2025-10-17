"""Utilities to construct plan-bound chart layers."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

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
