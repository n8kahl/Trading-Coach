from __future__ import annotations

from typing import Any, Literal

from ..lib.data_source import DataRoute
from ..providers.geometry import GeometryBundle, build_geometry
from ..providers.levels import backfill_key_levels_used
from ..providers.planner import plan as run_plan
from ..providers.series import SeriesBundle, ScaleMismatchError, fetch_series


def _as_dict(obj: Any) -> dict[str, Any]:
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "__dict__"):
        return {k: v for k, v in vars(obj).items() if not k.startswith("_")}
    return {}


def ensure_plan_schema(
    plan_obj: dict[str, Any],
    *,
    geometry: GeometryBundle | None,
    route: DataRoute,
    symbol_count: int = 1,
) -> dict[str, Any]:
    data_quality: dict[str, Any] = _as_dict(plan_obj.get("data_quality")) or {}
    snapshot = {
        "generated_at": route.as_of.isoformat(),
        "symbol_count": symbol_count,
    }
    plan_obj["planning_context"] = route.planning_context
    plan_obj["use_extended_hours"] = route.extended
    plan_obj.setdefault("warnings", [])
    plan_obj["data_quality"] = {
        **data_quality,
        "expected_move": data_quality.get("expected_move")
        if "expected_move" in data_quality
        else getattr(geometry, "expected_move", None),
        "remaining_atr": data_quality.get("remaining_atr")
        if "remaining_atr" in data_quality
        else getattr(geometry, "remaining_atr", None),
        "em_used": data_quality.get("em_used")
        if "em_used" in data_quality
        else getattr(geometry, "em_used", None),
        "snapshot": snapshot,
    }
    geo_trace = getattr(geometry, "snap_trace", None)
    if plan_obj.get("snap_trace") is None and geo_trace:
        plan_obj["snap_trace"] = list(geo_trace)
    symbol = plan_obj.get("symbol")
    if not symbol:
        raise ValueError("plan object missing symbol")
    if not plan_obj.get("key_levels_used"):
        plan_obj["key_levels_used"] = backfill_key_levels_used(symbol, geometry)
    plan_obj.setdefault("meta", {})
    plan_obj["meta"]["key_levels_used"] = plan_obj["key_levels_used"]
    if "planning_snapshot" not in plan_obj:
        plan_obj["planning_snapshot"] = {
            "entry_anchor": plan_obj.get("entry_anchor"),
            "entry_actionability": plan_obj.get("entry_actionability"),
            "entry_candidates": plan_obj.get("entry_candidates", []),
        }
    return plan_obj


async def _attempt_plan(
    symbol: str,
    *,
    route: DataRoute,
    mode_override: Literal["live", "lkg"] | None = None,
) -> tuple[dict[str, Any], SeriesBundle, GeometryBundle]:
    mode = mode_override or route.mode
    planning_context = route.planning_context if mode_override is None else ("frozen" if mode == "lkg" else "live")
    attempt_route = DataRoute(
        mode=mode,
        as_of=route.as_of,
        planning_context=planning_context,
        extended=route.extended,
    )
    series = await fetch_series(
        [symbol],
        mode=attempt_route.mode,
        as_of=attempt_route.as_of,
        extended=attempt_route.extended,
    )
    geometry = await build_geometry([symbol], series)
    plan_obj = await run_plan(symbol, series=series, geometry=geometry, route=attempt_route)
    plan_obj.setdefault("symbol", symbol)
    plan_obj = ensure_plan_schema(plan_obj, geometry=geometry, route=attempt_route, symbol_count=1)
    return plan_obj, series, geometry


async def compute_plan_with_fallback(symbol: str, route: DataRoute) -> dict[str, Any]:
    try:
        plan_obj, _, _ = await _attempt_plan(symbol, route=route)
        return plan_obj
    except ScaleMismatchError as err:
        return _scale_mismatch_stub(symbol, route, err)
    except Exception:
        pass

    if route.planning_context == "live":
        try:
            plan_obj, _, _ = await _attempt_plan(symbol, route=route, mode_override="lkg")
            plan_obj.setdefault("warnings", []).append("LIVE_FALLBACK_TO_LKG")
            return plan_obj
        except ScaleMismatchError as err:
            return _scale_mismatch_stub(symbol, route, err)
        except Exception:
            pass

    series: SeriesBundle | None = None
    geometry: GeometryBundle | None = None
    try:
        series = await fetch_series([symbol], mode=route.mode, as_of=route.as_of, extended=route.extended)
        geometry = await build_geometry([symbol], series)
    except ScaleMismatchError as err:
        return _scale_mismatch_stub(symbol, route, err)
    except Exception:
        series = None
        geometry = None

    key_levels = backfill_key_levels_used(symbol, geometry)
    stub = {
        "plan_id": f"{symbol}-STUB-{route.mode.upper()}",
        "version": 1,
        "trade_detail": "",
        "planning_context": route.planning_context,
        "symbol": symbol,
        "style": None,
        "targets": [],
        "target_meta": [],
        "targets_meta": [],
        "entry_candidates": [],
        "key_levels_used": key_levels,
        "meta": {"key_levels_used": key_levels},
        "data_quality": {
            "series_present": series is not None,
            "expected_move": getattr(geometry, "expected_move", None) if geometry else None,
            "remaining_atr": getattr(geometry, "remaining_atr", None) if geometry else None,
            "em_used": getattr(geometry, "em_used", None) if geometry else None,
            "snapshot": {
                "generated_at": route.as_of.isoformat(),
                "symbol_count": 1,
            },
        },
        "snap_trace": ["fallback: planning_stub"],
        "warnings": [
            "LKG_PARTIAL" if route.planning_context == "frozen" else "LIVE_PARTIAL",
        ],
        "planning_snapshot": {
            "entry_anchor": None,
            "entry_actionability": None,
            "entry_candidates": [],
        },
        "use_extended_hours": route.extended,
    }
    return stub


def _scale_mismatch_stub(symbol: str, route: DataRoute, error: ScaleMismatchError) -> dict[str, Any]:
    key_levels = backfill_key_levels_used(symbol, None)
    note = (
        f"{symbol} data scale mismatch: close {error.close_value:.2f} below "
        f"threshold {error.threshold:.2f} via {error.path}. Plan suppressed."
    )
    return {
        "plan_id": f"{symbol}-SCALE-MISMATCH",
        "version": 1,
        "trade_detail": "",
        "planning_context": route.planning_context,
        "symbol": symbol,
        "style": None,
        "targets": [],
        "target_meta": [],
        "targets_meta": [],
        "entry_candidates": [],
        "key_levels_used": key_levels,
        "meta": {"key_levels_used": key_levels},
        "data_quality": {
            "series_present": False,
            "snapshot": {
                "generated_at": route.as_of.isoformat(),
                "symbol_count": 1,
            },
        },
        "snap_trace": ["failure: scale_mismatch"],
        "warnings": ["SCALE_MISMATCH"],
        "notes": [note],
        "planning_snapshot": {
            "entry_anchor": None,
            "entry_actionability": None,
            "entry_candidates": [],
        },
        "use_extended_hours": route.extended,
    }
