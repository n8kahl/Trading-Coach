from __future__ import annotations

from typing import Any, Literal, Sequence

from ..lib.data_source import DataRoute
from ..providers.geometry import GeometryBundle, build_geometry
from ..providers.series import SeriesBundle, fetch_series
from ..providers.scanner import scan as run_scan


def build_placeholder_candidates(
    universe: Sequence[str],
    geometry: GeometryBundle | None,
) -> list[dict[str, Any]]:
    symbols = [symbol for symbol in universe if symbol] or ["ADHOC"]
    expected_move = getattr(geometry, "expected_move", None) if geometry else None
    remaining_atr = getattr(geometry, "remaining_atr", None) if geometry else None
    em_used = getattr(geometry, "em_used", None) if geometry else None
    key_levels = getattr(geometry, "key_levels", None)
    if not isinstance(key_levels, dict):
        key_levels = {"session": [], "structural": []}

    placeholders: list[dict[str, Any]] = []
    for index, symbol in enumerate(symbols, start=1):
        placeholders.append(
            {
                "symbol": symbol,
                "rank": index,
                "score": 0.1,
                "reasons": ["placeholder"],
                "plan_id": f"{symbol}-PLACEHOLDER",
                "entry_candidates": [],
                "snap_trace": ["fallback: placeholder"],
                "confluence": [],
                "accuracy_levels": [],
                "actionable_soon": False,
                "expected_move": expected_move,
                "remaining_atr": remaining_atr,
                "em_used": em_used,
                "key_levels_used": key_levels,
                "source_paths": {},
            }
        )
    return placeholders


def _ensure_candidates_present(
    page: dict[str, Any],
    universe: Sequence[str],
    geometry: GeometryBundle | None,
) -> dict[str, Any]:
    candidates = page.get("candidates")
    has_candidates = isinstance(candidates, list) and len(candidates) > 0
    if has_candidates:
        return page

    has_banner = bool(page.get("banner"))
    if has_banner and len(universe) == 0:
        placeholders = build_placeholder_candidates(universe, geometry)
        page["candidates"] = placeholders
        page["count_candidates"] = len(placeholders)
        warnings = page.setdefault("warnings", [])
        if "PLACEHOLDER" not in warnings:
            warnings.append("PLACEHOLDER")
        return page

    # Preserve empty list without injecting placeholders.
    if candidates is None:
        page["candidates"] = []
    page["count_candidates"] = len(page.get("candidates", []))
    return page


def ensure_scan_schema(
    page: dict[str, Any],
    *,
    route: DataRoute,
    symbols: Sequence[str],
    geometry: GeometryBundle | None,
) -> dict[str, Any]:
    snapshot = {
        "generated_at": route.as_of.isoformat(),
        "symbol_count": len(symbols),
    }
    data_quality = dict(page.get("data_quality") or {})
    page.setdefault("as_of", route.as_of.isoformat())
    page["planning_context"] = route.planning_context
    page["use_extended_hours"] = route.extended
    page.setdefault("warnings", [])
    page.setdefault("meta", {})
    page["meta"].setdefault("route", route.mode)
    page["meta"].setdefault("snapshot", snapshot)
    if route.extended:
        page["meta"].setdefault("session", "extended")
    page.setdefault("snap_trace", getattr(geometry, "snap_trace", None))
    page["data_quality"] = {
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
    return page


async def _attempt_scan(
    universe: Sequence[str],
    *,
    style: str,
    limit: int,
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
        list(universe),
        mode=attempt_route.mode,
        as_of=attempt_route.as_of,
        extended=attempt_route.extended,
    )
    geometry = await build_geometry(list(universe), series)
    page = await run_scan(
        list(universe),
        style=style,
        limit=limit,
        series=series,
        geometry=geometry,
        route=attempt_route,
    )
    page = ensure_scan_schema(page, route=attempt_route, symbols=universe, geometry=geometry)
    page = _ensure_candidates_present(page, universe, geometry)
    return page, series, geometry


async def compute_scan_with_fallback(
    universe: list[str],
    *,
    style: str,
    limit: int,
    route: DataRoute,
) -> dict[str, Any]:
    try:
        page, _, _ = await _attempt_scan(universe, style=style, limit=limit, route=route)
        return page
    except Exception:
        pass

    if route.planning_context == "live":
        try:
            page, _, _ = await _attempt_scan(universe, style=style, limit=limit, route=route, mode_override="lkg")
            page.setdefault("warnings", []).append("LIVE_FALLBACK_TO_LKG")
            return page
        except Exception:
            pass

    snapshot = {
        "generated_at": route.as_of.isoformat(),
        "symbol_count": len(universe),
    }
    stub = {
        "as_of": route.as_of.isoformat(),
        "planning_context": route.planning_context,
        "use_extended_hours": route.extended,
        "meta": {
            "snapshot": snapshot,
            "universe": {"name": "adhoc", "source": "planner", "count": len(universe)},
            **({"session": "extended"} if route.extended else {}),
        },
        "phase": "scan",
        "next_cursor": None,
        "snap_trace": ["fallback: scan_stub"],
        "warnings": [
            "LKG_PARTIAL" if route.planning_context == "frozen" else "LIVE_PARTIAL",
        ],
        "data_quality": {
            "planning_mode": route.planning_context == "frozen",
            "series_present": False,
            "expected_move": None,
            "remaining_atr": None,
            "em_used": None,
            "snapshot": snapshot,
        },
    }
    stub = _ensure_candidates_present(stub, universe, None)
    return stub
