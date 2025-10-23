from __future__ import annotations

from typing import Any, Dict, List

import httpx
from fastapi import FastAPI

from ..lib.data_route import DataRoute
from ..providers.geometry import build_geometry
from ..providers.options import select_contracts
from ..providers.planner import plan as run_plan
from ..providers.scanner import scan as run_scan
from ..providers.series import fetch_series


async def _chart_urls(app: FastAPI, payloads: List[Dict[str, Any]]) -> List[str | None]:
    if not payloads:
        return []
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://app.local") as client:
        results: List[str | None] = []
        for params in payloads:
            if params is None:
                results.append(None)
                continue
            response = await client.post("/gpt/chart-url", json=params)
            response.raise_for_status()
            results.append(response.json().get("interactive"))
        return results


async def generate_scan(
    *,
    symbols: List[str],
    style: str,
    limit: int,
    route: DataRoute,
    app: FastAPI,
) -> Dict[str, Any]:
    series = await fetch_series(symbols, mode=route.mode, as_of=route.as_of)
    geometry = await build_geometry(symbols, series)
    page = await run_scan(symbols, style=style, limit=limit, series=series, geometry=geometry, route=route)

    snapshot = {
        "generated_at": route.as_of.isoformat(),
        "symbol_count": len(symbols),
    }
    page["as_of"] = route.as_of.isoformat()
    page["planning_context"] = route.planning_context
    page.setdefault("warnings", [])
    page.setdefault("meta", {})
    page["meta"].setdefault("snapshot", snapshot)
    page["meta"]["style"] = style
    page["meta"]["limit"] = limit
    page["meta"]["route"] = route.mode
    page["meta"]["universe"] = {"name": "resolved", "count": len(symbols)}
    page.setdefault("data_quality", {})
    page["data_quality"].setdefault("expected_move", geometry.expected_move)
    page["data_quality"].setdefault("remaining_atr", geometry.remaining_atr)
    page["data_quality"].setdefault("em_used", geometry.em_used)
    page["data_quality"]["snapshot"] = snapshot

    candidates: List[Dict[str, Any]] = page.get("candidates", [])
    enriched: List[Dict[str, Any]] = []
    chart_payloads: List[Dict[str, Any] | None] = []

    for candidate in candidates:
        symbol = candidate.get("symbol")
        if not symbol:
            continue
        plan_obj = await run_plan(symbol, series=series, geometry=geometry, route=route)
        candidate.update(
            {
                "entry": plan_obj.get("entry"),
                "stop": plan_obj.get("stop"),
                "tps": plan_obj.get("targets") or [],
                "snap_trace": plan_obj.get("snap_trace"),
                "key_levels_used": plan_obj.get("key_levels_used"),
                "rr_t1": plan_obj.get("rr_to_t1"),
            }
        )
        data_quality = plan_obj.get("data_quality") or {}
        if "expected_move" in data_quality:
            candidate["expected_move"] = data_quality.get("expected_move")
        if "remaining_atr" in data_quality:
            candidate["remaining_atr"] = data_quality.get("remaining_atr")
        if "em_used" in data_quality:
            candidate["em_used"] = data_quality.get("em_used")
        contracts = await select_contracts(symbol, route.as_of, plan_obj)
        candidate["options_contracts"] = contracts.get("options_contracts", [])
        candidate["rejected_contracts"] = contracts.get("rejected_contracts", [])
        candidate["options_note"] = contracts.get("options_note")
        charts = plan_obj.get("charts") or {}
        params = charts.get("params") if isinstance(charts, dict) else None
        chart_payloads.append(params if isinstance(params, dict) else None)
        enriched.append(candidate)

    chart_urls = await _chart_urls(app, chart_payloads)
    for candidate, chart_url in zip(enriched, chart_urls):
        if chart_url:
            candidate["chart_url"] = chart_url

    if not enriched:
        page.setdefault("banner", "SCAN_NO_CANDIDATES")

    page["candidates"] = enriched
    page["count_candidates"] = len(enriched)
    return page


__all__ = ["generate_scan"]
