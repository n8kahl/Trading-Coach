from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

import httpx
from fastapi import FastAPI

from ..lib.data_route import DataRoute
from ..providers.geometry import build_geometry
from ..providers.options import select_contracts
from ..providers.planner import plan as run_plan
from ..providers.series import fetch_series


async def _resolve_chart_url(app: FastAPI, params: Dict[str, Any]) -> Optional[str]:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://app.local") as client:
        response = await client.post("/gpt/chart-url", json=params)
        response.raise_for_status()
        payload = response.json()
        return payload.get("interactive")


async def generate_plan(
    symbol: str,
    *,
    style: str | None,
    route: DataRoute,
    app: FastAPI,
) -> Dict[str, Any]:
    series = await fetch_series([symbol], mode=route.mode, as_of=route.as_of)
    geometry = await build_geometry([symbol], series)
    plan_obj = await run_plan(symbol, series=series, geometry=geometry, route=route)

    snapshot = {
        "generated_at": route.as_of.isoformat(),
        "symbol_count": 1,
    }

    existing_quality = plan_obj.get("data_quality") if isinstance(plan_obj.get("data_quality"), dict) else {}

    plan_obj["planning_context"] = route.planning_context
    plan_obj["data_quality"] = {
        **existing_quality,
        "expected_move": existing_quality.get("expected_move", geometry.expected_move),
        "remaining_atr": existing_quality.get("remaining_atr", geometry.remaining_atr),
        "em_used": existing_quality.get("em_used", geometry.em_used),
        "snapshot": snapshot,
    }

    detail = geometry.get(symbol)
    if detail and detail.snap_trace:
        plan_obj.setdefault("snap_trace", detail.snap_trace + ["planner:complete"])

    if detail and detail.key_levels_used and not plan_obj.get("key_levels_used"):
        plan_obj["key_levels_used"] = detail.key_levels_used

    contracts = await select_contracts(symbol, route.as_of, plan_obj)
    plan_obj["options_contracts"] = contracts.get("options_contracts", [])
    plan_obj["rejected_contracts"] = contracts.get("rejected_contracts", [])
    plan_obj["options_note"] = contracts.get("options_note")

    charts = plan_obj.get("charts") or {}
    params = charts.get("params")
    if isinstance(params, dict):
        chart_url = await _resolve_chart_url(app, params)
        plan_obj["chart_url"] = chart_url

    plan_obj.setdefault("style", style)
    plan_obj.setdefault("warnings", [])

    return plan_obj


__all__ = ["generate_plan"]
