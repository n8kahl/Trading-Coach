from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping, Optional

import httpx
from fastapi import FastAPI

from ..lib.data_route import DataRoute
from ..providers.geometry import GeometryDetail, build_geometry
from ..providers.options import select_contracts
from ..providers.planner import plan as run_plan
from ..providers.scanner import scan as run_scan
from ..providers.series import fetch_series
from .chart_levels import extract_supporting_levels
from .scan_confidence import compute_scan_confidence
from .chart_utils import (
    sanitize_chart_params,
    infer_session_label,
    normalize_confidence,
    normalize_style_token,
    build_ui_state,
)

logger = logging.getLogger(__name__)


def _extract_levels_for_chart(plan: Mapping[str, Any]) -> Optional[str]:
    if not isinstance(plan, Mapping):
        return None
    extras: list[Mapping[str, Any]] = []
    for key in ("key_levels_used", "structured_plan", "plan_layers"):
        value = plan.get(key)
        if isinstance(value, Mapping):
            extras.append(value)
    return extract_supporting_levels(plan, *extras)


async def _chart_urls(app: FastAPI, payloads: List[Dict[str, Any] | None]) -> List[str | None]:
    if not payloads:
        return []
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://app.local") as client:
        results: List[str | None] = []
        for params in payloads:
            sanitized = sanitize_chart_params(params if isinstance(params, dict) else None)
            if not sanitized:
                results.append(None)
                continue
            try:
                response = await client.post("/gpt/chart-url", json=sanitized)
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                logger.warning(
                    "chart-url endpoint rejected payload during scan enrichment",
                    extra={"status": exc.response.status_code, "detail": exc.response.text},
                )
                results.append(None)
                continue
            except httpx.RequestError as exc:
                logger.warning("chart-url request failed during scan enrichment", extra={"error": str(exc)})
                results.append(None)
                continue
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
    series = await fetch_series(symbols, mode=route.mode, as_of=route.as_of, extended=route.extended)
    geometry = await build_geometry(symbols, series)
    page = await run_scan(symbols, style=style, limit=limit, series=series, geometry=geometry, route=route)

    snapshot = {
        "generated_at": route.as_of.isoformat(),
        "symbol_count": len(symbols),
    }
    page["as_of"] = route.as_of.isoformat()
    page["planning_context"] = route.planning_context
    page["use_extended_hours"] = route.extended
    page.setdefault("warnings", [])
    page.setdefault("meta", {})
    page["meta"].setdefault("snapshot", snapshot)
    page["meta"]["style"] = style
    page["meta"]["limit"] = limit
    page["meta"]["route"] = route.mode
    if route.extended:
        page["meta"]["session"] = "extended"
    page["meta"]["universe"] = {"name": "resolved", "count": len(symbols)}
    page.setdefault("data_quality", {})
    page["data_quality"].setdefault("expected_move", geometry.expected_move)
    page["data_quality"].setdefault("remaining_atr", geometry.remaining_atr)
    page["data_quality"].setdefault("em_used", geometry.em_used)
    page["data_quality"]["snapshot"] = snapshot

    candidates: List[Dict[str, Any]] = page.get("candidates", [])
    enriched: List[Dict[str, Any]] = []
    chart_payloads: List[Dict[str, Any] | None] = []
    plan_payloads: List[Dict[str, Any]] = []
    session_label = infer_session_label(route.as_of)

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
        charts_container = plan_obj.get("charts")
        if not isinstance(charts_container, dict):
            charts_container = {}
            plan_obj["charts"] = charts_container

        params = charts_container.get("params") if isinstance(charts_container, dict) else None
        raw_params = dict(params) if isinstance(params, dict) else {}
        levels_token = _extract_levels_for_chart(plan_obj)
        if levels_token:
            raw_params["levels"] = levels_token
            raw_params["supportingLevels"] = "1"
        if route.extended:
            raw_params.setdefault("range", "1d")
            raw_params["session"] = "extended"
        style_token = normalize_style_token(plan_obj.get("style") or style)
        confidence_value = normalize_confidence(plan_obj.get("confidence"))
        raw_params["ui_state"] = build_ui_state(session=session_label, confidence=confidence_value, style=style_token)
        chart_params = sanitize_chart_params(raw_params if raw_params else None)
        if chart_params:
            charts_container["params"] = chart_params
            plan_obj["charts_params"] = chart_params
        chart_payloads.append(chart_params)
        plan_payloads.append(plan_obj)
        enriched.append(candidate)

    chart_urls = await _chart_urls(app, chart_payloads)
    planning_context = route.planning_context
    page_data_quality = page.get("data_quality") if isinstance(page.get("data_quality"), dict) else {}
    banner = page.get("banner")
    for candidate, chart_url, plan_obj in zip(enriched, chart_urls, plan_payloads):
        if chart_url:
            candidate["chart_url"] = chart_url
            charts_container = plan_obj.get("charts")
            if isinstance(charts_container, dict):
                charts_container["interactive"] = chart_url
            plan_obj["chart_url"] = chart_url
        symbol = candidate.get("symbol")
        detail: GeometryDetail | None = geometry.get(symbol) if symbol else None
        plan_data_quality = plan_obj.get("data_quality") if isinstance(plan_obj.get("data_quality"), dict) else {}
        combined_quality = {**page_data_quality, **plan_data_quality}
        confidence = None
        if detail is not None:
            warnings = plan_obj.get("warnings")
            if not warnings:
                confidence = compute_scan_confidence(
                    detail=detail,
                    plan_obj=plan_obj,
                    data_quality=combined_quality,
                    planning_context=planning_context,
                    banner=banner,
                )
        if confidence is not None:
            candidate["confidence"] = round(confidence, 2)
            source_paths = candidate.get("source_paths")
            if not isinstance(source_paths, dict):
                source_paths = {}
            source_paths["confidence"] = "scan_confidence_engine"
            candidate["source_paths"] = source_paths

    if not enriched:
        page.setdefault("banner", "SCAN_NO_CANDIDATES")

    page["candidates"] = enriched
    page["count_candidates"] = len(enriched)
    return page


__all__ = ["generate_scan"]
