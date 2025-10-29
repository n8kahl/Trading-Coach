from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Mapping, Iterable

import httpx
from fastapi import FastAPI
import pandas as pd

from ..lib.data_route import DataRoute
from ..providers.geometry import build_geometry
from ..providers.options import select_contracts
from ..providers.planner import plan as run_plan
from ..providers.series import fetch_series
from ..features.htf_levels import compute_intraday_htf_levels
from .chart_levels import extract_supporting_levels
from .chart_utils import (
    sanitize_chart_params,
    infer_session_label,
    normalize_confidence,
    normalize_style_token,
    build_ui_state,
)
from ..app.services import build_plan_layers

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


async def _resolve_chart_url(app: FastAPI, params: Dict[str, Any]) -> Optional[str]:
    sanitized = sanitize_chart_params(params)
    if not sanitized:
        params_keys = sorted(params.keys()) if isinstance(params, dict) else None
        logger.debug("chart params invalid or incomplete; skipping chart-url resolution", extra={"params_keys": params_keys})
        return None

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://app.local") as client:
        try:
            response = await client.post("/gpt/chart-url", json=sanitized)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            logger.warning(
                "chart-url endpoint rejected payload",
                extra={"status": exc.response.status_code, "detail": exc.response.text},
            )
            return None
        except httpx.RequestError as exc:
            logger.warning("chart-url request failed", extra={"error": str(exc)})
            return None
        payload = response.json()
        return payload.get("interactive")


def _interval_to_seconds(token: Optional[str]) -> Optional[int]:
    if not token:
        return None
    normalized = str(token).strip().lower()
    lookup = {
        "1": 60,
        "1m": 60,
        "3": 180,
        "3m": 180,
        "5": 300,
        "5m": 300,
        "10": 600,
        "10m": 600,
        "15": 900,
        "15m": 900,
        "30": 1800,
        "30m": 1800,
        "45": 2700,
        "45m": 2700,
        "60": 3600,
        "60m": 3600,
        "1h": 3600,
    }
    if normalized in lookup:
        return lookup[normalized]
    if normalized.endswith("m") and normalized[:-1].isdigit():
        return int(normalized[:-1]) * 60
    if normalized.endswith("h") and normalized[:-1].isdigit():
        return int(normalized[:-1]) * 3600
    if normalized.isdigit():
        return int(normalized) * 60
    return None


def _normalize_interval_label(token: Optional[str]) -> Optional[str]:
    if not token:
        return None
    normalized = str(token).strip().lower()
    mapping = {
        "1": "1m",
        "1m": "1m",
        "5": "5m",
        "5m": "5m",
        "10": "10m",
        "10m": "10m",
        "15": "15m",
        "15m": "15m",
        "30": "30m",
        "30m": "30m",
        "45": "45m",
        "45m": "45m",
        "60": "60m",
        "60m": "60m",
        "1h": "60m",
        "4h": "240m",
    }
    return mapping.get(normalized, normalized)


def _preferred_frame_sequence(hint: Optional[str]) -> Iterable[str]:
    normalized = _normalize_interval_label(hint)
    sequence: list[str] = []
    if normalized:
        sequence.append(normalized)
        if normalized == "60m":
            sequence.append("65m")
        elif normalized == "30m":
            sequence.extend(["15m", "5m"])
    sequence.extend(["5m", "15m", "65m", "1d"])
    seen: set[str] = set()
    for key in sequence:
        if key not in seen:
            seen.add(key)
            yield key


def _to_utc_epoch(value: datetime) -> int:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    else:
        value = value.astimezone(timezone.utc)
    return int(value.timestamp())


async def generate_plan(
    symbol: str,
    *,
    style: str | None,
    route: DataRoute,
    app: FastAPI,
) -> Dict[str, Any]:
    series = await fetch_series([symbol], mode=route.mode, as_of=route.as_of, extended=route.extended)
    geometry = await build_geometry([symbol], series)
    plan_obj = await run_plan(symbol, series=series, geometry=geometry, route=route)
    plan_obj.setdefault("use_extended_hours", route.extended)

    frames_for_symbol = series.frames.get(symbol) if hasattr(series, "frames") else None
    if isinstance(frames_for_symbol, Mapping):
        bars_60m = None
        bars_240m = None
        for key in ("60m", "60", "1h", "65m"):
            candidate = frames_for_symbol.get(key)
            if candidate is not None:
                bars_60m = candidate
                break
        for key in ("240m", "240", "4h", "195m", "260m"):
            candidate = frames_for_symbol.get(key)
            if candidate is not None:
                bars_240m = candidate
                break
        intraday_levels = compute_intraday_htf_levels(bars_60m, bars_240m)
        if intraday_levels:
            key_levels = plan_obj.get("key_levels")
            if not isinstance(key_levels, dict):
                key_levels = {}
                plan_obj["key_levels"] = key_levels
            for label, value in intraday_levels.items():
                key_levels.setdefault(label, value)

            nested_plan = plan_obj.get("plan") if isinstance(plan_obj.get("plan"), dict) else None
            if nested_plan is not None:
                nested_key_levels = nested_plan.get("key_levels")
                if not isinstance(nested_key_levels, dict):
                    nested_key_levels = {}
                    nested_plan["key_levels"] = nested_key_levels
                for label, value in intraday_levels.items():
                    nested_key_levels.setdefault(label, value)

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
    if contracts.get("options_quote_session"):
        plan_obj["options_quote_session"] = contracts["options_quote_session"]
    if contracts.get("options_as_of"):
        plan_obj["options_as_of"] = contracts["options_as_of"]

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
    session_label = infer_session_label(route.as_of)
    raw_params["ui_state"] = build_ui_state(session=session_label, confidence=confidence_value, style=style_token)
    chart_params = sanitize_chart_params(raw_params if raw_params else None)
    if chart_params:
        charts_container["params"] = chart_params
        plan_obj["charts_params"] = chart_params
        chart_url = await _resolve_chart_url(app, chart_params)
        if chart_url:
            plan_obj["chart_url"] = chart_url
            charts_container["interactive"] = chart_url

    plan_obj.setdefault("style", style or style_token)
    plan_obj.setdefault("warnings", [])

    plan_layers: Optional[Dict[str, Any]] = None
    try:
        plan_root = plan_obj.get("plan") if isinstance(plan_obj.get("plan"), dict) else plan_obj
        key_levels_sources = [
            plan_obj.get("key_levels"),
            plan_root.get("key_levels") if isinstance(plan_root, Mapping) else None,
        ]
        key_levels_payload: Dict[str, Any] = {}
        for source in key_levels_sources:
            if isinstance(source, Mapping):
                for label, value in source.items():
                    if isinstance(value, (int, float)):
                        key_levels_payload[label] = float(value)
        detail = geometry.get(symbol)
        if not key_levels_payload and detail and isinstance(detail.key_levels_used, Mapping):
            for namespace in detail.key_levels_used.values():
                if isinstance(namespace, Mapping):
                    for label, value in namespace.items():
                        if isinstance(value, (int, float)):
                            key_levels_payload.setdefault(label, float(value))
        overlays = plan_obj.get("context_overlays")
        if not isinstance(overlays, Mapping) and isinstance(plan_root, Mapping):
            root_overlays = plan_root.get("context_overlays")
            overlays = root_overlays if isinstance(root_overlays, Mapping) else None

        interval_hint = None
        if isinstance(chart_params, dict):
            interval_hint = chart_params.get("interval")
        if interval_hint is None:
            interval_hint = plan_obj.get("interval") or (
                plan_root.get("interval") if isinstance(plan_root, Mapping) else None
            )

        frame = None
        frames = frames_for_symbol if isinstance(frames_for_symbol, Mapping) else {}
        for candidate in _preferred_frame_sequence(interval_hint):
            lookup_key = candidate
            if lookup_key == "60m":
                lookup_key = "65m"
            frame_candidate = frames.get(lookup_key)
            if frame_candidate is not None and not frame_candidate.empty:
                frame = frame_candidate
                break

        last_time_s = None
        interval_s = None
        last_close_val: Optional[float] = None
        if frame is not None and not frame.empty:
            index = frame.index
            if isinstance(index, pd.DatetimeIndex) and len(index):
                last_ts = index[-1]
                if last_ts.tzinfo is None:
                    last_ts = last_ts.tz_localize("UTC")
                else:
                    last_ts = last_ts.tz_convert("UTC")
                last_time_s = int(last_ts.timestamp())
                if len(index) >= 2:
                    prev_ts = index[-2]
                    if prev_ts.tzinfo is None:
                        prev_ts = prev_ts.tz_localize("UTC")
                    else:
                        prev_ts = prev_ts.tz_convert("UTC")
                    delta_seconds = (last_ts - prev_ts).total_seconds()
                    if delta_seconds > 0:
                        interval_s = int(round(delta_seconds))
            try:
                last_close_val = float(frame["close"].iloc[-1])
            except Exception:  # pragma: no cover - defensive
                last_close_val = None

        if last_close_val is None:
            if detail and detail.last_close is not None:
                last_close_val = float(detail.last_close)
            else:
                last_close_val = float(series.latest_close.get(symbol)) if symbol in series.latest_close else None

        if last_time_s is None:
            last_time_s = _to_utc_epoch(route.as_of)
        if interval_s is None:
            interval_s = _interval_to_seconds(interval_hint) or 300

        raw_layers = build_plan_layers(
            symbol=symbol,
            interval=str(_normalize_interval_label(interval_hint) or "5m"),
            as_of=route.as_of.isoformat(),
            planning_context=route.planning_context,
            key_levels=key_levels_payload,
            overlays=overlays if isinstance(overlays, Mapping) else None,
            strategy_id=plan_obj.get("strategy_id") or plan_obj.get("setup") or (
                plan_root.get("strategy")
                if isinstance(plan_root, Mapping)
                else None
            ),
            direction=plan_obj.get("direction") or plan_obj.get("bias") or (
                plan_root.get("direction") if isinstance(plan_root, Mapping) else None
            ),
            waiting_for=plan_obj.get("waiting_for") or (
                plan_root.get("waiting_for") if isinstance(plan_root, Mapping) else None
            ),
            plan=plan_root if isinstance(plan_root, Mapping) else None,
            last_time_s=last_time_s,
            interval_s=interval_s,
            last_close=last_close_val,
        )
        if raw_layers:
            raw_layers["plan_id"] = plan_obj.get("plan_id")
            plan_layers = raw_layers
        else:
            plan_layers = None
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("plan_layers build failed for %s: %s", symbol, exc)
        plan_layers = None

    if plan_layers:
        plan_obj["plan_layers"] = plan_layers
        nested_plan = plan_obj.get("plan")
        if isinstance(nested_plan, dict):
            nested_plan["plan_layers"] = plan_layers
        plan_obj["layers_fetched"] = True
    else:
        plan_obj.setdefault("layers_fetched", False)

    return plan_obj


__all__ = ["generate_plan"]
