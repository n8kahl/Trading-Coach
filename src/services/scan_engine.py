"""Planning-mode scan engine."""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

import pandas as pd

from ..calculations import atr, ema
from ..levels import inject_style_levels
from ..plans import (
    EntryAnchor,
    EntryContext,
    build_plan_geometry,
    build_structured_geometry,
    compute_entry_candidates,
    is_actionable_soon,
    populate_recent_extrema,
    select_best_entry_plan,
)
from ..plans.entry import select_structural_entry
from ..core.geometry_engine import GeometrySummary, summarize_plan_geometry
from .contract_rules import ContractRuleBook, ContractTemplate
from .polygon_client import AggregatesResult, PolygonAggregatesClient
from .universe import UniverseSnapshot
from .persist import (
    FinalizationRecord,
    PlanningCandidateRecord,
    PlanningPersistence,
    PlanningRunRecord,
)

logger = logging.getLogger(__name__)

_STYLE_STOP_MULT = {
    "scalp": 1.2,
    "intraday": 2.0,
    "swing": 2.5,
    "leaps": 3.0,
}


def _nativeify(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, list):
        return [_nativeify(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_nativeify(item) for item in value)
    if isinstance(value, dict):
        return {key: _nativeify(item) for key, item in value.items()}
    return value


def _latest(series: pd.Series) -> Optional[float]:
    if series is None or series.empty:
        return None
    value = series.iloc[-1]
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _index_summary(frame: pd.DataFrame) -> Dict[str, float]:
    if frame is None or frame.empty:
        return {}
    closes = frame["close"]
    close_val = _latest(closes)
    prior = _latest(closes.iloc[:-1]) if len(closes) > 1 else None
    change_pct = None
    if close_val is not None and prior:
        change_pct = ((close_val - prior) / prior) * 100
    return {
        "close": close_val,
        "change_pct": change_pct,
    }


def _infer_tick_size(price: float) -> float:
    if price >= 500:
        return 0.1
    if price >= 200:
        return 0.05
    if price >= 50:
        return 0.02
    if price >= 10:
        return 0.01
    if price >= 1:
        return 0.005
    return 0.001


def _max_entry_distance_pct(style: str | None) -> float:
    token = (style or "intraday").strip().lower()
    if token in {"scalp", "intraday"}:
        return 0.003
    if token == "swing":
        return 0.01
    return 0.02


@dataclass
class PlanningCandidate:
    symbol: str
    readiness_score: float
    components: Dict[str, float]
    metrics: Dict[str, float]
    levels: Dict[str, float]
    contract_template: ContractTemplate
    requires_live_confirmation: bool
    missing_live_inputs: List[str]


@dataclass
class PlanningScanResult:
    as_of_utc: datetime
    run_id: Optional[int]
    universe: UniverseSnapshot
    indices_context: Dict[str, Dict[str, float]]
    candidates: List[PlanningCandidate]


class PlanningScanEngine:
    """Orchestrates planning-mode symbol analysis."""

    _AGG_WINDOWS = ("1d", "60", "30")
    _INDEX_SYMBOLS = ("I:SPX", "I:NDX", "I:RUT", "I:VIX")

    def __init__(
        self,
        polygon_client: PolygonAggregatesClient,
        persistence: PlanningPersistence,
        *,
        rulebook: Optional[ContractRuleBook] = None,
        min_readiness: float = 0.30,
    ) -> None:
        self._polygon = polygon_client
        self._persist = persistence
        self._rules = rulebook or ContractRuleBook()
        self._min_readiness = min_readiness
        self._probability_floor = 0.55

    async def run(self, universe: UniverseSnapshot, *, style: str) -> PlanningScanResult:
        as_of = datetime.now(timezone.utc)
        index_results = await self._polygon.fetch_many(self._INDEX_SYMBOLS, self._AGG_WINDOWS)
        indices_context: Dict[str, Dict[str, float]] = {}
        for symbol, agg in index_results.items():
            daily = agg.windows.get("1d")
            if daily is None:
                daily = agg.windows.get("d")
            indices_context[symbol] = _index_summary(daily) if daily is not None else {}

        symbol_results = await self._polygon.fetch_many(universe.symbols, self._AGG_WINDOWS)
        candidates = self._build_candidates(symbol_results, style=style)

        run_id = None
        if candidates:
            run_record = PlanningRunRecord(
                as_of_utc=as_of,
                universe_name=universe.name,
                universe_source=universe.source,
                tickers=universe.symbols,
                indices_context=indices_context,
                data_windows={"primary": list(self._AGG_WINDOWS)},
                notes=universe.metadata.get("notes") if universe.metadata else None,
                style=style,
            )
            run_id = await self._persist.create_scan_run(run_record)
            if run_id:
                await self._persist.ensure_schema()
                for candidate in candidates:
                    record = PlanningCandidateRecord(
                        scan_id=run_id,
                        symbol=candidate.symbol,
                        metrics=candidate.metrics,
                        levels=candidate.levels,
                        readiness_score=candidate.readiness_score,
                        components=candidate.components,
                        contract_template=candidate.contract_template.as_dict(),
                        requires_live_confirmation=candidate.requires_live_confirmation,
                        missing_live_inputs=candidate.missing_live_inputs,
                    )
                    await self._persist.store_candidate(record)

        return PlanningScanResult(
            as_of_utc=as_of,
            run_id=run_id,
            universe=universe,
            indices_context=indices_context,
            candidates=candidates,
        )

    def _build_candidates(
        self,
        aggregates: Dict[str, "AggregatesResult"],
        *,
        style: str,
    ) -> List[PlanningCandidate]:
        results: List[PlanningCandidate] = []
        for symbol, result in aggregates.items():
            candidate = self._score_symbol(symbol, result.windows, style=style)
            if candidate is not None:
                results.append(candidate)
        results.sort(key=lambda item: item.readiness_score, reverse=True)
        return results

    def _score_symbol(
        self,
        symbol: str,
        windows: Dict[str, pd.DataFrame],
        *,
        style: str,
    ) -> Optional[PlanningCandidate]:
        daily = windows.get("1d")
        if daily is None:
            daily = windows.get("d")
        if daily is None or len(daily) < 25:
            return None

        close_series = daily["close"]
        high_series = daily["high"]
        low_series = daily["low"]
        volume_series = daily["volume"] if "volume" in daily.columns else pd.Series([0.0] * len(daily))

        last_close = _latest(close_series)
        if last_close is None or last_close <= 0:
            return None

        ema9 = _latest(ema(close_series, 9))
        ema21 = _latest(ema(close_series, 21))
        ema50 = _latest(ema(close_series, 50))
        ema100 = _latest(ema(close_series, 100))
        atr_series = atr(high_series, low_series, close_series, 14)
        atr_val = _latest(atr_series)
        if atr_val is None or atr_val <= 0:
            return None
        atr_pct = (atr_val / last_close) * 100.0

        # Trend probability component.
        trend_component = 0.0
        trend_checks = 0
        for ema_val in (ema21, ema50, ema100):
            if ema_val:
                trend_checks += 1
                if last_close > ema_val:
                    trend_component += 1.0
        probability = trend_component / trend_checks if trend_checks else 0.0

        # Actionability based on pullback relative to ATR.
        pullback = abs(last_close - (ema21 or last_close))
        pullback_pct = (pullback / last_close) * 100.0 if last_close else None
        pullback_atr = pullback / atr_val if atr_val else None
        actionability = max(0.0, 1.0 - min((pullback_atr or 0.0), 3.0) / 3.0)

        levels_map: Dict[str, float] = {}
        try:
            levels_map["session_high"] = float(high_series.iloc[-1])
            levels_map["session_low"] = float(low_series.iloc[-1])
            if len(high_series) > 1:
                levels_map["pdh"] = float(high_series.iloc[-2])
                levels_map["pdl"] = float(low_series.iloc[-2])
                levels_map["pdc"] = float(close_series.iloc[-2])
        except Exception:
            pass
        daily_levels_ctx: List[Tuple[str, float]] = []
        weekly_levels_ctx: List[Tuple[str, float]] = []

        def _append_level(target: List[Tuple[str, float]], tag: str, value: Any) -> None:
            try:
                price = float(value)
            except (TypeError, ValueError):
                return
            if math.isfinite(price):
                target.append((tag, price))

        try:
            _append_level(daily_levels_ctx, "DAILY_HIGH", high_series.iloc[-1])
            _append_level(daily_levels_ctx, "DAILY_LOW", low_series.iloc[-1])
            _append_level(daily_levels_ctx, "DAILY_CLOSE", close_series.iloc[-1])
        except Exception:
            pass

        try:
            weekly_window = daily.tail(5)
            if not weekly_window.empty:
                _append_level(weekly_levels_ctx, "WEEKLY_HIGH", weekly_window["high"].max())
                _append_level(weekly_levels_ctx, "WEEKLY_LOW", weekly_window["low"].min())
                _append_level(weekly_levels_ctx, "WEEKLY_CLOSE", weekly_window["close"].iloc[-1])
        except Exception:
            pass

        inject_style_levels(
            levels_map,
            {
                "levels_daily": daily_levels_ctx,
                "levels_weekly": weekly_levels_ctx,
                "vol_profile_daily": {},
                "vol_profile_weekly": {},
            },
            style,
        )
        try:
            populate_recent_extrema(
                levels_map,
                list(high_series.tail(20)),
                list(low_series.tail(20)),
                window=6,
            )
        except Exception:
            pass
        realized_range = 0.0
        try:
            realized_range = abs(float(high_series.iloc[-1]) - float(low_series.iloc[-1]))
        except Exception:
            realized_range = 0.0
        atr_daily_series = atr(high_series, low_series, close_series, 14)
        atr_daily_val = _latest(atr_daily_series) or atr_val
        geometry = None
        fallback_geometry = False
        timestamp = None
        try:
            atr_for_entry = float(atr_val)
            if not math.isfinite(atr_for_entry):
                atr_for_entry = 0.0
        except (TypeError, ValueError):
            atr_for_entry = 0.0
        entry_seed = select_structural_entry(
            direction="long",
            style=style,
            close_price=float(last_close),
            levels=levels_map,
            atr=atr_for_entry,
            expected_move=None,
        )
        indicators_payload = {"rvol": trend_component, "liquidity_rank": None}
        prices_payload = {"close": float(last_close)}
        if ema9:
            prices_payload["ema9"] = float(ema9)
        if ema21:
            prices_payload["ema21"] = float(ema21)
        if ema50:
            prices_payload["ema50"] = float(ema50)
        try:
            entry_candidates = compute_entry_candidates(symbol, style, levels_map, indicators_payload, prices_payload)
        except Exception:
            entry_candidates = []
        if entry_seed and (not entry_candidates or not any(abs(float(candidate["level"]) - entry_seed) < 1e-4 for candidate in entry_candidates)):
            tick_size = _infer_tick_size(float(last_close))
            distance_pct = abs(entry_seed - last_close) / last_close if last_close else float("inf")
            distance_limit = _max_entry_distance_pct(style)
            if distance_pct <= distance_limit:
                distance_atr = abs(entry_seed - last_close) / atr_for_entry if atr_for_entry > 0 else float("inf")
                actionable = is_actionable_soon(entry_seed, last_close, atr_for_entry, tick_size, style)
                score = 0.40 * (1 - distance_pct) + 0.20 * (1 if actionable else 0)
                bars_to_trigger = max(int(round(distance_atr * 2.0)), 0)
                entry_candidates.insert(
                    0,
                    {
                        "level": round(entry_seed, 2),
                        "label": "STRUCTURAL",
                        "type": "STRUCTURAL",
                        "bars_to_trigger": bars_to_trigger,
                        "entry_distance_pct": round(distance_pct, 4),
                        "entry_distance_atr": round(distance_atr, 3) if math.isfinite(distance_atr) else None,
                        "actionable_soon": actionable,
                        "score": round(score, 4),
                        "structure_quality": 0.75,
                        "evaluation": {
                            "actionability": round(score, 4),
                            "distance_pct": round(distance_pct, 4),
                            "distance_atr": round(distance_atr, 3) if math.isfinite(distance_atr) else None,
                        },
                    },
                )
        entry_candidates = _nativeify(entry_candidates)
        index_payload = getattr(daily, "index", None)
        if index_payload is not None and len(index_payload) > 0:
            ts = index_payload[-1]
            if isinstance(ts, datetime):
                timestamp = ts
            else:
                try:
                    timestamp = datetime.fromisoformat(str(ts))
                except Exception:
                    timestamp = None
        entry_context_scan = EntryContext(
            direction="long",
            style=style,
            last_price=float(last_close),
            atr=float(atr_val),
            levels=levels_map,
            timestamp=timestamp,
            mtf_bias=None,
            session_phase=None,
            preferred_entries=[EntryAnchor(entry_seed, "structural")] if entry_seed else None,
            tick=_infer_tick_size(float(last_close)),
        )
        plan_timestamp_scan = timestamp.to_pydatetime() if isinstance(timestamp, pd.Timestamp) else timestamp
        plan_kwargs_scan = {
            "side": "long",
            "style": style,
            "strategy": None,
            "atr_tf": float(atr_val),
            "atr_daily": float(atr_daily_val or atr_val),
            "iv_expected_move": None,
            "realized_range": realized_range,
            "levels": dict(levels_map),
            "timestamp": plan_timestamp_scan,
            "em_points": None,
        }
        geometry, selected_entry = select_best_entry_plan(entry_context_scan, plan_kwargs_scan, builder=build_plan_geometry)
        entry_price = geometry.entry
        selected_tag = selected_entry.tag
        if selected_entry.tag != "reference":
            geometry.snap_trace.append(
                f"entry:{last_close:.2f}->{geometry.entry:.2f} via {selected_entry.tag.upper()}"
            )
        for candidate_meta in entry_candidates:
            evaluation = candidate_meta.setdefault("evaluation", {})
            try:
                level_val = float(candidate_meta.get("level"))
            except (TypeError, ValueError):
                evaluation["status"] = "invalid_level"
                continue
            if abs(level_val - selected_entry.entry) < 1e-4:
                evaluation["status"] = "selected"
                evaluation["actionability"] = round(selected_entry.actionability, 3)
            else:
                evaluation.setdefault("status", "considered")
        structured_warnings: List[str] = []
        geometry_summary: GeometrySummary | None = None
        if fallback_geometry or geometry is None:
            stop = float(entry_price - max(atr_val, 1e-6))
            target = float(entry_price + max(atr_val * 2.0, 1e-6))
            probability = max(probability, self._probability_floor)
            rr = (target - entry_price) / (entry_price - stop) if entry_price != stop else 0.0
            target_entries = [
                {
                    "price": round(target, 2),
                    "prob_touch": probability,
                    "distance": target - entry_price,
                    "rr_multiple": rr,
                    "em_fraction": None,
                    "snap_tag": "FALLBACK_RR",
                    "em_capped": False,
                }
            ]
            runner_payload = {
                "fraction": 0.2,
                "atr_multiple": 1.0,
                "atr_step": 0.5,
                "em_fraction_cap": 0.6,
                "notes": ["fallback_runner"],
                "trail": "ATR-only trail",
            }
            em_used_flag = False
            snap_trace_payload = ["fallback_rr"]
            structured_tp_reasons = [{"label": "TP1", "reason": "Fallback RR"}]
            key_levels_used_payload = {"session": [], "structural": []}
            geometry_summary = GeometrySummary(
                entry=round(entry_price, 2),
                stop=round(stop, 2),
                targets=[dict(item) for item in target_entries],
                rr_t1=round(rr, 3) if math.isfinite(rr) else None,
                atr_used=float(atr_val) if atr_val is not None else None,
                expected_move=None,
                remaining_atr=None,
                em_used=False,
                snap_trace=snap_trace_payload,
                key_levels_used={"session": [], "structural": []},
            )
        else:
            plan_time_dt = timestamp or datetime.now(timezone.utc)
            raw_targets = [float(meta.price) for meta in geometry.targets] or [entry_price + atr_val, entry_price + atr_val * 2, entry_price + atr_val * 3]
            structured_geometry = build_structured_geometry(
                symbol=symbol,
                style=style,
                direction="long",
                entry=entry_price,
                levels=levels_map,
                atr_value=atr_val,
                plan_time=plan_time_dt,
                raw_targets=raw_targets,
                rr_floor=geometry.stop.rr_min,
                em_hint=None,
            )
            structured_warnings = list(structured_geometry.warnings)
            invariant_broken = any(warning == "INVARIANT_BROKEN" for warning in structured_warnings)
            key_levels_used = structured_geometry.key_levels_used or {}
            structured_runner = dict(structured_geometry.runner_policy or {})
            structured_tp_reasons_raw = list(structured_geometry.tp_reasons or [])

            def _ensure_tp_reasons(reasons: List[Dict[str, Any]], count: int) -> List[Dict[str, Any]]:
                aligned: List[Dict[str, Any]] = []
                for idx in range(count):
                    if idx < len(reasons) and isinstance(reasons[idx], dict):
                        payload = dict(reasons[idx])
                    else:
                        payload = {"label": f"TP{idx + 1}", "reason": "Legacy geometry"}
                    payload.setdefault("label", f"TP{idx + 1}")
                    aligned.append(payload)
                return aligned

            if invariant_broken:
                logger.warning("planning scan invariants broken for %s; reverting to legacy geometry", symbol)
                if "STRUCTURED_GEOMETRY_FALLBACK" not in structured_geometry.warnings:
                    structured_geometry.warnings.append("STRUCTURED_GEOMETRY_FALLBACK")
                structured_warnings = list(structured_geometry.warnings)
                stop = round(float(geometry.stop.structural or geometry.stop.price or geometry.stop.volatility or entry_price), 2)
                targets_final = [round(float(meta.price), 2) for meta in geometry.targets] or raw_targets
                expected_move_points = structured_geometry.em_points or geometry.em_day
                clamp_flag = bool(geometry.em_used or structured_geometry.clamp_applied)
                if not structured_tp_reasons_raw:
                    structured_tp_reasons_raw = _ensure_tp_reasons([], len(targets_final))
                if not structured_runner:
                    structured_runner = {
                        "fraction": geometry.runner.fraction,
                        "atr_trail_mult": geometry.runner.atr_trail_mult,
                        "atr_trail_step": geometry.runner.atr_trail_step,
                        "em_fraction_cap": geometry.runner.em_fraction_cap,
                        "notes": list(getattr(geometry.runner, "notes", [])),
                    }
            else:
                stop = structured_geometry.stop
                targets_final = structured_geometry.targets
                expected_move_points = structured_geometry.em_points
                clamp_flag = structured_geometry.clamp_applied or geometry.em_used
                geometry.stop.price = stop
                geometry.stop.structural = stop
                geometry.stop.snapped = structured_geometry.stop_label
                geometry.em_day = expected_move_points
                geometry.em_used = clamp_flag
            if len(geometry.targets) < len(targets_final):
                geometry.targets.extend(geometry.targets[-1:] * (len(targets_final) - len(geometry.targets)))

            geometry.em_day = expected_move_points
            geometry.em_used = clamp_flag
            geometry.stop.price = stop
            geometry.stop.structural = stop
            if structured_geometry.stop_label:
                geometry.stop.snapped = structured_geometry.stop_label

            tp_reasons_aligned = _ensure_tp_reasons(structured_tp_reasons_raw, len(targets_final))
            target_entries = []
            risk_points = abs(entry_price - stop)
            for idx, meta in enumerate(geometry.targets, start=1):
                price_token = targets_final[idx - 1] if idx - 1 < len(targets_final) else getattr(meta, "price", entry_price)
                try:
                    price_val = round(float(price_token), 2)
                except (TypeError, ValueError):
                    price_val = round(float(entry_price), 2)
                meta.price = price_val
                meta.distance = round(abs(price_val - entry_price), 2)
                reward = price_val - entry_price
                rr_multiple = 0.0
                if risk_points > 0:
                    rr_multiple = max(reward, 0.0) / risk_points
                meta.rr_multiple = round(rr_multiple, 2)
                meta.em_fraction = round(meta.distance / expected_move_points, 2) if expected_move_points else None
                meta.em_capped = clamp_flag
                reason_payload = tp_reasons_aligned[idx - 1] if idx - 1 < len(tp_reasons_aligned) else {}
                meta.reason = reason_payload.get("snap_tag") or reason_payload.get("reason")
                target_entries.append(
                    {
                        "price": price_val,
                        "prob_touch": meta.prob_touch,
                        "distance": meta.distance,
                        "rr_multiple": meta.rr_multiple,
                        "em_fraction": meta.em_fraction,
                        "snap_tag": reason_payload.get("snap_tag"),
                        "em_capped": clamp_flag,
                        "reason": reason_payload.get("reason"),
                    }
                )

            target_entries = _nativeify(target_entries)

            try:
                geometry.runner.fraction = float(structured_runner.get("fraction", geometry.runner.fraction))
                geometry.runner.atr_trail_mult = float(
                    structured_runner.get("atr_trail_mult", geometry.runner.atr_trail_mult)
                )
                geometry.runner.atr_trail_step = float(
                    structured_runner.get("atr_trail_step", geometry.runner.atr_trail_step)
                )
                geometry.runner.em_fraction_cap = float(
                    structured_runner.get("em_fraction_cap", geometry.runner.em_fraction_cap)
                )
                geometry.runner.notes = list(structured_runner.get("notes", geometry.runner.notes))
            except Exception:
                geometry.runner.notes = list(structured_runner.get("notes", geometry.runner.notes))

            if not structured_runner.get("trail"):
                structured_runner["trail"] = f"ATR trail x {geometry.runner.atr_trail_mult:.2f}"

            stop = round(float(stop), 2)
            rr = (targets_final[0] - entry_price) / (entry_price - stop) if targets_final and entry_price != stop else 0.0
            probability = float(geometry.targets[0].prob_touch if geometry.targets else probability)
            runner_payload = {
                "fraction": structured_runner.get("fraction"),
                "atr_multiple": structured_runner.get("atr_trail_mult"),
                "atr_step": structured_runner.get("atr_trail_step"),
                "em_fraction_cap": structured_runner.get("em_fraction_cap"),
                "notes": structured_runner.get("notes"),
                "trail": structured_runner.get("trail"),
            }
            em_used_flag = bool(clamp_flag)
            snap_trace_payload = geometry.snap_trace
            runner_payload = _nativeify(runner_payload)
            structured_tp_reasons = _nativeify(tp_reasons_aligned)
            key_levels_used_payload = _nativeify({
                bucket: [dict(entry) for entry in entries] for bucket, entries in (key_levels_used or {}).items() if isinstance(entries, list)
            })
            geometry_summary = summarize_plan_geometry(
                geometry,
                entry=entry_price,
                atr_value=atr_val,
                expected_move=structured_geometry.em_points or geometry.em_day,
                key_levels_used=key_levels_used,
            )
        logger.debug(
            "planning_geometry",
            extra={
                "symbol": symbol,
                "entry": round(entry_price, 2),
                "stop": round(stop, 2),
                "targets": [entry["price"] for entry in target_entries],
                "expected_move": round(geometry.em_day, 4) if geometry and geometry.em_day else None,
                "remaining_atr": round(geometry.ratr, 4) if geometry and geometry.ratr else None,
                "em_used": em_used_flag,
            },
        )
        risk_reward = max(0.0, min(rr / 3.0, 1.0))  # normalise assuming 3:1 as ideal

        readiness = 0.45 * probability + 0.25 * actionability + 0.30 * risk_reward
        readiness = max(0.0, min(readiness, 1.0))
        include_candidate = readiness >= self._min_readiness or probability >= self._probability_floor
        if not include_candidate:
            return None

        template = self._rules.build(style)

        levels = {
            "entry": round(entry_price, 2),
            "invalidation": round(stop, 2),
            "targets": [entry["price"] for entry in target_entries] or [round(target, 2)],
        }
        target_meta_payload = target_entries
        metrics = _nativeify({
            "atr": round(atr_val, 4),
            "atr_pct": round(atr_pct, 2),
            "ema21": round(ema21, 4) if ema21 else None,
            "ema50": round(ema50, 4) if ema50 else None,
            "ema100": round(ema100, 4) if ema100 else None,
            "volume_avg": round(volume_series.tail(20).mean(), 2) if not volume_series.empty else None,
            "entry_distance_pct": round(pullback_pct, 2) if pullback_pct is not None else None,
            "entry_distance_atr": round(pullback_atr, 3) if pullback_atr is not None else None,
            "target_meta": target_meta_payload,
            "runner_policy": runner_payload,
            "atr_used": geometry_summary.atr_used if geometry_summary else (float(atr_val) if atr_val is not None else None),
            "expected_move": geometry_summary.expected_move if geometry_summary else (round(geometry.em_day, 4) if geometry and geometry.em_day else None),
            "remaining_atr": geometry_summary.remaining_atr if geometry_summary else (round(geometry.ratr, 4) if geometry and geometry.ratr else None),
            "em_used": geometry_summary.em_used if geometry_summary else em_used_flag,
            "snap_trace": geometry_summary.snap_trace if geometry_summary else snap_trace_payload,
            "entry_anchor": selected_tag,
            "entry_actionability": round(selected_entry.actionability, 3) if selected_entry else None,
            "stop_detail": {
                "structural": geometry.stop.structural if geometry else stop - last_close,
                "volatility": geometry.stop.volatility if geometry else 0.0,
                "snapped": geometry.stop.snapped if geometry else "fallback_rr",
            },
            "key_levels_used": geometry_summary.key_levels_used if geometry_summary else key_levels_used_payload,
            "tp_reasons": structured_tp_reasons,
            "entry_candidates": entry_candidates,
            "geometry_warnings": structured_warnings,
        })
        components = {
            "probability": round(probability, 3),
            "actionability": round(actionability, 3),
            "risk_reward": round(risk_reward, 3),
        }
        candidate = PlanningCandidate(
            symbol=symbol,
            readiness_score=round(readiness, 3),
            components=components,
            metrics=metrics,
            levels=levels,
            contract_template=template,
            requires_live_confirmation=True,
            missing_live_inputs=["iv", "spread", "oi"],
        )
        return candidate

    def replay_cached_run(
        self,
        run_record: Any,
        candidate_rows: Sequence[Any],
    ) -> PlanningScanResult:
        as_of = run_record["as_of_utc"]
        if not isinstance(as_of, datetime):
            as_of = datetime.fromisoformat(str(as_of))
        symbols_payload = run_record["tickers"]
        if isinstance(symbols_payload, str):
            try:
                symbols = json.loads(symbols_payload)
            except Exception:  # pragma: no cover - defensive
                symbols = []
        else:
            symbols = list(symbols_payload or [])
        indices_context = run_record["indices_context"] or {}
        if isinstance(indices_context, str):
            try:
                indices_context = json.loads(indices_context)
            except Exception:
                indices_context = {}
        data_windows = run_record["data_windows"] or {}
        if isinstance(data_windows, str):
            try:
                data_windows = json.loads(data_windows)
            except Exception:
                data_windows = {}
        metadata: Dict[str, Any] = {
            "style": run_record.get("style", "intraday"),
            "cache_replayed": True,
        }
        notes = run_record.get("notes")
        if notes:
            metadata["notes"] = notes
        universe_snapshot = UniverseSnapshot(
            name=run_record.get("universe_name", "fallback"),
            source="cache",
            as_of_utc=as_of,
            symbols=[str(sym) for sym in symbols],
            metadata=metadata,
        )
        candidates: List[PlanningCandidate] = []
        for row in candidate_rows:
            metrics = row["metrics"] or {}
            if isinstance(metrics, str):
                try:
                    metrics = json.loads(metrics)
                except Exception:
                    metrics = {}
            if isinstance(metrics, Mapping):
                entry_candidates_cached = metrics.get("entry_candidates")
                if isinstance(entry_candidates_cached, list):
                    metrics["entry_candidates"] = [
                        dict(candidate) for candidate in entry_candidates_cached if isinstance(candidate, Mapping)
                    ]
                elif entry_candidates_cached is not None:
                    metrics["entry_candidates"] = []
                key_levels_used_cached = metrics.get("key_levels_used")
                if isinstance(key_levels_used_cached, Mapping):
                    metrics["key_levels_used"] = {
                        bucket: [
                            dict(entry) for entry in entries if isinstance(entry, Mapping)
                        ]
                        for bucket, entries in key_levels_used_cached.items()
                        if isinstance(entries, list)
                    }
                elif key_levels_used_cached is not None:
                    metrics["key_levels_used"] = {}
                tp_reasons_cached = metrics.get("tp_reasons")
                if isinstance(tp_reasons_cached, list):
                    metrics["tp_reasons"] = [
                        dict(reason) for reason in tp_reasons_cached if isinstance(reason, Mapping)
                    ]
                elif tp_reasons_cached is not None:
                    metrics["tp_reasons"] = []
                runner_policy_cached = metrics.get("runner_policy")
                if isinstance(runner_policy_cached, Mapping):
                    metrics["runner_policy"] = dict(runner_policy_cached)
                elif runner_policy_cached is not None:
                    metrics["runner_policy"] = {}
                target_meta_cached = metrics.get("target_meta")
                if isinstance(target_meta_cached, list):
                    metrics["target_meta"] = [
                        dict(item) for item in target_meta_cached if isinstance(item, Mapping)
                    ]
            levels = row["levels"] or {}
            if isinstance(levels, str):
                try:
                    levels = json.loads(levels)
                except Exception:
                    levels = {}
            components = row["components"] or {}
            if isinstance(components, str):
                try:
                    components = json.loads(components)
                except Exception:
                    components = {}
            template_payload = row["contract_template"] or {}
            if isinstance(template_payload, str):
                try:
                    template_payload = json.loads(template_payload)
                except Exception:
                    template_payload = {}
            template = ContractTemplate.from_dict(template_payload)
            missing_inputs = row.get("missing_live_inputs") or []
            readiness = float(row["readiness_score"] or 0.0)
            candidate = PlanningCandidate(
                symbol=row["symbol"],
                readiness_score=readiness,
                components={str(k): float(v) if isinstance(v, (int, float)) else v for k, v in (components or {}).items()},
                metrics={str(k): v for k, v in (metrics or {}).items()},
                levels={str(k): v for k, v in (levels or {}).items()},
                contract_template=template,
                requires_live_confirmation=bool(row["requires_live_confirmation"]),
                missing_live_inputs=[str(item) for item in missing_inputs],
            )
            candidates.append(candidate)
        return PlanningScanResult(
            as_of_utc=as_of,
            run_id=run_record["id"],
            universe=universe_snapshot,
            indices_context=indices_context,
            candidates=candidates,
        )


__all__ = ["PlanningScanEngine", "PlanningCandidate", "PlanningScanResult"]
