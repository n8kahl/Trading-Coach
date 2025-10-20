"""Planning-mode scan engine."""

from __future__ import annotations

import logging
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from ..calculations import atr, ema
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
                as_of_utc=as_of.isoformat(),
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

        # Risk/reward using simple ATR band targets.
        stop = max(last_close - atr_val, 0.01)
        target = last_close + atr_val * 2.0
        rr = (target - last_close) / (last_close - stop) if last_close != stop else 0.0
        risk_reward = max(0.0, min(rr / 3.0, 1.0))  # normalise assuming 3:1 as ideal

        readiness = 0.45 * probability + 0.25 * actionability + 0.30 * risk_reward
        readiness = max(0.0, min(readiness, 1.0))
        include_candidate = readiness >= self._min_readiness or probability >= self._probability_floor
        if not include_candidate:
            return None

        template = self._rules.build(style)

        levels = {
            "entry": round(last_close, 2),
            "invalidation": round(stop, 2),
            "targets": [round(target, 2)],
        }
        metrics = {
            "atr": round(atr_val, 4),
            "atr_pct": round(atr_pct, 2),
            "ema21": round(ema21, 4) if ema21 else None,
            "ema50": round(ema50, 4) if ema50 else None,
            "ema100": round(ema100, 4) if ema100 else None,
            "volume_avg": round(volume_series.tail(20).mean(), 2) if not volume_series.empty else None,
            "entry_distance_pct": round(pullback_pct, 2) if pullback_pct is not None else None,
            "entry_distance_atr": round(pullback_atr, 3) if pullback_atr is not None else None,
        }
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
