"""Offline evaluator for trade plan snapshots.

The evaluator consumes cached plan snapshots (or fixture JSON payloads) and
simulates trade outcomes by walking through provided bar data.  It records
target hits, stop-outs, MFE/MAE statistics and reliability data for TP1
probability calibration.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence
import json
import math

from ..engine.calibration import CalibrationBin, CalibrationTable
from ..telemetry import (
    record_em_capped_tp,
    record_rr_below_min,
    record_sl_hit,
    record_tp_hit,
)


@dataclass(frozen=True)
class PriceBar:
    high: float
    low: float


@dataclass(frozen=True)
class TargetSpec:
    label: str
    price: float
    prob_touch: float
    em_capped: bool = False


@dataclass(frozen=True)
class PlanSnapshot:
    plan_id: str
    symbol: str
    style: str
    cohort: str
    direction: str
    entry: float
    stop: float
    targets: Sequence[TargetSpec]
    bars: Sequence[PriceBar]
    rr_to_t1: float | None = None


@dataclass
class PlanOutcome:
    plan_id: str
    tp_hit: bool
    sl_hit: bool
    mfe: float
    mae: float
    prob_touch: float
    rr_to_t1: float | None
    em_capped: bool


@dataclass
class EvaluationSummary:
    cohort: str
    style: str
    outcomes: Sequence[PlanOutcome]
    calibration_bins: Sequence[CalibrationBin]
    brier_score: float
    ece: float

    @property
    def total(self) -> int:
        return len(self.outcomes)

    @property
    def tp_hits(self) -> int:
        return sum(1 for outcome in self.outcomes if outcome.tp_hit)

    @property
    def sl_hits(self) -> int:
        return sum(1 for outcome in self.outcomes if outcome.sl_hit)

    @property
    def em_capped_tp(self) -> int:
        return sum(1 for outcome in self.outcomes if outcome.tp_hit and outcome.em_capped)

    def to_calibration_table(self) -> CalibrationTable:
        from datetime import datetime, timezone

        return CalibrationTable(
            style=self.style,
            cohort=self.cohort,
            bins=list(self.calibration_bins),
            sample_size=self.total,
            brier_score=self.brier_score,
            ece=self.ece,
            created_at=datetime.now(timezone.utc),
        )


class PlanEvaluator:
    """Replay engine for cached plan snapshots."""

    def __init__(self, *, bin_size: float = 0.2, telemetry_source: str | None = "offline") -> None:
        self.bin_size = max(0.05, min(0.5, float(bin_size)))
        self.telemetry_source = telemetry_source

    def evaluate_fixture(self, path: str | Path) -> EvaluationSummary:
        payload = json.loads(Path(path).read_text())
        return self.evaluate_payload(payload)

    def evaluate_payload(self, payload: Mapping[str, Any]) -> EvaluationSummary:
        cohort = str(payload.get("cohort") or "default")
        style = str(payload.get("style") or "intraday")
        snapshots = payload.get("plans") or []
        parsed = [self._parse_snapshot(item, style, cohort) for item in snapshots if item]
        outcomes = [self._simulate(snapshot) for snapshot in parsed]
        calibration_bins = self._build_bins(outcomes)
        brier = self._brier_score(outcomes)
        ece = self._expected_calibration_error(calibration_bins, len(outcomes))
        return EvaluationSummary(
            cohort=cohort,
            style=style,
            outcomes=outcomes,
            calibration_bins=calibration_bins,
            brier_score=brier,
            ece=ece,
        )

    def _parse_snapshot(self, raw: Mapping[str, Any], default_style: str, default_cohort: str) -> PlanSnapshot:
        plan_id = str(raw.get("plan_id") or raw.get("id") or raw.get("symbol") or "unknown")
        symbol = str(raw.get("symbol") or "UNKNOWN").upper()
        direction = str(raw.get("direction") or "long").lower()
        entry = float(raw.get("entry"))
        stop = float(raw.get("stop"))
        style = str(raw.get("style") or default_style)
        cohort = str(raw.get("cohort") or default_cohort)
        targets_payload = list(raw.get("targets") or raw.get("target_meta") or [])
        targets: List[TargetSpec] = []
        for item in targets_payload:
            if not isinstance(item, Mapping):
                continue
            label = str(item.get("label") or f"TP{len(targets) + 1}")
            price = float(item.get("price") or item.get("target") or 0.0)
            prob_touch = float(item.get("prob_touch") or item.get("probability") or 0.0)
            em_capped = bool(item.get("em_capped") or item.get("em_cap"))
            targets.append(TargetSpec(label=label, price=price, prob_touch=prob_touch, em_capped=em_capped))
        bars_payload = list(raw.get("bars") or [])
        bars: List[PriceBar] = []
        for bar in bars_payload:
            if not isinstance(bar, Mapping):
                continue
            try:
                high = float(bar.get("high"))
                low = float(bar.get("low"))
            except (TypeError, ValueError):
                continue
            bars.append(PriceBar(high=high, low=low))
        if not bars:
            raise ValueError(f"Snapshot {plan_id} missing bars for evaluation")
        rr_to_t1 = raw.get("rr_to_t1")
        try:
            rr_value = float(rr_to_t1) if rr_to_t1 is not None else None
        except (TypeError, ValueError):
            rr_value = None
        return PlanSnapshot(
            plan_id=plan_id,
            symbol=symbol,
            style=style,
            cohort=cohort,
            direction=direction,
            entry=entry,
            stop=stop,
            targets=targets,
            bars=bars,
            rr_to_t1=rr_value,
        )

    def _simulate(self, snapshot: PlanSnapshot) -> PlanOutcome:
        if not snapshot.targets:
            raise ValueError(f"Snapshot {snapshot.plan_id} missing targets")
        tp1 = snapshot.targets[0]
        tp_hit = False
        sl_hit = False
        mfe = 0.0
        mae = 0.0
        entry = snapshot.entry
        stop = snapshot.stop
        tp_price = tp1.price
        for bar in snapshot.bars:
            if snapshot.direction == "long":
                mfe = max(mfe, float(bar.high) - entry)
                mae = max(mae, entry - float(bar.low))
                if float(bar.low) <= stop:
                    sl_hit = True
                    break
                if float(bar.high) >= tp_price:
                    tp_hit = True
                    break
            else:
                mfe = max(mfe, entry - float(bar.low))
                mae = max(mae, float(bar.high) - entry)
                if float(bar.high) >= stop:
                    sl_hit = True
                    break
                if float(bar.low) <= tp_price:
                    tp_hit = True
                    break
        outcome = PlanOutcome(
            plan_id=snapshot.plan_id,
            tp_hit=tp_hit,
            sl_hit=sl_hit,
            mfe=round(mfe, 4),
            mae=round(mae, 4),
            prob_touch=_clamp(snapshot.targets[0].prob_touch),
            rr_to_t1=snapshot.rr_to_t1,
            em_capped=bool(tp1.em_capped),
        )
        source = self.telemetry_source
        if source:
            if outcome.tp_hit:
                record_tp_hit(source)
                if outcome.em_capped:
                    record_em_capped_tp(source)
            if outcome.sl_hit:
                record_sl_hit(source)
            if outcome.rr_to_t1 is not None and outcome.rr_to_t1 < 1.0:
                record_rr_below_min(source)
        return outcome

    def _build_bins(self, outcomes: Sequence[PlanOutcome]) -> List[CalibrationBin]:
        if not outcomes:
            return []
        size = self.bin_size
        bins: List[CalibrationBin] = []
        edges: List[float] = [round(i * size, 10) for i in range(int(math.ceil(1.0 / size)) + 1)]
        if edges[-1] < 1.0:
            edges.append(1.0)
        for idx in range(len(edges) - 1):
            lower = edges[idx]
            upper = edges[idx + 1]
            bucket = [outcome for outcome in outcomes if lower <= outcome.prob_touch < upper or (upper >= 1.0 and math.isclose(outcome.prob_touch, 1.0))]
            count = len(bucket)
            if count == 0:
                bins.append(CalibrationBin(lower=lower, upper=upper, count=0, avg_prediction=0.0, observed=0.0))
                continue
            avg_prediction = sum(item.prob_touch for item in bucket) / count
            observed = sum(1 for item in bucket if item.tp_hit) / count
            bins.append(
                CalibrationBin(
                    lower=lower,
                    upper=upper,
                    count=count,
                    avg_prediction=avg_prediction,
                    observed=observed,
                )
            )
        return bins

    @staticmethod
    def _brier_score(outcomes: Sequence[PlanOutcome]) -> float:
        if not outcomes:
            return 0.0
        total = len(outcomes)
        loss = 0.0
        for outcome in outcomes:
            actual = 1.0 if outcome.tp_hit else 0.0
            loss += (outcome.prob_touch - actual) ** 2
        return loss / total

    @staticmethod
    def _expected_calibration_error(bins: Sequence[CalibrationBin], total: int) -> float:
        if not bins or total <= 0:
            return 0.0
        error = 0.0
        for item in bins:
            if item.count == 0:
                continue
            error += abs(item.avg_prediction - item.observed) * (item.count / total)
        return error


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


__all__ = ["PlanEvaluator", "EvaluationSummary", "PlanOutcome"]
