"""Probability calibration utilities for plan targets.

This module provides a light-weight histogram based calibration layer that lets
us remap raw `prob_touch` estimates to empirically observed hit rates.  The
calibration tables are intentionally simple (piecewise linear interpolation
between reliability bins) so they can be persisted to JSON and applied both
offline (during evaluator replays) and live inside the `/gpt/plan` handler.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Tuple
import json
import math


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


@dataclass(frozen=True)
class CalibrationBin:
    """Reliability bin summarising raw vs observed hit rates."""

    lower: float
    upper: float
    count: int
    avg_prediction: float
    observed: float

    def to_dict(self) -> Dict[str, float | int]:
        payload = asdict(self)
        payload["lower"] = float(payload["lower"])
        payload["upper"] = float(payload["upper"])
        payload["avg_prediction"] = float(payload["avg_prediction"])
        payload["observed"] = float(payload["observed"])
        payload["count"] = int(payload["count"])
        return payload


@dataclass
class CalibrationTable:
    """Histogram based calibration table for a style/cohort."""

    style: str
    cohort: str | None
    bins: List[CalibrationBin]
    sample_size: int
    brier_score: float
    ece: float
    created_at: datetime

    def calibrate(self, probability: float) -> float:
        """Map a raw probability to the calibrated observation."""
        if not self.bins:
            return _clamp(probability)
        value = _clamp(float(probability))
        points: List[Tuple[float, float]] = []
        for item in self.bins:
            expected = _clamp(item.avg_prediction)
            observed = _clamp(item.observed)
            points.append((expected, observed))
        points.sort(key=lambda pair: pair[0])
        first_x, first_y = points[0]
        if value <= first_x:
            return _clamp(first_y)
        last_x, last_y = points[-1]
        if value >= last_x:
            return _clamp(last_y)
        for idx in range(len(points) - 1):
            x0, y0 = points[idx]
            x1, y1 = points[idx + 1]
            if value < x0:
                return _clamp(y0)
            if x0 <= value <= x1:
                span = x1 - x0
                if math.isclose(span, 0.0):
                    return _clamp((y0 + y1) / 2.0)
                ratio = (value - x0) / span
                return _clamp(y0 + ratio * (y1 - y0))
        return _clamp(points[-1][1])

    def to_payload(self) -> Dict[str, object]:
        """Serialise the table to a JSON friendly payload."""
        return {
            "style": self.style,
            "cohort": self.cohort,
            "sample_size": int(self.sample_size),
            "brier_score": round(float(self.brier_score), 6),
            "ece": round(float(self.ece), 6),
            "created_at": self.created_at.replace(tzinfo=timezone.utc).isoformat(),
            "bins": [bin_item.to_dict() for bin_item in self.bins],
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "CalibrationTable":
        style = str(payload.get("style") or "").strip() or "intraday"
        cohort = payload.get("cohort")
        sample_size = int(payload.get("sample_size") or 0)
        brier_score = float(payload.get("brier_score") or 0.0)
        ece = float(payload.get("ece") or 0.0)
        created_raw = payload.get("created_at")
        if isinstance(created_raw, str):
            try:
                created_at = datetime.fromisoformat(created_raw.replace("Z", "+00:00"))
            except ValueError:
                created_at = datetime.now(timezone.utc)
        else:
            created_at = datetime.now(timezone.utc)
        bins_payload = payload.get("bins") or []
        bins: List[CalibrationBin] = []
        for item in bins_payload:
            if not isinstance(item, Mapping):
                continue
            bins.append(
                CalibrationBin(
                    lower=float(item.get("lower") or 0.0),
                    upper=float(item.get("upper") or 0.0),
                    count=int(item.get("count") or 0),
                    avg_prediction=float(item.get("avg_prediction") or 0.0),
                    observed=float(item.get("observed") or 0.0),
                )
            )
        return cls(
            style=style,
            cohort=str(cohort) if cohort not in (None, "") else None,
            bins=bins,
            sample_size=sample_size,
            brier_score=brier_score,
            ece=ece,
            created_at=created_at if created_at.tzinfo else created_at.replace(tzinfo=timezone.utc),
        )


class CalibrationStore:
    """Registry holding calibration tables keyed by style/cohort."""

    def __init__(self) -> None:
        self._tables: Dict[Tuple[str, str | None], CalibrationTable] = {}

    @staticmethod
    def _key(style: str, cohort: str | None) -> Tuple[str, str | None]:
        style_token = (style or "intraday").strip().lower()
        cohort_token = (cohort or "").strip().lower() or None
        return style_token, cohort_token

    def register(self, table: CalibrationTable) -> None:
        key = self._key(table.style, table.cohort)
        self._tables[key] = table

    def calibrate(
        self,
        style: str,
        probability: float,
        *,
        cohort: str | None = None,
    ) -> Tuple[float, Dict[str, object] | None]:
        """Return calibrated probability alongside metadata payload."""
        table = self._lookup(style, cohort)
        if table is None:
            return _clamp(probability), None
        calibrated = table.calibrate(probability)
        payload = table.to_payload()
        payload["style"] = style
        payload["cohort"] = cohort or table.cohort
        return calibrated, payload

    def _lookup(self, style: str, cohort: str | None) -> CalibrationTable | None:
        specific_key = self._key(style, cohort)
        if specific_key in self._tables:
            return self._tables[specific_key]
        fallback_key = self._key(style, None)
        return self._tables.get(fallback_key)

    def to_payload(self) -> Dict[str, object]:
        """Serialise the entire store."""
        return {
            f"{style}:{cohort or 'default'}": table.to_payload()
            for (style, cohort), table in self._tables.items()
        }

    def save(self, path: str | Path) -> None:
        dest = Path(path)
        dest.write_text(json.dumps(self.to_payload(), indent=2, sort_keys=True))

    @classmethod
    def load(cls, path: str | Path) -> "CalibrationStore":
        store = cls()
        src = Path(path)
        if not src.exists():
            return store
        try:
            parsed = json.loads(src.read_text())
        except json.JSONDecodeError:
            return store
        if isinstance(parsed, Mapping):
            for value in parsed.values():
                if not isinstance(value, Mapping):
                    continue
                table = CalibrationTable.from_payload(value)
                store.register(table)
        return store

    def merge(self, tables: Iterable[CalibrationTable]) -> None:
        for table in tables:
            self.register(table)


__all__ = [
    "CalibrationBin",
    "CalibrationStore",
    "CalibrationTable",
]

