"""Deterministic geometry helpers used by scan and planning pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional

from ..plans.geometry import PlanGeometry, TargetMeta


def _normalise_levels(levels: Mapping[str, Any] | None) -> Dict[str, list[Dict[str, Any]]]:
    normalised: Dict[str, list[Dict[str, Any]]] = {}
    if not levels:
        return normalised
    for bucket, entries in levels.items():
        bucket_key = str(bucket or "").lower()
        if not bucket_key:
            continue
        bucket_items: list[Dict[str, Any]] = []
        if isinstance(entries, Mapping):
            entries = [entries]
        for entry in entries or []:
            if not isinstance(entry, Mapping):
                continue
            bucket_items.append(
                {
                    "role": entry.get("role"),
                    "label": entry.get("label"),
                    "price": entry.get("price"),
                    "distance": entry.get("distance"),
                    "source": entry.get("source"),
                }
            )
        if bucket_items:
            normalised[bucket_key] = bucket_items
    return normalised


def _serialise_targets(targets: Iterable[TargetMeta]) -> list[Dict[str, Any]]:
    serialised: list[Dict[str, Any]] = []
    for meta in targets:
        serialised.append(
            {
                "price": getattr(meta, "price", None),
                "distance": getattr(meta, "distance", None),
                "rr_multiple": getattr(meta, "rr_multiple", None),
                "prob_touch": getattr(meta, "prob_touch", None),
                "em_fraction": getattr(meta, "em_fraction", None),
                "mfe_quantile": getattr(meta, "mfe_quantile", None),
                "reason": getattr(meta, "reason", None),
                "em_capped": getattr(meta, "em_capped", None),
            }
        )
    return serialised


@dataclass(slots=True)
class GeometrySummary:
    entry: float
    stop: float
    targets: list[Dict[str, Any]]
    rr_t1: float | None
    atr_used: float | None
    expected_move: float | None
    remaining_atr: float | None
    em_used: bool
    snap_trace: list[str]
    key_levels_used: Dict[str, list[Dict[str, Any]]]


def summarize_plan_geometry(
    plan: PlanGeometry,
    *,
    entry: float,
    atr_value: float | None,
    expected_move: float | None = None,
    key_levels_used: Mapping[str, Any] | None = None,
) -> GeometrySummary:
    """Extract deterministic geometry metrics for downstream consumers."""

    targets_serialised = _serialise_targets(plan.targets)
    rr_t1 = None
    if targets_serialised:
        rr_raw = targets_serialised[0].get("rr_multiple")
        try:
            rr_t1 = float(rr_raw) if rr_raw is not None else None
        except (TypeError, ValueError):
            rr_t1 = None

    expected_move_val = expected_move
    if expected_move_val is None:
        expected_move_val = plan.em_day if getattr(plan, "em_day", None) else None

    summary = GeometrySummary(
        entry=float(entry),
        stop=float(plan.stop.price),
        targets=targets_serialised,
        rr_t1=rr_t1,
        atr_used=float(atr_value) if atr_value is not None else None,
        expected_move=float(expected_move_val) if expected_move_val is not None else None,
        remaining_atr=float(plan.ratr) if getattr(plan, "ratr", None) is not None else None,
        em_used=bool(getattr(plan, "em_used", False)),
        snap_trace=list(getattr(plan, "snap_trace", []) or []),
        key_levels_used=_normalise_levels(key_levels_used),
    )
    return summary


__all__ = ["GeometrySummary", "summarize_plan_geometry"]
