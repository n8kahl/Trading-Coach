"""Unified target and stop profile helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional


@dataclass(slots=True)
class TargetEngineResult:
    entry: float
    stop: float
    targets: List[float]
    probabilities: Dict[str, float]
    em_used: Optional[float]
    atr_used: Optional[float]
    snap_trace: List[Dict[str, Any]]
    meta: List[Dict[str, Any]]
    warnings: List[str]
    runner: Optional[Dict[str, Any]]
    bias: Optional[str] = None
    style: Optional[str] = None
    expected_move: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        return payload


def _clean_targets(values: List[Any]) -> List[float]:
    clean: List[float] = []
    for value in values or []:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        clean.append(numeric)
    return clean


def _probability_map(meta: List[Dict[str, Any]]) -> Dict[str, float]:
    probabilities: Dict[str, float] = {}
    for idx, item in enumerate(meta or [], start=1):
        if not isinstance(item, dict):
            continue
        prob = item.get("prob_touch") or item.get("probability")
        if prob is None:
            continue
        try:
            probability = float(prob)
        except (TypeError, ValueError):
            continue
        key = item.get("label") or f"tp{idx}"
        key = str(key).lower().replace(" ", "_")
        probabilities[key] = probability
    return probabilities


def build_target_profile(
    *,
    entry: float,
    stop: float,
    targets: List[Any],
    target_meta: Optional[List[Dict[str, Any]]],
    debug: Optional[Dict[str, Any]],
    runner: Optional[Dict[str, Any]],
    warnings: Optional[List[str]],
    atr_used: Optional[float],
    expected_move: Optional[float],
    style: Optional[str],
    bias: Optional[str],
) -> TargetEngineResult:
    meta = [dict(item) for item in target_meta or []]
    probabilities = _probability_map(meta)

    snap_trace: List[Dict[str, Any]] = []
    em_used: Optional[float] = None
    if debug and isinstance(debug, dict):
        em_candidate = debug.get("em_limit")
        try:
            em_used = float(em_candidate) if em_candidate is not None else None
        except (TypeError, ValueError):
            em_used = None
        trace = debug.get("meta") or debug.get("trace")
        if isinstance(trace, list):
            snap_trace = [dict(item) for item in trace if isinstance(item, dict)]
    cleaned_targets = _clean_targets(targets)

    atr_value = None
    if atr_used is not None:
        try:
            atr_value = float(atr_used)
        except (TypeError, ValueError):
            atr_value = None

    expected = None
    if expected_move is not None:
        try:
            expected = float(expected_move)
        except (TypeError, ValueError):
            expected = None

    return TargetEngineResult(
        entry=float(entry),
        stop=float(stop),
        targets=cleaned_targets,
        probabilities=probabilities,
        em_used=em_used,
        atr_used=atr_value,
        snap_trace=snap_trace,
        meta=meta,
        warnings=[str(w) for w in (warnings or [])],
        runner=dict(runner) if isinstance(runner, dict) else None,
        bias=bias,
        style=style,
        expected_move=expected,
    )


def build_structured_plan(
    *,
    plan_id: str,
    symbol: str,
    style: Optional[str],
    direction: Optional[str],
    profile: TargetEngineResult,
    confidence: Optional[float],
    rationale: Optional[str],
    options_block: Optional[Dict[str, Any]],
    chart_url: Optional[str],
    session: Optional[Dict[str, Any]],
    confluence: Optional[List[str]] = None,
) -> Dict[str, Any]:
    invalid = any("watch" in str(w).lower() for w in profile.warnings)
    entry_block = {"type": "limit", "level": profile.entry}
    structured = {
        "plan_id": plan_id,
        "symbol": symbol,
        "style": style,
        "direction": direction,
        "entry": entry_block,
        "invalid": invalid,
        "stop": profile.stop,
        "targets": profile.targets,
        "probabilities": profile.probabilities,
        "runner": profile.runner,
        "confluence": confluence or [],
        "confidence": confidence,
        "rationale": rationale,
        "options": options_block,
        "em_used": profile.em_used,
        "atr_used": profile.atr_used,
        "style_horizon_applied": profile.style or style,
        "chart_url": chart_url,
        "as_of": (session or {}).get("as_of") if session else None,
    }
    structured["snap_trace"] = profile.snap_trace
    structured["meta"] = profile.meta
    return structured
