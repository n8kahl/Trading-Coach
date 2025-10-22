"""Unified structural geometry workflow."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .clamp import clamp_targets_to_em
from .expected_move import session_expected_move
from .levels import profile_nodes
from .snap import ensure_monotonic
from .snap import snap_targets, stop_from_structure, build_key_levels_used
from .runner import compute_runner
from .invariants import assert_invariants, GeometryInvariantError


@dataclass
class StructuredGeometry:
    entry: float
    stop: float
    stop_label: str
    targets: List[float]
    tp_reasons: List[Dict[str, str]]
    key_levels_used: Dict[str, List[Dict[str, object]]]
    runner_policy: Dict[str, object]
    snap_tags: Set[str]
    em_points: float
    clamp_applied: bool
    warnings: List[str]


def _normalise_tp_reasons(reasons: Iterable[Dict[str, str | None]]) -> List[Dict[str, str]]:
    output: List[Dict[str, str]] = []
    for item in reasons:
        label = str(item.get("label") or "").upper()
        reason = str(item.get("reason") or "")
        snap_tag = item.get("snap_tag")
        payload = {"label": label, "reason": reason}
        if snap_tag:
            payload["snap_tag"] = str(snap_tag).upper()
        output.append(payload)
    return output


def build_structured_geometry(
    *,
    symbol: str,
    style: str,
    direction: str,
    entry: float,
    levels: Dict[str, float],
    atr_value: float,
    plan_time: datetime,
    raw_targets: Sequence[float],
    rr_floor: float,
    em_hint: Optional[float],
) -> StructuredGeometry:
    """
    Apply structural stop/target logic, EM clamp, and runner policy for a given entry.
    """

    warnings: List[str] = []
    direction = (direction or "").lower()
    style_token = (style or "").strip().lower()
    if style_token == "leap":
        style_token = "leaps"

    entry_val = float(entry)
    atr_val = abs(float(atr_value or 0.0))

    em_points = session_expected_move(symbol, plan_time, style_token)
    if (not em_points or em_points <= 0) and em_hint:
        em_points = float(em_hint)
    if (not em_points or em_points <= 0) and atr_val > 0:
        em_points = round(atr_val * 1.8, 4)
    if em_points < 0:
        em_points = 0.0

    snapped_tps, tp_reason_payload, snap_tags = snap_targets(
        entry=entry_val,
        direction=direction,
        raw_tps=list(raw_targets),
        levels=levels,
        atr_value=atr_val,
        style=style_token,
    )
    snapped_tps = [round(tp, 2) for tp in snapped_tps]
    tp_reasons = _normalise_tp_reasons(tp_reason_payload)

    clamped_tps = clamp_targets_to_em(
        entry=entry_val,
        direction=direction,
        tps=snapped_tps,
        em_points=em_points,
        snap_tags=snap_tags,
        style=style_token,
    )
    if atr_val > 0:
        prefer_threshold = atr_val * 0.25
    else:
        prefer_threshold = 0.0
    if prefer_threshold > 0:
        preferred: List[float] = []
        for idx, clamped in enumerate(clamped_tps):
            original = snapped_tps[idx] if idx < len(snapped_tps) else clamped
            reason = tp_reasons[idx] if idx < len(tp_reasons) else {}
            if reason.get("snap_tag") and abs(clamped - original) <= prefer_threshold + 1e-9:
                preferred.append(round(original, 2))
            else:
                preferred.append(clamped)
        clamped_tps = preferred
    clamp_applied = any(abs(a - b) > 1e-6 for a, b in zip(clamped_tps, snapped_tps))
    if clamp_applied:
        warnings.append("EM clamp applied")
        for idx, (clamped, reason) in enumerate(zip(clamped_tps, tp_reasons)):
            original = snapped_tps[idx] if idx < len(snapped_tps) else clamped
            if abs(clamped - original) > 1e-6:
                reason["reason"] = f"{reason['reason']} · EM clamp"

    clamped_tps = _respaced_targets_after_clamp(
        entry=entry_val,
        direction=direction,
        targets=[float(tp) for tp in clamped_tps],
        atr_value=atr_val,
        style=style_token,
        em_points=em_points,
    )

    stop_price, stop_label = stop_from_structure(
        entry=entry_val,
        direction=direction,
        levels=levels,
        atr_value=atr_val,
        style=style_token,
    )

    key_levels_used = build_key_levels_used(
        direction=direction,
        stop_level=(stop_price, stop_label),
        tp_prices=clamped_tps,
        tp_reasons=tp_reasons,
    )

    runner_policy = compute_runner(
        entry=entry_val,
        tps=clamped_tps,
        style=style_token,
        em_points=em_points,
        atr=atr_val,
        profile_nodes=profile_nodes(levels),
    )

    try:
        assert_invariants(direction, entry_val, stop_price, clamped_tps, rr_floor)
    except GeometryInvariantError as exc:
        warnings.append("INVARIANT_BROKEN")
        warnings.append(str(exc))

    return StructuredGeometry(
        entry=entry_val,
        stop=stop_price,
        stop_label=stop_label,
        targets=clamped_tps,
        tp_reasons=tp_reasons,
        key_levels_used=key_levels_used,
        runner_policy=runner_policy,
        snap_tags=snap_tags,
        em_points=em_points,
        clamp_applied=clamp_applied,
        warnings=warnings,
    )


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


def _respaced_targets_after_clamp(
    entry: float,
    direction: str,
    targets: List[float],
    atr_value: float,
    style: str,
    em_points: float,
) -> List[float]:
    if not targets:
        return []
    direction = (direction or "").lower()
    entry_val = float(entry)
    tick = _infer_tick_size(entry_val)
    atr_val = max(float(atr_value or 0.0), 0.0)
    style_token = (style or "intraday").strip().lower()
    if style_token == "leap":
        style_token = "leaps"
    base_spacing = atr_val * 0.35 if atr_val > 0 else tick * 3
    min_spacing = max(base_spacing, tick * 3)
    cap_offset = float(em_points or 0.0)

    respaced: List[float] = []
    for price in targets:
        try:
            candidate = float(price)
        except (TypeError, ValueError):
            continue
        prev = respaced[-1] if respaced else entry_val
        if direction == "long":
            lower_bound = prev + min_spacing
            candidate = max(candidate, lower_bound)
            if cap_offset > 0:
                candidate = min(candidate, entry_val + cap_offset)
            if candidate <= prev:
                continue
        elif direction == "short":
            upper_bound = prev - min_spacing
            candidate = min(candidate, upper_bound)
            if cap_offset > 0:
                candidate = max(candidate, entry_val - cap_offset)
            if candidate >= prev:
                continue
        candidate = round(candidate, 2)
        if respaced and abs(candidate - respaced[-1]) < tick:
            continue
        respaced.append(candidate)
    if not respaced:
        return []
    target_count = len(targets)
    if target_count > len(respaced):
        direction_factor = 1.0 if direction == "long" else -1.0
        step = direction_factor * max(min_spacing, tick * 3)
        cap_bound = entry_val + direction_factor * cap_offset if cap_offset > 0 else None
        while len(respaced) < target_count:
            prev = respaced[-1] if respaced else entry_val
            candidate = prev + step
            if cap_bound is not None:
                if direction == "long":
                    candidate = min(candidate, cap_bound)
                else:
                    candidate = max(candidate, cap_bound)
            candidate = round(candidate, 2)
            if respaced and (
                (direction == "long" and candidate <= respaced[-1])
                or (direction == "short" and candidate >= respaced[-1])
            ):
                break
            respaced.append(candidate)
    return ensure_monotonic(respaced, direction)


__all__ = ["StructuredGeometry", "build_structured_geometry"]
