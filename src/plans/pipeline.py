"""Unified structural geometry workflow."""

from __future__ import annotations

import logging
import math
import os

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


logger = logging.getLogger(__name__)
_DEBUG_SNAP_LADDER = str(os.getenv("DEBUG_SNAP_LADDER", "")).strip().lower() in {"1", "true", "yes", "on"}


_STYLE_MAX_EM_FRACTION = {
    "scalp": 0.35,
    "intraday": 0.60,
    "swing": 0.90,
    "leaps": 1.00,
}


@dataclass
class StructuredGeometry:
    entry: float
    stop: float
    stop_label: str
    stop_meta: Dict[str, object] | None
    targets: List[float]
    tp_reasons: List[Dict[str, str]]
    key_levels_used: Dict[str, List[Dict[str, object]]]
    runner_policy: Dict[str, object]
    snap_tags: Set[str]
    em_points: float
    clamp_applied: bool
    warnings: List[str]


def _normalise_tp_reasons(reasons: Iterable[Dict[str, object]]) -> List[Dict[str, object]]:
    output: List[Dict[str, object]] = []
    for item in reasons:
        label = str(item.get("label") or "").upper()
        reason = str(item.get("reason") or "")
        snap_tag = item.get("snap_tag")
        payload = {"label": label, "reason": reason}
        if snap_tag:
            payload["snap_tag"] = str(snap_tag).upper()
        if item.get("watch_plan"):
            payload["watch_plan"] = "true"
        for key in (
            "ideal_price",
            "ideal_distance",
            "ideal_fraction",
            "snap_price",
            "snap_distance",
            "snap_fraction",
            "snap_deviation",
            "candidate_nodes",
            "synthetic",
            "selected_node",
            "rr_multiple",
            "fraction",
            "em_cap_relaxed",
            "outside_ideal_band",
            "no_structural",
            "modifiers",
            "rr_floor",
            "distance",
            "synthetic_meta",
            "raw_rr_multiple",
            "snap_rr_multiple",
        ):
            if key in item:
                payload[key] = item[key]
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
    if style_token in {"0dte", "zero_dte"}:
        style_token = "scalp"
    if style_token == "leap":
        style_token = "leaps"

    entry_val = float(entry)
    atr_val = abs(float(atr_value or 0.0))
    max_em_fraction = _STYLE_MAX_EM_FRACTION.get(style_token, _STYLE_MAX_EM_FRACTION["intraday"])

    em_points = session_expected_move(symbol, plan_time, style_token)
    if (not em_points or em_points <= 0) and em_hint:
        em_points = float(em_hint)
    if (not em_points or em_points <= 0) and atr_val > 0:
        em_points = round(atr_val * 1.8, 4)
    if em_points < 0:
        em_points = 0.0

    stop_price, stop_label, stop_meta = stop_from_structure(
        entry=entry_val,
        direction=direction,
        levels=levels,
        atr_value=atr_val,
        style=style_token,
        expected_move=em_points,
    )

    risk_value: Optional[float] = None
    if stop_price is not None:
        if direction == "long":
            risk_candidate = entry_val - stop_price
        else:
            risk_candidate = stop_price - entry_val
        if risk_candidate > 0:
            risk_value = risk_candidate

    snapped_tps, tp_reason_payload, snap_tags = snap_targets(
        entry=entry_val,
        direction=direction,
        raw_tps=list(raw_targets),
        levels=levels,
        atr_value=atr_val,
        style=style_token,
        expected_move=em_points,
        max_em_fraction=max_em_fraction,
        stop_price=stop_price,
        rr_floor=rr_floor,
    )
    watch_plan_flag = "WATCH_PLAN" in snap_tags
    if watch_plan_flag:
        warnings.append("TP1 RR below floor")
    snap_reference = [round(tp, 2) for tp in snapped_tps]
    reason_reference = [dict(reason) for reason in tp_reason_payload]
    if _DEBUG_SNAP_LADDER and tp_reason_payload:
        candidate_nodes = tp_reason_payload[0].get("candidate_nodes") if tp_reason_payload else None
        if isinstance(candidate_nodes, list) and candidate_nodes:
            ladder_payload: List[Dict[str, object]] = []
            for node in candidate_nodes[:24]:
                if not isinstance(node, dict):
                    continue
                ladder_payload.append(
                    {
                        "index": node.get("index"),
                        "label": node.get("label"),
                        "price": node.get("price"),
                        "distance": node.get("distance"),
                        "rr": node.get("rr_multiple"),
                        "picked": node.get("picked"),
                        "decisions": node.get("decisions"),
                    }
                )
            if ladder_payload:
                logger.debug("tp1 ladder snapshot", extra={"debug": {"snap": {"ladder": ladder_payload}}})
    if snapped_tps and raw_targets:
        adjusted_tps: List[float] = []
        adjusted_reasons: List[Dict[str, str | None]] = []
        adjusted_tags: Set[str] = set()
        for idx, snapped_price in enumerate(snapped_tps):
            raw_price = float(raw_targets[idx]) if idx < len(raw_targets) else float(snapped_price)
            reason = dict(tp_reason_payload[idx]) if idx < len(tp_reason_payload) else {"label": f"TP{idx + 1}", "reason": "Stats target", "snap_tag": None}
            within_em_bounds = True
            if em_points and em_points > 0:
                distance = abs(float(snapped_price) - entry_val)
                within_em_bounds = distance <= float(em_points) * 1.001
            snap_tag_value = reason.get("snap_tag")
            synthetic_flag = bool(reason.get("synthetic"))
            has_structural = bool(reason.get("selected_node"))
            no_structural_flag = bool(reason.get("no_structural"))
            all_nodes_outside_em = False
            if idx == 0 and synthetic_flag and em_points and em_points > 0:
                max_fraction_style = _STYLE_MAX_EM_FRACTION.get(style_token)
                if max_fraction_style:
                    max_distance_cap = float(max_fraction_style) * float(em_points)
                    candidate_nodes = reason.get("candidate_nodes")
                    if isinstance(candidate_nodes, list) and candidate_nodes:
                        distances: List[float] = []
                        for node in candidate_nodes:
                            if not isinstance(node, dict):
                                continue
                            price_val = node.get("raw_price", node.get("price"))
                            try:
                                numeric_price = float(price_val)
                            except (TypeError, ValueError):
                                continue
                            distances.append(abs(numeric_price - entry_val))
                        if distances and all(distance > max_distance_cap + 1e-6 for distance in distances):
                            all_nodes_outside_em = True
            force_tp1_fallback = synthetic_flag and not has_structural and idx == 0 and (no_structural_flag or all_nodes_outside_em)
            accept_snap = False
            keep_reason: str | None = None
            if idx == 0:
                meets_rr = False
                meets_distance = False
                raw_distance = abs(float(raw_price) - entry_val)
                snap_distance = abs(float(snapped_price) - entry_val)
                atr_threshold_tp1 = atr_val * 0.25 if atr_val > 0 else 0.0
                if atr_threshold_tp1 > 0 and (raw_distance - snap_distance) >= atr_threshold_tp1 - 1e-9:
                    meets_distance = True
                if risk_value and risk_value > 0:
                    raw_reward = raw_price - entry_val if direction == "long" else entry_val - raw_price
                    snap_reward = snapped_price - entry_val if direction == "long" else entry_val - snapped_price
                    if raw_reward > 0 and snap_reward > 0:
                        raw_rr = raw_reward / risk_value
                        snap_rr = snap_reward / risk_value
                        reason.setdefault("raw_rr_multiple", round(raw_rr, 2))
                        reason.setdefault("snap_rr_multiple", round(snap_rr, 2))
                        if snap_rr - raw_rr >= 0.20 - 1e-9:
                            meets_rr = True
                base_accept = bool(snap_tag_value or has_structural)
                if base_accept and not force_tp1_fallback and within_em_bounds and (meets_rr or meets_distance):
                    accept_snap = True
                if not accept_snap and not force_tp1_fallback:
                    keep_reason = "KEEP_RAW_TARGET_BETTER_BALANCE"
            else:
                improves_rr = True
                if direction == "long" and snapped_price is not None:
                    improves_rr = float(snapped_price) >= float(raw_price) - 1e-6
                if direction == "short" and snapped_price is not None:
                    improves_rr = float(snapped_price) <= float(raw_price) + 1e-6
                if (snap_tag_value or improves_rr) and within_em_bounds:
                    accept_snap = True

            if force_tp1_fallback:
                accept_snap = False

            if accept_snap:
                adjusted_tps.append(round(float(snapped_price), 2))
                if reason.get("snap_tag"):
                    adjusted_tags.add(str(reason["snap_tag"]))
                adjusted_reasons.append(reason)
            else:
                fallback_price = round(float(raw_price), 2)
                if keep_reason:
                    adjusted_reasons.append({"label": str(reason.get("label") or f"TP{idx + 1}"), "reason": keep_reason})
                else:
                    reason["reason"] = f"{reason.get('reason') or 'Stats target'} · Snap skipped"
                    reason.pop("snap_tag", None)
                    reason.pop("selected_node", None)
                    reason.pop("modifiers", None)
                    adjusted_reasons.append(reason)
                adjusted_tps.append(fallback_price)
        snapped_tps = adjusted_tps
        tp_reason_payload = adjusted_reasons
        snap_tags = adjusted_tags
    snapped_tps = [round(tp, 2) for tp in snapped_tps]
    tp_reasons = _normalise_tp_reasons(tp_reason_payload)
    baseline_tp_reasons = _normalise_tp_reasons(reason_reference)

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
        preferred_prices: List[float] = []
        updated_snap_tags: Set[str] = set(snap_tags)
        em_val = float(em_points) if isinstance(em_points, (int, float)) else 0.0
        for idx, clamped in enumerate(clamped_tps):
            original = snapped_tps[idx] if idx < len(snapped_tps) else clamped
            reason = tp_reasons[idx] if idx < len(tp_reasons) else {}
            previous_tag = reason.get("snap_tag")
            candidate_nodes = reason.get("candidate_nodes") if isinstance(reason.get("candidate_nodes"), list) else []
            candidate_choice: Dict[str, object] | None = None
            if candidate_nodes:
                current_distance = abs(original - entry_val)
                current_rr = reason.get("rr_multiple")
                if current_rr is None and risk_value:
                    reward_current = original - entry_val if direction == "long" else entry_val - original
                    if reward_current > 0:
                        current_rr = round(reward_current / risk_value, 2)
                for node in candidate_nodes:
                    if not isinstance(node, dict):
                        continue
                    price_val = node.get("raw_price", node.get("price"))
                    try:
                        price_candidate = float(price_val)
                    except (TypeError, ValueError):
                        continue
                    distance_candidate = abs(price_candidate - entry_val)
                    if distance_candidate >= current_distance - 1e-9:
                        continue
                    if (current_distance - distance_candidate) > prefer_threshold + 1e-9:
                        continue
                    rr_candidate = node.get("rr_multiple")
                    if rr_candidate is None and risk_value:
                        reward_candidate = price_candidate - entry_val if direction == "long" else entry_val - price_candidate
                        if reward_candidate > 0:
                            rr_candidate = round(reward_candidate / risk_value, 2)
                    try:
                        rr_candidate_val = float(rr_candidate) if rr_candidate is not None else None
                    except (TypeError, ValueError):
                        rr_candidate_val = None
                    if rr_candidate_val is None:
                        continue
                    try:
                        current_rr_val = float(current_rr) if current_rr is not None else None
                    except (TypeError, ValueError):
                        current_rr_val = None
                    if current_rr_val is not None and rr_candidate_val + 1e-9 < current_rr_val - 1e-9:
                        continue
                    label_val = str(node.get("label") or "").upper()
                    if candidate_choice is None or distance_candidate < candidate_choice["distance"] - 1e-9 or (
                        math.isclose(distance_candidate, candidate_choice["distance"], abs_tol=1e-6)
                        and rr_candidate_val > candidate_choice["rr"] + 1e-9
                    ):
                        candidate_choice = {
                            "price": price_candidate,
                            "distance": distance_candidate,
                            "rr": rr_candidate_val,
                            "label": label_val,
                            "node_ref": node,
                        }
            prefer_original = reason.get("snap_tag") and abs(clamped - original) <= prefer_threshold + 1e-9
            if candidate_choice:
                chosen_price = round(candidate_choice["price"], 2)
                preferred_prices.append(chosen_price)
                if previous_tag:
                    previous_upper = str(previous_tag).upper()
                    if previous_upper != "WATCH_PLAN":
                        updated_snap_tags.discard(previous_upper)
                reason["reason"] = "Nearest structure within 0.25×ATR"
                reason["snap_price"] = chosen_price
                reason["snap_distance"] = round(candidate_choice["distance"], 3)
                if em_val > 0:
                    fraction_value = candidate_choice["distance"] / em_val
                    reason["snap_fraction"] = round(fraction_value, 3)
                    reason["fraction"] = round(fraction_value, 3)
                ideal_distance_val = reason.get("ideal_distance")
                if isinstance(ideal_distance_val, (int, float)):
                    reason["snap_deviation"] = round(candidate_choice["distance"] - float(ideal_distance_val), 3)
                reason["snap_tag"] = candidate_choice["label"]
                reason["selected_node"] = candidate_choice["label"]
                reason["rr_multiple"] = round(candidate_choice["rr"], 2)
                reason.pop("em_cap_relaxed", None)
                reason.pop("outside_ideal_band", None)
                reason["synthetic"] = False
                if candidate_nodes:
                    for node in candidate_nodes:
                        if not isinstance(node, dict):
                            continue
                        price_val = node.get("raw_price", node.get("price"))
                        try:
                            node_price = float(price_val)
                        except (TypeError, ValueError):
                            node_price = None
                        picked_flag = node_price is not None and math.isclose(node_price, candidate_choice["price"], abs_tol=1e-6)
                        node["picked"] = picked_flag
                        if picked_flag:
                            picked_for = node.setdefault("picked_for", [])
                            if isinstance(picked_for, list) and reason.get("label") not in picked_for:
                                picked_for.append(reason.get("label"))
                        else:
                            picked_for = node.get("picked_for")
                            if isinstance(picked_for, list) and reason.get("label") in picked_for:
                                picked_for.remove(reason.get("label"))
                updated_snap_tags.add(candidate_choice["label"])
            elif prefer_original:
                preferred_prices.append(round(original, 2))
                if previous_tag:
                    updated_snap_tags.add(str(previous_tag).upper())
            else:
                preferred_prices.append(round(clamped, 2))
                if previous_tag:
                    updated_snap_tags.add(str(previous_tag).upper())
        clamped_tps = preferred_prices
        snap_tags = updated_snap_tags
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

    if prefer_threshold > 0 and snap_reference:
        recomputed_tags: Set[str] = set()
        for idx, price in enumerate(clamped_tps):
            if idx >= len(snap_reference):
                continue
            original_price = snap_reference[idx]
            if abs(price - original_price) <= prefer_threshold + 1e-6:
                if idx < len(tp_reasons):
                    current_reason = tp_reasons[idx]
                    if current_reason.get("reason", "").endswith("Snap skipped"):
                        continue
                    if current_reason.get("reason") == "Nearest structure within 0.25×ATR":
                        continue
                baseline_reason = baseline_tp_reasons[idx] if idx < len(baseline_tp_reasons) else None
                if baseline_reason:
                    snap_tag = baseline_reason.get("snap_tag")
                    if snap_tag:
                        recomputed_tags.add(str(snap_tag).upper())
                        tp_reasons[idx] = dict(baseline_reason)
        if recomputed_tags:
            watch_preserved = "WATCH_PLAN" in snap_tags
            snap_tags = recomputed_tags
            if watch_preserved:
                snap_tags.add("WATCH_PLAN")

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

    if watch_plan_flag:
        warnings.append("WATCH_PLAN")
    else:
        try:
            assert_invariants(direction, entry_val, stop_price, clamped_tps, rr_floor)
        except GeometryInvariantError as exc:
            warnings.append("INVARIANT_BROKEN")
            warnings.append(str(exc))

    return StructuredGeometry(
        entry=entry_val,
        stop=stop_price,
        stop_label=stop_label,
        stop_meta=stop_meta,
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
