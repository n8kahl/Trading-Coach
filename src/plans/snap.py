"""Structural snapping for targets and stops."""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Sequence, Tuple, Set

from .clamp import ensure_monotonic
from .levels import directional_nodes, last_higher_low, last_lower_high, profile_nodes
from .targets import TPIdeal, tp_ideals

MIN_ATR_MULT_TP1 = {"scalp": 0.50, "intraday": 0.75, "swing": 1.0, "leaps": 1.0}
TP2_MIN_MULT = 1.5

MAJOR_NODES = {"vah", "val", "poc", "gap_top", "gap_bottom", "gap", "pwh", "pwl", "pwc", "pdh", "pdl", "pdc"}


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


def compute_adaptive_wick_buffer(atr: float, tick: float) -> float:
    tick = max(float(tick or 0.01), 1e-4)
    atr = max(float(atr or 0.0), 0.0)
    base_ticks = 0.15 * (atr / tick)
    buffer = base_ticks * tick
    return max(0.05, min(0.35, buffer))


def apply_atr_floor(entry: float, structural_stop: float, atr: float, direction: str, style: str) -> float:
    k_map = {"scalp": 0.6, "intraday": 0.9, "swing": 1.2, "leaps": 1.5}
    style_token = _style_token(style)
    k = k_map.get(style_token, 0.9)
    atr = max(float(atr or 0.0), 0.0)
    entry = float(entry)
    structural = float(structural_stop)
    if atr <= 0:
        return structural
    if direction == "short":
        floor = entry + k * atr
        return max(structural, floor)
    if direction == "long":
        floor = entry - k * atr
        return min(structural, floor)
    return structural


def _style_token(style: str) -> str:
    token = (style or "").strip().lower()
    if token == "leap":
        token = "leaps"
    return token or "intraday"


def snap_targets(
    entry: float,
    direction: str,
    raw_tps: Sequence[float],
    levels: Dict[str, float],
    atr_value: float,
    style: str,
    *,
    expected_move: float | None = None,
    max_em_fraction: float | None = None,
    stop_price: float | None = None,
    rr_floor: float | None = None,
) -> Tuple[List[float], List[Dict[str, object]], Set[str]]:
    """
    Snap each TP to the closest structural node, prioritising EM-based ideals when available.
    Returns (snapped_tps, tp_reasons, snap_tags_set).
    """

    if not raw_tps and (expected_move is None or expected_move <= 0):
        return [], [], set()

    expected_move_val = float(expected_move) if isinstance(expected_move, (int, float)) else 0.0
    direction = (direction or "").lower()
    style_token = _style_token(style)
    atr_value = abs(float(atr_value or 0.0))
    entry_val = float(entry)

    if expected_move_val <= 0:
        return _snap_targets_legacy(
            entry=entry_val,
            direction=direction,
            raw_tps=raw_tps,
            levels=levels,
            atr_value=atr_value,
            style=style_token,
        )

    ideals = tp_ideals(style_token, expected_move_val)
    if not ideals:
        return _snap_targets_legacy(
            entry=entry_val,
            direction=direction,
            raw_tps=raw_tps,
            levels=levels,
            atr_value=atr_value,
            style=style_token,
        )

    ladder = directional_nodes(levels, direction, entry_val)
    used_indices: Set[int] = set()
    snap_tags: Set[str] = set()
    snapped: List[float] = []
    reasons: List[Dict[str, object]] = []

    tick = _infer_tick_size(entry_val)
    tolerance = max(expected_move_val * 0.15, tick * 2)
    raw_len = len(raw_tps)
    max_fraction = float(max_em_fraction) if isinstance(max_em_fraction, (int, float)) and max_em_fraction > 0 else None

    stop_val = float(stop_price) if isinstance(stop_price, (int, float)) else None
    if stop_val is not None and not math.isfinite(stop_val):
        stop_val = None
    risk = None
    if stop_val is not None:
        risk_candidate = entry_val - stop_val if direction == "long" else stop_val - entry_val
        risk_candidate = abs(risk_candidate)
        if risk_candidate > tick * 0.25:
            risk = risk_candidate

    ladder_snapshot: List[Dict[str, object]] = []
    for price, label in ladder[:12]:
        price_val = float(price)
        distance_val = abs(price_val - entry_val)
        fraction_val = distance_val / expected_move_val if expected_move_val > 0 else None
        candidate: Dict[str, object] = {
            "label": str(label).upper(),
            "price": round(price_val, 2),
            "distance": round(distance_val, 3),
        }
        if fraction_val is not None and math.isfinite(fraction_val):
            candidate["fraction"] = round(fraction_val, 3)
        if risk and risk > 0:
            reward_candidate = price_val - entry_val if direction == "long" else entry_val - price_val
            rr_candidate = reward_candidate / risk if reward_candidate > 0 else 0.0
            if rr_candidate > 0:
                candidate["rr_multiple"] = round(rr_candidate, 2)
        ladder_snapshot.append(candidate)

    def _round_price(value: float) -> float:
        if tick > 0:
            steps = round(value / tick)
            value = steps * tick
        return round(value, 2)

    def _fraction_tag(fraction: float) -> str:
        scaled = max(0.0, fraction) * 100.0
        return f"EM{int(round(scaled))}"

    for idx, ideal in enumerate(ideals):
        if ideal.optional and raw_len < idx + 1:
            continue
        tp_label = ideal.label or f"TP{idx + 1}"
        min_distance, max_distance = ideal.distance_bounds(expected_move_val, max_fraction=max_fraction)
        ideal_distance = ideal.clamp_distance(expected_move_val, max_fraction=max_fraction)

        if snapped:
            previous = abs(snapped[-1] - entry_val)
            min_required = previous + max(tick, 0.01)
            if min_distance < min_required:
                min_distance = min_required
            if ideal_distance < min_distance:
                ideal_distance = min_distance
            if max_distance < min_distance:
                max_distance = min_distance

        ideal_distance_target = max(min_distance, min(ideal_distance, max_distance))
        ideal_price_val = entry_val + ideal_distance_target if direction == "long" else entry_val - ideal_distance_target
        ideal_fraction_val = ideal_distance_target / expected_move_val if expected_move_val > 0 else 0.0

        best_idx: int | None = None
        best_price: float | None = None
        best_label: str | None = None
        best_distance: float | None = None
        best_overshoot: float | None = None

        for ladder_idx, (price, label) in enumerate(ladder):
            if ladder_idx in used_indices:
                continue
            price_val = float(price)
            if direction == "long" and price_val <= entry_val + 1e-9:
                continue
            if direction == "short" and price_val >= entry_val - 1e-9:
                continue
            distance = abs(price_val - entry_val)
            if distance < min_distance - 1e-9:
                continue
            if distance < ideal_distance_target - 1e-9:
                continue
            if distance > max_distance + 1e-9:
                continue
            if snapped:
                last = snapped[-1]
                if direction == "long" and price_val <= last + 1e-6:
                    continue
                if direction == "short" and price_val >= last - 1e-6:
                    continue
            overshoot = distance - ideal_distance_target
            if best_idx is None or (
                overshoot < (best_overshoot or float("inf")) - 1e-9
                or (
                    best_overshoot is not None
                    and math.isclose(overshoot, best_overshoot, abs_tol=1e-6)
                    and distance < (best_distance or float("inf"))
                )
            ):
                best_idx = ladder_idx
                best_price = price_val
                best_label = label
                best_distance = distance
                best_overshoot = overshoot

        snap_tag: str | None = None
        price_final: float
        fraction_used: float
        synthetic_flag = False
        if (
            best_idx is not None
            and best_price is not None
            and (best_overshoot is None or best_overshoot <= tolerance + 1e-9)
        ):
            price_final = _round_price(best_price)
            used_indices.add(best_idx)
            snap_tag = str(best_label or "").upper()
            snap_tags.add(snap_tag)
            distance_used = abs(price_final - entry_val)
            fraction_used = distance_used / expected_move_val if expected_move_val > 0 else 0.0
            reason_text = f"Snapped to {snap_tag} · Ideal {_fraction_tag(fraction_used)}"
        else:
            synthetic_flag = True
            synthetic_distance = ideal_distance_target
            price_estimate = entry_val + synthetic_distance if direction == "long" else entry_val - synthetic_distance
            price_final = _round_price(price_estimate)
            distance_used = abs(price_final - entry_val)
            fraction_used = distance_used / expected_move_val if expected_move_val > 0 else 0.0
            synthetic_tag = _fraction_tag(fraction_used).upper()
            snap_tag = synthetic_tag
            snap_tags.add(synthetic_tag)
            reason_text = f"Ideal {synthetic_tag} (synthetic)"

        snapped.append(price_final)
        reason_payload: Dict[str, object] = {
            "label": tp_label.upper(),
            "reason": reason_text,
            "snap_tag": snap_tag,
            "ideal_price": round(ideal_price_val, 2),
            "ideal_distance": round(ideal_distance_target, 3),
            "ideal_fraction": round(ideal_fraction_val, 3),
            "snap_price": round(price_final, 2),
            "snap_distance": round(distance_used, 3),
            "snap_fraction": round(fraction_used, 3),
            "snap_deviation": round(distance_used - ideal_distance_target, 3),
            "synthetic": synthetic_flag,
        }
        if not synthetic_flag and snap_tag:
            reason_payload["selected_node"] = snap_tag
        if idx == 0 and ladder_snapshot:
            reason_payload["candidate_nodes"] = ladder_snapshot

        if risk and risk > 0 and price_final is not None:
            reward = price_final - entry_val if direction == "long" else entry_val - price_final
            rr_multiple = reward / risk if reward > 0 else 0.0
            if idx == 0 and rr_floor and rr_multiple < float(rr_floor) - 1e-6:
                snap_tags.add("WATCH_PLAN")
                reason_payload["reason"] = f"{reason_text} · RR {rr_multiple:.2f}<{float(rr_floor):.2f}"
                reason_payload["watch_plan"] = "true"
                reason_payload["rr_multiple"] = round(rr_multiple, 2)
        elif risk and risk > 0:
            reason_payload["rr_multiple"] = round((price_final - entry_val if direction == "long" else entry_val - price_final) / risk, 2) if risk > 0 else None

        reasons.append(reason_payload)

    if not snapped:
        return _snap_targets_legacy(
            entry=entry_val,
            direction=direction,
            raw_tps=raw_tps,
            levels=levels,
            atr_value=atr_value,
            style=style_token,
        )

    snapped = ensure_monotonic(snapped, direction)
    normalised_reasons: List[Dict[str, object]] = []
    for entry_reason in reasons:
        payload = {"label": str(entry_reason.get("label") or "").upper(), "reason": str(entry_reason.get("reason") or "")}
        snap_tag = entry_reason.get("snap_tag")
        if snap_tag:
            payload["snap_tag"] = str(snap_tag).upper()
        if "watch_plan" in entry_reason:
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
        ):
            if key in entry_reason:
                payload[key] = entry_reason[key]
        normalised_reasons.append(payload)
    return snapped, normalised_reasons, snap_tags


def _snap_targets_legacy(
    *,
    entry: float,
    direction: str,
    raw_tps: Sequence[float],
    levels: Dict[str, float],
    atr_value: float,
    style: str,
) -> Tuple[List[float], List[Dict[str, object]], Set[str]]:
    if not raw_tps:
        return [], [], set()

    ladder = directional_nodes(levels, direction, entry)
    used_indices: Set[int] = set()
    snap_tags: Set[str] = set()
    snapped: List[float] = []
    reasons: List[Dict[str, object]] = []

    htf_labels = {"pwh", "pwl", "pwc", "pdh", "pdl", "pdc"}

    def pick_node(
        minimum_mult: float,
        target_hint: float | None = None,
        prefer_major: bool = False,
    ) -> Tuple[float | None, str | None, int | None]:
        for idx, (price, label) in enumerate(ladder):
            if idx in used_indices:
                continue
            distance = abs(price - entry)
            if atr_value > 0 and distance < minimum_mult * atr_value:
                continue
            if target_hint is not None:
                if direction == "long" and price < target_hint:
                    continue
                if direction == "short" and price > target_hint and distance > abs(target_hint - entry):
                    continue
            if label.lower() in htf_labels and target_hint is not None:
                if abs(price - float(target_hint)) > max(atr_value * 0.25, 0.01):
                    continue
            if prefer_major and label.lower() not in MAJOR_NODES:
                continue
            return price, label, idx
        if prefer_major:
            return pick_node(minimum_mult, target_hint, prefer_major=False)
        return None, None, None

    tp1_hint = raw_tps[0] if len(raw_tps) >= 1 else None
    tp2_hint = raw_tps[1] if len(raw_tps) >= 2 else None
    tp3_hint = raw_tps[2] if len(raw_tps) >= 3 else None

    min_mult_tp1 = MIN_ATR_MULT_TP1.get(style, 0.75)
    price, label, idx = pick_node(min_mult_tp1, tp1_hint)
    if price is not None:
        snapped_price = round(price, 2)
        snapped.append(snapped_price)
        snap_tags.add(label.upper())
        used_indices.add(idx)
        reasons.append({"label": "TP1", "reason": f"Snapped to {label.upper()}", "snap_tag": label.upper()})
    else:
        fallback = round(float(tp1_hint or entry), 2)
        if direction == "short" and fallback >= entry:
            fallback = round(entry - max(atr_value * min_mult_tp1, 0.1), 2)
        elif direction == "long" and fallback <= entry:
            fallback = round(entry + max(atr_value * min_mult_tp1, 0.1), 2)
        snapped.append(fallback)
        reasons.append({"label": "TP1", "reason": "ATR multiple", "snap_tag": None})

    price, label, idx = pick_node(TP2_MIN_MULT, tp2_hint)
    if price is not None:
        valid_direction = (
            (direction == "long" and price > snapped[0]) or (direction == "short" and price < snapped[0])
        )
        if valid_direction:
            snapped_price = round(price, 2)
            snapped.append(snapped_price)
            snap_tags.add(label.upper())
            used_indices.add(idx)
            reasons.append({"label": "TP2", "reason": f"Snapped to {label.upper()}", "snap_tag": label.upper()})
        else:
            price = None
    if price is None:
        fallback = float(tp2_hint or snapped[0])
        if direction == "long":
            fallback = max(fallback, snapped[0] + max(atr_value * 0.5, 0.1))
        else:
            fallback = min(fallback, snapped[0] - max(atr_value * 0.5, 0.1))
        snapped.append(round(fallback, 2))
        reasons.append({"label": "TP2", "reason": "ATR/ordering", "snap_tag": None})

    price, label, idx = pick_node(TP2_MIN_MULT, tp3_hint, prefer_major=True)
    if price is not None:
        valid_direction = (
            (direction == "long" and price > snapped[-1]) or (direction == "short" and price < snapped[-1])
        )
        if valid_direction:
            snapped_price = round(price, 2)
            snapped.append(snapped_price)
            snap_tags.add(label.upper())
            used_indices.add(idx)
            reasons.append({"label": "TP3", "reason": f"Snapped to {label.upper()}", "snap_tag": label.upper()})
        else:
            price = None
    if price is None and len(raw_tps) >= 3:
        fallback = float(raw_tps[2])
        if direction == "long":
            fallback = max(fallback, snapped[-1] + max(atr_value * 0.5, 0.1))
        else:
            fallback = min(fallback, snapped[-1] - max(atr_value * 0.5, 0.1))
        snapped.append(round(fallback, 2))
        reasons.append({"label": "TP3", "reason": "EM/HTF preference", "snap_tag": None})
    elif price is None and len(snapped) < 3:
        snapped.append(snapped[-1])
        reasons.append({"label": "TP3", "reason": "Inherited from TP2", "snap_tag": None})

    snapped = ensure_monotonic(snapped, direction)
    normalised_reasons: List[Dict[str, object]] = []
    for entry_reason in reasons:
        payload = {"label": str(entry_reason.get("label") or "").upper(), "reason": str(entry_reason.get("reason") or "")}
        snap_tag = entry_reason.get("snap_tag")
        if snap_tag:
            payload["snap_tag"] = str(snap_tag).upper()
        normalised_reasons.append(payload)
    return snapped, normalised_reasons, snap_tags


def stop_from_structure(
    entry: float,
    direction: str,
    levels: Dict[str, float],
    atr_value: float,
    style: str,
    wick_buffer: float | None = None,
    tick: float | None = None,
) -> Tuple[float, str]:
    """
    Structure-first stop with ATR floor.
    """

    direction = (direction or "").lower()
    style = _style_token(style)
    entry_val = float(entry)
    atr_value = abs(float(atr_value or 0.0))
    tick_size = float(tick) if isinstance(tick, (int, float)) and tick and tick > 0 else _infer_tick_size(entry_val)
    wick = (
        float(wick_buffer)
        if isinstance(wick_buffer, (int, float)) and wick_buffer and wick_buffer > 0
        else compute_adaptive_wick_buffer(atr_value, tick_size)
    )

    fallback_mult = {
        "scalp": 1.2,
        "intraday": 1.8,
        "swing": 2.0,
        "leaps": 2.5,
    }.get(style, 1.8)

    def _clamp(price: float) -> float:
        if direction == "long":
            return min(price, entry_val - tick_size)
        return max(price, entry_val + tick_size)

    if direction == "short":
        base = levels.get("orh")
        label = "ORH"
        if not isinstance(base, (int, float)):
            base = last_lower_high(levels)
            label = "SWING_HIGH"
        if isinstance(base, (int, float)):
            stop_structural = float(base) + wick
        else:
            stop_structural = entry_val + fallback_mult * atr_value
            label = "ATR_FALLBACK"
        stop_price = apply_atr_floor(entry_val, stop_structural, atr_value, direction, style)
        stop_price = _clamp(stop_price)
    else:
        base = levels.get("orl")
        label = "ORL"
        if not isinstance(base, (int, float)):
            base = last_higher_low(levels)
            label = "SWING_LOW"
        if isinstance(base, (int, float)):
            stop_structural = float(base) - wick
        else:
            stop_structural = entry_val - fallback_mult * atr_value
            label = "ATR_FALLBACK"
        stop_price = apply_atr_floor(entry_val, stop_structural, atr_value, direction, style)
        stop_price = _clamp(stop_price)

    return round(stop_price, 2), label


def build_key_levels_used(
    direction: str,
    stop_level: Tuple[float, str] | None,
    tp_prices: Sequence[float],
    tp_reasons: Sequence[Dict[str, str | None]],
) -> Dict[str, List[Dict[str, object]]]:
    session_entries: List[Dict[str, object]] = []
    if stop_level is not None:
        price, label = stop_level
        session_entries.append(
            {"role": "stop", "label": label.upper(), "price": round(price, 2), "source": "session"}
        )
    for idx, (price, reason) in enumerate(zip(tp_prices, tp_reasons), start=1):
        label = reason.get("snap_tag")
        if label:
            session_entries.append(
                {"role": f"tp{idx}", "label": str(label).upper(), "price": round(float(price), 2), "source": "session"}
            )

    structural_entries: List[Dict[str, object]] = []
    return {
        "session": session_entries,
        "structural": structural_entries,
    }


__all__ = [
    "snap_targets",
    "stop_from_structure",
    "build_key_levels_used",
    "MIN_ATR_MULT_TP1",
    "compute_adaptive_wick_buffer",
    "apply_atr_floor",
]
