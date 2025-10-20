"""Structural snapping for targets and stops."""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple, Set

from .clamp import ensure_monotonic
from .levels import directional_nodes, last_higher_low, last_lower_high, profile_nodes

MIN_ATR_MULT_TP1 = {"scalp": 0.50, "intraday": 0.75, "swing": 1.0, "leaps": 1.0}
TP2_MIN_MULT = 1.5

MAJOR_NODES = {"vah", "val", "poc", "gap_top", "gap_bottom", "gap"}


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
) -> Tuple[List[float], List[Dict[str, str]], Set[str]]:
    """
    Snap each TP to the closest forward structural node (directional).
    Returns (snapped_tps, tp_reasons, snap_tags_set).
    """

    if not raw_tps:
        return [], [], set()

    direction = (direction or "").lower()
    style = _style_token(style)
    atr_value = abs(float(atr_value or 0.0))
    entry_val = float(entry)

    ladder = directional_nodes(levels, direction, entry_val)
    used_indices: Set[int] = set()
    snap_tags: Set[str] = set()
    snapped: List[float] = []
    reasons: List[Dict[str, str | None]] = []

    def pick_node(
        minimum_mult: float,
        target_hint: float | None = None,
        prefer_major: bool = False,
    ) -> Tuple[float | None, str | None, int | None]:
        for idx, (price, label) in enumerate(ladder):
            if idx in used_indices:
                continue
            distance = abs(price - entry_val)
            if atr_value > 0 and distance < minimum_mult * atr_value:
                continue
            if target_hint is not None:
                if direction == "long" and price < target_hint:
                    continue
                if direction == "short" and price > target_hint and distance > abs(target_hint - entry_val):
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
        fallback = round(float(tp1_hint or entry_val), 2)
        if direction == "short" and fallback >= entry_val:
            fallback = round(entry_val - max(atr_value * min_mult_tp1, 0.1), 2)
        elif direction == "long" and fallback <= entry_val:
            fallback = round(entry_val + max(atr_value * min_mult_tp1, 0.1), 2)
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
    return snapped, reasons, snap_tags


def stop_from_structure(
    entry: float,
    direction: str,
    levels: Dict[str, float],
    atr_value: float,
    style: str,
    wick_buffer: float = 0.15,
) -> Tuple[float, str]:
    """
    Structure-first stop with ATR floor.
    """

    direction = (direction or "").lower()
    style = _style_token(style)
    entry_val = float(entry)
    atr_value = abs(float(atr_value or 0.0))

    fallback_mult = {
        "scalp": 1.2,
        "intraday": 1.8,
        "swing": 2.0,
        "leaps": 2.5,
    }.get(style, 1.8)

    if direction == "short":
        base = levels.get("orh")
        label = "ORH"
        if not isinstance(base, (int, float)):
            base = last_lower_high(levels)
            label = "SWING_HIGH"
        if isinstance(base, (int, float)):
            stop_price = float(base) + wick_buffer
        else:
            stop_price = entry_val + fallback_mult * atr_value
            label = "ATR_FALLBACK"
        stop_floor = entry_val + max(0.6 * atr_value, 0.1)
        stop_price = max(stop_price, stop_floor)
    else:
        base = levels.get("orl")
        label = "ORL"
        if not isinstance(base, (int, float)):
            base = last_higher_low(levels)
            label = "SWING_LOW"
        if isinstance(base, (int, float)):
            stop_price = float(base) - wick_buffer
        else:
            stop_price = entry_val - fallback_mult * atr_value
            label = "ATR_FALLBACK"
        stop_floor = entry_val - max(0.6 * atr_value, 0.1)
        stop_price = min(stop_price, stop_floor)

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


__all__ = ["snap_targets", "stop_from_structure", "build_key_levels_used", "MIN_ATR_MULT_TP1"]
