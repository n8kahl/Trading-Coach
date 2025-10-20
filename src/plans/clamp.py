"""Target clamping utilities aligned with expected-move constraints."""

from __future__ import annotations

from typing import Iterable, List, Sequence

STRONG_SNAP_TAGS = {"VAH", "VAL", "POC", "GAP FILL", "GAP_FILL", "GAP", "MAJOR HTF", "HTF"}


def ensure_monotonic(tps: Sequence[float], direction: str) -> List[float]:
    """Return a strictly monotonic list of targets respecting trade direction."""

    direction = (direction or "").lower()
    if not tps:
        return []

    adjusted: List[float] = []
    if direction == "short":
        ceiling = float("inf")
        for tp in tps:
            try:
                value = round(float(tp), 2)
            except (TypeError, ValueError):
                continue
            if value >= ceiling:
                value = round(ceiling - 0.01, 2)
            ceiling = value
            adjusted.append(value)
    else:
        floor = float("-inf")
        for tp in tps:
            try:
                value = round(float(tp), 2)
            except (TypeError, ValueError):
                continue
            if value <= floor:
                value = round(floor + 0.01, 2)
            floor = value
            adjusted.append(value)
    return adjusted


def _normalise_tags(tags: Iterable[str]) -> set[str]:
    normalised = set()
    for tag in tags:
        if not tag:
            continue
        normalised.add(str(tag).strip().upper())
    return normalised


def clamp_targets_to_em(
    entry: float,
    direction: str,
    tps: Sequence[float],
    em_points: float,
    snap_tags: Iterable[str],
    style: str,
) -> List[float]:
    """
    Enforce |TPi - entry| <= em_points for intraday/scalp with conditional extension.

    Allow bounded extension to 1.10×EM ONLY if a strong snap tag exists ('VAH','VAL','POC','Gap fill','Major HTF').
    Never exceed 1.20×EM under any circumstance.
    """

    if not tps:
        return []

    direction = (direction or "").lower()
    style = (style or "").lower()
    entry_val = float(entry)
    em_points = max(float(em_points or 0.0), 0.0)

    tags = _normalise_tags(snap_tags)
    strong = bool(tags & STRONG_SNAP_TAGS)

    limit = em_points
    hard_cap = em_points * 1.20 if em_points else 0.0

    if style in {"intraday", "scalp"} and em_points > 0:
        limit = em_points * (1.10 if strong else 1.00)
        if hard_cap > 0:
            limit = min(limit, hard_cap)

    clamped: List[float] = []
    for tp in tps:
        try:
            value = float(tp)
        except (TypeError, ValueError):
            continue
        distance = abs(value - entry_val)
        if limit and distance > limit:
            offset = limit if direction != "short" else -limit
            value = entry_val + offset
        elif hard_cap and distance > hard_cap:
            offset = hard_cap if direction != "short" else -hard_cap
            value = entry_val + offset
        clamped.append(round(value, 2))

    min_step = 0.1 if style in {"intraday", "scalp"} else 0.05
    adjusted = ensure_monotonic(clamped, direction)
    if direction == "long":
        for idx in range(1, len(adjusted)):
            min_allowed = adjusted[idx - 1] + min_step
            if adjusted[idx] <= min_allowed - 1e-9:
                candidate = min_allowed
                if limit > 0:
                    candidate = min(candidate, entry_val + limit)
                adjusted[idx] = round(candidate, 2)
        if limit > 0:
            cap = entry_val + limit
            adjusted[-1] = min(adjusted[-1], cap)
            for idx in range(len(adjusted) - 2, -1, -1):
                max_allowed = cap - min_step * (len(adjusted) - 1 - idx)
                adjusted[idx] = min(adjusted[idx], max_allowed)
                if adjusted[idx] > adjusted[idx + 1] - min_step:
                    adjusted[idx] = round(adjusted[idx + 1] - min_step, 2)
    elif direction == "short":
        for idx in range(1, len(adjusted)):
            max_allowed = adjusted[idx - 1] - min_step
            if adjusted[idx] >= max_allowed + 1e-9:
                candidate = max_allowed
                if limit > 0:
                    candidate = max(candidate, entry_val - limit)
                adjusted[idx] = round(candidate, 2)
        floor = entry_val - limit if limit > 0 else None
        if floor is not None:
            adjusted[-1] = max(adjusted[-1], floor)
        for idx in range(len(adjusted) - 1):
            required = adjusted[idx] - min_step
            if adjusted[idx + 1] > required:
                candidate = required
                if floor is not None:
                    candidate = max(candidate, floor)
                adjusted[idx + 1] = round(candidate, 2)
    return adjusted


__all__ = ["clamp_targets_to_em", "ensure_monotonic"]
