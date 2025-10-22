"""Entry candidate generation and evaluation."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from typing import List, Mapping, Optional, Sequence
from zoneinfo import ZoneInfo

from .actionability import is_actionable_soon
from .geometry import PlanGeometry, _local_invalidation, _stop_bounds, build_plan_geometry

_NY_TZ = ZoneInfo("America/New_York")


@dataclass(frozen=True)
class EntryAnchor:
    level: float
    tag: str


@dataclass(frozen=True)
class EntryCandidate:
    entry: float
    stop: float
    tag: str
    actionability: float
    actionable_soon: bool
    entry_distance_pct: float
    entry_distance_atr: float
    bars_to_trigger: int


@dataclass
class EntryContext:
    direction: str
    style: str
    last_price: float
    atr: float
    levels: Mapping[str, float]
    timestamp: Optional[datetime]
    mtf_bias: Optional[str] = None
    session_phase: Optional[str] = None
    preferred_entries: Optional[Sequence[EntryAnchor]] = None
    tick: float = 0.0


def _session_phase_from_timestamp(ts: Optional[datetime]) -> Optional[str]:
    if ts is None:
        return None
    if ts.tzinfo is None:
        local = ts.replace(tzinfo=ZoneInfo("UTC")).astimezone(_NY_TZ)
    else:
        local = ts.astimezone(_NY_TZ)
    hour = local.hour
    minute = local.minute
    if hour == 15:
        return "power_hour"
    if hour < 10 or (hour == 10 and minute == 0):
        return "open"
    if 10 <= hour < 12:
        return "morning"
    if 12 <= hour < 14:
        return "midday"
    if 14 <= hour < 15:
        return "afternoon"
    return "other"


def _anchors_from_levels(direction: str, levels: Mapping[str, float]) -> List[EntryAnchor]:
    ordered: List[str]
    if direction == "long":
        ordered = [
            "opening_range_low",
            "session_low",
            "prev_low",
            "premarket_low",
            "vwap",
            "swing_low",
            "ema9",
            "ema20",
            "ema50",
        ]
    else:
        ordered = [
            "opening_range_high",
            "session_high",
            "prev_high",
            "premarket_high",
            "vwap",
            "swing_high",
            "ema9",
            "ema20",
            "ema50",
        ]
    anchors: List[EntryAnchor] = []
    for key in ordered:
        value = levels.get(key) or levels.get(key.upper())
        if value is None:
            continue
        try:
            price = float(value)
        except (TypeError, ValueError):
            continue
        anchors.append(EntryAnchor(level=price, tag=key))
    return anchors


def _actionability_score(entry: float, ctx: EntryContext) -> float:
    atr = ctx.atr if ctx.atr and ctx.atr > 0 else 1.0
    distance_atr = abs(entry - ctx.last_price) / atr
    distance_term = max(0.0, min(1.0, 1.0 - min(distance_atr, 0.3) / 0.3))
    session_phase = ctx.session_phase or _session_phase_from_timestamp(ctx.timestamp)
    session_boost = 1.0
    if session_phase == "power_hour":
        session_boost = 1.08
    elif session_phase == "open":
        session_boost = 1.02
    mtf_boost = 1.0
    if ctx.mtf_bias:
        bias_norm = ctx.mtf_bias.strip().lower()
        direction_norm = ctx.direction.strip().lower()
        if bias_norm == direction_norm:
            mtf_boost = 1.08
        elif bias_norm not in {"", "neutral"} and bias_norm != direction_norm:
            mtf_boost = 0.9
    score = distance_term * session_boost * mtf_boost
    return max(0.0, min(1.0, score))


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


def build_entry_candidates(ctx: EntryContext) -> List[EntryCandidate]:
    anchors: List[EntryAnchor] = []
    if ctx.preferred_entries:
        for anchor in ctx.preferred_entries:
            if isinstance(anchor, EntryAnchor):
                anchors.append(anchor)
            else:
                try:
                    level, tag = anchor  # type: ignore[misc]
                except Exception:  # pragma: no cover - defensive
                    continue
                anchors.append(EntryAnchor(level=float(level), tag=str(tag)))
    anchors.append(EntryAnchor(level=ctx.last_price, tag="reference"))
    anchors.extend(_anchors_from_levels(ctx.direction, ctx.levels))
    seen: set[float] = set()
    tick_size = ctx.tick if isinstance(ctx.tick, (int, float)) and ctx.tick > 0 else _infer_tick_size(ctx.last_price)
    min_stop_ratio, max_stop_ratio = _stop_bounds(ctx.style)
    candidates: List[EntryCandidate] = []
    for anchor in anchors:
        level = float(anchor.level)
        if level in seen:
            continue
        seen.add(level)
        stop_price, _ = _local_invalidation(ctx.direction, level, ctx.atr, ctx.levels)
        ratio = abs(level - stop_price) / ctx.atr if ctx.atr > 0 else float("inf")
        is_structural = anchor.tag.lower() == "structural"
        if not is_structural and (ratio < min_stop_ratio - 1e-6 or ratio > max_stop_ratio + 1e-6):
            continue
        distance_pct = abs(level - ctx.last_price) / ctx.last_price if ctx.last_price else float("inf")
        distance_atr = abs(level - ctx.last_price) / ctx.atr if ctx.atr > 0 else float("inf")
        bars_to_trigger = max(int(round(distance_atr * 2.0)), 0)
        actionable_soon = is_actionable_soon(level, ctx.last_price, ctx.atr, tick_size, ctx.style)
        actionability = _actionability_score(level, ctx)
        if is_structural:
            actionability = max(actionability, 0.99)
        candidates.append(
            EntryCandidate(
                entry=round(level, 4),
                stop=round(stop_price, 4),
                tag=anchor.tag,
                actionability=actionability,
                actionable_soon=actionable_soon,
                entry_distance_pct=distance_pct,
                entry_distance_atr=distance_atr,
                bars_to_trigger=bars_to_trigger,
            )
        )
    candidates.sort(
        key=lambda item: (
            -item.actionability,
            item.bars_to_trigger,
            item.entry_distance_atr,
            item.entry_distance_pct,
        )
    )
    return candidates


def _readiness_score(plan: PlanGeometry, actionability: float) -> float:
    if not plan.targets:
        return 0.0
    first = plan.targets[0]
    probability = float(getattr(first, "prob_touch", 0.0) or 0.0)
    rr_multiple = float(getattr(first, "rr_multiple", 0.0) or 0.0)
    return max(0.0, probability) * max(0.0, rr_multiple) * max(0.0, min(1.0, actionability))


def select_best_entry_plan(
    ctx: EntryContext,
    plan_kwargs: Mapping[str, object],
    *,
    builder=build_plan_geometry,
    min_actionability: float | None = None,
) -> Tuple[PlanGeometry, EntryCandidate]:
    candidates_all = build_entry_candidates(ctx)
    if not candidates_all:
        fallback_payload = dict(plan_kwargs)
        fallback_payload["entry"] = ctx.last_price
        fallback_payload.pop("_builder", None)
        best_plan = builder(**fallback_payload)
        distance_pct = abs(best_plan.entry - ctx.last_price) / ctx.last_price if ctx.last_price else float("inf")
        distance_atr = abs(best_plan.entry - ctx.last_price) / ctx.atr if ctx.atr > 0 else float("inf")
        bars_to_trigger = max(int(round(distance_atr * 2.0)), 0)
        tick_size = ctx.tick if isinstance(ctx.tick, (int, float)) and ctx.tick > 0 else _infer_tick_size(ctx.last_price)
        actionable_soon = is_actionable_soon(best_plan.entry, ctx.last_price, ctx.atr, tick_size, ctx.style)
        candidate = EntryCandidate(
            entry=round(best_plan.entry, 4),
            stop=round(best_plan.stop.price, 4),
            tag="reference",
            actionability=_actionability_score(best_plan.entry, ctx),
            actionable_soon=actionable_soon,
            entry_distance_pct=distance_pct,
            entry_distance_atr=distance_atr,
            bars_to_trigger=bars_to_trigger,
        )
        return best_plan, candidate

    threshold = None
    if isinstance(min_actionability, (int, float)):
        threshold = float(min_actionability)
    filtered_candidates = (
        [candidate for candidate in candidates_all if threshold is None or candidate.actionability >= threshold]
        if candidates_all
        else []
    )

    sorted_candidates = sorted(
        filtered_candidates or candidates_all,
        key=lambda item: (
            -item.actionability,
            item.bars_to_trigger,
            item.entry_distance_atr,
            item.entry_distance_pct,
        ),
    )
    preferred = [candidate for candidate in sorted_candidates if candidate.actionable_soon] or sorted_candidates
    for candidate in preferred:
        payload = dict(plan_kwargs)
        payload["entry"] = candidate.entry
        payload.pop("_builder", None)
        try:
            plan = builder(**payload)
        except ValueError:
            continue
        candidate_with_stop = replace(candidate, stop=round(float(plan.stop.price), 4))
        return plan, candidate_with_stop

    # Fall back to reference candidate if none built successfully.
    fallback_payload = dict(plan_kwargs)
    fallback_payload["entry"] = ctx.last_price
    fallback_payload.pop("_builder", None)
    best_plan = builder(**fallback_payload)
    distance_pct = abs(best_plan.entry - ctx.last_price) / ctx.last_price if ctx.last_price else float("inf")
    distance_atr = abs(best_plan.entry - ctx.last_price) / ctx.atr if ctx.atr > 0 else float("inf")
    bars_to_trigger = max(int(round(distance_atr * 2.0)), 0)
    tick_size = ctx.tick if isinstance(ctx.tick, (int, float)) and ctx.tick > 0 else _infer_tick_size(ctx.last_price)
    actionable_soon = is_actionable_soon(best_plan.entry, ctx.last_price, ctx.atr, tick_size, ctx.style)
    fallback_candidate = EntryCandidate(
        entry=round(best_plan.entry, 4),
        stop=round(best_plan.stop.price, 4),
        tag="reference",
        actionability=_actionability_score(best_plan.entry, ctx),
        actionable_soon=actionable_soon,
        entry_distance_pct=distance_pct,
        entry_distance_atr=distance_atr,
        bars_to_trigger=bars_to_trigger,
    )
    return best_plan, fallback_candidate


__all__ = [
    "EntryAnchor",
    "EntryCandidate",
    "EntryContext",
    "build_entry_candidates",
    "select_best_entry_plan",
]
