"""Canonical strategy catalog for plan normalization."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Mapping, Optional, Sequence


@dataclass(frozen=True)
class StrategyProfile:
    """Normalized strategy description used across planning surfaces."""

    name: str
    trigger: Sequence[str]
    invalidation: str
    management: str
    reload: Optional[str] = None
    runner: Optional[str] = None
    badges: Sequence[str] = ()

    def to_payload(self) -> Dict[str, object]:
        payload = asdict(self)
        payload["trigger"] = list(self.trigger)
        payload["badges"] = list(self.badges)
        return payload


_CATALOG: Dict[str, StrategyProfile] = {
    "baseline_auto": StrategyProfile(
        name="Baseline Geometry",
        trigger=[
            "Price tests the highlighted structural level with confirming volume",
            "Bias follows the session trend (EMA stack & VWAP alignment)",
        ],
        invalidation="Candle close beyond structural stop or VWAP pivot failure.",
        management="Scale at TP1, trail remainder using plan runner defaults.",
        reload="Only after fresh confluence rebuild; do not average into losers.",
        runner="ATR trail per plan runner (publish fraction + step).",
        badges=("Baseline",),
    ),
    "orb_retest": StrategyProfile(
        name="Opening Range Retest",
        trigger=[
            "Breakout from opening range high/low",
            "Retest holds with rising volume and stacked EMAs",
        ],
        invalidation="Close back inside opening range with VWAP failure.",
        management="Take partials at ATR ladder; tighten stop once price clears OR high/low.",
        reload="Allowed on second retest if VWAP & OR boundary hold.",
        runner="Trail with 0.8× ATR below higher lows.",
        badges=("Opening Range", "Power Move"),
    ),
    "power_hour_trend": StrategyProfile(
        name="Power Hour Continuation",
        trigger=[
            "3–4pm ET, price above/below session VWAP",
            "Short EMAs stacked in trade direction with rising ADX",
        ],
        invalidation="Close back through VWAP or ADX rollover below threshold.",
        management="Trim into prior high/low liquidity; trail remainder into the close.",
        reload="Avoid re-entries after 3:45pm ET unless fresh VWAP reclaim.",
        runner="Tighten to 0.6× ATR once TP1 hits.",
        badges=("Power Hour",),
    ),
    "gap_fill_open": StrategyProfile(
        name="Opening Gap Fill",
        trigger=[
            "09:30–10:00 ET gap fails to extend",
            "VWAP cross in direction of the fill with momentum divergence",
        ],
        invalidation="Close back outside prior day range or failure to reclaim VWAP.",
        management="Target prior close, then previous session VWAP.",
        reload="Single attempt only; no reload if first trade fails.",
        runner="Trail with 1.0× ATR behind intraday swings.",
        badges=("Gap Fill",),
    ),
    "vwap_reclaim": StrategyProfile(
        name="VWAP Reclaim/Reject",
        trigger=[
            "VWAP reclaimed/rejected with 2-3 bar confirmation",
            "Confluence with opening range boundary",
        ],
        invalidation="Lose VWAP on closing basis or re-enter opening range.",
        management="Trim into prior structure (OR / PDH / PDL), trail per runner guidance.",
        reload="Single reload permitted on subsequent VWAP test with confirmation.",
        runner="Trail 0.8× ATR while VWAP holds in trade direction.",
        badges=("VWAP", "Structure"),
    ),
    "range_break_retest": StrategyProfile(
        name="Range Break & Retest",
        trigger=[
            "Session or prior day range breaks",
            "Price retests and holds the breakout level",
        ],
        invalidation="Close back inside the broken range.",
        management="Scale at first structural target, trail stop below reclaimed level.",
        reload="Allowed on second retest if volume confirms acceptance.",
        runner="Trail 1.0× ATR beneath reclaimed structure.",
        badges=("Range Break",),
    ),
    "ema_pullback_trend": StrategyProfile(
        name="EMA Pullback Trend",
        trigger=[
            "EMA stack aligned with trade bias",
            "Price pulls back to EMA cluster with rejection wick",
        ],
        invalidation="Close through 20 EMA or breakdown of swing structure.",
        management="Scale at TP1, trail remaining position with adaptive EMA runner.",
        reload="Reload permitted on next EMA tag if ADX remains rising.",
        runner="Trail 0.9× ATR under higher lows (or above lower highs).",
        badges=("Trend",),
    ),
    "midday_mean_revert": StrategyProfile(
        name="Midday VWAP Reversion",
        trigger=[
            "11:30–14:00 ET mean-reversion window",
            "ADX < 15 with price stretched >0.6× ATR from VWAP",
        ],
        invalidation="Close beyond extension bands or VWAP slope flip.",
        management="Use measured move back to VWAP; exit remainder at VWAP touch.",
        reload="Allow one reload if VWAP rejects with low ADX.",
        runner="Trail shallow at 0.5× ATR due to chop.",
        badges=("VWAP", "Mean Reversion"),
    ),
}


def get_strategy_profile(strategy_id: Optional[str], style: Optional[str] = None) -> Dict[str, object]:
    """Return normalized strategy payload for a given identifier."""

    if strategy_id:
        key = strategy_id.strip().lower()
        profile = _CATALOG.get(key)
        if profile:
            return profile.to_payload()

    style_token = (style or "").strip().lower()
    fallback_badges: List[str] = []
    if style_token:
        fallback_badges.append(style_token.title())

    fallback = StrategyProfile(
        name="Discretionary Setup",
        trigger=[
            "Use confluence badges to confirm alignment.",
            "Execute only when price respects published entry geometry.",
        ],
        invalidation="Close through structural stop or loss of trend confluence.",
        management="Scale per plan targets; respect runner policy for remainder.",
        reload="Avoid reloads without fresh structure confirmation.",
        runner="Follow published runner defaults.",
        badges=tuple(fallback_badges),
    )
    return fallback.to_payload()


def compose_strategy_badges(
    strategy_payload: Mapping[str, object],
    *,
    bias: Optional[str],
    style: Optional[str],
    confluence: Iterable[str],
    extra_badges: Iterable[str] | None = None,
    limit: int = 5,
) -> List[Dict[str, str]]:
    """Compose prioritized badges for plan surfaces."""

    ordered: List[Dict[str, str]] = []

    name = str(strategy_payload.get("name") or "").strip()
    if name:
        ordered.append({"label": name, "kind": "strategy"})

    for token in strategy_payload.get("badges", []):
        label = str(token).strip()
        if label:
            ordered.append({"label": label, "kind": "strategy"})

    style_token = (style or "").strip()
    if style_token:
        ordered.append({"label": style_token.title(), "kind": "style"})

    bias_token = (bias or "").strip()
    if bias_token:
        ordered.append({"label": bias_token.title(), "kind": "bias"})

    for item in confluence:
        label = str(item).strip()
        if label:
            ordered.append({"label": label, "kind": "confluence"})

    if extra_badges:
        for item in extra_badges:
            label = str(item).strip()
            if label:
                ordered.append({"label": label, "kind": "meta"})

    deduped: List[Dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for badge in ordered:
        key = (badge["label"].lower(), badge["kind"])
        if key in seen:
            continue
        deduped.append(badge)
        seen.add(key)
        if len(deduped) >= limit:
            break
    return deduped


__all__ = ["StrategyProfile", "get_strategy_profile", "compose_strategy_badges"]
