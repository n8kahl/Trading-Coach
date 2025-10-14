"""Context aggregation and scoring helpers."""

from __future__ import annotations

from typing import Any, Dict

from ..providers import macro, sector, internals


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _normalize_score(value: float | None) -> float:
    if value is None:
        return 0.5
    if not isinstance(value, (int, float)):
        return 0.5
    return _clamp(float(value))


def _score(ev: Dict[str, Any], sec: Dict[str, Any], peer: Dict[str, Any], ints: Dict[str, Any]) -> float:
    components = []
    minutes = ev.get("min_minutes_to_event")
    if isinstance(minutes, int):
        components.append(_clamp(1.0 - min(max(minutes, 0), 240) / 240.0))
    rel = sec.get("rel_vs_spy")
    if isinstance(rel, (int, float)):
        components.append(_clamp(0.5 + (float(rel) - 0.5) * 0.8))
    peer_vs = peer.get("rs_vs_benchmark")
    if isinstance(peer_vs, (int, float)):
        components.append(_clamp(float(peer_vs)))
    breadth = ints.get("breadth")
    if isinstance(breadth, (int, float)):
        components.append(_clamp(0.4 + min(float(breadth), 3000.0) / 6000.0))
    if not components:
        return 0.5
    return _clamp(sum(components) / len(components))


def _macro_summary(ev: Dict[str, Any]) -> str:
    upcoming = ev.get("upcoming") or []
    if not upcoming:
        return "No high-impact events within 4h."
    first = upcoming[0]
    name = first.get("name") or "Event"
    minutes = first.get("minutes")
    if isinstance(minutes, int):
        return f"{name} in {minutes}m"
    return name


def build_context(symbol: str, as_of: str) -> Dict[str, Any]:
    """Aggregate macro, sector, and internal signals into a context block."""

    events = macro.get_event_window(as_of)
    sector_snapshot = sector.sector_strength(symbol, as_of)
    peer_snapshot = sector.peer_rel_strength(symbol, as_of)
    internal_snapshot = internals.market_internals(as_of)
    score = _score(events, sector_snapshot, peer_snapshot, internal_snapshot)
    return {
        "macro": _macro_summary(events),
        "sector": {
            "name": sector_snapshot.get("sector"),
            "rel_vs_spy": sector_snapshot.get("rel_vs_spy"),
            "z": sector_snapshot.get("zscore"),
        },
        "internals": {
            "breadth": internal_snapshot.get("breadth"),
            "vix": internal_snapshot.get("vix"),
            "tick": internal_snapshot.get("tick"),
        },
        "rs": {"vs_benchmark": peer_snapshot.get("rs_vs_benchmark")},
        "context_score": score,
    }


__all__ = ["build_context"]
