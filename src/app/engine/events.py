"""Event gating utilities for Fancy Trader."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

_SEVERITY_ORDER = {"low": 1, "medium": 2, "high": 3, "critical": 4}


@dataclass(slots=True)
class EventDecision:
    action: str
    triggered: List[Dict[str, object]]
    reason: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "action": self.action,
            "triggered": self.triggered,
            "reason": self.reason,
        }


def _normalize_severity(value: object) -> int:
    token = str(value or "low").lower()
    return _SEVERITY_ORDER.get(token, 1)


def apply_event_gating(
    events: Iterable[Dict[str, object]] | None,
    *,
    minutes_threshold: int = 90,
    severity_threshold: str = "medium",
    mode: str = "defined_risk",
) -> EventDecision:
    """Determine whether a plan should be adjusted due to upcoming events."""

    if not events:
        return EventDecision(action="allow", triggered=[])

    severity_cutoff = _normalize_severity(severity_threshold)
    triggered: List[Dict[str, object]] = []
    for raw in events:
        if not isinstance(raw, dict):
            continue
        minutes = raw.get("minutes_to_event")
        try:
            minutes_val = float(minutes)
        except (TypeError, ValueError):
            continue
        if minutes_val < 0:
            continue
        severity_rank = _normalize_severity(raw.get("severity"))
        if severity_rank >= severity_cutoff and minutes_val <= minutes_threshold:
            triggered.append(raw)

    if not triggered:
        return EventDecision(action="allow", triggered=[])

    mode_token = mode.lower()
    if mode_token not in {"defined_risk", "suppress"}:
        mode_token = "defined_risk"

    reason = ", ".join(
        f"{event.get('label') or event.get('type') or 'event'} @ {event.get('minutes_to_event')}m"
        for event in triggered
    )

    if mode_token == "suppress":
        return EventDecision(action="suppress", triggered=triggered, reason=reason)
    return EventDecision(action="defined_risk", triggered=triggered, reason=reason)


__all__ = ["apply_event_gating", "EventDecision"]
