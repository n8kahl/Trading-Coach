from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


class PlanStatus(str, Enum):
    INTACT = "intact"
    AT_RISK = "at_risk"
    INVALIDATED = "invalidated"
    REVERSAL = "reversal"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _coerce_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


@dataclass
class PlanMonitor:
    plan_id: str
    symbol: str
    direction: str
    entry: float
    stop: float
    targets: List[float] = field(default_factory=list)
    version: int = 1
    min_rr: float = 1.2
    confidence: Optional[float] = None
    runner: Optional[Dict[str, Any]] = None
    buffer_pct: float = 0.003

    status: PlanStatus = PlanStatus.INTACT
    last_rr: Optional[float] = None
    last_breach: Optional[str] = None
    last_note: Optional[str] = None
    last_event_ts: Optional[str] = None
    has_initial_emit: bool = False
    last_price: Optional[float] = None

    def update_snapshot(self, plan: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        symbol = plan.get("symbol") or self.symbol
        direction = (plan.get("bias") or plan.get("direction") or self.direction or "long").lower()
        entry = _coerce_float(plan.get("entry")) or self.entry
        stop = _coerce_float(plan.get("stop")) or self.stop
        targets_raw = plan.get("targets") or []
        targets = [float(t) for t in targets_raw if _coerce_float(t) is not None]
        version = plan.get("version") or self.version
        confidence = _coerce_float(plan.get("confidence"))
        runner = plan.get("runner") or None
        rr = _coerce_float(plan.get("rr_to_t1"))
        if rr is not None and rr > 0:
            self.min_rr = max(0.8, float(rr) * 0.6)

        baseline_changed = (
            not math.isclose(self.entry, entry, rel_tol=1e-6)
            or not math.isclose(self.stop, stop, rel_tol=1e-6)
            or (targets and targets[0] != (self.targets[0] if self.targets else None))
            or direction != self.direction
        )

        self.symbol = symbol
        self.direction = direction
        self.entry = entry
        self.stop = stop
        self.targets = targets
        self.version = int(version) if isinstance(version, (int, float)) else self.version
        self.confidence = confidence
        self.runner = runner
        self.status = PlanStatus.INTACT
        self.last_rr = rr
        self.last_breach = None
        self.last_note = None
        self.last_event_ts = None
        self.has_initial_emit = False
        self.last_price = None

        if baseline_changed:
            event = self._build_state_event(reason="plan_updated")
        else:
            event = self._build_state_event(reason="plan_registered")
        self.last_note = event["changes"].get("note")
        self.last_event_ts = event["changes"].get("timestamp")
        return event

    def _risk_reward(self, price: float) -> Optional[float]:
        if not self.targets:
            return None
        tp = self.targets[0]
        if self.direction == "long":
            risk = price - self.stop
            reward = tp - price
        else:
            risk = self.stop - price
            reward = price - tp
        if risk <= 0 or reward <= 0:
            return None
        return round(float(reward / risk), 2)

    def _status_for_price(self, price: float) -> tuple[PlanStatus, Optional[str]]:
        buffer = self.stop * (1 + self.buffer_pct) if self.direction == "long" else self.stop * (1 - self.buffer_pct)
        if self.direction == "long":
            if price <= self.stop:
                return PlanStatus.INVALIDATED, "stop_hit"
            if price <= buffer:
                return PlanStatus.AT_RISK, "stop_buffer_touched"
        else:
            if price >= self.stop:
                return PlanStatus.INVALIDATED, "stop_hit"
            if price >= buffer:
                return PlanStatus.AT_RISK, "stop_buffer_touched"
        return PlanStatus.INTACT, None

    def _note_for_status(self, status: PlanStatus, price: float, rr: Optional[float], breach: Optional[str]) -> str:
        price_copy = f"{price:.2f}"
        if status == PlanStatus.INVALIDATED:
            return f"Stop level triggered at {price_copy}"
        if status == PlanStatus.AT_RISK:
            if breach == "rr_deterioration" and rr is not None:
                return f"Reward-to-risk fell to {rr:.2f}"
            return f"Price is within stop buffer ({price_copy})"
        if status == PlanStatus.REVERSAL:
            return "Reversal criteria met"
        return "Plan intact. Risk profile unchanged."

    def _next_step_for_status(self, status: PlanStatus) -> str:
        if status == PlanStatus.INVALIDATED:
            return "plan_invalidated"
        if status == PlanStatus.AT_RISK:
            return "tighten_stop"
        if status == PlanStatus.REVERSAL:
            return "consider_reversal"
        return "hold_plan"

    def _build_state_event(self, *, price: Optional[float] = None, reason: str) -> Dict[str, Any]:
        rr = self.last_rr
        note = self.last_note
        if not note:
            reference_price = price if price is not None else self.entry
            note = self._note_for_status(self.status, reference_price, self.last_rr, self.last_breach)
        changes: Dict[str, Any] = {
            "status": self.status.value,
            "next_step": self._next_step_for_status(self.status),
            "note": note,
            "rr_to_t1": rr,
            "breach": self.last_breach,
            "timestamp": self.last_event_ts or _utc_now(),
        }
        if price is not None:
            changes["last_price"] = price
        payload = {
            "t": "plan_delta",
            "plan_id": self.plan_id,
            "version": self.version,
            "changes": changes,
            "reason": reason,
        }
        return payload

    def handle_price(self, price: float, *, partial: bool = False) -> Optional[Dict[str, Any]]:
        if price is None or not math.isfinite(price):
            return None

        self.last_price = price
        previous_rr = self.last_rr
        rr = self._risk_reward(price)
        status, breach = self._status_for_price(price)

        if status == PlanStatus.INTACT and rr is not None and rr < self.min_rr:
            status = PlanStatus.AT_RISK
            breach = "rr_deterioration"

        note = self._note_for_status(status, price, rr, breach)
        timestamp = _utc_now()

        should_emit = False
        if not self.has_initial_emit:
            should_emit = True
            self.has_initial_emit = True
        elif status != self.status:
            should_emit = True
        elif breach != self.last_breach:
            should_emit = True
        elif rr is not None and (previous_rr is None or abs(rr - previous_rr) >= 0.05):
            should_emit = True

        self.status = status
        self.last_breach = breach
        self.last_note = note
        self.last_event_ts = timestamp
        self.last_rr = rr

        if not should_emit:
            return None

        return self._build_state_event(price=price, reason="partial_bar" if partial else "price_update")


class LivePlanEngine:
    """Tracks plan health across incoming market events."""

    def __init__(self) -> None:
        self._monitors: Dict[str, Dict[str, PlanMonitor]] = {}
        self._lock = asyncio.Lock()

    async def register_snapshot(self, snapshot: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        plan = snapshot.get("plan") or {}
        plan_id = str(plan.get("plan_id") or "").strip()
        symbol = str(plan.get("symbol") or "").strip()
        if not plan_id or not symbol:
            return None
        symbol_key = symbol.upper()
        async with self._lock:
            symbol_plans = self._monitors.setdefault(symbol_key, {})
            monitor = symbol_plans.get(plan_id)
            if monitor is None:
                direction = (plan.get("bias") or plan.get("direction") or "long").lower()
                entry = _coerce_float(plan.get("entry")) or 0.0
                stop = _coerce_float(plan.get("stop")) or 0.0
                targets_raw = plan.get("targets") or []
                targets = [float(t) for t in targets_raw if _coerce_float(t) is not None]
                monitor = PlanMonitor(
                    plan_id=plan_id,
                    symbol=symbol_key,
                    direction=direction,
                    entry=entry,
                    stop=stop,
                    targets=targets,
                    version=int(plan.get("version") or 1),
                    min_rr=_coerce_float(plan.get("rr_to_t1")) or 1.2,
                    confidence=_coerce_float(plan.get("confidence")),
                    runner=plan.get("runner"),
                )
                symbol_plans[plan_id] = monitor
                return monitor.update_snapshot(plan)
            return monitor.update_snapshot(plan)

    async def handle_market_event(self, symbol: str, event: Dict[str, Any]) -> List[Dict[str, Any]]:
        symbol_key = symbol.upper()
        async with self._lock:
            monitors = list((self._monitors.get(symbol_key) or {}).values())
        if not monitors:
            return []
        event_type = event.get("t")
        emitted: List[Dict[str, Any]] = []
        if event_type in {"tick", "bar"}:
            price = _coerce_float(event.get("p") or event.get("close"))
            if price is None:
                return []
            partial = bool(event.get("partial"))
            for monitor in monitors:
                payload = monitor.handle_price(price, partial=partial)
                if payload:
                    emitted.append(payload)
        elif event_type == "plan_full":
            payload = event.get("payload")
            if isinstance(payload, dict):
                update_event = await self.register_snapshot({"plan": payload.get("plan") or payload})
                if update_event:
                    emitted.append(update_event)
        return emitted

    async def active_plan_states(self, symbol: str) -> List[Dict[str, Any]]:
        symbol_key = symbol.upper()
        async with self._lock:
            monitors = list((self._monitors.get(symbol_key) or {}).values())
        states: List[Dict[str, Any]] = []
        for monitor in monitors:
            states.append(
                {
                    "plan_id": monitor.plan_id,
                    "version": monitor.version,
                    "status": monitor.status.value,
                    "rr_to_t1": monitor.last_rr,
                    "next_step": monitor._next_step_for_status(monitor.status),
                    "note": monitor.last_note,
                    "timestamp": monitor.last_event_ts,
                }
            )
        return states
