from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Tuple

from src.agent_server import build_plan_layers


def stub_plan_components(symbol: str) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Return plan, plan_layers, session_state tuples for stubbed tests."""

    entry = 430.0
    stop = 428.6
    target = 432.4
    now = datetime(2025, 10, 28, 14, 30, tzinfo=timezone.utc)
    as_of_iso = now.isoformat().replace("+00:00", "Z")
    session_state: Dict[str, Any] = {
        "status": "open",
        "as_of": as_of_iso,
        "next_open": None,
        "tz": "America/New_York",
    }
    plan_core: Dict[str, Any] = {
        "plan_id": f"{symbol}-demo",
        "symbol": symbol,
        "direction": "long",
        "entry": entry,
        "stop": stop,
        "targets": [target],
        "meta": {"entry_status": {"state": "waiting"}},
        "waiting_for": "1m close above VWAP",
        "session_state": session_state,
        "context_overlays": {"volume_profile": {"vwap": entry - 0.12}},
        "key_levels": {"PDL": entry - 1.0},
        "price": {"close": entry - 0.25},
    }
    plan_layers = build_plan_layers(
        symbol=symbol,
        interval="5m",
        as_of=as_of_iso,
        planning_context="live",
        key_levels=plan_core["key_levels"],
        overlays=plan_core["context_overlays"],
        strategy_id="orb_retest",
        direction="long",
        waiting_for=plan_core["waiting_for"],
        plan=plan_core,
        last_time_s=int(now.timestamp()),
        interval_s=300,
        last_close=entry - 0.25,
    )
    plan_layers["plan_id"] = plan_core["plan_id"]
    plan_snapshot = dict(plan_core)
    plan_snapshot["plan_layers"] = plan_layers
    return plan_snapshot, plan_layers, session_state


__all__ = ["stub_plan_components"]
