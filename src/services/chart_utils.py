from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, Optional, Set
from zoneinfo import ZoneInfo


ALLOWED_CHART_PARAM_KEYS: Set[str] = {
    "symbol",
    "interval",
    "direction",
    "entry",
    "stop",
    "tp",
    "title",
    "plan_id",
    "plan_version",
    "strategy",
    "range",
    "focus",
    "center_time",
    "scale_plan",
    "notes",
    "live",
    "last_update",
    "data_source",
    "data_mode",
    "data_age_ms",
    "runner",
    "tp_meta",
    "view",
    "levels",
    "ema",
    "session",
    "supportingLevels",
    "ui_state",
}

REQUIRED_CHART_PARAM_KEYS: Set[str] = {
    "symbol",
    "interval",
    "direction",
    "entry",
    "stop",
    "tp",
}

_STRINGY_EMPTY = {"", " ", "\n", "\t"}

_ET = ZoneInfo("America/New_York")
_SUPPORTED_STYLES = ("scalp", "intraday", "swing")


def infer_session_label(as_of: datetime | None) -> str:
    """Return session token ('premkt'|'live'|'after') inferred from an as_of timestamp."""
    if as_of is None:
        return "live"
    if as_of.tzinfo is None:
        aware = as_of.replace(tzinfo=_ET)
    else:
        aware = as_of.astimezone(_ET)
    total_minutes = aware.hour * 60 + aware.minute
    if 4 * 60 <= total_minutes < 9 * 60 + 30:
        return "premkt"
    if 9 * 60 + 30 <= total_minutes < 16 * 60:
        return "live"
    return "after"


def normalize_style_token(style: Any) -> str:
    """Map arbitrary style descriptors into canonical tokens."""
    if isinstance(style, str):
        token = style.strip().lower()
    else:
        token = ""
    if not token:
        return "intraday"
    for candidate in _SUPPORTED_STYLES:
        if token == candidate or candidate in token:
            return candidate
    return "intraday"


def normalize_confidence(value: Any) -> float:
    """Clamp confidence into [0,1], accommodating percentage inputs."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not 0.0 <= numeric <= 1.0 and 1.0 < numeric <= 100.0:
        numeric /= 100.0
    if numeric < 0.0:
        return 0.0
    if numeric > 1.0:
        return 1.0
    return numeric


def build_ui_state(*, session: str, confidence: float, style: str) -> str:
    """Return a JSON payload describing the chart UI state."""
    payload = {
        "session": session or "live",
        "confidence": round(confidence, 3),
        "style": style or "intraday",
    }
    return json.dumps(payload, separators=(",", ":"), sort_keys=True)


def sanitize_chart_params(raw: Dict[str, Any] | None) -> Optional[Dict[str, Any]]:
    """Return a payload that the /gpt/chart-url endpoint will accept.

    The FastAPI route enforces ``extra='forbid'`` on the request model, so we must
    drop any unknown keys and coerce values into the shapes the model expects.
    """

    if not isinstance(raw, dict):
        return None

    sanitized: Dict[str, Any] = {}

    for key in ALLOWED_CHART_PARAM_KEYS:
        if key not in raw:
            continue

        value = raw[key]
        if value is None:
            continue
        if isinstance(value, str) and value.strip() == "":
            continue
        if isinstance(value, str) and value in _STRINGY_EMPTY:
            continue

        if key in {"entry", "stop"}:
            try:
                sanitized[key] = float(value)
            except (TypeError, ValueError):
                sanitized[key] = str(value)
            continue

        if key == "tp":
            if isinstance(value, (list, tuple)):
                joined = ",".join(str(item).strip() for item in value if str(item).strip())
                if joined:
                    sanitized[key] = joined
            else:
                text = str(value).strip()
                if text:
                    sanitized[key] = text
            continue

        if key == "ema":
            if isinstance(value, (list, tuple)):
                joined = ",".join(str(item).strip() for item in value if str(item).strip())
                if joined:
                    sanitized[key] = joined
            else:
                text = str(value).strip()
                if text:
                    sanitized[key] = text
            continue

        if key == "data_age_ms":
            if isinstance(value, (int, float)):
                sanitized[key] = int(value)
            else:
                sanitized[key] = str(value)
            continue

        sanitized[key] = str(value)

    if REQUIRED_CHART_PARAM_KEYS - sanitized.keys():
        return None

    return sanitized


__all__ = [
    "sanitize_chart_params",
    "ALLOWED_CHART_PARAM_KEYS",
    "REQUIRED_CHART_PARAM_KEYS",
    "infer_session_label",
    "normalize_style_token",
    "normalize_confidence",
    "build_ui_state",
]
