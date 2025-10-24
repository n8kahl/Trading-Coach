from __future__ import annotations

from typing import Any, Dict, Optional, Set


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


__all__ = ["sanitize_chart_params", "ALLOWED_CHART_PARAM_KEYS", "REQUIRED_CHART_PARAM_KEYS"]
