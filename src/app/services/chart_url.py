"""Canonical chart URL builder used by /gpt/chart-url."""

from __future__ import annotations

import json
from urllib.parse import urlencode, urlsplit, urlunsplit
from typing import Any, Dict, Iterable, Mapping

from .instrument_precision import get_precision

ALLOWED_KEYS: frozenset[str] = frozenset(
    {
        "symbol",
        "interval",
        "direction",
        "entry",
        "stop",
        "tp",
        "ema",
        "focus",
        "center_time",
        "scale_plan",
        "view",
        "range",
        "theme",
        "plan_id",
        "plan_version",
        "force_interval",
        "ui_state",
    }
)

STYLE_DEFAULTS: Dict[str, Dict[str, str]] = {
    "scalp": {"interval": "1", "view": "30m", "range": "1d"},
    "intraday": {"interval": "5", "view": "30m", "range": "1d"},
    "swing": {"interval": "60", "view": "3M", "range": "15d"},
    "leaps": {"interval": "1D", "view": "1Y", "range": "6M"},
}

DEFAULT_CHART_PATH = "/chart"


def _extract_style(ui_state: Any) -> str:
    """Return normalized style token from a UI state payload."""
    if isinstance(ui_state, Mapping):
        token = ui_state.get("style")
        return str(token or "").strip().lower()
    if isinstance(ui_state, str):
        try:
            parsed = json.loads(ui_state)
        except Exception:
            return ""
        if isinstance(parsed, Mapping):
            token = parsed.get("style")
            return str(token or "").strip().lower()
    return ""


def _is_intraday(token: str) -> bool:
    """Identify intraday-style intervals (minutes/hours)."""
    normalized = (token or "").strip().lower()
    if not normalized:
        return True
    if normalized.endswith("d") or normalized.endswith("w"):
        return False
    normalized = normalized.replace("m", "").replace("h", "").strip()
    try:
        return int(normalized) < 1440
    except Exception:
        return True


def coerce_by_style(params: Dict[str, Any]) -> Dict[str, Any]:
    params = dict(params or {})
    if not params:
        return params

    style_token = _extract_style(params.get("ui_state"))
    if not style_token:
        for fallback in ("style", "style_hint"):
            raw = params.get(fallback)
            if raw:
                style_token = str(raw).strip().lower()
                if style_token:
                    break
    if not style_token:
        return params

    defaults = STYLE_DEFAULTS.get(style_token)
    if not defaults:
        return params

    interval = str(params.get("interval") or "").strip()
    force = str(params.get("force_interval") or "").strip() == "1"

    should_upgrade = (not interval) or (
        style_token in {"swing", "leaps"} and _is_intraday(interval) and not force
    )

    if should_upgrade:
        params["interval"] = defaults["interval"]
        params.setdefault("view", defaults["view"])
        params.setdefault("range", defaults["range"])
    else:
        params.setdefault("view", defaults["view"])
        params.setdefault("range", defaults["range"])

    return params


def _format_number(value: float | str, decimals: int) -> str:
    try:
        formatted = f"{float(value):.{decimals}f}"
    except (TypeError, ValueError):
        return str(value)
    if "." in formatted:
        formatted = formatted.rstrip("0").rstrip(".")
    return formatted


def _coerce_list(value: Iterable, formatter) -> str:
    return ",".join(formatter(item) for item in value)


def make_chart_url(
    params: Dict[str, object],
    *,
    base_url: str,
    precision_map: Dict[str, int] | None = None,
) -> str:
    """Return canonical /chart URL composed from allow-listed params."""

    params_dict: Dict[str, Any] = coerce_by_style(dict(params))
    params_dict.pop("style_hint", None)

    symbol = str(params_dict.get("symbol") or "").upper()
    precision = get_precision(symbol, precision_map=precision_map)

    payload: Dict[str, str] = {}
    if symbol:
        payload["symbol"] = symbol

    for key, raw_value in params_dict.items():
        if key not in ALLOWED_KEYS:
            continue
        if raw_value in (None, "", [], ()):
            continue
        if key == "symbol":
            continue
        if key in {"entry", "stop"}:
            payload[key] = _format_number(raw_value, precision)
        elif key == "tp":
            if isinstance(raw_value, (list, tuple)):
                payload[key] = _coerce_list(
                    raw_value, lambda x: _format_number(x, precision)
                )
            else:
                payload[key] = str(raw_value)
        elif key == "ema":
            if isinstance(raw_value, (list, tuple)):
                payload[key] = _coerce_list(raw_value, lambda x: str(int(x)))
            else:
                payload[key] = str(raw_value)
        else:
            payload[key] = str(raw_value)

    items = sorted(payload.items(), key=lambda pair: pair[0])
    query = urlencode(items, doseq=False, safe=":,")

    parsed = urlsplit(base_url)
    scheme = parsed.scheme or "https"
    netloc = parsed.netloc or parsed.path
    path = parsed.path if parsed.netloc else DEFAULT_CHART_PATH
    if not path or path == "/":
        path = DEFAULT_CHART_PATH

    return urlunsplit((scheme, netloc, path, query, ""))


__all__ = ["make_chart_url", "ALLOWED_KEYS", "coerce_by_style"]
