"""Canonical chart URL builder used by /gpt/chart-url."""

from __future__ import annotations

from urllib.parse import urlencode, urlsplit, urlunsplit
from typing import Dict, Iterable

from .precision import get_price_precision

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
    }
)


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
    """Return canonical /tv URL composed from allow-listed params."""

    symbol = str(params.get("symbol") or "").upper()
    precision = get_price_precision(symbol, precision_map=precision_map)

    payload: Dict[str, str] = {}
    if symbol:
        payload["symbol"] = symbol

    for key, raw_value in params.items():
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
                payload[key] = _coerce_list(raw_value, lambda x: _format_number(x, precision))
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
    path = parsed.path if parsed.netloc else "/tv"
    if not path or path == "/":
        path = "/tv"

    return urlunsplit((scheme, netloc, path, query, ""))


__all__ = ["make_chart_url", "ALLOWED_KEYS"]
