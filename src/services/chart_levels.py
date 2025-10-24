from __future__ import annotations

import math
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

MAX_LEVEL_TOKENS = 40

_LEVEL_PRIORITY: Sequence[tuple[str, str]] = (
    ("ORH", "ORH"),
    ("ORL", "ORL"),
    ("SessionH", "SessionH"),
    ("SessionL", "SessionL"),
    ("PDH", "PDH"),
    ("PDL", "PDL"),
    ("PDC", "PDC"),
    ("GapFill", "GapFill"),
    ("VAH", "VAH"),
    ("POC", "POC"),
    ("VAL", "VAL"),
    ("Pivot", "Pivot"),
    ("R1", "R1"),
    ("S1", "S1"),
    ("Fib38", "Fib38"),
    ("Fib50", "Fib50"),
    ("Fib61", "Fib61"),
    ("H1H", "H1H"),
    ("H1L", "H1L"),
    ("H4H", "H4H"),
    ("H4L", "H4L"),
    ("DH", "DH"),
    ("DL", "DL"),
    ("WH", "WH"),
    ("WL", "WL"),
)


def _normalize_key(token: str) -> str:
    return "".join(ch for ch in token.lower() if ch.isalnum())


_ALIAS_TO_CANONICAL: Mapping[str, str] = {
    # Opening range
    "orh": "ORH",
    "openingrangehigh": "ORH",
    "openingrangeh": "ORH",
    "openinghigh": "ORH",
    "orl": "ORL",
    "openingrangelow": "ORL",
    "openinglow": "ORL",
    # Session
    "sessionhigh": "SessionH",
    "sessionh": "SessionH",
    "session_low": "SessionL",
    "sessionlow": "SessionL",
    "sessionl": "SessionL",
    # Previous day
    "prevhigh": "PDH",
    "previoushigh": "PDH",
    "previousdayhigh": "PDH",
    "priordayhigh": "PDH",
    "pdh": "PDH",
    "prevlow": "PDL",
    "previouslow": "PDL",
    "previousdaylow": "PDL",
    "priordaylow": "PDL",
    "pdl": "PDL",
    "prevclose": "PDC",
    "previousclose": "PDC",
    "priordayclose": "PDC",
    "pdc": "PDC",
    # Gap
    "gapfill": "GapFill",
    "gap": "GapFill",
    # Volume profile
    "vah": "VAH",
    "valueareahigh": "VAH",
    "valuehigh": "VAH",
    "poc": "POC",
    "pointofcontrol": "POC",
    "val": "VAL",
    "valuearealow": "VAL",
    "valuelow": "VAL",
    # Pivots
    "pivot": "Pivot",
    "dailypivot": "Pivot",
    "r1": "R1",
    "resistance1": "R1",
    "s1": "S1",
    "support1": "S1",
    # Fibonacci
    "fib38": "Fib38",
    "fib382": "Fib38",
    "fib0.382": "Fib38",
    "fib50": "Fib50",
    "fib0.5": "Fib50",
    "fib61": "Fib61",
    "fib618": "Fib61",
    "fib0.618": "Fib61",
    "fib62": "Fib61",
    # Higher timeframe
    "h1high": "H1H",
    "h1h": "H1H",
    "intradayhigh": "H1H",
    "h1low": "H1L",
    "h1l": "H1L",
    "intradaylow": "H1L",
    "h4high": "H4H",
    "h4h": "H4H",
    "h4low": "H4L",
    "h4l": "H4L",
    "dailyhigh": "DH",
    "dayhigh": "DH",
    "dh": "DH",
    "dhigh": "DH",
    "dailylow": "DL",
    "daylow": "DL",
    "dl": "DL",
    "dlow": "DL",
    "weeklyhigh": "WH",
    "wh": "WH",
    "w1high": "WH",
    "weeklylow": "WL",
    "wl": "WL",
    "w1low": "WL",
}


def _coerce_price(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        for key in ("price", "level", "value", "close"):
            if key in value:
                maybe = _coerce_price(value[key])
                if maybe is not None:
                    return maybe
        return None
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        # Take the first numeric from the sequence
        for item in value:
            maybe = _coerce_price(item)
            if maybe is not None:
                return maybe
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def _record_level(levels: MutableMapping[str, float], label: str, value: Any) -> None:
    canonical = _ALIAS_TO_CANONICAL.get(_normalize_key(label))
    if canonical is None:
        return
    if canonical in levels:
        return
    price = _coerce_price(value)
    if price is None:
        return
    levels[canonical] = price


def _collect_levels_from_mapping(levels: MutableMapping[str, float], mapping: Mapping[str, Any]) -> None:
    for key, value in mapping.items():
        _record_level(levels, key, value)
        _collect_levels(levels, value, treat_mapping=True)


def _collect_levels(levels: MutableMapping[str, float], container: Any, *, treat_mapping: bool = False) -> None:
    if isinstance(container, Mapping):
        if treat_mapping:
            _collect_levels_from_mapping(levels, container)
        for key in ("key_levels", "relevant_levels", "levels"):
            nested = container.get(key)
            if isinstance(nested, Mapping):
                _collect_levels_from_mapping(levels, nested)
        key_used = container.get("key_levels_used")
        if isinstance(key_used, Mapping):
            for value in key_used.values():
                if isinstance(value, Mapping):
                    _collect_levels(levels, value, treat_mapping=True)
                else:
                    _collect_levels(levels, value)
        plan_layers = container.get("plan_layers")
        if isinstance(plan_layers, Mapping):
            _collect_levels(levels, plan_layers)
        structured = container.get("structured_plan")
        if isinstance(structured, Mapping):
            _collect_levels(levels, structured)
        nested_plan = container.get("plan")
        if isinstance(nested_plan, Mapping):
            _collect_levels(levels, nested_plan)
    elif isinstance(container, Sequence) and not isinstance(container, (str, bytes, bytearray)):
        for item in container:
            if isinstance(item, Mapping):
                label = item.get("label") or item.get("name") or item.get("role")
                if label:
                    _record_level(levels, str(label), item)
            else:
                _collect_levels(levels, item)


def _infer_decimals(plan: Mapping[str, Any], level_values: Iterable[float]) -> int:
    explicit = plan.get("decimals")
    if isinstance(explicit, int) and 0 <= explicit <= 6:
        return explicit

    numbers: list[float] = []
    for key in ("entry", "stop"):
        value = plan.get(key)
        if isinstance(value, (int, float)) and math.isfinite(value):
            numbers.append(float(value))
    targets = plan.get("targets")
    if isinstance(targets, Sequence):
        for item in targets:
            if isinstance(item, (int, float)) and math.isfinite(item):
                numbers.append(float(item))
    for value in level_values:
        numbers.append(value)

    if not numbers:
        return 2

    def _count_decimals(number: float) -> int:
        text = f"{number:.8f}".rstrip("0").rstrip(".")
        if "." not in text:
            return 0
        return len(text.split(".")[1])

    return max(0, min(max(_count_decimals(num) for num in numbers), 6))


def _format_price(value: float, decimals: int) -> str:
    quantizer = Decimal("1") if decimals == 0 else Decimal("1").scaleb(-decimals)
    try:
        rounded = Decimal(str(value)).quantize(quantizer, rounding=ROUND_HALF_UP)
    except (InvalidOperation, ValueError):
        rounded = Decimal(0)
    return f"{rounded:.{decimals}f}"


def _build_tokens(plan: Mapping[str, Any], level_map: Mapping[str, float]) -> list[str]:
    if not level_map:
        return []
    decimals = _infer_decimals(plan, level_map.values())
    seen: set[tuple[str, str]] = set()
    tokens: list[str] = []

    for canonical, output_label in _LEVEL_PRIORITY:
        if canonical not in level_map:
            continue
        price = level_map[canonical]
        formatted = _format_price(price, decimals)
        key = (formatted, output_label)
        if key in seen:
            continue
        tokens.append(f"{formatted}|{output_label}")
        seen.add(key)
        if len(tokens) >= MAX_LEVEL_TOKENS:
            break

    if len(tokens) >= MAX_LEVEL_TOKENS:
        return tokens

    for canonical, price in level_map.items():
        output_label = next((label for key, label in _LEVEL_PRIORITY if key == canonical), None)
        if output_label is None:
            continue
        formatted = _format_price(price, decimals)
        key = (formatted, output_label)
        if key in seen:
            continue
        tokens.append(f"{formatted}|{output_label}")
        seen.add(key)
        if len(tokens) >= MAX_LEVEL_TOKENS:
            break

    return tokens


def extract_supporting_levels(plan: Mapping[str, Any], *extras: Mapping[str, Any]) -> str | None:
    """
    Return a semicolon-delimited string of ``price|label`` tokens for supporting levels.
    """

    level_map: dict[str, float] = {}
    _collect_levels(level_map, plan)
    for extra in extras:
        if isinstance(extra, Mapping):
            _collect_levels(level_map, extra)

    tokens = _build_tokens(plan, level_map)
    if not tokens:
        return None
    return ";".join(tokens)


__all__ = ["extract_supporting_levels"]
