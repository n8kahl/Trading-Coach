"""Async Tradier client helpers used for option contract pricing."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx
import pandas as pd

from .config import get_settings

logger = logging.getLogger(__name__)


class TradierNotConfiguredError(RuntimeError):
    """Raised when Tradier credentials are missing."""


@dataclass(slots=True)
class TradierConfig:
    token: str
    base_url: str


def _get_tradier_config() -> TradierConfig:
    settings = get_settings()
    token = settings.tradier_token
    if not token:
        raise TradierNotConfiguredError("Tradier API token is not configured.")
    base_url = settings.tradier_base_url.rstrip("/")
    return TradierConfig(token=token, base_url=base_url)


async def _fetch_json(client: httpx.AsyncClient, url: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    response = await client.get(url, params=params)
    response.raise_for_status()
    return response.json()


async def _resolve_expiration(client: httpx.AsyncClient, symbol: str, requested: str | None) -> str:
    if requested:
        return requested
    data = await _fetch_json(
        client,
        "/v1/markets/options/expirations",
        params={"symbol": symbol, "includeAllRoots": "true", "strikes": "false"},
    )
    expirations = data.get("expirations", {})
    dates = expirations.get("date") if isinstance(expirations, dict) else []
    if isinstance(dates, str):
        return dates
    if isinstance(dates, list) and dates:
        return dates[0]
    raise RuntimeError(f"No option expirations returned for {symbol}")


def _chain_to_dataframe(options: List[Dict[str, Any]]) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    for opt in options:
        greeks = opt.get("greeks") or {}
        bid = opt.get("bid")
        ask = opt.get("ask")
        mid = None
        spread_pct = None
        if isinstance(bid, (int, float)) and isinstance(ask, (int, float)) and (bid or ask):
            mid = (bid + ask) / 2 if bid is not None and ask is not None else None
            if mid and mid > 0:
                spread_pct = (ask - bid) / mid
        expiration = opt.get("expiration_date")
        dte = None
        if expiration:
            try:
                expiry_ts = pd.Timestamp(expiration).normalize()
                dte = (expiry_ts - pd.Timestamp.utcnow().normalize()).days
            except Exception:  # pragma: no cover - defensive
                dte = None
        records.append(
            {
                "symbol": opt.get("symbol"),
                "underlying": opt.get("underlying"),
                "option_type": opt.get("option_type"),
                "strike": opt.get("strike"),
                "bid": bid,
                "ask": ask,
                "mid": mid,
                "delta": greeks.get("delta"),
                "volume": opt.get("volume") or 0,
                "open_interest": opt.get("open_interest") or 0,
                "dte": dte,
                "expiration_date": expiration,
                "spread_pct": spread_pct if spread_pct is not None else 9.99,
            }
        )
    return pd.DataFrame.from_records(records)


async def fetch_option_chain(symbol: str, expiration: str | None = None) -> pd.DataFrame:
    """Retrieve a Tradier option chain as a pandas DataFrame."""
    config = _get_tradier_config()
    headers = {
        "Authorization": f"Bearer {config.token}",
        "Accept": "application/json",
    }
    async with httpx.AsyncClient(base_url=config.base_url, headers=headers, timeout=10.0) as client:
        resolved_expiry = await _resolve_expiration(client, symbol, expiration)
        chain_payload = await _fetch_json(
            client,
            "/v1/markets/options/chains",
            params={"symbol": symbol, "expiration": resolved_expiry},
        )
    options = chain_payload.get("options", {}).get("option", [])
    if not options:
        return pd.DataFrame()
    if isinstance(options, dict):
        options = [options]
    return _chain_to_dataframe(options)


async def select_tradier_contract(symbol: str, prefer_delta: float | None = 0.5) -> Dict[str, Any] | None:
    """Fetch the option chain and return a single contract suggestion."""
    from .contract_selector import filter_chain, pick_best_contract  # late import to avoid cycle

    chain = await fetch_option_chain(symbol)
    if chain.empty:
        logger.warning("Tradier chain empty for %s", symbol)
        return None

    rules = {
        "dte_range": (0, 90),
        "delta_range": (0.2, 0.8),
        "max_spread_pct": 0.25,
        "min_open_interest": 10,
        "min_volume": 1,
    }
    filtered = filter_chain(chain, rules)
    best = pick_best_contract(filtered, prefer_delta=prefer_delta)
    if best is None:
        logger.info("No contract matched filters for %s", symbol)
        return None
    return {
        "symbol": best.get("symbol"),
        "strike": best.get("strike"),
        "expiration": best.get("expiration_date"),
        "option_type": best.get("option_type"),
        "bid": best.get("bid"),
        "ask": best.get("ask"),
        "mid": best.get("mid"),
        "delta": best.get("delta"),
        "dte": best.get("dte"),
        "spread_pct": best.get("spread_pct"),
    }
