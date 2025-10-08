"""Trading Coach backend tailored for GPT Actions integrations.

This service focuses on a small, well-documented HTTP surface that a custom
GPT (via Actions) can call to reason about trade ideas, manage active trades,
and store lightweight user data such as watchlists or journal notes.

The implementation keeps the original quantitative helpers (scanner,
trade follower, indicator utilities) so the GPT still has access to rich
trading context, but removes anything related to hosted UIs or ChatKit.
"""

from __future__ import annotations

import asyncio
import json
import os
import logging
from typing import Any, Dict, List
from urllib.parse import urlencode

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from fastapi.responses import HTMLResponse

from .config import get_settings
from .follower import TradeFollower
from .scanner import scan_market
from .tradier import select_tradier_contract


logger = logging.getLogger(__name__)

app = FastAPI(
    title="Trading Coach GPT Backend",
    description="Backend utilities for a custom GPT that offers trading guidance.",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Auth helper
# ---------------------------------------------------------------------------

class AuthedUser(BaseModel):
    user_id: str


async def require_api_key(
    authorization: str | None = Header(default=None, alias="Authorization"),
    x_user_id: str | None = Header(default=None, alias="X-User-Id"),
) -> AuthedUser:
    """Optional API key check for GPT Actions.

    If `BACKEND_API_KEY` is set we enforce it. Otherwise the app falls back to
    a permissive mode that uses `X-User-Id` (or `anonymous`) to scope data.
    """

    settings = get_settings()
    expected = settings.backend_api_key

    if expected:
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Missing bearer token")
        token = authorization.split(" ", 1)[1]
        if token != expected:
            raise HTTPException(status_code=401, detail="Invalid bearer token")

    user_id = x_user_id or "anonymous"
    return AuthedUser(user_id=user_id)


# ---------------------------------------------------------------------------
# In-memory stores (swap with a real database when ready)
# ---------------------------------------------------------------------------

_WATCHLISTS: Dict[str, List[str]] = {}
_NOTES: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
_TRADES: Dict[str, Dict[str, TradeFollower]] = {}


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class ScanUniverse(BaseModel):
    tickers: List[str] = Field(..., description="Ticker symbols to analyse")
    style: str | None = Field(
        default=None,
        description="Optional style filter: 'scalp', 'intraday', or 'swing'.",
    )


class FollowTradeIn(BaseModel):
    symbol: str = Field(..., description="Underlying symbol, e.g. 'AAPL'")
    direction: str = Field(..., pattern="^(long|short)$")
    entry_price: float = Field(..., gt=0)
    trade_id: str | None = Field(
        default=None,
        description="Existing trade identifier to continue following.",
    )


class WatchlistUpdate(BaseModel):
    tickers: List[str]


class NoteIn(BaseModel):
    date: str = Field(..., description="ISO date string, e.g. 2025-10-08")
    text: str = Field(..., description="Journal entry")


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _style_for_strategy(strategy_id: str) -> str:
    sid = strategy_id.lower()
    if "orb" in sid:
        return "scalp"
    if "vwap" in sid or "inside" in sid:
        return "intraday"
    return "swing"


def _synth_ohlcv(ticker: str, bars: int = 60) -> pd.DataFrame:
    """Temporary data source until Polygon integration is wired up."""
    idx = pd.date_range(end=pd.Timestamp.utcnow(), periods=bars, freq="T")
    base = 100 + np.random.rand() * 25
    prices = base + np.cumsum(np.random.randn(bars))
    return pd.DataFrame(
        {
            "open": prices,
            "high": prices + np.random.rand(bars),
            "low": prices - np.random.rand(bars),
            "close": prices,
            "volume": np.random.randint(1_000, 5_000, size=bars),
        },
        index=idx,
    )


# Strategy utilities ---------------------------------------------------------

def _direction_for_strategy(strategy_id: str) -> str:
    sid = strategy_id.lower()
    if "short" in sid or "put" in sid:
        return "short"
    return "long"


def _suggest_levels(entry: float, atr_value: float | None, direction: str) -> tuple[float, float, float]:
    atr = abs(float(atr_value)) if atr_value else max(entry * 0.01, 0.25)
    if direction == "short":
        stop = entry + atr
        target = max(entry - 2 * atr, 0.01)
    else:
        stop = max(entry - atr, 0.01)
        target = entry + 2 * atr
    return entry, stop, target


def _indicators_for_strategy(strategy_id: str) -> List[str]:
    sid = strategy_id.lower()
    if "vwap" in sid:
        return ["VWAP", "EMA20"]
    if "orb" in sid:
        return ["VWAP", "EMA9", "EMA20"]
    if "adx" in sid:
        return ["VWAP", "ADX"]
    return ["VWAP"]


def _timeframe_for_style(style: str | None) -> str:
    mapping = {"scalp": "1", "intraday": "5", "swing": "60"}
    return mapping.get(style or "", "5")


# ---------------------------------------------------------------------------
# GPT router
# ---------------------------------------------------------------------------

gpt = APIRouter(prefix="/gpt", tags=["gpt"])


@gpt.get("/health", summary="Lightweight readiness probe")
async def gpt_health(_: AuthedUser = Depends(require_api_key)) -> Dict[str, str]:
    return {"status": "ok"}


@gpt.post("/scan", summary="Rank trade setups across a list of tickers")
async def gpt_scan(
    universe: ScanUniverse,
    request: Request,
    user: AuthedUser = Depends(require_api_key),
) -> List[Dict[str, Any]]:
    if not universe.tickers:
        raise HTTPException(status_code=400, detail="No tickers provided")

    # TODO: replace with Polygon data fetch using settings.polygon_api_key.
    market_data = {ticker: _synth_ohlcv(ticker) for ticker in universe.tickers}
    signals = await scan_market(universe.tickers, market_data)

    contract_suggestions: Dict[str, Dict[str, Any] | None] = {}
    unique_symbols = sorted({signal.symbol for signal in signals})
    if unique_symbols:
        settings = get_settings()
        if not settings.tradier_token:
            logger.info("Tradier token not configured; skipping contract suggestions.")
        else:
            try:
                tasks = [select_tradier_contract(symbol) for symbol in unique_symbols]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for symbol, result in zip(unique_symbols, results):
                    if isinstance(result, Exception):
                        logger.warning("Tradier contract lookup failed for %s: %s", symbol, result)
                        contract_suggestions[symbol] = None
                    else:
                        contract_suggestions[symbol] = result
            except Exception as exc:  # pragma: no cover - safety net
                logger.warning("Tradier integration error: %s", exc)

    payload: List[Dict[str, Any]] = []
    for signal in signals:
        style = _style_for_strategy(signal.strategy_id)
        if universe.style and universe.style != style:
            continue
        latest_row = market_data[signal.symbol].iloc[-1]
        entry_price = float(latest_row["close"])
        atr_value = signal.features.get("atr")
        direction = _direction_for_strategy(signal.strategy_id)
        entry, stop, target = _suggest_levels(entry_price, atr_value, direction)
        indicators = _indicators_for_strategy(signal.strategy_id)
        chart_params = {
            "entry": f"{entry:.2f}",
            "stop": f"{stop:.2f}",
            "target": f"{target:.2f}",
            "direction": direction,
            "tf": _timeframe_for_style(style),
            "indicators": ",".join(indicators),
        }
        chart_url = str(request.url_for("chart_page", symbol=signal.symbol.upper()))
        chart_url = f"{chart_url}?{urlencode(chart_params)}"
        payload.append(
            {
                "symbol": signal.symbol,
                "style": style,
                "strategy_id": signal.strategy_id,
                "description": signal.description,
                "score": signal.score,
                "contract_suggestion": contract_suggestions.get(signal.symbol),
                "direction": direction,
                "levels": {
                    "entry": round(entry, 2),
                    "stop": round(stop, 2),
                    "target": round(target, 2),
                },
                "chart_url": chart_url,
                "features": signal.features,
            }
        )

    logger.info("scan universe=%s user=%s results=%d", universe.tickers, user.user_id, len(payload))
    return payload


@gpt.post("/follow", summary="Start or resume ATR-based trade management")
async def gpt_follow(
    body: FollowTradeIn,
    user: AuthedUser = Depends(require_api_key),
) -> Dict[str, Any]:
    store = _TRADES.setdefault(user.user_id, {})
    trade_id = body.trade_id or f"t_{os.urandom(6).hex()}"

    follower = store.get(trade_id)
    if follower is None:
        follower = TradeFollower(symbol=body.symbol, entry_price=body.entry_price, direction=body.direction)
        store[trade_id] = follower

    # Placeholder ATR; replace with real intraday data when available.
    message = follower.update_from_price(body.entry_price, atr_value=1.0)
    return {
        "trade_id": trade_id,
        "symbol": follower.symbol,
        "direction": follower.direction,
        "stop": follower.stop_price,
        "target": follower.tp_price,
        "message": message,
        "state": follower.state.value,
    }


@gpt.get("/trades/{trade_id}", summary="Retrieve the latest trade follower state")
async def gpt_trade_state(trade_id: str, user: AuthedUser = Depends(require_api_key)) -> Dict[str, Any]:
    follower = _TRADES.get(user.user_id, {}).get(trade_id)
    if follower is None:
        raise HTTPException(status_code=404, detail="Trade not found")
    return {
        "trade_id": trade_id,
        "symbol": follower.symbol,
        "direction": follower.direction,
        "stop": follower.stop_price,
        "target": follower.tp_price,
        "scaled": follower.scaled,
        "state": follower.state.value,
    }


@gpt.get("/watchlist", summary="Get the caller's watchlist")
async def gpt_watchlist(user: AuthedUser = Depends(require_api_key)) -> Dict[str, Any]:
    return {"tickers": _WATCHLISTS.get(user.user_id, [])}


@gpt.post("/watchlist", summary="Replace the caller's watchlist")
async def gpt_update_watchlist(
    update: WatchlistUpdate,
    user: AuthedUser = Depends(require_api_key),
) -> Dict[str, Any]:
    tickers = sorted({ticker.upper() for ticker in update.tickers})
    _WATCHLISTS[user.user_id] = tickers
    return {"tickers": tickers}


@gpt.get("/notes", summary="Read trading journal entries for a date")
async def gpt_get_notes(date: str, user: AuthedUser = Depends(require_api_key)) -> Dict[str, Any]:
    entries = _NOTES.get(user.user_id, {}).get(date, [])
    return {"date": date, "notes": entries}


@gpt.post("/notes", summary="Append a trading journal entry")
async def gpt_add_note(note: NoteIn, user: AuthedUser = Depends(require_api_key)) -> Dict[str, Any]:
    notebook = _NOTES.setdefault(user.user_id, {})
    entries = notebook.setdefault(note.date, [])
    entry = {"id": f"n_{os.urandom(6).hex()}", "text": note.text}
    entries.append(entry)
    return {"date": note.date, "note": entry}


@gpt.get("/widgets/{kind}", summary="Generate lightweight dashboard widgets")
async def gpt_widget(kind: str, symbol: str | None = None, user: AuthedUser = Depends(require_api_key)) -> Dict[str, Any]:
    if kind == "ticker_wedge" and symbol:
        return {
            "type": "ticker_wedge",
            "symbol": symbol.upper(),
            "pattern": "rising_wedge",
            "confidence": 0.72,
            "levels": {"support": 98.4, "resistance": 102.6},
        }
    if kind == "playbook_today":
        watchlist = _WATCHLISTS.get(user.user_id, [])
        return {
            "type": "playbook_today",
            "tickers": watchlist[:6],
            "themes": ["trend_continuation", "mean_reversion"],
        }
    raise HTTPException(status_code=404, detail="Unknown widget kind or missing params")


# Register GPT endpoints with the application
app.include_router(gpt)


# ---------------------------------------------------------------------------
# Platform health endpoints
# ---------------------------------------------------------------------------


@app.get("/chart/{symbol}", response_class=HTMLResponse, name="chart_page")
async def chart_page(symbol: str, entry: float, stop: float, target: float, direction: str = "long", tf: str = "5", indicators: str = "VWAP") -> HTMLResponse:
    indicator_list = [item.strip() for item in indicators.split(",") if item.strip()]
    data = {
        "symbol": symbol.upper(),
        "entry": entry,
        "stop": stop,
        "target": target,
        "direction": direction,
        "timeframe": tf,
        "indicators": indicator_list,
    }
    html = f"""
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <title>{symbol.upper()} trading plan</title>
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <style>
      html, body {{ margin: 0; height: 100%; background-color: #0f172a; color: #e2e8f0; font-family: sans-serif; }}
      #tv_container {{ height: 100vh; width: 100vw; }}
      .legend {{ position: absolute; top: 12px; left: 12px; background: rgba(15, 23, 42, 0.7); padding: 12px 16px; border-radius: 8px; }}
      .legend h1 {{ margin: 0 0 8px 0; font-size: 20px; }}
      .legend p {{ margin: 4px 0; font-size: 14px; }}
    </style>
  </head>
  <body>
    <div id="tv_container"></div>
    <div class="legend">
      <h1>{symbol.upper()}</h1>
      <p>Direction: {direction.capitalize()}</p>
      <p>Entry: {entry:.2f}</p>
      <p>Stop: {stop:.2f}</p>
      <p>Target: {target:.2f}</p>
    </div>
    <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
    <script>
      const data = {json.dumps(data)};
      function addIndicator(chart, name) {{
        if (name === "VWAP") {{
          chart.createStudy("VWAP", false, false);
        }} else if (name.startsWith("EMA")) {{
          const length = parseInt(name.slice(3), 10) || 20;
          chart.createStudy("Moving Average Exponential", false, false, [length]);
        }} else if (name === "ADX") {{
          chart.createStudy("Average Directional Index", false, false, [14]);
        }}
      }}
      const widget = new TradingView.widget({{
        autosize: true,
        symbol: data.symbol,
        interval: data.timeframe,
        timezone: "Etc/UTC",
        theme: "dark",
        style: "1",
        container_id: "tv_container",
        allow_symbol_change: false,
        hide_side_toolbar: false,
        studies_overrides: {{}},
        overrides: {{
          "paneProperties.backgroundType": "solid",
          "paneProperties.background": "#0f172a",
        }},
        locale: "en"
      }});
      widget.onChartReady(function() {{
        const chart = widget.activeChart();
        chart.setResolution(data.timeframe, function(){{}});
        data.indicators.forEach((indicator) => addIndicator(chart, indicator));
        const entryLine = chart.createHorizontalLine(data.entry, {{ text: "Entry {entry:.2f}", color: "#2563eb" }});
        entryLine.setExtendLeft(true);
        const stopLine = chart.createHorizontalLine(data.stop, {{ text: "Stop {stop:.2f}", color: "#ef4444" }});
        stopLine.setExtendLeft(true);
        const targetLine = chart.createHorizontalLine(data.target, {{ text: "Target {target:.2f}", color: "#22c55e" }});
        targetLine.setExtendLeft(true);
      }});
    </script>
  </body>
</html>
"""
    return HTMLResponse(content=html)

@app.get("/healthz", summary="Readiness probe used by Railway")
async def healthz() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/", summary="Service metadata")
async def root() -> Dict[str, Any]:
    return {
        "name": "trading-coach-gpt-backend",
        "description": "Backend endpoints intended for a custom GPT Action.",
        "routes": {
            "scan": "/gpt/scan",
            "follow": "/gpt/follow",
            "trade_state": "/gpt/trades/{trade_id}",
            "watchlist": "/gpt/watchlist",
            "notes": "/gpt/notes",
            "widgets": "/gpt/widgets/{kind}",
        },
    }
