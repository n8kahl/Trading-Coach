"""FastAPI server exposing endpoints for scanning, following trades and creating ChatKit sessions.

This module wires together the other components of the trading assistant into a
web API.  It uses the OpenAI Python SDK to create ChatKit sessions, the
`scanner` module to find setups, and the `follower` module to manage trade
state machines.  In a production environment the endpoints would be secured
with authentication and connect to persistent storage.  Here we keep things
simple for demonstration purposes.
"""

from __future__ import annotations

import os
import asyncio
import logging
from typing import List, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import pandas as pd
import numpy as np

import httpx
import openai

from .config import get_settings
from .scanner import scan_market, Signal
from .follower import TradeFollower


app = FastAPI(title="AI Trading Assistant API")

logger = logging.getLogger(__name__)

# In‑memory store of active trade followers keyed by trade ID
followers: Dict[str, TradeFollower] = {}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ScanRequest(BaseModel):
    tickers: List[str]
    # In a full implementation you might include timeframe and other params


class FollowRequest(BaseModel):
    trade_id: str
    symbol: str
    direction: str  # "long" or "short"
    entry_price: float
    # Additional parameters (ATR period, risk multiples) could be added here


@app.on_event("startup")
async def on_startup() -> None:
    """Initialize global state on startup."""
    settings = get_settings()
    openai.api_key = settings.openai_api_key


@app.post("/api/chatkit/session")
async def create_chatkit_session() -> Dict[str, str]:
    """Create a new ChatKit session and return the client secret.

    This endpoint proxies the OpenAI ChatKit SDK and returns the `client_secret`
    needed by the frontend to initialize ChatKit.  If the ChatKit SDK fails,
    the endpoint returns a dummy secret for testing.
    """
    settings = get_settings()
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                "https://api.openai.com/v1/chatkit/sessions",
                headers={
                    "Authorization": f"Bearer {settings.openai_api_key}",
                    "Content-Type": "application/json",
                    "OpenAI-Beta": "chatkit_beta=v1",
                },
                json={
                    "workflow": {
                        "id": settings.workflow_id,
                    },
                    "user": settings.chatkit_user_id,
                },
            )
        resp.raise_for_status()
        data = resp.json()
        client_secret = data.get("client_secret")
        if not client_secret:
            raise HTTPException(status_code=502, detail="OpenAI response missing client_secret")
        return {"client_secret": client_secret}
    except httpx.HTTPStatusError as exc:
        logger.error("ChatKit session request failed: %s", exc.response.text)
        # Bubble up failure so the frontend can show a meaningful message.
        raise HTTPException(status_code=exc.response.status_code, detail=exc.response.text)
    except Exception:
        logger.exception("Failed to create ChatKit session from OpenAI")
        # Fallback: return a dummy secret in development
        return {"client_secret": "dummy-client-secret"}


@app.post("/api/scan", response_model=List[Dict[str, Any]])
async def api_scan(req: ScanRequest) -> List[Dict[str, Any]]:
    """Scan the provided tickers and return ranked signals.

    The scanner currently uses placeholder data.  To implement real scanning,
    replace the mock data generation with calls to Polygon’s APIs and pass
    the resulting DataFrames into `scan_market`.
    """
    if not req.tickers:
        raise HTTPException(status_code=400, detail="No tickers provided")

    # Generate dummy OHLCV data for demonstration.  In real usage, fetch from Polygon.
    market_data: Dict[str, pd.DataFrame] = {}
    for ticker in req.tickers:
        # Create a DataFrame with 60 rows of synthetic data
        index = pd.date_range(end=pd.Timestamp.utcnow(), periods=60, freq="T")
        base_price = 100 + np.random.rand() * 20
        prices = base_price + np.cumsum(np.random.randn(60))
        highs = prices + np.random.rand(60)
        lows = prices - np.random.rand(60)
        volumes = np.random.randint(1000, 5000, size=60)
        df = pd.DataFrame({
            "open": prices,
            "high": highs,
            "low": lows,
            "close": prices,
            "volume": volumes,
        }, index=index)
        market_data[ticker] = df

    signals: List[Signal] = await scan_market(req.tickers, market_data)
    # Convert dataclass instances into JSON‑serializable dicts
    results: List[Dict[str, Any]] = []
    for sig in signals:
        results.append({
            "symbol": sig.symbol,
            "strategy_id": sig.strategy_id,
            "description": sig.description,
            "score": sig.score,
            "contract": sig.contract,
            "features": sig.features,
        })
    return results


@app.post("/api/follow")
async def api_follow(req: FollowRequest) -> Dict[str, Any]:
    """Register a new trade follower and return initial guidance.

    The endpoint creates a `TradeFollower` instance for the given trade ID and
    symbol, stores it in memory, and returns the first set of instructions.
    In a full implementation you would stream subsequent instructions to the
    client via WebSocket or Server‑Sent Events.
    """
    if req.trade_id in followers:
        raise HTTPException(status_code=400, detail="Trade ID already exists")
    follower = TradeFollower(
        symbol=req.symbol,
        entry_price=req.entry_price,
        direction=req.direction,
    )
    followers[req.trade_id] = follower
    # On first update, compute a dummy ATR value (should come from real data)
    dummy_atr = 1.0
    msg = follower.update_from_price(req.entry_price, dummy_atr)
    return {
        "message": msg,
        "stop": follower.stop_price,
        "target": follower.tp_price,
    }
