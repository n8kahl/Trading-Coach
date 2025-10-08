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

from fastapi import FastAPI, HTTPException, Response, APIRouter, Depends, Header, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from pydantic import BaseModel

import pandas as pd
import numpy as np

import httpx
import openai

from .config import get_settings
from .scanner import scan_market, Signal
from .follower import TradeFollower
from .agents_runtime import run_agent_turn


app = FastAPI(title="AI Trading Assistant API")

logger = logging.getLogger(__name__)

# In‑memory store of active trade followers keyed by trade ID
followers: Dict[str, TradeFollower] = {}

# Simple in-memory stores for GPT facade endpoints (per-user scoping)
_GPT_WATCHLISTS: Dict[str, List[str]] = {}
_GPT_NOTES: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
_GPT_TRADES: Dict[str, Dict[str, TradeFollower]] = {}

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


class AgentMessageRequest(BaseModel):
    message: str
    conversation_id: str | None = None


class ClientLog(BaseModel):
    level: str = "info"
    message: str
    context: dict | None = None


@app.on_event("startup")
async def on_startup() -> None:
    """Initialize global state on startup."""
    settings = get_settings()
    openai.api_key = settings.openai_api_key


# ---- API key auth for GPT Action facade ------------------------------------

class AuthedUser(BaseModel):
    user_id: str


async def require_api_key(
    authorization: str | None = Header(default=None, alias="Authorization"),
    x_user_id: str | None = Header(default=None, alias="X-User-Id"),
) -> AuthedUser:
    """Validate Bearer API key and extract the caller's user id.

    When BACKEND_API_KEY is configured we enforce it; otherwise we operate in
    development mode and allow unauthenticated access (useful for quick GPT
    prototyping on Railway).

    - If BACKEND_API_KEY is set, expect `Authorization: Bearer <BACKEND_API_KEY>`
      and optionally `X-User-Id` for per-user scoping.
    - If BACKEND_API_KEY is missing, allow the request and derive a user id from
      `X-User-Id` (defaulting to "anonymous").
    """
    settings = get_settings()
    expected = settings.backend_api_key
    if expected:
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing bearer token")
        token = authorization.split(" ", 1)[1]
        if token != expected:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid bearer token")
        user_id = x_user_id or "anonymous"
        return AuthedUser(user_id=user_id)

    # No backend API key configured: fall back to permissive mode.
    user_id = x_user_id or "anonymous"
    return AuthedUser(user_id=user_id)


# ---- GPT facade router ------------------------------------------------------

gpt = APIRouter(prefix="/gpt", tags=["gpt"])


class GPTScanRequest(BaseModel):
    tickers: List[str]
    style: str | None = None  # "scalp" | "intraday" | "swing" (optional filter)


@gpt.post("/scan")
async def gpt_scan(req: GPTScanRequest, user: AuthedUser = Depends(require_api_key)) -> List[Dict[str, Any]]:
    if not req.tickers:
        raise HTTPException(status_code=400, detail="No tickers provided")

    # Reuse the existing scanner with mock data; integrate Polygon later
    market_data: Dict[str, pd.DataFrame] = {}
    for t in req.tickers:
        idx = pd.date_range(end=pd.Timestamp.utcnow(), periods=60, freq="T")
        base = 100.0
        prices = base + np.cumsum(np.random.randn(60))
        df = pd.DataFrame(
            {
                "open": prices,
                "high": prices + np.random.rand(60),
                "low": prices - np.random.rand(60),
                "close": prices,
                "volume": np.random.randint(1000, 5000, size=60),
            },
            index=idx,
        )
        market_data[t] = df

    signals = await scan_market(req.tickers, market_data)
    # Map strategy ids to generic styles for the GPT
    def to_style(sid: str) -> str:
        if "orb" in sid:
            return "scalp"
        if "vwap" in sid or "inside" in sid:
            return "intraday"
        return "swing"

    items: List[Dict[str, Any]] = []
    for s in signals:
        style = to_style(s.strategy_id)
        if req.style and req.style != style:
            continue
        items.append(
            {
                "symbol": s.symbol,
                "style": style,
                "strategy_id": s.strategy_id,
                "description": s.description,
                "score": s.score,
                "contract_suggestion": s.contract,
                "features": s.features,
            }
        )
    return items


@gpt.get("/health")
async def gpt_health(user: AuthedUser = Depends(require_api_key)) -> Dict[str, str]:
    return {"status": "ok"}


class GPTFollowRequest(BaseModel):
    symbol: str
    direction: str  # "long" | "short"
    entry_price: float
    trade_id: str | None = None


@gpt.post("/follow")
async def gpt_follow(req: GPTFollowRequest, user: AuthedUser = Depends(require_api_key)) -> Dict[str, Any]:
    user_trades = _GPT_TRADES.setdefault(user.user_id, {})
    trade_id = req.trade_id or f"t_{os.urandom(6).hex()}"
    if trade_id in user_trades:
        follower = user_trades[trade_id]
    else:
        follower = TradeFollower(symbol=req.symbol, entry_price=req.entry_price, direction=req.direction)
        user_trades[trade_id] = follower
    msg = follower.update_from_price(req.entry_price, atr=1.0)
    return {
        "trade_id": trade_id,
        "symbol": follower.symbol,
        "direction": follower.direction,
        "stop": follower.stop_price,
        "target": follower.tp_price,
        "message": msg,
    }


@gpt.get("/trades/{trade_id}")
async def gpt_trade_state(trade_id: str, user: AuthedUser = Depends(require_api_key)) -> Dict[str, Any]:
    follower = _GPT_TRADES.get(user.user_id, {}).get(trade_id)
    if not follower:
        raise HTTPException(status_code=404, detail="Trade not found")
    return {
        "trade_id": trade_id,
        "symbol": follower.symbol,
        "direction": follower.direction,
        "stop": follower.stop_price,
        "target": follower.tp_price,
        "last_event": follower.last_event,
    }


@gpt.get("/watchlist")
async def gpt_get_watchlist(user: AuthedUser = Depends(require_api_key)) -> Dict[str, Any]:
    return {"tickers": _GPT_WATCHLISTS.get(user.user_id, [])}


class GPTWatchlistUpdate(BaseModel):
    tickers: List[str]


@gpt.post("/watchlist")
async def gpt_set_watchlist(update: GPTWatchlistUpdate, user: AuthedUser = Depends(require_api_key)) -> Dict[str, Any]:
    uniq = sorted({t.upper() for t in update.tickers})
    _GPT_WATCHLISTS[user.user_id] = uniq
    return {"tickers": uniq}


@gpt.get("/notes")
async def gpt_get_notes(date: str, user: AuthedUser = Depends(require_api_key)) -> Dict[str, Any]:
    notes = _GPT_NOTES.get(user.user_id, {}).get(date, [])
    return {"date": date, "notes": notes}


class GPTNoteIn(BaseModel):
    date: str  # ISO date string
    text: str


@gpt.post("/notes")
async def gpt_add_note(note: GPTNoteIn, user: AuthedUser = Depends(require_api_key)) -> Dict[str, Any]:
    user_notes = _GPT_NOTES.setdefault(user.user_id, {})
    day = user_notes.setdefault(note.date, [])
    item = {"id": f"n_{os.urandom(6).hex()}", "text": note.text}
    day.append(item)
    return {"date": note.date, "note": item}


@gpt.get("/widgets/{kind}")
async def gpt_widget(kind: str, symbol: str | None = None, user: AuthedUser = Depends(require_api_key)) -> Dict[str, Any]:
    """Return compact JSON payloads suitable for rendering as markdown cards."""
    if kind == "ticker_wedge" and symbol:
        return {
            "type": "ticker_wedge",
            "symbol": symbol.upper(),
            "pattern": "rising_wedge",
            "confidence": 0.72,
            "levels": {"support": 98.4, "resistance": 102.6},
        }
    if kind == "playbook_today":
        wl = _GPT_WATCHLISTS.get(user.user_id, [])
        return {
            "type": "playbook_today",
            "tickers": wl[:6],
            "themes": ["trend_continuation", "mean_reversion"],
        }
    raise HTTPException(status_code=404, detail="Unknown widget kind or missing params")


@app.post("/api/chatkit/session")
async def create_chatkit_session() -> Dict[str, str]:
    """Create a new ChatKit session and return the client secret.

    This endpoint proxies the OpenAI ChatKit SDK and returns the `client_secret`
    needed by the frontend to initialize ChatKit.  If the ChatKit SDK fails,
    the endpoint returns a dummy secret for testing.
    """
    settings = get_settings()
    try:
        user_id = settings.chatkit_user_id or f"web-user-{os.urandom(6).hex()}"
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
                    "user": user_id,
                },
            )
        resp.raise_for_status()
        data = resp.json()
        client_secret = data.get("client_secret")
        if not client_secret:
            raise HTTPException(status_code=502, detail="OpenAI response missing client_secret")
        return {
            "client_secret": client_secret,
            "workflow_id": settings.workflow_id,
            "user_id": user_id,
        }
    except httpx.HTTPStatusError as exc:
        logger.error("ChatKit session request failed: %s", exc.response.text)
        # Bubble up failure so the frontend can show a meaningful message.
        raise HTTPException(status_code=exc.response.status_code, detail=exc.response.text)
    except Exception:
        logger.exception("Failed to create ChatKit session from OpenAI")
        # Fallback: return a dummy secret in development
        return {
            "client_secret": "dummy-client-secret",
            "workflow_id": settings.workflow_id,
            "user_id": "demo-user",
        }


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


@app.post("/api/agent/respond")
async def agent_respond(req: AgentMessageRequest) -> Dict[str, Any]:
    """Proxy a message to the OpenAI Agents SDK trading assistant.

    This endpoint complements ChatKit by allowing the frontend (or tools)
    to retrieve structured responses that may include widget payloads.
    """
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")
    try:
        result = await run_agent_turn(req.message, conversation_id=req.conversation_id)
        return result
    except Exception as exc:  # pragma: no cover - surface useful error detail
        logger.exception("Agent turn failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/debug/log")
async def debug_log(entry: ClientLog) -> Dict[str, Any]:
    """Accept client-side debug logs and write them to server logs.

    Useful on hosted environments (e.g., Railway) where browser console output
    isn't visible in server logs. Do not send secrets in `context`.
    """
    lvl = (entry.level or "info").lower()
    msg = entry.message
    ctx = entry.context or {}
    if lvl == "error":
        logger.error("client: %s | ctx=%s", msg, ctx)
    elif lvl == "warning" or lvl == "warn":
        logger.warning("client: %s | ctx=%s", msg, ctx)
    else:
        logger.info("client: %s | ctx=%s", msg, ctx)
    return {"ok": True}


@app.get("/healthz")
async def healthz() -> Dict[str, str]:
    return {"status": "ok"}


# Cache for proxied ChatKit script to avoid repeated upstream fetches
_CHATKIT_JS_CACHE: Dict[str, bytes] = {}


@app.get("/assets/chatkit.js")
async def chatkit_js_proxy() -> Response:
    """Serve the ChatKit web component script from our origin.

    Some environments block public CDNs; proxying the script avoids those
    issues and ensures the custom element registers.
    """
    cached = _CHATKIT_JS_CACHE.get("content")
    if cached:
        return Response(cached, media_type="application/javascript")

    sources = [
        "https://cdn.jsdelivr.net/npm/@openai/chatkit-widget@latest/dist/web.js",
        "https://cdn.platform.openai.com/deployments/chatkit/chatkit.js",
        "https://unpkg.com/@openai/chatkit-widget@latest/dist/web.js",
    ]
    last_error: str | None = None
    async with httpx.AsyncClient(timeout=12.0) as client:
        for url in sources:
            try:
                r = await client.get(url)
                r.raise_for_status()
                content = r.content
                _CHATKIT_JS_CACHE["content"] = content
                logger.info("chatkit.js proxied from %s (len=%d)", url, len(content))
                return Response(content, media_type="application/javascript")
            except Exception as exc:  # pragma: no cover
                last_error = f"{type(exc).__name__}: {exc}"
                logger.warning("failed to fetch chatkit.js from %s: %s", url, exc)

    raise HTTPException(status_code=502, detail=f"Failed to fetch ChatKit script: {last_error}")


# Register GPT facade routes before static so they take precedence
app.include_router(gpt)

# Mount static frontend last so API routes take precedence and POSTs to /api/*
# are not intercepted by the static handler (which would return 405).
# Prefer the built React app in `frontend_dist/` if present, otherwise fall back
# to the legacy static demo in `frontend/`.
dist_dir = Path("frontend_dist")
static_root = "frontend_dist" if dist_dir.is_dir() else "frontend"
app.mount("/", StaticFiles(directory=static_root, html=True), name="static")
