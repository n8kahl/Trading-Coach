# Trading Coach GPT Backend

A quick note: latest deploy refresh triggered on 2025-10-08 15:15 UTC.

A lightweight FastAPI service that exposes trading utilities for consumption by a custom GPT (via Actions).  
It keeps the quantitative bits—market scanning, ATR-based trade management, watchlists, and journaling—while stripping out the legacy ChatKit UI and OpenAI Agents runtime.

> ⚠️ **Disclaimer:** All code and strategy examples are for educational purposes only.  
> They are not a recommendation to trade securities. Always do your own research and consult a licensed professional.

## Features

- **`POST /gpt/scan`** – rank tickers using the built-in strategy library backed by live Polygon/Yahoo OHLCV.  
- **`POST /gpt/follow` & `GET /gpt/trades/{id}`** – manage trades with an ATR-driven follower.  
- **`GET/POST /gpt/watchlist`** – read & update user-specific ticker lists.  
- **`GET/POST /gpt/notes`** – lightweight daily journal.  
- **`GET /gpt/widgets/{kind}`** – generate compact card payloads you can render as Markdown/JSON in the GPT response.  
- Optional bearer auth (`BACKEND_API_KEY`) plus `X-User-Id` scoping; falls back to anonymous mode for quick prototyping.

## Project layout

```
.
├── README.md
├── requirements.txt
├── nixpacks.toml            # Deploys a Python-only image on Railway
├── Procfile                 # uvicorn launch command
└── src/
    ├── agent_server.py      # FastAPI app (all /gpt endpoints + health checks)
    ├── calculations.py      # Indicator helpers (EMA, ATR, VWAP, ADX, etc.)
    ├── contract_selector.py # Option contract filters (placeholder)
    ├── follower.py          # TradeFollower state machine
    ├── scanner.py           # Strategy evaluation engine
    ├── strategy_library.py  # Declarative strategy definitions
    ├── backtester.py        # Stub for future historical analysis
    └── config.py            # Environment settings (Polygon, backend API key)
```

## Getting started

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn src.agent_server:app --reload --port 8000
```

Environment variables (recommended):

```bash
# .env
POLYGON_API_KEY=pk_xxx          # not yet used, but reserved for real data
BACKEND_API_KEY=super-secret    # omit to allow anonymous access during dev
# Tradier sandbox (provided example values)
TRADIER_SANDBOX_TOKEN=3QP4qlzY6acyDQujsYDp3lk15Xyj
TRADIER_SANDBOX_ACCOUNT=VA52364852
```

Visit `http://localhost:8000/docs` for interactive Swagger docs, or call endpoints directly:

```bash
curl http://localhost:8000/gpt/scan \
  -H "Content-Type: application/json" \
  -d '{"tickers":["AAPL","MSFT","TSLA"]}'
```

## Connecting to myGPT (Actions)

1. In the GPT Builder, open **Actions → Add action**.  
2. Use the deployed base URL (e.g. `https://your-railway-domain.up.railway.app`).  
3. Set the schema URL to `https://.../openapi.json`. FastAPI exposes an up-to-date OpenAPI doc.  
4. Configure security:  
   - If `BACKEND_API_KEY` is set, add a Bearer token in the Action request headers and supply an `X-User-Id` header.  
   - Otherwise leave auth empty—the backend will accept anonymous calls and store data under `anonymous`.  
5. Provide the GPT with short usage tips, e.g.:

   ```
   - Use /gpt/scan with a handful of tickers and optionally style=intraday.
   - Call /gpt/follow after recommending a trade; read status via /gpt/trades/{trade_id}.
   - Persist user watchlists via /gpt/watchlist and journal entries via /gpt/notes.
   ```

The GPT can now reason about the user’s portfolio, recall notes, and fetch strategy-driven scans.

## Implementation notes

- **Market data:** `/gpt/scan` and `/charts` pull Polygon aggregates with a Yahoo fallback. Provide credentials before production use.  
- **Persistence:** Everything lives in in-memory dictionaries (`_WATCHLISTS`, `_NOTES`, `_TRADES`). Swap these for a real database layer before production.  
- **Strategies:** `strategy_library.py` defines several sample setups (ORB retest, VWAP/AVWAP, etc.). Expand or tweak scoring in `scanner.py` as needed.  
- **Trade management:** `TradeFollower` tracks ATR-based stops and trailing behaviour. The current endpoints feed it only the entry price; extend with live price/ATR data for meaningful guidance.

## Next steps

1. Fetch real market data (Polygon aggregates into `market_data` inside `/gpt/scan`).  
2. Persist user artefacts in Postgres or Firestore.  
3. Harden auth (JWT, OAuth) if you expose the service publicly.  
4. Extend the Tradier integration to respect strategy-specific filters (DTE, delta targets).  
5. Add additional GPT-friendly endpoints (e.g. `/gpt/volatility`, `/gpt/backtest`).  
6. Write tests for the critical logic (scanner heuristics, trade follower transitions).

Happy building, and trade responsibly!
