# AI‑Driven Options Trading Assistant

This repository contains a **plug‑and‑play starter kit** for building an AI‑powered options trading assistant.  
It combines proven technical trading strategies, real‑time market data, and OpenAI’s AgentKit/ChatKit to deliver actionable trade ideas and step‑by‑step coaching.

> **Disclaimer:** This codebase and the accompanying strategy library are for educational and demonstration purposes only.  
> They do **not** constitute financial advice or an offer to trade securities.  
> You are solely responsible for your trading decisions and should consult a qualified financial advisor before placing any trades.

## Project structure

```
trading_bot/
├── README.md             # This file
├── .gitignore
├── requirements.txt       # Python dependencies
├── src/                   # Backend Python code
│   ├── __init__.py
│   ├── agent_server.py    # FastAPI app exposing endpoints for scanning, following and ChatKit sessions
│   ├── calculations.py    # Core indicator calculations (ATR, EMA, VWAP, etc.)
│   ├── contract_selector.py # Functions for selecting option contracts based on delta/DTE/liquidity
│   ├── config.py          # Configuration and environment variable management
│   ├── follower.py        # Trade follower state machine for real‑time coaching
│   ├── scanner.py         # Market scanner that looks for A+ setups
│   ├── strategy_library.py # Declarative strategy definitions used by the scanner
│   └── backtester.py      # Placeholder for backtesting logic (for future work)
├── docs/
│   └── strategies_calculations.md # Detailed description of every strategy and indicator
├── frontend/
│   ├── index.html         # Minimal example embedding ChatKit via a Web Component
│   └── chatkit.js         # Helper code for initializing ChatKit on the front‑end
└── project.zip            # Generated zip ready to upload to GitHub (see below)
```

## Prerequisites

1. **Python 3.10+** – Used for the backend services and indicator computations.  
   You can install Python from [python.org](https://python.org) or via [Homebrew](https://brew.sh) on macOS.  
2. **Node.js 18+** (optional) – Only required if you want to extend the frontend beyond the provided static example.  
3. **OpenAI API key** – Obtainable from the [OpenAI dashboard](https://platform.openai.com/account/api-keys).  
4. **Polygon.io API key** – Sign up for a free or paid account at [polygon.io](https://polygon.io) to access real‑time and historical market data.  
5. **PostgreSQL database** (optional) – For storing backtest results and signals.  The starter kit runs without a database, but backtesting features expect one.

## Quick start

The following steps will get you up and running with a basic agent that can scan for trade setups and respond to user queries via ChatKit.  All commands assume you run them from the repository root (`trading_bot/`).

### 1. Install backend dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The `requirements.txt` file pins dependencies such as `fastapi`, `uvicorn`, `openai`, `pydantic`, and technical analysis utilities.

### 2. Set environment variables

Create a `.env` file in the repository root (this file is ignored by Git).  Populate it with your own API keys and configuration:

```bash
POLYGON_API_KEY=your_polygon_key_here
OPENAI_API_KEY=your_openai_key_here
WORKFLOW_ID=your_agent_builder_workflow_id
CHATKIT_API_KEY=your_chatkit_secret_key  # optional if you use advanced integration
DB_URL=postgresql+asyncpg://user:password@localhost:5432/trading
```

The `WORKFLOW_ID` refers to the ID of an agent workflow you have created using **Agent Builder**, OpenAI’s visual canvas for designing multi‑step agent workflows【128171747671312†screenshot】.

### 3. Run the backend server

The backend exposes three primary endpoints:

* **`/api/chatkit/session`** – Generates a ChatKit session token via the OpenAI Python SDK and returns the `client_secret`【220893600390724†screenshot】.  The ChatKit frontend uses this token to authenticate the user.
* **`/api/scan`** – Scans the market universe for A+ setups based on the strategy library and returns a ranked list of signals.  Placeholder logic is included; you should implement the actual scanning using Polygon’s REST and WebSocket endpoints.
* **`/api/follow/{trade_id}`** – Subscribes to a trade identified by `trade_id` and streams real‑time coaching instructions (e.g. update stops, scale out at targets).  The implementation uses an internal state machine.
* **`/api/agent/respond`** – Proxies a user prompt to the OpenAI Agents-based trading coach (`openai-agents` SDK) and returns the model’s final output plus optional widget payloads that the frontend can render alongside ChatKit.

Start the server locally using `uvicorn`:

```bash
uvicorn src.agent_server:app --reload --host 0.0.0.0 --port 8000
```

Visit `http://localhost:8000/docs` for interactive Swagger documentation of the API.  You can test the `/api/chatkit/session` endpoint here and verify that a `client_secret` is returned.

### 4. Frontend (bundled ChatKit React)

We now ship a bundled React front‑end (Vite) that renders ChatKit without relying on public CDNs.

Build locally:

```bash
# from repo root
cd web
npm ci
npm run build   # outputs to ../frontend_dist/
cd ..
uvicorn src.agent_server:app --reload --host 0.0.0.0 --port 8000
# open http://localhost:8000
```

On Railway, Nixpacks will run `npm ci && npm run build` if you add a root build step, or you can create a deploy hook to run `npm --prefix web ci && npm --prefix web run build` before starting the Python process. The server prefers `frontend_dist/` (bundled app) and falls back to `frontend/` if the bundle isn’t present.

## Implementation overview

### Backend services

The backend consists of three high‑level modules:

1. **Scanner** – Periodically scans the market (via Polygon) for new setups defined in `strategy_library.py`.  It computes indicators such as ATR, EMA, VWAP, and ADX using functions from `calculations.py`, filters the options chain for liquidity via `contract_selector.py`, and returns a list of candidate trades.  The scanning logic is asynchronous so that it can ingest WebSocket streams without blocking other requests.

2. **Follower** – Maintains a state machine for every subscribed trade.  It monitors price movements, recalculates stop‑loss and take‑profit levels using the ATR and supertrend/Chandelier exit rules, and sends updates to the user via ChatKit (or returns them via API).  See `follower.py` for an example implementation.

3. **ChatKit session endpoint** – Creates a ChatKit session using the OpenAI Python SDK.  The official docs provide a similar example using FastAPI【220893600390724†screenshot】.  We replicate that pattern here: the endpoint returns a `client_secret`, which the front‑end passes to the ChatKit component to authenticate the user.
4. **Agents runtime** – `src/agents_runtime.py` wraps the [openai-agents](https://github.com/openai/openai-agents-python) SDK to run a dedicated Trading Coach agent.  The `/api/agent/respond` route proxies prompts to this agent and returns conversational output plus widget-ready JSON payloads so the frontend can surface trade ideas alongside ChatKit.

### Frontend integration

The `frontend/index.html` file demonstrates how to embed ChatKit as a Web Component.  It loads the ChatKit script from OpenAI’s CDN, fetches a session token from your backend, and mounts the ChatKit UI into a `<div>` container.  Customizing the look, feel, and prompts of the chat is as simple as editing the attributes on the `<chat-kit>` element or overriding CSS variables.  Refer to the official guide to learn more about theming and actions【430922328384056†L146-L154】.

If you prefer to build a more complex interface (e.g. a dashboard with trade cards and charts), create a new Next.js app and install the `chatkit-js` library.  Then follow the steps in the ChatKit guide to set up the session endpoint, initialize the component in React, and handle events.  The relevant documentation is linked in the `docs/strategies_calculations.md` file.

## Next steps

The code provided here is a starting point.  To turn this into a production‑ready tool you should:

* Implement the scanning logic using Polygon’s WebSocket and REST APIs.  See the comments in `scanner.py` for guidance.
* Expand the strategy library with additional setups or adjust the thresholds for your own risk tolerance.  The library is defined in a declarative format so that adding new strategies does not require touching the scanning code.
* Connect to a database (e.g. PostgreSQL) to log trades, backtest results, and update the “A+” thresholds automatically.
* Harden the real‑time follower with reconnection logic, concurrency handling, and message queuing.
* Add authentication and user management if you plan to offer this as a hosted product.

When you are ready to deploy, you can push this repository to GitHub and set up continuous deployment to your preferred hosting provider (e.g. Railway or Vercel).  Remember to store API keys securely using environment variables or secrets management services.

## Generating the zip archive

To generate a zip file that you can upload directly to GitHub, run the following command from the root of the project:

```bash
zip -r project.zip . -x "*.env" -x "project.zip"
```

The `project.zip` file is included in this repository for your convenience.  It contains the entire `trading_bot/` directory tree except for secret files.  You can upload it to GitHub using the web interface or via the `gh` CLI.

---

If you have questions or would like help expanding this project, feel free to ask.  Happy building!
