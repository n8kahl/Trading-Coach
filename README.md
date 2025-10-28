# Trading Coach GPT Backend

_Latest deploy refresh: 2025‑10‑17 04:49 UTC_

A lightweight FastAPI service that prepares market data, trading plans, volatility context, and option contract picks for a custom GPT agent. The backend owns the quantitative plumbing so the GPT can focus on reasoning with traders.

> ⚠️ **Disclaimer:** All code and strategy examples are for educational purposes only. Nothing here is a recommendation to trade securities.

---

## Production Status

- Status: Production ready (current GitHub build)
- Branch: `main`
- Commit: `90fe80d`
- Host: `https://trading-coach-production.up.railway.app`

See `docs/production_readiness.md` for scope, validation checks, and verification steps.

---

## Current surface area

| Endpoint | Purpose | Notes |
| --- | --- | --- |
| `POST /gpt/scan` | Evaluate strategy playbooks (ORB retest, VWAP cluster, gap fill, midday fade, etc.) on any ticker list and return grounded plans (entry/stop/targets/confidence) plus overlays and indicators. | Shares the same Phase 3 metadata as `/gpt/plan` (probabilities, runner telemetry, composite scoring, guardrails). |
| `GET /gpt/context/{symbol}` | Stream the latest OHLCV bars + indicator series for a single interval. | Use when the GPT needs extra bars for bespoke analysis. |
| `POST /gpt/multi-context` | Fetch multiple intervals in one call (e.g., `["5m","1h","4h","1D"]`) and attach a volatility regime block (ATM IV, IV rank/percentile, HV20/60/120, IV↔HV ratio). | Responses are cached for 30 s per symbol+interval+lookback. |
| `POST /gpt/contracts` | Rank Tradier option contracts with liquidity gates (spread, Δ, DTE, OI) and compute scenario P/L using plan anchors (delta/gamma/vega/theta). | `risk_amount` (defaults $100) is used only for sizing projections; no budget filtering occurs. |
| `POST /gpt/chart-url` | Return a canonical `/tv` link (symbol, price targets, plan metadata only). | Canonicalises params; no session/market keys or inline levels when `FF_CHART_CANONICAL_V1=1`. |
| `GET /api/v1/gpt/chart-layers` | Fetch plan-bound levels/zones/annotations for a given `plan_id`. | Drives overlays for `/tv`; layers persisted with each generated plan. |
| `GET /tv` | Serves the TradingView Advanced UI when bundled; otherwise falls back to Lightweight Charts with EMA labels, white VWAP, plan bands, and overlay lines. | `scale_plan=auto` rescales historic plans to current price regimes. |

Support routes: `/tv-api/*` (Lightweight Charts datafeed), `/gpt/widgets/{kind}` (legacy dashboards), `/charts/html` (static renderer), `/api/v1/gpt/chart-layers` (plan overlays).

Scan responses expose additional actionability metrics (`entry_distance_pct`, `entry_distance_atr`, `bars_to_trigger`, `actionable_soon`) so clients can prioritise nearer-term setups.

Authentication is optional. Set `BACKEND_API_KEY` to require Bearer tokens; include `X-User-Id` to scope data per user.

---

## Feature flags

- `FF_CHART_CANONICAL_V1` – emit canonical `/tv` links and persist `plan_layers` with every plan.
- `FF_LAYERS_ENDPOINT` – expose `GET /api/v1/gpt/chart-layers` and require persisted plan overlays for rendering.
- `FF_OPTIONS_ALWAYS` – append deterministic `confluence_tags`, `tp_reasons`, and `options_contracts` (plus fallback `options_note`) to plan payloads, ensuring the GPT never fabricates contract picks.

### Persistence checks

- Set `DB_URL` or `DATABASE_URL` to your Railway Postgres connection string; startup logs should include `database connection pool initialised`.
- Run `psql "$DB_URL"` (or `psql "$DATABASE_URL"`) and inspect `idea_snapshots` after a `/gpt/plan` call.
- Hit `/gpt/plan` followed by `curl "$BASE_URL/api/v1/gpt/chart-layers?plan_id=..."` to confirm a 200 response with `levels`.
- Without a live database the service falls back to in-memory caching, and `/tv` overlays break after restarts.

---

## Scenario Plans (Market Replay)

- Live plan: one auto-updating plan per symbol (single source of truth).
- Scenarios: zero or more frozen snapshots per style (Scalp/Intraday/Swing; Reversal gated until server strategy exists).
- Adopt: promote a scenario to Live in the UI; optionally regenerate via `/gpt/plan` first.
- Charts: canonical `/tv` URLs only; no `session_*` or inline levels in links. Overlays fetched by `plan_id` via `GET /api/v1/gpt/chart-layers`.
  Plans also surface `confluence_tags`, multi-timeframe `confluence`, `key_levels_used`, `risk_block`, `execution_rules`, and (when `FF_OPTIONS_ALWAYS=1`) top-ranked `options_contracts` so the GPT can explain levels, risk, and contract selection without fabricating details.
  Access in `trade-coach-ui` under `/replay/:symbol`.

## Simulated Live Mode

Strategy engines can now evaluate setups even when the market session is closed. Opt in per request by passing `"simulate_open": true` in the JSON body (or the `X-Simulate-Open: 1` header). Responses remain deterministic—the `as_of` timestamp is unchanged—and include a banner such as `Simulated live — analysis as of 2024-03-02 16:00 ET`. Chart links behave like live scans (`live=1` + `last_update`), and both scan/plan payloads surface `meta.simulated_open = true` so clients can highlight the simulated context.

```bash
curl -X POST "$BASE_URL/gpt/scan" \
  -H "Content-Type: application/json" \
  -d '{
    "universe": ["TSLA", "NVDA", "META"],
    "style": "intraday",
    "limit": 10,
    "simulate_open": true
  }'
```

```bash
curl -X POST "$BASE_URL/gpt/plan" \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "TSLA",
    "style": "intraday",
    "simulate_open": true
  }'
```

Legacy clients can keep their payloads unchanged and simply add `X-Simulate-Open: 1` to trigger the simulated-live path.

---

## Project layout

```
.
├── README.md
├── requirements.txt
├── nixpacks.toml            # Railway build config
├── Procfile                 # uvicorn launch command
├── enrich_service.py        # Finnhub enrichment sidecar (sentiment/events/earnings)
├── docs/
│   ├── gpt_integration.md   # API schemas + GPT usage guide
│   └── progress.md          # Change log, roadmap, operational notes
└── src/
    ├── agent_server.py      # FastAPI app (all /gpt + /tv/* routes)
    ├── scanner.py           # Strategy evaluation engine
    ├── strategy_library.py  # Declarative strategy definitions
    ├── context_overlays.py  # Supply/demand, liquidity, volatility extras
    ├── tradier.py           # Tradier option helpers + caching
    ├── polygon_options.py   # Polygon option helpers (fallback only)
    ├── calculations.py      # Indicator utilities (EMA/ATR/VWAP/ADX…)
    └── config.py            # Settings loader (env vars, caching)
```

Static assets for the fallback chart renderer live in `static/tv/`.

---

## Local development

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn src.agent_server:app --reload --port 8000
# Optional sidecar for sentiment/events/earnings
# Recommended: run via the src.* module path
uvicorn src.enrich_service:app --reload --port 8081

On Railway, do not pass "$PORT" directly to uvicorn (the runner may not expand
it). Use the starter modules that read PORT from the environment:

- Main API: `python -m src.start_server`
- Enrichment sidecar: `python -m src.start_enrich`
- Finnhub sidecar: `python -m src.start_finnhub_sidecar`
```

Environment variables (read via Pydantic):

```
# .env (example)
POLYGON_API_KEY=pk_your_key              # Required for premium Polygon data (optional)
BACKEND_API_KEY=super-secret             # Omit for anonymous access during dev
BASE_URL=https://<your-app>.up.railway.app/tv
FINNHUB_API_KEY=your_finnhub_key         # Required for enrich_service.py
TE_API_KEY=replace_me_key:replace_me_secret  # Required Trading Economics primary macro events source (key:secret format)
ENRICH_SERVICE_URL=http://localhost:8081 # Override if deploying enrichment elsewhere
DB_URL=postgresql://user:pass@host:5432/dbname  # Optional; enables persistent idea snapshots (alias: DATABASE_URL)
SELF_API_BASE_URL=https://trading-coach-production.up.railway.app  # Used for auto-replan callbacks
PUBLIC_BASE_URL=https://trading-coach-production.up.railway.app    # Absolute base used for link unfurls (/tv)

# Tradier (sandbox defaults shown)
TRADIER_SANDBOX_TOKEN=XXXXXXXXXXXX
TRADIER_SANDBOX_ACCOUNT=VA0000000000
TRADIER_SANDBOX_BASE_URL=https://sandbox.tradier.com
```

The enrichment sidecar continues to source earnings and sentiment from Finnhub. Trading Economics
is the primary macro events provider when `TE_API_KEY` is configured; Finnhub serves as the
fallback. The health probe still pings Finnhub's news-sentiment endpoint.

Use `pytest` to run the current unit test suite (indicator maths only right now).

---

## Live plan console (Next.js)

The new coach UI lives in [`trade-coach-ui/`](trade-coach-ui/) and ships a reactive Next.js experience with Polygon streaming, plan deltas, and coaching timelines.

```
cd trade-coach-ui
npm install
npm run dev
```

Environment variables (optional):

```
NEXT_PUBLIC_API_BASE_URL=https://trading-coach-production.up.railway.app
NEXT_PUBLIC_WS_BASE_URL=wss://trading-coach-production.up.railway.app
# Only set the API key if your backend requires it—never expose secrets in client builds
NEXT_PUBLIC_BACKEND_API_KEY=optional-bearer-token
```

The root `package.json` exposes helper scripts:

```
npm run dev        # trade-coach-ui dev server
npm run build      # build the live plan console bundle
npm run start      # start the live plan console (Next.js) on $PORT
```

`trade-coach-ui` renders live plan data by calling `/idea/{plan_id}` for the initial snapshot, `/ws/plans/{plan_id}` for coaching deltas, and `/stream/{symbol}` for price ticks. As you extend the TradeFollower, the UI will automatically surface trail-stop adjustments and auto-replan events.

`/tv` is the canonical interactive viewer for plans and scans. Legacy Idea Page bundles have been removed; continue using the `/gpt/chart-url` surface to build shareable `/tv` links.

---

## Architecture primer

- **FastAPI app (`src/agent_server.py`)** – hosts GPT endpoints, the `/tv` viewer, and a `/tv-api` datafeed. Uses async helpers for I/O.
- **Market data** – Polygon aggregates with Yahoo fallback for OHLCV; Tradier chains & quotes (batched + cached for 15 s) provide option prices, greeks, and IV. Polygon option snapshots are best-effort; failures simply fall back to Tradier.
- **Strategy engine (`scanner.py`)** – builds real trade plans from intraday context (anchored VWAPs, ATR, EMA alignment, breakout checks). Plans include direction, entry, stop, target ladder, confidence, ATR, notes, R:R, and overlays.
- **Volatility metrics** – `_compute_iv_metrics` in `agent_server.py` caches ATM IV, IV rank/percentile, and HV20/60/120 for 2 minutes per symbol.
- **Chart renderer (`static/tv`)** – Lightweight Charts fallback draws labelled EMAs, white VWAP, plan lines, supply/demand bands, liquidity pools, fair value gaps, and anchored VWAPs. Autoscale + plan rescaling keep everything on-screen.
- **Realtime bars** – Polygon websocket stream keeps SPX/NDX charts ticking between full plan refreshes (1–2 minute cadence). If Polygon drops, the server falls back to proxy-translated bars and banners the chart as degraded.
- **Caching** – Multi-context responses (30 s), IV metrics (120 s), Tradier chains/quotes (15 s). All caches are in-memory; restart clears them.

## Index sniper mode (SPX/NDX)

- Enable via `INDEX_SNIPER_MODE=1` to prioritize SPX/NDX structure while automatically substituting SPY/QQQ contracts when index feeds degrade.
- Polygon index snapshots (`/v3/snapshot/options/I:SPX|I:NDX`) are used first; Tradier greeks (requires the ORATS entitlement) are the secondary source.
- When both feeds are degraded the planner falls back to the ETF proxy using a rolling 60-minute gamma regression to translate entry/stop/target levels (γ, ratio, drift exposed in `execution_proxy`).
- Plans include a banner (`fallback_banner`) and settlement note to flag American-style ETF contracts; charts receive the same metadata for the `/tv` renderer.

---

## Wiring a custom GPT (Actions)

1. In GPT Builder, add a new Action pointing at your deployment (e.g. `https://<railway-app>.up.railway.app`).
2. For the schema URL, use `https://<host>/openapi.json`. FastAPI serves the latest OpenAPI document.
3. Configure auth:  
   - If `BACKEND_API_KEY` is set, add a Bearer token and send `X-User-Id` on each call.  
   - Otherwise leave headers empty; the backend will scope data to `anonymous`.
4. Suggested GPT instructions:
   ```
   - Start every analysis with /gpt/scan.
   - Pull additional bars with /gpt/context or /gpt/multi-context.
   - Use /gpt/contracts with plan anchors to source option ideas.
- Generate shareable charts using /gpt/chart-url and present the returned /tv link (plan overlays fetched by `plan_id`).
   ```

See `docs/gpt_integration.md` for full schemas and sample payloads.

---

## Operational notes & troubleshooting

| Symptom | Cause / mitigation |
| --- | --- |
| `Polygon option snapshot failed ... 400 Bad Request` spam | The account tied to `POLYGON_API_KEY` lacks the options snapshot entitlement. These warnings are expected; the service automatically falls back to Tradier quotes. |
| `/gpt/multi-context` returns 400 | At least one interval token is invalid. Use tokens such as `1m`, `5m`, `15m`, `1h`, `4h`, `1D`. Duplicates are ignored silently. |
| `/gpt/contracts` returns empty `best` | Liquidity filters removed everything. The service widens Δ by ±0.05 and DTE by ±2 once each; if it still returns empty there genuinely isn’t a liquid contract under the constraints. |
| Chart missing plan bands | Ensure you pass `entry`, `stop`, and `tp` (comma separated) when calling `/gpt/chart-url`. The GPT should forward the plan payload (including `plan_id`) from `/gpt/scan`. |
| Plan URLs stop working after restart | Configure `DB_URL`/`DATABASE_URL` with your Railway Postgres connection so plan/idea snapshots persist across deploys; otherwise the in-memory cache resets. |
| `/tv` renders a blank chart | Confirm you are using the canonical link emitted by `/gpt/plan` (includes `plan_id` + `plan_version`) and review the viewer data flow in `docs/tv_data_flow.md`. Hard-refresh to pull the latest `tradingview-init.js`. |

Deployment is currently handled by Railway (`nixpacks.toml` + `Procfile`). Logs will show cache hits (`cached=true`) and option snapshot warnings.

---

## Roadmap & hand-off

The detailed progress log and backlog live in [`docs/progress.md`](docs/progress.md). Highlights:

- **Charts:** Add `focus=plan` zooming and upgrade to TradingView overlays once the Advanced bundle is packaged.
- **Contracts:** Improve IV rank by incorporating historical IV time series; expose skew/term slope.
- **Scanning:** Persist user watchlists / plan history; expose playbook metadata for GPT explanation generation.
- **Ops:** Optionally suppress Polygon snapshot warnings when entitlement is missing; consider Redis-based caching for multi-instance deployments.
- **Tests:** Expand beyond indicator math—add API contract tests and regression coverage for scanner heuristics.

Refer to the progress doc for the active TODO list, open questions, and a changelog of recent enhancements.

---

## Additional references

- [`docs/gpt_integration.md`](docs/gpt_integration.md) – API schemas, GPT request examples, and contract/multi-context usage notes.
- [`docs/progress.md`](docs/progress.md) – Chronological change log, roadmap, operational caveats.
- [`static/tv/`](static/tv/) – Fallback chart renderer assets.
- [`tests/`](tests/) – Current unit test coverage.

Happy building—and trade responsibly.
