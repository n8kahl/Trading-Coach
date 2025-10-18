# `/tv` Plan Viewer Data Flow

This note traces how a canonical plan URL such as

```
https://trading-coach-production.up.railway.app/tv/?symbol=META&plan_id=meta-intraday-long-2025-10&plan_version=4&entry=725.49&stop=723.61&tp=727.75%2C729.05%2C730.75&interval=5m&range=15d&focus=plan&center_time=latest&theme=dark&scale_plan=auto
```

turns into the rendered chart and overlays that users see in the webview.

## High-level sequence

1. **Static shell**  
   `static/tv/index.html` serves the markup, styles, and two scripts:
   - Lightweight Charts (`lightweight-charts.standalone.production.js`)
   - The viewer bootstrap (`static/tv/tradingview-init.js`)

2. **Bootstrap & URL normalisation** (`tradingview-init.js`:40-338)  
   - Parse query-string for plan + chart metadata.
   - Derive the canonical symbol (prefers `symbol`, falls back to `plan_meta` or the `plan_id` slug).
   - When a `plan_id` is present, load plan layers via `GET /api/v1/gpt/chart-layers`.
   - Merge URL params with any embedded `plan_meta`, then seed `basePlan`/`currentPlan`.
   - Normalise feature flags (follow-live, “show more levels”) and set the initial UI state.

3. **Plan snapshot enrichment** (`tradingview-init.js`:1420-2310)  
   - If an SSE `plan_full` payload arrives, replace `currentPlan`/`mergedPlanMeta`.
   - Auto-adopts regenerated plans when the stream signals a newer `plan_id` and the viewer has opted in.
   - Renders the plan panel (targets, checklist, confluence, statistics).

4. **Market data fetch** (`tradingview-init.js`:3031-3332)  
   - `fetchBars()` builds a `/tv-api/bars` request using the resolved symbol, resolution, and time window.
   - Responses are the TradingView-compatible payloads emitted from `src/agent_server.py:3382-3576`.
   - On success, the viewer:
     - Sets candlestick + histogram data.
     - Recomputes EMA/VWAP overlays and key levels.
     - Applies plan-driven price lines (entry, stops, targets, runner anchor).
     - Recentres/zooms the chart according to `focus`, `center_time`, and `range` query params.
   - On failure (`s != "ok"` or network errors) the viewer shows a debug banner and keeps the last known data.

5. **Streaming updates** (`tradingview-init.js`:1426-1514)  
   - Establishes an `EventSource` to `/stream/{symbol}`:
     - `tick` / `bar` → update last price + heartbeat.
     - `plan_state` / `plan_delta` / `plan_full` → adjust plan status, notes, or adopt a new plan.
     - `market_status` → update market-phase pill and status note.

6. **Auto-refresh cadence**  
   - `fetchBars()` runs immediately and every `TIMEFRAME_REFRESH_MS` (default 60 s).
   - Streaming heartbeat fallback keeps the “Streaming Data” pill updated even without ticks.

## Back-end responsibilities

| Endpoint | Path | Purpose | Key refs |
| --- | --- | --- | --- |
| Idea snapshot | `/idea/{plan_id}` | Load persisted plan snapshot (serves the plan panel). | `src/agent_server.py:6083-6130` |
| Plan layers | `/api/v1/gpt/chart-layers` | Return persisted `plan_layers` for overlays. | `src/agent_server.py:7242-7346` |
| Market bars | `/tv-api/bars` | Supply OHLCV payloads for the chart. | `src/agent_server.py:3382-3576` |
| Streams | `/stream/{symbol}` | SSE fan-out for live prices + plan deltas. | `src/agent_server.py:1836-1884` |

Market data comes from `src/agent_server.py:_load_remote_ohlcv()`, which tries Polygon first, falls back to Yahoo if Polygon is stale/missing, and ultimately returns cached Polygon data with a “stale” flag so the front-end can surface the degraded state.

## Failure modes & mitigations

| Symptom | Likely cause | Mitigation |
| --- | --- | --- |
| Viewer throws `ReferenceError: Cannot access 'levelsToggleEl' before initialization` and stops bootstrapping. | Cached pre-20251118 bundle exposed a TDZ bug. | Hard-refresh (cache bust via `?v=20251118`); bundle fix shipped in `static/tv/tradingview-init.js` (Git commit `f472bcd`). |
| Debug banner: `Error loading data: No data` | `/tv-api/bars` returned `s="no_data"` (symbol/interval mismatch or upstream outage). | Inspect server logs near `tv-api/bars` request, confirm Polygon/Yahoo health, or retry with a coarser interval (viewer now falls back automatically). |
| Plan panel empty | URL lacked `plan_id`/`plan_version`, so viewer could not hydrate metadata. | Always use canonical links from `/gpt/plan` (`trade_detail` or `/gpt/chart-url` response) which include the slug + version. |
| Streaming pill stays “Idle” | SSE to `/stream/{symbol}` blocked or API key missing. | Check network console for 401/403/5xx on `/stream`; ensure `BACKEND_API_KEY` is set and forwarded when required. |

## Testing checklist

1. `pytest` — full suite covers plan serialization and DB persistence (`tests/test_chart_layers_api.py`, `tests/test_config.py`).
2. Manual:
   - `curl "$HOST/gpt/plan"` → ensure `plan_layers` present in the response and `/api/v1/gpt/chart-layers` returns 200.
   - `curl "$HOST/tv-api/bars?symbol=AAPL&resolution=5"` → expect `{"s":"ok", ...}` with recent timestamps.
   - Hard refresh `/tv/?plan_id=...` → verify candlesticks render, plan panel populated, “Streaming Data” pill toggles to green after the first heartbeat.
3. Production log watch:
   - Confirm startup logs include `database connection pool initialised`.
   - Observe `tv-api/bars OK ... bars=...` entries when the viewer loads.
   - Monitor for `Polygon data is stale` warnings; if persistent, investigate Polygon API health.

Keep this doc current when tweaking either the viewer or `/tv-api` surface; it is the canonical reference for how trade plans become charted webviews.
