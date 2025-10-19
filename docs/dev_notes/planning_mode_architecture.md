# Planning Mode Evening/Weekend Scan Architecture

**Status:** Draft implementation plan  
**Intent:** Produce high-probability setups off-hours using Polygon data only (no pre-computed lists, no fabrication).

## Objectives
- Recompute liquid universes dynamically every run (S&P 500, NASDAQ 100, watchlist) with Polygon constituents or market-cap top lists.
- In planning mode, run price-structure readiness scoring off latest Polygon aggregates, without emitting option symbols until live quotes are available.
- Persist evening scans for replay/backtests and finalize contracts at the next market open.

## Service Layout
```
src/
  services/
    universe.py          # tiered universe provider (constituents → market-cap → watchlist)
    polygon_client.py    # async aggregates client (rate-limited, cached, resilient)
    scan_engine.py       # orchestrates evening scan (indices regime + readiness scoring)
    contract_rules.py    # rules-based option templates (delta/DTE bands per style)
    persist.py           # DB writes for runs, candidates, finalizations
  planning/
    planning_scan.py     # callable entry-point for planning mode scans
  routers/
    gpt.py               # exposes /gpt/scan (planning_mode) + /gpt/finalize
```

## Data Flow
1. **Universe expansion** (`UniverseProvider`):
   - Tier A: Polygon index constituents (SPX/NDX).  
   - Tier B: Polygon reference tickers sorted by market cap.  
   - Tier C: Configured watchlist.  
   - Cache TTL ≈ 12h; persist snapshot in `evening_scan_universe_snapshots`.
2. **Aggregates fetch** (`AggClient`):
   - Daily + 60/30-min bars for equities and indices.  
   - Async semaphore, backoff, short Redis TTL (5–15 min).  
   - Returns dict[symbol] → {daily, m30, m60}.
3. **Readiness scoring** (`scan_engine`):
   - Compute trend regime (EMA stack), ATR%, pullback/extension, structure.  
   - Risk gating from indices (I:SPX, I:NDX, I:RUT, I:VIX).  
   - Score = 0.45·probability + 0.25·actionability + 0.30·risk_reward (store components).
4. **Persistence** (`persist`):
   - Tables  
     - `evening_scan_runs(id, as_of_utc, universe_name, universe_source, tickers, indices_context, data_windows, notes)`  
     - `evening_candidates(id, scan_id, symbol, metrics, levels, readiness_score, contract_template, requires_live_confirmation, missing_live_inputs)`  
     - `plan_finalizations(id, candidate_id, finalized_at, live_inputs, selected_contracts, status, reject_reason)`
5. **Contract templates** (`contract_rules`):
   - Intraday: delta 0.30–0.40, DTE 0–3.  
   - Swing: delta 0.20–0.35, DTE 5–15.  
   - Leaps: delta 0.60–0.75, DTE 6–12 mo.  
   - Liquidity guardrails: OI ≥ threshold, spread_pct ≤ threshold, etc.  
   - Output structured template + `requires_live_confirmation=true`.
6. **Finalizer** (`/gpt/finalize` or scheduled job):
   - Pull live chains, snap templates to contracts if delta/DTE/LIQ/ spread gates pass.  
   - Update `plan_finalizations` with status (`finalized`, `rejected`, `deferred`).  
   - Propagate to `/gpt/plan` responses (options_contracts or options_note).

## Planning Mode Response Contract
```
{
  "symbol": "AAPL",
  "planning_mode": true,
  "readiness_score": 0.72,
  "components": {"probability":0.78,"actionability":0.65,"risk_reward":0.70},
  "levels": {"entry":223.4,"invalidation":221.9,"targets":[225.8,227.4]},
  "contract_template": {
    "style":"intraday","type":"CALL",
    "delta_range":[0.30,0.40],"dte_range":[0,3],
    "min_oi":500,"max_spread_pct":4.0
  },
  "requires_live_confirmation": true,
  "missing_live_inputs": ["iv","spread","oi"]
}
```

## Testing / Operations
- **Unit tests**: Universe provider fallback; scoring determinism; contract template generation; finalizer gating.
- **Integration tests**: Planning scan with recorded Polygon fixtures; finalizer with sample chains.
- **Cron**:  
  - `evening_scan` (e.g., 22:00 CT) → run `planning_scan`.  
  - `preopen_finalize` (08:15 ET) → finalize templates; retry at open if data missing.

## Next Steps
- Implement modules above and wire into `/gpt/scan` planning-mode branch.  
- Update OpenAPI (`ScanRequest.planning_mode`, planning metadata).  
- Extend UI to show “Planning Mode — requires live confirmation” badge and finalization status.
