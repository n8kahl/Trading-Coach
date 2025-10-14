# FT-Context (Macro • Sector • Internals)

**Purpose:** Provide the GPT assistant with a single, compact signal describing external market conditions (macro events, sector strength, internals breadth). The server converts this into the `context` block attached to each setup.

## Inputs

- `macro.get_event_window(as_of)` — fetches FOMC/CPI/NFP countdowns from the enrichment sidecar (Finnhub data) and highlights anything inside the next four hours.
- `sector.sector_strength(symbol, as_of)` — computes daily percentage change of the mapped sector ETF versus SPY using Polygon daily aggregates.
- `sector.peer_rel_strength(symbol, as_of)` — compares the symbol’s daily change with SPY to generate a relative-strength score.
- `internals.market_internals(as_of)` — pulls Polygon advancer/decliner snapshots and the latest VIX close to approximate breadth, VIX, and a TICK-style reading.

All timestamps are normalised to ISO-8601 with timezone information.

## Summary Text

`context.macro` should be a short deterministic sentence (e.g., “FOMC in 45m” or “No high-impact events within 4h.”).

## Usage in Output

Every setup includes:

```json
"context": {
  "macro": "FOMC in 45m",
  "sector": {"name": "XLK", "rel_vs_spy": 0.32, "z": 1.1},
  "internals": {"breadth": 1820, "vix": 14.2, "tick": 820},
  "rs": {"vs_benchmark": 1.06}
}
```

The GPT must reference this block when explaining confidence, event risk, or sector emphasis.
