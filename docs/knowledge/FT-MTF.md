# FT-MTF (Multi-Timeframe Confluence)

Defines how 5m → 1D frames are evaluated and surfaced.

## Frame Schema

```
{
  "trend": "bullish|bearish|sideways",
  "ema_state": "9>20>50" (or similar stack),
  "structure": "higher-low" / "lower-high" / "range",
  "momentum": "ADX 24 rising" / "Momentum cooling" / etc.
}
```

The server returns a dictionary of frames (5m, 15m, 1h, 4h, 1D). When only HTF bias is available, populate `_1h` with the broader bias so the assistant still has deterministic data.

## Snap Rules

- Target snapping prioritises HTF structure → volume profile → Fib levels.
- Stops trail HTF swing points when the frame shows alignment (e.g., 1h higher-low).
- Runner anchor inherits the highest timeframe that aligns with the trade direction.

## Scoring

Feed the MTF bias into `trend_alignment` component (1.0 when HTF bias matches direction, 0.5 baseline otherwise).

## GPT Usage

- Include concise references (e.g., "5m trend bullish; 1h neutral"), not raw datasets.
- `mtf_analysis` attaches to each setup for downstream UI cards.
