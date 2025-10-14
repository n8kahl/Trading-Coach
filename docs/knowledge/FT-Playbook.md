# FT-Playbook (Setups & Probability Components)

Deterministic library describing how each core setup is scored and validated. Use together with `FT-MTF`, `FT-Context`, and `FT-Risk`.

## Probability Components Mapping

| Component | Source | Notes |
| --- | --- | --- |
| `trend_alignment` | MTF bias + structure | Higher when direction matches HTF bias and structure (higher-high / higher-low for longs, etc.). |
| `liquidity_structure` | Key levels + confluence | Reward overlapping levels (VWAP, POC, supply/demand) and valid invalidation bands. |
| `momentum_signal` | Intraday momentum cues | Uses recent ADX/EMA stack state, squeeze flags, and volume pacing. |
| `volatility_regime` | Expected move + ATR | Down-weights trades when volatility regime is extreme relative to style.

Each component returns `[0, 1]`. Pass the dict to `overall_confidence(components, context_score)`.

## Confidence Rule

```
confidence = clamp(0.6*trend_alignment + 0.2*liquidity_structure + 0.2*volatility_regime + adjustment)
adjustment = clamp(context_score - 0.5, -0.05, 0.05)
```

## Trade Quality

Use `quality_grade(confidence)` to convert to the public-facing grade: `A+` ≥ 0.85, `A` ≥ 0.78, …, `D` otherwise.

## Practical Notes

- Keep component scoring deterministic per strategy.
- Snap targets to HTF structure (see `FT-MTF`).
- When context is risk-off, confidence falls but never by more than ±0.05.
- Feed final `confidence`, `probability_components`, and `trade_quality_score` to the API response.
