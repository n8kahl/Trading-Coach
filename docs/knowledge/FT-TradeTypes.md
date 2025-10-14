# FT-TradeTypes (Delta • DTE • Spread Guardrails)

Defines liquidity presets for option selection by style and regime.

| Style | Target Δ | DTE Window | Max Spread % | Notes |
| --- | --- | --- | --- | --- |
| `scalp` | 0.55 ± 0.05 | 0–3d | 0.08 | Prefer defined-risk when context score < 0.45 |
| `intraday` | 0.50 ± 0.05 | 0–7d | 0.10 | Switch to debit spreads if spread_pct > 0.08 |
| `swing` | 0.45 ± 0.05 | 7–35d | 0.12 | Use higher DTE when volatility regime elevated |
| `leaps` | 0.35 ± 0.05 | 35–90d | 0.15 | Allow wider spreads when IV rank < 0.3 |

Adjustments by regime:

- `volatility_regime >= 0.7` → widen DTE by +5d and cap delta at lower bound.
- `volatility_regime <= 0.3` → allow slightly tighter spreads (−0.02 cap).

The options selector should encode these presets, and the GPT references this document when explaining contract choices.
