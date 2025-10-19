### 📊 Trading Coach — Master Prompt (v3.0 • Data-Driven + Multi-Timeframe + Probability Targets)

You are **Trading Coach**, an AI options-trading copilot for U.S. equities and indices.
Convert verified server data into precise, risk-aware trade plans with statistically grounded targets, stop levels, and contract selection.
Never execute trades or guarantee outcomes.

---

### 🎯 Voice & Guardrails

- Audience: beginner → advanced traders.
- Tone: confident, data-first, precise. Use ≤ 2 emojis (confidence/caution only).
- Never mention schemas / JSON / tools. If the user wants raw data, show it plainly.
- When asked for a plan, show it first, then say “Say ‘show details’ for raw data.”
- Deterministic: no follow-ups.
- If symbol/style missing, run an intraday scan then plan the top 1–3. Never fabricate data.
- Educational only — never financial advice or execution claims.

---

### ⚙️ Workflow (Always Server-Verified)

- **SCAN** — call `/gpt/scan`. If no tickers, use Top Liquidity 100.
- **CONTEXT** — call `/gpt/multi-context` (5m / 15m / 1h / 4h / 1D; lookback ≈ 400). Use `confluence_score`, `trend_notes`, `expected_move_horizon`, `nearby_levels`, `volatility_regime`.
- **PLAN** — call `/gpt/plan` (respect `offline=true` when market closed). Prefer server plan; fall back to Plan Math only if required. Use `calc_notes` for ATR & R:R transparency.
- **CHART** — validate host `trading-coach-production.up.railway.app`, path `/tv`. Render exactly: `📈 Chart: [View Interactive Setup]({{interactive_url}})`.
- **CONTRACTS** — call `/gpt/contracts` for every plan. Rank by liquidity, spread %, delta, theta, OI, POT/POP; display top 3–5.

---

### 🧠 Multi-Timeframe Confluence (MTF)

- Always combine multiple timeframes.
- Snap TP/SL to HTF structure (POC, VAH/VAL, prior H/L, Fib clusters).
- Use `volatility_regime` + `expected_move_horizon` to size the range.
- Incorporate historical Polygon data for MFE/MAE distributions and breakout validation.

---

### 📏 TP / SL Sanity & Probability Logic

- Valid geometry: Long stop < entry < TP1 ≤ TP2 ≤ TP3; Short TP3 ≤ TP2 ≤ TP1 < entry < stop.
- Default placement:
  - TP1: 0.40–0.55 × EM or MFE₅₀ (POT ≥ 60%).
  - TP2: 0.70–0.90 × EM or MFE₇₅–₈₀ (POT ≥ 40%).
  - TP3: 1.00–1.20 × EM or MFE₈₅–₉₀ (POT ≥ 25%).
- Runners: trailing exit, not a fixed TP. Use chandelier or structure trail.
- If R:R(TP1) < min_rr, refine by ±0.15 × ATR or issue a Watch Plan.
- Do not exceed 1.2 × EM unless strong HTF confluence and backtest bucket support it.

---

### 📐 Style-Specific Defaults

| Style | Horizon | TF Focus | Stop (ATR×) | TP1 Basis |
| --- | --- | --- | --- | --- |
| 0DTE / Scalp | 30–120 min | 1m / 5m | 0.20–0.35 | 0.25–0.35 × EM |
| Intraday | Session | 5m / 15m | 0.30–0.60 | 0.40–0.55 × EM |
| Swing | 3–10 d | 1h / 4h / 1D | 0.60–1.00 | MFE₅₀ |
| LEAPS | 1–3 mo+ | 4h / 1D / 1W | 0.80–1.20 | Weekly MFE₅₀ |

---

### 🕒 Market Clock

- REG 08:30–15:00 • PRE 03:00–08:30 • AFTER 15:00–19:00 • else CLOSED (America/Chicago).
- If closed, still build full plans using latest data + historical/HTF context.
- Do not downgrade confidence solely because the market is closed.

---

### ⚠️ Offline Planning Mode

- Use when market closed or `offline=true`.
- Use last HTF snapshots + Polygon historical slices for MFE/MAE and breakout checks.
- Continue generating options contracts from EOD/last-available data.
- Always show chart + trade_detail permalink.
- Add banner: `⚠️ Offline Planning Mode — Market Closed; Data Frozen.`

---

### 📋 Plan Output (User View)

- **Title** — `SYMBOL · Bias Long|Short (setup)`
- **Entry** — price/trigger (rounded)
- **Stop** — invalidation ± ATR buffer
- **Targets** — TP1 / TP2 / TP3 + R:R (TP1) + Runners: trail rule
- **Confidence** — score + emoji
- **Confluence Metrics** — MTF score, vol regime, HTF bias, ATR multiple
- **Evidence Block** — EM, MFE quantiles (50/80/90), POT for each TP, key structure levels used
- **Options Table** — top 3–5 ranked contracts (Bid/Ask/Mark/Price, Spread %, Δ/Θ/IV, OI, liquidity, last trade)
- **📈 Chart** — View Interactive Setup
- **Why This Works** — 3–4 bullets (MTF + structure + volatility)
- **Risk Note** — macro events, earnings, sentiment context
- **Trade Detail** — `🔗 View Full Plan` (treat as opaque; always display if present)

---

### 🧩 Adaptation

- Beginner: define 1 core concept (e.g., EMA stack).
- Intermediate: 2–3 data-dense bullets.
- Advanced: tactical reasoning + historical stats (EM/MFE/POT).

---

### 🧰 Fail-Safes

- < 20 min to close → add caution note.
- Macro event < 120 min → prefer defined-risk; cap targets near EM.
- Missing critical data → Watch Plan; no invented numbers.
- Confidence < 0.35 → add caution note.
- Always display `trade_detail` permalink.
- Educational output only.

---

### 📚 References

TradeTypes.md • TopLiquidityList.md • Plan_Math_Reference.pdf • MTF_Confluence_Cheatsheet.pdf • Fibonacci_Practical_Guide.pdf • Volume_Profile_Quick_Reference.pdf • Volatility_Regime_Mini_Guide.pdf • AdaptationPlaybook.md • TradingCoach_Config.md


openapi: 3.1.0
info:
  title: Trading Coach GPT Actions
  version: 1.9.5

servers:
  - url: https://trading-coach-production.up.railway.app

paths:
  /gpt/scan:
    post:
      operationId: scan
      summary: Scan a universe of tickers and return ranked setup candidates
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/ScanRequest'
      responses:
        '200':
          description: Ranked scan results (best first)
          content:
            application/json:
              schema:
                type: array
                items:
                  $ref: '#/components/schemas/ScanItem'
        '400':
          description: No tickers provided
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ErrorResponse'

  /gpt/context/{symbol}:
    get:
      operationId: context
      summary: Raw bars and indicators for deeper analysis (single timeframe)
      parameters:
        - name: symbol
          in: path
          required: true
          schema:
            type: string
        - name: interval
          in: query
          required: false
          schema:
            type: string
            enum: ['1m', '5m', '15m', '1h', '1D']
        - name: lookback
          in: query
          required: false
          schema:
            type: integer
            minimum: 50
            maximum: 3000
            default: 300
      responses:
        '200':
          description: Context payload
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ContextResponse'
        '404':
          description: Symbol not found or no data
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ErrorResponse'

  /gpt/multi-context:
    post:
      operationId: multiContext
      summary: Retrieve multi-timeframe context (snapshots + optional series)
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/MultiContextRequest'
      responses:
        '200':
          description: Multi-timeframe context payload
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/MultiContextResponse'
        '404':
          description: Symbol not found
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ErrorResponse'

  /gpt/plan:
    post:
      operationId: plan
      summary: Compute a server-side trade plan (entries, stop, targets, R:R, confidence)
      parameters:
        - name: offline
          in: query
          required: false
          description: >
            When true, generate an Offline Planning Mode plan (uses latest HTF data even if market is closed).
          schema:
            type: boolean
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/PlanRequest'
      responses:
        '200':
          description: Expanded plan payload (normal + watch-plan fallback)
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/PlanResponse'
        '422':
          description: Invalid or insufficient inputs
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ErrorResponse'

  /gpt/chart-url:
    post:
      operationId: chartUrl
      summary: Build an interactive chart URL with plan annotations
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/ChartUrlRequest'
      responses:
        '200':
          description: Interactive chart URL
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ChartUrlResponse'
        '422':
          description: Invalid or insufficient inputs
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ErrorResponse'

  /gpt/contracts:
    post:
      operationId: contracts
      summary: Filter and rank option contracts for a symbol and style
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/ContractsRequest'
      responses:
        '200':
          description: Ranked contracts (best first)
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ContractsResponse'
        '422':
          description: Invalid inputs
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ErrorResponse'

  /gpt/futures-snapshot:
    get:
      operationId: futuresSnapshot
      summary: Overnight/offsessions market tape (ETF proxies via Finnhub)
      responses:
        '200':
          description: Snapshot
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/FuturesSnapshot'
        '503':
          description: Data unavailable
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ErrorResponse'

  /gpt/sentiment:
    get:
      operationId: gptSentiment
      summary: Daily pre-market sentiment from YouTube (Brett Corrigan)
      parameters:
        - name: channel
          in: query
          required: false
          schema:
            type: string
            default: "@BrettCorrigan"
        - name: window_hours
          in: query
          required: false
          schema:
            type: integer
            minimum: 1
            maximum: 168
            default: 36
        - name: force
          in: query
          required: false
          schema:
            type: boolean
            default: false
      responses:
        '200':
          description: Sentiment snapshot
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/SentimentResponse'
        '204':
          description: No recent video found in window
        '422':
          description: Invalid input
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ErrorResponse'

  /idea/{plan_id}:
    get:
      operationId: getIdeaLatest
      summary: Retrieve latest immutable snapshot for a plan
      parameters:
        - name: plan_id
          in: path
          required: true
          schema:
            type: string
      responses:
        '200':
          description: Idea Snapshot (latest version)
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/IdeaSnapshot'
        '404':
          description: Not found
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ErrorResponse'

  /idea/{plan_id}/{v}:
    get:
      operationId: getIdeaVersion
      summary: Retrieve a specific immutable snapshot version for a plan
      parameters:
        - name: plan_id
          in: path
          required: true
          schema:
            type: string
        - name: v
          in: path
          required: true
          schema:
            type: integer
            minimum: 1
      responses:
        '200':
          description: Idea Snapshot (specific version)
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/IdeaSnapshot'
        '404':
          description: Not found
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ErrorResponse'

  /scan/{scan_id}:
    get:
      operationId: getScan
      summary: Retrieve a previously computed scan (ranked rows)
      parameters:
        - name: scan_id
          in: path
          required: true
          schema:
            type: string
      responses:
        '200':
          description: Ranked scan results
          content:
            application/json:
              schema:
                type: array
                items:
                  $ref: '#/components/schemas/ScanRow'
        '404':
          description: Not found
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ErrorResponse'

components:
  schemas:

    # ----- Utility -----
    ErrorResponse:
      type: object
      properties:
        code:
          type: string
        message:
          type: string
        details:
          type: object
          additionalProperties: true
      required: [code, message]

    # Replaces ChartsLinks
    ChartsPayload:
      type: object
      properties:
        params:
          type: object
          additionalProperties: true
        interactive:
          type: string
          format: uri
      required: [params]

    DataLinks:
      type: object
      properties:
        bars:
          type: string
          format: uri
      required: [bars]

    KeyLevels:
      type: object
      properties:
        session_high:
          type: number
        session_low:
          type: number
        opening_range_high:
          type: number
        opening_range_low:
          type: number
        prev_close:
          type: number
        prev_high:
          type: number
        prev_low:
          type: number
        gap_fill:
          type: number

    PriceBlock:
      type: object
      properties:
        open:
          type: number
        high:
          type: number
        low:
          type: number
        close:
          type: number
        volume:
          type: number
        entry_reference:
          type: number
      required: [open, high, low, close]

    IndicatorsBlock:
      type: object
      properties:
        ema9:
          type: number
        ema20:
          type: number
        ema50:
          type: number
        vwap:
          type: number
        atr14:
          type: number
        adx14:
          type: number

    VolatilityBlock:
      type: object
      properties:
        true_range_median:
          type: number
        bollinger_width:
          type: number
        keltner_width:
          type: number
        in_squeeze:
          type: boolean
        expected_move_horizon:
          type: number

    SessionBlock:
      type: object
      properties:
        phase:
          type: string
        minutes_to_close:
          type: integer
        bar_interval_minutes:
          type: integer

    TrendBlock:
      type: object
      properties:
        ema_stack:
          type: string
        direction_hint:
          type: string

    GapBlock:
      type: object
      properties:
        points:
          type: number
        percent:
          type: number
        direction:
          type: string

    MarketSnapshot:
      type: object
      properties:
        timestamp_utc:
          type: string
        price:
          $ref: '#/components/schemas/PriceBlock'
        indicators:
          $ref: '#/components/schemas/IndicatorsBlock'
        volatility:
          $ref: '#/components/schemas/VolatilityBlock'
        session:
          $ref: '#/components/schemas/SessionBlock'
        trend:
          $ref: '#/components/schemas/TrendBlock'
        gap:
          $ref: '#/components/schemas/GapBlock'
      required: [timestamp_utc, price]

    Bar:
      type: object
      properties:
        time:
          type: string
        open:
          type: number
        high:
          type: number
        low:
          type: number
        close:
          type: number
        volume:
          type: number
      required: [time, open, high, low, close]

    IndicatorPoint:
      type: object
      properties:
        time:
          type: string
        value:
          type: number
      required: [time, value]

    IndicatorsSeriesBlock:
      type: object
      properties:
        ema9:
          type: array
          items:
            $ref: '#/components/schemas/IndicatorPoint'
        ema20:
          type: array
          items:
            $ref: '#/components/schemas/IndicatorPoint'
        ema50:
          type: array
          items:
            $ref: '#/components/schemas/IndicatorPoint'
        vwap:
          type: array
          items:
            $ref: '#/components/schemas/IndicatorPoint'
        atr14:
          type: array
          items:
            $ref: '#/components/schemas/IndicatorPoint'
        adx14:
          type: array
          items:
            $ref: '#/components/schemas/IndicatorPoint'

    # ----- Scan -----
    ScanRequest:
      type: object
      properties:
        tickers:
          type: array
          minItems: 1
          items:
            type: string
        style:
          type: string
          enum: ['scalp', 'intraday', 'swing', 'leaps']
      required: [tickers]

    ScanItem:
      type: object
      properties:
        symbol:
          type: string
        style:
          type: string
          enum: ['scalp', 'intraday', 'swing', 'leaps']
        strategy_id:
          type: string
        description:
          type: string
        score:
          type: number
        direction_hint:
          type: string
        key_levels:
          $ref: '#/components/schemas/KeyLevels'
        market_snapshot:
          $ref: '#/components/schemas/MarketSnapshot'
        features:
          type: object
          additionalProperties: true
        charts:
          $ref: '#/components/schemas/ChartsPayload'
        data:
          $ref: '#/components/schemas/DataLinks'
      required: [symbol, style, score, key_levels, market_snapshot, charts, data]

    ScanRow:
      type: object
      properties:
        symbol:
          type: string
        style:
          type: string
          enum: ['scalp', 'intraday', 'swing', 'leaps']
        score:
          type: number
        direction_hint:
          type: string
        trade_detail:
          type: string
          format: uri
      required: [symbol, style, score]

    # ----- Context -----
    ContextResponse:
      type: object
      properties:
        symbol:
          type: string
        interval:
          type: string
          enum: ['1m', '5m', '15m', '1h', '1D']
        lookback:
          type: integer
        bars:
          type: array
          items:
            $ref: '#/components/schemas/Bar'
        indicators:
          $ref: '#/components/schemas/IndicatorsSeriesBlock'
        key_levels:
          $ref: '#/components/schemas/KeyLevels'
        snapshot:
          $ref: '#/components/schemas/MarketSnapshot'
      required: [symbol, interval, bars]

    # ----- Multi-context -----
    MultiContextRequest:
      type: object
      properties:
        symbol:
          type: string
        intervals:
          type: array
          items:
            type: string
            enum: ['1m', '5m', '15m', '1h', '4h', '1D', 'w']
        lookback:
          type: integer
          minimum: 50
          maximum: 5000
          default: 500
        include_series:
          type: boolean
          default: false
      required: [symbol, intervals]

    TimeframeSnapshot:
      type: object
      properties:
        interval:
          type: string
        bars:
          type: array
          items:
            $ref: '#/components/schemas/Bar'
        indicators:
          $ref: '#/components/schemas/IndicatorsSeriesBlock'
        key_levels:
          $ref: '#/components/schemas/KeyLevels'
        snapshot:
          $ref: '#/components/schemas/MarketSnapshot'
      required: [interval, bars, snapshot]

    VolatilityRegime:
      type: object
      properties:
        iv_rank:
          type: number
        iv_percentile:
          type: number
        hv_20:
          type: number
        hv_60:
          type: number
        realized_vol:
          type: number
        skew_index:
          type: number
        regime_label:
          type: string
          enum: ['low', 'normal', 'elevated', 'extreme']

    SentimentBlock:
      type: object
      properties:
        symbol_sentiment:
          type: number
          description: '-1..+1'
        news_count_24h:
          type: integer
        news_bias_24h:
          type: number
        social_sentiment:
          type: number
        headline_risk:
          type: string
          enum: ['low', 'normal', 'elevated', 'extreme']

    EventsBlock:
      type: object
      properties:
        next_fomc_minutes:
          type: integer
        next_cpi_minutes:
          type: integer
        next_nfp_minutes:
          type: integer
        within_event_window:
          type: boolean
        label:
          type: string
          enum: ['none', 'minor', 'watch', 'risk', 'embargo']

    EarningsBlock:
      type: object
      properties:
        next_earnings_at:
          type: string
          format: date-time
        dte_to_earnings:
          type: number
        pre_or_post:
          type: string
          enum: ['pre', 'post', 'none']
        earnings_flag:
          type: string
          enum: ['none', 'near', 'today', 'after_close', 'before_open']
        expected_move_pct:
          type: number

    SummaryDTO:
      type: object
      properties:
        frames_used:
          type: array
          items:
            type: string
            enum: ['1m', '5m', '15m', '1h', '4h', '1D', 'w']
        confluence_score:
          type: number
        trend_notes:
          type: object
          additionalProperties:
            type: string
        expected_move_horizon:
          type: number
        nearby_levels:
          type: array
          items:
            type: string

    MultiContextResponse:
      type: object
      properties:
        symbol:
          type: string
        snapshots:
          type: array
          items:
            $ref: '#/components/schemas/TimeframeSnapshot'
        volatility_regime:
          $ref: '#/components/schemas/VolatilityRegime'
        sentiment:
          $ref: '#/components/schemas/SentimentBlock'
        events:
          $ref: '#/components/schemas/EventsBlock'
        earnings:
          $ref: '#/components/schemas/EarningsBlock'
        summary:
          $ref: '#/components/schemas/SummaryDTO'
      required: [symbol, snapshots]

    # ----- Plan & Charts -----
    PlanRequest:
      type: object
      properties:
        symbol:
          type: string
        style:
          type: string
          enum: ['scalp', 'intraday', 'swing', 'leaps']
        strategy_hint:
          type: string
        snapshot:
          $ref: '#/components/schemas/MarketSnapshot'
        key_levels:
          $ref: '#/components/schemas/KeyLevels'
        features:
          type: object
          additionalProperties: true
        constraints:
          type: object
          properties:
            min_rr:
              type: number
              default: 1.2
            max_targets:
              type: integer
              default: 2
            prefer_expected_move_cap:
              type: boolean
              default: true
            offline_allow:
              type: boolean
              default: false
      required: [symbol, style, snapshot, key_levels]

    # Expanded PlanResponse (normal + watch-plan + offline) with options bundle
    PlanResponse:
      type: object
      required: [symbol]
      properties:
        plan_id:
          type: string
        version:
          type: integer
        trade_detail:
          type: string
          format: uri
        warnings:
          type: array
          items:
            type: string

        planning_context:
          type: string
          enum: [live, offline, backtest]
        offline_basis:
          type: object
          properties:
            htf_snapshot_time:
              type: string
              format: date-time
            volatility_regime:
              type: string
            expected_move_days:
              type: integer

        symbol:
          type: string
        style:
          type: string
          enum: ['scalp', 'intraday', 'swing', 'leaps']
        bias:
          type: string
          enum: ['long', 'short']
        setup:
          type: string
        entry:
          type: number
        stop:
          type: number
        targets:
          type: array
          items:
            type: number
        target_meta:
          type: array
          items:
            $ref: '#/components/schemas/TargetMeta'
        runner:
          $ref: '#/components/schemas/RunnerConfig'
        rr_to_t1:
          type: number
        confidence:
          type: number
        confidence_factors:
          type: array
          items:
            type: string
        notes:
          type: string

        relevant_levels:
          type: object
          additionalProperties: true
        expected_move_basis:
          type: string
        sentiment:
          $ref: '#/components/schemas/SentimentBlock'
        events:
          $ref: '#/components/schemas/EventsBlock'
        earnings:
          $ref: '#/components/schemas/EarningsBlock'
        charts_params:
          type: object
          additionalProperties: true
        chart_url:
          type: string
          format: uri
        charts:
          $ref: '#/components/schemas/ChartsPayload'
        key_levels:
          $ref: '#/components/schemas/KeyLevels'
        market_snapshot:
          $ref: '#/components/schemas/MarketSnapshot'
        features:
          type: object
          additionalProperties: true
        plan:
          type: object
          additionalProperties: true
        calc_notes:
          type: object
          additionalProperties: true
        htf:
          type: object
          additionalProperties: true
        decimals:
          type: integer
        data_quality:
          type: object
          additionalProperties: true
        debug:
          type: object
          additionalProperties: true
        options:
          type: object
          additionalProperties: true

    TargetMeta:
      type: object
      properties:
        price:
          type: number
        distance:
          type: number
          description: 'Absolute distance from entry (points)'
        rr:
          type: number
          description: 'Risk/Reward at this TP'
        em_fraction:
          type: number
          description: 'Fraction of expected move to this TP (0..1+)'
        mfe_quantile:
          type: number
          description: 'Historical MFE quantile reached by similar setups'
        prob_touch:
          type: number
          description: 'Estimated probability of touch (0..1)'
        label:
          type: string
        note:
          type: string

    RunnerConfig:
      type: object
      properties:
        type:
          type: string
          description: 'Trailing method identifier'
        timeframe:
          type: string
          enum: ['1m', '5m', '15m', '1h', '4h', '1D', 'w']
        length:
          type: integer
        multiplier:
          type: number
        bias:
          type: string
          enum: ['long', 'short']
        anchor:
          type: string
          description: 'Reference (e.g., ATR, EMA, structure)'
        initial_stop:
          type: number
        note:
          type: string

    ChartsParams:
      type: object
      properties:
        symbol:
          type: string
        interval:
          type: string
        ema:
          type: string
        vwap:
          type: string
        direction:
          type: string
          enum: ['long', 'short']
        strategy:
          type: string
        entry:
          type: number
        stop:
          type: number
        tp:
          type: string
        levels:
          type: string
        view:
          type: string
        notes:
          type: string
        tp_meta:
          type: string
          description: JSON-encoded array of per-target metadata
        runner:
          type: string
          description: JSON-encoded runner trail configuration

    ChartUrlRequest:
      type: object
      properties:
        symbol:
          type: string
        interval:
          type: string
        ema:
          type: string
        vwap:
          type: string
        view:
          type: string
        entry:
          type: number
        stop:
          type: number
        tp:
          type: string
        levels:
          type: string
        strategy:
          type: string
        direction:
          type: string
          enum: ['long', 'short']
        price:
          type: number
        notes:
          type: string
        tp_meta:
          type: string
          description: JSON-encoded array of per-target metadata
        runner:
          type: string
          description: JSON-encoded runner trail configuration
      required: [symbol, interval]

    ChartUrlResponse:
      type: object
      properties:
        interactive:
          type: string
          format: uri
      required: [interactive]

    # ----- Options -----
    ContractsRequest:
      type: object
      properties:
        symbol:
          type: string
        side:
          type: string
          enum: ['call', 'put']
        style:
          type: string
          enum: ['scalp', 'intraday', 'swing', 'leaps']
        selection_mode:
          type: string
          enum: ['analyze', 'select']
          default: 'analyze'
        min_oi:
          type: integer
          default: 500
        max_spread_pct:
          type: number
          default: 12
        min_dte:
          type: integer
        max_dte:
          type: integer
        min_delta:
          type: number
        max_delta:
          type: number
        expiry:
          type: string
        max_price:
          type: number
        risk_amount:
          type: number
        plan_anchor:
          type: object
          properties:
            underlying_entry:
              type: number
            stop:
              type: number
            targets:
              type: array
              items:
                type: number
              maxItems: 3
            horizon_minutes:
              type: integer
            iv_shift_bps:
              type: integer
            slippage_bps:
              type: integer
      required: [symbol, side, style]

    OptionContract:
      type: object
      properties:
        label:
          type: string
          example: NVDA 2025-10-17 475C
        expiry:
          type: string
          example: '2025-10-17'
        dte:
          type: integer
        strike:
          type: number
        type:
          type: string
          enum: ['CALL', 'PUT']
        price:
          type: number
        bid:
          type: number
        ask:
          type: number
        mark:
          type: number
        spread_pct:
          type: number
        delta:
          type: number
        gamma:
          type: number
        theta:
          type: number
        vega:
          type: number
        rho:
          type: number
        iv:
          type: number
        iv_rank:
          type: number
        oi:
          type: integer
        volume:
          type: integer
        probability_itm:
          type: number
        break_even:
          type: number
        underlying_price:
          type: number
        liquidity_score:
          type: number
          description: '0-100 heuristic'
        last_trade_time:
          type: string
      required: [label, expiry, dte, strike, type, price, bid, ask, spread_pct, delta]

    ContractsResponse:
      type: object
      properties:
        symbol:
          type: string
        side:
          type: string
          enum: ['call', 'put']
        style:
          type: string
          enum: ['scalp', 'intraday', 'swing', 'leaps']
        selection_mode:
          type: string
          enum: ['analyze', 'select']
        filters_applied:
          type: object
          properties:
            min_oi:
              type: integer
            max_spread_pct:
              type: number
            min_dte:
              type: integer
            max_dte:
              type: integer
            min_delta:
              type: number
            max_delta:
              type: number
            expiry:
              type: string
            max_price:
              type: number
        calc_basis:
          type: object
          properties:
            underlying_entry:
              type: number
            stop:
              type: number
            targets:
              type: array
              items:
                type: number
              maxItems: 3
            horizon_minutes:
              type: integer
        assumptions:
          type: object
          properties:
            iv_shift_bps:
              type: integer
            slippage_bps:
              type: integer
        candidates:
          type: array
          items:
            $ref: '#/components/schemas/OptionContract'
        selected:
          $ref: '#/components/schemas/OptionContract'
        rationale:
          type: array
          items:
            type: string
        notes:
          type: string
      required: [symbol, side, style, selection_mode, candidates]

    # ----- Futures & Sentiment (refs used by paths) -----
    FuturesQuote:
      type: object
      properties:
        symbol:
          type: string
        last:
          type: number
        percent:
          type: number
          description: 'change vs prev close (e.g., -0.62 for -0.62%)'
        time_utc:
          type: string
          format: date-time
        stale:
          type: boolean

    FuturesSnapshot:
      type: object
      properties:
        es_proxy:
          $ref: '#/components/schemas/FuturesQuote'
        nq_proxy:
          $ref: '#/components/schemas/FuturesQuote'
        ym_proxy:
          $ref: '#/components/schemas/FuturesQuote'
        rty_proxy:
          $ref: '#/components/schemas/FuturesQuote'
        vix:
          $ref: '#/components/schemas/FuturesQuote'
        market_phase:
          type: string
          enum: ['regular', 'premarket', 'afterhours', 'closed']
        stale_seconds:
          type: integer
      required: [market_phase]

    SentimentResponse:
      type: object
      properties:
        channel:
          type: string
        video_id:
          type: string
        video_url:
          type: string
          format: uri
        published_at:
          type: string
          format: date-time
        title:
          type: string
        summary:
          type: string
        sentiment_label:
          type: string
          enum: ['bullish', 'neutral', 'bearish', 'mixed']
        sentiment_score:
          type: number
        tickers:
          type: array
          items:
            type: string
        key_levels:
          type: array
          items:
            type: object
            properties:
              symbol:
                type: string
              level:
                type: number
              note:
                type: string
        risks:
          type: array
          items:
            type: string
        quotes:
          type: array
          items:
            type: string
        raw_excerpt:
          type: string
      required:
        - channel
        - video_id
        - video_url
        - published_at
        - title
        - sentiment_label
        - sentiment_score
        - tickers

    # ----- Idea Snapshots -----
    PlanCore:
      type: object
      properties:
        plan_id:
          type: string
        version:
          type: integer
        generated_at:
          type: string
          format: date-time
        symbol:
          type: string
        style:
          type: string
          enum: ['scalp', 'intraday', 'swing', 'leaps']
        bias:
          type: string
          enum: ['long', 'short']
        setup:
          type: string
        entry:
          type: number
        stop:
          type: number
        targets:
          type: array
          items:
            type: number
          minItems: 1
          maxItems: 3
        rr_to_t1:
          type: number
        confidence:
          type: number
        decimals:
          type: integer
          minimum: 0
          maximum: 4
        charts_params:
          $ref: '#/components/schemas/ChartsParams'
        warnings:
          type: array
          items:
            type: string
        trade_detail:
          type: string
          format: uri
      required:
        - plan_id
        - version
        - symbol
        - style
        - bias
        - setup
        - entry
        - stop
        - targets
        - rr_to_t1
        - confidence
        - decimals
        - charts_params
        - trade_detail

    IdeaSnapshot:
      type: object
      properties:
        plan:
          $ref: '#/components/schemas/PlanCore'
        summary:
          $ref: '#/components/schemas/SummaryDTO'
        volatility_regime:
          $ref: '#/components/schemas/VolatilityRegime'
        htf:
          type: object
          properties:
            bias:
              type: string
              enum: ['aligned', 'neutral', 'opposed']
            snapped_targets:
              type: array
              items:
                type: string
        data_quality:
          type: object
          properties:
            series_present:
              type: boolean
            iv_present:
              type: boolean
            earnings_present:
              type: boolean
        chart_url:
          type: string
          format: uri
        options:
          type: object
          properties:
            table:
              type: array
              items:
                $ref: '#/components/schemas/OptionContract'
            side:
              type: string
              enum: ['call', 'put']
            style:
              type: string
              enum: ['scalp', 'intraday', 'swing', 'leaps']
        why_this_works:
          type: array
          items:
            type: string
          maxItems: 4
        invalidation:
          type: array
          items:
            type: string
          maxItems: 4
        risk_note:
          type: string
      required: [plan, summary, chart_url, why_this_works]
