(function () {
  const params = new URLSearchParams(window.location.search);

  const normalizeResolution = (value) => {
    const token = (value || '').toString().trim().toLowerCase();
    if (!token) return '1';
    if (token.endsWith('m')) return String(parseInt(token.replace('m', ''), 10) || 1);
    if (token.endsWith('h')) return String((parseInt(token.replace('h', ''), 10) || 1) * 60);
    if (token.endsWith('w')) return '1W';
    if (token === 'd' || token === '1d') return '1D';
    return token.toUpperCase();
  };

  const resolutionToSeconds = (resolution) => {
    const token = (resolution || '').trim().toUpperCase();
    if (token.endsWith('D')) {
      const days = parseInt(token, 10) || 1;
      return days * 24 * 60 * 60;
    }
    if (token.endsWith('W')) {
      const weeks = parseInt(token, 10) || 1;
      return weeks * 7 * 24 * 60 * 60;
    }
    const minutes = parseInt(token, 10);
    if (!Number.isFinite(minutes) || minutes <= 0) return 60;
    return minutes * 60;
  };

  const parseList = (value) =>
    (value || '')
      .split(',')
      .map((chunk) => chunk.trim())
      .filter(Boolean);

  const parseFloatList = (value) =>
    parseList(value)
      .map((item) => parseFloat(item))
      .filter((num) => Number.isFinite(num));

  const parseNamedLevels = (value) => {
    if (!value) return [];
    const chunks = value.includes(';') ? value.split(';') : value.split(',');
    return chunks
      .map((raw) => {
        const token = raw.trim();
        if (!token) return null;
        const [pricePart, labelPart] = token.split('|').map((part) => part.trim());
        const price = parseFloat(pricePart);
        if (!Number.isFinite(price)) return null;
        return { price, label: labelPart || null };
      })
      .filter(Boolean);
  };

  const parseNumber = (value) => {
    if (value === null || value === undefined || value === '') return null;
    const num = Number.parseFloat(value);
    return Number.isFinite(num) ? num : null;
  };

  const symbol = (params.get('symbol') || 'AAPL').toUpperCase();
  let currentResolution = normalizeResolution(params.get('interval') || '15');
  const theme = params.get('theme') === 'light' ? 'light' : 'dark';
  const baseUrl = `${window.location.protocol}//${window.location.host}`;
  const planIdParam = (params.get('plan_id') || '').trim() || null;
  const planVersionParam = (params.get('plan_version') || '').trim() || null;

  const toNumber = (value) => {
    const num = Number(value);
    return Number.isFinite(num) ? num : null;
  };

  const basePlan = {
    entry: parseNumber(params.get('entry')),
    stop: parseNumber(params.get('stop')),
    tps: parseFloatList(params.get('tp')),
    direction: (params.get('direction') || '').toLowerCase(),
    strategy: params.get('strategy'),
    atr: parseNumber(params.get('atr')),
    notes: params.get('notes'),
    title: params.get('title'),
    tpMeta: (() => {
      try {
        return JSON.parse(params.get('tp_meta') || '[]');
      } catch {
        return [];
      }
    })(),
    runner: (() => {
      try {
        return JSON.parse(params.get('runner') || 'null');
      } catch {
        return null;
      }
    })(),
  };

  const clonePlan = () => JSON.parse(JSON.stringify(basePlan));
  let currentPlan = clonePlan();

  let keyLevels = parseNamedLevels(params.get('levels'));

  let planMeta = {};
  try {
    planMeta = JSON.parse(params.get('plan_meta') || '{}');
  } catch {
    planMeta = {};
  }

  const paramConfidence = parseNumber(params.get('confidence'));
  const mergedPlanMeta = {
    symbol,
    style: planMeta.style || params.get('style'),
    style_display: planMeta.style_display || null,
    bias: planMeta.bias || basePlan.direction || null,
    confidence: toNumber(planMeta.confidence) ?? paramConfidence,
    risk_reward: toNumber(planMeta.risk_reward ?? planMeta.rr_to_t1),
    notes: planMeta.notes || params.get('notes') || null,
    warnings: Array.isArray(planMeta.warnings) ? planMeta.warnings : [],
    runner: planMeta.runner || basePlan.runner,
    targets: Array.isArray(planMeta.targets) && planMeta.targets.length ? planMeta.targets : basePlan.tps,
    target_meta: Array.isArray(planMeta.target_meta) && planMeta.target_meta.length ? planMeta.target_meta : basePlan.tpMeta,
    entry: toNumber(planMeta.entry) ?? basePlan.entry,
    stop: toNumber(planMeta.stop) ?? basePlan.stop,
    atr: toNumber(planMeta.atr) ?? basePlan.atr,
    strategy: planMeta.strategy_label || basePlan.strategy,
    expected_move: toNumber(planMeta.expected_move),
    horizon_minutes: toNumber(planMeta.horizon_minutes),
    key_levels: planMeta.key_levels || null,
  };

  if ((!keyLevels || !keyLevels.length) && mergedPlanMeta.key_levels) {
    keyLevels = Object.entries(mergedPlanMeta.key_levels)
      .map(([name, value]) => {
        const numeric = toNumber(value);
        if (!Number.isFinite(numeric)) return null;
        const label = name
          .replace(/_/g, ' ')
          .replace(/\b([a-z])/g, (match) => match.toUpperCase());
        return { price: numeric, label };
      })
      .filter(Boolean);
  }
  const dedupedLevels = [];
  const levelSet = new Set();
  (keyLevels || []).forEach((item) => {
    if (!item || !Number.isFinite(item.price)) return;
    const token = `${item.label || ''}:${item.price.toFixed(4)}`;
    if (levelSet.has(token)) return;
    levelSet.add(token);
    dedupedLevels.push(item);
  });
  keyLevels = dedupedLevels;

  const headerSymbolEl = document.getElementById('header_symbol');
  const headerStrategyEl = document.getElementById('header_strategy');
  const headerBiasEl = document.getElementById('header_bias');
  const headerConfidenceEl = document.getElementById('header_confidence');
  const headerRREl = document.getElementById('header_rr');
  const headerDurationEl = document.getElementById('header_duration');
  const headerLastPriceEl = document.getElementById('header_lastprice');
  const headerPlanStatusEl = document.getElementById('header_planstatus');
  const headerMarketEl = document.getElementById('header_market');
  const planStatusNoteEl = document.getElementById('plan_status_note');
  const timeframeSwitcherEl = document.getElementById('timeframe_switcher');
  const planPanelEl = document.getElementById('plan_panel');
  const planPanelBodyEl = document.getElementById('plan_panel_body');
  const debugEl = document.getElementById('debug_banner');

  const showError = (message) => {
    if (!debugEl) return;
    debugEl.style.display = 'block';
    debugEl.textContent = message;
  };

  window.addEventListener('error', (event) => {
    showError(`Runtime error: ${event.message}`);
  });

  window.addEventListener('unhandledrejection', (event) => {
    showError(`Unhandled promise rejection: ${event.reason}`);
  });

  const TIMEFRAMES = [
    { label: '1m', resolution: '1' },
    { label: '5m', resolution: '5' },
    { label: '10m', resolution: '10' },
    { label: '30m', resolution: '30' },
    { label: '1H', resolution: '60' },
    { label: '1D', resolution: '1D' },
    { label: '1W', resolution: '1W' },
  ];
  let activeTimeframe =
    TIMEFRAMES.find((tf) => normalizeResolution(tf.resolution) === currentResolution) || TIMEFRAMES[0];

  const layoutBase = {
    background: { type: 'solid', color: theme === 'light' ? '#ffffff' : '#0b0f14' },
    textColor: theme === 'light' ? '#1b2733' : '#e6edf3',
  };
  const gridBase = {
    vertLines: { color: theme === 'light' ? '#e5e9f0' : '#1f2933' },
    horzLines: { color: theme === 'light' ? '#e5e9f0' : '#1f2933' },
  };

  const container = document.getElementById('tv_chart_container');
  const chart = LightweightCharts.createChart(container, {
    layout: layoutBase,
    grid: gridBase,
    crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
    timeScale: { borderColor: theme === 'light' ? '#d1d5db' : '#1f2933' },
    rightPriceScale: {
      borderColor: theme === 'light' ? '#d1d5db' : '#1f2933',
      scaleMargins: { top: 0.1, bottom: 0.25 },
    },
    leftPriceScale: {
      visible: true,
      borderColor: theme === 'light' ? '#d1d5db' : '#1f2933',
      scaleMargins: { top: 0.8, bottom: 0.02 },
    },
    watermark: {
      visible: true,
      fontSize: 22,
      horzAlign: 'left',
      vertAlign: 'bottom',
      color: theme === 'light' ? 'rgba(15,23,42,0.08)' : 'rgba(148,163,184,0.08)',
      text: '',
    },
  });

  const candleSeries = chart.addCandlestickSeries({
    upColor: '#22c55e',
    downColor: '#ef4444',
    borderUpColor: '#22c55e',
    borderDownColor: '#ef4444',
    wickUpColor: '#22c55e',
    wickDownColor: '#ef4444',
  });

  const volumeSeries = chart.addHistogramSeries({
    priceScaleId: 'left',
    color: theme === 'light' ? '#94a3b8' : '#334155',
    priceFormat: { type: 'volume' },
    scaleMargins: { top: 0.7, bottom: 0 },
  });

  const emaPalette = ['#38bdf8', '#a855f7', '#facc15', '#f97316'];
  const emaTokensInput = parseList(params.get('ema'));
  const emaTokens = emaTokensInput.length ? emaTokensInput : ['9', '21', '50'];
  const emaSeries = emaTokens.reduce((acc, token, idx) => {
    const span = parseInt(token, 10);
    if (!Number.isFinite(span) || span <= 0) return acc;
    const series = chart.addLineSeries({
      lineWidth: 2,
      color: emaPalette[idx % emaPalette.length],
      title: `EMA${span}`,
    });
    acc.push({ span, series });
    return acc;
  }, []);

  let vwapSeries = null;
  const scalePlanToken = (params.get('scale_plan') || 'auto').toLowerCase();

  let lastKnownPrice = null;
  let fetchToken = 0;
  let currentPlanStatus = 'intact';
  let latestPlanNote = 'Plan intact. Risk profile unchanged.';
  let latestNextStep = 'hold_plan';
  let latestMarketNote = null;
  let currentMarketPhase = null;
  let streamSource = null;
  let latestCandleData = [];
  let latestVolumeData = [];
  const DEFAULT_REPLAY_MINUTES = 10;
  const REPLAY_MAX_MINUTES = 180;
  const REPLAY_STEP_MS = 750;
  let isReplaying = false;
  let replayQueue = [];
  let replayIndex = 0;
  let replayTimer = null;
  let replayPrevPhase = null;
  let replayPrevNote = null;
  let replayHadStream = false;
  let replayFetchToken = 0;
  let replaySavedCandleData = [];
  let replaySavedVolumeData = [];
  let replaySavedVisibleRange = null;
  let replayStatusEl = null;
  let replayStartButton = null;
  let replayStopButton = null;
  let replayMinutesInput = null;
  let replayStatusMessage = '';
  const replayConfig = { minutes: DEFAULT_REPLAY_MINUTES };

  const PLAN_STATUS_META = {
    intact: { label: 'Plan Intact', className: 'status-pill--intact' },
    at_risk: { label: 'Plan At Risk', className: 'status-pill--risk' },
    invalidated: { label: 'Plan Invalidated', className: 'status-pill--invalid' },
    reversal: { label: 'Plan Reversal', className: 'status-pill--reversal' },
  };

  const NEXT_STEP_LABELS = {
    hold_plan: 'Hold plan',
    tighten_stop: 'Tighten stop',
    plan_invalidated: 'Plan invalidated',
    consider_reversal: 'Consider reversal',
  };

  const MARKET_PHASE_META = {
    regular: { label: 'Market Open', className: 'status-pill--intact', note: 'Market open.' },
    premarket: { label: 'Pre-Market', className: 'status-pill--risk', note: 'Pre-market session — liquidity thinner.' },
    afterhours: { label: 'After Hours', className: 'status-pill--risk', note: 'After hours — liquidity thinner.' },
    closed: { label: 'Market Closed', className: 'status-pill--invalid', note: 'Market closed — live updates limited.' },
    replay: { label: 'Replay Mode', className: 'status-pill--risk', note: 'Replaying recent price action.' },
  };

  const formatNextStep = (token) => {
    if (!token) return null;
    const normalized = token.toLowerCase();
    if (NEXT_STEP_LABELS[normalized]) return NEXT_STEP_LABELS[normalized];
    return normalized.replace(/_/g, ' ').replace(/\b\w/g, (ch) => ch.toUpperCase());
  };

  const clampReplayMinutes = (value) => {
    const num = Number.parseInt(value, 10);
    if (!Number.isFinite(num)) return DEFAULT_REPLAY_MINUTES;
    const clamped = Math.min(Math.max(num, 1), REPLAY_MAX_MINUTES);
    return clamped;
  };

  const parseEventTimestamp = (payload) => {
    if (!payload) return null;
    const raw = payload.ts || payload.timestamp || payload.time || payload.t;
    if (raw === null || raw === undefined) return null;
    if (typeof raw === 'number') {
      if (raw > 10_000_000_000) {
        return Math.floor(raw / 1000);
      }
      return Math.floor(raw);
    }
    if (typeof raw === 'string') {
      const num = Number(raw);
      if (Number.isFinite(num)) {
        return num > 10_000_000_000 ? Math.floor(num / 1000) : Math.floor(num);
      }
      const date = new Date(raw);
      if (!Number.isNaN(date.getTime())) {
        return Math.floor(date.getTime() / 1000);
      }
    }
    if (raw instanceof Date && !Number.isNaN(raw.getTime())) {
      return Math.floor(raw.getTime() / 1000);
    }
    return null;
  };

  const updateStatusNote = () => {
    if (!planStatusNoteEl) return;
    const parts = [];
    if (latestPlanNote) parts.push(latestPlanNote);
    if (latestNextStep) {
      const label = formatNextStep(latestNextStep);
      if (label) parts.push(`Next: ${label}`);
    }
    if (latestMarketNote) parts.push(latestMarketNote);
    if (parts.length) {
      planStatusNoteEl.textContent = parts.join(' • ');
      planStatusNoteEl.classList.add('active');
    } else {
      planStatusNoteEl.textContent = '';
      planStatusNoteEl.classList.remove('active');
    }
  };

  const applyPlanStatus = (status, note, nextStep, rrValue) => {
    if (typeof status === 'string' && status) {
      currentPlanStatus = status.toLowerCase();
    }
    if (typeof note === 'string' && note.trim()) {
      latestPlanNote = note.trim();
    }
    if (typeof nextStep === 'string' && nextStep.trim()) {
      latestNextStep = nextStep.trim();
    }
    const meta = PLAN_STATUS_META[currentPlanStatus] || PLAN_STATUS_META.intact;
    if (headerPlanStatusEl) {
      headerPlanStatusEl.textContent = meta.label;
      headerPlanStatusEl.className = `status-pill ${meta.className}`;
    }
    if (typeof rrValue === 'number' && headerRREl) {
      headerRREl.textContent = `R:R (TP1): ${rrValue.toFixed(2)}`;
    }
    updateStatusNote();
  };

  const applyMarketStatus = (phase, note) => {
    if (typeof phase === 'string' && phase.trim()) {
      currentMarketPhase = phase.trim().toLowerCase();
    } else if (!currentMarketPhase) {
      currentMarketPhase = 'closed';
    }
    const meta = MARKET_PHASE_META[currentMarketPhase] || MARKET_PHASE_META.closed;
    if (headerMarketEl) {
      headerMarketEl.textContent = meta.label;
      headerMarketEl.className = `status-pill ${meta.className}`;
    }
    if (note && typeof note === 'string' && note.trim()) {
      latestMarketNote = note.trim();
    } else {
      latestMarketNote = meta.note;
    }
    updateStatusNote();
  };

  applyPlanStatus(currentPlanStatus, latestPlanNote, latestNextStep, null);
  applyMarketStatus('closed', 'Market closed — live updates limited.');

  const matchesPlan = (incomingPlanId) => {
    if (!planIdParam) return true;
    return (incomingPlanId || '').trim() === planIdParam;
  };

  const handlePlanStateEvent = (plans) => {
    if (!Array.isArray(plans) || plans.length === 0) return;
    let candidate = null;
    if (planIdParam) {
      candidate = plans.find((item) => (item?.plan_id || '').trim() === planIdParam);
    }
    if (!candidate) {
      candidate = plans[0];
    }
    if (!candidate) return;
    applyPlanStatus(candidate.status || currentPlanStatus, candidate.note, candidate.next_step, candidate.rr_to_t1);
  };

  const handlePlanDeltaEvent = (payload) => {
    if (!payload || !matchesPlan(payload.plan_id)) return;
    const changes = payload.changes || {};
    const statusToken = changes.status || payload.status || currentPlanStatus;
    const rrValue = Number.isFinite(changes.rr_to_t1) ? changes.rr_to_t1 : null;
    applyPlanStatus(statusToken, changes.note, changes.next_step, rrValue);
    if (Number.isFinite(changes.last_price)) {
      lastKnownPrice = changes.last_price;
      updateHeaderPricing(lastKnownPrice);
      renderPlanPanel(lastKnownPrice);
      updatePlanPanelLastPrice(lastKnownPrice);
    }
  };

  const handlePlanFullEvent = (payload) => {
    if (!payload) return;
    const planBlock = payload.plan || {};
    if (!matchesPlan(planBlock.plan_id)) return;
    applyPlanStatus('intact', 'Plan updated. Review levels.', 'hold_plan', planBlock.rr_to_t1);
  };

  const handleTickEvent = (payload) => {
    if (!payload || isReplaying) return;
    const price = Number.isFinite(payload.p) ? payload.p : Number.isFinite(payload.close) ? payload.close : null;
    if (price === null) return;
    lastKnownPrice = price;
    updateHeaderPricing(price);
    updatePlanPanelLastPrice(price);
    updateRealtimeBar(price, payload);
  };

  const connectStream = () => {
    if (isReplaying || typeof EventSource === 'undefined') return;
    const streamUrl = `${baseUrl}/stream/${symbol}`;
    if (streamSource) {
      streamSource.close();
      streamSource = null;
    }
    try {
      streamSource = new EventSource(streamUrl);
      streamSource.onmessage = (msg) => {
        if (!msg?.data) return;
        let envelope;
        try {
          envelope = JSON.parse(msg.data);
        } catch (err) {
          console.warn('Failed to parse stream message', err);
          return;
        }
        const event = envelope?.event;
        if (!event || !event.t) return;
        switch (event.t) {
          case 'plan_state':
            handlePlanStateEvent(event.plans);
            break;
          case 'plan_delta':
            handlePlanDeltaEvent(event);
            break;
          case 'plan_full':
            handlePlanFullEvent(event.payload);
            break;
          case 'tick':
            handleTickEvent(event);
            break;
          case 'bar':
            handleTickEvent(event);
            break;
          case 'market_status':
            applyMarketStatus(event.phase, event.note);
            break;
          default:
            break;
        }
      };
      streamSource.onerror = () => {
        if (streamSource) {
          streamSource.close();
          streamSource = null;
        }
        window.setTimeout(connectStream, 5000);
      };
    } catch (err) {
      console.error('Stream connection failed', err);
      window.setTimeout(connectStream, 5000);
    }
  };

  const priceLineMap = new Map();
  const setPriceLine = (id, options) => {
    if (!Number.isFinite(options.price)) return;
    const existing = priceLineMap.get(id);
    if (existing) {
      existing.applyOptions(options);
      return existing;
    }
    const line = candleSeries.createPriceLine(options);
    priceLineMap.set(id, line);
    return line;
  };
  const prunePriceLines = (activeIds) => {
    for (const [id, line] of priceLineMap.entries()) {
      if (!activeIds.has(id)) {
        candleSeries.removePriceLine(line);
        priceLineMap.delete(id);
      }
    }
  };

  const setWatermark = () => {
    const tfLabel = activeTimeframe ? activeTimeframe.label : currentResolution;
    chart.applyOptions({
      watermark: {
        text: `${symbol} · ${tfLabel}`,
      },
    });
  };

  const TIMEFRAME_REFRESH_MS = 60000;

  const initializeTimeframes = () => {
    timeframeSwitcherEl.innerHTML = '';
    TIMEFRAMES.forEach((tf) => {
      const button = document.createElement('button');
      button.className = 'timeframe-button';
      button.textContent = tf.label;
      button.dataset.resolution = tf.resolution;
      if (normalizeResolution(tf.resolution) === currentResolution) {
        button.classList.add('active');
        activeTimeframe = tf;
      }
      button.addEventListener('click', () => {
        if (normalizeResolution(tf.resolution) === currentResolution) return;
        currentResolution = normalizeResolution(tf.resolution);
        activeTimeframe = tf;
        params.set('interval', tf.label);
        const newUrl = `${window.location.pathname}?${params.toString()}`;
        window.history.replaceState({}, '', newUrl);
        Array.from(timeframeSwitcherEl.querySelectorAll('button')).forEach((btn) => btn.classList.remove('active'));
        button.classList.add('active');
        setWatermark();
        fetchBars();
      });
      timeframeSwitcherEl.appendChild(button);
    });
  };

  const formatPrice = (value) => (Number.isFinite(value) ? value.toFixed(2) : '—');
  const formatPercentage = (value) => {
    const num = toNumber(value);
    return num !== null ? `${(num * 100).toFixed(0)}%` : '—';
  };

  const formatProbability = (value) => {
    const num = toNumber(value);
    if (num === null) return null;
    return `${Math.round(num * 100)}% POT`;
  };

  const estimateDuration = () => {
    if (Number.isFinite(mergedPlanMeta.horizon_minutes)) {
      const minutes = mergedPlanMeta.horizon_minutes;
      if (minutes >= 1440) {
        const days = minutes / 1440;
        return `Stay in trade ≈ ${days.toFixed(days >= 2 ? 0 : 1)} day${days >= 2 ? 's' : ''}`;
      }
      if (minutes >= 60) {
        const hours = minutes / 60;
        return `Stay in trade ≈ ${hours.toFixed(hours >= 2 ? 0 : 1)} hour${hours >= 2 ? 's' : ''}`;
      }
      return `Stay in trade ≈ ${minutes.toFixed(0)} minutes`;
    }
    const styleToken = (mergedPlanMeta.style || '').toLowerCase();
    if (styleToken === 'scalp' || styleToken === '0dte') return 'Stay in trade ≈ 30–60 minutes';
    if (styleToken === 'intraday') return 'Stay in trade ≈ 2–4 hours';
    if (styleToken === 'swing') return 'Stay in trade ≈ 3–5 days';
    if (styleToken === 'leaps') return 'Stay in trade: multi-week campaign';
    return null;
  };

  const renderHeader = () => {
    const bias = (mergedPlanMeta.bias || currentPlan.direction || '').toLowerCase();
    if (headerSymbolEl) headerSymbolEl.textContent = symbol;
    if (headerStrategyEl) {
      const styleLabel = mergedPlanMeta.style_display || (mergedPlanMeta.style || '').toUpperCase();
      const strategyLabel = mergedPlanMeta.strategy || currentPlan.strategy || '';
      headerStrategyEl.textContent = [styleLabel, strategyLabel].filter(Boolean).join(' · ');
    }
    if (headerBiasEl) headerBiasEl.textContent = bias ? `Bias: ${bias === 'long' ? 'Long 🟢' : 'Short 🔴'}` : '';
    const rawConfidence =
      toNumber(mergedPlanMeta.confidence) ??
      toNumber(planMeta.confidence_score) ??
      toNumber(planMeta.plan_confidence) ??
      toNumber(planMeta.plan?.confidence) ??
      paramConfidence;
    const displayConfidence = rawConfidence !== null && rawConfidence > 0 ? rawConfidence : null;
    if (headerConfidenceEl) {
      headerConfidenceEl.textContent = displayConfidence !== null ? `Confidence: ${(displayConfidence * 100).toFixed(0)}%` : '';
    }
    const fallbackRRForHeader = (() => {
      const entry =
        Number.isFinite(currentPlan.entry) && Number.isFinite(currentPlan.stop)
          ? currentPlan.entry
          : mergedPlanMeta.entry;
      const stop =
        Number.isFinite(currentPlan.entry) && Number.isFinite(currentPlan.stop)
          ? currentPlan.stop
          : mergedPlanMeta.stop;
      const targetsSource =
        Number.isFinite(currentPlan.entry) && Number.isFinite(currentPlan.stop)
          ? currentPlan.tps
          : Array.isArray(mergedPlanMeta.targets)
            ? mergedPlanMeta.targets
            : [];
      const target = targetsSource?.find((value) => Number.isFinite(value));
      const direction = (mergedPlanMeta.bias || currentPlan.direction || 'long').toLowerCase();
      if (!Number.isFinite(entry) || !Number.isFinite(stop) || !Number.isFinite(target)) return null;
      const risk = direction === 'long' ? entry - stop : stop - entry;
      const reward = direction === 'long' ? target - entry : entry - target;
      if (risk <= 0 || reward <= 0) return null;
      return reward / risk;
    })();
    if (headerRREl) {
      const rrValue = toNumber(mergedPlanMeta.risk_reward);
      const rrDisplay = rrValue !== null && rrValue > 0 ? rrValue : fallbackRRForHeader;
      headerRREl.textContent = rrDisplay !== null ? `R:R (TP1): ${rrDisplay.toFixed(2)}` : '';
    }
    if (headerDurationEl) {
      headerDurationEl.textContent = estimateDuration() || '';
    }
  };

  const renderPlanPanel = (lastPrice) => {
    if (!planPanelBodyEl) return;
    const targetsMeta = Array.isArray(mergedPlanMeta.target_meta) ? mergedPlanMeta.target_meta : [];
    const warnings = Array.isArray(mergedPlanMeta.warnings) ? mergedPlanMeta.warnings : [];

    const targetsList = currentPlan.tps
      .map((tp, idx) => {
        if (!Number.isFinite(tp)) return null;
        const meta = targetsMeta[idx] || {};
        const sequence = Number.isFinite(meta.sequence) ? meta.sequence : idx + 1;
        const label = sequence >= 3 ? 'Runner' : `TP${sequence}`;
        const rrVal = toNumber(meta.rr);
        const rr = rrVal !== null ? ` · R:R ${rrVal.toFixed(2)}` : '';
        const pot = formatProbability(meta.prob_touch);
        const emVal = toNumber(meta.em_fraction);
        const em = emVal !== null ? ` · ${emVal.toFixed(2)}×EM` : '';
        return `<li><strong>${label}:</strong> ${formatPrice(tp)}${em}${pot ? ` · ${pot}` : ''}${rr}</li>`;
      })
      .filter(Boolean)
      .join('');

    const warningsList = warnings
      .map((warning) => `<li>${warning}</li>`)
      .join('');

    const rawConfidence =
      toNumber(mergedPlanMeta.confidence) ??
      toNumber(planMeta.confidence_score) ??
      toNumber(planMeta.plan_confidence) ??
      toNumber(planMeta.plan?.confidence) ??
      paramConfidence;
    const confidenceCopy = rawConfidence !== null && rawConfidence > 0 ? `${(rawConfidence * 100).toFixed(0)}%` : '—';
    const rrValue = toNumber(mergedPlanMeta.risk_reward);
    const rrFallback = (() => {
      const useScaled = Number.isFinite(currentPlan.entry) && Number.isFinite(currentPlan.stop);
      const entry = useScaled ? currentPlan.entry : mergedPlanMeta.entry;
      const stop = useScaled ? currentPlan.stop : mergedPlanMeta.stop;
      const targetPool = useScaled
        ? currentPlan.tps
        : Array.isArray(mergedPlanMeta.targets)
          ? mergedPlanMeta.targets
          : [];
      const target = targetPool?.find((value) => Number.isFinite(value));
      const direction = (mergedPlanMeta.bias || currentPlan.direction || 'long').toLowerCase();
      if (!Number.isFinite(entry) || !Number.isFinite(stop) || !Number.isFinite(target)) return null;
      const risk = direction === 'long' ? entry - stop : stop - entry;
      const reward = direction === 'long' ? target - entry : entry - target;
      if (risk <= 0 || reward <= 0) return null;
      return reward / risk;
    })();
    const rrCopy = rrValue !== null && rrValue > 0 ? rrValue.toFixed(2) : rrFallback !== null ? rrFallback.toFixed(2) : '—';
    const runnerNote = mergedPlanMeta.runner && mergedPlanMeta.runner.note ? mergedPlanMeta.runner.note : null;

    const lastPriceCopy = Number.isFinite(lastPrice) ? formatPrice(lastPrice) : '—';
    const replayMinutesValue = clampReplayMinutes(replayConfig.minutes);
    replayConfig.minutes = replayMinutesValue;
    const replayStatusText = replayStatusMessage || (isReplaying ? 'Replay in progress…' : '');
    const startDisabledAttr = isReplaying ? 'disabled' : '';
    const stopDisabledAttr = isReplaying ? '' : 'disabled';

    planPanelBodyEl.innerHTML = `
      <div class="plan-panel__section">
        <div class="plan-metrics">
          <span><small>Entry</small><strong>${formatPrice(mergedPlanMeta.entry)}</strong></span>
          <span><small>Stop</small><strong>${formatPrice(mergedPlanMeta.stop)}</strong></span>
          <span><small>Last Price</small><strong id="plan_last_price_value">${lastPriceCopy}</strong></span>
          <span><small>Confidence</small><strong>${confidenceCopy}</strong></span>
          <span><small>R:R (TP1)</small><strong>${rrCopy}</strong></span>
        </div>
      </div>
      <div class="plan-panel__section">
        <h3>Targets</h3>
        <ul class="plan-panel__targets">
          ${targetsList || '<li>No targets supplied.</li>'}
        </ul>
      </div>
      ${
        runnerNote
          ? `<div class="plan-panel__section">
              <h3>Runner Guidance</h3>
              <p>${runnerNote}</p>
            </div>`
          : ''
      }
      <div class="plan-panel__section">
        <h3>Pre-Entry Checklist</h3>
        <ul class="plan-panel__warnings">
          ${
            warningsList ||
            '<li>Confirm structure alignment, volume tone, and that price reclaims entry trigger before committing capital.</li>'
          }
        </ul>
      </div>
      <div class="plan-panel__section plan-replay">
        <h3>Market Replay</h3>
        <div class="plan-replay__controls">
          <input
            id="market_replay_minutes"
            class="plan-replay__input"
            type="number"
            min="1"
            max="${REPLAY_MAX_MINUTES}"
            step="1"
            value="${replayMinutesValue}"
            aria-label="Minutes to replay"
          />
          <button id="market_replay_start" type="button" class="plan-replay__button" ${startDisabledAttr}>Start Replay</button>
          <button id="market_replay_stop" type="button" class="plan-replay__button" ${stopDisabledAttr}>Stop</button>
        </div>
        <p id="market_replay_status" class="plan-replay__status">${replayStatusText}</p>
      </div>
      ${
        mergedPlanMeta.notes
          ? `<div class="plan-panel__section">
              <h3>Plan Notes</h3>
              <p>${mergedPlanMeta.notes}</p>
            </div>`
          : ''
      }
    `;
    if (planPanelEl) {
      if (window.innerWidth <= 1024) {
        planPanelEl.open = false;
      } else {
        planPanelEl.open = true;
      }
    }
    attachReplayControls();
    updatePlanPanelLastPrice(lastPrice);
  };

  const updatePlanPanelLastPrice = (value) => {
    const lastEl = document.getElementById('plan_last_price_value');
    if (!lastEl) return;
    lastEl.textContent = Number.isFinite(value) ? formatPrice(value) : '—';
  };

  const updateRealtimeBar = (price, payload) => {
    if (!latestCandleData.length || !Number.isFinite(price)) return;
    const secondsPerBar = Math.max(resolutionToSeconds(currentResolution), 60);
    const eventTs = parseEventTimestamp(payload) ?? Math.floor(Date.now() / 1000);
    const bucketTime = Math.floor(eventTs / secondsPerBar) * secondsPerBar;

    const lastIdx = latestCandleData.length - 1;
    const lastBar = latestCandleData[lastIdx];
    if (!lastBar || !Number.isFinite(lastBar?.time)) {
      return;
    }

    if (bucketTime < lastBar.time) {
      // Out-of-order tick; ignore to avoid rewinding the chart.
      return;
    }

    if (bucketTime === lastBar.time) {
      const updated = {
        ...lastBar,
        close: price,
        high: Math.max(Number.isFinite(lastBar.high) ? lastBar.high : price, price),
        low: Math.min(Number.isFinite(lastBar.low) ? lastBar.low : price, price),
      };
      latestCandleData[lastIdx] = updated;
      candleSeries.update(updated);

      const volBar = latestVolumeData[latestVolumeData.length - 1];
      if (volBar) {
        const updatedVol = {
          ...volBar,
          color: updated.close >= updated.open ? '#22c55e88' : '#ef444488',
        };
        latestVolumeData[latestVolumeData.length - 1] = updatedVol;
        volumeSeries.update(updatedVol);
      }
      return;
    }

    // Advance to a new bar bucket.
    const newBar = {
      time: bucketTime,
      open: price,
      high: price,
      low: price,
      close: price,
    };
    latestCandleData.push(newBar);
    candleSeries.update(newBar);

    const newVolumeBar = {
      time: bucketTime,
      value: 0,
      color: price >= newBar.open ? '#22c55e88' : '#ef444488',
    };
    latestVolumeData.push(newVolumeBar);
    volumeSeries.update(newVolumeBar);
  };

  const setReplayStatusMessage = (message) => {
    replayStatusMessage = message || '';
    if (replayStatusEl) {
      replayStatusEl.textContent = replayStatusMessage;
    }
  };

  const syncReplayControls = () => {
    if (replayMinutesInput) {
      if (!isReplaying) {
        replayMinutesInput.value = clampReplayMinutes(replayConfig.minutes);
      }
      replayMinutesInput.disabled = isReplaying;
    }
    if (replayStartButton) replayStartButton.disabled = isReplaying;
    if (replayStopButton) replayStopButton.disabled = !isReplaying;
    if (replayStatusEl) {
      replayStatusEl.textContent = replayStatusMessage;
    }
  };

  const fetchReplayBars = async (minutes) => {
    const minutesClamped = clampReplayMinutes(minutes);
    const resolutionToken = currentResolution || '1';
    const secondsPerBar = Math.max(resolutionToSeconds(resolutionToken), 60);
    const nowSec = Math.floor(Date.now() / 1000);
    const minBarCount = Math.max(Math.ceil((minutesClamped * 60) / secondsPerBar), 1);
    const targetBarCount = Math.max(minBarCount, 2);
    const spanSeconds = Math.max(secondsPerBar * (targetBarCount + 1), secondsPerBar * 2);
    const from = nowSec - spanSeconds;
    const qs = new URLSearchParams({
      symbol,
      resolution: resolutionToken,
      from: String(from),
      to: String(nowSec),
    });
    const response = await fetch(`${baseUrl}/tv-api/bars?${qs.toString()}`, { cache: 'no-store' });
    if (!response.ok) {
      throw new Error(`Replay data request failed (${response.status})`);
    }
    const payload = await response.json();
    if (payload.s !== 'ok' || !Array.isArray(payload.t) || !payload.t.length) {
      return [];
    }
    const bars = payload.t.map((time, idx) => ({
      time: payload.t[idx],
      open: payload.o[idx],
      high: payload.h[idx],
      low: payload.l[idx],
      close: payload.c[idx],
      volume: Array.isArray(payload.v) ? payload.v[idx] || 0 : 0,
    }));
    bars.sort((a, b) => a.time - b.time);
    const cutoff = nowSec - minutesClamped * 60;
    const recent = bars.filter((bar) => bar.time >= cutoff);
    if (recent.length >= 2) {
      return recent.slice(-targetBarCount);
    }
    if (recent.length === 1) {
      const lastIndex = bars.findIndex((bar) => bar.time === recent[0].time);
      if (lastIndex > 0) {
        return bars.slice(Math.max(0, lastIndex - (targetBarCount - 1)), lastIndex + 1);
      }
      return recent;
    }
    return bars.slice(-targetBarCount);
  };

  const stopMarketReplay = ({
    restoreChart = true,
    resumeStream = true,
    message = '',
    skipFetch = false,
  } = {}) => {
    if (replayTimer) {
      window.clearTimeout(replayTimer);
      replayTimer = null;
    }
    replayFetchToken += 1;
    const wasReplaying = isReplaying;
    isReplaying = false;
    replayQueue = [];
    replayIndex = 0;

    if (restoreChart && replaySavedCandleData.length) {
      candleSeries.setData(replaySavedCandleData);
      volumeSeries.setData(replaySavedVolumeData);
      if (replaySavedVisibleRange) {
        try {
          chart.timeScale().setVisibleRange(replaySavedVisibleRange);
        } catch {
          chart.timeScale().fitContent();
        }
      } else {
        chart.timeScale().fitContent();
      }
    }
    replaySavedCandleData = [];
    replaySavedVolumeData = [];
    replaySavedVisibleRange = null;

    if (replayPrevPhase || replayPrevNote) {
      applyMarketStatus(replayPrevPhase || 'closed', replayPrevNote || undefined);
    }
    replayPrevPhase = null;
    replayPrevNote = null;

    if (resumeStream && replayHadStream) {
      connectStream();
    }
    replayHadStream = false;

    if (message) {
      setReplayStatusMessage(message);
    } else if (wasReplaying) {
      setReplayStatusMessage('');
    } else {
      setReplayStatusMessage(replayStatusMessage);
    }
    syncReplayControls();

    if (!skipFetch) {
      fetchBars();
    }
  };

  const playNextReplayBar = () => {
    if (!isReplaying) return;
    if (replayIndex >= replayQueue.length) {
      stopMarketReplay({
        restoreChart: true,
        resumeStream: true,
        message: 'Replay complete. Resuming live stream…',
      });
      return;
    }
    const bar = replayQueue[replayIndex];
    candleSeries.update({
      time: bar.time,
      open: bar.open,
      high: bar.high,
      low: bar.low,
      close: bar.close,
    });
    volumeSeries.update({
      time: bar.time,
      value: bar.volume || 0,
      color: bar.close >= bar.open ? '#22c55e88' : '#ef444488',
    });
    lastKnownPrice = bar.close;
    updateHeaderPricing(lastKnownPrice);
    updatePlanPanelLastPrice(lastKnownPrice);
    replayIndex += 1;
    setReplayStatusMessage(`Playing ${Math.min(replayIndex, replayQueue.length)}/${replayQueue.length}`);
    syncReplayControls();
    replayTimer = window.setTimeout(playNextReplayBar, REPLAY_STEP_MS);
  };

  const startMarketReplay = async () => {
    if (isReplaying) return;
    const minutesRaw = replayMinutesInput ? replayMinutesInput.value : replayConfig.minutes;
    const minutes = clampReplayMinutes(minutesRaw);
    replayConfig.minutes = minutes;
    if (replayMinutesInput) replayMinutesInput.value = minutes;

    replayFetchToken += 1;
    const token = replayFetchToken;
    isReplaying = true;
    replayQueue = [];
    replayIndex = 0;
    replayPrevPhase = currentMarketPhase;
    replayPrevNote = latestMarketNote;
    replayHadStream = Boolean(streamSource);
    if (streamSource) {
      try {
        streamSource.close();
      } catch {
        // ignore
      }
      streamSource = null;
    }
    const tfLabel = (activeTimeframe && activeTimeframe.label) || currentResolution || '1m';
    applyMarketStatus('replay', `Replaying last ${minutes} minutes using ${tfLabel} bars.`);
    setReplayStatusMessage(`Preparing ${tfLabel} replay…`);
    syncReplayControls();

    try {
      const bars = await fetchReplayBars(minutes);
      if (!isReplaying || token !== replayFetchToken) {
        return;
      }
      if (!Array.isArray(bars) || bars.length === 0) {
        stopMarketReplay({
          restoreChart: true,
          resumeStream: true,
          message: 'No intraday data available for that window.',
          skipFetch: true,
        });
        return;
      }

      replaySavedCandleData = latestCandleData.map((item) => ({ ...item }));
      replaySavedVolumeData = latestVolumeData.map((item) => ({ ...item }));
      try {
        replaySavedVisibleRange = chart.timeScale().getVisibleRange();
      } catch {
        replaySavedVisibleRange = null;
      }

      replayQueue = bars;
      const firstBar = replayQueue[0];
      candleSeries.setData([
        {
          time: firstBar.time,
          open: firstBar.open,
          high: firstBar.high,
          low: firstBar.low,
          close: firstBar.close,
        },
      ]);
      volumeSeries.setData([
        {
          time: firstBar.time,
          value: firstBar.volume || 0,
          color: firstBar.close >= firstBar.open ? '#22c55e88' : '#ef444488',
        },
      ]);
      lastKnownPrice = firstBar.close;
      updateHeaderPricing(lastKnownPrice);
      updatePlanPanelLastPrice(lastKnownPrice);
      try {
        chart.timeScale().setVisibleRange(replaySavedVisibleRange || {
          from: replayQueue[0].time,
          to: replayQueue[replayQueue.length - 1].time,
        });
      } catch {
        try {
          chart.timeScale().setVisibleRange({
            from: replayQueue[0].time,
            to: replayQueue[replayQueue.length - 1].time,
          });
        } catch {
          chart.timeScale().fitContent();
        }
      }

      replayIndex = 1;
      setReplayStatusMessage(`Playing ${Math.min(replayIndex, replayQueue.length)}/${replayQueue.length}`);
      syncReplayControls();
      replayTimer = window.setTimeout(playNextReplayBar, REPLAY_STEP_MS);
    } catch (err) {
      console.error('Market replay failed', err);
      stopMarketReplay({
        restoreChart: true,
        resumeStream: true,
        message: `Replay failed: ${err.message || err}`,
      });
    }
  };

  const attachReplayControls = () => {
    replayStatusEl = document.getElementById('market_replay_status');
    replayStartButton = document.getElementById('market_replay_start');
    replayStopButton = document.getElementById('market_replay_stop');
    replayMinutesInput = document.getElementById('market_replay_minutes');

    if (replayMinutesInput) {
      replayMinutesInput.value = clampReplayMinutes(replayConfig.minutes);
      replayMinutesInput.addEventListener('change', () => {
        const updated = clampReplayMinutes(replayMinutesInput.value);
        replayConfig.minutes = updated;
        replayMinutesInput.value = updated;
      });
      replayMinutesInput.addEventListener('input', () => {
        replayConfig.minutes = clampReplayMinutes(replayMinutesInput.value);
      });
    }

    if (replayStartButton) {
      replayStartButton.addEventListener('click', (event) => {
        event.preventDefault();
        startMarketReplay();
      });
    }
    if (replayStopButton) {
      replayStopButton.addEventListener('click', (event) => {
        event.preventDefault();
        if (isReplaying) {
          stopMarketReplay({
            restoreChart: true,
            resumeStream: true,
            message: 'Replay stopped. Resuming live stream…',
          });
        }
      });
    }
    syncReplayControls();
  };

  const updateHeaderPricing = (lastPrice) => {
    if (headerLastPriceEl) {
      headerLastPriceEl.textContent = Number.isFinite(lastPrice) ? `Last: ${formatPrice(lastPrice)}` : '';
    }
  };

  const rangeTokenRaw = (params.get('range') || '').toLowerCase();

  const resolveSpanSeconds = () => {
    const base = resolutionToSeconds(currentResolution) * 600;
    if (!rangeTokenRaw) return base;

    if (/^\d+$/.test(rangeTokenRaw)) {
      const barsCount = parseInt(rangeTokenRaw, 10);
      if (Number.isFinite(barsCount) && barsCount > 0) {
        return resolutionToSeconds(currentResolution) * barsCount;
      }
    }

    const match = rangeTokenRaw.match(/^(\d+)([dwmy])$/);
    if (match) {
      const value = parseInt(match[1], 10);
      const unit = match[2];
      const UNIT_SECONDS = { d: 86400, w: 604800, m: 2629800, y: 31557600 };
      const mult = UNIT_SECONDS[unit];
      if (mult && Number.isFinite(value) && value > 0) {
        return value * mult;
      }
    }

    if (rangeTokenRaw === 'fit') {
      return base;
    }

    return base;
  };

  const fetchBars = async () => {
    if (isReplaying) return;
    const token = ++fetchToken;
    try {
      const now = Math.floor(Date.now() / 1000);
      const minSpan = resolutionToSeconds(currentResolution) * 200;
      const maxSpan = 60 * 60 * 24 * 365 * 2;
      const span = Math.min(Math.max(resolveSpanSeconds(), minSpan), maxSpan);
      const from = now - span;
      const qs = new URLSearchParams({
        symbol,
        resolution: currentResolution,
        from: String(from),
        to: String(now),
      }).toString();
      const response = await fetch(`${baseUrl}/tv-api/bars?${qs}`);
      if (!response.ok) throw new Error(`bars request failed (${response.status})`);
      const payload = await response.json();
      if (payload.s !== 'ok' || !payload.t?.length) throw new Error('No data');
      if (token !== fetchToken || isReplaying) {
        return;
      }

      const bars = payload.t.map((time, idx) => ({
        time: payload.t[idx],
        open: payload.o[idx],
        high: payload.h[idx],
        low: payload.l[idx],
        close: payload.c[idx],
        volume: payload.v[idx] || 0,
      }));

      const planForFrame = clonePlan();

      const candleData = bars.map((bar) => ({
        time: bar.time,
        open: bar.open,
        high: bar.high,
        low: bar.low,
        close: bar.close,
      }));
      candleSeries.setData(candleData);
      latestCandleData = candleData.map((bar) => ({ ...bar }));

      const volumeData = bars.map((bar) => ({
        time: bar.time,
        value: bar.volume,
        color: bar.close >= bar.open ? '#22c55e88' : '#ef444488',
      }));
      volumeSeries.setData(volumeData);
      latestVolumeData = volumeData.map((bar) => ({ ...bar }));

      const lastPrice = bars[bars.length - 1]?.close ?? null;
      lastKnownPrice = lastPrice;
      updateHeaderPricing(lastPrice);
      renderPlanPanel(lastPrice);

      let scaleMultiplier = 1;
      if (scalePlanToken !== 'off') {
        if (scalePlanToken === 'auto') {
          if (Number.isFinite(planForFrame.entry) && Number.isFinite(lastPrice) && planForFrame.entry && lastPrice) {
            const ratio = lastPrice / planForFrame.entry;
            if (ratio > 1.2 || ratio < 0.8) {
              scaleMultiplier = ratio;
            }
          }
        } else {
          const manual = parseFloat(scalePlanToken);
          if (Number.isFinite(manual) && manual > 0) {
            scaleMultiplier = manual;
          }
        }
      }

      if (scaleMultiplier !== 1 && Number.isFinite(scaleMultiplier) && scaleMultiplier > 0) {
        if (Number.isFinite(planForFrame.entry)) planForFrame.entry *= scaleMultiplier;
        if (Number.isFinite(planForFrame.stop)) planForFrame.stop *= scaleMultiplier;
        planForFrame.tps = planForFrame.tps.map((tp) => (Number.isFinite(tp) ? tp * scaleMultiplier : tp));
        if (planForFrame.runner && Number.isFinite(planForFrame.runner.anchor)) {
          planForFrame.runner.anchor *= scaleMultiplier;
        }
      }

      currentPlan = planForFrame;

      const overlayValues = [
        planForFrame.entry,
        planForFrame.stop,
        ...planForFrame.tps,
        ...keyLevels.map((lvl) => lvl.price),
      ].filter((value) => Number.isFinite(value));

      const vwapRequested = (params.get('vwap') || '').toLowerCase() !== 'false';
      if (vwapRequested && !vwapSeries) {
        vwapSeries = chart.addLineSeries({
          lineWidth: 2,
          color: '#f8fafc',
          title: 'VWAP',
        });
      }
      let lastVwap = null;
      if (vwapSeries) {
        const values = [];
        let cumulativePV = 0;
        let cumulativeVol = 0;
        let currentSession = null;
        for (let i = 0; i < candleData.length; i += 1) {
          const bar = candleData[i];
          const vol = volumeData[i]?.value || 0;
          const typical = (bar.high + bar.low + bar.close) / 3;
          const sessionKey = new Date(bar.time * 1000).toISOString().slice(0, 10);
          if (sessionKey !== currentSession) {
            currentSession = sessionKey;
            cumulativePV = 0;
            cumulativeVol = 0;
          }
          cumulativePV += typical * vol;
          cumulativeVol += vol;
          const vwapValue = cumulativeVol > 0 ? cumulativePV / cumulativeVol : typical;
          if (Number.isFinite(vwapValue)) {
            lastVwap = vwapValue;
          }
          values.push({ time: bar.time, value: vwapValue });
        }
        vwapSeries.setData(values);
      }
      if (Number.isFinite(lastVwap)) {
        overlayValues.push(lastVwap);
      }

      emaSeries.forEach(({ span, series }) => {
        const values = [];
        const weight = 2 / (span + 1);
        let emaValue = null;
        for (let i = 0; i < candleData.length; i += 1) {
          const close = candleData[i].close;
          if (close == null) continue;
          if (emaValue === null) {
            emaValue = close;
          } else {
            emaValue = close * weight + emaValue * (1 - weight);
          }
          values.push({ time: candleData[i].time, value: emaValue });
        }
        series.setData(values);
      });

      const activeLineIds = new Set();
      const registerLine = (id, options) => {
        const line = setPriceLine(id, options);
        if (line) {
          activeLineIds.add(id);
        }
      };

      registerLine('entry', {
        price: planForFrame.entry,
        color: '#facc15',
        title: 'Entry',
        lineWidth: 2,
        lineStyle: LightweightCharts.LineStyle.Solid,
      });

      registerLine('stop', {
        price: planForFrame.stop,
        color: '#ef4444',
        title: 'Stop',
        lineWidth: 2,
        lineStyle: LightweightCharts.LineStyle.Solid,
      });

      planForFrame.tps.forEach((tp, idx) => {
        if (!Number.isFinite(tp)) return;
        const meta = Array.isArray(mergedPlanMeta.target_meta) ? mergedPlanMeta.target_meta[idx] || {} : {};
        const sequence = Number.isFinite(meta.sequence) ? meta.sequence : idx + 1;
        const isRunner = sequence >= 3;
        const label = isRunner ? 'Runner' : meta.label || `TP${sequence}`;
        const color = isRunner ? '#c084fc' : '#22c55e';
        const id = isRunner ? 'runner-tp' : `tp:${sequence}`;
        registerLine(id, {
          price: tp,
          color,
          title: label,
          lineWidth: 2,
          lineStyle: LightweightCharts.LineStyle.Dashed,
        });
      });

      if (Number.isFinite(lastVwap)) {
        registerLine('vwap', {
          price: lastVwap,
          color: '#f8fafc',
          title: 'VWAP',
          lineWidth: 2,
          lineStyle: LightweightCharts.LineStyle.Solid,
        });
      }

      [...keyLevels]
        .filter((level) => Number.isFinite(level.price))
        .sort((a, b) => b.price - a.price)
        .forEach((level, idx) => {
          const label = level.label ? level.label : `Level ${idx + 1}`;
          const key = `level:${label}:${level.price.toFixed(4)}`;
          registerLine(key, {
            price: level.price,
            color: '#94a3b8',
            title: label,
            lineWidth: 1,
            lineStyle: LightweightCharts.LineStyle.Dotted,
          });
        });
      if (planForFrame.runner && Number.isFinite(planForFrame.runner.anchor)) {
        const runnerLabel = planForFrame.runner.label || 'Runner Trail';
        registerLine('runner-anchor', {
          price: planForFrame.runner.anchor,
          color: '#ff4fa3',
          title: runnerLabel,
          lineWidth: 2,
          lineStyle: LightweightCharts.LineStyle.Dashed,
        });
      }

      prunePriceLines(activeLineIds);

      candleSeries.setMarkers([]);

      chart.priceScale('right').applyOptions({ autoScale: true });

      debugEl.style.display = 'none';
      debugEl.textContent = '';
    } catch (err) {
      debugEl.style.display = 'block';
      debugEl.textContent = `Error loading data: ${err.message}`;
    }
  };

  renderHeader();
  initializeTimeframes();
  setWatermark();
  fetchBars();
  connectStream();
  window.addEventListener('resize', () => {
    chart.resize(container.clientWidth, container.clientHeight);
    if (planPanelEl && window.innerWidth > 1024) {
      planPanelEl.open = true;
    }
    renderPlanPanel(lastKnownPrice);
  });
  window.setInterval(fetchBars, TIMEFRAME_REFRESH_MS);
  window.addEventListener('beforeunload', () => {
    if (streamSource) {
      try {
        streamSource.close();
      } catch (err) {
        // ignore
      }
    }
  });
})();
