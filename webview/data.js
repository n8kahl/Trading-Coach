const PLAN_ENDPOINT = '/api/v1/plan/current';

export async function fetchPlan({ symbol }) {
  const url = new URL(PLAN_ENDPOINT, window.location.origin);
  if (symbol) url.searchParams.set('symbol', symbol);

  const response = await fetch(url.toString(), {
    headers: { Accept: 'application/json' },
    credentials: 'same-origin',
  });

  if (!response.ok) {
    throw new Error(`Plan fetch failed (${response.status})`);
  }

  const payload = await response.json();
  return normalizePlan(payload);
}

export function normalizePlan(payload) {
  const plan = payload?.plan || {};
  const charts = payload?.charts || {};
  const params = charts?.params || {};
  const uiState = safeJson(
    params.ui_state,
    payload?.session_state || { session: 'live', style: 'intraday' }
  );

  const symbol =
    payload?.symbol ||
    plan?.symbol ||
    params?.symbol ||
    payload?.charts?.symbol ||
    'SPY';

  const entry = num(plan.entry ?? params.entry);
  const stop = num(plan.stop ?? params.stop);
  const targets =
    Array.isArray(plan.targets) && plan.targets.length
      ? plan.targets.map(num).filter(isFinite)
      : String(params.tp || '')
          .split(',')
          .map(num)
          .filter(isFinite);

  const rr = computeRR(entry, stop, targets.at(0));
  const runnerPolicy = plan.runner_policy || plan.runner || {};
  const runners = normalizeRunners(plan.runners, runnerPolicy);

  const rawConfidence = firstFinite(
    plan.confidence,
    payload?.confidence,
    uiState?.confidence,
    0
  );
  const confidence = clamp01(rawConfidence <= 1 ? rawConfidence : rawConfidence / 100);

  const confluence = extractConfidenceComponents(payload, plan);
  const evidence = extractEvidence(plan, payload, confluence);

  const levelsTokens = String(params.levels || '');
  const supportingLevels = params.supportingLevels ?? '1';

  const dataAge =
    parseDate(plan.as_of) ||
    parseDate(payload?.as_of) ||
    parseDate(charts?.meta?.as_of) ||
    null;

  return {
    symbol,
    entry,
    stop,
    tps: targets,
    rr,
    runners,
    confidence,
    confluence,
    evidence,
    levelsTokens,
    supportingLevels,
    uiState: {
      session: uiState?.session || payload?.session_state?.status || 'live',
      style: uiState?.style || plan?.style || payload?.style || 'intraday',
      confidence: confidence,
    },
    charts,
    dataAge,
    raw: payload,
  };
}

function normalizeRunners(runners, runnerPolicy) {
  if (Array.isArray(runners) && runners.length) {
    return runners;
  }

  const fraction = runnerPolicy?.fraction ?? runnerPolicy?.runner_fraction;
  const trail = runnerPolicy?.trail ?? runnerPolicy?.atr_trail_mult;
  const notes = runnerPolicy?.notes;

  const out = [];
  if (isFinite(fraction)) out.push(`${Math.round(fraction * 100)}% trail`);
  if (isFinite(trail)) out.push(`ATR x${trail}`);
  if (notes && Array.isArray(notes)) out.push(...notes);
  return out;
}

function extractConfidenceComponents(payload, plan) {
  const source =
    plan?.confidence_components ||
    plan?.confidence_breakdown ||
    payload?.confidence_components ||
    payload?.plan?.confidence_breakdown ||
    {};

  const lookup = (keys, fallback = 0) => {
    for (const key of keys) {
      const value = source?.[key];
      const n = num(value);
      if (isFinite(n)) return n;
    }
    return fallback;
  };

  return {
    atr: lookup(['atr', 'atr_score', 'volatility']),
    vwap: lookup(['vwap', 'vwap_score']),
    emas: lookup(['emas', 'ema', 'ema_score']),
    orderflow: lookup(['order_flow', 'orderflow', 'flow']),
    liquidity: lookup(['liquidity', 'liquidity_score']),
    why:
      plan?.confluence?.why ||
      payload?.confluence?.why ||
      plan?.notes ||
      payload?.plan?.notes ||
      'Model alignment across frames.',
    htf_bias: plan?.htf?.bias || plan?.bias || payload?.bias || null,
    vol_regime:
      plan?.volatility_regime ||
      payload?.volatility_regime ||
      payload?.plan?.volatility_regime ||
      null,
    snap: Array.isArray(plan?.snap_trace) ? plan.snap_trace[0] : null,
  };
}

function extractEvidence(plan, payload, components = {}) {
  const confluence = plan?.confluence || payload?.confluence || {};
  return {
    htf_bias: components.htf_bias || confluence.htf_bias || plan?.htf?.bias || null,
    vol_regime: components.vol_regime || confluence.vol_regime || null,
    snap: components.snap || confluence.snap || (plan?.snap_trace || [])[0] || null,
    why: components.why || confluence.why,
  };
}

function num(value) {
  const n = Number(value);
  return Number.isFinite(n) ? n : NaN;
}

function firstFinite(...values) {
  for (const value of values) {
    const n = num(value);
    if (Number.isFinite(n)) return n;
  }
  return NaN;
}

function computeRR(entry, stop, tp1) {
  if (!isFinite(entry) || !isFinite(stop) || !isFinite(tp1) || entry === stop) return null;
  const risk = Math.abs(entry - stop);
  const reward = Math.abs(tp1 - entry);
  if (!risk) return null;
  return +(reward / risk).toFixed(2);
}

function clamp01(value) {
  if (!isFinite(value)) return 0;
  return Math.min(1, Math.max(0, value));
}

function parseDate(value) {
  if (!value) return null;
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? null : date;
}

function safeJson(raw, defaultValue) {
  if (!raw) return defaultValue;
  try {
    return JSON.parse(raw);
  } catch {
    return defaultValue;
  }
}
