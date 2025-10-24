import { loadDiagnostics } from './diag.js';

const routes = {
  validate: '/api/v1/plan/validate',
  sizing: '/api/v1/sizing/suggest',
  alerts: '/api/v1/alerts/set',
  coach: '/api/v1/coach/chat',
  order: '/api/v1/broker_tradier/place',
};

const budgets = {
  sizing: 800,
  validate: 800,
  alerts: 500,
  coach: 1200,
  order: 1200,
};

export async function mountActionsDock(rootSel, ctx = {}) {
  let root = document.querySelector(rootSel);
  if (!root) throw new Error(`Actions dock root not found: ${rootSel}`);

  if (root.__actionsHandler) {
    root.removeEventListener('click', root.__actionsHandler);
  }

  root.innerHTML = '';
  const { symbol, session } = ctx;

  const diagnostics = ctx.diagnostics || (await loadDiagnostics());
  const caps = readCaps(diagnostics);

  const buttons = [
    makeBtn('Size It', 'sizing'),
    makeBtn('Validate Plan', 'validate'),
    makeBtn('Set Alert', 'alerts'),
    makeBtn('Ask Coach', 'coach'),
    makeBtn('Place Order', 'order', {
      disabled: !caps.broker.ready,
      reason: caps.broker.reason,
    }),
  ];

  buttons.forEach((button) => root.appendChild(button));

  const handler = async (event) => {
    const btn = event.target.closest('button[data-action]');
    if (!btn || btn.disabled) return;
    const action = btn.dataset.action;

    btn.disabled = true;
    const payload = buildPayload(action, { symbol, session });
    if (payload === null) {
      btn.disabled = false;
      return;
    }

    try {
      const duration = await post(routes[action], payload);
      noteLatency(action, duration);
    } catch (error) {
      console.error(`[actions] ${action} failed`, error);
    } finally {
      btn.disabled = false;
    }
  };

  root.addEventListener('click', handler);
  root.__actionsHandler = handler;
}

function makeBtn(label, action, opts = {}) {
  const button = document.createElement('button');
  button.type = 'button';
  button.textContent = label;
  button.dataset.action = action;
  if (opts.disabled) {
    button.disabled = true;
    button.title = opts.reason || 'Unavailable';
    button.setAttribute('aria-disabled', 'true');
  } else {
    button.removeAttribute('aria-disabled');
    button.removeAttribute('title');
  }
  return button;
}

function buildPayload(action, { symbol, session }) {
  switch (action) {
    case 'sizing':
      return { symbol, risk_budget: '1R', session };
    case 'validate':
      return { symbol, session };
    case 'alerts':
      return { symbol, preset: 'plusMinus1R', session };
    case 'coach':
      return { symbol, ask: 'checklist', session };
    case 'order':
      return confirm(
        'Place staged order? Orders route via sandbox until broker is verified.'
      )
        ? { symbol, side: 'buy', qty: 1, sandbox: true }
        : null;
    default:
      return { symbol };
  }
}

async function post(url, body) {
  const started = performance.now();
  const response = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });

  const duration = performance.now() - started;
  console.debug('[actions] POST', url, 'duration=', Math.round(duration));

  if (!response.ok) {
    throw new Error(`${url} failed with ${response.status}`);
  }

  try {
    await response.json();
  } catch {
    // No body returned — acceptable.
  }

  return duration;
}

function noteLatency(action, durationMs) {
  const budget = budgets[action];
  if (!budget) return;
  if (durationMs > budget) {
    console.warn(`[actions] ${action} exceeded budget ${Math.round(durationMs)}ms > ${budget}ms`);
  }
}

function readCaps(diagnostics = {}) {
  const health = diagnostics.health || {};
  const ready = diagnostics.ready || {};
  const brokerOk =
    !!health?.providers?.tradier?.ok ||
    !!health?.caps?.broker_tradier ||
    !!ready?.caps?.broker_tradier;

  return {
    broker: {
      ready: brokerOk,
      reason: brokerOk ? '' : 'Connect broker in Settings',
    },
  };
}
