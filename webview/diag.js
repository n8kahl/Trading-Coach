const markRegistry = new Map();
const budgets = {
  'webview:init': 400,
  'ws:connect': 500,
  'plan:fetch': 800,
};

let diagnosticsCache = null;

export const Perf = {
  mark(name) {
    const markName = `${name}::start::${Date.now()}`;
    markRegistry.set(name, markName);
    performance.mark(markName);
  },
  end(name) {
    const startMark = markRegistry.get(name);
    if (!startMark) return;
    const endMark = `${name}::end::${Date.now()}`;
    performance.mark(endMark);
    performance.measure(name, startMark, endMark);
    markRegistry.delete(name);
  },
  last(name) {
    const entries = performance.getEntriesByName(name);
    const last = entries?.[entries.length - 1];
    return last?.duration ?? 0;
  },
  clear(name) {
    performance.clearMeasures(name);
  },
};

export async function loadDiagnostics(force = false) {
  if (diagnosticsCache && !force) return diagnosticsCache;
  diagnosticsCache = await fetchDiagnostics();
  return diagnosticsCache;
}

export async function mountObservabilityPanel(rootSel) {
  const root = document.querySelector(rootSel);
  if (!root) return null;

  const diagnostics = await loadDiagnostics();

  root.innerHTML = `
    <details class="diag-details" open>
      <summary>Diagnostics</summary>
      <div class="diag-metrics">
        <ul class="kv" role="list">
          <li data-metric="init"><span>Init</span><strong id="obsInit">—</strong></li>
          <li data-metric="ws"><span>WS</span><strong id="obsWS">—</strong></li>
          <li data-metric="plan"><span>Plan</span><strong id="obsPlan">—</strong></li>
          <li data-metric="data"><span>Data Age</span><strong id="obsDataAge">—</strong></li>
        </ul>
        <p class="muted diag-line">Providers: <span id="diagProviders">—</span></p>
        <p class="muted diag-line">Routes: <span id="diagRoutes">—</span></p>
      </div>
    </details>
  `;

  updateDiagnostics(diagnostics);
  updatePerf();

  return diagnostics;
}

export function updatePerf() {
  applyMetric('obsInit', Perf.last('webview:init'), budgets['webview:init']);
  applyMetric('obsWS', Perf.last('ws:connect'), budgets['ws:connect']);
  applyMetric('obsPlan', Perf.last('plan:fetch'), budgets['plan:fetch']);
}

export function updateDataAge(date) {
  const el = document.getElementById('obsDataAge');
  if (!el) return;
  if (!(date instanceof Date) || Number.isNaN(date.getTime())) {
    el.textContent = '—';
    el.removeAttribute('data-age');
    return;
  }
  el.textContent = formatRelative(date);
  el.dataset.age = date.toISOString();
}

function applyMetric(id, duration, budget) {
  const el = document.getElementById(id);
  if (!el) return;
  if (!duration) {
    el.textContent = '—';
    el.classList.remove('over-budget');
    return;
  }
  el.textContent = `${Math.round(duration)} ms`;
  if (budget && duration > budget) {
    el.classList.add('over-budget');
  } else {
    el.classList.remove('over-budget');
  }
}

function updateDiagnostics(diagnostics) {
  const providersEl = document.getElementById('diagProviders');
  const routesEl = document.getElementById('diagRoutes');

  if (providersEl) {
    providersEl.textContent = formatProviders(diagnostics?.health?.providers);
  }
  if (routesEl) {
    const routes = diagnostics?.routes;
    if (Array.isArray(routes)) {
      routesEl.textContent = `${routes.length}`;
    } else if (routes && typeof routes === 'object') {
      routesEl.textContent = `${Object.keys(routes).length}`;
    } else {
      routesEl.textContent = '—';
    }
  }
}

async function fetchDiagnostics() {
  const [health, ready, routes] = await Promise.all([
    fetchJSON('/api/v1/diag/health'),
    fetchJSON('/api/v1/diag/ready'),
    fetchJSON('/api/v1/diag/routes'),
  ]);

  return {
    health: health || {},
    ready: ready || {},
    routes: routes || [],
    fetchedAt: new Date(),
  };
}

async function fetchJSON(path) {
  try {
    const response = await fetch(path, { credentials: 'same-origin' });
    if (!response.ok) throw new Error(`Failed with ${response.status}`);
    return await response.json();
  } catch (error) {
    console.warn('[diag] fetch failed', path, error);
    return null;
  }
}

function formatProviders(providers = {}) {
  const entries = Object.entries(providers);
  if (!entries.length) return '—';
  return entries
    .map(([name, meta]) => {
      const ok = meta?.ok ?? meta?.status === 'ok';
      return `${name}${ok ? ' ✓' : ' ⚠'}`;
    })
    .join(', ');
}

function formatRelative(date) {
  const diffMs = Date.now() - date.getTime();
  const diffMinutes = Math.round(diffMs / 60000);
  if (Math.abs(diffMinutes) < 1) return 'just now';

  const formatter =
    typeof Intl !== 'undefined' && Intl.RelativeTimeFormat
      ? new Intl.RelativeTimeFormat(undefined, { numeric: 'auto' })
      : null;

  if (!formatter) {
    return `${diffMinutes} min${Math.abs(diffMinutes) === 1 ? '' : 's'} ago`;
  }

  if (Math.abs(diffMinutes) < 60) {
    return formatter.format(-diffMinutes, 'minute');
  }

  const diffHours = Math.round(diffMinutes / 60);
  if (Math.abs(diffHours) < 24) {
    return formatter.format(-diffHours, 'hour');
  }

  const diffDays = Math.round(diffHours / 24);
  return formatter.format(-diffDays, 'day');
}
