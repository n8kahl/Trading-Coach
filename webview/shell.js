import { mountActionsDock } from './actions.js';
import { mountChart } from './chart.js';
import { mountConfidenceSurface, bindConfidenceData } from './confidence.js';
import { mountPlanSummary, mountEvidence } from './plan.js';
import { mountMobileUX } from './mobile.js';
import { fetchPlan } from './data.js';
import {
  Perf,
  mountObservabilityPanel,
  updatePerf,
  updateDataAge,
  loadDiagnostics,
} from './diag.js';

const qs = (selector) => document.querySelector(selector);
const DEFAULT_SYMBOL = 'SPY';

let currentSymbol = DEFAULT_SYMBOL;
let currentModel = null;

document.addEventListener('DOMContentLoaded', initShell);

async function initShell() {
  Perf.mark('webview:init');

  applyStoredTheme();
  bindThemeToggle();
  bindSymbolInteractions();
  attachKeyboardShortcuts();

  mountConfidenceSurface('#confidenceSurface');

  await mountObservabilityPanel('#diagPanel');

  await loadSymbol(getSymbolFromQuery());

  Perf.end('webview:init');
  updatePerf();
}

async function loadSymbol(nextSymbol) {
  const target = (nextSymbol || DEFAULT_SYMBOL).toUpperCase();
  currentSymbol = target;
  setTitle(target);
  setSymbolField(target);

  Perf.mark('plan:fetch');
  try {
    const model = await fetchPlan({ symbol: target });
    Perf.end('plan:fetch');
    currentModel = model;

    const session = model.uiState.session || 'live';
    updateStatus(session);
    decorateSession(session);

    const diag = await loadDiagnostics();
    await mountActionsDock('#actionsDock', {
      symbol: target,
      session,
      diagnostics: diag,
    });

    mountChart('#chartRoot', {
      symbol: target,
      params: model.charts?.params || {},
      levelsTokens: model.levelsTokens,
      supportingLevels: model.supportingLevels,
      prices: {
        entry: model.entry,
        stop: model.stop,
        targets: model.tps,
      },
    });

    mountPlanSummary('#planSummary', model);
    mountEvidence('#evidence', model);
    bindConfidenceData('#confidenceSurface', model);
    mountMobileUX({ sheetSel: '#bottomSheet', fabSel: '#fab', session });

    updateUrl(model);
    updatePerf();
    updateDataAge(model.dataAge);
  } catch (error) {
    Perf.end('plan:fetch');
    reportError(error);
    updateDataAge(null);
    updatePerf();
  }
}

function bindSymbolInteractions() {
  const form = qs('#symbolForm');
  if (!form) return;

  form.addEventListener('submit', (event) => {
    event.preventDefault();
    const input = qs('#symbolInput');
    if (!input) return;
    const value = input.value.trim().toUpperCase();
    if (!value) return;
    if (value === currentSymbol) {
      input.blur();
      return;
    }
    loadSymbol(value);
  });
}

function attachKeyboardShortcuts() {
  window.addEventListener(
    'keydown',
    (event) => {
      if (event.key === '/' && !event.metaKey && !event.ctrlKey && !event.altKey) {
        const input = qs('#symbolInput');
        if (!input) return;
        event.preventDefault();
        input.focus();
        input.select();
      }
      if (event.key === 'Escape') {
        const active = document.activeElement;
        if (active?.id === 'symbolInput') {
          active.blur();
        }
      }
    },
    true
  );
}

function setTitle(symbol) {
  const title = qs('#symbolTitle');
  if (title) title.textContent = symbol;
  document.title = `Trading Coach • ${symbol}`;
}

function setSymbolField(symbol) {
  const input = qs('#symbolInput');
  if (input) input.value = symbol;
}

function updateStatus(currentSession) {
  const pill = qs('#statusBanner');
  if (!pill) return;

  pill.hidden = false;
  pill.classList.remove('status-pill--premkt', 'status-pill--after');

  if (currentSession === 'live') {
    pill.textContent = 'Live';
  } else if (currentSession === 'premkt') {
    pill.textContent = 'Pre-Market';
    pill.classList.add('status-pill--premkt');
  } else if (currentSession === 'after') {
    pill.textContent = 'After Hours';
    pill.classList.add('status-pill--after');
  } else {
    pill.hidden = true;
  }
}

function decorateSession(currentSession) {
  document.body.dataset.session = currentSession;
  const dock = qs('#actionsDock');
  const sheet = qs('#bottomSheet');

  if (dock) {
    dock.classList.toggle('after-hours', currentSession === 'after');
  }
  if (sheet) {
    sheet.dataset.session = currentSession;
  }
}

function reportError(error) {
  console.error('[shell] Failed to load plan', error);
  const planRoot = qs('#planSummary');
  if (planRoot) {
    planRoot.innerHTML = `
      <header class="section-header"><h3>Plan</h3></header>
      <p class="muted">Unable to load plan data. Please retry shortly.</p>
    `;
  }
}

function applyStoredTheme() {
  const stored = localStorage.getItem('webview.theme');
  const prefersLight = window.matchMedia('(prefers-color-scheme: light)').matches;
  const theme = stored || (prefersLight ? 'light' : 'dark');
  setTheme(theme);
}

function bindThemeToggle() {
  const toggle = qs('#themeToggle');
  if (!toggle) return;
  toggle.addEventListener('click', () => {
    const next = document.documentElement.classList.contains('light') ? 'dark' : 'light';
    setTheme(next);
  });
}

function setTheme(theme) {
  const isLight = theme === 'light';
  document.documentElement.classList.toggle('light', isLight);
  localStorage.setItem('webview.theme', theme);
  const toggle = qs('#themeToggle');
  if (toggle) {
    toggle.textContent = isLight ? 'Dark Mode' : 'Light Mode';
    toggle.setAttribute('aria-pressed', String(isLight));
  }
}

function getSymbolFromQuery() {
  const params = new URLSearchParams(window.location.search);
  return params.get('symbol') || DEFAULT_SYMBOL;
}

function updateUrl(model) {
  const params = new URLSearchParams(window.location.search);
  params.set('symbol', model.symbol || currentSymbol);
  if (model.charts?.params?.interval) {
    params.set('interval', model.charts.params.interval);
  }
  if (isFinite(model.entry)) params.set('entry', model.entry);
  if (isFinite(model.stop)) params.set('stop', model.stop);
  if (Array.isArray(model.tps) && model.tps.length) {
    params.set('tp', model.tps.filter(isFinite).join(','));
  }
  if (model.levelsTokens) params.set('levels', model.levelsTokens);
  if (model.supportingLevels !== undefined) {
    params.set('supportingLevels', String(model.supportingLevels));
  }

  const uiState =
    model.charts?.params?.ui_state || JSON.stringify(model.uiState || { session: 'live' });
  params.set('ui_state', uiState);

  const url = `${window.location.pathname}?${params.toString()}${window.location.hash || ''}`;
  window.history.replaceState({}, '', url);
}
