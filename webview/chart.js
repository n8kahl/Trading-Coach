let viewerInstance = null;

export function mountChart(selector, opts = {}) {
  const host = document.querySelector(selector);
  if (!host) throw new Error(`Chart root not found at selector: ${selector}`);

  host.innerHTML = '<div id="tvRoot" class="tv-root"></div>';
  const urlParams = new URLSearchParams(location.search);
  const paramOverrides = opts.params || {};
  const symbol =
    opts.symbol ||
    paramOverrides.symbol ||
    urlParams.get('symbol') ||
    'SPY';
  const interval =
    paramOverrides.interval ||
    urlParams.get('interval') ||
    opts.interval ||
    '5';

  initViewer('#tvRoot', { symbol, interval });

  const levelsRaw =
    opts.levelsTokens ??
    paramOverrides.levels ??
    (urlParams.get('levels') || '');
  const supportingFlag =
    opts.supportingLevels ??
    paramOverrides.supportingLevels ??
    urlParams.get('supportingLevels') ??
    '1';
  const supporting = `${supportingFlag}` !== '0';
  const levels = parseLevels(levelsRaw);
  const overlay = new LevelsOverlay({ getViewer, guidesClass: 'guide--supporting' });
  overlay.show(supporting);
  overlay.set(levels);

  const extras = collectPrices(opts.prices, paramOverrides, urlParams);
  overlay.autoscale({ extra: extras });

  const toggleBtn = document.getElementById('btnToggleSupportingLevels');
  if (toggleBtn) {
    if (toggleBtn.__levelsHandler) {
      toggleBtn.removeEventListener('click', toggleBtn.__levelsHandler);
    }
    const syncButton = () => {
      const visible = overlay.visible;
      toggleBtn.textContent = visible ? 'Hide Supporting Levels' : 'Show Supporting Levels';
      toggleBtn.setAttribute('aria-pressed', visible ? 'true' : 'false');
    };
    syncButton();
    const handler = () => {
      overlay.toggle();
      syncButton();
    };
    toggleBtn.addEventListener('click', handler);
    toggleBtn.__levelsHandler = handler;
  }
}

export function parseLevels(raw) {
  if (!raw) return [];
  const tokens = raw.split(/[;,]/).map((token) => token.trim()).filter(Boolean);
  const seen = new Set();
  const out = [];

  for (const token of tokens) {
    const [priceStr, labelRaw] = token.split('|');
    const price = Number(priceStr);
    if (!Number.isFinite(price)) continue;
    const label = (labelRaw || '').trim() || 'Lvl';
    const key = `${price.toFixed(4)}|${label}`;
    if (seen.has(key)) continue;
    seen.add(key);
    out.push({ price, label });
    if (out.length >= 40) break;
  }

  return out;
}

export class LevelsOverlay {
  constructor({ getViewer, guidesClass }) {
    this.getViewer = getViewer;
    this.guidesClass = guidesClass;
    this.lines = [];
    this.visible = true;
    this.data = [];
  }

  set(levels) {
    this.data = Array.isArray(levels) ? levels : [];
    if (this.visible) this._render();
  }

  show(visible = true) {
    this.visible = Boolean(visible);
    this._render();
  }

  toggle() {
    this.show(!this.visible);
  }

  clear() {
    const tv = this.getViewer();
    if (this.lines.length && tv?.removeLine) {
      this.lines.forEach((id) => tv.removeLine(id));
    }
    this.lines = [];
  }

  _render() {
    this.clear();
    if (!this.visible) return;
    const tv = this.getViewer();
    if (!tv?.drawHLine) return;

    this.data.forEach(({ price, label }) => {
      const id = tv.drawHLine(price, { label, className: this.guidesClass });
      if (id) this.lines.push(id);
    });
  }

  autoscale({ extra = [] } = {}) {
    const tv = this.getViewer();
    if (!tv?.setPriceRange) return;
    const prices = [
      ...this.data.map((item) => item.price),
      ...extra.filter((value) => Number.isFinite(value)),
    ].filter((value) => Number.isFinite(value));

    if (!prices.length) return;
    const min = Math.min(...prices);
    const max = Math.max(...prices);
    const pad = (max - min) * 0.1 || Math.abs(min * 0.001) || 1;
    tv.setPriceRange({ min: min - pad, max: max + pad });
  }
}

function collectPrices(prices = {}, params = {}, urlParams = new URLSearchParams()) {
  const entry =
    prices.entry ??
    num(params.entry) ??
    num(urlParams.get('entry'));
  const stop =
    prices.stop ??
    num(params.stop) ??
    num(urlParams.get('stop'));
  const targets =
    Array.isArray(prices.targets) && prices.targets.length
      ? prices.targets
      : parseTargets(params.tp ?? urlParams.get('tp'));
  return [entry, stop, ...targets].filter((value) => Number.isFinite(value));
}

function num(value) {
  if (value === null || value === undefined || value === '') return NaN;
  const n = Number(value);
  return Number.isFinite(n) ? n : NaN;
}

function parseTargets(raw) {
  if (!raw) return [];
  if (Array.isArray(raw)) {
    return raw.map(num).filter((value) => Number.isFinite(value));
  }
  return String(raw)
    .split(',')
    .map(num)
    .filter((value) => Number.isFinite(value));
}

function getViewer() {
  return viewerInstance;
}

function initViewer(rootSel, { symbol, interval }) {
  const root = document.querySelector(rootSel);
  if (!root) throw new Error(`Viewer root not found: ${rootSel}`);
  viewerInstance = new LightweightViewer(root, { symbol, interval });
  return viewerInstance;
}

class LightweightViewer {
  constructor(root, { symbol, interval }) {
    this.root = root;
    this.root.classList.add('tv-root');

    this.viewport = document.createElement('div');
    this.viewport.className = 'tv-viewport';

    this.placeholder = document.createElement('div');
    this.placeholder.className = 'tv-placeholder';
    this.placeholder.innerHTML = `
      <div class="tv-symbol">${symbol}</div>
      <div class="tv-interval">${interval.toUpperCase()}</div>
    `;

    this.overlay = document.createElement('div');
    this.overlay.className = 'tv-overlay';

    this.viewport.appendChild(this.placeholder);
    this.viewport.appendChild(this.overlay);
    this.root.innerHTML = '';
    this.root.appendChild(this.viewport);

    this.lines = new Map();
    this.counter = 0;
    this.priceRange = { min: 0, max: 1 };
    this.setPriceRange({ min: 0, max: 1 });
  }

  drawHLine(price, { label, className } = {}) {
    const id = `line-${++this.counter}`;
    const line = document.createElement('div');
    line.className = ['guide', className].filter(Boolean).join(' ');
    line.dataset.price = price;

    const labelEl = document.createElement('span');
    labelEl.className = 'guide-label';
    labelEl.textContent = label || '';
    line.appendChild(labelEl);

    this.overlay.appendChild(line);
    this.lines.set(id, { element: line, price });
    this.positionLine(id);
    return id;
  }

  removeLine(id) {
    const entry = this.lines.get(id);
    if (!entry) return;
    entry.element.remove();
    this.lines.delete(id);
  }

  setPriceRange(range) {
    const { min, max } = range || {};
    if (!Number.isFinite(min) || !Number.isFinite(max) || min === max) return;
    this.priceRange = { min, max };
    this.lines.forEach((_, id) => this.positionLine(id));
  }

  setInstrument({ symbol, interval }) {
    if (symbol) this.placeholder.querySelector('.tv-symbol').textContent = symbol;
    if (interval) this.placeholder.querySelector('.tv-interval').textContent = interval.toUpperCase();
  }

  positionLine(id) {
    const entry = this.lines.get(id);
    if (!entry) return;
    const { price } = entry;
    const { min, max } = this.priceRange;
    const span = max - min;
    if (!span) return;
    const ratio = 1 - (price - min) / span;
    const clamped = Math.min(1, Math.max(0, ratio));
    entry.element.style.top = `${clamped * 100}%`;
  }
}
