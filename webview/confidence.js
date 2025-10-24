const gates = {
  unfavorable: [0, 49],
  mixed: [50, 69],
  favorable: [70, 100],
};

export function mountConfidenceSurface(rootSel) {
  const root = document.querySelector(rootSel);
  if (!root) throw new Error(`Confidence surface root not found: ${rootSel}`);

  root.innerHTML = `
    <header class="section-header">
      <h3>Confidence</h3>
      <time id="confidenceAge" class="muted">—</time>
    </header>
    <div id="confidenceBand" class="band"><span id="confidenceScore">—</span></div>
    <ul id="confidenceChips" class="chips" role="list"></ul>
    <p id="confidenceWhy" class="muted"></p>
  `;
}

export function bindConfidenceData(rootSel, model = {}) {
  const root = typeof rootSel === 'string' ? document.querySelector(rootSel) : rootSel;
  if (!root) return;

  const band = root.querySelector('#confidenceBand');
  const scoreEl = root.querySelector('#confidenceScore');
  const chips = root.querySelector('#confidenceChips');
  const whyEl = root.querySelector('#confidenceWhy');
  const ageEl = root.querySelector('#confidenceAge');

  const score = clampToScore(model.confidence);
  if (scoreEl) scoreEl.textContent = `${score}`;
  if (band) band.dataset.state = computeState(score);

  const factors = buildFactors(model.confluence);
  if (chips) {
    chips.innerHTML = '';
    factors.forEach(({ k, v }) => {
      const li = document.createElement('li');
      li.className = 'chip';
      li.textContent = `${k} ${formatContribution(v)}`;
      chips.appendChild(li);
    });
  }

  if (whyEl) {
    const why = model.confluence?.why || model.evidence?.why || 'Model alignment across frames.';
    whyEl.textContent = why;
  }

  if (ageEl && model.dataAge instanceof Date && !Number.isNaN(model.dataAge.getTime())) {
    ageEl.dateTime = model.dataAge.toISOString();
    ageEl.textContent = model.dataAge.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  }
}

function buildFactors(confluence = {}) {
  return [
    { k: 'ATR', v: confluence.atr ?? 0 },
    { k: 'VWAP', v: confluence.vwap ?? 0 },
    { k: 'EMAs', v: confluence.emas ?? 0 },
    { k: 'Order-Flow', v: confluence.orderflow ?? 0 },
    { k: 'Liquidity', v: confluence.liquidity ?? 0 },
  ];
}

function clampToScore(value) {
  if (!isFinite(value)) return 0;
  const pct = value <= 1 ? value * 100 : value;
  return Math.round(Math.max(0, Math.min(100, pct)));
}

function computeState(score) {
  if (score >= gates.favorable[0]) return 'favorable';
  if (score >= gates.mixed[0]) return 'mixed';
  return 'unfavorable';
}

function formatContribution(value) {
  if (!isFinite(value)) return '+0';
  const scaled = Math.round(value * 100);
  const prefix = scaled >= 0 ? '+' : '';
  return `${prefix}${scaled}`;
}
