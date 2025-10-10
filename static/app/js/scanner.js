import { API_BASE, fetchJSON, showToast, buildIdeaUrl } from './common.js';

const form = document.getElementById('scan-form');
const resultsEl = document.getElementById('scan-results');
const biasFilter = document.getElementById('bias-filter');
const scoreFilter = document.getElementById('score-filter');
const regimeFilter = document.getElementById('regime-filter');
const cardTemplate = document.getElementById('scan-card-template');

let lastResults = [];

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const tickersRaw = document.getElementById('tickers').value.trim();
  const style = document.getElementById('style').value || null;
  const body = {
    tickers: tickersRaw
      ? tickersRaw.split(/[\s,]+/).filter(Boolean).map((t) => t.toUpperCase())
      : [],
  };
  if (style) body.style = style;

  try {
    resultsEl.innerHTML = '<p class="muted">Running scan…</p>';
    const res = await fetchJSON(`${API_BASE}/gpt/scan`, {
      method: 'POST',
      body: JSON.stringify(body),
    });
    lastResults = res;
    render();
    showToast('Scan complete');
  } catch (err) {
    console.error(err);
    resultsEl.innerHTML = `<p class="muted">Scan failed: ${err.message}</p>`;
  }
});

[biasFilter, scoreFilter, regimeFilter].forEach((el) =>
  el.addEventListener('change', render)
);

function render() {
  if (!Array.isArray(lastResults) || lastResults.length === 0) {
    resultsEl.innerHTML = '<p class="muted">No results yet.</p>';
    return;
  }

  resultsEl.innerHTML = '';
  const biasValue = biasFilter.value;
  const minScore = Number(scoreFilter.value || 0);
  const regimeValue = regimeFilter.value;

  lastResults
    .filter((item) => {
      const bias = (item.features?.direction_bias || '').toLowerCase();
      const score = Number(item.score || 0);
      if (biasValue !== 'all' && bias !== biasValue) return false;
      if (score < minScore) return false;
      if (regimeValue !== 'all') {
        const regime = item.market_snapshot?.volatility?.regime_label;
        if ((regime || '').toLowerCase() !== regimeValue) return false;
      }
      return true;
    })
    .forEach((item) => {
      const node = cardTemplate.content.cloneNode(true);
      const card = node.querySelector('.card');
      card.querySelector('.symbol').textContent = item.symbol;
      card.querySelector('.description').textContent = item.description || '—';
      card.querySelector('.score').textContent = (item.score || 0).toFixed(2);
      card.querySelector('.setup').textContent = item.strategy_id || '—';
      card.querySelector('.confidence').textContent = (item.features?.plan_confidence ?? item.score ?? 0).toFixed(2);
      card.querySelector('.rr').textContent = (item.plan?.risk_reward ?? 0).toFixed(2);

      const biasChip = card.querySelector('.chip.bias');
      const bias = (item.features?.direction_bias || '').toLowerCase();
      biasChip.textContent = bias || 'n/a';
      biasChip.classList.remove('long', 'short');
      if (bias === 'long') biasChip.classList.add('long');
      if (bias === 'short') biasChip.classList.add('short');

      card.querySelector('.chip.style').textContent = item.style || 'auto';

      const levelsEl = card.querySelector('.levels');
      const levels = item.key_levels || {};
      levelsEl.innerHTML = Object.entries(levels)
        .slice(0, 4)
        .map(([k, v]) => `<span>${k}: ${Number(v).toFixed(2)}</span>`)
        .join('');

      const planBtn = card.querySelector('.plan-btn');
      planBtn.addEventListener('click', () => openIdea(item));

      const chartBtn = card.querySelector('.chart-btn');
      if (item.charts?.params) {
        const params = new URLSearchParams(item.charts.params).toString();
        chartBtn.onclick = () => window.open(`${API_BASE}/tv?${params}`, '_blank');
      } else {
        chartBtn.disabled = true;
      }

      resultsEl.appendChild(node);
    });

  if (!resultsEl.innerHTML) {
    resultsEl.innerHTML = '<p class="muted">No rows match the filters.</p>';
  }
}

async function openIdea(item) {
  try {
    const body = { symbol: item.symbol, style: item.style };
    const plan = await fetchJSON(`${API_BASE}/gpt/plan`, {
      method: 'POST',
      body: JSON.stringify(body),
    });
    if (plan.plan_id) {
      const url = buildIdeaUrl(plan.plan_id, plan.version || 1);
      window.open(url, '_blank');
    } else if (plan.idea_url) {
      window.open(plan.idea_url, '_blank');
    } else {
      throw new Error('Plan did not include idea URL');
    }
  } catch (err) {
    console.error(err);
    showToast(`Failed to open idea: ${err.message}`);
  }
}

render();
