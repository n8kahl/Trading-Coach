import { API_BASE, fetchJSON, showToast, formatNumber, copy, roundToDecimals } from './common.js';

const params = new URLSearchParams(window.location.search);
const planId = params.get('plan_id');
const version = params.get('v');

const element = (id) => document.getElementById(id);

const symbolEl = element('plan-symbol');
const biasEl = element('plan-bias');
const entryEl = element('plan-entry');
const stopEl = element('plan-stop');
const tp1El = element('plan-tp1');
const tp2El = element('plan-tp2');
const rrEl = element('plan-rr');
const ideaUrlEl = element('idea-url');
const warningsEl = element('plan-warnings');
const calcNotesEl = element('calc-notes');
const htfSnapsEl = element('htf-snaps');
const dataQualityEl = element('data-quality');
const confluenceEl = element('confluence-content');
const educationEl = element('education-cards');
const optionsTable = document.querySelector('#options-table tbody');
const optionsEmpty = element('options-empty');
const coachEventsEl = element('coach-events');
const chartFrame = element('chart-frame');
const chartWarning = element('chart-warning');
const copyBtn = element('copy-levels');
const liveBtn = element('live-btn');
const replayBtn = element('replay-btn');
const ivSlider = element('iv-shift');
const ivValue = element('iv-value');
const slippageSlider = element('slippage');
const slippageValue = element('slippage-value');

let ideaSnapshot = null;
let decimals = 2;
let liveSource = null;
let replaySource = null;

async function init() {
  if (!planId) {
    showToast('Missing plan_id in URL');
    return;
  }
  try {
    const url = version
      ? `${API_BASE}/idea/${encodeURIComponent(planId)}/${encodeURIComponent(version)}`
      : `${API_BASE}/idea/${encodeURIComponent(planId)}`;
    ideaSnapshot = await fetchJSON(url);
    populate();
    startLive();
  } catch (err) {
    console.error(err);
    showToast(`Failed to load idea: ${err.message}`);
  }
}

function populate() {
  const plan = ideaSnapshot.plan || {};
  const planData = ideaSnapshot.plan || {};
  const targets = planData.targets || [];
  decimals = planData.decimals ?? ideaSnapshot.summary?.decimals ?? 2;

  symbolEl.textContent = planData.symbol || '—';
  biasEl.textContent = planData.bias || '—';
  entryEl.textContent = formatNumber(planData.entry, decimals);
  stopEl.textContent = formatNumber(planData.stop, decimals);
  tp1El.textContent = formatNumber(targets[0], decimals);
  tp2El.textContent = formatNumber(targets[1], decimals);
  rrEl.textContent = formatNumber(planData.rr_to_t1, 2);
  const tradeDetailUrl = planData.trade_detail || planData.idea_url;
  ideaUrlEl.href = tradeDetailUrl || '#';
  ideaUrlEl.classList.toggle('muted', !tradeDetailUrl);

  renderWarnings(planData.warnings || ideaSnapshot.warnings || []);
  renderCalcNotes(ideaSnapshot.calc_notes || plan.calc_notes);
  renderHtf(ideaSnapshot.htf || plan.htf);
  renderDataQuality(ideaSnapshot.data_quality);
  renderConfluence(ideaSnapshot.summary, ideaSnapshot.volatility_regime);
  renderEducationCards(ideaSnapshot);
  renderOptions(ideaSnapshot.options);

  const chartUrl = ideaSnapshot.chart_url || plan.trade_detail || plan.idea_url;
  if (chartUrl && chartUrl.includes('/tv')) {
    chartFrame.src = chartUrl;
    chartWarning.classList.add('hidden');
  } else {
    chartWarning.classList.remove('hidden');
  }

  copyBtn.addEventListener('click', () => {
    const text = `Symbol ${planData.symbol}\nBias ${planData.bias}\nEntry ${planData.entry}\nStop ${planData.stop}\nTargets ${targets.join(', ')}`;
    copy(text);
    showToast('Levels copied to clipboard');
  });

  liveBtn.addEventListener('click', startLive);
  replayBtn.addEventListener('click', startReplay);

  ivSlider.addEventListener('input', updateOptionScenarios);
  slippageSlider.addEventListener('input', updateOptionScenarios);
}

function renderWarnings(warnings) {
  warningsEl.innerHTML = '';
  if (!warnings || warnings.length === 0) {
    warningsEl.innerHTML = '<li class="muted">None</li>';
    return;
  }
  warnings.forEach((w) => {
    const li = document.createElement('li');
    li.textContent = w;
    warningsEl.appendChild(li);
  });
}

function renderCalcNotes(calcNotes) {
  calcNotesEl.textContent = JSON.stringify(calcNotes ?? {}, null, 2);
}

function renderHtf(htf) {
  htfSnapsEl.textContent = JSON.stringify(htf ?? {}, null, 2);
}

function renderDataQuality(dataQuality) {
  dataQualityEl.textContent = JSON.stringify(dataQuality ?? {}, null, 2);
}

function renderConfluence(summary, volRegime) {
  confluenceEl.innerHTML = '';
  if (!summary) {
    confluenceEl.innerHTML = '<p class="muted">Summary unavailable.</p>';
    return;
  }
  const list = document.createElement('ul');
  list.className = 'meta';
  const pairs = [
    ['Confluence Score', formatNumber(summary.confluence_score, 2)],
    ['Frames', (summary.frames_used || []).join(', ') || '—'],
    ['Trend Notes', JSON.stringify(summary.trend_notes || {})],
    ['Expected Move Horizon', formatNumber(summary.expected_move_horizon, decimals)],
    ['Nearby Levels', (summary.nearby_levels || []).join(', ') || '—'],
  ];
  pairs.forEach(([label, value]) => {
    const li = document.createElement('li');
    const span = document.createElement('span');
    span.textContent = label;
    li.appendChild(span);
    const strong = document.createElement('strong');
    strong.textContent = value;
    li.appendChild(strong);
    list.appendChild(li);
  });
  confluenceEl.appendChild(list);

  if (volRegime) {
    const pre = document.createElement('pre');
    pre.textContent = JSON.stringify(volRegime, null, 2);
    confluenceEl.appendChild(pre);
  }
}

function renderEducationCards(snapshot) {
  educationEl.innerHTML = '';
  const cards = [];
  const plan = snapshot.plan || {};
  const snapped = (snapshot.htf?.snapped_targets || []).join(',');
  const summary = snapshot.summary || {};

  if ((plan.setup || '').toLowerCase().includes('vwap') || snapped.includes('VWAP')) {
    cards.push({
      title: 'VWAP Reclaim',
      body: 'Price reclaimed VWAP with HTF alignment. Expect pullbacks towards VWAP to hold if confluence stays intact.',
      link: 'https://docs.trading-coach.app/playbooks/vwap-reclaim',
    });
  }
  if (snapped.includes('VAH') || snapped.includes('VAL') || snapped.includes('POC')) {
    cards.push({
      title: 'Volume Profile Levels',
      body: 'Targets align with VAH/VAL/POC. These levels often act as magnets or barriers—watch them closely.',
      link: 'https://docs.trading-coach.app/concepts/volume-profile',
    });
  }
  if (snapped.toLowerCase().includes('fib')) {
    cards.push({
      title: 'Fibonacci Extensions',
      body: 'Fib 1.0/1.272 targets represent natural expansion zones. Combine with confluence score before scaling risk.',
      link: 'https://docs.trading-coach.app/concepts/fibonacci',
    });
  }
  if ((summary.volatility_regime || {}).regime_label === 'elevated') {
    cards.push({
      title: 'Elevated IV',
      body: 'Implied volatility is elevated. Prefer defined-risk structures or scale down size; EM caps enforced automatically.',
      link: 'https://docs.trading-coach.app/concepts/volatility-regimes',
    });
  }

  if (cards.length === 0) {
    educationEl.innerHTML = '<p class="muted">No education notes for this plan.</p>';
    return;
  }

  cards.forEach((card) => {
    const div = document.createElement('article');
    div.className = 'card';
    div.innerHTML = `
      <h3>${card.title}</h3>
      <p>${card.body}</p>
      <a href="${card.link}" target="_blank" rel="noopener">Read guide →</a>
    `;
    educationEl.appendChild(div);
  });
}

function renderOptions(optionsData) {
  optionsTable.innerHTML = '';
  if (!optionsData?.table || optionsData.table.length === 0) {
    optionsEmpty.classList.remove('hidden');
    return;
  }
  optionsEmpty.classList.add('hidden');

  optionsData.table.forEach((row) => {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>${row.label || '—'}</td>
      <td>${formatNumber(row.price, 2)}</td>
      <td>${formatNumber(row.bid, 2)}</td>
      <td>${formatNumber(row.ask, 2)}</td>
      <td>${formatNumber(row.delta, 2)}</td>
      <td>${formatNumber(row.theta, 2)}</td>
      <td>${formatNumber(row.iv, 2)}</td>
      <td>${formatNumber(row.spread_pct, 2)}</td>
      <td>${row.oi ?? '—'}</td>
      <td>${row.liquidity_score ?? '—'}</td>
      <td class="scenario">—</td>
    `;
    tr.dataset.row = JSON.stringify(row);
    optionsTable.appendChild(tr);
  });
  updateOptionScenarios();
}

function updateOptionScenarios() {
  if (!optionsTable.children.length) return;
  ivValue.textContent = ivSlider.value;
  slippageValue.textContent = slippageSlider.value;
  const ivShift = Number(ivSlider.value || 0) / 10000; // convert bps
  const slippage = Number(slippageSlider.value || 0) / 10000;

  Array.from(optionsTable.children).forEach((tr) => {
    const row = JSON.parse(tr.dataset.row || '{}');
    const price = Number(row.price || 0);
    const delta = Number(row.delta || 0);
    const gamma = Number(row.gamma || 0);
    const vega = Number(row.vega || 0);

    const plan = ideaSnapshot.plan || {};
    const entry = Number(plan.entry || 0);
    const tp1 = Number((plan.targets || [])[0] || entry);
    const move = tp1 - entry;

    const scenario =
      price +
      delta * move +
      0.5 * gamma * move * move +
      vega * ivShift -
      price * slippage;

    const cell = tr.querySelector('.scenario');
    cell.textContent = formatNumber(scenario, 2);
  });
}

function appendEvent(event) {
  const div = document.createElement('div');
  div.className = 'event';
  div.innerHTML = `
    <strong>${event.state || 'event'}</strong><br />
    ${event.coaching || ''}<br />
    <span class="muted">Price: ${formatNumber(event.price, decimals)} | ${new Date(event.time || Date.now()).toLocaleTimeString()}</span>
  `;
  coachEventsEl.prepend(div);
  while (coachEventsEl.children.length > 50) {
    coachEventsEl.removeChild(coachEventsEl.lastChild);
  }
}

function startLive() {
  stopStreams();
  liveBtn.classList.add('active');
  replayBtn.classList.remove('active');
  if (!planId || !ideaSnapshot?.plan?.symbol) return;
  const symbol = ideaSnapshot.plan.symbol;
  liveSource = new EventSource(`${API_BASE}/stream/market?symbol=${encodeURIComponent(symbol)}`);
  liveSource.onmessage = (event) => {
    try {
      const payload = JSON.parse(event.data || '{}');
      if (payload.event) appendEvent(payload.event);
    } catch (err) {
      console.warn('stream parse error', err);
    }
  };
  liveSource.onerror = () => {
    showToast('Live stream error');
    stopStreams();
  };
}

function startReplay() {
  stopStreams();
  liveBtn.classList.remove('active');
  replayBtn.classList.add('active');
  const plan = ideaSnapshot.plan || {};
  const symbol = plan.symbol;
  if (!symbol) return;
  const params = new URLSearchParams({
    symbol,
    minutes: '30',
    entry: plan.entry,
    stop: plan.stop,
    tp1: (plan.targets || [])[0],
    tp2: (plan.targets || [])[1] ?? '',
    direction: plan.bias || 'long',
  });
  replaySource = new EventSource(`${API_BASE}/simulate?${params.toString()}`);
  replaySource.onmessage = (e) => {
    try {
      const payload = JSON.parse(e.data || '{}');
      appendEvent(payload);
    } catch (err) {
      console.warn('simulate parse error', err);
    }
  };
  replaySource.onerror = () => {
    showToast('Replay finished');
    stopStreams();
  };
}

function stopStreams() {
  if (liveSource) {
    liveSource.close();
    liveSource = null;
  }
  if (replaySource) {
    replaySource.close();
    replaySource = null;
  }
}

window.addEventListener('beforeunload', stopStreams);

init();
