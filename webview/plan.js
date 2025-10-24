export function mountPlanSummary(rootSel, model = {}) {
  const root = document.querySelector(rootSel);
  if (!root) return;

  const { entry, stop, tps = [], rr, runners = [] } = model;
  const targets = tps.filter(isFinite);

  root.innerHTML = `
    <header class="section-header">
      <h3>Plan</h3>
    </header>
    <ul class="kv" role="list">
      <li><span>Entry</span><strong>${fmt(entry)}</strong></li>
      <li><span>Stop</span><strong>${fmt(stop)}</strong></li>
      <li><span>TPs</span><strong>${targets.length ? targets.map(fmt).join(' · ') : '—'}</strong></li>
      <li><span>R:R</span><strong>${rr ?? '—'}</strong></li>
      ${
        runners.length
          ? `<li><span>Runners</span><strong>${runners.map(strip).join(', ')}</strong></li>`
          : ''
      }
    </ul>
    <div id="evidence" class="evidence-block"></div>
  `;
}

export function mountEvidence(rootSel, model = {}) {
  const root =
    typeof rootSel === 'string' ? document.querySelector(rootSel) : rootSel;
  if (!root) return;

  const { evidence = {} } = model;
  const bullets = [
    evidence.htf_bias && `HTF Bias: ${evidence.htf_bias}`,
    evidence.vol_regime && `Volatility: ${evidence.vol_regime}`,
    evidence.snap && `Snap: ${evidence.snap}`,
  ].filter(Boolean);

  const why = evidence.why;

  root.innerHTML = `
    ${
      bullets.length
        ? `<ul class="bullets" role="list">${bullets.map(renderBullet).join('')}</ul>`
        : `<p class="muted">Awaiting confluence details…</p>`
    }
    ${
      why
        ? `<p class="muted evidence-why" aria-live="polite">${escapeHtml(why)}</p>`
        : ''
    }
  `;
}

function fmt(value) {
  if (!isFinite(value)) return '—';
  const abs = Math.abs(value);
  if (abs >= 1000) return value.toLocaleString(undefined, { maximumFractionDigits: 2 });
  return Number(value).toFixed(2);
}

function strip(value) {
  return typeof value === 'string' ? value : String(value ?? '');
}

function renderBullet(content) {
  return `<li>${escapeHtml(content)}</li>`;
}

function escapeHtml(value) {
  return String(value ?? '').replace(/[&<>"']/g, (match) => {
    const entities = {
      '&': '&amp;',
      '<': '&lt;',
      '>': '&gt;',
      '"': '&quot;',
      "'": '&#39;',
    };
    return entities[match];
  });
}
