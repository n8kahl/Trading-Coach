export const API_BASE = window.location.origin;

export async function fetchJSON(url, options = {}) {
  const res = await fetch(url, {
    headers: { 'content-type': 'application/json', ...(options.headers || {}) },
    ...options,
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || res.statusText);
  }
  return res.json();
}

export function showToast(message, ms = 2400) {
  const toast = document.getElementById('toast');
  if (!toast) return;
  toast.textContent = message;
  toast.classList.add('show');
  setTimeout(() => toast.classList.remove('show'), ms);
}

export function formatNumber(value, decimals = 2) {
  if (value === null || value === undefined || Number.isNaN(value)) return '—';
  return Number(value).toFixed(decimals);
}

export function copy(text) {
  if (!navigator.clipboard) {
    const textarea = document.createElement('textarea');
    textarea.value = text;
    document.body.appendChild(textarea);
    textarea.select();
    document.execCommand('copy');
    document.body.removeChild(textarea);
    return;
  }
  return navigator.clipboard.writeText(text);
}

export function buildIdeaUrl(planId, version = 1) {
  const params = new URLSearchParams();
  if (planId) params.set('plan_id', planId);
  const safeVersion = Number.isFinite(Number(version)) ? String(version) : '1';
  params.set('plan_version', safeVersion);
  const query = params.toString();
  return query ? `/tv?${query}` : '/tv';
}

export function roundToDecimals(value, decimals) {
  const factor = 10 ** decimals;
  return Math.round(value * factor) / factor;
}
