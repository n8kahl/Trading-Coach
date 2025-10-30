export const API_BASE = (process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8000').replace(/\/+$/, '');

export const WS_BASE = (process.env.NEXT_PUBLIC_WS_BASE_URL || '')
  ? String(process.env.NEXT_PUBLIC_WS_BASE_URL).replace(/\/+$/, '')
  : API_BASE.replace(/^http/i, (m) => (m.toLowerCase() === 'https' ? 'wss' : 'ws'));

export const CHART_HOST = (process.env.NEXT_PUBLIC_CHART_HOST || API_BASE).replace(/\/+$/, '');

