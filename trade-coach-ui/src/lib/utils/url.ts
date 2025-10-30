/**
 * Enforces canonical chart URL usage.
 * Only allows server-hosted paths that start with /tv.
 * Returns a safe fallback '/tv' on invalid input.
 */
export function validateTvUrl(url: string): string {
  try {
    const u = new URL(
      url,
      typeof window !== 'undefined' ? window.location.origin : 'http://localhost'
    );
    if (!u.pathname.startsWith('/tv')) throw new Error('Non-canonical chart URL');
    return u.pathname + (u.search || '') + (u.hash || '');
  } catch {
    // fail closed: no fabricated link, just a neutral viewer route
    return '/tv';
  }
}

import { CHART_HOST } from '@/lib/hosts';

export function safeChartUrl(u: string | null | undefined): string | null {
  if (!u) return null;
  try {
    const url = new URL(u, typeof window !== 'undefined' ? window.location.origin : 'http://localhost');
    const okHost = new URL(CHART_HOST).host;
    return url.host === okHost ? url.toString() : null;
  } catch {
    return null;
  }
}
