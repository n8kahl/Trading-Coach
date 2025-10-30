const CANONICAL_HOST = "trading-coach-production.up.railway.app";
const CANONICAL_PATH_PREFIX = "/chart";

function parseUrl(candidate: string): URL | null {
  try {
    return new URL(candidate);
  } catch {
    return null;
  }
}

export function isCanonicalChartUrl(candidate: unknown): candidate is string {
  if (typeof candidate !== "string" || !candidate.trim()) {
    return false;
  }
  const parsed = parseUrl(candidate.trim());
  if (!parsed) return false;
  if (parsed.protocol !== "https:") return false;
  if (parsed.hostname !== CANONICAL_HOST) return false;
  if (!parsed.pathname.startsWith(CANONICAL_PATH_PREFIX)) return false;
  return true;
}

export function ensureCanonicalChartUrl(candidate: unknown): string | null {
  return isCanonicalChartUrl(candidate) ? candidate.trim() : null;
}

export const CANONICAL_CHART_HOST = `https://${CANONICAL_HOST}`;
