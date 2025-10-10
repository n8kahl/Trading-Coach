const VALID_HOST = "trading-coach-production.up.railway.app";

export function validateChartUrl(url: string | null | undefined): string | null {
  if (!url) return null;
  try {
    const parsed = new URL(url);
    if (parsed.host !== VALID_HOST) return null;
    if (!parsed.pathname.startsWith("/tv")) return null;
    return parsed.toString();
  } catch (error) {
    return null;
  }
}
