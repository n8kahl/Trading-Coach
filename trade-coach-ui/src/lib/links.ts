import { PUBLIC_UI_BASE_URL } from "@/lib/env";

export function buildUiPlanLink(planId: string): string {
  return `${PUBLIC_UI_BASE_URL}/plan/${encodeURIComponent(planId)}`;
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function buildTvLinkFromPlan(plan: any): string | null {
  const interactive = plan?.charts?.interactive || plan?.chart?.interactive || plan?.chart_url || null;
  if (typeof interactive === 'string' && interactive) return interactive;
  const base = (process.env.NEXT_PUBLIC_CHART_BASE || '').replace(/\/+$/, '');
  const symbol = plan?.symbol || plan?.plan?.symbol;
  const interval = plan?.charts?.params?.interval || plan?.interval || '5m';
  if (!base || !symbol) return null;
  const url = new URL(`${base}/html`);
  url.searchParams.set('symbol', String(symbol).toUpperCase());
  url.searchParams.set('interval', String(interval));
  if (plan?.entry != null) url.searchParams.set('entry', String(plan.entry));
  if (plan?.stop != null) url.searchParams.set('stop', String(plan.stop));
  if (Array.isArray(plan?.targets) && plan.targets.length) url.searchParams.set('tp', plan.targets.join(','));
  return url.toString();
}

export function parsePlanIdFromMaybeUrl(input: string): string | null {
  if (!input) return null;
  if (/^https?:\/\//i.test(input)) {
    try {
      const u = new URL(input);
      return u.searchParams.get('plan_id');
    } catch {
      return null;
    }
  }
  return input;
}
