import "server-only";

import { API_BASE_URL, withAuthHeaders } from "./env";
import { ensureCanonicalChartUrl } from "./chartUrl";
import type { PlanSnapshot } from "./types";

export async function fetchPlanSnapshot(planId: string): Promise<PlanSnapshot> {
  const url = `${API_BASE_URL}/idea/${encodeURIComponent(planId)}`;
  const res = await fetch(url, {
    headers: withAuthHeaders({
      Accept: "application/json",
    }),
    cache: "no-store",
  });

  if (!res.ok) {
    throw new Error(`Failed to fetch plan snapshot (${res.status})`);
  }

  return (await res.json()) as PlanSnapshot;
}

export async function fetchLatestPlanId(symbol: string, style?: string): Promise<string | null> {
  // Not available on the server yet; return null to avoid 404s
  return null;
}

export async function fetchPlanForSymbol(symbol: string, style?: string): Promise<PlanSnapshot | null> {
  const body: Record<string, unknown> = { symbol: symbol.toUpperCase() };
  if (style) body.style = style;
  const res = await fetch(`${API_BASE_URL}/gpt/plan`, {
    method: 'POST',
    headers: withAuthHeaders({ 'Content-Type': 'application/json', Accept: 'application/json' }),
    body: JSON.stringify(body),
    cache: 'no-store',
  });
  if (!res.ok) return null;
  const data = await res.json();
  const plan = data?.plan ?? data ?? {};
  const plan_id: string | undefined = plan.plan_id;
  if (!plan_id) return null;
  const structured = plan.structured_plan ?? null;
  const entry = plan.entry ?? structured?.entry?.level ?? null;
  const stop = plan.stop ?? structured?.stop ?? null;
  const targets = Array.isArray(plan.targets) ? plan.targets : structured?.targets ?? [];
  const snapshot: PlanSnapshot = {
    plan: {
      plan_id,
      symbol: plan.symbol ?? symbol.toUpperCase(),
      style: plan.style ?? structured?.style ?? null,
      entry,
      stop,
      targets,
      rr_to_t1: plan.rr_to_t1 ?? null,
      confidence: plan.confidence ?? null,
      session_state: plan.session_state ?? null,
      structured_plan: structured,
    },
    chart_url: ensureCanonicalChartUrl(plan.chart_url ?? plan.trade_detail ?? data?.charts?.interactive ?? null),
  } as PlanSnapshot;
  return snapshot;
}
