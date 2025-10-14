import "server-only";

import { API_BASE_URL, withAuthHeaders } from "./env";
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
  const searchParams = new URLSearchParams({ symbol: symbol.toUpperCase() });
  if (style) {
    searchParams.set("style", style);
  }

  const url = `${API_BASE_URL}/internal/idea/latest?${searchParams.toString()}`;
  const res = await fetch(url, {
    headers: withAuthHeaders({
      Accept: "application/json",
    }),
    cache: "no-store",
  });

  if (!res.ok) {
    return null;
  }

  const data = (await res.json()) as { plan_id?: string };
  return data.plan_id ?? null;
}
