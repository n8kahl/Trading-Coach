'use client';

import { useCallback, useEffect, useMemo, useState } from 'react';
import { API_BASE_URL, withAuthHeaders } from '@/lib/env';

export type ScenarioStyle = 'scalp' | 'intraday' | 'swing' | 'reversal';

export type ScenarioPlan = {
  plan_id: string;
  symbol: string;
  scenario_of: string | null; // origin live plan_id
  scenario_style: ScenarioStyle | null;
  linked_to_live: boolean;
  frozen: boolean; // always true for scenarios
  direction?: string | null;
  entry?: number | null;
  stop?: number | null;
  tps?: number[];
  confidence?: number | null;
  rr_to_t1?: number | null;
  chart_url?: string | null;
  created_at: string; // ISO
};

type StoreState = {
  livePlanId: string | null;
  scenarios: ScenarioPlan[];
};

const KEY_PREFIX = 'tc:scenarios:'; // per symbol key
const LIVE_KEY_PREFIX = 'tc:live:';

function readStore(symbol: string): StoreState {
  const raw = localStorage.getItem(KEY_PREFIX + symbol.toUpperCase());
  const live = localStorage.getItem(LIVE_KEY_PREFIX + symbol.toUpperCase());
  try {
    const parsed = raw ? (JSON.parse(raw) as ScenarioPlan[]) : [];
    return { livePlanId: live || null, scenarios: parsed };
  } catch {
    return { livePlanId: live || null, scenarios: [] };
  }
}

function writeStore(symbol: string, state: StoreState) {
  localStorage.setItem(KEY_PREFIX + symbol.toUpperCase(), JSON.stringify(state.scenarios));
  if (state.livePlanId) localStorage.setItem(LIVE_KEY_PREFIX + symbol.toUpperCase(), state.livePlanId);
  else localStorage.removeItem(LIVE_KEY_PREFIX + symbol.toUpperCase());
}

export function useScenarioStore(symbol: string) {
  const upper = symbol.toUpperCase();
  const [{ livePlanId, scenarios }, setState] = useState<StoreState>(() => ({ livePlanId: null, scenarios: [] }));

  useEffect(() => {
    setState(readStore(upper));
  }, [upper]);

  useEffect(() => {
    writeStore(upper, { livePlanId, scenarios });
  }, [upper, livePlanId, scenarios]);

  const addScenario = useCallback((plan: ScenarioPlan) => {
    setState((prev) => {
      const existing = prev.scenarios.filter((s) => s.symbol === upper);
      // De-dup identical style if very recent (60s)
      const now = Date.now();
      const recentSame = existing.find(
        (s) => s.scenario_style === plan.scenario_style && now - new Date(s.created_at).getTime() < 60_000,
      );
      if (recentSame) return prev; // drop duplicate
      // Enforce <=3 per symbol
      const nextScenarios = [...prev.scenarios.filter((s) => s.symbol !== upper), ...existing, plan].slice(-3);
      return { ...prev, scenarios: nextScenarios };
    });
  }, [upper]);

  const removeScenario = useCallback((plan_id: string) => {
    setState((prev) => ({ ...prev, scenarios: prev.scenarios.filter((s) => s.plan_id !== plan_id) }));
  }, []);

  const setLinked = useCallback((plan_id: string, linked: boolean) => {
    setState((prev) => ({
      ...prev,
      scenarios: prev.scenarios.map((s) => (s.plan_id === plan_id ? { ...s, linked_to_live: linked } : s)),
    }));
  }, []);

  const adoptAsLive = useCallback((plan_id: string) => {
    setState((prev) => ({ ...prev, livePlanId: plan_id }));
  }, []);

  const regenerateScenario = useCallback(async (style: ScenarioStyle, originLivePlanId: string | null) => {
    // Minimal /gpt/plan client
    const res = await fetch(`${API_BASE_URL}/gpt/plan`, {
      method: 'POST',
      headers: withAuthHeaders({ 'Content-Type': 'application/json', Accept: 'application/json' }),
      body: JSON.stringify({ symbol: upper, style }),
    });
    if (!res.ok) throw new Error(`/gpt/plan failed (${res.status})`);
    const data = await res.json();
    const plan = data?.plan ?? data ?? {};
    const planId: string = plan.plan_id || data.plan_id;
    const chartUrl: string | null = plan.chart_url || plan.trade_detail || data?.charts?.interactive || null;
    const entry: number | null = plan.entry ?? plan.structured_plan?.entry?.level ?? null;
    const stop: number | null = plan.stop ?? plan.structured_plan?.stop ?? null;
    const tps: number[] = Array.isArray(plan.targets) ? plan.targets : plan.structured_plan?.targets || [];
    const scenario: ScenarioPlan = {
      plan_id: planId,
      symbol: upper,
      scenario_of: originLivePlanId,
      scenario_style: style,
      linked_to_live: false,
      frozen: true,
      direction: plan.direction ?? plan.structured_plan?.direction ?? null,
      entry,
      stop,
      tps,
      confidence: plan.confidence ?? null,
      rr_to_t1: plan.rr_to_t1 ?? null,
      chart_url: chartUrl,
      created_at: new Date().toISOString(),
    };
    addScenario(scenario);
    // Telemetry (client-only)
    try { console.debug('telemetry', { t: 'scenario_generated', symbol: upper, style }); } catch {}
    return scenario;
  }, [upper, addScenario]);

  return useMemo(() => ({
    livePlanId,
    scenarios,
    addScenario,
    removeScenario,
    setLinked,
    adoptAsLive,
    regenerateScenario,
  }), [livePlanId, scenarios, addScenario, removeScenario, setLinked, adoptAsLive, regenerateScenario]);
}

