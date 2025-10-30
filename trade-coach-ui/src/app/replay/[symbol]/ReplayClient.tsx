'use client';

import { useEffect, useMemo, useReducer, useState } from 'react';
import Link from 'next/link';
import PriceChart from '@/components/PriceChart';
import { API_BASE_URL, WS_BASE_URL } from '@/lib/env';
import type { PlanDeltaEvent, PlanSnapshot, SymbolTickEvent } from '@/lib/types';
import { useScenarioStore, type ScenarioStyle } from '@/state/scenarioStore';

type ReplayClientProps = {
  symbol: string;
  initialLivePlanId: string | null;
  initialSnapshot: PlanSnapshot | null;
};

type LiveState = {
  planId: string | null;
  status: string;
  rr?: number | null;
  lastPrice?: number | null;
  entry?: number | null;
  stop?: number | null;
  targets?: number[];
};

function deriveInitialLive(snapshot: PlanSnapshot | null, planId: string | null): LiveState {
  const plan = snapshot?.plan;
  const structured = plan?.structured_plan as any;
  const entry = structured?.entry?.level ?? plan?.entry ?? null;
  const stop = plan?.stop ?? structured?.stop ?? null;
  const targets = structured?.targets?.length ? structured.targets : plan?.targets ?? [];
  return {
    planId: planId ?? plan?.plan_id ?? null,
    status: structured?.invalid ? 'invalid' : 'planned',
    rr: plan?.rr_to_t1 ?? null,
    lastPrice: entry ?? null,
    entry,
    stop,
    targets,
  };
}

export default function ReplayClient({ symbol, initialLivePlanId, initialSnapshot }: ReplayClientProps) {
  const upperSymbol = symbol.toUpperCase();
  const [live, setLive] = useState<LiveState>(() => deriveInitialLive(initialSnapshot, initialLivePlanId));
  const [priceSeries, setPriceSeries] = useState<{ time: number; value: number }[]>([]);
  const { livePlanId, scenarios, adoptAsLive, regenerateScenario, removeScenario, setLinked } = useScenarioStore(symbol);
  const activeLivePlanId = livePlanId || live.planId;

  // Subscribe to live plan deltas
  useEffect(() => {
    if (!activeLivePlanId) return;
    const wsUrl = `${WS_BASE_URL}/ws/plans/${encodeURIComponent(activeLivePlanId)}`;
    const socket = new WebSocket(wsUrl);
    socket.onmessage = (event) => {
      try {
        const payload = JSON.parse(event.data) as PlanDeltaEvent;
        if (payload.t !== 'plan_delta') return;
        setLive((prev) => ({
          ...prev,
          status: payload.changes.status || prev.status,
          rr: payload.changes.rr_to_t1 ?? prev.rr,
          lastPrice: payload.changes.last_price ?? prev.lastPrice,
        }));
        // Live invalidated => offer regen for linked scenarios
        if (payload.changes.status === 'invalidated' || payload.changes.status === 'plan_invalidated') {
          scenarios
            .filter((s) => s.linked_to_live)
            .forEach(async (s) => {
              try {
                const next = await regenerateScenario(s.scenario_style || 'intraday', activeLivePlanId);
                // Replace old card by deleting; addScenario already appended new one in regenerate
                removeScenario(s.plan_id);
                console.debug('telemetry', { t: 'scenario_regenerated', symbol: upperSymbol, style: s.scenario_style });
              } catch (e) {
                console.warn('linked scenario regen failed', e);
              }
            });
        }
      } catch (e) {
        console.warn('ws parse', e);
      }
    };
    return () => socket.close();
  }, [activeLivePlanId, regenerateScenario, removeScenario, scenarios, upperSymbol]);

  // Price stream for symbol
  useEffect(() => {
    const url = `${API_BASE_URL}/stream/${encodeURIComponent(upperSymbol)}`;
    const sse = new EventSource(url);
    sse.onmessage = (event) => {
      try {
        const payload = JSON.parse(event.data) as SymbolTickEvent;
        if (payload.t === 'tick') {
          const ts = Math.floor(new Date(payload.ts).getTime() / 1000);
          setPriceSeries((prev) => {
            const next = [...prev, { time: ts, value: payload.p }];
            if (next.length > 720) next.splice(0, next.length - 720);
            return next;
          });
          setLive((prev) => ({ ...prev, lastPrice: payload.p }));
        }
      } catch {}
    };
    sse.onerror = () => sse.close();
    return () => sse.close();
  }, [upperSymbol]);

  const [style, setStyle] = useState<ScenarioStyle>('intraday');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const generate = async () => {
    if (busy) return;
    setBusy(true);
    setError(null);
    try {
      await regenerateScenario(style, activeLivePlanId || null);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Generation failed');
    } finally {
      setBusy(false);
    }
  };

  const [compareId, setCompareId] = useState<string | null>(null);
  const comparePlan = useMemo(() => scenarios.find((s) => s.plan_id === compareId) || null, [scenarios, compareId]);

  return (
    <div className="space-y-8 px-6 py-10 sm:px-10">
      <header className="flex flex-col gap-2 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <div className="text-xs uppercase tracking-[0.3em] text-neutral-400">Market Replay</div>
          <h1 className="mt-1 text-3xl font-semibold text-white">{upperSymbol} · Live + Scenarios</h1>
          <p className="mt-1 text-sm text-neutral-400">Live plan auto-updates; scenarios are frozen snapshots unless linked.</p>
        </div>
        {activeLivePlanId ? (
          <Link href={`/plan/${encodeURIComponent(activeLivePlanId)}`} className="text-sm text-sky-300 underline">
            Open Live Console ↗
          </Link>
        ) : null}
      </header>

      <section className="rounded-3xl border border-neutral-800/80 bg-neutral-900/50 p-6 backdrop-blur">
        <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <div className="text-sm font-medium text-neutral-400">Live price</div>
            <div className="mt-1 text-3xl font-semibold text-emerald-300">{live.lastPrice ? live.lastPrice.toFixed(2) : '—'}</div>
          </div>
          <div className="flex items-center gap-2">
            {(['scalp','intraday','swing','reversal'] as ScenarioStyle[]).map((opt) => (
              <button
                key={opt}
                type="button"
                onClick={() => setStyle(opt)}
                disabled={opt === 'reversal'}
                className={`rounded-full px-3 py-1 text-sm ${style===opt? 'bg-sky-500/20 text-sky-200 border border-sky-500/40':'bg-neutral-800/60 text-neutral-200 border border-neutral-700/60'} ${opt==='reversal'?'opacity-50 cursor-not-allowed':''}`}
                title={opt==='reversal' ? 'Reversal strategy gated until server support' : ''}
              >
                {opt.charAt(0).toUpperCase() + opt.slice(1)}
              </button>
            ))}
            <button
              type="button"
              onClick={generate}
              disabled={busy}
              className="rounded-full border border-emerald-400/60 bg-emerald-400/10 px-4 py-2 text-sm font-semibold text-emerald-200 transition hover:bg-emerald-400/20 disabled:opacity-60"
            >
              {busy ? 'Generating…' : 'Generate Plan'}
            </button>
          </div>
        </div>
        <div className="mt-6 overflow-hidden rounded-2xl border border-neutral-800/70 bg-neutral-950/40">
          <PriceChart
            data={priceSeries}
            lastPrice={live.lastPrice ?? undefined}
            entry={live.entry}
            stop={live.stop}
            targets={live.targets}
            compare={comparePlan ? { entry: comparePlan.entry ?? undefined, stop: comparePlan.stop ?? undefined, targets: comparePlan.tps ?? [], label: 'Scenario' } : null}
          />
        </div>
        {error && <p className="mt-3 text-sm text-rose-300">{error}</p>}
      </section>

      <section className="space-y-4">
        <h2 className="text-xs font-semibold uppercase tracking-[0.3em] text-neutral-400">Scenarios</h2>
        {scenarios.length === 0 ? (
          <p className="text-sm text-neutral-400">No scenarios yet. Choose a style and generate a plan.</p>
        ) : (
          <ul className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
            {scenarios.map((s) => (
              <li key={s.plan_id} className="rounded-2xl border border-neutral-800/70 bg-neutral-900/60 p-4">
                <div className="mb-2 flex items-center justify-between">
                  <div className="flex items-center gap-2 text-xs">
                    <span className="rounded-full bg-neutral-800/80 px-2 py-0.5 text-neutral-200">Scenario</span>
                    <span className="rounded-full bg-sky-800/40 px-2 py-0.5 text-sky-200">{s.scenario_style}</span>
                    {s.plan_id === activeLivePlanId ? (
                      <span className="rounded-full bg-emerald-800/40 px-2 py-0.5 text-emerald-200">Live</span>
                    ) : (
                      <span className="rounded-full bg-neutral-800/60 px-2 py-0.5 text-neutral-200">Frozen</span>
                    )}
                  </div>
                  {s.chart_url ? (
                    <Link href={s.chart_url} target="_blank" className="text-xs text-sky-300 underline">
                      Open chart ↗
                    </Link>
                  ) : (
                    <span className="text-xs text-neutral-500">Chart unavailable</span>
                  )}
                </div>
                <div className="flex items-center justify-between text-sm">
                  <div className="text-neutral-400">Entry</div>
                  <div className="font-mono text-neutral-100">{s.entry?.toFixed(2) ?? '—'}</div>
                </div>
                <div className="mt-1 flex items-center justify-between text-sm">
                  <div className="text-neutral-400">Stop</div>
                  <div className="font-mono text-rose-200">{s.stop?.toFixed(2) ?? '—'}</div>
                </div>
                <div className="mt-1 flex items-center justify-between text-sm">
                  <div className="text-neutral-400">TPs</div>
                  <div className="font-mono text-emerald-200">{(s.tps||[]).slice(0,3).map(v=>v.toFixed(2)).join(', ') || '—'}</div>
                </div>
                <div className="mt-3 flex flex-wrap gap-2 text-xs">
                  <button
                    type="button"
                    onClick={() => setCompareId(s.plan_id === compareId ? null : s.plan_id)}
                    className={`rounded-full px-3 py-1 ${compareId===s.plan_id?'bg-sky-500/20 text-sky-200 border border-sky-500/40':'bg-neutral-800/60 text-neutral-200 border border-neutral-700/60'}`}
                  >
                    {compareId === s.plan_id ? 'Hide Compare' : 'Compare'}
                  </button>
                  <button
                    type="button"
                    onClick={() => { adoptAsLive(s.plan_id); console.debug('telemetry', { t: 'scenario_adopted' }); }}
                    className="rounded-full border border-emerald-400/60 bg-emerald-400/10 px-3 py-1 text-emerald-200"
                  >
                    Adopt as Live
                  </button>
                  <button
                    type="button"
                    onClick={async () => { await regenerateScenario(s.scenario_style || 'intraday', activeLivePlanId || null); console.debug('telemetry', { t: 'scenario_regenerated' }); }}
                    className="rounded-full border border-amber-400/60 bg-amber-400/10 px-3 py-1 text-amber-200"
                  >
                    Regenerate
                  </button>
                  <button
                    type="button"
                    onClick={() => { removeScenario(s.plan_id); console.debug('telemetry', { t: 'scenario_deleted' }); }}
                    className="rounded-full border border-rose-400/60 bg-rose-400/10 px-3 py-1 text-rose-200"
                  >
                    Delete
                  </button>
                  <label className="ml-auto flex items-center gap-2 text-neutral-300">
                    <input type="checkbox" checked={!!s.linked_to_live} onChange={(e)=> setLinked(s.plan_id, e.target.checked)} />
                    <span>Link to Live</span>
                  </label>
                </div>
              </li>
            ))}
          </ul>
        )}
      </section>
    </div>
  );
}
