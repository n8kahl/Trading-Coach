'use client';

import React, { useEffect, useMemo, useState } from 'react';
import clsx from 'clsx';
import { API_BASE_URL, withAuthHeaders } from '@/lib/env';
import type {
  Badge,
  ExpectedDuration,
  PlanSnapshot,
  StrategyProfile,
  StructuredPlan,
  TargetMetaEntry,
} from '@/lib/types';

type PlanDetailProps = {
  plan: PlanSnapshot['plan'];
  structured?: StructuredPlan | null;
  planId: string;
};

type OverlayState =
  | { status: 'idle' | 'loading'; data: null; warning: null; error: null }
  | { status: 'ready'; data: ChartLayers; warning: null; error: null }
  | { status: 'warning'; data: ChartLayers | null; warning: string; error: null }
  | { status: 'error'; data: null; warning: null; error: string };

type ChartLayers = {
  plan_id: string;
  as_of?: string;
  levels: Array<Record<string, unknown>>;
  zones: Array<Record<string, unknown>>;
  annotations: Array<Record<string, unknown>>;
  meta?: Record<string, unknown>;
};

function formatNumber(value: number | null | undefined, decimals = 2): string {
  if (value == null || Number.isNaN(value)) return '—';
  return Number(value).toFixed(decimals);
}

function renderTargetMeta(meta: TargetMetaEntry | undefined): JSX.Element {
  if (!meta) return <span className="text-xs text-neutral-500">No meta</span>;

  const chips: JSX.Element[] = [];
  const lines: JSX.Element[] = [];
  const snapFraction =
    typeof meta.snap_fraction === 'number'
      ? meta.snap_fraction
      : typeof meta.em_fraction === 'number'
        ? meta.em_fraction
        : undefined;
  const idealFraction = typeof meta.ideal_fraction === 'number' ? meta.ideal_fraction : undefined;
  const deviationFraction =
    idealFraction != null && snapFraction != null ? Math.abs(snapFraction - idealFraction) : undefined;

  if (typeof meta.snap_price === 'number') {
    const fractionLabel = snapFraction != null ? `${(snapFraction * 100).toFixed(0)}% EM` : null;
    lines.push(
      <span key="snap" className="font-mono text-emerald-200">
        Snapped {meta.snap_price.toFixed(2)}
        {fractionLabel ? <span className="text-neutral-400"> ({fractionLabel})</span> : null}
      </span>,
    );
  }
  if (typeof meta.ideal_price === 'number') {
    const fractionLabel = idealFraction != null ? `${(idealFraction * 100).toFixed(0)}% EM` : null;
    lines.push(
      <span key="ideal" className="font-mono text-cyan-200">
        Ideal {meta.ideal_price.toFixed(2)}
        {fractionLabel ? <span className="text-neutral-400"> ({fractionLabel})</span> : null}
      </span>,
    );
  }
  if (typeof meta.snap_deviation === 'number') {
    lines.push(
      <span key="delta" className="text-xs text-neutral-400">
        Δ {meta.snap_deviation >= 0 ? '+' : ''}
        {meta.snap_deviation.toFixed(2)} pts
      </span>,
    );
  }

  if (typeof meta.prob_touch_calibrated === 'number') {
    chips.push(
      <span key="prob" className="rounded-full bg-emerald-500/20 px-2 py-0.5 text-xs text-emerald-200">
        {(meta.prob_touch_calibrated * 100).toFixed(0)}% touch
      </span>,
    );
  } else if (typeof meta.prob_touch === 'number') {
    chips.push(
      <span key="prob" className="rounded-full bg-emerald-500/20 px-2 py-0.5 text-xs text-emerald-200">
        {(meta.prob_touch * 100).toFixed(0)}% touch
      </span>,
    );
  }
  if (typeof meta.em_fraction === 'number') {
    chips.push(
      <span key="em" className="rounded-full bg-cyan-500/20 px-2 py-0.5 text-xs text-cyan-200">
        {(meta.em_fraction * 100).toFixed(0)}% EM
      </span>,
    );
  }
  if (meta.snap_tag) {
    chips.push(
      <span key="snap-tag" className="rounded-full bg-neutral-700/40 px-2 py-0.5 text-xs text-neutral-200">
        {meta.snap_tag}
      </span>,
    );
  }
  if (meta.em_capped) {
    chips.push(
      <span key="cap" className="rounded-full bg-amber-500/20 px-2 py-0.5 text-xs text-amber-200">EM capped</span>,
    );
  }
  if (meta.synthetic) {
    chips.push(
      <span key="synthetic" className="rounded-full bg-neutral-700/20 px-2 py-0.5 text-xs text-neutral-300">
        Synthetic ideal
      </span>,
    );
  }
  if (meta.watch_plan === 'true' || meta.watch_plan === true) {
    chips.push(
      <span key="watch" className="rounded-full bg-amber-500/20 px-2 py-0.5 text-xs text-amber-200">
        Watch plan
      </span>,
    );
  }
  if (deviationFraction != null && deviationFraction > 0.15) {
    chips.push(
      <span key="deviation" className="rounded-full bg-rose-500/20 px-2 py-0.5 text-xs text-rose-200">
        Snapped far from ideal · POT re-checked
      </span>,
    );
  }

  if (!chips.length && !lines.length) {
    return <span className="text-xs text-neutral-500">No meta</span>;
  }

  return (
    <div className="space-y-1">
      {lines.length ? <div className="flex flex-wrap gap-2 text-xs text-neutral-300">{lines}</div> : null}
      {chips.length ? <div className="flex flex-wrap gap-1">{chips}</div> : null}
    </div>
  );
}

function Badges({ badges }: { badges: Badge[] | undefined }) {
  if (!badges || badges.length === 0) return null;
  return (
    <div className="flex flex-wrap gap-2">
      {badges.slice(0, 5).map((badge) => (
        <span
          key={`${badge.kind}-${badge.label}`}
          className={clsx(
            'rounded-full px-3 py-1 text-xs font-semibold uppercase tracking-wide',
            badge.kind === 'strategy' && 'bg-emerald-500/15 text-emerald-200 border border-emerald-500/30',
            badge.kind === 'style' && 'bg-sky-500/15 text-sky-200 border border-sky-500/30',
            badge.kind === 'bias' && 'bg-rose-500/15 text-rose-200 border border-rose-500/30',
            badge.kind === 'confluence' && 'bg-neutral-700/30 text-neutral-100 border border-neutral-600/50',
            badge.kind === 'meta' && 'bg-amber-500/15 text-amber-200 border border-amber-500/30',
          )}
        >
          {badge.label}
        </span>
      ))}
    </div>
  );
}

export default function PlanDetail({ plan, structured, planId }: PlanDetailProps) {
  const [overlays, setOverlays] = useState<OverlayState>({ status: 'idle', data: null, warning: null, error: null });

  const targetMeta = useMemo(() => {
    const entries = Array.isArray(plan.target_meta) ? plan.target_meta : [];
    if (entries.length === 0 && Array.isArray(structured?.targets)) {
      return structured?.targets.map((price, index) => ({ label: `TP${index + 1}`, price }));
    }
    return entries as TargetMetaEntry[];
  }, [plan.target_meta, structured?.targets]);

  const watchPlan = useMemo(
    () => targetMeta.some((entry) => entry?.watch_plan === 'true' || entry?.watch_plan === true),
    [targetMeta],
  );

  const { nodes: candidateNodes, selectedNode } = useMemo(() => {
    const root = targetMeta.find(
      (entry) => Array.isArray(entry?.candidate_nodes) && (entry?.candidate_nodes as unknown[]).length > 0,
    );
    if (!root || !Array.isArray(root.candidate_nodes)) {
      return { nodes: [] as Array<{ label: string; price: number | null; distance?: number | null; fraction?: number | null; rr_multiple?: number | null }>, selectedNode: undefined as string | undefined };
    }
    const normaliseNumber = (value: unknown): number | null => {
      if (typeof value === 'number' && Number.isFinite(value)) return value;
      const parsed = Number(value);
      return Number.isFinite(parsed) ? parsed : null;
    };
    const nodes = (root.candidate_nodes as Array<Record<string, unknown>>)
      .map((node) => {
        const price = normaliseNumber(node.price);
        const distance = normaliseNumber(node.distance);
        const fraction = normaliseNumber(node.fraction);
        const rr = normaliseNumber(node.rr_multiple);
        return {
          label: String(node.label ?? 'LEVEL').toUpperCase(),
          price,
          distance,
          fraction,
          rr_multiple: rr,
        };
      })
      .filter((node) => node.price != null);
    const selected =
      typeof root.selected_node === 'string' && root.selected_node.trim()
        ? root.selected_node.toUpperCase()
        : undefined;
    return { nodes, selectedNode: selected };
  }, [targetMeta]);

  useEffect(() => {
    let cancelled = false;
    const controller = new AbortController();

    async function loadOverlays() {
      setOverlays({ status: 'loading', data: null, warning: null, error: null });
      try {
        const res = await fetch(`${API_BASE_URL}/api/v1/gpt/chart-layers?plan_id=${encodeURIComponent(planId)}`, {
          headers: withAuthHeaders({ Accept: 'application/json' }),
          signal: controller.signal,
        });
        if (cancelled) return;
        if (res.status === 409) {
          const detail = (await res.json().catch(() => ({}))) as Record<string, unknown>;
          setOverlays({
            status: 'warning',
            data: null,
            warning: (detail?.message as string) ?? 'Plan overlays are stale.',
            error: null,
          });
          return;
        }
        if (!res.ok) {
          setOverlays({ status: 'error', data: null, warning: null, error: `Request failed (${res.status})` });
          return;
        }
        const payload = (await res.json()) as ChartLayers;
        if (cancelled) return;
        setOverlays({ status: 'ready', data: payload, warning: null, error: null });
      } catch (error) {
        if (cancelled) return;
        setOverlays({
          status: 'error',
          data: null,
          warning: null,
          error: error instanceof Error ? error.message : 'Overlay request failed',
        });
      }
    }

    void loadOverlays();
    return () => {
      cancelled = true;
      controller.abort();
    };
  }, [planId]);

  const badges = plan.badges ?? structured?.badges ?? [];
  const strategyProfile = (plan.strategy_profile ?? structured?.strategy_profile) as StrategyProfile | undefined;
  const expectedDuration = (plan.expected_duration ?? structured?.expected_duration) as ExpectedDuration | undefined;
  const structuredRecord = structured && typeof structured === 'object' ? (structured as Record<string, unknown>) : null;
  const riskBlock = (plan.risk_block as Record<string, unknown> | undefined) ?? (structuredRecord?.risk_block as Record<string, unknown> | undefined);
  const executionRules =
    (plan.execution_rules as Record<string, unknown> | undefined) ??
    (structuredRecord?.execution_rules as Record<string, unknown> | undefined);
  const sourcePaths = (plan.source_paths ?? {}) as Record<string, string>;
  const optionsContracts = Array.isArray(plan.options_contracts) ? plan.options_contracts : [];
  const rejectedContracts = Array.isArray(plan.rejected_contracts) ? plan.rejected_contracts : [];

  return (
    <section className="space-y-6 rounded-3xl border border-neutral-800/70 bg-neutral-900/60 p-6 shadow-lg shadow-emerald-500/5">
      <header className="space-y-2">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h2 className="text-sm font-semibold uppercase tracking-[0.3em] text-neutral-400">Plan Detail</h2>
            {plan.session_state?.banner ? (
              <div className="mt-2 rounded-lg bg-amber-500/20 px-3 py-2 text-xs text-amber-100">
                {plan.session_state.banner}
              </div>
            ) : null}
          </div>
          <Badges badges={badges} />
        </div>
        {strategyProfile ? (
          <div className="rounded-2xl border border-neutral-800/80 bg-neutral-950/60 p-4 text-sm text-neutral-200">
            <h3 className="text-base font-semibold text-white">{strategyProfile.name}</h3>
            <div className="mt-2 grid gap-2 sm:grid-cols-2">
              <div>
                <p className="text-xs uppercase tracking-[0.2em] text-neutral-500">Trigger</p>
                <ul className="mt-1 list-disc pl-5 text-sm text-neutral-200">
                  {strategyProfile.trigger.map((line) => (
                    <li key={line}>{line}</li>
                  ))}
                </ul>
              </div>
              <div className="space-y-2 text-sm text-neutral-200">
                <div>
                  <p className="text-xs uppercase tracking-[0.2em] text-neutral-500">Invalidation</p>
                  <p className="mt-1 leading-relaxed text-neutral-300">{strategyProfile.invalidation}</p>
                </div>
                <div>
                  <p className="text-xs uppercase tracking-[0.2em] text-neutral-500">Management</p>
                  <p className="mt-1 leading-relaxed text-neutral-300">{strategyProfile.management}</p>
                </div>
                {strategyProfile.reload ? (
                  <div>
                    <p className="text-xs uppercase tracking-[0.2em] text-neutral-500">Reload</p>
                    <p className="mt-1 leading-relaxed text-neutral-300">{strategyProfile.reload}</p>
                  </div>
                ) : null}
                {strategyProfile.runner ? (
                  <div>
                    <p className="text-xs uppercase tracking-[0.2em] text-neutral-500">Runner</p>
                    <p className="mt-1 leading-relaxed text-neutral-300">{strategyProfile.runner}</p>
                  </div>
                ) : null}
                {expectedDuration ? (
                  <div>
                    <p className="text-xs uppercase tracking-[0.2em] text-neutral-500">Expected Duration</p>
                    <p className="mt-1 leading-relaxed text-neutral-200">{expectedDuration.label}</p>
                    <p className="text-xs text-neutral-500">
                      ~{expectedDuration.minutes} min
                      {expectedDuration.basis?.length ? ` · ${expectedDuration.basis.join(', ')}` : ''}
                    </p>
                  </div>
                ) : null}
              </div>
            </div>
          </div>
        ) : expectedDuration ? (
          <div className="rounded-2xl border border-neutral-800/80 bg-neutral-950/60 p-4 text-sm text-neutral-200">
            <p className="text-xs uppercase tracking-[0.2em] text-neutral-500">Expected Duration</p>
            <p className="mt-2 text-base font-semibold text-white">{expectedDuration.label}</p>
            <p className="mt-1 text-xs text-neutral-500">
              ~{expectedDuration.minutes} min
              {expectedDuration.basis?.length ? ` · ${expectedDuration.basis.join(', ')}` : ''}
            </p>
          </div>
        ) : null}
      </header>

      {watchPlan ? (
        <div className="rounded-2xl border border-amber-500/50 bg-amber-500/10 px-4 py-3 text-xs text-amber-100">
          RR below floor on TP1 &mdash; treating plan as watch-only until structure improves.
        </div>
      ) : null}

      <div className="overflow-hidden rounded-2xl border border-neutral-800/70">
        <table className="min-w-full text-sm text-neutral-200">
          <thead className="bg-neutral-900/70 text-xs uppercase tracking-[0.25em] text-neutral-400">
            <tr>
              <th className="px-4 py-3 text-left">Leg</th>
              <th className="px-4 py-3 text-left">Value</th>
              <th className="px-4 py-3 text-left">Meta</th>
            </tr>
          </thead>
          <tbody>
            <tr className="border-t border-neutral-800/70">
              <td className="px-4 py-3 font-medium text-neutral-100" title={sourcePaths['entry'] ? `Source: ${sourcePaths['entry']}` : undefined}>
                Entry
              </td>
              <td className="px-4 py-3 font-mono">{formatNumber(plan.entry ?? structured?.entry?.level ?? null)}</td>
              <td className="px-4 py-3 text-xs text-neutral-500">Reference entry published by geometry engine.</td>
            </tr>
            <tr className="border-t border-neutral-800/70">
              <td className="px-4 py-3 font-medium text-neutral-100" title={sourcePaths['stop'] ? `Source: ${sourcePaths['stop']}` : undefined}>
                Stop
              </td>
              <td className="px-4 py-3 font-mono text-rose-200">{formatNumber(plan.stop ?? structured?.stop ?? null)}</td>
              <td className="px-4 py-3 text-xs text-neutral-500">Invalidation level derived from structure/ATR.</td>
            </tr>
            {targetMeta.map((meta, index) => (
              <tr key={meta.label ?? index} className="border-t border-neutral-800/70">
                <td
                  className="px-4 py-3 font-medium text-neutral-100"
                  title={sourcePaths[`tp${index + 1}`] ? `Source: ${sourcePaths[`tp${index + 1}`]}` : undefined}
                >
                  {meta.label ?? `TP${index + 1}`}
                </td>
                <td className="px-4 py-3 font-mono text-emerald-200">{formatNumber(meta.price ?? plan.targets?.[index])}</td>
                <td className="px-4 py-3">{renderTargetMeta(meta)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {candidateNodes.length ? (
        <details className="rounded-2xl border border-neutral-800/70 bg-neutral-950/60 p-4 text-sm text-neutral-200" open={watchPlan}>
          <summary className="cursor-pointer list-none text-xs uppercase tracking-[0.3em] text-neutral-500 outline-none">
            Candidate TP ladder
            <span className="ml-2 text-[0.65rem] text-neutral-400">
              ({candidateNodes.length})
            </span>
          </summary>
          <ul className="mt-3 space-y-2 text-sm">
            {candidateNodes.slice(0, 10).map((node, idx) => {
              const isSelected = selectedNode && node.label === selectedNode;
              const distanceLabel =
                typeof node.distance === 'number' && Number.isFinite(node.distance)
                  ? `${node.distance.toFixed(2)} pts`
                  : null;
              const fractionLabel =
                typeof node.fraction === 'number' && Number.isFinite(node.fraction)
                  ? `${(node.fraction * 100).toFixed(0)}% EM`
                  : null;
              const rrLabel =
                typeof node.rr_multiple === 'number' && Number.isFinite(node.rr_multiple)
                  ? `RR ${node.rr_multiple.toFixed(2)}`
                  : null;
              return (
                <li
                  key={`${node.label}-${idx}`}
                  className={clsx(
                    'rounded-xl border px-3 py-2 transition',
                    isSelected
                      ? 'border-emerald-500/50 bg-emerald-500/10 text-emerald-100'
                      : 'border-neutral-800/70 bg-neutral-900/50 text-neutral-200',
                  )}
                >
                  <div className="flex items-center justify-between">
                    <span className="font-semibold">{node.label}</span>
                    <span className="font-mono text-sm">{formatNumber(node.price)}</span>
                  </div>
                  <div className="mt-1 flex flex-wrap gap-3 text-xs text-neutral-400">
                    {distanceLabel ? <span>{distanceLabel}</span> : null}
                    {fractionLabel ? <span>{fractionLabel}</span> : null}
                    {rrLabel ? <span>{rrLabel}</span> : null}
                  </div>
                </li>
              );
            })}
          </ul>
        </details>
      ) : null}

      {riskBlock ? (
        <div className="rounded-2xl border border-neutral-800/80 bg-neutral-950/60 p-4 text-sm text-neutral-200">
          <h3 className="text-xs uppercase tracking-[0.3em] text-neutral-500">Risk Block</h3>
          <div className="mt-2 grid gap-3 sm:grid-cols-2">
            {Object.entries(riskBlock).map(([key, value]) => (
              <div key={key}>
                <div className="text-xs uppercase tracking-[0.2em] text-neutral-500">{key.replace(/_/g, ' ')}</div>
                <div className="mt-1 text-neutral-200">{typeof value === 'number' ? value.toString() : String(value)}</div>
              </div>
            ))}
          </div>
        </div>
      ) : null}

      {executionRules ? (
        <div className="rounded-2xl border border-neutral-800/80 bg-neutral-950/60 p-4 text-sm text-neutral-200">
          <h3 className="text-xs uppercase tracking-[0.3em] text-neutral-500">Execution Rules</h3>
          <ul className="mt-2 list-disc pl-5">
            {Object.entries(executionRules).map(([key, value]) => (
              <li key={key} className="leading-relaxed text-neutral-300">
                <span className="font-medium text-neutral-200">{key.replace(/_/g, ' ')}:</span> {String(value)}
              </li>
            ))}
          </ul>
        </div>
      ) : null}

      <div className="grid gap-4 lg:grid-cols-2">
        <div className="rounded-2xl border border-neutral-800/60 bg-neutral-950/70 p-4">
          <h3 className="text-xs uppercase tracking-[0.3em] text-neutral-500">Options</h3>
          {optionsContracts.length > 0 ? (
            <ul className="mt-2 space-y-2 text-sm">
              {optionsContracts.slice(0, 3).map((contract, index) => (
                <li key={index} className="rounded-xl border border-neutral-800/70 bg-neutral-900/60 px-3 py-2">
                  <div className="font-mono text-neutral-100">{contract['symbol'] ?? contract['label'] ?? 'Contract'}</div>
                  <div className="mt-1 flex flex-wrap gap-3 text-xs text-neutral-400">
                    {contract['delta'] != null ? <span>Δ {Number(contract['delta']).toFixed(2)}</span> : null}
                    {contract['spread_pct'] != null ? <span>Spread {Number(contract['spread_pct']).toFixed(2)}%</span> : null}
                    {contract['open_interest'] != null ? <span>OI {contract['open_interest']}</span> : null}
                  </div>
                </li>
              ))}
            </ul>
          ) : (
            <p className="mt-2 text-sm text-neutral-400">{plan.options_note ?? 'No eligible contracts supplied.'}</p>
          )}
          {rejectedContracts.length > 0 ? (
            <div className="mt-3 space-y-1 text-xs text-neutral-500">
              {rejectedContracts.map((item, idx) => (
                <div key={`${item['symbol']}-${idx}`}>⚠ {item['reason']}: {item['message'] ?? item['symbol']}</div>
              ))}
            </div>
          ) : null}
        </div>

        <div className="rounded-2xl border border-neutral-800/60 bg-neutral-950/70 p-4">
          <h3 className="text-xs uppercase tracking-[0.3em] text-neutral-500">Overlays</h3>
          {overlays.status === 'loading' && <p className="text-sm text-neutral-400">Loading overlays…</p>}
          {overlays.status === 'error' && (
            <p className="text-sm text-rose-300">{overlays.error}</p>
          )}
          {overlays.status === 'warning' && (
            <div className="rounded-xl border border-amber-500/50 bg-amber-500/15 px-3 py-2 text-xs text-amber-100">
              {overlays.warning}
            </div>
          )}
          {overlays.status === 'ready' && overlays.data ? (
            <div className="mt-3 grid gap-3 text-sm text-neutral-200">
              <div>
                <div className="text-xs uppercase tracking-[0.2em] text-neutral-500">Levels</div>
                <ul className="mt-1 space-y-1 text-xs text-neutral-300">
                  {overlays.data.levels.length === 0 ? (
                    <li>No levels published.</li>
                  ) : (
                    overlays.data.levels.slice(0, 6).map((level, idx) => (
                      <li key={idx}>
                        {level['label'] ?? 'Level'} · {level['price'] ?? '—'}
                      </li>
                    ))
                  )}
                </ul>
              </div>
              <div>
                <div className="text-xs uppercase tracking-[0.2em] text-neutral-500">Zones</div>
                <ul className="mt-1 space-y-1 text-xs text-neutral-300">
                  {overlays.data.zones.length === 0 ? (
                    <li>No zones published.</li>
                  ) : (
                    overlays.data.zones.slice(0, 6).map((zone, idx) => (
                      <li key={idx}>
                        {zone['label'] ?? zone['kind'] ?? 'Zone'} · {zone['low']} → {zone['high']}
                      </li>
                    ))
                  )}
                </ul>
              </div>
            </div>
          ) : null}
        </div>
      </div>
    </section>
  );
}
