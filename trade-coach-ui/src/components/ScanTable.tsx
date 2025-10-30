'use client';

import React, { useEffect, useMemo, useState } from 'react';
import clsx from 'clsx';
import { API_BASE_URL, withAuthHeaders } from '@/lib/env';
import { ensureCanonicalChartUrl } from '@/lib/chartUrl';

export type ScanTableRow = {
  plan_id?: string;
  rank: number;
  symbol: string;
  bias?: string | null;
  confidence?: number | null;
  setup?: string | null;
  confluence?: string[] | null;
  notes?: string[] | null;
  entry?: number | null;
  stop?: number | null;
  tps?: number[] | null;
  metrics?: Record<string, unknown> | null;
  charts?: { params?: Record<string, unknown> | null } | null;
  [key: string]: unknown;
};

type ChartLinkState = {
  href: string | null;
  loading: boolean;
  error: string | null;
};

type ScanTableProps = {
  rows: ScanTableRow[];
};

async function buildChartLink(row: ScanTableRow): Promise<string | null> {
  const params = row.charts?.params;
  if (!params || Object.keys(params).length === 0) return null;

  const body: Record<string, unknown> = { ...params };
  body.symbol = body.symbol ?? row.symbol;
  body.direction = body.direction ?? row.bias ?? row['direction'];
  const entry = body.entry ?? row.entry ?? row['plan']?.entry;
  const stop = body.stop ?? row.stop ?? row['plan']?.stop;
  const targets = body.tp ?? row.tps ?? row['plan']?.targets;
  if (entry != null) body.entry = entry;
  if (stop != null) body.stop = stop;
  if (!body.tp && Array.isArray(targets)) {
    body.tp = targets.filter((value): value is number => typeof value === 'number').join(',');
  }

  try {
    const res = await fetch(`${API_BASE_URL}/gpt/chart-url`, {
      method: 'POST',
      headers: withAuthHeaders({ 'Content-Type': 'application/json', Accept: 'application/json' }),
      body: JSON.stringify(body),
      cache: 'no-store',
    });
    if (!res.ok) {
      return null;
    }
    const data = (await res.json()) as { interactive?: string };
    return ensureCanonicalChartUrl(data?.interactive ?? null);
  } catch {
    return null;
  }
}

export default function ScanTable({ rows }: ScanTableProps) {
  const [links, setLinks] = useState<Record<string, ChartLinkState>>({});

  const keyedRows = useMemo(
    () =>
      rows.map((row) => ({
        ...row,
        key: row.plan_id ?? `${row.symbol}-${row.rank}`,
      })),
    [rows],
  );

  useEffect(() => {
    let cancelled = false;
    const nextState: Record<string, ChartLinkState> = {};
    keyedRows.forEach((row) => {
      const key = row.key as string;
      nextState[key] = links[key] ?? { href: null, loading: !!row.charts?.params, error: null };
    });
    setLinks(nextState);

    async function resolve() {
      for (const row of keyedRows) {
        const key = row.key as string;
        if (!row.charts?.params) continue;
        setLinks((prev) => ({ ...prev, [key]: { href: prev[key]?.href ?? null, loading: true, error: null } }));
        const href = await buildChartLink(row);
        if (cancelled) return;
        setLinks((prev) => ({ ...prev, [key]: { href, loading: false, error: href ? null : 'Unavailable' } }));
      }
    }

    resolve();
    return () => {
      cancelled = true;
    };
  }, [keyedRows]);

  if (rows.length === 0) {
    return <p className="text-sm text-neutral-400">No candidates available.</p>;
  }

  return (
    <div className="overflow-x-auto rounded-3xl border border-neutral-800/60 bg-neutral-900/50">
      <table className="min-w-full text-left text-sm text-neutral-200">
        <thead className="border-b border-neutral-800 bg-neutral-900/70 text-xs uppercase tracking-[0.25em] text-neutral-500">
          <tr>
            <th className="px-4 py-3">Rank</th>
            <th className="px-4 py-3">Symbol</th>
            <th className="px-4 py-3">Bias</th>
            <th className="px-4 py-3">Confidence</th>
            <th className="px-4 py-3">Trigger</th>
            <th className="px-4 py-3">Context</th>
            <th className="px-4 py-3">Last</th>
            <th className="px-4 py-3">EM%</th>
            <th className="px-4 py-3">RVOL</th>
            <th className="px-4 py-3">Liquidity</th>
            <th className="px-4 py-3">Momentum</th>
            <th className="px-4 py-3">ETA</th>
            <th className="px-4 py-3">Chart</th>
          </tr>
        </thead>
        <tbody>
          {keyedRows.map((row) => {
            const key = row.key as string;
            const metrics = (row.metrics ?? {}) as Record<string, unknown>;
            const contextItems = (row.confluence ?? metrics['confluence'] ?? []) as string[];
            const emPercent = metrics['expected_move_pct'] ?? metrics['em_pct'] ?? row['em_percent'];
            const rvol = metrics['rvol'] ?? row['rvol'];
            const liquidity = metrics['liquidity'] ?? row['liquidity'];
            const momentum = metrics['momentum'] ?? row['momentum'];
            const trigger = row.setup ?? (row['strategy_id'] as string | undefined) ?? '—';
            const badgeLink = links[key];
            const last = metrics['last'] ?? metrics['price'] ?? row['last'];
            const planBlock = (row['plan'] ?? {}) as Record<string, unknown>;
            const structuredPlan = (row['structured_plan'] ?? {}) as Record<string, unknown>;
            const expectedDuration = (planBlock['expected_duration'] ?? structuredPlan['expected_duration']) as
              | { minutes?: number }
              | undefined;
            const etaDisplay =
              expectedDuration && typeof expectedDuration.minutes === 'number'
                ? `~${Math.round(Number(expectedDuration.minutes))}m`
                : '—';
            return (
              <tr key={key} className="border-b border-neutral-800/60 hover:bg-neutral-900/80">
                <td className="px-4 py-3 font-semibold text-neutral-300">{row.rank}</td>
                <td className="px-4 py-3 font-mono text-sm tracking-widest text-white">{row.symbol.toUpperCase()}</td>
                <td className="px-4 py-3 text-neutral-200">{(row.bias ?? row['direction'] ?? '—').toString().toUpperCase()}</td>
                <td className="px-4 py-3 text-neutral-200">{row.confidence != null ? `${(Number(row.confidence) * 100).toFixed(0)}%` : '—'}</td>
                <td className="px-4 py-3 text-neutral-200">{trigger.replace(/_/g, ' ')}</td>
                <td className="px-4 py-3 text-neutral-400">
                  {Array.isArray(contextItems) && contextItems.length > 0 ? contextItems.slice(0, 3).join(' · ') : '—'}
                </td>
                <td className="px-4 py-3 text-neutral-200">{typeof last === 'number' ? last.toFixed(2) : '—'}</td>
                <td className="px-4 py-3 text-neutral-200">{typeof emPercent === 'number' ? `${(Number(emPercent) * 100).toFixed(0)}%` : '—'}</td>
                <td className="px-4 py-3 text-neutral-200">{typeof rvol === 'number' ? rvol.toFixed(2) : '—'}</td>
                <td className="px-4 py-3 text-neutral-200">{typeof liquidity === 'number' ? liquidity.toFixed(2) : '—'}</td>
                <td className="px-4 py-3 text-neutral-200">{typeof momentum === 'number' ? momentum.toFixed(2) : '—'}</td>
                <td className="px-4 py-3 text-neutral-200">{etaDisplay}</td>
                <td className="px-4 py-3">
                  {row.charts?.params ? (
                    badgeLink?.loading ? (
                      <span className="text-xs text-neutral-500">Loading…</span>
                    ) : badgeLink?.href ? (
                      <a
                        href={badgeLink.href}
                        className="text-xs font-semibold text-sky-300 underline decoration-dotted underline-offset-4"
                        target="_blank"
                        rel="noreferrer"
                      >
                        Open chart
                      </a>
                    ) : (
                      <span className="text-xs text-rose-300">No link</span>
                    )
                  ) : (
                    <span className="text-xs text-neutral-500">—</span>
                  )}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
