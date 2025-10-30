'use client';

import * as React from 'react';

/**
 * Renders a deterministic ladder view for plan numbers.
 * Absolutely no client-side fabrication or re-computation.
 */
function Row({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="grid grid-cols-3 items-center gap-2">
      <div className="col-span-1 text-sm text-[var(--muted)]">{label}</div>
      <div className="col-span-2 font-mono">{value}</div>
    </div>
  );
}

export default function PlanSummaryCard({
  plan,
  collapsed,
}: {
  plan: any; // server-provided plan object; shape unchanged
  collapsed?: boolean;
}) {
  const bias = plan?.bias; // 'LONG' | 'SHORT' (from backend)
  const rr = plan?.riskReward; // number or { tp1: number } from backend
  const structure = plan?.structure; // 'INTACT' | 'INVALID'
  const details = plan?.details || {}; // entry, stop, last, horizon (names from backend)
  const targets = plan?.targets || []; // [{ label, price }, ...]

  return (
    <section
      aria-labelledby="plan-summary-title"
      className="rounded-xl border border-[var(--border)] bg-[var(--surface)] shadow-[var(--elev-1)]"
    >
      <header className="flex items-center justify-between px-3 py-2 border-b border-[var(--border)]">
        <h2 id="plan-summary-title" className="text-base font-semibold tracking-wide">
          {plan?.symbol}{' '}
          <span className="text-sm text-[var(--muted)]">{plan?.style}</span>
        </h2>
        <div className="flex items-center gap-1">
          {bias && (
            <span className="px-2 py-1 rounded bg-[var(--chip)] text-xs" aria-label={`Bias ${bias}`}>
              BIAS: {bias}
            </span>
          )}
          {rr != null && (
            <span className="px-2 py-1 rounded bg-[var(--chip)] text-xs" aria-label="Risk Reward">
              R:R (TP1): {typeof rr === 'number' ? rr : rr.tp1}
            </span>
          )}
          {structure && (
            <span
              className={`px-2 py-1 rounded text-xs ${
                structure === 'INTACT'
                  ? 'bg-emerald-500/10 text-emerald-300'
                  : 'bg-[var(--danger)]/10 text-[var(--danger)]'
              }`}
              aria-label={`Structure ${structure}`}
            >
              {structure === 'INTACT' ? 'PLAN INTACT' : 'INVALIDATED'}
            </span>
          )}
        </div>
      </header>

      {!collapsed && (
        <div className="grid gap-3 p-3">
          <div className="grid gap-2">
            {'entry' in details && <Row label="Entry" value={<span>{details.entry}</span>} />}
            {'stop' in details && (
              <Row label="Stop" value={<span className="text-[var(--danger)]">{details.stop}</span>} />
            )}
            {'last' in details && <Row label="Last Price" value={<span>{details.last}</span>} />}
            {'horizon' in details && <Row label="Horizon" value={<span>{details.horizon}</span>} />}
            {rr != null && (
              <Row
                label="R:R (TP1)"
                value={<span className="font-medium">{typeof rr === 'number' ? rr : rr.tp1}</span>}
              />
            )}
          </div>

          {targets?.length > 0 && (
            <div>
              <h3 className="mb-1 text-sm text-[var(--muted)]">Targets</h3>
              <ul className="grid gap-1 font-mono">
                {targets.map((t: any, i: number) => (
                  <li key={`${t.label ?? i}-${t.price}`} className="flex items-center gap-2">
                    <span className="inline-block w-12 text-xs text-[var(--muted)]">
                      {t.label ?? `TP${i + 1}`}
                    </span>
                    <span>{t.price}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {Array.isArray(plan?.checklist) && plan.checklist.length > 0 && (
            <div>
              <h3 className="mb-1 text-sm text-[var(--muted)]">Pre-entry checklist</h3>
              <ul className="list-disc pl-5 text-sm">
                {plan.checklist.map((line: string, idx: number) => (
                  <li key={idx}>{line}</li>
                ))}
              </ul>
            </div>
          )}

          {plan?.confluence && (
            <div className="flex items-center gap-2">
              <span className="text-sm text-[var(--muted)]">Confluence</span>
              <span className="px-2 py-1 rounded bg-[var(--chip)] text-xs">{plan.confluence}</span>
            </div>
          )}

          {plan?.next && (
            <div className="rounded bg-[var(--surface-2)] p-2 text-sm">
              <strong>Next:</strong> {plan.next}
            </div>
          )}
        </div>
      )}
    </section>
  );
}
