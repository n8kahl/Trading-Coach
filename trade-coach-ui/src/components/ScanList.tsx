'use client';

import * as React from 'react';
import Link from 'next/link';

/**
 * Renders scan results exactly in server order.
 * No client-side sorting or fabrication.
 */
export default function ScanList({ items }: { items: any[] }) {
  if (!items || items.length === 0) {
    return (
      <div className="rounded-lg border border-[var(--border)] bg-[var(--surface)] p-4 text-sm text-[var(--muted)]">
        No scan results.
      </div>
    );
  }

  return (
    <>
      {/* Desktop table */}
      <div className="hidden md:block overflow-x-auto rounded-lg border border-[var(--border)]">
        <table className="min-w-full bg-[var(--surface)] text-sm">
          <thead className="bg-[var(--surface-2)] text-[var(--muted)]">
            <tr>
              <th className="px-3 py-2 text-left">#</th>
              <th className="px-3 py-2 text-left">Symbol</th>
              <th className="px-3 py-2 text-left">Bias</th>
              <th className="px-3 py-2 text-left">Reason</th>
              <th className="px-3 py-2 text-left">Action</th>
            </tr>
          </thead>
          <tbody>
            {items.map((it, i) => (
              <tr key={it.id ?? i} className="border-b border-[var(--border)]">
                <td className="px-3 py-2">{i + 1}</td>
                <td className="px-3 py-2 font-mono">{it.symbol}</td>
                <td className="px-3 py-2">{it.bias}</td>
                <td className="px-3 py-2 text-[var(--muted)]">{it.reason}</td>
                <td className="px-3 py-2">
                  <Link
                    href={`/plan/${it.planId}`}
                    className="rounded bg-[var(--chip)] px-2.5 py-1.5 hover:bg-[var(--border)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--ring)]"
                  >
                    Open Plan
                  </Link>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Mobile cards */}
      <ul className="md:hidden grid gap-2">
        {items.map((it, i) => (
          <li
            key={it.id ?? i}
            className="rounded-xl border border-[var(--border)] bg-[var(--surface)] p-3"
          >
            <div className="flex items-center justify-between">
              <div className="font-medium">
                {i + 1}. <span className="font-mono">{it.symbol}</span>
              </div>
              <span className="text-xs px-2 py-1 rounded bg-[var(--chip)]">{it.bias}</span>
            </div>
            <p className="mt-2 text-sm text-[var(--muted)]">{it.reason}</p>
            <div className="mt-2 flex justify-end">
              <Link
                className="rounded bg-[var(--chip)] px-2.5 py-1.5 hover:bg-[var(--border)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--ring)]"
                href={`/plan/${it.planId}`}
              >
                Open
              </Link>
            </div>
          </li>
        ))}
      </ul>
    </>
  );
}
