'use client';

import * as React from 'react';
import Link from 'next/link';
import clsx from 'clsx';
import * as ReactDOM from 'react';
import { useEffect, useState } from 'react';
import { fetchServerInfo } from '@/lib/serverInfo';
import { API_BASE } from '@/lib/hosts';

export function ResponsiveShell({
  title,
  children,
}: { title?: string; children: React.ReactNode }) {
  return (
    <div className="min-h-dvh bg-[var(--bg)] text-[var(--text)]">
      {/* Accessibility: Skip to content */}
      <a
        href="#main"
        className="sr-only focus:not-sr-only focus:absolute focus:top-2 focus:left-2 focus:z-50 bg-[var(--chip)] px-3 py-2 rounded"
      >
        Skip to content
      </a>

      <header className="sticky top-0 z-40 border-b border-[var(--border)] bg-[var(--surface)]/80 backdrop-blur">
        <nav
          aria-label="Primary"
          className="mx-auto flex max-w-7xl items-center justify-between px-3 sm:px-4 h-12"
        >
          <div className="flex items-center gap-3">
            <Link href="/" className="font-semibold tracking-wide">
              Trading Coach
            </Link>
            {title && (
              <span className="text-[var(--muted)] text-sm hidden sm:inline">
                {title}
              </span>
            )}
          </div>
          <ul className="flex items-center gap-2 text-sm">
            <li>
              <Link
                className="px-2 py-1 rounded hover:bg-[var(--chip)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--ring)]"
                href="/scans"
              >
                Scans
              </Link>
            </li>
            <li>
              <Link
                className="px-2 py-1 rounded hover:bg-[var(--chip)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--ring)]"
                href="/plan"
              >
                Plan
              </Link>
            </li>
            <li>
              <Link
                className="px-2 py-1 rounded hover:bg-[var(--chip)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--ring)]"
                href="/replay"
              >
                Replay
              </Link>
            </li>
          </ul>
        </nav>
      </header>

      <main id="main" role="main" className="mx-auto max-w-7xl px-3 sm:px-4 py-3">
        {children}
      </main>

      <footer className="mt-6 border-t border-[var(--border)] bg-[var(--surface)]/60">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-3 sm:px-4 py-2 text-xs text-[var(--muted)]">
          <ServerBadge />
          <span>UI: fancytraderui</span>
        </div>
      </footer>
    </div>
  );
}

export default ResponsiveShell;

function ServerBadge() {
  const [label, setLabel] = useState<string>('API: resolving…');
  useEffect(() => {
    let alive = true;
    fetchServerInfo()
      .then((info) => {
        if (!alive) return;
        if (!info) {
          setLabel(`API: ${new URL(API_BASE).host}`);
          return;
        }
        const host = new URL(API_BASE).host;
        const sha = (info.git_sha || 'unknown').slice(0, 7);
        setLabel(`API: ${info.service ?? host} (${sha})`);
      })
      .catch(() => {
        if (!alive) return;
        setLabel(`API: ${new URL(API_BASE).host}`);
      });
    return () => {
      alive = false;
    };
  }, []);
  return <span className="inline-block rounded bg-[var(--chip)] px-2 py-0.5">{label}</span>;
}
