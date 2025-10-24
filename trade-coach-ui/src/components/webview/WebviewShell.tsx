"use client";

import clsx from "clsx";
import type { ReactNode } from "react";

type WebviewShellProps = {
  statusStrip: ReactNode;
  chartPanel: ReactNode;
  planPanel: ReactNode;
  actionsDock?: ReactNode;
  mobileSheet?: ReactNode;
  className?: string;
};

export default function WebviewShell({
  statusStrip,
  chartPanel,
  planPanel,
  actionsDock,
  mobileSheet,
  className,
}: WebviewShellProps) {
  return (
    <div className={clsx("relative min-h-screen bg-[#050709]", className)}>
      <div className="pointer-events-none fixed inset-0 bg-gradient-to-b from-emerald-500/8 via-transparent to-transparent" aria-hidden />
      <header className="sticky top-0 z-30 border-b border-neutral-900/70 backdrop-blur-md">{statusStrip}</header>
      <main className="relative z-10 mx-auto flex w-full max-w-[1400px] flex-1 flex-col gap-6 px-4 pb-24 pt-6 sm:px-6 lg:px-10">
        <div className="grid gap-6 lg:grid-cols-[minmax(0,1.9fr),minmax(0,1fr)]">
          <section className="min-h-[420px] rounded-3xl border border-neutral-800/70 bg-neutral-950/50 p-5 shadow-xl shadow-emerald-500/10 backdrop-blur">
            {chartPanel}
          </section>
          <aside className="hidden rounded-3xl border border-neutral-800/70 bg-neutral-950/40 p-5 shadow-lg shadow-emerald-500/5 backdrop-blur lg:block">
            {planPanel}
          </aside>
        </div>
      </main>
      {actionsDock ? <div className="fixed bottom-6 right-6 z-30 hidden md:block">{actionsDock}</div> : null}
      {mobileSheet}
    </div>
  );
}
