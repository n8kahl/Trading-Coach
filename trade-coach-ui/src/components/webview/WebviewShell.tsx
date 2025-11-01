"use client";

import clsx from "clsx";
import type { ReactNode } from "react";

type WebviewShellProps = {
  statusStrip: ReactNode;
  chartPanel: ReactNode;
  planPanel: ReactNode;
  watchlistPanel?: ReactNode;
  actionsDock?: ReactNode;
  mobileSheet?: ReactNode;
  className?: string;
  theme?: "dark" | "light";
  collapsed?: boolean;
};

export default function WebviewShell({
  statusStrip,
  chartPanel,
  planPanel,
  watchlistPanel,
  actionsDock,
  mobileSheet,
  className,
  theme = "dark",
  collapsed = false,
}: WebviewShellProps) {
  const isLight = theme === "light";
  const surfaceMain = isLight
    ? "border-slate-200 bg-white/90 shadow-slate-200/40"
    : "border-neutral-800/70 bg-neutral-950/50 shadow-emerald-500/10";
  const surfaceAside = isLight
    ? "border-slate-200 bg-white/85 shadow-slate-200/30"
    : "border-neutral-800/70 bg-neutral-950/40 shadow-emerald-500/5";
  const hasWatchlist = Boolean(watchlistPanel);

  return (
    <div
      className={clsx(
        "relative min-h-screen transition-colors duration-300",
        isLight ? "bg-slate-50 text-slate-900" : "bg-[#050709] text-neutral-100",
        className,
      )}
    >
      <div
        className={clsx(
          "pointer-events-none fixed inset-0 bg-gradient-to-b via-transparent to-transparent",
          isLight ? "from-emerald-400/10" : "from-emerald-500/8",
        )}
        aria-hidden
      />
      <header
        className={clsx(
          "sticky top-0 z-30 backdrop-blur-md",
          isLight ? "border-b border-slate-200/80 bg-white/80" : "border-b border-neutral-900/70 bg-[#050709]/70",
        )}
      >
        {statusStrip}
      </header>
      <main
        className={clsx(
          "relative z-10 mx-auto flex w-full max-w-[1400px] flex-1 flex-col gap-6 px-4 pb-24 pt-6 sm:px-6 lg:px-10",
          collapsed && "pt-3 sm:pt-4",
        )}
        data-collapsed={collapsed ? "true" : "false"}
      >
        <div
          className={clsx(
            "grid gap-6",
            hasWatchlist
              ? "lg:grid-cols-[minmax(0,1.05fr),minmax(0,1.9fr),minmax(0,1fr)]"
              : "lg:grid-cols-[minmax(0,1.9fr),minmax(0,1fr)]",
          )}
        >
          {hasWatchlist ? (
            <div className="hidden lg:block">
              {watchlistPanel}
            </div>
          ) : null}
          <section
            className={clsx(
              "min-h-[420px] rounded-3xl border p-5 backdrop-blur transition-colors duration-300",
              surfaceMain,
            )}
          >
            {chartPanel}
          </section>
          <aside
            className={clsx(
              "hidden rounded-3xl border p-5 backdrop-blur transition-colors duration-300 lg:block",
              surfaceAside,
            )}
          >
            {planPanel}
          </aside>
        </div>
      </main>
      {actionsDock ? <div className="fixed bottom-6 right-6 z-30 hidden md:block">{actionsDock}</div> : null}
      {mobileSheet}
    </div>
  );
}
