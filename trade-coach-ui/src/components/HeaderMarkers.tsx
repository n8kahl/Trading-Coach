"use client";

import clsx from "clsx";
import * as React from "react";
import type { Level } from "@/lib/plan/coach";

type HeaderMarkersProps = {
  levels: Array<Level & { hidden?: boolean }>;
  onHighlight?: (levelId: string) => void;
  onToggleVisibility?: (levelId: string) => void;
  highlightedId?: string | null;
};

const formatter = new Intl.NumberFormat("en-US", {
  style: "decimal",
  minimumFractionDigits: 2,
  maximumFractionDigits: 2,
});

const LONG_PRESS_DELAY = 450;

export default function HeaderMarkers({
  levels,
  onHighlight,
  onToggleVisibility,
  highlightedId,
}: HeaderMarkersProps) {
  const pressTimerRef = React.useRef<Record<string, number | null>>({});

  const startPressTimer = React.useCallback(
    (levelId: string) => {
      if (!onToggleVisibility) return;
      clearPressTimer(levelId, pressTimerRef);
      const timer = window.setTimeout(() => {
        clearPressTimer(levelId, pressTimerRef);
        onToggleVisibility(levelId);
      }, LONG_PRESS_DELAY);
      pressTimerRef.current[levelId] = timer;
    },
    [onToggleVisibility],
  );

  const cancelPressTimer = React.useCallback((levelId: string) => {
    clearPressTimer(levelId, pressTimerRef);
  }, []);

  React.useEffect(() => {
    return () => {
      Object.values(pressTimerRef.current).forEach((id) => {
        if (id != null) {
          window.clearTimeout(id);
        }
      });
      pressTimerRef.current = {};
    };
  }, []);

  if (levels.length === 0) {
    return <p className="text-xs text-neutral-500">No trade levels published yet.</p>;
  }

  return (
    <div className="flex flex-wrap gap-2">
      {levels.map((level) => {
        const tone = pickTone(level.type);
        const isHighlighted = highlightedId === level.id;
        const label = `${level.label}: ${formatter.format(level.price)}`;
        return (
          <button
            key={level.id}
            type="button"
            className={clsx(
              "group inline-flex items-center gap-2 rounded-full border px-3 py-1 text-xs font-semibold uppercase tracking-[0.16em] transition focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-400",
              tone,
              level.hidden && "opacity-40",
              isHighlighted && "ring-2 ring-offset-1 ring-emerald-400/80 ring-offset-neutral-950",
            )}
            onClick={() => onHighlight?.(level.id)}
            onPointerDown={() => startPressTimer(level.id)}
            onPointerUp={() => cancelPressTimer(level.id)}
            onPointerLeave={() => cancelPressTimer(level.id)}
            title={level.label}
            aria-pressed={isHighlighted}
          >
            <span>{level.label}</span>
            <span className="tabular-nums text-neutral-50">{formatter.format(level.price)}</span>
          </button>
        );
      })}
    </div>
  );
}

function clearPressTimer(levelId: string, ref: React.MutableRefObject<Record<string, number | null>>) {
  const timer = ref.current[levelId];
  if (timer != null) {
    window.clearTimeout(timer);
    ref.current[levelId] = null;
  }
}

function pickTone(type: Level["type"]): string {
  switch (type) {
    case "entry":
      return "border-emerald-500/40 bg-emerald-500/10 text-emerald-100 hover:border-emerald-400/80";
    case "stop":
      return "border-rose-500/40 bg-rose-500/10 text-rose-100 hover:border-rose-400/80";
    case "tp":
      return "border-sky-500/40 bg-sky-500/10 text-sky-100 hover:border-sky-400/80";
    case "reentry":
      return "border-amber-500/40 bg-amber-500/10 text-amber-100 hover:border-amber-400/80";
    case "vwap":
    case "ema":
    default:
      return "border-neutral-700/60 bg-neutral-900/70 text-neutral-100 hover:border-emerald-400/60";
  }
}
