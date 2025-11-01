"use client";

import clsx from "clsx";
import { useMemo } from "react";
import { useStore } from "@/store/useStore";
import type { PlanLayers, PlanSnapshot } from "@/lib/types";

type CoachPanelProps = {
  plan: PlanSnapshot["plan"];
  layers: PlanLayers | null;
};

type ObjectiveMeta = {
  state?: string;
  why?: unknown;
  progress?: number;
};

type ProbabilityChip = {
  key: string;
  label: string;
  probability: number;
};

function clamp(value: number, min = 0, max = 1): number {
  if (Number.isNaN(value)) return min;
  return Math.min(max, Math.max(min, value));
}

function formatPercent(value: number | null | undefined, precision = 0): string {
  if (value == null || Number.isNaN(value)) return "—";
  const pct = value * 100;
  return `${pct.toFixed(precision)}%`;
}

function formatRelativeTime(iso: string | null): string {
  if (!iso) return "—";
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return "—";
  const diff = Date.now() - date.getTime();
  if (diff < 1_000) return "just now";
  const minutes = Math.floor(diff / 60_000);
  if (minutes < 1) {
    const seconds = Math.floor(diff / 1_000);
    return `${seconds}s ago`;
  }
  if (minutes < 60) {
    return `${minutes}m ago`;
  }
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  return `${days}d ago`;
}

function resolveObjectiveMeta(layers: PlanLayers | null): ObjectiveMeta | null {
  const meta = layers?.meta;
  if (!meta || typeof meta !== "object") return null;
  const objective = (meta as Record<string, unknown>).next_objective;
  if (!objective || typeof objective !== "object") return null;
  return objective as ObjectiveMeta;
}

function resolveHtfProximity(plan: PlanSnapshot["plan"]): string | null {
  const planRecord = plan as Record<string, unknown>;
  const candidates: Array<unknown> = [];
  const htfBlock = planRecord.htf;
  if (htfBlock && typeof htfBlock === "object") {
    const htf = htfBlock as Record<string, unknown>;
    candidates.push(htf.proximity_label, htf.proximity, htf.bias, htf.context);
  }
  const summary = planRecord.summary;
  if (summary && typeof summary === "object") {
    const block = summary as Record<string, unknown>;
    candidates.push(block.htf_proximity, block.htf_bias);
  }
  candidates.push(planRecord.htf_proximity, planRecord.htf_bias);
  const token = candidates.find((value) => typeof value === "string" && value.trim().length > 0) as string | undefined;
  return token ? token.trim() : null;
}

function normaliseWhyTokens(meta: ObjectiveMeta | null): string[] {
  if (!meta) return [];
  const raw = meta.why;
  if (!raw) return [];
  if (Array.isArray(raw)) {
    return raw.filter((value): value is string => typeof value === "string" && value.trim().length > 0);
  }
  if (typeof raw === "string" && raw.trim()) {
    return raw.split(/[,;]+/).map((entry) => entry.trim()).filter(Boolean);
  }
  return [];
}

function buildProbabilityChips(plan: PlanSnapshot["plan"]): ProbabilityChip[] {
  const meta = Array.isArray(plan.target_meta) ? (plan.target_meta as Array<Record<string, unknown>>) : [];
  if (!meta.length) return [];
  return meta
    .slice(0, 2)
    .map((entry, index) => {
      const label = typeof entry.label === "string" && entry.label.trim() ? entry.label.trim() : `TP${index + 1}`;
      const probability =
        typeof entry.prob_touch === "number"
          ? entry.prob_touch
          : typeof entry.prob_touch_calibrated === "number"
            ? entry.prob_touch_calibrated
            : typeof entry.prob_touch_raw === "number"
              ? entry.prob_touch_raw
              : null;
      return probability != null
        ? {
            key: `tp-${index}`,
            label,
            probability: clamp(probability, 0, 1),
          }
        : null;
    })
    .filter((entry): entry is ProbabilityChip => entry !== null);
}

function renderChip(label: string, value: string, tone: "emerald" | "amber" | "rose" | "neutral" = "neutral") {
  const toneClasses: Record<typeof tone, string> = {
    emerald: "border-[color:var(--tc-chip-emerald-border)] bg-[color:var(--tc-chip-emerald-surface)] text-[color:var(--tc-emerald-200)]",
    amber: "border-[color:var(--tc-chip-amber-border)] bg-[color:var(--tc-chip-amber-surface)] text-[color:var(--tc-amber-200)]",
    rose: "border-[color:var(--tc-chip-red-border)] bg-[color:var(--tc-chip-red-surface)] text-[color:var(--tc-red-300)]",
    neutral: "border-[color:var(--tc-chip-neutral-border)] bg-[color:var(--tc-chip-neutral-surface)] text-[color:var(--tc-neutral-200)]",
  };
  return (
    <span
      key={`${label}-${value}`}
      className={clsx(
        "inline-flex items-center gap-1 rounded-full border px-3 py-1 text-[0.65rem] font-semibold uppercase tracking-[0.2em]",
        toneClasses[tone],
      )}
    >
      <span>{label}</span>
      <span>{value}</span>
    </span>
  );
}

function resolveNextAction(diff: ReturnType<typeof useStore.getState>["coach"]["diff"]): { action: string; waiting?: string | null } {
  if (!diff) {
    return { action: "Awaiting guidance…" };
  }
  const action = diff.next_action && diff.next_action.trim().length > 0 ? diff.next_action.trim() : diff.waiting_for ?? "Awaiting guidance…";
  return {
    action,
    waiting: diff.waiting_for ?? null,
  };
}

export function CoachPanel({ plan, layers }: CoachPanelProps) {
  const coach = useStore((state) => state.coach);
  const session = useStore((state) => state.session);

  const objectiveMeta = useMemo(() => resolveObjectiveMeta(layers), [layers]);
  const nextObjectiveProgress = clamp(
    (coach.diff?.objective_progress?.progress ?? objectiveMeta?.progress ?? 0) || 0,
    0,
    1,
  );

  const mtfDelta = coach.diff?.confluence_delta?.mtf ?? 0;
  const vwapSide = coach.diff?.confluence_delta?.vwap_side ?? null;
  const htfProximity = useMemo(() => resolveHtfProximity(plan), [plan]);
  const whyTokens = useMemo(() => normaliseWhyTokens(objectiveMeta), [objectiveMeta]);
  const probabilityChips = useMemo(() => buildProbabilityChips(plan), [plan]);
  const { action: nextAction, waiting } = useMemo(() => resolveNextAction(coach.diff), [coach.diff]);

  const entryDistancePct = coach.diff?.objective_progress?.entry_distance_pct ?? null;
  const showNoTrade = nextObjectiveProgress < 0.2 && mtfDelta <= -2;

  const timeline = useMemo(() => {
    const items = coach.timeline.slice(-6);
    return [...items].reverse();
  }, [coach.timeline]);

  return (
    <section
      className="space-y-4 rounded-3xl border border-[color:var(--tc-border-subtle)] bg-[color:var(--tc-surface-muted)]/90 p-5 text-[color:var(--tc-neutral-100)] backdrop-blur"
      data-testid="coach-panel"
    >
      {showNoTrade ? (
        <div
          className="rounded-2xl border border-[color:var(--tc-chip-red-border)] bg-[color:var(--tc-chip-red-surface)] px-4 py-3 text-sm font-semibold text-[color:var(--tc-red-300)]"
          data-testid="coach-no-trade-banner"
        >
          No-trade zone — MTF momentum unfavorable ({mtfDelta >= 0 ? `+${mtfDelta}` : mtfDelta}) and progress {formatPercent(nextObjectiveProgress, 0)}
        </div>
      ) : null}

      <header className="space-y-2">
        <div className="text-[0.65rem] uppercase tracking-[0.3em] text-[color:var(--tc-neutral-400)]">Next Action</div>
        <h3 className="text-xl font-semibold text-[color:var(--tc-neutral-50)]">{nextAction}</h3>
        {waiting ? (
          <p className="text-sm text-[color:var(--tc-neutral-300)]">
            Waiting for <span className="font-semibold text-[color:var(--tc-neutral-100)]">{waiting}</span>
          </p>
        ) : null}
        <div className="text-xs text-[color:var(--tc-neutral-400)]">
          Session · {session?.status ?? "—"} {session?.as_of ? `· ${new Date(session.as_of).toLocaleTimeString([], { hour: "numeric", minute: "2-digit" })}` : ""}
        </div>
      </header>

      <section className="space-y-2">
        <div className="text-[0.65rem] uppercase tracking-[0.28em] text-[color:var(--tc-neutral-400)]">Why</div>
        <div className="flex flex-wrap items-center gap-2">
          {renderChip("MTF", mtfDelta > 0 ? `↑ ${mtfDelta}` : mtfDelta < 0 ? `↓ ${Math.abs(mtfDelta)}` : "Flat", mtfDelta > 0 ? "emerald" : mtfDelta < 0 ? "rose" : "neutral")}
          {vwapSide ? renderChip("VWAP", vwapSide === "above" ? "Above" : "Below", vwapSide === "above" ? "emerald" : "rose") : null}
          {htfProximity ? renderChip("HTF", htfProximity, "neutral") : null}
          {whyTokens.map((token) => renderChip("Confluence", token, "neutral"))}
        </div>
      </section>

      {coach.diff?.risk_cue ? (
        <section className="space-y-2">
          <div className="text-[0.65rem] uppercase tracking-[0.28em] text-[color:var(--tc-neutral-400)]">Risk</div>
          <div className="inline-flex items-center rounded-full border border-[color:var(--tc-chip-amber-border)] bg-[color:var(--tc-chip-amber-surface)] px-3 py-1 text-[0.68rem] font-semibold uppercase tracking-[0.22em] text-[color:var(--tc-amber-200)]">
            {coach.diff.risk_cue.replace(/[_-]/g, " ")}
          </div>
        </section>
      ) : null}

      <section className="grid gap-3 md:grid-cols-2">
        <div className="rounded-2xl border border-[color:var(--tc-border-subtle)] bg-[color:var(--tc-surface-primary)]/80 px-4 py-3">
          <div className="text-[0.6rem] uppercase tracking-[0.25em] text-[color:var(--tc-neutral-400)]">Objective Progress</div>
          <div className="mt-2 h-2 w-full rounded-full bg-[color:var(--tc-neutral-900)]/80">
            <div
              className={clsx(
                "h-full rounded-full transition-[width]",
                nextObjectiveProgress >= 0.66
                  ? "bg-[color:var(--tc-emerald-400)]"
                  : nextObjectiveProgress >= 0.33
                    ? "bg-[color:var(--tc-amber-400)]"
                    : "bg-[color:var(--tc-neutral-400)]",
              )}
              style={{ width: `${Math.round(nextObjectiveProgress * 100)}%` }}
            />
          </div>
          <div className="mt-2 flex items-center justify-between text-xs text-[color:var(--tc-neutral-300)]">
            <span>{formatPercent(nextObjectiveProgress, 0)}</span>
            <span>Δ {formatPercent(entryDistancePct ?? null, 1)}</span>
          </div>
        </div>
        {probabilityChips.length ? (
          <div className="rounded-2xl border border-[color:var(--tc-border-subtle)] bg-[color:var(--tc-surface-primary)]/80 px-4 py-3">
            <div className="text-[0.6rem] uppercase tracking-[0.25em] text-[color:var(--tc-neutral-400)]">Probability Ladder</div>
            <div className="mt-2 flex flex-wrap items-center gap-2">
              {probabilityChips.map((chip) =>
                renderChip(chip.label, formatPercent(chip.probability, 0), chip.probability >= 0.65 ? "emerald" : chip.probability >= 0.45 ? "amber" : "neutral"),
              )}
            </div>
          </div>
        ) : null}
      </section>

      {timeline.length ? (
        <section className="space-y-2">
          <div className="text-[0.65rem] uppercase tracking-[0.28em] text-[color:var(--tc-neutral-400)]">Coach Timeline</div>
          <ol className="space-y-2 text-sm text-[color:var(--tc-neutral-300)]" data-testid="coach-timeline">
            {timeline.map((entry) => {
              const text =
                entry.diff.next_action && entry.diff.next_action.trim()
                  ? entry.diff.next_action.trim()
                  : entry.diff.waiting_for ?? "Pulse received";
              const relative = formatRelativeTime(entry.ts ?? null);
              return (
                <li
                  key={entry.ts}
                  className="flex items-start justify-between rounded-2xl border border-[color:var(--tc-border-subtle)]/60 bg-[color:var(--tc-surface-primary)]/80 px-3 py-2"
                  data-testid="coach-timeline-entry"
                >
                  <span className="pr-3 text-[0.82rem] text-[color:var(--tc-neutral-100)]">{text}</span>
                  <span className="text-[0.65rem] uppercase tracking-[0.24em] text-[color:var(--tc-neutral-500)]">{relative}</span>
                </li>
              );
            })}
          </ol>
        </section>
      ) : null}
    </section>
  );
}

export default CoachPanel;
