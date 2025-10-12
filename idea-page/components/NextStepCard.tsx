"use client";

import { useMemo } from "react";

import { PlanStreamState } from "@/hooks/usePlanStream";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";

type NextStepCardProps = {
  planState: PlanStreamState;
  onKeepPlan: () => void;
  onApplyUpdate: () => void;
  disabled?: boolean;
};

const STATUS_META: Record<
  PlanStreamState["status"],
  { label: string; badgeClass: string; descriptionClass: string }
> = {
  intact: {
    label: "Plan Intact",
    badgeClass: "bg-emerald-500/10 text-emerald-400 border border-emerald-500/30",
    descriptionClass: "text-emerald-100/90",
  },
  at_risk: {
    label: "Plan At Risk",
    badgeClass: "bg-amber-500/10 text-amber-300 border border-amber-400/40",
    descriptionClass: "text-amber-100/90",
  },
  invalidated: {
    label: "Plan Invalidated",
    badgeClass: "bg-red-500/10 text-red-300 border border-red-500/40",
    descriptionClass: "text-red-100/90",
  },
  reversal: {
    label: "Reversal Opportunity",
    badgeClass: "bg-sky-500/10 text-sky-300 border border-sky-500/40",
    descriptionClass: "text-sky-100/90",
  },
};

function formatTimestamp(timestamp: string | null): string {
  if (!timestamp) return "";
  try {
    const dt = new Date(timestamp);
    if (Number.isNaN(dt.getTime())) return "";
    return dt.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit", second: "2-digit" });
  } catch {
    return "";
  }
}

function formatRR(value: number | null): string {
  if (typeof value !== "number" || Number.isNaN(value)) {
    return "—";
  }
  return value.toFixed(2);
}

export default function NextStepCard({ planState, onKeepPlan, onApplyUpdate, disabled }: NextStepCardProps) {
  const statusMeta = STATUS_META[planState.status] ?? STATUS_META.intact;
  const lastUpdatedCopy = useMemo(() => formatTimestamp(planState.timestamp), [planState.timestamp]);

  return (
    <Card className="border border-primary/10 bg-primary/5">
      <CardContent className="flex flex-col gap-4 p-4 sm:p-5">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <span className={`inline-flex items-center rounded-full px-3 py-1 text-xs font-medium ${statusMeta.badgeClass}`}>
            {statusMeta.label}
          </span>
          {lastUpdatedCopy ? <span className="text-xs text-muted-foreground">Updated {lastUpdatedCopy}</span> : null}
        </div>
        {planState.note ? (
          <p className={`text-sm leading-relaxed ${statusMeta.descriptionClass}`}>{planState.note}</p>
        ) : (
          <p className="text-sm text-muted-foreground">Awaiting live plan guidance.</p>
        )}
        <div className="flex flex-wrap items-center justify-between gap-3 text-sm text-muted-foreground">
          <span>
            <strong className="text-foreground">R:R (TP1):</strong> {formatRR(planState.rrToT1)}
          </span>
          {planState.lastPrice !== null ? (
            <span>
              <strong className="text-foreground">Last:</strong> {planState.lastPrice.toFixed(2)}
            </span>
          ) : null}
        </div>
        <div className="flex flex-wrap justify-end gap-2">
          <Button variant="ghost" size="sm" disabled={disabled} onClick={onKeepPlan}>
            Keep Plan
          </Button>
          <Button size="sm" disabled={disabled} onClick={onApplyUpdate}>
            Apply Update
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}
