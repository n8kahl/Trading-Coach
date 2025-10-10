import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { buildLevelsCopy, formatPrice, formatRR } from "@/lib/format";
import type { TIdeaSnapshot } from "@/lib/types";
import { copyToClipboard } from "@/lib/copy";
import { cn } from "@/lib/utils";

type QuickPlanProps = {
  idea: TIdeaSnapshot;
};

export default function QuickPlan({ idea }: QuickPlanProps) {
  const plan = idea.plan;
  const decimals = plan.decimals;

  const tp1 = plan.targets[0] ?? null;
  const tp2 = plan.targets[1] ?? null;

  const isLong = plan.bias === "long";
  const geometryInvalid =
    (isLong && (plan.stop >= plan.entry || (tp1 !== null && tp1 <= plan.entry))) ||
    (!isLong && (plan.stop <= plan.entry || (tp1 !== null && tp1 >= plan.entry)));

  const copy = () => copyToClipboard(buildLevelsCopy(plan), "Plan copied");

  return (
    <Card>
      <CardContent className="flex flex-col gap-6 p-6">
        {geometryInvalid && (
          <div className="rounded-md border border-destructive/40 bg-destructive/10 px-4 py-3 text-sm text-destructive">
            Sanity check failed: target/stop ordering does not match {plan.bias} geometry. Review before executing.
          </div>
        )}
        <div className="grid gap-4 md:grid-cols-4">
          <PlanMetric label="Entry" value={formatPrice(plan.entry, decimals)} accent="text-foreground" />
          <PlanMetric label="Stop" value={formatPrice(plan.stop, decimals)} accent="text-destructive" />
          <PlanMetric label="TP1" value={tp1 !== null ? formatPrice(tp1, decimals) : "—"} accent="text-success" />
          <PlanMetric label="TP2" value={tp2 !== null ? formatPrice(tp2, decimals) : "—"} accent="text-success" />
        </div>
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <span className="font-medium text-foreground">R:R to TP1</span>
            <span>{formatRR(plan.rr_to_t1)}</span>
          </div>
          <Button variant="outline" size="sm" onClick={copy}>
            Copy Plan
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}

type PlanMetricProps = {
  label: string;
  value: string;
  accent?: string;
};

function PlanMetric({ label, value, accent }: PlanMetricProps) {
  return (
    <div className="rounded-lg border border-border/60 bg-muted/40 px-4 py-3">
      <p className="text-xs uppercase tracking-wide text-muted-foreground">{label}</p>
      <p className={cn("mt-1 text-lg font-semibold text-foreground", accent)}>{value}</p>
    </div>
  );
}
