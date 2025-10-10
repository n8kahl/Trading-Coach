import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent } from "@/components/ui/card";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { buildLevelsCopy, buildPlanString, formatConfidence, formatDateTime, formatMarketPhase } from "@/lib/format";
import type { TIdeaSnapshot } from "@/lib/types";
import { copyToClipboard } from "@/lib/copy";
import { cn } from "@/lib/utils";

type IdeaHeaderProps = {
  idea: TIdeaSnapshot;
  isRefreshing?: boolean;
};

export default function IdeaHeader({ idea, isRefreshing }: IdeaHeaderProps) {
  const { plan, summary, htf } = idea;
  const confidence = formatConfidence(plan.confidence);
  const defaultPhase = plan.setup.includes("watch") ? "closed" : "regular";
  const marketPhase = formatMarketPhase((summary?.trend_notes?.phase as string | undefined) ?? defaultPhase);
  const timestampRaw =
    (summary as { generated_at?: string })?.generated_at ??
    (plan as { generated_at?: string })?.generated_at ??
    undefined;
  const timestamp = formatDateTime(timestampRaw ?? new Date().toISOString());
  const snappedTargets = htf?.snapped_targets ?? [];
  const planningContext = idea.planning_context ?? plan.planning_context ?? null;

  const biasVariant = plan.bias === "long" ? "success" : "destructive";

  const copyLevels = () => copyToClipboard(buildLevelsCopy(plan), "Levels copied");
  const copyPlanString = () => copyToClipboard(buildPlanString(plan, snappedTargets), "Plan string copied");

  const openBroker = () => {
    const brokerUrl = `https://www.tradingview.com/chart/?symbol=${plan.symbol}`;
    window.open(brokerUrl, "_blank", "noopener");
  };

  return (
    <Card>
      <CardContent className="flex flex-col gap-4 p-6">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div className="flex flex-col gap-2">
            <div className="flex flex-wrap items-center gap-2">
              <Badge variant="outline" className="border-foreground/20 bg-muted/40 px-2.5 py-1 text-sm font-semibold tracking-wide">
                {plan.symbol}
              </Badge>
              <Badge variant={biasVariant} className="px-2.5 py-1 text-xs uppercase tracking-wide">
                {plan.bias === "long" ? "Long Bias" : "Short Bias"}
              </Badge>
              <Badge variant="subtle" className="px-2.5 py-1 text-xs uppercase tracking-wide">
                {plan.setup}
              </Badge>
              <Badge variant="outline" className="px-2.5 py-1 text-xs uppercase tracking-wide">
                {marketPhase}
              </Badge>
              {isRefreshing && <span className="flex items-center text-xs text-muted-foreground">Refreshing…</span>}
            </div>
            <div className="flex flex-wrap items-center gap-3 text-sm text-muted-foreground">
              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger className="flex items-center gap-1 font-medium text-foreground">
                    <span>{confidence.emoji}</span>
                    <span>Confidence {confidence.label}</span>
                  </TooltipTrigger>
                  <TooltipContent>Server-calculated confidence score</TooltipContent>
                </Tooltip>
              </TooltipProvider>
              <span>•</span>
              <span>Updated {timestamp}</span>
              <span>•</span>
              <span className="uppercase tracking-wide">Style: {plan.style}</span>
            </div>
          </div>
          {planningContext === "offline" && (
            <div className="rounded-md border border-amber-400/60 bg-amber-500/15 px-3 py-2 text-xs font-medium text-amber-700">
              ⚠️ Offline Planning Mode — Market Closed; HTF &amp; Volatility data from last valid session.
            </div>
          )}
          <div className="flex flex-wrap items-center gap-2">
            <Button variant="secondary" onClick={copyLevels}>
              Copy Levels
            </Button>
            <Button variant="outline" onClick={copyPlanString}>
              Copy Plan String
            </Button>
            <Button onClick={openBroker}>Open in Broker</Button>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
