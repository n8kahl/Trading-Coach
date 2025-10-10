import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { TIdeaSnapshot } from "@/lib/types";
import { cn } from "@/lib/utils";

type ConfluencePanelProps = {
  idea: TIdeaSnapshot;
};

export default function ConfluencePanel({ idea }: ConfluencePanelProps) {
  const summary = idea.summary;
  const vol = idea.volatility_regime;
  const snapped = idea.htf?.snapped_targets ?? [];
  const plan = idea.plan;
  const offlineBasis = idea.offline_basis;

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Confluence Metrics</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <section className="space-y-2">
          <h3 className="text-sm font-semibold text-muted-foreground">MTF Alignment</h3>
          <div className="flex flex-wrap items-center gap-2">
            {(summary?.frames_used ?? []).map((frame) => (
              <Badge key={frame} variant="subtle" className="bg-muted px-2.5 py-1 text-xs uppercase tracking-wide">
                {frame}
              </Badge>
            ))}
          </div>
          <div className="text-sm text-muted-foreground">
            Confluence score{" "}
            <span className="font-semibold text-foreground">{summary?.confluence_score?.toFixed(2) ?? "—"}</span>
          </div>
          {summary?.trend_notes && (
            <ul className="ml-4 list-disc text-sm text-muted-foreground">
              {Object.entries(summary.trend_notes).map(([frame, note]) => (
                <li key={frame}>
                  <span className="font-medium text-foreground">{frame}:</span> {note}
                </li>
              ))}
            </ul>
          )}
        </section>

        <section className="space-y-2">
          <h3 className="text-sm font-semibold text-muted-foreground">Volatility Regime</h3>
          {vol ? (
            <div className="flex flex-wrap items-center gap-2 text-sm">
              <Badge
                variant={
                  vol.regime_label === "elevated"
                    ? "warning"
                    : vol.regime_label === "extreme"
                      ? "destructive"
                      : vol.regime_label === "low"
                        ? "success"
                        : "secondary"
                }
                className="px-2.5 py-1 uppercase"
              >
                {vol.regime_label ?? "Unknown"}
              </Badge>
              <span>IV Rank: {vol.iv_rank ?? "—"}</span>
              <span>IV Percentile: {vol.iv_percentile ?? "—"}</span>
              <span>HV20: {vol.hv_20 ?? "—"}</span>
              <span>HV60: {vol.hv_60 ?? "—"}</span>
            </div>
          ) : (
            <p className="text-sm text-muted-foreground">Volatility data unavailable.</p>
          )}
        </section>

        <section className="space-y-2">
          <h3 className="text-sm font-semibold text-muted-foreground">Structure Anchors</h3>
          {snapped.length > 0 ? (
            <div className="flex flex-wrap gap-2">
              {snapped.map((tag) => (
                <Badge key={tag} variant="outline" className="px-2.5 py-1 text-xs uppercase">
                  {tag}
                </Badge>
              ))}
            </div>
          ) : (
            <p className="text-sm text-muted-foreground">No snapped targets available.</p>
          )}
        </section>

        <section className="space-y-2">
          <h3 className="text-sm font-semibold text-muted-foreground">Math</h3>
          <div className="grid gap-3 sm:grid-cols-2">
            <div className="rounded-md border border-border/60 bg-muted/40 px-3 py-2">
              <p className="text-xs text-muted-foreground">ATR</p>
              <p className="text-sm font-semibold text-foreground">
                {idea.calc_notes?.atr14 ? idea.calc_notes.atr14.toFixed(2) : "—"}
              </p>
            </div>
            <div className="rounded-md border border-border/60 bg-muted/40 px-3 py-2">
              <p className="text-xs text-muted-foreground">Stop Multiple</p>
              <p className="text-sm font-semibold text-foreground">
                {idea.calc_notes?.stop_multiple ? idea.calc_notes.stop_multiple.toFixed(2) : "—"}
              </p>
            </div>
          </div>
          {plan.warnings && plan.warnings.length > 0 && (
            <ul className="ml-4 list-disc text-xs text-destructive">
              {plan.warnings.map((warning) => (
                <li key={warning}>{warning}</li>
              ))}
            </ul>
          )}
          {offlineBasis && (
            <div className="rounded-md border border-primary/30 bg-primary/10 px-3 py-2 text-xs text-primary">
              <p className="font-semibold uppercase tracking-wide">Offline basis</p>
              <p>HTF snapshot: {offlineBasis.htf_snapshot_time ?? "—"}</p>
              <p>Vol regime: {offlineBasis.volatility_regime ?? "—"}</p>
              <p>Expected move horizon: {offlineBasis.expected_move_days ?? "—"} days</p>
            </div>
          )}
        </section>
      </CardContent>
    </Card>
  );
}
