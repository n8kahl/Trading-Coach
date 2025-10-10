import { useState } from "react";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { TIdeaSnapshot } from "@/lib/types";
import { safeJsonStringify } from "@/lib/utils";

type ProvenanceProps = {
  idea: TIdeaSnapshot;
};

export default function Provenance({ idea }: ProvenanceProps) {
  const [showRaw, setShowRaw] = useState(false);

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between">
        <CardTitle className="text-base">Provenance</CardTitle>
        <Button variant="outline" size="sm" onClick={() => setShowRaw((prev) => !prev)}>
          {showRaw ? "Hide raw JSON" : "View raw JSON"}
        </Button>
      </CardHeader>
      <CardContent className="space-y-4 text-sm text-muted-foreground">
        <div className="space-y-1">
          <p>
            <span className="font-semibold text-foreground">Plan ID: </span>
            {idea.plan.plan_id}
          </p>
          <p>
            <span className="font-semibold text-foreground">Version: </span>
            v{idea.plan.version}
          </p>
          <p>
            <span className="font-semibold text-foreground">Style: </span>
            {idea.plan.style}
          </p>
        </div>
        {idea.plan.warnings && idea.plan.warnings.length > 0 && (
          <div className="rounded-md border border-warning/40 bg-warning/10 px-3 py-2 text-xs text-amber-700">
            <span className="font-semibold">Server warnings:</span> {idea.plan.warnings.join("; ")}
          </div>
        )}
        {showRaw && (
          <pre className="max-h-[420px] overflow-auto rounded-lg border border-border/60 bg-muted/40 p-4 text-xs leading-relaxed text-muted-foreground">
            {safeJsonStringify(idea, 2)}
          </pre>
        )}
      </CardContent>
    </Card>
  );
}
