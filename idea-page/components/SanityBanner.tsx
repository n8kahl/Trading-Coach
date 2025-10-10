import { AlertTriangle } from "lucide-react";

import type { PlanCore } from "@/lib/types";

type SanityBannerProps = {
  plan: PlanCore;
};

export default function SanityBanner({ plan }: SanityBannerProps) {
  const tp1 = plan.targets[0] ?? null;
  const tp2 = plan.targets[1] ?? null;
  const isLong = plan.bias === "long";

  const geometryInvalid =
    (isLong && (plan.stop >= plan.entry || (tp1 !== null && tp1 <= plan.entry) || (tp2 !== null && tp2 <= plan.entry))) ||
    (!isLong && (plan.stop <= plan.entry || (tp1 !== null && tp1 >= plan.entry) || (tp2 !== null && tp2 >= plan.entry)));

  if (!geometryInvalid) return null;

  return (
    <div className="rounded-lg border border-destructive/40 bg-destructive/15 px-4 py-3 text-sm text-destructive">
      <div className="flex items-center gap-2">
        <AlertTriangle className="h-4 w-4" />
        <span>Sanity warning: Target or stop ordering violates {plan.bias} geometry. Validate before using this plan.</span>
      </div>
    </div>
  );
}
