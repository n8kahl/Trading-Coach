import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { TIdeaSnapshot } from "@/lib/types";

type PlaybookProps = {
  idea: TIdeaSnapshot;
};

export default function Playbook({ idea }: PlaybookProps) {
  const why = idea.why_this_works ?? [];
  const invalidation = idea.invalidation ?? [];
  const riskNote = idea.risk_note;

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Playbook</CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        <section>
          <h3 className="text-sm font-semibold text-muted-foreground">Why this works</h3>
          <ul className="mt-2 space-y-2 text-sm text-muted-foreground">
            {why.map((item) => (
              <li key={item} className="flex items-start gap-2">
                <span className="mt-1.5 h-1.5 w-1.5 rounded-full bg-emerald-500" />
                <span>{item}</span>
              </li>
            ))}
            {why.length === 0 && <li className="text-xs text-muted-foreground/70">No rationale provided.</li>}
          </ul>
        </section>
        <section>
          <h3 className="text-sm font-semibold text-muted-foreground">Invalidation & If-Then</h3>
          <ul className="mt-2 space-y-2 text-sm text-muted-foreground">
            {invalidation.map((item) => (
              <li key={item} className="flex items-start gap-2">
                <span className="mt-1.5 h-1.5 w-1.5 rounded-full bg-destructive" />
                <span>{item}</span>
              </li>
            ))}
            {invalidation.length === 0 && <li className="text-xs text-muted-foreground/70">No invalidation notes recorded.</li>}
          </ul>
        </section>
        {riskNote && (
          <section className="rounded-md border border-amber-400/40 bg-amber-500/10 px-3 py-2 text-sm text-amber-700">
            <span className="font-semibold">Risk note:</span> {riskNote}
          </section>
        )}
      </CardContent>
    </Card>
  );
}
