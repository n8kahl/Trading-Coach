import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { TIdeaSnapshot } from "@/lib/types";

const EDUCATION_TOPICS: Array<{ keyword: RegExp; title: string; blurb: string; url: string }> = [
  {
    keyword: /VWAP/i,
    title: "VWAP 101",
    blurb: "Volume-weighted average price acts as a real-time mean reversion anchor. Learn how Trading Coach uses VWAP for bias.",
    url: "https://learn.tradingcoach.ai/concepts/vwap",
  },
  {
    keyword: /POC|VAH|VAL/i,
    title: "Volume Profile Levels",
    blurb: "POC, VAH, and VAL highlight liquidity magnets from prior sessions. Understand how to plan take profits around them.",
    url: "https://learn.tradingcoach.ai/concepts/volume-profile",
  },
  {
    keyword: /1\.272|Fib/i,
    title: "Fibonacci Extensions",
    blurb: "The 1.272 extension is a reliable first thrust target on continuation trades. See examples and risk management guidelines.",
    url: "https://learn.tradingcoach.ai/concepts/fibonacci",
  },
];

type EducationRailProps = {
  idea: TIdeaSnapshot;
};

export default function EducationRail({ idea }: EducationRailProps) {
  const keywords = [
    idea.plan.charts_params.notes,
    idea.why_this_works?.join(" "),
    idea.htf?.snapped_targets?.join(" "),
  ]
    .filter(Boolean)
    .join(" ");

  const cards = EDUCATION_TOPICS.filter((topic) => topic.keyword.test(keywords));

  if (cards.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Education</CardTitle>
        </CardHeader>
        <CardContent className="text-sm text-muted-foreground">
          No contextual lessons detected for this snapshot. When the playbook references VWAP, volume profile, or Fibonacci extensions you’ll see quick refreshers here.
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="grid gap-4">
      {cards.map((card) => (
        <Card key={card.title} className="border-primary/20 bg-primary/5">
          <CardHeader>
            <CardTitle className="text-sm">{card.title}</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3 text-sm text-muted-foreground">
            <p>{card.blurb}</p>
            <a
              href={card.url}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex text-sm font-semibold text-primary underline-offset-4 hover:underline"
            >
              View guide →
            </a>
          </CardContent>
        </Card>
      ))}
    </div>
  );
}
