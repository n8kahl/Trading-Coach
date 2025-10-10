import { Card, CardContent } from "@/components/ui/card";
import { validateChartUrl } from "@/lib/validateChartUrl";

type ChartEmbedProps = {
  chartUrl: string | null | undefined;
};

export default function ChartEmbed({ chartUrl }: ChartEmbedProps) {
  const validUrl = validateChartUrl(chartUrl);

  if (!validUrl) {
    return (
      <Card className="min-h-[360px]">
        <CardContent className="flex h-full flex-col items-center justify-center gap-2 text-center text-muted-foreground">
          <p className="text-sm font-medium text-foreground">Chart unavailable</p>
          <p className="text-xs leading-relaxed">
            The interactive chart link could not be validated. Request a refreshed plan or check back later.
          </p>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className="overflow-hidden">
      <CardContent className="p-0">
        <iframe
          key={validUrl}
          title="Interactive chart"
          src={validUrl}
          className="min-h-[360px] w-full rounded-lg border-0"
          sandbox="allow-scripts allow-same-origin allow-forms allow-popups"
          loading="lazy"
        />
      </CardContent>
    </Card>
  );
}
