"use client";

import { useEffect, useMemo, useState } from "react";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";
import { API_BASE } from "@/lib/api";
import type { TIdeaSnapshot } from "@/lib/types";
import { formatDateTime, formatPrice } from "@/lib/format";

type LiveCoachingProps = {
  idea: TIdeaSnapshot;
};

type StreamMessage = {
  symbol?: string;
  event?: CoachingEvent;
};

type CoachingEvent = {
  state?: string;
  price?: number;
  time?: string;
  coaching?: string;
  type?: string;
};

export default function LiveCoaching({ idea }: LiveCoachingProps) {
  const symbol = idea.plan.symbol;
  const [events, setEvents] = useState<CoachingEvent[]>([]);
  const [status, setStatus] = useState<"idle" | "connecting" | "connected" | "error">("idle");
  const [paused, setPaused] = useState(false);

  useEffect(() => {
    if (!symbol || typeof window === "undefined" || paused) return;
    setStatus("connecting");
    const url = `${API_BASE}/stream/market?symbol=${encodeURIComponent(symbol)}`;
    const source = new EventSource(url);

    source.onopen = () => setStatus("connected");
    source.onmessage = (message) => {
      try {
        const parsed: StreamMessage = JSON.parse(message.data || "{}");
        const payload = parsed.event ?? (parsed as CoachingEvent);
        if (!payload) return;
        setEvents((prev) => {
          const next = [payload, ...prev];
          return next.slice(0, 50);
        });
      } catch (err) {
        console.warn("[LiveCoaching] parse error", err);
      }
    };

    source.onerror = () => {
      setStatus("error");
      source.close();
    };

    return () => {
      source.close();
      setStatus("idle");
    };
  }, [symbol, paused]);

  const statusLabel = useMemo(() => {
    switch (status) {
      case "connected":
        return "Live feed active";
      case "connecting":
        return "Connecting…";
      case "error":
        return "Connection lost";
      default:
        return paused ? "Paused" : "Idle";
    }
  }, [status, paused]);

  return (
    <Card className="flex h-full flex-col">
      <CardHeader className="flex flex-row items-center justify-between gap-3">
        <CardTitle className="text-base">Live Coaching</CardTitle>
        <div className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
          <span>{statusLabel}</span>
          <div className="flex items-center gap-1">
            <Button variant="ghost" size="sm" onClick={() => setPaused((prev) => !prev)}>
              {paused ? "Resume" : "Pause"}
            </Button>
            <Button variant="ghost" size="sm" onClick={() => setEvents([])}>
              Clear
            </Button>
          </div>
        </div>
      </CardHeader>
      <CardContent className="flex flex-1 flex-col gap-3">
        <ScrollArea className="h-[360px] rounded-md border border-border/60 bg-muted/30 p-3">
          {events.length === 0 ? (
            <p className="text-sm text-muted-foreground">
              Awaiting stream updates for <span className="font-semibold text-foreground">{symbol}</span>. Feed will populate once real-time events arrive.
            </p>
          ) : (
            <ul className="space-y-2">
              {events.map((event, idx) => (
                <li key={`${event.time}-${idx}`} className="rounded-md border border-border/50 bg-background/80 px-3 py-2 text-sm">
                  <div className="flex items-center justify-between gap-3">
                    <span className="font-semibold capitalize text-foreground">{event.state ?? event.type ?? "Update"}</span>
                    <span className="text-xs text-muted-foreground">{formatDateTime(event.time ?? new Date().toISOString())}</span>
                  </div>
                  <div className="mt-1 text-sm text-muted-foreground">
                    {event.coaching ?? "Monitoring…"}
                    {Number.isFinite(event.price) && (
                      <span className="ml-2 font-semibold text-foreground">Price {formatPrice(event.price ?? null, idea.plan.decimals)}</span>
                    )}
                  </div>
                </li>
              ))}
            </ul>
          )}
        </ScrollArea>
        <p className="text-xs text-muted-foreground">
          Streaming endpoint: <code className="rounded bg-muted px-1 py-0.5">{symbol}</code> · Events expire after 50 messages.
        </p>
      </CardContent>
    </Card>
  );
}
