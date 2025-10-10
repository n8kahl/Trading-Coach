"use client";

import { useMemo, useState } from "react";
import { motion } from "framer-motion";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import type { TIdeaSnapshot } from "@/lib/types";
import { computeWhatIf } from "@/lib/math";
import { cn } from "@/lib/utils";

type ContractsTableProps = {
  idea: TIdeaSnapshot;
  highlightedRows?: Set<string>;
  onRefresh?: () => void;
};

export default function ContractsTable({ idea, highlightedRows = new Set(), onRefresh }: ContractsTableProps) {
  const contracts = idea.options?.table?.slice(0, 6) ?? [];
  const [fillPrice, setFillPrice] = useState(() => (contracts[0]?.price ?? 0).toFixed(2));
  const [ivShift, setIvShift] = useState(0);

  const whatIf = useMemo(() => {
    const baseline = contracts[0];
    if (!baseline) return null;
    const result = computeWhatIf(
      {
        price: baseline.price,
        bid: baseline.bid,
        ask: baseline.ask,
        mark: baseline.mark ?? null,
        delta: baseline.delta,
        theta: baseline.theta ?? null,
        iv: baseline.iv ?? null,
      },
      { fillPrice: Number(fillPrice) || baseline.price, ivShiftBps: ivShift },
    );
    return result;
  }, [contracts, fillPrice, ivShift]);

  return (
    <Card className="h-full">
      <CardHeader className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
        <CardTitle className="text-base">Options Candidates</CardTitle>
        <div className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
          <label className="flex items-center gap-1">
            Fill Price
            <input
              type="number"
              step="0.05"
              value={fillPrice}
              onChange={(event) => setFillPrice(event.target.value)}
              className="h-8 w-20 rounded border border-border bg-background px-2"
            />
          </label>
          <label className="flex items-center gap-1">
            IV shift (bps)
            <input
              type="range"
              min={-200}
              max={200}
              value={ivShift}
              onChange={(event) => setIvShift(Number(event.target.value))}
              className="h-2 w-28"
            />
            <span className="w-12 text-right">{ivShift}</span>
          </label>
          {onRefresh && (
            <Button variant="ghost" size="sm" onClick={onRefresh}>
              Refresh
            </Button>
          )}
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {whatIf && (
          <div className="flex flex-wrap gap-3 rounded-lg border border-border/60 bg-muted/40 px-3 py-2 text-xs text-muted-foreground">
            <span className="font-medium text-foreground">What-if P/L</span>
            <span>Optimistic: {whatIf.optimistic.toFixed(2)}</span>
            <span>Neutral: {whatIf.neutral.toFixed(2)}</span>
            <span>Pessimistic: {whatIf.pessimistic.toFixed(2)}</span>
          </div>
        )}
        <div className="max-h-[360px] overflow-auto rounded-lg border border-border/70">
          <Table>
            <TableHeader>
              <TableRow className="bg-muted/60">
                <TableHead>Label</TableHead>
                <TableHead>Expiry / DTE</TableHead>
                <TableHead>Strike</TableHead>
                <TableHead>Type</TableHead>
                <TableHead>Bid</TableHead>
                <TableHead>Ask</TableHead>
                <TableHead>Mark</TableHead>
                <TableHead>Price</TableHead>
                <TableHead>Spread%</TableHead>
                <TableHead>Δ</TableHead>
                <TableHead>Θ</TableHead>
                <TableHead>IV</TableHead>
                <TableHead>OI</TableHead>
                <TableHead>Liquidity</TableHead>
                <TableHead>Last Trade</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {contracts.map((row) => {
                const changeHighlight = highlightedRows.has(row.label);
                return (
                  <TableRow key={row.label} className={cn(changeHighlight && "bg-emerald-500/10")}>
                    <TableCell className="font-medium text-foreground">
                      {changeHighlight ? <motion.span animate={{ scale: [1, 1.03, 1] }} transition={{ duration: 0.6 }}>{row.label}</motion.span> : row.label}
                    </TableCell>
                    <TableCell>
                      <div className="flex flex-col">
                        <span>{row.expiry}</span>
                        <span className="text-xs text-muted-foreground">{row.dte} DTE</span>
                      </div>
                    </TableCell>
                    <TableCell>{row.strike.toFixed(2)}</TableCell>
                    <TableCell className={cn("font-semibold", row.type === "CALL" ? "text-emerald-600" : "text-red-500")}>{row.type}</TableCell>
                    <TableCell>{row.bid.toFixed(2)}</TableCell>
                    <TableCell>{row.ask.toFixed(2)}</TableCell>
                    <TableCell>{row.mark ? row.mark.toFixed(2) : "—"}</TableCell>
                    <TableCell>{row.price.toFixed(2)}</TableCell>
                    <TableCell>{row.spread_pct.toFixed(2)}</TableCell>
                    <TableCell>{row.delta.toFixed(2)}</TableCell>
                    <TableCell>{row.theta ? row.theta.toFixed(2) : "—"}</TableCell>
                    <TableCell>{row.iv ? row.iv.toFixed(2) : "—"}</TableCell>
                    <TableCell>{row.oi ?? "—"}</TableCell>
                    <TableCell>{row.liquidity_score ?? "—"}</TableCell>
                    <TableCell>{row.last_trade_time ?? "—"}</TableCell>
                  </TableRow>
                );
              })}
              {contracts.length === 0 && (
                <TableRow>
                  <TableCell colSpan={15} className="text-center text-muted-foreground">
                    No contracts available for this plan.
                  </TableCell>
                </TableRow>
              )}
            </TableBody>
          </Table>
        </div>
      </CardContent>
    </Card>
  );
}
