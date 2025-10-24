import type { ChartParams } from "./types";

export type SupportingLevel = {
  price: number;
  label: string;
};

export type ParsedUiState = {
  session: "live" | "premkt" | "after";
  confidence: number;
  style: "scalp" | "intraday" | "swing" | "unknown";
};

const SESSION_TOKENS = new Set(["live", "premkt", "after"]);
const STYLE_TOKENS = new Set(["scalp", "intraday", "swing"]);

export function parsePrice(value: unknown): number | null {
  if (value == null) return null;
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "string") {
    const numeric = Number.parseFloat(value);
    if (Number.isFinite(numeric)) return numeric;
  }
  return null;
}

export function parseTargets(value: unknown): number[] {
  if (!value) return [];
  if (Array.isArray(value)) {
    return value.map(parsePrice).filter((item): item is number => item != null);
  }
  if (typeof value === "string") {
    return value
      .split(",")
      .map((token) => parsePrice(token.trim()))
      .filter((item): item is number => item != null);
  }
  return [];
}

export function parseSupportingLevels(token: string | undefined | null): SupportingLevel[] {
  if (!token || typeof token !== "string") return [];
  return token
    .split(";")
    .map((entry) => entry.trim())
    .filter(Boolean)
    .map((entry) => {
      const [priceToken, ...labelParts] = entry.split("|");
      const price = parsePrice(priceToken);
      const label = labelParts.join("|").trim() || "Level";
      if (price == null) return null;
      return { price, label };
    })
    .filter((item): item is SupportingLevel => item !== null);
}

export function parseUiState(raw: string | undefined | null): ParsedUiState {
  if (!raw || typeof raw !== "string") {
    return { session: "live", confidence: 0, style: "unknown" };
  }
  try {
    const parsed = JSON.parse(raw) as Record<string, unknown>;
    const sessionRaw = typeof parsed.session === "string" ? parsed.session.toLowerCase() : "";
    const session = SESSION_TOKENS.has(sessionRaw) ? (sessionRaw as ParsedUiState["session"]) : "live";
    let confidence = parsePrice(parsed.confidence) ?? 0;
    if (confidence > 1) confidence = Math.min(confidence / 100, 1);
    if (confidence < 0) confidence = 0;
    const styleRaw = typeof parsed.style === "string" ? parsed.style.toLowerCase() : "";
    const style = STYLE_TOKENS.has(styleRaw) ? (styleRaw as ParsedUiState["style"]) : "unknown";
    return { session, confidence, style };
  } catch (error) {
    if (process.env.NODE_ENV !== "production") {
      console.warn("Failed to parse chart ui_state", error);
    }
    return { session: "live", confidence: 0, style: "unknown" };
  }
}

export function normalizeChartParams(params?: ChartParams | null): Record<string, string> {
  if (!params) return {};
  const entries: [string, string][] = [];
  for (const [key, value] of Object.entries(params)) {
    if (value == null) continue;
    entries.push([key, String(value)]);
  }
  return Object.fromEntries(entries);
}
