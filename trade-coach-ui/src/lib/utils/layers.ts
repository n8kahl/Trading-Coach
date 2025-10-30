import type { SupportingLevel } from "@/lib/chart";
import type { PlanLayers, PlanLayerLevel } from "@/lib/types";

function toSupportingLevel(entry: PlanLayerLevel | null | undefined): SupportingLevel | null {
  if (!entry) return null;
  const price = typeof entry.price === "number" ? entry.price : Number(entry.price);
  if (!Number.isFinite(price)) return null;
  const label = typeof entry.label === "string" && entry.label.trim() ? entry.label : entry.kind || "Level";
  return { price, label };
}

export function extractSupportingLevels(layers: PlanLayers | null | undefined): SupportingLevel[] {
  if (!layers) return [];
  const meta = layers.meta;
  const groups = meta && typeof meta === "object" ? (meta as Record<string, unknown>).level_groups : null;
  const supplemental = groups && typeof groups === "object" ? (groups as Record<string, unknown>).supplemental : null;
  if (Array.isArray(supplemental)) {
    return supplemental
      .map((item) => {
        if (!item || typeof item !== "object") return null;
        return toSupportingLevel(item as PlanLayerLevel);
      })
      .filter((item): item is SupportingLevel => item !== null);
  }
  return layers.levels.map((item) => toSupportingLevel(item)).filter((item): item is SupportingLevel => item !== null);
}

export function extractPrimaryLevels(layers: PlanLayers | null | undefined): SupportingLevel[] {
  if (!layers) return [];
  const meta = layers.meta;
  const groups = meta && typeof meta === "object" ? (meta as Record<string, unknown>).level_groups : null;
  const primary = groups && typeof groups === "object" ? (groups as Record<string, unknown>).primary : null;
  if (Array.isArray(primary)) {
    return primary
      .map((item) => {
        if (!item || typeof item !== "object") return null;
        return toSupportingLevel(item as PlanLayerLevel);
      })
      .filter((item): item is SupportingLevel => item !== null);
  }
  return [];
}
