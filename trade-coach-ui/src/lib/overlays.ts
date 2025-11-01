"use client";

type OverlayPayload = Record<string, unknown>;

export type OverlayEntry = {
  id: string;
  version: number;
  kind?: string;
  data: OverlayPayload;
};

export type OverlayDiff =
  | ({ id: string; version: number; remove?: false } & Partial<OverlayEntry>)
  | { id: string; version: number; remove: true };

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return !!value && typeof value === "object" && !Array.isArray(value);
}

function mergePayload(base: OverlayPayload, delta: OverlayPayload): { changed: boolean; value: OverlayPayload } {
  let changed = false;
  const next: OverlayPayload = { ...base };
  for (const [key, value] of Object.entries(delta)) {
    const current = next[key];
    if (isPlainObject(current) && isPlainObject(value)) {
      const nested = mergePayload(current, value);
      if (nested.changed) {
        next[key] = nested.value;
        changed = true;
      }
      continue;
    }
    if (!Object.is(current, value)) {
      next[key] = value;
      changed = true;
    }
  }
  return { changed, value: changed ? next : base };
}

export class OverlayRegistry {
  private readonly overlays = new Map<string, OverlayEntry>();

  constructor(initial?: OverlayEntry[]) {
    if (initial) {
      initial.forEach((entry) => this.upsert(entry));
    }
  }

  upsert(entry: OverlayEntry): OverlayEntry {
    const existing = this.overlays.get(entry.id);
    if (existing && existing.version > entry.version) {
      return existing;
    }
    let payload = entry.data ?? {};
    if (existing && existing.version === entry.version) {
      const merged = mergePayload(existing.data, payload);
      payload = merged.changed ? merged.value : existing.data;
    } else if (existing && existing.version < entry.version) {
      const merged = mergePayload(existing.data, payload);
      payload = merged.changed ? merged.value : existing.data;
    }
    const normalized: OverlayEntry = {
      id: entry.id,
      version: Math.max(existing?.version ?? 0, entry.version),
      kind: entry.kind ?? existing?.kind,
      data: payload,
    };
    this.overlays.set(entry.id, normalized);
    return normalized;
  }

  applyDiff(diff: OverlayDiff): OverlayEntry | null {
    const existing = this.overlays.get(diff.id);
    if (!existing) {
      if ("remove" in diff && diff.remove) {
        return null;
      }
      if (!("data" in diff) || !isPlainObject(diff.data)) {
        return null;
      }
      return this.upsert({
        id: diff.id,
        version: diff.version,
        kind: "kind" in diff ? diff.kind : undefined,
        data: diff.data ?? {},
      });
    }
    if (diff.version < existing.version) {
      return existing;
    }
    if ("remove" in diff && diff.remove) {
      this.overlays.delete(diff.id);
      return null;
    }
    const payload = mergePayload(existing.data, (diff.data ?? {}) as OverlayPayload);
    if (!payload.changed && (diff.kind === undefined || diff.kind === existing.kind)) {
      if (diff.version > existing.version) {
        const updated: OverlayEntry = {
          ...existing,
          version: diff.version,
        };
        this.overlays.set(diff.id, updated);
        return updated;
      }
      return existing;
    }
    const updated: OverlayEntry = {
      id: diff.id,
      version: diff.version,
      kind: diff.kind ?? existing.kind,
      data: payload.value,
    };
    this.overlays.set(diff.id, updated);
    return updated;
  }

  remove(id: string): boolean {
    return this.overlays.delete(id);
  }

  get(id: string): OverlayEntry | undefined {
    return this.overlays.get(id);
  }

  snapshot(): OverlayEntry[] {
    return Array.from(this.overlays.values());
  }

  clear(): void {
    this.overlays.clear();
  }
}

export function applyOverlayDiffs(registry: OverlayRegistry, diffs: OverlayDiff[] | OverlayDiff): OverlayEntry[] {
  const applied: OverlayEntry[] = [];
  const list = Array.isArray(diffs) ? diffs : [diffs];
  for (const diff of list) {
    const result = registry.applyDiff(diff);
    if (result) {
      applied.push(result);
    }
  }
  return applied;
}
