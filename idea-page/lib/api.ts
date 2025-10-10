import { IdeaSnapshot, type TIdeaSnapshot } from "@/lib/types";

const BASE_URL = "https://trading-coach-production.up.railway.app";

export class SnapshotNotFoundError extends Error {
  constructor(message = "Snapshot not found") {
    super(message);
    this.name = "SnapshotNotFoundError";
  }
}

export async function fetchIdea(planId: string, version?: string | number): Promise<TIdeaSnapshot> {
  const path = version ? `/idea/${planId}/${version}` : `/idea/${planId}`;
  const res = await fetch(`${BASE_URL}${path}`, { cache: "no-store" });

  if (!res.ok) {
    if (res.status === 404) {
      throw new SnapshotNotFoundError();
    }
    throw new Error("Unable to load idea snapshot");
  }

  const json = await res.json();
  return IdeaSnapshot.parse(json);
}
