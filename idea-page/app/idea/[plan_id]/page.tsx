"use server";

import IdeaSnapshotClient from "@/components/IdeaSnapshotClient";
import { fetchIdea, SnapshotNotFoundError } from "@/lib/api";
import { notFound } from "next/navigation";

type IdeaPageProps = {
  params: {
    plan_id: string;
  };
};

export default async function IdeaPage({ params }: IdeaPageProps) {
  try {
    const snapshot = await fetchIdea(params.plan_id);
    return <IdeaSnapshotClient initialData={snapshot} planId={params.plan_id} />;
  } catch (error) {
    if (error instanceof SnapshotNotFoundError) {
      notFound();
    }
    throw error;
  }
}

export async function generateMetadata({ params }: IdeaPageProps) {
  try {
    const snapshot = await fetchIdea(params.plan_id);
    return {
      title: `${snapshot.plan.symbol} Idea · ${snapshot.plan.bias.toUpperCase()} setup`,
      description: `${snapshot.plan.setup} plan for ${snapshot.plan.symbol} with targets ${snapshot.plan.targets.map((t) => t.toFixed(snapshot.plan.decimals)).join(", ")}.`,
    };
  } catch (error) {
    return {
      title: "Idea not found",
      description: "Unable to retrieve the requested plan.",
    };
  }
}
