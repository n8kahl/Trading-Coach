"use server";

import IdeaSnapshotClient from "@/components/IdeaSnapshotClient";
import { fetchIdea, SnapshotNotFoundError } from "@/lib/api";
import { notFound } from "next/navigation";

type IdeaVersionPageProps = {
  params: {
    plan_id: string;
    v: string;
  };
};

export default async function IdeaVersionPage({ params }: IdeaVersionPageProps) {
  try {
    const snapshot = await fetchIdea(params.plan_id, params.v);
    return <IdeaSnapshotClient initialData={snapshot} planId={params.plan_id} version={params.v} />;
  } catch (error) {
    if (error instanceof SnapshotNotFoundError) {
      notFound();
    }
    throw error;
  }
}

export async function generateMetadata({ params }: IdeaVersionPageProps) {
  try {
    const snapshot = await fetchIdea(params.plan_id, params.v);
    return {
      title: `${snapshot.plan.symbol} Idea v${snapshot.plan.version}`,
      description: `${snapshot.plan.setup} plan version ${snapshot.plan.version} for ${snapshot.plan.symbol}.`,
    };
  } catch (error) {
    return {
      title: "Idea version not found",
      description: "Unable to retrieve the requested plan version.",
    };
  }
}
