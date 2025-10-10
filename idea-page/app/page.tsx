import Link from "next/link";

export default function HomePage() {
  return (
    <main className="mx-auto flex min-h-screen max-w-2xl flex-col items-start justify-center gap-6 px-6 py-20">
      <h1 className="text-3xl font-semibold text-foreground">Trading Coach · Idea Pages</h1>
      <p className="text-muted-foreground">
        Provide a plan identifier to view its server-verified snapshot.
      </p>
      <Link
        href="/idea/SAMPLE_PLAN_ID"
        className="rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground transition hover:bg-primary/80"
      >
        Try sample plan
      </Link>
    </main>
  );
}
