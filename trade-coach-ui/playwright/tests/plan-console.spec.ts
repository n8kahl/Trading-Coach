import { test, expect } from "@playwright/test";

const planId = process.env.E2E_PLAN_ID;
const watchlistPlanId = process.env.E2E_WATCHLIST_PLAN_ID || planId;
const coachPlanId = process.env.E2E_COACH_PLAN_ID || planId;
const expectNoTradeBanner = process.env.E2E_EXPECT_NO_TRADE === "1";
const replaySymbol = process.env.E2E_REPLAY_SYMBOL;

const planDescribe = planId ? test.describe : test.describe.skip;

planDescribe("Plan console", () => {
  test("loads plan console and displays session + objective", async ({ page }) => {
    if (!planId) return;
    await page.goto(`/plan/${planId}`);

    await expect(page.getByText(/Objective/i).first()).toBeVisible({ timeout: 10_000 });
    await expect(page.getByText(/Progress \d+%/i)).toBeVisible();
    await expect(page.getByText(/As of/i)).toBeVisible();
  });

  test("opens canonical chart link", async ({ page, context }) => {
    if (!planId) return;
    await page.goto(`/plan/${planId}`);

    const [popup] = await Promise.all([
      context.waitForEvent("page"),
      page.getByRole("link", { name: /open chart/i }).click(),
    ]);

    await popup.waitForLoadState("load");
    expect(popup.url()).toContain("/chart");
    await popup.close();
  });

  test("watchlist sorts by actionability and navigates", async ({ page }) => {
    if (!watchlistPlanId) return;
    await page.goto(`/plan/${watchlistPlanId}`);

    const items = page.locator('[data-testid="watchlist-item"]');
    await expect(items.first()).toBeVisible({ timeout: 20_000 });

    const statuses = await items.evaluateAll((nodes) => nodes.map((node) => node.getAttribute("data-actionable") || ""));
    const lastSoon = statuses.lastIndexOf("soon");
    const firstWatching = statuses.indexOf("watching");
    if (lastSoon !== -1 && firstWatching !== -1) {
      expect(lastSoon).toBeLessThan(firstWatching);
    }

    const rawDistances = await items.evaluateAll((nodes) => nodes.map((node) => node.getAttribute("data-distance")));
    const distances = rawDistances
      .map((value) => {
        if (!value) return null;
        const parsed = Number.parseFloat(value);
        return Number.isFinite(parsed) ? parsed : null;
      })
      .filter((value): value is number => value != null);
    for (let idx = 1; idx < distances.length; idx += 1) {
      expect(distances[idx]).toBeGreaterThanOrEqual(distances[idx - 1] - 1e-6);
    }

    const targetPlanId = await items.first().getAttribute("data-plan-id");
    await Promise.all([
      page.waitForURL((url) => url.pathname.includes("/plan/"), { timeout: 15_000 }),
      items.first().locator("button").click(),
    ]);
    if (targetPlanId) {
      const currentPath = decodeURIComponent(new URL(page.url()).pathname);
      expect(currentPath).toContain(targetPlanId);
    }
    await expect(page.getByText(/Objective/i).first()).toBeVisible();
  });

  test("coach panel renders timeline", async ({ page }) => {
    if (!coachPlanId) return;
    await page.goto(`/plan/${coachPlanId}`);
    await expect(page.getByTestId("coach-panel")).toBeVisible({ timeout: 15_000 });
    const timelineEntries = page.getByTestId("coach-timeline-entry");
    await expect(timelineEntries.first()).toBeVisible({ timeout: 20_000 });
    if (expectNoTradeBanner) {
      await expect(page.getByTestId("coach-no-trade-banner")).toBeVisible({ timeout: 20_000 });
    }
  });
});

const replayDescribe = replaySymbol ? test.describe : test.describe.skip;

replayDescribe("Simulated dojo", () => {
  test("displays simulated banner and overlays", async ({ page }) => {
    if (!replaySymbol) return;
    await page.goto(`/replay/${replaySymbol}`);

    await expect(page.getByText(/Simulated Dojo/i)).toBeVisible({ timeout: 15_000 });
    await expect(page.getByText(/Simulated live/i)).toBeVisible({ timeout: 15_000 });
    await expect(page.locator("canvas").first()).toBeVisible({ timeout: 15_000 });

    const primaryLevels = page.getByText(/Primary Levels/i).locator("..").locator("li");
    await expect(primaryLevels.first()).toBeVisible({ timeout: 15_000 });
  });
});
