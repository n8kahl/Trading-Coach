import { test, expect } from "@playwright/test";

const planId = process.env.E2E_PLAN_ID;

const describeBlock = planId ? test.describe : test.describe.skip;

describeBlock("Plan console", () => {
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
});
