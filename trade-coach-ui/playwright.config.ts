import { defineConfig } from "@playwright/test";

const baseURL = process.env.E2E_UI_BASE_URL || "http://localhost:3000";

export default defineConfig({
  testDir: "playwright/tests",
  timeout: 30_000,
  use: {
    baseURL,
    headless: true,
    trace: "on-first-retry",
  },
});
