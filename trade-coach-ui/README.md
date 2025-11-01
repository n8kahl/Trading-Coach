This is a [Next.js](https://nextjs.org) project bootstrapped with [`create-next-app`](https://nextjs.org/docs/app/api-reference/cli/create-next-app).

## Getting Started

First, run the development server:

```bash
npm run dev
# or
yarn dev
# or
pnpm dev
# or
bun dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.

Routes
- `/plan/:planId` – Live plan console with deltas, coaching timeline, and price stream.
- `/replay/:symbol` – Market Replay with Scenario Plans (generate Scalp/Intraday/Swing scenarios, compare overlays, adopt/regenerate).

Feature Flags
- `NEXT_PUBLIC_API_BASE_URL` – Backend REST base URL.
- `NEXT_PUBLIC_WS_BASE_URL` – Backend WebSocket base URL.
- `NEXT_PUBLIC_BACKEND_API_KEY` – Optional Bearer token.
  Scenario Plans UI is enabled by default (no feature flag).

## Testing

- `npm run test` executes the Vitest unit suite (components, hooks, and stores).
- `npm run e2e` runs Playwright scenarios. Set `E2E_UI_BASE_URL` (defaults to `http://localhost:3000`) and `E2E_PLAN_ID` to point at a real plan before running.

You can start editing the page by modifying `app/page.tsx`. The page auto-updates as you edit the file.

This project uses [`next/font`](https://nextjs.org/docs/app/building-your-application/optimizing/fonts) to automatically optimize and load [Geist](https://vercel.com/font), a new font family for Vercel.

## Learn More

To learn more about Next.js, take a look at the following resources:

- [Next.js Documentation](https://nextjs.org/docs) - learn about Next.js features and API.
- [Learn Next.js](https://nextjs.org/learn) - an interactive Next.js tutorial.

You can check out [the Next.js GitHub repository](https://github.com/vercel/next.js) - your feedback and contributions are welcome!

## Deploy on Vercel

The easiest way to deploy your Next.js app is to use the [Vercel Platform](https://vercel.com/new?utm_medium=default-template&filter=next.js&utm_source=create-next-app&utm_campaign=create-next-app-readme) from the creators of Next.js.

Check out our [Next.js deployment documentation](https://nextjs.org/docs/app/building-your-application/deploying) for more details.
