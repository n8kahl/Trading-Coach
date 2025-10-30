# QA Checklist

- [x] Opening a canonical `/chart` link with `plan_id` hydrates the plan panel immediately, including entry/stop/targets/confluence/rules and overlays.
- [x] Opening a non-canonical `/chart` link lacking `plan_id` triggers auto-canonicalization via `/gpt/chart-url`, updates the URL, then hydrates overlays and plan data.
- [x] Timeframe buttons update the chart interval/range, refresh candles, adjust the header label, and push state so browser back/forward restores prior settings.
- [x] The “Reversal” control flips the plan bias, refreshes overlays/guidance, updates the URL direction parameter, and supports toggling back.
- [x] Killing or dropping the `/stream` EventSource shows a reconnecting status, retries with exponential backoff (max 30s), and resumes updates when the feed returns.
- [x] Plans reporting targets but delivering empty parsed data surface the subtle “Targets unavailable (awaiting updates)” banner without throwing.
