#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${TE_API_KEY:-}" ]]; then
  echo "TE_API_KEY is required (format key:secret)" >&2
  exit 1
fi

FROM=$(date -u +"%Y-%m-%d")
if command -v date >/dev/null 2>&1; then
  if date -u -d "+14 days" +"%Y-%m-%d" >/dev/null 2>&1; then
    TO=$(date -u -d "+14 days" +"%Y-%m-%d")
  else
    TO=$(date -u -v+14d +"%Y-%m-%d")
  fi
else
  echo "date command not available" >&2
  exit 1
fi

curl -s "https://api.tradingeconomics.com/calendar?country=united%20states&from=$FROM&to=$TO&c=${TE_API_KEY}" \
  | jq 'length as $n | {count: $n}'
