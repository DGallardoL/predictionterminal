#!/usr/bin/env bash
# Smoke-test the deployed API. Exits non-zero on first failure.
# Usage: ./scripts/smoke_test.sh [BASE_URL]
#   default BASE_URL: http://localhost:8000

set -euo pipefail

BASE=${1:-http://localhost:8000}
echo "Testing $BASE..."

curl -sf "$BASE/health" > /dev/null && echo "  ok  /health"
curl -sf "$BASE/health/detail" > /dev/null && echo "  ok  /health/detail"
curl -sf "$BASE/metrics" > /dev/null && echo "  ok  /metrics"
curl -sf "$BASE/factors" > /dev/null && echo "  ok  /factors"

if command -v jq >/dev/null 2>&1; then
  count=$(curl -sf "$BASE/openapi.json" | jq -r '.paths | keys | length')
  echo "  endpoints: $count"
else
  curl -sf "$BASE/openapi.json" > /dev/null && echo "  ok  /openapi.json (install jq for endpoint count)"
fi

curl -sf "$BASE/alpha-hub/graveyard" > /dev/null && echo "  ok  /alpha-hub/graveyard"
curl -sf "$BASE/alpha/decay" > /dev/null && echo "  ok  /alpha/decay"

echo "All checks passed."
