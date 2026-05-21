# Research Notes — Polymarket API (verified 2026-04-23)

This document captures what has been **verified from official docs and a real GitHub issue** so that Claude Code doesn't have to re-discover. Cite this when asked "are you sure about X?"

## Two APIs, different base URLs

Polymarket has multiple services. For this POC we use two:

### Gamma API — market metadata
- **Base URL:** `https://gamma-api.polymarket.com`
- Purpose: discover markets, get metadata (slug, dates, volumes, clobTokenIds)
- No auth required for read endpoints
- Returns rich JSON with many fields; we only care about a few

### CLOB API — price history (the main one)
- **Base URL:** `https://clob.polymarket.com`
- Purpose: historical prices, order books, trading (we only read)
- No auth required for public endpoints (`/prices-history` is public)

## Endpoints we use

### 1. Get market metadata by slug

```
GET https://gamma-api.polymarket.com/markets?slug={slug}
```

Or:

```
GET https://gamma-api.polymarket.com/markets/slug/{slug}
```

**How to find a slug:** go to `https://polymarket.com/event/XXX` — the `XXX` is the slug.

Example: `https://polymarket.com/event/fed-decision-in-october` → slug = `fed-decision-in-october`

**Response (fields we care about):**
```json
{
  "id": "...",
  "question": "Will the Fed cut rates in October?",
  "slug": "fed-decision-in-october",
  "conditionId": "0x...",
  "clobTokenIds": "[\"71321...\", \"52114...\"]",   // JSON string, needs json.loads
  "startDate": "2025-07-15T00:00:00Z",
  "endDate": "2026-11-01T00:00:00Z",
  "active": true,
  "closed": false,
  "volume": "1234567.89",
  "liquidity": "123456.78",
  "volumeClob": 950000,
  "liquidityClob": 100000
}
```

**IMPORTANT:** `clobTokenIds` is a **string containing JSON**, not a native array. Must parse twice:
```python
import json
clob_token_ids = json.loads(market["clobTokenIds"])  # ['yes_token_id', 'no_token_id']
yes_token_id = clob_token_ids[0]
```

### 2. Get price history

```
GET https://clob.polymarket.com/prices-history
```

**Query parameters** (verified from official docs):

| Name | Type | Required | Description |
|---|---|---|---|
| `market` | string | **YES** | The asset ID (token ID) to query. NOT the condition ID or slug. |
| `startTs` | number | no | Unix timestamp (seconds) lower bound |
| `endTs` | number | no | Unix timestamp (seconds) upper bound |
| `interval` | enum | no | One of: `max`, `all`, `1m`, `1w`, `1d`, `6h`, `1h` |
| `fidelity` | integer | no | Bucket size in **minutes**. Default = 1. |

**Interval vs fidelity:**
- `interval` = lookback window (how far back to fetch)
- `fidelity` = granularity (minutes between points)

**Response:**
```json
{
  "history": [
    {"t": 1706745600, "p": 0.42},
    {"t": 1706832000, "p": 0.45},
    ...
  ]
}
```
- `t` is unix seconds
- `p` is price in [0, 1]

### 3. ⚠️ Known limitation: resolved markets

**Source:** https://github.com/Polymarket/py-clob-client/issues/216 (filed Dec 2025)

For markets that are **resolved/closed**, the endpoint **only returns data when `fidelity >= 720`** (12 hours). Sub-12h requests return empty arrays.

**Our mitigation:** always use `fidelity=1440` (daily). Daily resolution is what we want anyway for daily returns regressions.

## Rate limits (verified)

From https://docs.polymarket.com/api-reference/rate-limits :

| Endpoint | Limit | Window | Auth |
|---|---|---|---|
| General | 15,000 | 10s | No |
| Health check | 100 | 10s | No |
| CLOB API general | 9,000 | 10s | No |
| **`/prices-history`** | **1,000** | **10s** | **No** |

Our cache TTL of 1h means even with 100 concurrent users we'd never hit the limit.

## End-to-end flow (this is the recipe)

```python
import json
import requests

BASE_GAMMA = "https://gamma-api.polymarket.com"
BASE_CLOB = "https://clob.polymarket.com"

def fetch_factor_history(slug: str, start_ts: int | None = None) -> list[dict]:
    # Step 1: Gamma API — get metadata
    r = requests.get(f"{BASE_GAMMA}/markets", params={"slug": slug}, timeout=15)
    r.raise_for_status()
    markets = r.json()
    if not markets:
        raise ValueError(f"No market found for slug={slug}")
    market = markets[0]
    
    # Step 2: Parse the embedded JSON string
    token_ids = json.loads(market["clobTokenIds"])
    yes_token_id = token_ids[0]
    
    # Step 3: CLOB API — get price history
    params = {
        "market": yes_token_id,
        "interval": "max",
        "fidelity": 1440,  # daily
    }
    if start_ts is not None:
        params["startTs"] = start_ts
    
    r2 = requests.get(f"{BASE_CLOB}/prices-history", params=params, timeout=15)
    r2.raise_for_status()
    history = r2.json()["history"]
    return history  # [{'t': unix_seconds, 'p': 0.0..1.0}, ...]
```

## Unknowns / to-verify-at-build-time

- What happens if a slug has multiple markets (compound events)? → test during build; likely take first or expose parameter
- Exact format of `startTs`/`endTs` in some edge cases → use `int(datetime.timestamp())` always
- Does `interval=max` always override `startTs`? → test; use `startTs` explicitly if set and omit `interval`

## Links (all verified accessible 2026-04-23)

- API docs hub: https://docs.polymarket.com/
- prices-history spec: https://docs.polymarket.com/api-reference/markets/get-prices-history
- Rate limits: https://docs.polymarket.com/api-reference/rate-limits
- Get market by slug: https://docs.polymarket.com/api-reference/markets/get-market-by-slug
- Fetching markets guide: https://docs.polymarket.com/market-data/fetching-markets
- py-clob-client repo: https://github.com/Polymarket/py-clob-client
- Resolved-market fidelity issue: https://github.com/Polymarket/py-clob-client/issues/216
