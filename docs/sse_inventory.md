# SSE / Realtime endpoint inventory

The FastAPI app exposes two Server-Sent Events endpoints. Both live under
`/terminal/*` and both emit `text/event-stream` with `event: <name>\ndata:
<json>\n\n` framing. Two other "live" paths (`/alpha-hub/live-panel` and
`/signals/live`) sound streamy but are actually **plain JSON GETs** — they
return a snapshot of the precomputed live-signals state, not a stream.

## Endpoints

| Path | Status | Event types emitted | Data schema (per event) | UI panel(s) that could use it |
| --- | --- | --- | --- | --- |
| `GET /terminal/stream?subs=kind:slug,kind:slug` | **Current** (multiplexed hub) | `ready`, `tick`, `book`, `tape`, `hb`, `bye` | `{"type": <kind>, "slug": <slug>, "data": …, "ts": <unix>}` — `tick.data = {mid, bid, ask}`; `book.data = {bids:[{price,size}], asks:[…]}`; `tape.data = {fills:[{price,size,side,ts}]}`; `ready.data = {cid, subs, ts}`; `hb.data = {ts}`; `bye.data = {reason}` | Terminal hero price tile (`tick`), L1 book table (`book`), recent-trades tape (`tape`); BTC arb Poly-side midpoint ticker (`tick`). |
| `GET /terminal/live-stream?slugs=a,b,c&hz=0.5` | **Deprecated** (one poller per connection — slated for v0.2 removal) | `ready`, `tick`, `bye` | `tick` payload: `{slug, mid, bid, ask, ts}`; `ready`: `{slugs, hz, interval_s}`; `bye`: `{reason}` | Anything that already wires to it (back-compat only — prefer `/terminal/stream` for new code). |

### Subscription syntax — `/terminal/stream`

`subs` is a comma-separated list of `kind:slug` pairs.

- `kind` ∈ `{tick, book, tape}` (registered in `pfm/realtime/pollers.py:255`).
- `slug` is a Polymarket market slug. Unknown kinds → 400, malformed → 400,
  >60 subs per client → 400, hub at 500-client cap → 503 with `Retry-After: 30`.
- The endpoint emits a `ready` frame immediately on connect, then `tick`
  / `book` / `tape` frames at ~2 s cadence (`DEFAULT_POLL_INTERVAL_S`), and
  a `hb` heartbeat every 10 s to defeat proxy idle timeouts.

Example: `GET /terminal/stream?subs=tick:will-the-fed-cut-in-june-2026,book:will-the-fed-cut-in-june-2026`

### Subscription syntax — `/terminal/live-stream` (legacy)

`slugs` is comma-separated (max 30); `hz` is 0.1–5.0 (default 0.5). The
endpoint resolves each slug to a YES token once at connect, then loops
emitting one `tick` per slug per `1/hz` seconds until the 300 s server-side
deadline elapses (clients reconnect).

## What is NOT available

These do not exist as SSE endpoints — anything labelled "live" elsewhere
is a plain JSON GET:

- `/alpha-hub/live-panel` → JSON snapshot of `web/data/live_signals.json`.
- `/signals/live` → JSON snapshot of the live-signals job state.
- No SSE for Binance / external CEX prices. The BTC latency-arb panel
  must either poll the JSON REST endpoint or wire its own WebSocket
  to Binance directly from the browser.

## TODO — endpoints that *almost* work

- **Binance midpoint stream.** The BTC up/down arb panel needs the
  CEX leg of the spread to pulse in sync with the Polymarket leg. There
  is no `/terminal/stream` kind for Binance — `pollers.py` only knows
  `tick|book|tape` against Polymarket CLOB. Cheapest fix: a new poller
  `kind="cex_mid"` that hits `https://api.binance.com/api/v3/ticker/bookTicker?symbol=BTCUSDT`
  every 2 s and emits `{type:"cex_mid", slug:"BTCUSDT", data:{mid,bid,ask}, ts}`.
- **Aggregated Δ-logit / spread stream.** A common UI pattern is to
  show `logit(p_poly) - logit(p_binance_implied)` ticking. Today the
  client must subscribe to both and compute the spread in JS. A server-
  side composite kind (e.g. `kind="spread_btc_updown"`) would halve the
  in-browser work and avoid races on heartbeat-interleaved frames.
- **`/terminal/live-stream` removal.** It's tagged DEPRECATED 2026-05-08
  in the source. Once all front-end callers have been migrated to
  `/terminal/stream`, delete the module and drop the router include in
  `main.py`. Until then the inventory has to list both.
