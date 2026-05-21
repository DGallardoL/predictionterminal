# ADR-0017: SSE vs WebSocket — picking the right push primitive

- **Status**: Accepted
- **Date**: 2026-05-16
- **Wave**: wave-13 (task W13-41)
- **Deciders**: Damian + Claude Code sessions
- **Related**: ADR-0001 (FastAPI), ADR-0007 (multi-session coordination), ADR-0011 (cache-stampede single-flight)

## Context

Two streaming primitives now coexist in the Prediction Terminal backend, and we
need a written rule for when to reach for each so future contributors stop
relitigating the question on every new live panel.

1. **Server-Sent Events (SSE)** — `GET /strategies/arb/stream`, a 2 s tick used
   by the Cross-venue Arb dashboard. Implemented as a FastAPI
   `StreamingResponse` with `media_type="text/event-stream"`. The browser uses
   the native `EventSource` API, which gives us automatic reconnect, the
   `Last-Event-ID` resume header, and HTTP semantics throughout the stack
   (proxies, TLS, gzip, auth cookies, CORS — everything Just Works™).
2. **WebSocket (WS)** — `GET /ws/live` (shipped in W13-11), a bidirectional
   channel that multiplexes `arb`, `jumps`, and `sentiment` feeds. Clients
   send `{op: "subscribe", channel: "arb"}` / `{op: "unsubscribe", ...}` and
   the server pushes channel-tagged frames plus 25 s heartbeats. Capped at
   100 concurrent connections.

Both technologies can deliver a "live tile" in the UI, so a default-correct
guideline avoids one-off bikeshedding when somebody adds a new panel.

## Decision

- **Default to SSE.** Use SSE whenever the data flow is strictly server →
  client, the message shape is small JSON, and the panel is content with a
  fixed cadence (1–5 s). The arb dashboard, sentiment leaderboard, and any
  future "ticker strip" fall here.
- **Use WebSocket only when the client must talk back.** That includes
  channel subscribe/unsubscribe, dynamic filter changes (e.g. "only NVDA
  jumps with z > 3"), back-pressure acks, or anything binary (future Parquet
  frames, protobuf orderbook snapshots). `/ws/live` is the canonical example.
- **Both stay production.** We do not migrate the arb SSE to WS, and we do
  not split `/ws/live` back into three SSE endpoints. Each one earned its
  place.

## Tradeoffs

| Concern               | SSE                                      | WebSocket                                  |
| --------------------- | ---------------------------------------- | ------------------------------------------ |
| Infra simplicity      | Plain HTTP/1.1+/2; proxies pass through  | Needs `Upgrade: websocket`; some LBs strip |
| Reconnect             | Built-in `EventSource` retry             | Hand-rolled exponential backoff in JS      |
| Resume after drop     | `Last-Event-ID` header, native           | App-level cursor in subscribe payload      |
| Direction             | Server → client only                     | Full duplex                                |
| Payload type          | UTF-8 text (JSON works fine)             | Text or binary                             |
| Auth                  | Cookies / `Authorization` like any GET   | Cookies work; bearer-in-query is awkward   |
| HTTP/2 & multiplexing | Friendly                                 | Separate TCP stream per socket             |
| Server resources      | One open file descriptor per client      | Same, plus app-level state machine         |
| Browser support       | Universal (modern), trivial JS           | Universal, but more code per panel         |
| Testing               | `httpx` streaming + assertion is trivial | `pytest-asyncio` + websockets client harness |

## Consequences

- **Positive**: New panels get a one-sentence answer ("does the client need
  to send anything? → WS, else SSE"). The arb dashboard keeps its proven
  EventSource resume story. `/ws/live` keeps its multiplexed channel model
  so we are not opening three SSE streams from every browser tab.
- **Negative**: We carry two streaming code paths, two test harnesses, and
  two reconnect bugs to chase. The 100-conn cap on `/ws/live` is a real
  ceiling and will require connection pooling on the frontend once we ship
  the planned "Live Edge" tab.
- **Mitigations**: Document the rule in `docs/USER_GUIDE.md` next to the
  streaming examples; add a lint check in `scripts/validate_endpoints.py`
  that flags any new SSE endpoint accepting query-string filters that
  change mid-stream (a smell that it should have been WS).

## Alternatives considered

- **WS for everything.** Rejected: the arb SSE has zero client→server
  traffic and gains nothing from upgrade negotiation, while losing native
  reconnect.
- **SSE for everything (multiplexed by `event:` field).** Rejected:
  unsubscribing a single channel mid-stream requires tearing down the whole
  connection, which churns proxies and burns reconnect budget.
- **gRPC-Web bidi streaming.** Rejected: adds a protobuf toolchain to a
  POC whose grading rubric values simplicity (ADR-0009 frontend-vanilla-html).
- **Long polling.** Rejected on latency grounds; arb opportunities decay in
  seconds.

## Revisit triggers

Re-open this ADR if (a) we add a binary frame format, (b) connection counts
on `/ws/live` hit the 100 cap in production, or (c) a CDN/edge layer starts
buffering SSE in a way `EventSource` cannot detect.
