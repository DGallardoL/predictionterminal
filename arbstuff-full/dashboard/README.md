# ARB · Kalshi × Polymarket Monitor

Real-time arbitrage monitoring dashboard. Single-page React app that consumes
an SSE stream from a Flask backend and surfaces live cross-market
opportunities, scan logs, PnL, and event config.

## Run

```bash
npm install
npm run dev
```

Dev server runs on `http://localhost:5173` and proxies `/api/*` to
`http://localhost:5060` (your Flask backend). Adjust in `vite.config.js` if
your backend lives elsewhere.

```bash
npm run build       # production build -> dist/
npm run preview     # preview the built bundle
```

## Routes

- `/dashboard` — main monitor (default)
- `/review` — match review page (placeholder; wire your endpoints)

## Backend contract

All relative to `/api`:

- `GET /dashboard/stream` — SSE pushing full state every ~2s
- `GET /dashboard/orderbook?kalshi_ticker=…&poly_token=…`
- `GET /dashboard/pnl` — `{trades, total_pnl, count}`
- `GET /dashboard/config-stats` — `{reviewed, main, combined_mapped}`
- `GET /config-events` — `{events: [...]}`
- `POST /dashboard/blacklist` — `{arb_key}`
- `DELETE /dashboard/blacklist` — clear all
- `POST /dashboard/settings` — `{email_enabled, threshold, min_alert_profit, scan_mode}`

## Design

Terminal-trader aesthetic: dense, monospaced numerics (JetBrains Mono),
editorial italic serif headers (Instrument Serif), near-black canvas with
subtle atmospheric gradients + grain, phosphor-green signal accents, 3px
scrollbars, sticky status bar with pulsing connection dot. No UI libraries —
pure CSS variables with a small design system in `src/styles.css`.

## File map

```
src/
  main.jsx                      router + providers
  styles.css                    full design system
  hooks/useSSE.js               auto-reconnecting SSE hook
  pages/
    Dashboard.jsx               main shell
    Review.jsx                  /review placeholder
  components/
    StatusBar.jsx               top bar: status, balances, badges, settings
    Metrics.jsx                 5-card metric row
    OpportunitiesTab.jsx        split table + detail + orderbook
    ScanLogTab.jsx              sticky-header scan log with filters
    PnLTab.jsx                  SVG cumulative chart + trade log
    ConfigTab.jsx               event config browser with search
    SettingsPanel.jsx           slide-in overlay settings
    icons.jsx                   inline SVG icon set
```
