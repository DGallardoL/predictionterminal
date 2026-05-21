# Screenshots — capture recipe

The README links these files via `![alt](docs/screenshots/<file>.png)`. They
are intentionally absent from version control until captured against a
live demo deploy so they reflect real production data, not synthetic
fixtures.

## Files to capture

| File | What it shows | How to capture |
|---|---|---|
| `terminal.png` | Terminal landing with theme heatmap, top movers, calendar, PM-VIX. | Open `http://localhost:8080`, default tab. Resize browser to 1440×900, full-page screenshot. |
| `alphahub.png` | α Hub with B_VALIDATED card filter applied, three cards visible. | Strategies → α Hub. Filter `Min tier = B_VALIDATED`. Capture with at least three cards in frame. |
| `regression.png` | `/fit` response: coefficient table, residual plot, VIF panel. | Regression mode. Pre-load `NVDA` + `nvda-ai-mentions-by-2026q3` + `gpt5-by-eoy`. Click **Run fit**. Capture after charts render. |
| `quote.png` | Quote page detail: hero, price chart, eight-panel chart grid. | Click any market in Top movers → quote page. Wait for all eight panels. Capture in 1440×900. |
| `graveyard.png` | Alpha Graveyard tab with the six retired strategies + death certificates. | Strategies → α Hub → Graveyard tab. Capture full table + first death certificate expanded. |
| `cmdk.png` | `Cmd+K` modal open with autocomplete results. | Press `Cmd+K`. Type `nvda`. Capture with three results visible and Recents below. |

## Conventions

- **Resolution**: 1440×900 minimum, 2880×1800 (Retina) preferred.
- **Format**: PNG, no alpha channel.
- **Compression**: pass through `pngquant` to keep each file under
  500 KB.
- **No personally identifying info**: cover or blur any logged-in
  user details. The frontend has no auth today, so this should not
  apply.
- **Disclaimer footer must be in frame** for every screenshot that
  shows price / strategy data. The "Not investment advice" footer is
  load-bearing.

## Refresh cadence

Re-capture before each tagged release (e.g. after `v0.2.0`, `v0.3.0`).
Keep the file names stable so the README links don't rot.
