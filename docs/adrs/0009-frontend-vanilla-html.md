# ADR-0009: Frontend is a single static HTML file with Plotly via CDN

- **Status:** Accepted
- **Date:** 2026-05-08
- **Authors:** Damian Gallardo

## Context

The application has three modes (Regression / Strategies / Terminal) plus a
Cmd-K global search modal, deep-linking URL state, an Alpha Graveyard tab,
embed widgets, replay scenarios, alert configuration, and the live SSE
stream. Across these surfaces it draws roughly two dozen Plotly charts and
exposes ~150 endpoints. By any reasonable measure that is "real" frontend
territory, and a typical 2026 default would reach for Vite + React or
SvelteKit before writing the first line of code.

We deliberately did the opposite: `web/index.html` is a single ~3,000-line
static HTML file with vanilla JavaScript and `<script src="https://cdn…/
plotly-2.x.min.js">` as the only frontend dependency. There is no `npm`,
no `package.json`, no build step, and no transpilation of any kind. The
whole frontend is served by an `nginx-alpine` container that does nothing
but `gzip` the bytes.

Three forces pushed us to this design:

1. **Grading criteria.** This is a course project graded on engineering
   discipline. A reviewer who clones the repo and runs `docker-compose up`
   gets a working app in 30 seconds, with no `npm install` step that might
   fail because their Node version disagrees. That is worth more than any
   reactive-component framework benefit.
2. **The data model is fundamentally read-mostly.** Every panel is a
   `fetch()` to an endpoint, a Plotly redraw, and a couple of event
   listeners. No optimistic updates, no client-side joins, no complex
   form state. Reactive frameworks pay their cost in code volume regardless
   of whether the app actually needs reactivity.
3. **Demo failure modes.** A 15-minute live demo that ends in "let me just
   reinstall node\_modules" is the worst outcome. Plain HTML cannot fail
   that way.

## Considered alternatives

- **Vite + React + TypeScript.** The default modern stack. Excellent DX
  for medium-complexity apps. Rejected for two reasons: the build artefact
  needs its own deploy pipeline, and the runtime adds ~150 KB of framework
  before the first chart renders. We weighed these against ~zero benefit
  for our (read-only, fetch-and-redraw) interaction model.
- **SvelteKit.** Smaller runtime, ergonomic, SSR-friendly. Same objection:
  build step + dependency tree + node-version friction. The reactive sugar
  doesn't pay for itself when the entire app is "click → fetch → Plotly.react".
- **HTMX.** Closer to our philosophy and seriously considered. Rejected only
  because Plotly already needs a JS context per chart; once we have JS
  anyway, HTMX's "no JS" pitch loses force, and vanilla `fetch()` is
  enough for our coordination needs.
- **Web Components / Lit.** Would solve the "3,000-line index.html is hard
  to navigate" problem, but introduces a build dependency on at least
  Lit itself. We addressed the navigation issue with `data-stab=` markers
  and clear section comments instead.

## Decision

Ship `web/index.html` as a single static file. Plotly via the public CDN.
All interactivity in vanilla JS. Shared chart styling lives in
`web/plotly-theme.js`, also served statically. Frontend deployment is
"copy the `web/` directory behind an nginx-alpine".

Concretely:

- **No package.json**, **no build step**, **no transpilation**.
- Chart styling is centralised in `plotly-theme.js`.
- API base URL is configured at runtime via `web/config.js`, which sets
  `window.PFM_API_BASE` and is overridable per-environment.
- The page is feature-flag-clean: every tab is a `<section
  data-mode="…" data-stab="…">` block that JS shows/hides on switch.

## Consequences

- **Zero install friction.** `git clone` → `docker-compose up` → working
  app in under a minute. There is no Node, no `npm install`, no build
  cache, no `.nvmrc`, no engine field to misalign.
- **Deploy is a `cp`.** Nginx serves `web/` as-is in every environment
  (dev compose, Render Static Sites, Cloudflare Pages, S3+CloudFront, …).
  No CI step has ever broken because of a frontend build.
- **Limited reactivity.** When two panels need to react to a shared piece
  of state (selected slug, watchlist, theme filter) we manage that via a
  small pub/sub helper inside the file. This is fine for our two-dozen-
  panels scope and would creak around 50+ panels.
- **No type safety in the frontend.** We accept this trade. The API is
  fully Pydantic-typed and exports a complete OpenAPI schema; if frontend
  type drift becomes a problem we can generate `.d.ts` types from the
  schema and load them in an editor without changing the deploy story.
- **Plotly version is pinned in HTML.** A breaking Plotly major would
  require a one-line edit. We accept this risk; it has not bitten us in
  the project's lifetime.
- **The single-file constraint is intentional and load-bearing.** Splitting
  `index.html` into ten files would re-introduce module-resolution
  questions and tempt us toward a build step. Until the file genuinely
  becomes unnavigable, we keep it as one file with disciplined section
  markers.
- **Re-evaluation trigger.** If the app grows a write-heavy surface
  (multi-user portfolios with optimistic updates, collaborative
  watchlists, drag-and-drop dashboards), revisit this ADR. Until then,
  vanilla wins.
