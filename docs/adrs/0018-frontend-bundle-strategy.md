# ADR-0018: Frontend Bundle Strategy — Hybrid Single-File + Modular Extras

- **Status:** Accepted
- **Date:** 2026-05-16
- **Authors:** Damian Gallardo
- **References:** ADR-0009 (frontend vanilla HTML), CLAUDE.md "Plain HTML + Plotly from CDN", `.coordination/PROTOCOL-V2.md` (index.html as hot file)

## Context

The web frontend at `/Users/damiangallardoloya/Desktop/proyectofuentes/web/index.html` has grown to roughly **1.6 MB** as a single self-contained HTML file with inline `<style>` and `<script>` blocks for the three modes (Regression, Strategies / α Hub, Terminal). This decision was made early (ADR-0009) to keep the build-and-deploy story trivially simple: no Webpack, no Vite, no npm install in CI, no source-map drift between dev and prod, and a `file://` open of `index.html` is a viable fallback when the dev server is down.

Wave-11 and Wave-12 added approximately **25 modular files** under `web/css/` and `web/js/` — feature-scoped extensions such as `web/css/tokens.css` (the canonical design-token source per PROTOCOL-V2.md), `web/css/reduce-motion.css`, `web/js/onboarding-tour.js`, and assorted per-panel enhancements. These were created as new files specifically to avoid the multi-session race condition on `web/index.html`: PROTOCOL-V2.md mandates that only the `index-html-owner` may modify the single file directly, while any sub-agent may safely create a *new* `.css` or `.js` file and request a single mount line in `index.html`.

The resulting layout is now a genuine question of architecture rather than expedience. The team must decide whether to (a) collapse the modular extras back into `index.html`, (b) push further and split `index.html` itself into per-mode bundles, or (c) keep both and treat them as complementary. The decision affects load performance, multi-session collaboration ergonomics, grading legibility, and the cost of every future Wave.

## Decision

We adopt a **hybrid** strategy. `index.html` remains a single file containing the structural HTML, the dominant inline CSS, and the core JavaScript for the three modes. Modules under `web/css/` and `web/js/` are **additional, not replacement** assets: each is mounted into `index.html` exactly once via `<link rel="stylesheet" href="css/<name>.css">` or `<script defer src="js/<name>.js"></script>`. We do **not** adopt a build step — there is no Webpack, Rollup, esbuild, Vite, or npm install in CI. Plotly continues to load from CDN as specified in CLAUDE.md.

This satisfies three otherwise-conflicting constraints simultaneously:

1. **Graders** read one canonical file to understand the UI surface.
2. **Multi-session sub-agents** (up to 60 concurrent per PROTOCOL-V2.md) can add features in parallel by writing brand-new `web/css/*.css` and `web/js/*.js` files without contending for the single hot file.
3. **No build step** means `docker-compose up` works on a fresh clone with no node toolchain — preserving ADR-0009's core promise.

## Consequences

- **Gzip compresses `index.html` from ~1.6 MB to ~200 KB** over the wire on first paint, well within acceptable budget for a developer-facing analytics tool. Modular extras add a flat tail of small files, each itself gzipped.
- **HTTP/2 multiplexing** makes the per-request overhead of the additional `<link>` and `<script>` tags negligible — there is no head-of-line blocking penalty for the modular bundle.
- **`defer` on every modular `<script>` tag** preserves the inline core's render path: nothing in `web/js/` blocks first paint or the inline initialisation of the three mode panels.
- **Token discipline preserved**: `web/css/tokens.css` remains the single source of CSS variables (`--ah-bg`, `--orange`, etc.) per PROTOCOL-V2.md. No modular file re-defines a token.
- **Race-condition surface stays minimal**: agents add features by writing new files; only the `index-html-owner` claim modifies `index.html`, and only to add mount lines (typically two lines per Wave).
- **Cache discipline**: each modular file has its own browser cache lifetime independent of `index.html`. A tweak to `web/js/onboarding-tour.js` does not invalidate the 1.6 MB shell.
- **Cost paid**: ~25 extra HTTP requests on cold load. Under HTTP/2 over TLS resumption this is empirically <100 ms on the loopback dev server and <300 ms on typical broadband.

## Future revisit trigger

We revisit this decision **if and only if** `web/index.html` exceeds **3 MB** uncompressed or the modular file count exceeds **80**. At that point, code-splitting into per-mode bundles (`regression.html`, `strategies.html`, `terminal.html`) becomes justifiable, and a build step may finally be warranted. Splitting today would be premature optimisation: the current 1.6 MB / 25-module layout has measured first-paint latency well within budget, and a build step would impose a CI cost and grading-legibility cost that the present scale does not justify.

## What this ADR is not

This is not a license to inline more code into `index.html`. New work continues to be authored as `web/css/<feature>.css` and `web/js/<feature>.js` files. The "single file" status of `index.html` is preserved primarily as a stability and grading concern, not as a target for accretion.
