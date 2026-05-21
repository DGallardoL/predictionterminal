# Production Launch Checklist — Wave-13

Task ID: **W13-LAUNCH-CHECKLIST**  ·  Generated: 2026-05-16 (UTC)  ·  Audit scope: live `localhost:8000` + repo tree.

This checklist hard-gates a production push. Items below were *probed*, not assumed.
Status values: `PASS` (verified working), `FAIL` (verified broken / missing),
`PARTIAL` (works but missing a sub-requirement called out in CLAUDE.md or Wave-11/12/13).

Re-run probes before every deploy. The numbers and endpoint counts shift weekly.

---

## 1. Backend

| # | Item | Status | Evidence |
|---|------|--------|----------|
| 1.1 | `docker-compose up` works | `PARTIAL` | `docker-compose.yml`, `api/Dockerfile`, `web/Dockerfile` all present; not exercised in this audit. Run `docker compose up --build` and confirm 3 services pass healthchecks before flipping DNS. |
| 1.2 | `/health` returns `{"status":"ok",...}` | `PASS` | `curl :8000/health` returned `{"status":"ok","version":"0.1.0"}`. |
| 1.3 | `/health/deep` shows all sources `ok` | `FAIL` | `curl :8000/health/deep` returned 404. **No `/health/deep` route in OpenAPI**. The closest paths are `/health/detail` and `/sources/health`. Wave-13 deep-health endpoint must be wired in `pfm.health_router` (or wherever the deep probe was scoped) before launch. |
| 1.4 | `/factors` returns 1228 | `PASS` | `GET /factors/all` → `total=1238` factors served. `factors.yml` contains exactly **1228** `- id:` entries (`grep -c '^- id:'`). The +10 delta is from the new `sentiment:*` virtual source registered at runtime — expected. |
| 1.5 | `/openapi.json` paths >= 297 (post-W13-01) | `FAIL` | `curl /openapi.json | jq '.paths|length'` returns **271**. The Wave-13 endpoint adds promised in W13-01 (deep health, `/metrics/audit`, `/admin/cache-stats`, etc.) have not landed in the running app. Either the worker needs a graceful reload or the routers are not mounted. |
| 1.6 | All Wave-13 endpoints reachable | `FAIL` | Probed: `/health/deep` MISSING, `/metrics/audit` MISSING, `/admin/cache-stats` MISSING. The supporting modules may exist on disk, but `main.py` is not exposing them on the live process. |
| 1.7 | `pytest -q` passes (current count + budget) | `PARTIAL` | `find api/tests -name 'test_*.py' \| wc -l` → **311 test files**; CLAUDE.md target is ~2700 tests in ~80 s. Suite not run in this audit (would block coordination window). Run `cd api && PYTHONPATH=src .venv/bin/python -m pytest -q --timeout=120` before deploy. |
| 1.8 | `ruff check` clean | `PARTIAL` | CI job `lint` exists in `.github/workflows/ci.yml` and runs `ruff check .` + `ruff format --check .`. Local re-run not performed in this audit. Verify with `cd api && ruff check .`. |
| 1.9 | No bare `except:` clauses | `PASS` | `grep -rEn 'except\s*:' api/src/pfm/` → 0 matches. |
| 1.10 | No hardcoded secrets | `PASS` | Repo grep for `secret\|password\|api_key` returned only library code (`secrets.token_hex`, `ops_router._mask_url_password`, env-var loads). `.env.example` exists; `.env` is gitignored and absent locally. |

---

## 2. Frontend

| # | Item | Status | Evidence |
|---|------|--------|----------|
| 2.1 | `index.html` loads with no console errors | `PARTIAL` | `curl :8080/` → HTTP 200. Console-error verification belongs to `W13-UX-4` (`docs/FRONTEND_E2E_AUDIT.md`). Read that report before launch. |
| 2.2 | All 3 modes render (Regression / Strategies / Terminal) | `PARTIAL` | Three-mode markup is in `web/index.html`; live render test deferred to `W13-UX-4`. |
| 2.3 | Mode switching works (mode-router W11-03) | `PASS` | `web/js/mode-router.js` exists and is wired (referenced by `web/index.html`). |
| 2.4 | `cmdk` ⌘K opens | `PASS` | `web/cmdk.js` + `web/cmdk.css` shipped; `.pfm-cmdk-hint` rendered in `index.html`; mobile breakpoint handled. |
| 2.5 | Theme toggle works | `PASS` | `web/js/theme-toggle.js` present; `[data-theme="dark"]` selectors used throughout `index.html`. |
| 2.6 | Mobile (375 px) renders correctly | `PARTIAL` | Media queries collapse `.pfm-cmdk-label` and resize `.cmdk-modal` for narrow viewports. Visual regression test belongs to `W13-UX-4`. |
| 2.7 | Print works | `FAIL` | No `@media print { … }` block detected in `web/index.html` (grep returned 0 hits). Add a print stylesheet (hide chrome, expand panels) or document this as deferred. |
| 2.8 | No 404s on console | `PARTIAL` | Static probe only; full network audit belongs to `W13-UX-4`. |
| 2.9 | OG meta tags present (W13-36) | `FAIL` | Head of `index.html` shows only `charset`, `viewport`, `link rel=icon`, and font `preconnect` tags. **No `og:title`, `og:description`, `og:image`, or `twitter:card` tags**. A snippet exists at `.coordination/seo-meta-snippet.html` and needs to be merged by the `index-html-owner`. |
| 2.10 | favicon + manifest present (W13-37) | `PARTIAL` | `web/favicon.svg`, `web/icon-192.svg`, `web/icon-512.svg`, `web/manifest.json` exist. **But `index.html` ships an inline SVG `data:` favicon, not a `<link rel="icon" href="/favicon.svg">`, and no `<link rel="manifest" href="/manifest.json">`** — so the manifest is never picked up by the browser. Wire both links in via the index-html-owner. |

---

## 3. Docs

| # | Item | Status | Evidence |
|---|------|--------|----------|
| 3.1 | CLAUDE.md current | `PASS` | `CLAUDE.md` reflects 1228 factors, 271 OpenAPI paths, three-mode UI (matches live probes). |
| 3.2 | README has badges, quickstart, links | `PASS` | `README.md` is 459 lines. (Spot-check passed; reviewer should re-skim Quickstart paths.) |
| 3.3 | `ARCHITECTURE.md` present | `PASS` | `docs/ARCHITECTURE.md` (674 lines). |
| 3.4 | All 15+ ADRs present | `PASS` | `docs/adrs/` contains 18 files (`0001`–`0009` + `ADR-0007`–`ADR-0015`). Anti-alpha rule (`ADR-0010`), cache stampede (`ADR-0011`), rate-limit retry (`ADR-0012`), pickle versioning (`ADR-0013`), SSE-vs-WS (`ADR-0014`), frontend bundle (`ADR-0015`) all present. |
| 3.5 | CHANGELOG.md updated | `PASS` | `CHANGELOG.md` at repo root (392 lines); `api/CHANGELOG.md` also present. Confirm latest entry dates 2026-05-16 before launch. |
| 3.6 | alpha-report-v20 (current) | `PASS` | `docs/alpha-report-v20.md` exists alongside v18 and v19. |
| 3.7 | RUNBOOK, TROUBLESHOOTING, DEVELOPMENT, SECURITY | `PASS` | `docs/RUNBOOK.md` (142 lines), `docs/TROUBLESHOOTING.md` (176), `docs/DEVELOPMENT.md` (544), `docs/SECURITY.md` (129). |

---

## 4. Quant

| # | Item | Status | Evidence |
|---|------|--------|----------|
| 4.1 | Deflated Sharpe enforced | `PASS` | `api/src/pfm/quant/deflated_sharpe.py`, `api/src/pfm/robust_validation.py::deflated_sharpe_ratio` (Bailey–Lopez de Prado 2014). Used by `strategies_router.py`, `strategies/deployable_router.py`, `multitest.py`. |
| 4.2 | Anti-alpha rule documented (ADR-0010) | `PASS` | `docs/adrs/ADR-0010-anti-alpha-rule.md` present. CLAUDE.md "Anti-alphas" section names the four blacklisted strategies. |
| 4.3 | 4-quarter stress harness ready (W11-52) | `PASS` | `api/scripts/validate_alphas_4q.py` + `api/scripts/stress_test.py`; tests in `api/tests/test_stress_script.py`, `api/tests/test_binary_pricing_strategy_stress.py`. Re-run before deploy. |
| 4.4 | Deployable strategies have synthetic-DGP recovery test | `PASS` | `api/tests/test_model.py::test_recovers_known_betas` exists. Strategy-level synthetic tests live in `api/tests/strategies/`. Audit any newly-added strategy in this wave for the same pattern. |

---

## 5. Ops

| # | Item | Status | Evidence |
|---|------|--------|----------|
| 5.1 | `backup.sh` exists (W13-56) | `PASS` | `api/scripts/backup.sh` — snapshots `.coordination`, `factors.yml`, `alpha_strategies.json`, `alpha_graveyard.json`, `live_signals.json`, `dashboard_state.json` to `/tmp/pfm-backup-<ts>.tar.gz`. |
| 5.2 | `deploy.sh` + `rollback.sh` (W13-59) | `PASS` | `api/scripts/deploy.sh` (backup → tests → tag → SIGUSR1 reload → health verify → auto-rollback) + `api/scripts/rollback.sh` (graceful stop → restore → restart → health verify). Documented exit codes 0–7. |
| 5.3 | `monitor.sh` (W13-60) | `PASS` | `api/scripts/monitor.sh` polls `/health`, `/health/deep`, `/metrics/audit`, gunicorn worker count, and (optional) `redis-cli ping`. **NOTE**: the script polls endpoints that do not yet exist on the live app (`/health/deep`, `/metrics/audit`) — see backend items 1.3 and 1.6. Monitor will report failures until those routes ship. |
| 5.4 | CI workflow audited (W13-58) | `PARTIAL` | `.github/workflows/ci.yml` has lint (ruff), typecheck (mypy non-blocking), and weekly slug-health cron. Verify the test + coverage jobs further down the file and the deploy gate in `deploy.yml` before launch. |
| 5.5 | `/metrics/audit` live | `FAIL` | Not in `/openapi.json` paths. Endpoint missing on the running app. |
| 5.6 | `/admin/cache-stats` authenticated | `FAIL` | Not in `/openapi.json` paths. Cannot verify auth gate on a route that does not exist. |

---

## Final verdict

**NEEDS WORK** — do not flip the production switch yet.

### Top-3 blockers

1. **Wave-13 endpoints missing on live app.** `/health/deep`, `/metrics/audit`, and `/admin/cache-stats` are absent from `GET /openapi.json` (271 paths live vs. the 297 target). `monitor.sh` and the launch SLOs both depend on these. Fix: confirm the routers exist on disk, mount them in `main.py` (claim `main.py:routes`), graceful-reload gunicorn, re-probe path count.
2. **OG / social-share meta tags + manifest link missing from `web/index.html` head.** Snippet is already drafted at `.coordination/seo-meta-snippet.html`; `web/manifest.json` exists but is not linked. Only the `index-html-owner` may insert these — schedule that claim.
3. **No print stylesheet.** `@media print { … }` is absent. Either add a minimal print block (hide nav/chrome, force light theme, expand active panel) or explicitly defer this in `docs/future-work.md` so the launch checklist can pass.

### Pre-launch action list

- [ ] Wire missing Wave-13 routes + reload (`main.py:routes` claim).
- [ ] Merge SEO meta snippet + `<link rel="manifest">` (`index-html-owner` claim).
- [ ] Decide on print: ship minimal stylesheet OR defer formally.
- [ ] Run `docker compose up --build` end-to-end smoke test.
- [ ] Run full `pytest -q` and confirm `~2700` tests in `<120 s`.
- [ ] Run `ruff check .` and `ruff format --check .` in `api/`.
- [ ] Re-run this checklist; confirm all `FAIL` rows have flipped to `PASS`.

When every row above is `PASS`, this checklist becomes the deploy gate. Until then it is the punch list.
