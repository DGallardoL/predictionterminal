# Security — Threat Model & Mitigations

This document describes the security posture of the `prediction-factor-model` POC
("Prediction Terminal"). The service is a read-mostly research dashboard that
serves a public web UI and a FastAPI backend; it does **not** custody funds,
execute trades, or store user PII. Despite the limited blast radius, the project
is graded in part on engineering discipline, so the following threat model and
mitigations have been adopted.

## 1. Threat Model

### 1.1 External adversaries

- **Scraping (LOW risk).** All `/factors`, `/terminal/*`, `/alpha-hub/*` and
  `/strategies/arb/*` endpoints are public by design — they expose the same
  Polymarket / Kalshi / yfinance data anyone can fetch upstream. We accept this.
  The cost of scraping is bounded by our caching layer (Redis L2 + per-process
  LRU), so a scraper actually warms the cache for legitimate users.
- **DDoS (MEDIUM risk).** Most endpoints are unauthenticated. A determined
  attacker could exhaust the gunicorn worker pool. We rely on upstream
  termination (Cloudflare / nginx) for L7 rate limiting; in-process throttling
  is best-effort only.
- **SSRF / injection (LOW risk).** The fit endpoint accepts factor slugs but
  resolves them through a typed Pydantic-validated registry; the underlying
  Polymarket fetcher uses a fixed allow-listed base URL.

### 1.2 Supply chain

- **pip dependencies.** 43 direct dependencies pinned in `pyproject.toml`,
  plus 200+ transitive. The largest exposure surface is `statsmodels`,
  `fastapi`, `pandas`, `httpx`. We accept the risk that an upstream package
  could be hijacked between releases; mitigation is `pip audit` on a weekly
  cadence (see §5).

### 1.3 Insider

- N/A — single-developer project. The threat model assumes Damian is the only
  privileged operator. If the project goes multi-user, this section needs to be
  rewritten before production deployment.

### 1.4 Data integrity

- **Factor catalog tampering.** `factors.yml` is the source of truth for 1228
  factor slugs. If an attacker (or a buggy migration script) corrupts this
  file, the regression endpoint will silently fit against wrong markets.
  Mitigation: file is checked into git, validated by `scripts/validate_factors.py`
  in CI, and FS-readonly in production.
- **Fake market data.** A man-in-the-middle on the Polymarket fetch could
  inject manipulated prices. Mitigation: HTTPS to Polymarket is non-negotiable
  (`httpx` verifies certs by default; we never pass `verify=False`).

## 2. Mitigations

| Threat | Mitigation | Status |
|---|---|---|
| CSRF / cross-origin abuse | CORS strict to `http://localhost:8080` and the deployed origin only; no wildcards in production | implemented |
| Polymarket rate-limit exhaustion | Token-bucket limiter in `pfm.polymarket.client` (1000 req / 10 s, matching upstream) | implemented |
| Split-brain on engine processes | Redis `SETNX` leader election in `pfm.arb_scanner` prevents double-engine writes to `dashboard_state.json` | implemented |
| Pickle format-confusion on Redis L2 | All cache values are tagged with a `pfm-pickle-v3` magic header; reads from a wrong version are dropped, not deserialized | implemented |
| Privileged admin endpoints | `/admin/*` requires `Authorization: Bearer <PFM_ADMIN_TOKEN>` (W12-18); 401 on missing or mismatched token | implemented |
| Personal data leakage | No login, no cookies, no PII collected; access logs strip IP after 7 days (handled at edge) | by-design |

## 3. Disclosure

Damian Gallardo (`sieclaudeag@gmail.com`) is the sole security contact. There
is **no bug bounty programme.** Good-faith vulnerability reports are welcome
via email and will receive an acknowledgement within 7 days. The project does
not run on production infrastructure for end users, so the realistic blast
radius of any finding is limited to the maintainer's dev environment.

## 4. Known Issues

- **No HTTPS termination at the app layer.** The FastAPI app serves plain
  HTTP; TLS is expected from an upstream reverse proxy (nginx, Cloudflare,
  Caddy). If you deploy this without a TLS terminator in front, every request
  including the admin Bearer token will traverse the wire in clear text.
- **`/metrics/audit` is unauthenticated.** It exposes aggregated counters
  (request counts, cache hit ratios, factor coverage). Low value to an
  attacker, but operationally it leaks fleet size. Consider adding HTTP Basic
  auth in front of `/metrics/*` when productionising.
- **`factors.yml` writable on disk.** In dev the file is editable by the
  service account; in production set `chmod 0444 factors.yml` and run the
  service as a user that does not own the file. This prevents tampering by a
  compromised worker.

## 5. Dependencies Audit

- Run `pip audit` against the locked environment **weekly** (Mondays). CVE
  findings with CVSS ≥ 7.0 block release; lower severity findings get a 30-day
  remediation window.
- All direct dependencies are pinned to exact versions in `pyproject.toml`
  (`==`, not `>=`). Transitive deps are pinned through a generated
  `requirements.lock` to ensure deterministic Docker builds.
- Dependabot is opt-in on the GitHub repo; weekly PRs get reviewed and merged
  after CI is green.

## 6. Secrets Management

- All secrets are read from environment variables: `PFM_ADMIN_TOKEN`,
  `REDIS_URL`, `POLYMARKET_API_KEY` (if ever introduced), `KALSHI_API_KEY`.
- **Never commit `.env`** — it is `.gitignore`d and the pre-commit hook
  (W11-50) refuses commits where a file >100 kB or matches `*.env`,
  `*credentials*`, `*secret*` slips into the index.
- Local dev uses `.env.example` as the canonical template; secrets are filled
  in by hand or via a password manager.

## 7. Logging

- Request logs strip the `Authorization` header before serialisation; this is
  enforced in the structlog processor chain (`_redact_auth_header`).
- URL query parameters whose key matches `(?i)(token|secret|password|key)` are
  masked to `***` before logging.
- Tracebacks include file/line but no request body by default; opt-in body
  logging is gated on `PFM_LOG_REQUEST_BODY=1` and refuses to start in
  production environments (detected via `PFM_ENV=prod`).

## 8. Future Work

- **OWASP ZAP scan in CI.** A weekly scheduled job runs ZAP baseline against
  the dockerised stack; findings posted to a GitHub issue.
- **Snyk dependency scan.** Replace or supplement `pip audit` with Snyk for
  richer vulnerability metadata and SBOM generation.
- **Admin port isolation.** Move `/admin/*` behind a separate gunicorn port
  (e.g. `:8001`) bound to `127.0.0.1` only, with an IP allow-list at the
  reverse proxy. This decouples admin compromise from public-API compromise.
- **Per-tenant secrets** when (if) the project grows beyond single-developer
  use. Currently any operator with `PFM_ADMIN_TOKEN` has full admin rights;
  scoped tokens with roles (read-only metrics vs cache-flush vs engine-restart)
  would be a worthwhile follow-on.
