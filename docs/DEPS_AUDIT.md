# Dependency Security Audit — W13-LUPA-6

**Date:** 2026-05-16
**Scope:** `api/.venv` (Python 3.14.4) — pip + transitive. No `package.json` (no npm audit needed).
**Tools:** `pip-audit 2.10.0` (PyPI Advisory DB + OSV), `pip list --outdated`, `pip list --format=json`.
**Status:** AUDIT-ONLY. No `requirements.txt` / `pyproject.toml` edits performed.

## Summary

| Metric | Count |
|---|---|
| Total installed packages | 141 |
| Vulnerable packages | **1** (urllib3) |
| Distinct advisories | **2** (both GHSA, no MITRE CVE alias yet) |
| Outdated packages | 23 |
| Critical / High severity | 0 |
| Medium severity | 2 |
| npm packages | n/a (no package.json) |

## Findings (pip-audit)

### urllib3 2.6.3  →  fix in 2.7.0  *(transitive, via `requests` + `types-requests`)*

| Advisory | Aliases | Severity (est.) | Impact |
|---|---|---|---|
| **GHSA-qccp-gfcp-xxvc** (CVE-2026-44431) | — | Medium | Sensitive headers (`Authorization`, `Cookie`, `Proxy-Authorization`) leak across cross-origin redirects when using the **low-level** `ProxyManager.connection_from_url().urlopen(assert_same_host=False)` API. Our codebase uses `httpx` directly; `requests`/`urllib3` only enters via `yfinance` and `sentry-sdk`, neither of which uses the low-level proxy path. Exposure: **low**. |
| **GHSA-mf9v-mfxr-j63j** (CVE-2026-44432) | — | Medium (CWE-409 zip-bomb / DoS) | Streaming-API decompression may decode the *whole* response on the 2nd `read()` (Brotli) or after `drain_conn()` — excessive CPU/memory from compressed responses of untrusted origin. Our outbound HTTP targets are Polymarket, yfinance, Binance, Kalshi — semi-trusted. Exposure: **low-medium**. |

**Recommended action:** Pin `urllib3>=2.7.0,<3.0` in `requirements.txt`. urllib3 is not a direct dep, so add it as an explicit constraint to force the upgrade past `requests`'s loose floor.

## Outdated packages (23)

Non-security upgrades worth folding into a routine bump PR:

| Package | Installed | Latest | Notes |
|---|---|---|---|
| pandas | 2.3.3 | **3.0.3** | Major version — review breaking changes (Index API, copy-on-write default). Pin stays `<3.0` per `requirements.txt`. |
| mypy | 2.0.0 | 2.1.0 | Patch — safe. |
| numpy | 2.4.4 | 2.4.5 | Patch — safe. |
| pydantic | 2.13.3 | 2.13.4 | Patch — safe. |
| pydantic-settings | 2.14.0 | 2.14.1 | Patch — safe. |
| pydantic_core | 2.46.3 | 2.46.4 | Patch — safe. |
| uvicorn | 0.46.0 | 0.47.0 | Minor — `requirements.txt` cap is `<0.33` (stale). |
| requests | 2.33.1 | 2.34.2 | Minor — pulls in urllib3 2.7.x fix. |
| idna | 3.13 | 3.15 | Minor. |
| ruff | 0.15.12 | 0.15.13 | Patch. |
| coverage | 7.13.5 | 7.14.0 | Minor. |
| hypothesis | 6.152.4 | 6.152.7 | Patch. |
| fonttools | 4.62.1 | 4.63.0 | Minor (matplotlib dep). |
| markdown-it-py | 4.0.0 | 4.2.0 | Minor. |
| eth-keyfile | 0.8.1 | 0.9.1 | Minor (Polymarket / web3 transitive). |
| parsimonious | 0.10.0 | 0.11.0 | Minor (web3 transitive). |
| ast_serialize | 0.3.0 | 0.4.0 | Minor. |
| librt | 0.10.0 | 0.11.0 | Minor. |
| pytz | 2026.1.post1 | 2026.2 | Patch — TZ data refresh. |
| types-PyYAML | 6.0.12.20260508 | 6.0.12.20260510 | Stubs. |
| types-requests | 2.33.0.20260508 | 2.33.0.20260513 | Stubs. |
| urllib3 | **2.6.3** | **2.7.0** | **Security — see above.** |
| pip | 26.1 | 26.1.1 | Patch. |

## Recommended pins for `api/requirements.txt`

Add / tighten:

```text
# Security: GHSA-qccp-gfcp-xxvc & GHSA-mf9v-mfxr-j63j (2026-05).
urllib3>=2.7.0,<3.0

# Loosen stale caps blocking patch upgrades:
uvicorn[standard]>=0.32,<0.48    # was <0.33 — blocks 0.47
requests>=2.34,<3.0              # currently unpinned (transitive)
```

For `pyproject.toml` `[project.dependencies]`, no change needed — the runtime floors there (`fastapi>=0.116`, `pydantic>=2.9`, etc.) are open-ended and already permit the patched versions.

## CI hardening (suggested follow-up, out of scope for this audit)

1. Add `pip-audit --strict` to `.github/workflows/ci.yml` (a 4th job after `tests`, `ruff`, `mypy`). Cache the OSV DB to keep CI fast (~10 s warm).
2. Optional: Dependabot or Renovate weekly for security-only PRs (`open-pull-requests-limit: 5`, group patch updates).
3. Re-run this audit any time `requirements.txt` is touched.

## Artifacts

- Inventory snapshot: `/tmp/pip_inventory.json` (141 packages)
- Raw pip-audit JSON: `/tmp/pip_audit.json` (2 vulns)
- Outdated JSON: `/tmp/outdated.json` (23 packages)

## Sign-off

- No critical or high-severity vulnerabilities.
- 2 medium-severity advisories in 1 transitive package (`urllib3`), low real-world exposure given our HTTP-client mix (`httpx` primary).
- 23 routine version bumps available; only `urllib3 → 2.7.0` is security-driving.
