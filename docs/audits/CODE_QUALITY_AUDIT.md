# Code Quality Audit — W13-LUPA-4

**Scope:** `api/src/pfm/` — full ruff lint + format check, plus mypy 2.0.0 type-check on the four priority modules: `pfm.model`, `pfm.regression_core`, `pfm.cache_pool`, `pfm.main`.

**Date:** 2026-05-16
**Tools:** ruff (via `.venv`), mypy 2.0.0. **pyright not installed** in this venv (no module `pyright`).
**Mode:** Audit-only. No code changed.

## Headline numbers

| Tool                          | Result                                                                   |
| ----------------------------- | ------------------------------------------------------------------------ |
| `ruff check src/pfm/`         | **169 errors** across the package (106 auto-fixable with `--fix`)        |
| `ruff format --check src/pfm/`| **243 files would be reformatted** (68 already formatted; 311 total)     |
| `mypy` (4 priority modules)   | **13 errors in 3 files** (`cache_pool.py`, `model.py`, `main.py`)        |
| `pyright`                     | Not available in `api/.venv` — install if you want a second opinion      |

Raw outputs saved to `/tmp/ruff-out.txt`, `/tmp/ruff-fmt.txt`, `/tmp/mypy-out.txt`.

## Per-priority-module summary

### `pfm/model.py` (priority — model math)
- Ruff: **0 lint errors**. Format-only: would be reformatted.
- Mypy: **1 error** — `model.py:42` `[return-value]` Incompatible return value: `np.log(...)` returns `ndarray`, the function is annotated as returning `pd.Series`. Real bug-shaped issue: the function lies about its return type. Caller flow may not hit it (pandas dispatches), but the annotation is wrong. **Fix:** wrap result in `pd.Series` or change annotation to `pd.Series | np.ndarray`.

### `pfm/regression_core.py` (priority — OLS/HAC)
- Ruff: **0 lint errors**. Format-only: would be reformatted.
- Mypy: **0 errors**. Cleanest of the four.

### `pfm/cache_pool.py` (priority — Redis pool)
- Ruff: **5 errors** (4 RUF100 unused-noqa for non-enabled `BLE001`; 1 SIM105 suppressible-exception at line 368).
- Mypy: **4 errors** — lines 250/276/278/369 `[union-attr]`: `self._redis` is typed `Any | None` but called without a None-guard. Either narrow with `if self._redis is not None:` before each call, or change the attribute type to `Redis` after construction and route the None case through a different code path.
- Format: would be reformatted.

### `pfm/main.py` (priority — app entrypoint)
- Ruff: **5 errors** (3 I001 unsorted-imports at lines 2340/2411/2622; 1 SIM105 at 1606; 1 RUF100 at 2383).
- Mypy: **8 errors**:
  - `main.py:33` `[misc]` + `[assignment]` — `BrotliMiddleware = None` after a fallback `except ImportError` clobbers the imported class symbol. **Fix:** use a `try/except` that assigns to a different name (e.g. `BrotliMiddleware: type | None = None`) or guard with `if TYPE_CHECKING`.
  - `main.py:927` `[arg-type]` — `float(df.iloc[-1, 0])`: pandas scalar from `.iloc` is too wide for `float()`. Cast via `pd.to_numeric` or `np.float64()` first.
  - `main.py:1035`, `main.py:1139` `[no-redef]` — `prices` shadow-redefined in nested branches (lines 1010 / 1116 earlier in same scope). Rename inner shadows.
  - `main.py:1377/1378` `[assignment]` — `_clob_lock_key`/`_clob_lock_token` declared as `str`/`bytes` but assigned `Any | None` from `getattr`. Either widen to `str | None` / `bytes | None`, or assert non-None after the getattr.
  - `main.py:2361` `[unused-ignore]` — `# type: ignore` comment is no longer needed; remove.
- Format: would be reformatted.

## Ruff rule frequency (top 15, full package)

| Count | Rule    | Description                                                | Auto-fix |
| ----- | ------- | ---------------------------------------------------------- | -------- |
| 24    | UP017   | Use `datetime.UTC` instead of `datetime.timezone.utc`      | yes      |
| 23    | RUF100  | Unused `noqa` directives (rules not enabled in config)     | yes      |
| 16    | RUF022  | `__all__` not sorted                                       | yes      |
| 14    | UP037   | Remove quotes from already-PEP-604 annotations             | yes      |
| 10    | E741    | Ambiguous variable name (`l`, `I`, `O`)                    | no       |
| 8     | F401    | Unused imports                                             | yes      |
| 8     | N806    | Non-lowercase variable in function (`K`, `W`, `L`)         | no       |
| 6     | E402    | Module-level import not at top of file                     | no       |
| 6     | UP035   | Deprecated import (`typing.Callable` → `collections.abc`)  | yes      |
| 5     | PTH118  | `os.path.join` should be `Path / "x"`                      | no       |
| 5     | UP041   | `asyncio.TimeoutError` → builtin `TimeoutError`            | yes      |
| 4     | F841    | Unused local variable                                      | no       |
| 4     | PLR1730 | `if a > b: a = b` → `a = min(a, b)`                        | yes      |
| 4     | SIM105  | Use `contextlib.suppress(...)` instead of `try/except/pass`| no       |
| 3     | B905    | `zip()` without `strict=`                                  | no       |

**Lint hotspots** (most errors per file): `quant/realized_vol.py` (11), `pricing/binary_models.py` (11), `admin/cache_invalidate_router.py` (10), `arb_matching/event_similarity.py` (9), `strategies/calendar_lambda_ratio.py` (7), `alpha_hub_router.py` (7), `alerts/configure_router.py` (7), `ws_live_router.py` (6), `terminal/event_impact_router.py` (6), `arb/quality_router.py` (6).

## Top 20 prioritized issues

Ranked by impact: real bugs / latent crashes first, then style debt that touches the priority modules, then easy fleet-wide auto-fixes.

| # | Location                              | Rule / category     | Sev   | Recommended fix |
|---|----------------------------------------|---------------------|-------|------------------|
| 1 | `pfm/main.py:33`                       | mypy `[misc][assignment]` | HIGH | `BrotliMiddleware = None` after `except ImportError` overwrites the class symbol with `None`. Use `BrotliMiddleware: type[Any] | None = None` in the except branch or guard usages with `if BrotliMiddleware is not None`. |
| 2 | `pfm/cache_pool.py:250,276,278,369`    | mypy `[union-attr]` | HIGH | `self._redis` typed as `Any \| None` called without None-check 4× — real `AttributeError` risk if `_redis` ever falls back to None mid-flight. Add `if self._redis is None: return` early, or type as `Redis` after construction. |
| 3 | `pfm/model.py:42`                      | mypy `[return-value]` | HIGH | `logit()` annotated to return `pd.Series` but returns `np.ndarray` from `np.log(...)`. Wrap with `pd.Series(..., index=clipped.index)` or update annotation. Affects downstream type-narrowing in `regression_core`. |
| 4 | `pfm/main.py:1035,1139`                | mypy `[no-redef]`   | MED   | `prices: dict[str, float]` redeclared inside conditional after first declaration in same function. Rename inner shadow (e.g. `peer_prices`) to avoid name reuse and reader confusion. |
| 5 | `pfm/main.py:927`                      | mypy `[arg-type]`   | MED   | `float(df.iloc[-1, 0])` — pandas scalar too wide. Use `float(pd.to_numeric(df.iloc[-1, 0]))` or `float(np.asarray(...).item())`. Crashes on Timedelta/Timestamp cells if column type ever drifts. |
| 6 | `pfm/main.py:1377,1378`                | mypy `[assignment]` | MED   | `_clob_lock_key`/`_clob_lock_token` typed `str`/`bytes` but `getattr(..., None)` returns `Any \| None`. Widen the declarations to `str \| None` / `bytes \| None` and add a None-guard before use. |
| 7 | `pfm/portfolio_import_router.py:230` (×2) | B023             | MED   | `B023` function-uses-loop-variable: closures capture `raw` by reference inside a loop — classic late-binding bug. Bind via default arg: `lambda x, raw=raw: ...`. |
| 8 | `pfm/factors_correlation_matrix_router.py:412` | F841        | MED   | `n` assigned but never used — likely a forgotten check (`if n < min_obs`). Either delete or finish the validation. Worth a manual look before removal. |
| 9 | `pfm/main.py:2340,2411,2622`           | I001                | LOW   | Three unsorted import blocks inside function bodies (deferred imports). Run `ruff check --fix --select I001` after confirming nothing else in main.py is mid-edit. |
| 10| `pfm/main.py:1606`, `pfm/cache_pool.py:368` | SIM105         | LOW   | `try / except / pass` → `with contextlib.suppress(Exception):`. Clearer intent, same behavior. |
| 11| `pfm/alpha_hub_router.py:37-43`        | E402 (×6)           | LOW   | Module-level imports below other code. Usually intentional (lazy import to break cycles or guard slow ones). If intentional, add `# noqa: E402` with a comment explaining why; otherwise move to top. |
| 12| `pfm/quant/realized_vol.py:105,107` + 8 more | E741          | LOW   | Ambiguous names `l`, `K`, `W`, `L` — math-paper variables. In numeric code these are conventional; consider per-file `# ruff: noqa: E741, N806` rather than renaming. Not auto-fixable. |
| 13| `pfm/pricing/binary_models.py:264,276,452,454` | N806        | LOW   | Same family as #12 — `K`, `W` are option-pricing conventions. Per-module noqa is cleaner than renaming to `strike_k`. |
| 14| 24× `UP017` package-wide               | UP017               | LOW   | `datetime.timezone.utc` → `datetime.UTC` (3.11+). Auto-fixable; safe one-shot `ruff check --fix --select UP017`. |
| 15| 23× `RUF100` package-wide              | RUF100              | LOW   | Unused `# noqa: BLE001` / `# noqa: SLF001` directives — rules aren't enabled in `pyproject.toml`. Auto-fixable; check the config first to confirm we don't *want* those rules enabled instead of stripping the noqas. |
| 16| 16× `RUF022` package-wide (incl. `cache_pool.py` neighbours) | RUF022 | LOW   | `__all__` lists unsorted. Pure style; auto-fix. |
| 17| 14× `UP037` package-wide               | UP037               | LOW   | Quoted annotations no longer needed under `from __future__ import annotations`. Auto-fix. |
| 18| 8× `F401` package-wide                 | F401                | LOW   | Unused imports. Auto-fix, but skim first — re-exports sometimes look unused. |
| 19| 5× `UP041` (incl. `ws_live_router.py`, `health_deep_router.py`) | UP041 | LOW | `asyncio.TimeoutError` → builtin `TimeoutError` (aliased since 3.11). Auto-fix safe. |
| 20| `ruff format --check` — **243 / 311 files** would reformat | format | LOW | Run `ruff format src/pfm/` in a dedicated wave-N PR (no semantic diff). Coordinate via `active-edits.json` — touches almost every file and will clobber concurrent edits. **Do this when no other waves are live.** |

## Format-check observations

- **243 of 311 files** would be reformatted — i.e. ~78% of the package. This suggests `ruff format` has either never been run or the config was tightened after the codebase was written. The four priority modules (`model.py`, `regression_core.py`, `cache_pool.py`, `main.py`) all need reformatting.
- Pure-whitespace diff — zero semantic risk — but the volume means a single PR will look enormous and conflict with every active-edits claim. Recommend running once during a quiet window and committing under a dedicated "fleet format" task.

## Mypy notes

- Run config: `mypy --ignore-missing-imports --no-incremental` on the four priority files only. Strictness was kept at defaults; turning on `--strict` would surface hundreds of `Any`-related warnings, mostly from third-party stubs.
- **`regression_core.py` was clean.** This is the core OLS/HAC module — good sign that the math layer's type discipline is solid.
- **`cache_pool.py`'s 4 union-attr errors are the most concrete latent bug** in this audit: a deliberate `Any | None` annotation that's never narrowed before each `.get/.set/.delete` call. Worth fixing even if the runtime path always populates `_redis`.

## Recommended remediation sequence

1. **One commit:** ruff auto-fixes for `UP017`, `RUF100`, `RUF022`, `UP037`, `F401`, `UP041`, `UP035`, `PLR1730`, `I001`, `C420`, `UP034`, `PLR5501` — all auto-fixable, ~106 of 169 issues. Run tests.
2. **One commit:** fix the 13 mypy errors (issues #1–#6 above). These are real type bugs in priority modules.
3. **One commit (quiet window only):** `ruff format src/pfm/`. 243-file diff; needs exclusive lock on the package.
4. **One commit:** manual review of the non-auto-fixable lint debt (`E741`, `N806`, `E402`, `B023`, `F841`, `SIM105`, `B905`, `PTH*`) — many are deliberate or naming-convention disagreements; consider per-file `# ruff: noqa` instead of rewrites in math-heavy modules.

## What was not checked

- **pyright** — not installed in `api/.venv`. Install via `.venv/bin/pip install pyright` to get a second opinion on the type errors.
- **Strict mypy** (`--strict`) — would surface much more, mostly noise.
- **Per-test-file lint** — scope was `src/pfm/` only, not `tests/`.
- **Cyclomatic complexity / dead code** (radon, vulture) — out of scope for this audit.
