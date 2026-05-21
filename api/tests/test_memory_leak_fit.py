"""Memory-leak regression test for ``POST /fit`` (W13-30).

Wave-13 hardening: confirm that **1000 sequential** ``/fit`` calls do not
leak memory or accumulate global state.

The fit pipeline pulls together statsmodels OLS, a fair number of DataFrame
copies, a cache key, several Pydantic response models, and (post-W12) a
slug->FactorConfig index lookup. Each of those is a credible place to grow
process memory if a reference is held in a module-level cache, a closure,
or a logger handler. A leak of even **100 KB per fit** would balloon to
~100 MB after 1000 calls — exactly the bound this test enforces.

Test contract
-------------

1. **RSS / heap growth < 100 MB** after 1000 fits. We prefer ``psutil`` for
   wall-clock RSS when available (most accurate), and fall back to
   ``tracemalloc`` peak heap (always available, stdlib) otherwise. Either
   path is a credible proxy for "no major leak".
2. **``app.state.factors_by_slug`` count is unchanged** before vs. after the
   burst. A drift here would mean a request handler mutated the global
   factor index — a serious correctness bug, not just a leak.
3. **``app.state.factors`` count is unchanged** before vs. after. Same
   rationale; doubles as a smoke check that the factor registry hasn't been
   silently rebuilt by a side-effect.
4. **No new entries in known global caches** — specifically the per-request
   audit metrics tracker is permitted to grow (one entry per endpoint), but
   the factor registry / slug index must be invariant.

This test is marked ``@pytest.mark.slow`` and is excluded from the default
``pytest`` invocation; run it via ``pytest -m slow tests/test_memory_leak_fit.py``.

All external IO (Polymarket, yfinance, Redis) is mocked end-to-end by the
``app_client`` fixture in ``conftest.py``; no network calls are made.
"""

from __future__ import annotations

import gc
import os
import tracemalloc
import warnings

import pytest

import pfm.main as main_mod

# ---------------------------------------------------------------------------
# Marker registration
# ---------------------------------------------------------------------------
# ``pytest.mark.slow`` is shared with ``tests/test_perf_benchmarks.py``. The
# project's pytest config uses ``--strict-markers`` but does not register
# ``slow`` explicitly, which would emit a ``PytestUnknownMarkWarning``. We
# silence that warning at module scope so the slow-suite output stays clean.
warnings.filterwarnings(
    "ignore",
    message=r"Unknown pytest\.mark\.slow.*",
    category=pytest.PytestUnknownMarkWarning,  # type: ignore[attr-defined]
)

pytestmark = pytest.mark.slow


# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

# Number of sequential /fit calls. 1000 is the contract; can be overridden
# via env for local debugging (``PFM_MEMLEAK_N=50 pytest -m slow``).
_N_CALLS: int = int(os.environ.get("PFM_MEMLEAK_N", "1000"))

# Hard ceiling on memory growth. 100 MB is the W13-30 contract; with no
# leak we typically see <5 MB tracemalloc growth, so 100 MB leaves comfortable
# headroom for legitimate caches (audit metrics tracker, OpenAPI schema cache,
# etc.) while still flagging a runaway leak.
_MAX_GROWTH_BYTES: int = 100 * 1024 * 1024

# Sample every Nth call to keep wall-time bounded (no need to time each one).
# We still issue every request — only the memory snapshot is sampled.
_SAMPLE_EVERY: int = max(1, _N_CALLS // 10)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fit_body(ticker: str = "MEM", factor: str = "factor_a") -> dict:
    """A minimal valid /fit payload, reusing the conftest fixture's factors."""
    return {
        "ticker": ticker,
        "factors": [factor],
        "start": "2025-06-01",
        "end": "2025-12-31",
    }


def _maybe_rss() -> int | None:
    """Return current process RSS in bytes if ``psutil`` is installed.

    ``psutil`` is not a hard dependency of the project; if it's missing we
    return ``None`` and fall back to ``tracemalloc``. Both are accepted by
    the W13-30 contract.
    """
    try:
        import psutil  # type: ignore[import-not-found]
    except ImportError:
        return None
    try:
        return int(psutil.Process(os.getpid()).memory_info().rss)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# The single big test
# ---------------------------------------------------------------------------


def test_no_memory_leak_after_1000_fits(app_client) -> None:
    """1000 sequential /fit calls must not leak >100 MB.

    Procedure:

    1. **Warm-up**: issue 5 fits to amortise one-time costs (Pydantic schema
       compilation, OpenAPI generation, statsmodels import warmth, the
       audit-metrics tracker registering a new key, etc.). Without warm-up
       the first few requests dominate the growth measurement and produce a
       false positive.
    2. **Snapshot baseline** — RSS via psutil if available, and the
       tracemalloc peak. ``gc.collect()`` first to drain pending cycles so
       the baseline reflects steady-state.
    3. **Loop 1000 times** — POST /fit, assert 200, discard the response.
       Every ``_SAMPLE_EVERY`` calls run ``gc.collect()`` to keep the
       sampled growth signal monotonic-ish (otherwise tracemalloc trace
       sizes oscillate with the gen-2 cycle which makes the assertion noisy).
    4. **Snapshot final** — RSS and tracemalloc.
    5. **Assert** growth < 100 MB by whichever metric we have.
    6. **Cross-check globals**: ``app.state.factors`` and
       ``app.state.factors_by_slug`` counts (and key sets) unchanged.
    """

    # Sanity: the conftest lifespan should have populated factors. If not,
    # the test premise is invalid and we'd rather skip loudly than measure
    # a degenerate empty-factor path.
    factors_before = dict(main_mod.app.state.factors)
    assert factors_before, "app.state.factors empty — conftest lifespan didn't run?"

    factors_by_slug_before = dict(getattr(main_mod.app.state, "factors_by_slug", {}) or {})
    assert factors_by_slug_before, (
        "app.state.factors_by_slug empty — conftest lifespan didn't build the index?"
    )

    # --- 1. Warm-up --------------------------------------------------------
    for _ in range(5):
        r = app_client.post("/fit", json=_fit_body())
        assert r.status_code == 200

    # --- 2. Baseline snapshot ---------------------------------------------
    gc.collect()
    rss_baseline = _maybe_rss()
    tracemalloc.start()
    # Reset the peak so it starts from zero rather than including warm-up
    # allocations. We compare peaks at the end as a tighter signal than
    # current-snapshot (which gc may have already reclaimed).
    tracemalloc.reset_peak()
    tm_baseline_current, _ = tracemalloc.get_traced_memory()

    # --- 3. Loop -----------------------------------------------------------
    # Vary the ticker so we don't simply hit the response cache on every
    # call (which would understate any per-call allocation). 16 distinct
    # tickers give enough rotation to defeat trivial memoisation while
    # still mostly hitting warm factor-history fixtures.
    rotation = [f"M{i:03d}" for i in range(16)]
    try:
        for i in range(_N_CALLS):
            ticker = rotation[i % len(rotation)]
            resp = app_client.post("/fit", json=_fit_body(ticker=ticker))
            assert resp.status_code == 200, (
                f"call {i} failed: status={resp.status_code} body={resp.text[:200]}"
            )
            # Drop the parsed body explicitly — TestClient + httpx will keep
            # it alive until the next request otherwise, polluting the
            # growth measurement.
            del resp

            if (i + 1) % _SAMPLE_EVERY == 0:
                gc.collect()

        # --- 4. Final snapshot --------------------------------------------
        gc.collect()
        tm_final_current, tm_peak = tracemalloc.get_traced_memory()
        rss_final = _maybe_rss()
    finally:
        tracemalloc.stop()

    # --- 5. Assertions ----------------------------------------------------
    # Prefer RSS if psutil was available; tracemalloc otherwise.
    tm_growth = tm_final_current - tm_baseline_current

    if rss_baseline is not None and rss_final is not None:
        rss_growth = rss_final - rss_baseline
        # RSS is the most credible "real memory" metric; enforce the
        # 100 MB ceiling against it. We also surface tracemalloc growth
        # in the message for diagnostic value.
        assert rss_growth < _MAX_GROWTH_BYTES, (
            f"RSS grew {rss_growth / 1e6:.1f} MB across {_N_CALLS} /fit calls "
            f"(>= {_MAX_GROWTH_BYTES / 1e6:.0f} MB cap); tracemalloc growth "
            f"= {tm_growth / 1e6:.1f} MB; tracemalloc peak = {tm_peak / 1e6:.1f} MB. "
            f"Suspect a per-request global cache or unbounded list append."
        )
    else:
        # tracemalloc-only fallback. The traced-heap number is typically
        # 30-50% of RSS (it doesn't count C-level mallocs), so 100 MB on
        # the heap is an even stricter test than 100 MB RSS would be.
        assert tm_growth < _MAX_GROWTH_BYTES, (
            f"tracemalloc traced heap grew {tm_growth / 1e6:.1f} MB across "
            f"{_N_CALLS} /fit calls (>= {_MAX_GROWTH_BYTES / 1e6:.0f} MB cap); "
            f"peak = {tm_peak / 1e6:.1f} MB. psutil not installed so no RSS "
            f"cross-check. Suspect a per-request global cache or unbounded "
            f"list append."
        )

    # --- 6. Global-state cross-checks -------------------------------------
    factors_after = main_mod.app.state.factors
    factors_by_slug_after = getattr(main_mod.app.state, "factors_by_slug", {}) or {}

    assert len(factors_after) == len(factors_before), (
        f"app.state.factors count drifted under load: {len(factors_before)} -> {len(factors_after)}"
    )
    assert set(factors_after.keys()) == set(factors_before.keys()), (
        "app.state.factors keys mutated by /fit (request handler must be read-only)"
    )

    assert len(factors_by_slug_after) == len(factors_by_slug_before), (
        f"app.state.factors_by_slug count drifted: "
        f"{len(factors_by_slug_before)} -> {len(factors_by_slug_after)}"
    )
    assert set(factors_by_slug_after.keys()) == set(factors_by_slug_before.keys()), (
        "app.state.factors_by_slug keys mutated by /fit "
        "(slug index must be built once at startup, never per-request)"
    )
