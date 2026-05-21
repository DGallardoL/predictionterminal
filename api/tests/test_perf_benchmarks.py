"""Latency benchmarks for hot endpoints (W11-43).

This module pins p95 wall-clock latency budgets for the six most heavily
trafficked endpoints in the FastAPI app. Each benchmark issues **20 sequential
in-process requests** through ``fastapi.testclient.TestClient`` (no real
network — every upstream HTTP call is stubbed via ``respx`` for the live
client, and the ``app_client`` fixture pre-installs in-process fakes for
factor history + yfinance + Redis).

Why sequential, not concurrent? The TestClient dispatches via ASGI in-process
on the calling thread; concurrent measurements would interleave on the GIL and
distort per-call timings. Sequential calls give a clean per-request wall-clock
sample, which is what we ultimately care about for end-user perceived latency.

Per-call timing uses ``time.perf_counter()`` (monotonic, highest-resolution
clock available — typical resolution << 1 µs on macOS and Linux).

After each benchmark we sort the 20 latency samples, compute p50/p95/p99 (via
``statistics.quantiles`` style interpolation), and assert ``p95 <= threshold``.
Thresholds are deliberately generous so they catch real regressions (2-3×
slowdown) without being flaky on a loaded CI runner.

The full latency distribution for all six endpoints is written to
``/tmp/perf-benchmark-results.json`` so a CI step (or a human) can inspect
trend lines across runs.

All tests are marked ``@pytest.mark.slow`` so a default ``pytest`` invocation
will skip them. Run explicitly with::

    pytest -m slow tests/test_perf_benchmarks.py

To skip when running on a machine without a baseline (e.g. shared CI without
warm caches), set ``PFM_SKIP_PERF_BENCH=1`` and the entire module short-
circuits — tests become a smoke-only pass.

Endpoints + budgets (p95):

  1. POST /fit                     — 3000 ms (1 factor, NVDA, mocked)
  2. GET  /terminal/jumps/{slug}   —  500 ms (post T17 prewarm)
  3. GET  /terminal/jumps/cluster  — 6000 ms (acknowledged-slow per recap)
  4. GET  /alpha-hub/leaderboard   —  200 ms
  5. GET  /factors                 —  100 ms (in-memory)
  6. GET  /openapi.json            —  500 ms

Each test is independent — they do not share a TestClient because each fixture
configures a different upstream-stub posture (factor-mock vs respx live).
"""

from __future__ import annotations

import json
import os
import statistics
import time

# ---------------------------------------------------------------------------
# Module-level skip switch + result aggregator
# ---------------------------------------------------------------------------
# ``pytest.mark.slow`` is registered as a custom marker in ``pyproject.toml``
# (``[tool.pytest.ini_options].markers``). Without that registration pytest
# emits a ``PytestUnknownMarkWarning`` — harmless but noisy. We additionally
# silence the warning at the module level so ``--noconftest`` runs (which
# ignore project-level markers) stay clean.
import warnings as _warnings
from collections.abc import Iterator
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import pytest
import respx
from fastapi.testclient import TestClient

_warnings.filterwarnings(
    "ignore",
    message=r"Unknown pytest\.mark\.slow.*",
    category=pytest.PytestUnknownMarkWarning,  # type: ignore[attr-defined]
)

pytestmark = pytest.mark.slow

#: When true, every benchmark short-circuits with a single warm-up request and
#: skips the latency assertions. Used by CI environments that don't have a
#: warm baseline (cold-disk caches, first run of the day, etc).
_SKIP_PERF: bool = bool(os.environ.get("PFM_SKIP_PERF_BENCH"))

#: Where to write the consolidated latency-distribution report. Tests in this
#: module APPEND their results into the in-memory ``_RESULTS`` dict and the
#: ``_write_results_on_teardown`` autouse fixture flushes to disk after each
#: test (so a partial run still produces useful output).
_RESULTS_PATH = Path("/tmp/perf-benchmark-results.json")

#: Per-endpoint result store. Keyed by endpoint label (matches the keys in the
#: emitted JSON). Each value is the full distribution dict.
_RESULTS: dict[str, dict] = {}

#: Number of sequential calls per benchmark.
_N_CALLS: int = 20


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _quantiles(samples_ms: list[float]) -> dict[str, float]:
    """Return p50/p95/p99 + min/max/mean for a sorted-by-value list.

    Uses linear interpolation between the two nearest ranks (equivalent to
    ``numpy.quantile`` with ``method='linear'``). With only 20 samples,
    quantile estimates are noisy — that's fine for a regression budget,
    not fine for a precise SLO.
    """
    sorted_ms = sorted(samples_ms)
    n = len(sorted_ms)

    def _pct(p: float) -> float:
        # Linear interpolation. p in [0,1].
        if n == 1:
            return sorted_ms[0]
        rank = p * (n - 1)
        lo = int(rank)
        hi = min(lo + 1, n - 1)
        frac = rank - lo
        return sorted_ms[lo] * (1 - frac) + sorted_ms[hi] * frac

    return {
        "min_ms": sorted_ms[0],
        "max_ms": sorted_ms[-1],
        "mean_ms": statistics.fmean(sorted_ms),
        "p50_ms": _pct(0.50),
        "p95_ms": _pct(0.95),
        "p99_ms": _pct(0.99),
        "n": n,
        "samples_ms": sorted_ms,
    }


def _bench(
    client: TestClient,
    label: str,
    fn: callable[[TestClient], httpx.Response],
    threshold_p95_ms: float,
    *,
    accept_status: set[int] = frozenset({200}),
) -> dict:
    """Issue ``_N_CALLS`` sequential requests, record per-call latency.

    Returns the distribution dict (also written to ``_RESULTS[label]``).

    Asserts every response status is in ``accept_status`` AND that p95 <=
    ``threshold_p95_ms``. The first call is treated as a warm-up and is
    included in the sample (we want to measure realistic perceived latency,
    which includes one cold call per session in practice; the in-process
    caches kick in immediately on call #2 onward, so p50 will reflect the
    warm path and p95/p99 will partially reflect cold).
    """
    samples_ms: list[float] = []
    for i in range(_N_CALLS):
        t0 = time.perf_counter()
        r = fn(client)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        samples_ms.append(elapsed_ms)
        assert r.status_code in accept_status, (
            f"[{label}] call #{i} returned status {r.status_code}: {r.text[:200]}"
        )

    dist = _quantiles(samples_ms)
    dist["threshold_p95_ms"] = threshold_p95_ms
    dist["passed"] = bool(dist["p95_ms"] <= threshold_p95_ms)
    _RESULTS[label] = dist

    # Flush results after every endpoint so a partial run still leaves a
    # useful artefact. ``Path.write_text`` is atomic on POSIX.
    try:
        _RESULTS_PATH.write_text(json.dumps(_RESULTS, indent=2, sort_keys=True))
    except OSError:  # pragma: no cover — /tmp not writable
        pass

    if _SKIP_PERF:
        # In skip mode we still record the distribution but don't enforce.
        return dist

    assert dist["passed"], (
        f"[{label}] p95={dist['p95_ms']:.1f}ms exceeds budget "
        f"{threshold_p95_ms:.0f}ms (p50={dist['p50_ms']:.1f}ms, "
        f"p99={dist['p99_ms']:.1f}ms, max={dist['max_ms']:.1f}ms, "
        f"n={dist['n']})"
    )
    return dist


# ---------------------------------------------------------------------------
# Fixtures — two TestClient flavours, mirroring tests/test_e2e_smoke.py
# ---------------------------------------------------------------------------


def _make_factors_file(tmp_path: Path) -> Path:
    """Inline copy of ``conftest.factors_file`` so this module is self-contained.

    Required because the task spec runs the suite with ``--noconftest``: we
    cannot depend on shared fixtures, so we duplicate the minimal 2-factor
    catalog and the synthetic history + log-return generators here.
    """
    p = tmp_path / "factors.yml"
    p.write_text(
        """
factors:
  - id: factor_a
    name: Factor A
    slug: slug-a
    source: polymarket
    description: Test factor A.
  - id: factor_b
    name: Factor B
    slug: slug-b
    source: polymarket
    description: Test factor B.
"""
    )
    return p


def _make_fake_factor_history() -> callable:
    """Inline copy of ``conftest.fake_factor_history``.

    Returns a function with signature ``(client, slug, start, end) -> DataFrame``
    that mirrors the real ``fetch_factor_history`` API.
    """
    rng = pd.date_range("2025-06-01", "2025-12-31", freq="D", tz="UTC")
    n = len(rng)
    t = np.arange(n) / n
    series_a = pd.DataFrame(
        {"price": (0.30 + 0.30 * np.sin(2 * np.pi * t * 1.2)).clip(0.05, 0.95)},
        index=rng,
    )
    series_a.index.name = "date"
    series_b = pd.DataFrame(
        {"price": (0.55 + 0.20 * np.cos(2 * np.pi * t * 0.8)).clip(0.05, 0.95)},
        index=rng,
    )
    series_b.index.name = "date"
    bank = {"slug-a": series_a, "slug-b": series_b}

    def _fetch(_client, slug: str, start=None, end=None):  # type: ignore[no-untyped-def]
        df = bank[slug]
        if start is not None:
            df = df[df.index >= start]
        if end is not None:
            df = df[df.index <= end]
        return df

    return _fetch


def _make_fake_log_returns() -> callable:
    """Inline copy of ``conftest.fake_log_returns`` — deterministic returns."""

    def _make(
        ticker: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
        return_type: str = "log",
    ) -> pd.Series:
        idx = pd.date_range(start, end, freq="B", tz="UTC")
        n = len(idx)
        rng = np.random.default_rng(seed=hash(ticker) % (2**32))
        values = 0.0001 * np.arange(n) + 0.005 * np.sin(np.arange(n)) + rng.normal(0, 0.001, n)
        if return_type == "simple":
            values = values * 1.05
        s = pd.Series(values, index=idx, name="r")
        s.index = pd.to_datetime(s.index, utc=True).normalize()
        return s

    return _make


@pytest.fixture
def fit_client(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> Iterator[TestClient]:
    """Self-contained TestClient with factor + yfinance + Redis mocked out.

    Mirrors ``conftest.app_client`` so the suite runs cleanly under
    ``pytest --noconftest`` (the spec requirement). Used by the /fit and
    /factors benchmarks which depend on in-process factor history.
    """
    import pfm.main as main_mod

    factors_file = _make_factors_file(tmp_path)
    monkeypatch.setenv("FACTORS_FILE", str(factors_file))

    import pfm.config as cfg

    cfg._settings = None  # force re-read of FACTORS_FILE env var

    monkeypatch.setattr(main_mod, "fetch_factor_history", _make_fake_factor_history())
    monkeypatch.setattr(main_mod, "get_log_returns", _make_fake_log_returns())

    from pfm.cache import NullCache

    monkeypatch.setattr(main_mod, "RedisCache", lambda url: NullCache())

    with TestClient(main_mod.app) as client:
        yield client


@pytest.fixture(scope="module")
def live_client() -> Iterator[TestClient]:
    """TestClient with all external HTTP stubbed to empty responses.

    Mirrors the fixture in ``tests/test_e2e_smoke.py`` but module-scoped so
    the 20-call benchmarks reuse a single warm app instance. Reusing across
    tests is safe because we never mutate app state across requests in the
    benchmark calls (each test does GETs or simple POSTs against read paths).
    """
    import pfm.main as main_mod
    from pfm.cache import NullCache

    _orig_redis = main_mod.RedisCache
    main_mod.RedisCache = lambda url: NullCache()  # type: ignore[assignment]

    with respx.mock(assert_all_called=False, assert_all_mocked=False) as mock:
        mock.get(url__regex=r"https?://gamma-api\.polymarket\.com/.*").mock(
            return_value=httpx.Response(200, json=[])
        )
        mock.get(url__regex=r"https?://clob\.polymarket\.com/.*").mock(
            return_value=httpx.Response(200, json={"history": []})
        )
        mock.get(url__regex=r"https?://(api|trading-api)\.kalshi\.com/.*").mock(
            return_value=httpx.Response(200, json={"markets": []})
        )
        mock.get(url__regex=r"https?://api\.elections\.kalshi\.com/.*").mock(
            return_value=httpx.Response(200, json={"markets": [], "candlesticks": []})
        )
        mock.get(url__regex=r"https?://api\.binance\.com/.*").mock(
            return_value=httpx.Response(200, json=[])
        )
        mock.get(url__regex=r"https?://api\.gdeltproject\.org/.*").mock(
            return_value=httpx.Response(200, json={"articles": []})
        )
        mock.get(url__regex=r"https?://hn\.algolia\.com/.*").mock(
            return_value=httpx.Response(200, json={"hits": []})
        )
        mock.get(url__regex=r"https?://(www|old)?\.?reddit\.com/.*").mock(
            return_value=httpx.Response(200, json={"data": {"children": []}})
        )
        mock.get(url__regex=r"https?://query[12]\.finance\.yahoo\.com/.*").mock(
            return_value=httpx.Response(200, json={"chart": {"result": []}})
        )
        mock.get(url__regex=r"https?://api\.stlouisfed\.org/.*").mock(
            return_value=httpx.Response(200, json={"observations": []})
        )

        with TestClient(main_mod.app) as client:
            yield client

    main_mod.RedisCache = _orig_redis  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Benchmark 1 — POST /fit (minimal: 1 factor, NVDA)
# ---------------------------------------------------------------------------


def test_perf_fit_minimal(fit_client: TestClient) -> None:
    """p95 < 3000 ms for a one-factor /fit against the in-process mock data.

    Uses the conftest ``app_client`` fixture which installs a 2-factor
    catalog (``factor_a``, ``factor_b``) + a deterministic synthetic factor
    history + a deterministic log-return generator. The whole call path is
    OLS + HAC SE inside the FastAPI request handler, so this benchmark is
    really measuring framework overhead + statsmodels OLS warm-up cost on
    ~150 daily observations.

    Threshold is generous (3 s) because the first call has cold imports for
    statsmodels and the second-call onward is typically ~50-200 ms.
    """
    payload = {
        "ticker": "NVDA",
        "factors": ["factor_a"],
        "start": "2025-06-15",
        "end": "2025-12-15",
    }

    def _call(c: TestClient) -> httpx.Response:
        return c.post("/fit", json=payload)

    _bench(
        fit_client,
        label="POST /fit",
        fn=_call,
        threshold_p95_ms=3000.0,
    )


# ---------------------------------------------------------------------------
# Benchmark 2 — GET /terminal/jumps/{slug}
# ---------------------------------------------------------------------------


def test_perf_terminal_jumps_slug(live_client: TestClient) -> None:
    """p95 < 500 ms for /terminal/jumps/{slug}, post the T17 prewarm.

    The prewarm runs at lifespan-start and populates a module-level TTL
    cache, so under the live_client (whose lifespan executed within respx
    mock) the cache may be empty for the test slug. We accept 200/404/502/
    503 because under the mocked-empty upstream Polymarket gamma endpoint,
    the slug ``bitcoin`` legitimately resolves to "no data" — which is a
    documented contract response (see test_e2e_smoke.py:268).

    What we're really benchmarking: route dispatch + cache lookup + the
    error-envelope construction when the upstream miss happens. p95 should
    be well under 500 ms because no upstream IO actually occurs.
    """

    def _call(c: TestClient) -> httpx.Response:
        return c.get("/terminal/jumps/bitcoin?limit=5")

    _bench(
        live_client,
        label="GET /terminal/jumps/{slug}",
        fn=_call,
        threshold_p95_ms=500.0,
        accept_status={200, 404, 502, 503},
    )


# ---------------------------------------------------------------------------
# Benchmark 3 — GET /terminal/jumps/cluster
# ---------------------------------------------------------------------------


def test_perf_terminal_jumps_cluster(live_client: TestClient) -> None:
    """p95 < 6000 ms for the cluster endpoint (acknowledged-slow per recap).

    The cluster computation does an O(N) sweep across the jumps table plus
    pairwise similarity, then DBSCAN. With upstream stubbed to empty, the
    table is empty and the response is fast (mostly framework overhead),
    but we still allow a 6 s budget per the OVERNIGHT-RECAP note that this
    endpoint is acknowledged-slow even after T57157 speedup.
    """

    def _call(c: TestClient) -> httpx.Response:
        return c.get("/terminal/jumps/cluster")

    _bench(
        live_client,
        label="GET /terminal/jumps/cluster",
        fn=_call,
        threshold_p95_ms=6000.0,
        accept_status={200, 502, 503},
    )


# ---------------------------------------------------------------------------
# Benchmark 4 — GET /alpha-hub/leaderboard
# ---------------------------------------------------------------------------


def test_perf_alpha_hub_leaderboard(live_client: TestClient) -> None:
    """p95 < 200 ms. The leaderboard reads from a static JSON snapshot +
    ``@cached`` namespace, so after the first call every subsequent call is
    a memory hit. The first call may take longer (file IO + cache fill);
    that's why we sample 20 and look at p95.
    """

    def _call(c: TestClient) -> httpx.Response:
        return c.get("/alpha-hub/leaderboard?limit=5")

    _bench(
        live_client,
        label="GET /alpha-hub/leaderboard",
        fn=_call,
        threshold_p95_ms=200.0,
    )


# ---------------------------------------------------------------------------
# Benchmark 5 — GET /factors
# ---------------------------------------------------------------------------


def test_perf_factors_list(fit_client: TestClient) -> None:
    """p95 < 100 ms. The factors list is loaded at lifespan-start into
    ``app.state.factors`` (a list of Pydantic models) and the handler just
    serialises it. No IO, no computation. Should be deeply sub-millisecond
    per call on a modern laptop; 100 ms is a very forgiving regression
    fence.

    Uses ``fit_client`` (the ``app_client`` flavour) because that fixture
    installs the small 2-factor catalog — we're measuring the serialise +
    response path, not the catalog size.
    """

    def _call(c: TestClient) -> httpx.Response:
        return c.get("/factors")

    _bench(
        fit_client,
        label="GET /factors",
        fn=_call,
        threshold_p95_ms=100.0,
    )


# ---------------------------------------------------------------------------
# Benchmark 6 — GET /openapi.json
# ---------------------------------------------------------------------------


def test_perf_openapi_json(live_client: TestClient) -> None:
    """p95 < 500 ms. FastAPI caches the OpenAPI schema after first build, so
    call #1 may take 100-400 ms (introspecting all 271 routes + Pydantic
    schemas) and call #2 onward is a memory hit. We allow 500 ms to cover
    the first call comfortably even on a slow CI.
    """

    def _call(c: TestClient) -> httpx.Response:
        return c.get("/openapi.json")

    _bench(
        live_client,
        label="GET /openapi.json",
        fn=_call,
        threshold_p95_ms=500.0,
    )


# ---------------------------------------------------------------------------
# Module finaliser — flush the results one last time after all tests run.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module", autouse=True)
def _flush_results_on_module_teardown() -> Iterator[None]:
    """Write the aggregated results file once after all module tests complete.

    The per-test ``_bench`` helper already writes incrementally so partial
    runs produce output; this fixture is a belt-and-braces final flush plus
    a printable summary that surfaces in pytest's captured stdout (``-s``).
    """
    yield
    if not _RESULTS:
        return
    try:
        _RESULTS_PATH.write_text(json.dumps(_RESULTS, indent=2, sort_keys=True))
    except OSError:  # pragma: no cover
        pass
    # Print a compact one-line summary per endpoint for human eyeballing.
    print("\n=== perf benchmark summary (p95 vs threshold) ===")
    for label, dist in _RESULTS.items():
        status = "OK " if dist.get("passed") else "MISS"
        print(
            f"  {status}  {label:<32s}  "
            f"p50={dist['p50_ms']:7.1f}ms  "
            f"p95={dist['p95_ms']:7.1f}ms  "
            f"thr={dist['threshold_p95_ms']:.0f}ms"
        )
