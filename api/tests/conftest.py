"""Shared pytest fixtures."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import pfm.main as main_mod
from pfm.cache_utils import get_cache as _get_cache

_VOLATILE_CACHE_NAMESPACES = (
    "predictit_all",
    "predictit_market",
    "multi_venue_search",
    "multi_venue_concept",
    "manifold_search",
    "polymarket_search",
    "kalshi_search",
    "arb_scanner",
    "terminal_countdown",
    # Wave-N+1 (post-real-data) additions — module-level caches that leak
    # state between tests when the suite runs in one process.
    "polygon_consensus",
    "polygon_calendar",
    "earnings_whisper",
    "earnings_whisper_dashboard",
    "earnings_calendar",
    "pm_vix",
    "pm_vix_slugs",
    "live_signals_fetch",
    "live_signals",
    "decay_real",
    "alpha_hub_leaderboard",
    # 2026-05-15 latency-audit fixes added module-level caches that pull
    # state forward across respx mocks; clear between tests so each test
    # sees a clean cold path.
    "terminal_calendar",
    "terminal_orderbook_tokens",
    "terminal_orderbook_book",
    "terminal_quality",
    "terminal_prob_fan",
    "terminal_flow",
    "terminal_trades_cid",
    "terminal_trades_tape",
    # Smart-factor-picker L1 cache — perf fix 2026-05-15 added a 1 h TTL
    # process-local bucket on top of the new Redis L2. Clear between tests
    # so the existing cache-hit test in test_regression_enriched.py and the
    # new tests in test_factors_suggest_for_ticker.py see a cold path.
    "factors_suggest_for_ticker",
)


@pytest.fixture(autouse=True)
def _disable_background_prewarms(monkeypatch: pytest.MonkeyPatch) -> None:
    """Disable lifespan background tasks that make real HTTP calls.

    The jumps prewarm (and friends) iterate a hardcoded slug list against
    real Polymarket; without explicit opt-out the TestClient lifespan
    triggers minutes of upstream traffic and tests hang. Production
    workers still default to ``"1"`` (see ``pfm.main`` lifespan).
    """
    monkeypatch.setenv("PFM_JUMPS_PREWARM_ENABLED", "0")
    monkeypatch.setenv("PFM_EXTRA_PREWARMS_ENABLED", "0")
    monkeypatch.setenv("PFM_LIVE_SIGNALS_ENABLED", "0")
    monkeypatch.setenv("PFM_DECAY_REFRESH_ENABLED", "0")
    monkeypatch.setenv("PFM_PMVIX_PREWARM_ENABLED", "0")
    monkeypatch.setenv("PFM_EARNINGS_PREWARM_ENABLED", "0")
    monkeypatch.setenv("PFM_CRYPTO_WS_ENABLED", "0")
    monkeypatch.setenv("PFM_CRYPTO_5MIN_ENABLED", "0")
    monkeypatch.setenv("PFM_ARB_ENGINE_AUTOSTART", "0")
    monkeypatch.setenv("PFM_FACTOR_PREWARM_ENABLED", "0")
    # Lifespan reads this and skips the +10 curated-sentiment factor
    # injection so any TestClient gets only the fixture catalogue. Keeps
    # ``tests/test_sentiment_factor_unit.py`` (which inspects the curated
    # map directly) working since that test doesn't build a TestClient.
    monkeypatch.setenv("PFM_SUPPRESS_CURATED_SENTIMENT", "1")


@pytest.fixture(autouse=True)
def _reset_volatile_caches():
    """Clear external-data caches before AND after each test.

    Without this, tests that hit predictit / multi-venue / manifold see
    polluted state from earlier tests in the same session — call counts and
    cache-hit assertions end up off-by-one.
    """

    # Also force-clear module-level cache singletons by importing them
    # directly. Some modules captured the cache at import time, so
    # `get_cache(name).clear()` may not refer to the same object if the
    # module passed a `store=` to the factory.
    def _clear_all() -> None:
        try:
            from pfm.sources import predictit as _predictit_mod

            _predictit_mod._ALL_CACHE.clear()
        except Exception:
            pass
        try:
            from pfm import multi_venue_search as _mvs_mod

            _mvs_mod._SEARCH_CACHE.clear()
            _mvs_mod._CONCEPT_CACHE.clear()
        except Exception:
            pass
        try:
            from pfm import arb_scanner as _arb_mod

            _arb_mod._SCANNER_CACHE.clear()
            _arb_mod._MATCH_CACHE.clear()
        except Exception:
            pass
        try:
            from pfm import terminal_countdown as _tc_mod

            _tc_mod._COUNTDOWN_CACHE.clear()
            # Per-slug meta cache added 2026-05-15 to absorb concurrent
            # fanout from market-detail open; reset between tests so each
            # test sees a clean cold Gamma call.
            if hasattr(_tc_mod, "_SLUG_META_CACHE"):
                _tc_mod._SLUG_META_CACHE.clear()
        except Exception:
            pass
        # Polygon, pm_vix and earnings_whisper hold refs at import time;
        # if a previous test called ``reset_caches()`` the namespace lookup
        # returns a fresh instance while the module retains the old one.
        try:
            from pfm.sources import polygon as _polygon_mod

            _polygon_mod._CONSENSUS_CACHE.clear()
            _polygon_mod._CALENDAR_CACHE.clear()
        except Exception:
            pass
        try:
            from pfm import pm_vix as _pm_vix_mod

            _pm_vix_mod._VIX_CACHE.clear()
            _pm_vix_mod._SLUG_MEMORY_CACHE.clear()
        except Exception:
            pass
        try:
            from pfm import earnings_whisper as _ew_mod

            _ew_mod._WHISPER_CACHE.clear()
            _ew_mod._DASHBOARD_CACHE.clear()
            _ew_mod._CALENDAR_CACHE.clear()
        except Exception:
            pass
        # Terminal-layer module-level dict caches that don't go through
        # ``get_cache``; they collide between tests that share a slug.
        try:
            from pfm.terminal import quality_score as _qs_mod

            _qs_mod._QUALITY_GAMMA_CACHE.clear()
        except Exception:
            pass
        try:
            from pfm.terminal import calendar_curated as _cc_mod

            _cc_mod._CLUSTERS_CACHE.clear()
        except Exception:
            pass
        # 2026-05-15 upstream-hardening pass added process-local caches to
        # source clients so 429s on shared endpoints don't cascade to 502s.
        # Each cache leaks state between tests if not cleared per-test.
        # Also clears the polymarket source's slug→metadata cache so tests
        # that re-use the same slug (e.g. "x", "fed-decision") see a clean
        # cold path on every test.
        try:
            from pfm.sources import polymarket as _polymarket_mod

            _polymarket_mod._METADATA_CACHE.clear()
            _polymarket_mod._DISCOVER_CACHE.clear()
        except Exception:
            pass
        try:
            from pfm.sources import binance as _binance_mod

            _binance_mod._KLINES_CACHE.clear()
        except Exception:
            pass
        try:
            from pfm.sources import kalshi as _kalshi_mod

            _kalshi_mod._MARKET_CACHE.clear()
        except Exception:
            pass
        try:
            from pfm.sources import manifold as _manifold_mod

            _manifold_mod._SEARCH_CACHE.clear()
            _manifold_mod._MARKET_CACHE.clear()
        except Exception:
            pass
        try:
            from pfm.sources import predictit as _predictit_mod2

            _predictit_mod2._MARKET_CACHE.clear()
        except Exception:
            pass
        for n in _VOLATILE_CACHE_NAMESPACES:
            _get_cache(n).clear()

    _clear_all()
    yield
    _clear_all()


@pytest.fixture
def factors_file(tmp_path: Path) -> Path:
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


@pytest.fixture
def fake_factor_history() -> callable:
    """Return a function ``(slug, start, end) -> DataFrame`` matching the API."""
    rng = pd.date_range("2025-06-01", "2025-12-31", freq="D", tz="UTC")
    n = len(rng)
    # Smooth oscillating series in [0.10, 0.90] so neither the rising nor
    # falling factor saturates at the clip bounds (which would zero-out the
    # variance of one of them and make conditional-regression ill-posed).
    import numpy as np

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

    def _fetch(_client, slug: str, start=None, end=None):
        df = bank[slug]
        if start is not None:
            df = df[df.index >= start]
        if end is not None:
            df = df[df.index <= end]
        return df

    return _fetch


@pytest.fixture
def fake_log_returns() -> callable:
    """Return a deterministic log-return series builder."""

    def _make(
        ticker: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
        return_type: str = "log",
    ) -> pd.Series:
        import numpy as np

        idx = pd.date_range(start, end, freq="B", tz="UTC")
        n = len(idx)
        rng = np.random.default_rng(seed=hash(ticker) % (2**32))
        values = 0.0001 * np.arange(n) + 0.005 * np.sin(np.arange(n)) + rng.normal(0, 0.001, n)
        if return_type == "simple":
            values = values * 1.05  # tiny scale shift so simple != log in tests
        s = pd.Series(values, index=idx, name="r")
        s.index = pd.to_datetime(s.index, utc=True).normalize()
        return s

    return _make


@pytest.fixture
def app_client(
    monkeypatch: pytest.MonkeyPatch,
    factors_file: Path,
    fake_factor_history: callable,
    fake_log_returns: callable,
) -> Iterator[TestClient]:
    """TestClient with all external IO patched out (Polymarket, yfinance, redis)."""
    # Point config at the temp factors file BEFORE the lifespan runs.
    monkeypatch.setenv("FACTORS_FILE", str(factors_file))
    # Reset the cached Settings singleton so the env var is picked up.
    import pfm.config as cfg

    cfg._settings = None

    # Patch the data layer on ``pfm.main`` — after the 2026-05 split,
    # ``pfm.regression_core`` resolves these symbols via ``_main_attr("name")``
    # (a getattr against the main module) so a single setattr here propagates
    # into the helper functions that actually call them.
    monkeypatch.setattr(main_mod, "fetch_factor_history", fake_factor_history)
    monkeypatch.setattr(main_mod, "get_log_returns", fake_log_returns)

    # Force NullCache so tests don't depend on Redis.
    from pfm.cache import NullCache

    monkeypatch.setattr(main_mod, "RedisCache", lambda url: NullCache())

    with TestClient(main_mod.app) as client:
        yield client


# ---------------------------------------------------------------------------
# Live uvicorn fixture — for SSE / WebSocket tests that need a real TCP
# socket because httpx.ASGITransport buffers streaming reads under Python
# 3.14 (see tests/test_sse_concurrent_load.py for the upstream incompat
# write-up). Spins up uvicorn on a random port in a daemon thread so the
# test process and the server share the same Python interpreter — which
# means module-level monkeypatches still propagate to handlers.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def live_server_factory():
    """Return a callable ``start(app, *, host="127.0.0.1") -> base_url`` that
    boots ``app`` on a random free port and tears down at module teardown.

    Usage in a test::

        @pytest.fixture(scope="module")
        def server(live_server_factory):
            app = FastAPI()
            app.include_router(router)
            return live_server_factory(app)

        def test_thing(server):
            r = httpx.get(server + "/health", timeout=5)
            ...
    """
    import socket
    import threading
    import time as _time
    from contextlib import closing

    import uvicorn

    servers: list[uvicorn.Server] = []
    threads: list[threading.Thread] = []

    def _free_port() -> int:
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]

    def _start(app, host: str = "127.0.0.1") -> str:
        port = _free_port()
        config = uvicorn.Config(
            app=app,
            host=host,
            port=port,
            log_level="warning",
            access_log=False,
            lifespan="on",
        )
        server = uvicorn.Server(config)
        servers.append(server)

        thread = threading.Thread(target=server.run, daemon=True)
        threads.append(thread)
        thread.start()

        # Wait up to 5s for the server to accept connections.
        deadline = _time.time() + 5.0
        while _time.time() < deadline:
            if server.started:
                break
            _time.sleep(0.05)
        else:
            raise RuntimeError(f"uvicorn did not start within 5s on port {port}")

        return f"http://{host}:{port}"

    yield _start

    # Teardown — request shutdown on every spawned server.
    for s in servers:
        s.should_exit = True
    for t in threads:
        t.join(timeout=3)
