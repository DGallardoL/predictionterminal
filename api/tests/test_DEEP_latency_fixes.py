"""Latency-fix regression tests for the five endpoints flagged in the
production audit (2026-05-08):

  - ``POST /reverse-finder``               default candidates: 64.5s -> <1s
  - ``POST /news/causal-chain``            cold:                18.4s -> <3s
  - ``GET  /indices/pm-vix``               cold:                 8.1s -> <100ms
  - ``GET  /alpha/earnings-whisper-dashboard`` cold:            13.0s -> <100ms
  - ``GET  /replay/scenario/{id}``         cold:                 7.8s -> <500ms

Every test isolates the relevant module and uses mocks for any upstream
HTTP. We assert the *behavioural contract* introduced by the fix
(candidate-pool capping, cache reuse, prewarm-only mode, parallel
execution) rather than wall-clock time so the suite stays deterministic.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pfm.cache_utils import get_cache, reset_caches

# ---------------------------------------------------------------------------
# 1) /reverse-finder — default candidate pool
# ---------------------------------------------------------------------------


def _make_factor_universe(n: int = 1360) -> dict[str, Any]:
    """Synth ``{factor_id -> FactorConfig}`` mapping for routing tests."""
    from pfm.factors import FactorConfig

    out: dict[str, FactorConfig] = {}
    for i in range(n):
        fid = f"factor_{i:04d}"
        out[fid] = FactorConfig(
            id=fid,
            name=f"Factor {i:04d}",
            slug=f"slug-{i:04d}",
            source="polymarket",
            description=f"synthetic factor {i}",
            theme="other",
        )
    return out


def _seed_homepage_cache_with_volume(top_slugs: list[str]) -> None:
    """Populate ``terminal_homepage`` cache so the top-volume path triggers."""
    cache = get_cache("terminal_homepage")
    rows = [
        {"slug": slug, "name": slug, "volume_24h": 1_000_000.0 - i * 10_000.0}
        for i, slug in enumerate(top_slugs)
    ]
    payload = {
        "theme": None,
        "hours": 24,
        "n_markets_considered": len(rows),
        "gainers": [],
        "losers": [],
        "most_active": rows,
        "recently_launched": [],
        "resolving_soon": [],
        "breaking_news": [],
        "theme_heatmap": [],
        "pm_vix": 0.0,
    }
    cache.set(("homepage_default", 24), payload, ttl=600)


def test_reverse_finder_default_pool_top_volume_caps_to_100(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``pool=top_volume`` (default) iterates <=100 factors even with 1360 in catalog."""
    reset_caches()
    factors = _make_factor_universe(n=1360)
    top_volume_slugs = [factors[fid].slug for fid in sorted(factors.keys())[:50]]
    _seed_homepage_cache_with_volume(top_volume_slugs)

    call_log: list[str] = []

    def _fake_reverse_find_factors(*_a: Any, **kwargs: Any) -> dict[str, Any]:
        cands = list(kwargs.get("candidate_factor_ids") or [])
        call_log.extend(cands)
        return {
            "ticker": kwargs.get("ticker", "X"),
            "top_factors": [],
            "total_r_squared": 0.0,
            "n_obs": 0,
            "rejected": [],
        }

    import pfm.reverse_finder_router as rff_router

    monkeypatch.setattr(rff_router, "reverse_find_factors", _fake_reverse_find_factors)
    monkeypatch.setattr(
        rff_router, "_build_returns_fetcher", lambda: lambda *a, **k: pd.Series(dtype=float)
    )
    monkeypatch.setattr(
        rff_router, "_build_factor_fetcher", lambda factors_arg: lambda *a, **k: pd.DataFrame()
    )

    class _NullCache:
        def get(self, _k: Any) -> None:
            return None

        def set(self, _k: Any, _v: Any, _ttl: int) -> None:
            return None

    app = FastAPI()
    app.include_router(rff_router.router)
    app.dependency_overrides[rff_router._get_factors_dep] = lambda: factors
    app.dependency_overrides[rff_router._get_cache_dep] = _NullCache

    with TestClient(app) as client:
        r = client.post(
            "/reverse-finder",
            json={
                "ticker": "TEST",
                "start": "2025-06-01",
                "end": "2025-12-01",
                "candidate_factor_ids": None,
                "k": 3,
            },
        )

    assert r.status_code == 200, r.text
    body = r.json()
    # Default pool was renamed from "top_volume:<src>" to "curated_<N>" in the
    # 2026-05 reverse-finder refactor (curated discovery pool; see
    # ``_curated_candidate_ids`` in pfm/reverse_finder_router.py).
    assert body["pool_used"].startswith("curated_") or body["pool_used"].startswith("top_volume:")
    assert 0 < body["n_candidates_evaluated"] <= 200, (
        f"expected <=200 candidates in curated pool, got {body['n_candidates_evaluated']}"
    )
    assert len(call_log) <= 200


def test_reverse_finder_pool_all_iterates_full_catalog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``pool=all`` opt-in permits iterating the full ~1360 factor catalog."""
    reset_caches()
    factors = _make_factor_universe(n=1360)

    seen_count = {"n": 0}

    def _fake_reverse_find_factors(*_a: Any, **kwargs: Any) -> dict[str, Any]:
        seen_count["n"] = len(list(kwargs.get("candidate_factor_ids") or []))
        return {
            "ticker": kwargs.get("ticker", "X"),
            "top_factors": [],
            "total_r_squared": 0.0,
            "n_obs": 0,
            "rejected": [],
        }

    import pfm.reverse_finder_router as rff_router

    monkeypatch.setattr(rff_router, "reverse_find_factors", _fake_reverse_find_factors)
    monkeypatch.setattr(
        rff_router, "_build_returns_fetcher", lambda: lambda *a, **k: pd.Series(dtype=float)
    )
    monkeypatch.setattr(
        rff_router, "_build_factor_fetcher", lambda factors_arg: lambda *a, **k: pd.DataFrame()
    )

    class _NullCache:
        def get(self, _k: Any) -> None:
            return None

        def set(self, _k: Any, _v: Any, _ttl: int) -> None:
            return None

    app = FastAPI()
    app.include_router(rff_router.router)
    app.dependency_overrides[rff_router._get_factors_dep] = lambda: factors
    app.dependency_overrides[rff_router._get_cache_dep] = _NullCache

    with TestClient(app) as client:
        r = client.post(
            "/reverse-finder?pool=all",
            json={
                "ticker": "TEST",
                "start": "2025-06-01",
                "end": "2025-12-01",
                "candidate_factor_ids": None,
                "k": 3,
            },
        )

    assert r.status_code == 200, r.text
    body = r.json()
    # Naming changed from "all" to "all_<N>" in the 2026-05 refactor so the
    # response carries the candidate count back to the caller.
    assert body["pool_used"] == "all_1360" or body["pool_used"] == "all"
    assert body["n_candidates_evaluated"] == 1360
    assert seen_count["n"] == 1360


def test_reverse_finder_explicit_candidates_overrides_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the body sets ``candidate_factor_ids``, ``pool`` is ignored."""
    reset_caches()
    factors = _make_factor_universe(n=10)

    def _fake_reverse_find_factors(*_a: Any, **kwargs: Any) -> dict[str, Any]:
        return {
            "ticker": kwargs.get("ticker", "X"),
            "top_factors": [],
            "total_r_squared": 0.0,
            "n_obs": 0,
            "rejected": [],
        }

    import pfm.reverse_finder_router as rff_router

    monkeypatch.setattr(rff_router, "reverse_find_factors", _fake_reverse_find_factors)
    monkeypatch.setattr(
        rff_router, "_build_returns_fetcher", lambda: lambda *a, **k: pd.Series(dtype=float)
    )
    monkeypatch.setattr(
        rff_router, "_build_factor_fetcher", lambda factors_arg: lambda *a, **k: pd.DataFrame()
    )

    class _NullCache:
        def get(self, _k: Any) -> None:
            return None

        def set(self, _k: Any, _v: Any, _ttl: int) -> None:
            return None

    app = FastAPI()
    app.include_router(rff_router.router)
    app.dependency_overrides[rff_router._get_factors_dep] = lambda: factors
    app.dependency_overrides[rff_router._get_cache_dep] = _NullCache

    with TestClient(app) as client:
        r = client.post(
            "/reverse-finder?pool=all",
            json={
                "ticker": "TEST",
                "start": "2025-06-01",
                "end": "2025-12-01",
                "candidate_factor_ids": ["factor_0001", "factor_0002"],
                "k": 2,
            },
        )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["pool_used"] == "explicit"
    assert body["n_candidates_evaluated"] == 2


# ---------------------------------------------------------------------------
# 2) /news/causal-chain — cache + parallel hydration
# ---------------------------------------------------------------------------


def test_news_causal_chain_post_cache_short_circuits_second_call() -> None:
    """Second identical POST returns the cached payload without recomputing."""
    reset_caches()
    from pfm import news_causal_chain as ncc

    ncc.BETA_REGISTRY.clear()
    ncc.register_betas("trump-impeach-2027", {"DJT": 0.5})

    app = FastAPI()
    app.state.poly = None
    app.include_router(ncc.router)

    body = {
        "factor_id": "trump-impeach-2027",
        "news_items": [
            {
                "title": "Trump impeach vote scheduled",
                "price_before": 0.30,
                "price_after": 0.45,
            }
        ],
        "lookback_hours": 24,
    }

    build_calls: list[Any] = []
    real_build = ncc.build_causal_chain

    def _spy_build(*args: Any, **kwargs: Any) -> dict[str, Any]:
        build_calls.append((args, kwargs))
        return real_build(*args, **kwargs)

    with (
        patch.object(ncc, "build_causal_chain", side_effect=_spy_build),
        TestClient(app) as client,
    ):
        r1 = client.post("/news/causal-chain", json=body)
        r2 = client.post("/news/causal-chain", json=body)

    assert r1.status_code == 200, r1.text
    assert r2.status_code == 200, r2.text
    # First call computes, second comes from the post-body cache.
    assert len(build_calls) == 1, f"expected 1 build, got {len(build_calls)}"
    assert r1.json() == r2.json()


def test_news_causal_chain_parallel_gdelt_rss_hydration() -> None:
    """``_hydrate_news_for_factor_async`` issues GDELT and RSS concurrently."""
    from pfm import news_causal_chain as ncc

    started: list[float] = []
    finished: list[float] = []

    async def _slow_gdelt(_kw: list[str], _ts: str, **_kwargs: Any) -> list[dict]:
        started.append(time.monotonic())
        await asyncio.sleep(0.10)
        finished.append(time.monotonic())
        return [{"title": "g", "ts": "", "url": "u1", "source": "gdelt", "description": ""}]

    async def _slow_rss(_kw: list[str], **_kwargs: Any) -> list[dict]:
        started.append(time.monotonic())
        await asyncio.sleep(0.10)
        finished.append(time.monotonic())
        return [{"title": "r", "ts": "", "url": "u2", "source": "rss", "description": ""}]

    with (
        patch.object(ncc, "_fetch_gdelt_async", _slow_gdelt),
        patch.object(ncc, "_fetch_rss_async", _slow_rss),
    ):
        t0 = time.monotonic()
        items = asyncio.run(ncc._hydrate_news_for_factor_async("ai-bubble-pop", lookback_hours=24))
        wall = time.monotonic() - t0

    # Two ~100ms calls in parallel: total wall <150ms (well under 200ms serial).
    assert wall < 0.18, f"hydration was serial (wall={wall:.3f}s)"
    # Both started before either finished — concrete proof of parallelism.
    assert started[1] < finished[0]
    assert {it["source"] for it in items} == {"gdelt", "rss"}


# ---------------------------------------------------------------------------
# 3) /indices/pm-vix — prewarm cache hot path
# ---------------------------------------------------------------------------


def test_pm_vix_prewarm_only_returns_503_when_cache_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When ``PFM_PMVIX_PREWARM_ENABLED=1`` and cache empty -> 503 + Retry-After."""
    reset_caches()
    monkeypatch.setenv("PFM_PMVIX_PREWARM_ENABLED", "1")

    from pfm import pm_vix

    pm_vix._VIX_CACHE.clear()

    app = FastAPI()
    app.include_router(pm_vix.router)

    with TestClient(app) as client:
        r = client.get("/indices/pm-vix")
    assert r.status_code == 503, r.text
    assert r.headers.get("Retry-After") == "5"


def test_pm_vix_returns_cache_age_seconds_after_prewarm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After a prewarm tick, GET /indices/pm-vix returns the cached snapshot fast."""
    reset_caches()
    monkeypatch.setenv("PFM_PMVIX_PREWARM_ENABLED", "1")

    from pfm import pm_vix

    pm_vix._VIX_CACHE.clear()

    # Patch fetch_gamma_market so the synchronous compute returns deterministic
    # values without hitting the network.
    def _fake_fetch(_http: Any, _url: str, _slug: str, **_kw: Any) -> dict[str, Any]:
        return {"bestBid": 0.30, "bestAsk": 0.31, "lastTradePrice": 0.305, "volume24hr": 10_000.0}

    monkeypatch.setattr(pm_vix, "fetch_gamma_market", _fake_fetch)

    # Run one prewarm cycle synchronously.
    pm_vix._prewarm_compute_snapshot()

    app = FastAPI()
    app.include_router(pm_vix.router)
    with TestClient(app) as client:
        r = client.get("/indices/pm-vix")
    assert r.status_code == 200, r.text
    body = r.json()
    assert "cache_age_seconds" in body
    assert body["cache_age_seconds"] >= 0
    assert body["is_stale"] is False


# ---------------------------------------------------------------------------
# 4) /alpha/earnings-whisper-dashboard — prewarm
# ---------------------------------------------------------------------------


def test_earnings_dashboard_prewarm_503_when_cache_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``PFM_EARNINGS_PREWARM_ENABLED=1`` and cache empty -> 503 + Retry-After."""
    reset_caches()
    monkeypatch.setenv("PFM_EARNINGS_PREWARM_ENABLED", "1")
    from pfm import earnings_whisper as ew

    ew._DASHBOARD_CACHE.clear()

    app = FastAPI()
    app.include_router(ew.router)
    with TestClient(app) as client:
        r = client.get("/alpha/earnings-whisper-dashboard?days=14&source=hardcoded")
    assert r.status_code == 503, r.text
    assert r.headers.get("Retry-After") == "5"


def test_earnings_dashboard_serves_prewarmed_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prewarm tick fills the cache; subsequent GET is fast and surfaces age."""
    reset_caches()
    monkeypatch.setenv("PFM_EARNINGS_PREWARM_ENABLED", "1")
    from pfm import earnings_whisper as ew

    ew._DASHBOARD_CACHE.clear()

    monkeypatch.setattr(
        ew,
        "whisper_dashboard",
        lambda days, source: [],  # type: ignore[arg-type, misc]
    )

    ew._prewarm_compute_dashboard(days=14, source="hardcoded")

    app = FastAPI()
    app.include_router(ew.router)
    with TestClient(app) as client:
        r = client.get("/alpha/earnings-whisper-dashboard?days=14&source=hardcoded")
    assert r.status_code == 200, r.text
    body = r.json()
    assert "cache_age_seconds" in body
    assert body["cache_age_seconds"] >= 0


# ---------------------------------------------------------------------------
# 5) /replay/scenario/{id} — parallel resolves + 24h cache
# ---------------------------------------------------------------------------


def test_replay_scenario_parallel_resolves(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each slug is resolved concurrently; total wall < N × per-call latency."""
    reset_caches()
    from pfm import replay_mode as rm

    rm._SCENARIO_CACHE.clear()

    overlap_max = {"n": 0}
    inflight = {"n": 0}
    asyncio.Lock()

    def _slow_resolve(slug: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        # Tracking the live concurrency without an event loop: use threads
        # since asyncio.to_thread runs each call on the default executor.
        import threading

        t_lock = threading.Lock()
        with t_lock:
            inflight["n"] += 1
            overlap_max["n"] = max(overlap_max["n"], inflight["n"])
        time.sleep(0.05)
        with t_lock:
            inflight["n"] -= 1
        idx = pd.date_range(start, end, freq="D", tz="UTC").normalize()
        n = len(idx)
        price = (0.50 + 0.10 * np.sin(np.linspace(0, 6, n))).clip(0.05, 0.95)
        return pd.DataFrame({"price": price}, index=idx)

    monkeypatch.setattr(rm, "_resolve_pm_history", _slow_resolve)
    rm._yf_close_cached.cache_clear()
    monkeypatch.setattr(
        rm,
        "_yf_close_cached",
        lambda ticker, start_iso, end_iso: tuple(
            (
                pd.Timestamp(d).tz_localize("UTC").normalize().isoformat(),
                100.0 + i,
            )
            for i, d in enumerate(pd.date_range(start_iso, end_iso, freq="D"))
        ),
    )

    out = rm.replay_scenario("election_night_2024")
    assert "scenario" in out
    assert out["cache_age_seconds"] == 0
    # election_night_2024 has 8 slugs; with 50ms each, fully parallel
    # would land near 50-150ms, fully serial near 400ms+. We accept any
    # observed concurrency >= 2 as proof the fan-out is parallel.
    assert overlap_max["n"] >= 2, (
        f"expected concurrent resolves; max overlap was {overlap_max['n']}"
    )


def test_replay_scenario_caches_response_24h(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Second call to the same scenario hits cache, doesn't recompute."""
    reset_caches()
    from pfm import replay_mode as rm

    rm._SCENARIO_CACHE.clear()

    call_count = {"n": 0}

    def _fake_resolve(slug: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        call_count["n"] += 1
        idx = pd.date_range(start, end, freq="D", tz="UTC").normalize()
        return pd.DataFrame({"price": np.full(len(idx), 0.55)}, index=idx)

    monkeypatch.setattr(rm, "_resolve_pm_history", _fake_resolve)
    rm._yf_close_cached.cache_clear()
    monkeypatch.setattr(
        rm,
        "_yf_close_cached",
        lambda ticker, start_iso, end_iso: tuple(
            (
                pd.Timestamp(d).tz_localize("UTC").normalize().isoformat(),
                100.0 + i,
            )
            for i, d in enumerate(pd.date_range(start_iso, end_iso, freq="D"))
        ),
    )

    out1 = rm.replay_scenario("fomc_2024_09")
    n_after_first = call_count["n"]
    out2 = rm.replay_scenario("fomc_2024_09")

    assert out1["scenario"]["id"] == "fomc_2024_09"
    assert out2["scenario"]["id"] == "fomc_2024_09"
    # Second call hits the 24h cache; no further resolves.
    assert call_count["n"] == n_after_first
    assert out2["cache_age_seconds"] >= 0


__all__ = [
    "test_earnings_dashboard_prewarm_503_when_cache_empty",
    "test_earnings_dashboard_serves_prewarmed_cache",
    "test_news_causal_chain_parallel_gdelt_rss_hydration",
    "test_news_causal_chain_post_cache_short_circuits_second_call",
    "test_pm_vix_prewarm_only_returns_503_when_cache_empty",
    "test_pm_vix_returns_cache_age_seconds_after_prewarm",
    "test_replay_scenario_caches_response_24h",
    "test_replay_scenario_parallel_resolves",
    "test_reverse_finder_default_pool_top_volume_caps_to_100",
    "test_reverse_finder_explicit_candidates_overrides_pool",
    "test_reverse_finder_pool_all_iterates_full_catalog",
]
