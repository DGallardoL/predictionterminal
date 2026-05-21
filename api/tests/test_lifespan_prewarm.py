"""Tests for the /terminal/vol-distribution + /terminal/factor-clusters
lifespan warm-cache prewarmers (``pfm.prewarm``).

What is verified
----------------
1. ``_top_slugs_for_voldist`` walks themes in priority order and caps at N.
2. ``warm_voldist_lookup`` and ``warm_clusters_lookup`` honour the 60s TTL.
3. ``prewarm_voldist`` / ``prewarm_factor_clusters`` populate ``app.state``
   on a TestClient startup and log "prewarm: voldist ok" / "factor-clusters
   ok".
4. The HTTP handlers short-circuit (no live compute call) when warm and
   recompute when the warm entry is older than the TTL.

We patch the heavy compute helpers (``_compute_voldist_snapshots`` and
``_compute_factor_clusters_default``) so the test runs in <100 ms without
needing the on-disk strat-7 pickle.
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any
from unittest import mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pfm import prewarm
from pfm.factors import FactorConfig
from pfm.prewarm import (
    TOP_N_VOLDIST,
    WARM_TTL_SECONDS,
    _top_slugs_for_voldist,
    prewarm_factor_clusters,
    prewarm_voldist,
    warm_clusters_lookup,
    warm_voldist_lookup,
)
from pfm.terminal.factor_clusters import router as clusters_router
from pfm.terminal.vol_distribution import (
    _get_factors_dep,
    _get_history_path_dep,
)
from pfm.terminal.vol_distribution import router as voldist_router

# ---------------------------------------------------------------------------
# Shared synthetic factor catalog (no on-disk pickle needed).
# ---------------------------------------------------------------------------


def _f(fid: str, slug: str, theme: str) -> FactorConfig:
    return FactorConfig(
        id=fid,
        name=fid.replace("_", " ").title(),
        slug=slug,
        source="polymarket",
        description=f"synthetic {fid}",
        theme=theme,
    )


@pytest.fixture
def factors_catalog() -> dict[str, FactorConfig]:
    """Mixed-theme synthetic catalog. Includes politics + earnings + macro
    + crypto + sentiment + 'other' so the priority ordering can be checked."""
    return {
        "pol_a": _f("pol_a", "slug-pol-a", "politics"),
        "pol_b": _f("pol_b", "slug-pol-b", "politics"),
        "earn_a": _f("earn_a", "slug-earn-a", "earnings"),
        "macro_a": _f("macro_a", "slug-macro-a", "macro"),
        "macro_b": _f("macro_b", "slug-macro-b", "macro"),
        "crypto_a": _f("crypto_a", "slug-crypto-a", "crypto"),
        "sent_a": _f("sent_a", "slug-sent-a", "sentiment"),
        "other_a": _f("other_a", "slug-other-a", "other"),
    }


# ---------------------------------------------------------------------------
# 1. _top_slugs_for_voldist
# ---------------------------------------------------------------------------


def test_top_slugs_respects_theme_priority(
    factors_catalog: dict[str, FactorConfig],
) -> None:
    """politics first, then earnings → macro → crypto → sentiment."""
    out = _top_slugs_for_voldist(factors_catalog, n=6)
    # First two should be politics (insertion order within theme is stable).
    assert out[:2] == ["slug-pol-a", "slug-pol-b"]
    # Then earnings, then macro …
    assert out[2] == "slug-earn-a"
    assert out[3] in {"slug-macro-a", "slug-macro-b"}
    assert "slug-crypto-a" in out
    # 'other' is the lowest priority and only included after the priority
    # themes are exhausted.
    assert "slug-other-a" not in out  # we capped at 6 priority slots first


def test_top_slugs_caps_at_n(factors_catalog: dict[str, FactorConfig]) -> None:
    assert len(_top_slugs_for_voldist(factors_catalog, n=3)) == 3
    assert _top_slugs_for_voldist(factors_catalog, n=0) == []
    assert _top_slugs_for_voldist({}, n=5) == []


def test_top_slugs_pads_from_other_themes_when_priority_empty() -> None:
    """When no priority-theme factors exist, fall through to whatever's there."""
    only_other = {"x": _f("x", "slug-x", "other"), "y": _f("y", "slug-y", "weather")}
    assert set(_top_slugs_for_voldist(only_other, n=5)) == {"slug-x", "slug-y"}


def test_top_slugs_default_n_matches_module_constant(
    factors_catalog: dict[str, FactorConfig],
) -> None:
    """The exported TOP_N_VOLDIST drives the cap when the caller omits N."""
    # Build a catalog larger than TOP_N_VOLDIST so the cap actually bites.
    big = {f"id_{i}": _f(f"id_{i}", f"slug-{i}", "politics") for i in range(TOP_N_VOLDIST + 5)}
    out = _top_slugs_for_voldist(big)
    assert len(out) == TOP_N_VOLDIST


# ---------------------------------------------------------------------------
# 2. Freshness lookup helpers
# ---------------------------------------------------------------------------


class _AppShim:
    """Minimal stand-in for ``FastAPI`` exposing ``.state`` only."""

    def __init__(self) -> None:
        self.state = type("S", (), {})()


def test_warm_voldist_lookup_returns_none_when_unset() -> None:
    app = _AppShim()
    assert warm_voldist_lookup(app, "slug-x") is None  # type: ignore[arg-type]


def test_warm_voldist_lookup_returns_payload_when_fresh() -> None:
    app = _AppShim()
    payload = {"slug": "slug-x", "current_vol": 0.42}
    app.state.warm_voldist = {
        "computed_at": time.time(),
        "snapshots": {"slug-x": payload},
    }
    assert warm_voldist_lookup(app, "slug-x") == payload  # type: ignore[arg-type]


def test_warm_voldist_lookup_returns_none_when_stale() -> None:
    app = _AppShim()
    app.state.warm_voldist = {
        "computed_at": time.time() - (WARM_TTL_SECONDS + 5),
        "snapshots": {"slug-x": {"current_vol": 0.42}},
    }
    assert warm_voldist_lookup(app, "slug-x") is None  # type: ignore[arg-type]


def test_warm_voldist_lookup_returns_none_for_unknown_slug() -> None:
    app = _AppShim()
    app.state.warm_voldist = {
        "computed_at": time.time(),
        "snapshots": {"slug-x": {"current_vol": 0.42}},
    }
    assert warm_voldist_lookup(app, "slug-y") is None  # type: ignore[arg-type]


def test_warm_clusters_lookup_only_for_default_query() -> None:
    """Theme=None + min_corr=0.5 hits; anything else falls through."""
    app = _AppShim()
    payload = {"n_factors_in": 4, "n_clusters": 1, "clusters": [], "theme": None, "min_corr": 0.5}
    app.state.warm_clusters = {"computed_at": time.time(), "payload": payload}
    assert warm_clusters_lookup(app, theme=None, min_corr=0.5) == payload  # type: ignore[arg-type]
    assert warm_clusters_lookup(app, theme="politics", min_corr=0.5) is None  # type: ignore[arg-type]
    assert warm_clusters_lookup(app, theme=None, min_corr=0.6) is None  # type: ignore[arg-type]


def test_warm_clusters_lookup_returns_none_when_stale() -> None:
    app = _AppShim()
    app.state.warm_clusters = {
        "computed_at": time.time() - (WARM_TTL_SECONDS + 5),
        "payload": {
            "n_factors_in": 0,
            "n_clusters": 0,
            "clusters": [],
            "theme": None,
            "min_corr": 0.5,
        },
    }
    assert warm_clusters_lookup(app, theme=None, min_corr=0.5) is None  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 3. Async prewarm tasks populate app.state and log success
# ---------------------------------------------------------------------------


def test_prewarm_voldist_populates_app_state(
    factors_catalog: dict[str, FactorConfig],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The async task fans out via to_thread and stores snapshots on state."""
    app = FastAPI()
    app.state.factors = factors_catalog

    fake_snapshots = {
        "slug-pol-a": {
            "slug": "slug-pol-a",
            "current_vol": 0.5,
            "theme": "politics",
            "n_peers": 1,
            "percentile_in_theme": 50.0,
            "vol_distribution": {"p10": 0.1, "p25": 0.2, "p50": 0.3, "p75": 0.4, "p90": 0.5},
            "current_z_score": 0.0,
            "peers_higher_vol": [],
            "peers_lower_vol": [],
        },
    }

    with (
        mock.patch.object(
            prewarm,
            "_compute_voldist_snapshots",
            return_value=fake_snapshots,
        ) as compute_mock,
        caplog.at_level(logging.INFO, logger="pfm.prewarm"),
    ):
        asyncio.run(prewarm_voldist(app))

    # Compute helper was called with (factors, slugs).
    compute_mock.assert_called_once()
    args, _ = compute_mock.call_args
    assert args[0] is factors_catalog
    assert isinstance(args[1], list) and len(args[1]) > 0

    # State populated with a fresh timestamp and the mocked payload.
    assert isinstance(app.state.warm_voldist, dict)
    assert app.state.warm_voldist["snapshots"] == fake_snapshots
    assert time.time() - app.state.warm_voldist["computed_at"] < 2.0

    # Log line confirms format.
    msgs = " | ".join(rec.getMessage() for rec in caplog.records)
    assert "prewarm: voldist ok" in msgs


def test_prewarm_factor_clusters_populates_app_state(
    caplog: pytest.LogCaptureFixture,
) -> None:
    app = FastAPI()
    fake_payload = {
        "n_factors_in": 12,
        "n_clusters": 3,
        "clusters": [],
        "theme": None,
        "min_corr": 0.5,
    }
    with (
        mock.patch.object(
            prewarm,
            "_compute_factor_clusters_default",
            return_value=fake_payload,
        ) as compute_mock,
        caplog.at_level(logging.INFO, logger="pfm.prewarm"),
    ):
        asyncio.run(prewarm_factor_clusters(app))

    compute_mock.assert_called_once()
    assert isinstance(app.state.warm_clusters, dict)
    assert app.state.warm_clusters["payload"] == fake_payload
    assert time.time() - app.state.warm_clusters["computed_at"] < 2.0
    msgs = " | ".join(rec.getMessage() for rec in caplog.records)
    assert "prewarm: factor-clusters ok" in msgs


def test_prewarm_voldist_swallows_compute_failure(
    factors_catalog: dict[str, FactorConfig],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A raise inside the compute thread is logged but never propagates."""
    app = FastAPI()
    app.state.factors = factors_catalog
    with (
        mock.patch.object(
            prewarm,
            "_compute_voldist_snapshots",
            side_effect=RuntimeError("pickle on fire"),
        ),
        caplog.at_level(logging.WARNING, logger="pfm.prewarm"),
    ):
        asyncio.run(prewarm_voldist(app))  # MUST NOT raise

    # State left unset (we set it to None at lifespan entry; here the
    # attribute simply doesn't exist on the bare FastAPI()).
    assert not hasattr(app.state, "warm_voldist") or app.state.warm_voldist is None
    msgs = " | ".join(rec.getMessage() for rec in caplog.records)
    assert "voldist failed" in msgs


def test_prewarm_factor_clusters_handles_none_payload(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """When the compute helper returns None (no history), skip silently."""
    app = FastAPI()
    with (
        mock.patch.object(
            prewarm,
            "_compute_factor_clusters_default",
            return_value=None,
        ),
        caplog.at_level(logging.INFO, logger="pfm.prewarm"),
    ):
        asyncio.run(prewarm_factor_clusters(app))
    assert not hasattr(app.state, "warm_clusters") or app.state.warm_clusters is None
    msgs = " | ".join(rec.getMessage() for rec in caplog.records)
    assert "factor-clusters skipped" in msgs


# ---------------------------------------------------------------------------
# 4. End-to-end: handler short-circuits on warm, recomputes when stale.
# ---------------------------------------------------------------------------


def _build_app_with_voldist_routes(
    factors: dict[str, FactorConfig],
    pkl_path: Path,
) -> FastAPI:
    app = FastAPI()
    app.include_router(voldist_router)
    app.dependency_overrides[_get_factors_dep] = lambda: factors
    app.dependency_overrides[_get_history_path_dep] = lambda: pkl_path
    return app


def test_voldist_endpoint_returns_warm_payload_without_recomputing(
    factors_catalog: dict[str, FactorConfig],
    tmp_path: Path,
) -> None:
    """Fresh warm entry → handler must skip compute_vol_distribution entirely."""
    app = _build_app_with_voldist_routes(factors_catalog, tmp_path / "missing.pkl")

    warm_payload = {
        "slug": "slug-pol-a",
        "current_vol": 0.99,
        "theme": "politics",
        "n_peers": 7,
        "percentile_in_theme": 85.0,
        "vol_distribution": {"p10": 0.1, "p25": 0.2, "p50": 0.3, "p75": 0.4, "p90": 0.5},
        "current_z_score": 1.2,
        "peers_higher_vol": [],
        "peers_lower_vol": [],
    }
    app.state.warm_voldist = {
        "computed_at": time.time(),
        "snapshots": {"slug-pol-a": warm_payload},
    }

    # Patch compute_vol_distribution at the module the handler resolves it
    # against — any call would fail this assert. We use a sentinel raise so
    # if the warm path is bypassed the test surfaces a clear error.
    with (
        mock.patch(
            "pfm.terminal.vol_distribution.compute_vol_distribution",
            side_effect=AssertionError("must not recompute on warm hit"),
        ),
        TestClient(app) as client,
    ):
        r = client.get("/terminal/vol-distribution/slug-pol-a")

    assert r.status_code == 200, r.text
    assert r.json() == warm_payload


def test_voldist_endpoint_recomputes_when_stale(
    factors_catalog: dict[str, FactorConfig],
    tmp_path: Path,
) -> None:
    """A stale warm entry must NOT short-circuit — handler falls through."""
    app = _build_app_with_voldist_routes(factors_catalog, tmp_path / "missing.pkl")

    app.state.warm_voldist = {
        "computed_at": time.time() - (WARM_TTL_SECONDS + 5),
        "snapshots": {"slug-pol-a": {"slug": "slug-pol-a", "current_vol": 0.99}},
    }

    # Fake the (live) compute path so the test doesn't need a pickle.
    from pfm.terminal.vol_distribution import VolDistributionResult

    fresh = VolDistributionResult(
        slug="slug-pol-a",
        current_vol=0.10,
        theme="politics",
        n_peers=1,
        percentile_in_theme=10.0,
        vol_distribution={"p10": 0.01, "p25": 0.02, "p50": 0.03, "p75": 0.04, "p90": 0.05},
        current_z_score=-1.0,
        peers_higher_vol=[],
        peers_lower_vol=[],
    )
    with (
        mock.patch(
            "pfm.terminal.vol_distribution.compute_vol_distribution",
            return_value=fresh,
        ) as compute_mock,
        TestClient(app) as client,
    ):
        r = client.get("/terminal/vol-distribution/slug-pol-a")

    assert r.status_code == 200, r.text
    body = r.json()
    # The handler must have run live-compute → the recomputed (low) current_vol
    # leaks through rather than the stale 0.99.
    assert body["current_vol"] == pytest.approx(0.10)
    compute_mock.assert_called_once()


def test_voldist_endpoint_recomputes_when_custom_window(
    factors_catalog: dict[str, FactorConfig],
    tmp_path: Path,
) -> None:
    """A non-default ?window= must always bypass the warm path."""
    app = _build_app_with_voldist_routes(factors_catalog, tmp_path / "missing.pkl")
    app.state.warm_voldist = {
        "computed_at": time.time(),
        "snapshots": {"slug-pol-a": {"slug": "slug-pol-a", "current_vol": 0.99}},
    }
    from pfm.terminal.vol_distribution import VolDistributionResult

    fresh = VolDistributionResult(
        slug="slug-pol-a",
        current_vol=0.42,
        theme="politics",
        n_peers=1,
        percentile_in_theme=50.0,
        vol_distribution={"p10": 0.01, "p25": 0.02, "p50": 0.03, "p75": 0.04, "p90": 0.05},
        current_z_score=0.0,
        peers_higher_vol=[],
        peers_lower_vol=[],
    )
    with (
        mock.patch(
            "pfm.terminal.vol_distribution.compute_vol_distribution",
            return_value=fresh,
        ) as compute_mock,
        TestClient(app) as client,
    ):
        r = client.get("/terminal/vol-distribution/slug-pol-a?window=60")
    assert r.status_code == 200, r.text
    assert r.json()["current_vol"] == pytest.approx(0.42)
    compute_mock.assert_called_once()


def test_clusters_endpoint_returns_warm_payload_without_recomputing() -> None:
    """Default-query factor-clusters call returns the prewarmed payload."""
    app = FastAPI()
    app.include_router(clusters_router)

    fake_payload: dict[str, Any] = {
        "n_factors_in": 10,
        "n_clusters": 2,
        "clusters": [],
        "theme": None,
        "min_corr": 0.5,
        "degraded_mode": False,
        "reason": None,
    }
    app.state.warm_clusters = {"computed_at": time.time(), "payload": fake_payload}

    # If any of these are touched, the warm path was bypassed.
    with (
        mock.patch(
            "pfm.terminal.factor_clusters._load_cached_history",
            side_effect=AssertionError("must not load pickle on warm hit"),
        ),
        TestClient(app) as client,
    ):
        r = client.get("/terminal/factor-clusters")

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["n_factors_in"] == 10
    assert body["n_clusters"] == 2


def test_clusters_endpoint_bypasses_warm_on_non_default_query() -> None:
    """``?theme=politics`` is not in the prewarmed snapshot → falls through.

    We assert "falls through" by checking the response NEVER matches the
    distinctive warm payload (n_factors_in=999). The handler instead enters
    its degraded-mode branch (empty history → 200 with degraded_mode=true)
    or the recompute path — either way, the warm payload's marker value
    cannot leak out.
    """
    app = FastAPI()
    app.include_router(clusters_router)
    app.state.warm_clusters = {
        "computed_at": time.time(),
        "payload": {
            "n_factors_in": 999,
            "n_clusters": 0,
            "clusters": [],
            "theme": None,
            "min_corr": 0.5,
            "degraded_mode": False,
            "reason": None,
        },
    }
    with (
        mock.patch(
            "pfm.terminal.factor_clusters._load_cached_history",
            return_value={},
        ),
        TestClient(app) as client,
    ):
        r = client.get("/terminal/factor-clusters?theme=politics")
    assert r.status_code == 200, r.text
    body = r.json()
    # Crucial: warm payload (n_factors_in=999) was NOT served — the warm
    # short-circuit correctly stayed dormant for the non-default query.
    assert body["n_factors_in"] != 999
    # And the response matches the recompute branch's degraded-mode shape.
    assert body["degraded_mode"] is True


# ---------------------------------------------------------------------------
# 5. Lifespan integration: fire-and-forget creates the tasks
# ---------------------------------------------------------------------------


def test_lifespan_dispatches_both_prewarm_tasks_via_create_task() -> None:
    """Importing pfm.main and entering its lifespan should schedule both tasks.

    We patch the prewarm coroutines so the lifespan startup stays fast and
    deterministic — the only thing we care about here is that asyncio.create_task
    was invoked for each.
    """
    # Import lazily so any earlier pytest collection cost is amortized.
    from pfm import main as main_mod

    seen: list[str] = []

    async def _fake_prewarm_voldist(app: FastAPI) -> None:
        seen.append("voldist")
        app.state.warm_voldist = {"computed_at": time.time(), "snapshots": {}}

    async def _fake_prewarm_clusters(app: FastAPI) -> None:
        seen.append("clusters")
        app.state.warm_clusters = {
            "computed_at": time.time(),
            "payload": {
                "n_factors_in": 0,
                "n_clusters": 0,
                "clusters": [],
                "theme": None,
                "min_corr": 0.5,
                "degraded_mode": True,
                "reason": "test",
            },
        }

    with (
        mock.patch.object(prewarm, "prewarm_voldist", _fake_prewarm_voldist),
        mock.patch.object(prewarm, "prewarm_factor_clusters", _fake_prewarm_clusters),
        TestClient(main_mod.app) as client,
    ):
        # Probe a cheap endpoint to force the lifespan to run.
        r = client.get("/health")
        assert r.status_code == 200

        # Give the two fire-and-forget tasks a moment to land.
        for _ in range(50):
            if {"voldist", "clusters"}.issubset(set(seen)):
                break
            time.sleep(0.02)

    assert "voldist" in seen, f"voldist prewarm task never ran (seen={seen})"
    assert "clusters" in seen, f"clusters prewarm task never ran (seen={seen})"
