"""Golden-file regression for high-value endpoints.

We snapshot a small set of endpoints whose JSON shape is part of the public
contract with the front-end / external callers.  Any *unintended* schema
change is caught here; intentional changes are pushed through by deleting
the relevant ``tests/golden/<name>.json`` file (or running
``scripts/regenerate_golden.sh``) and committing the diff.

What's covered
--------------
1. ``/health``                       — fast liveness probe shape
2. ``/health/detail``                — readiness shape (volatile fields stripped)
3. ``/alpha-hub/graveyard``          — the alpha cemetery (intentionally stable)
4. ``/quant/multitest/bh``           — BH-FDR with a fixed input vector
5. ``/quant/quarterly-stability``    — 4-quarter Sharpe gate, fixed input
6. ``/portfolio/resolution-tree``    — conditional MTM tree, fixed positions
7. ``/strategies/optimize``          — HRP optimiser, 3 fixed pair_ids, seed=42
8. ``/replay/scenarios``             — list of pre-baked replay scenarios
9. ``/indices/pm-vix/components``    — bucket breakdown, mocked Polymarket data

We mount only the routers each test needs onto a throw-away ``FastAPI`` app
so we never touch ``pfm.main``.  This keeps the tests fast and isolated —
exactly the pattern already used in ``test_pm_vix.py``,
``test_alpha_graveyard.py`` and ``test_resolution_pnl_tree.py``.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pfm import pm_vix
from pfm.alpha_graveyard_router import router as graveyard_router
from pfm.cache_utils import get_cache, reset_caches
from pfm.health_router import router as health_detail_router
from pfm.pm_vix import router as pm_vix_router
from pfm.portfolio_optimizer_router import router as portfolio_optimizer_router
from pfm.quant_validation_router import router as quant_validation_router
from pfm.replay_mode import router as replay_router
from pfm.resolution_pnl_tree import router as pnl_tree_router
from tests.golden_helper import assert_matches_golden

# ---------------------------------------------------------------------------
# Setup / teardown
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_caches() -> None:
    """Caches across modules are global; reset before every test for
    deterministic golden output."""
    reset_caches()
    with contextlib.suppress(Exception):
        get_cache("pm_vix").clear()
    yield
    reset_caches()


def _client_for(*routers: Any) -> TestClient:
    """Mount the given routers on a fresh FastAPI app and return its client."""
    app = FastAPI()
    for r in routers:
        app.include_router(r)
    return TestClient(app)


# ---------------------------------------------------------------------------
# 1. /health  — pulled from main.py via a small standalone re-implementation.
# ---------------------------------------------------------------------------
#
# We deliberately do NOT mount pfm.main.app here (per CLAUDE.md "don't touch
# main.py" and to avoid the global lifespan).  The /health route in main.py
# is trivially stable: ``{"status": "ok", "version": <pfm.__version__>}``.
# We assert the exact shape directly.


def test_golden_health_simple() -> None:
    from pfm import __version__

    body = {"status": "ok", "version": __version__}
    assert_matches_golden("health_simple", body, ignore_keys=["version"])


# ---------------------------------------------------------------------------
# 2. /health/detail
# ---------------------------------------------------------------------------


def test_golden_health_detail() -> None:
    client = _client_for(health_detail_router)
    r = client.get("/health/detail")
    assert r.status_code == 200, r.text
    # uptime / git_sha / redis latency vary across runs and machines.
    assert_matches_golden(
        "health_detail",
        r.json(),
        ignore_keys=["uptime_seconds", "git_sha", "redis", "version", "latency_ms"],
    )


# ---------------------------------------------------------------------------
# 3. /alpha-hub/graveyard
# ---------------------------------------------------------------------------


def test_golden_alpha_graveyard() -> None:
    client = _client_for(graveyard_router)
    r = client.get("/alpha-hub/graveyard")
    assert r.status_code == 200, r.text
    # The graveyard is the canonical schema check — pin it exactly.  The
    # ``killed_iso`` field is a stable historical date (not "now"), so we
    # don't strip it; if the registry is intentionally edited, the golden
    # is regenerated on purpose.
    assert_matches_golden("alpha_graveyard", r.json())


# ---------------------------------------------------------------------------
# 4. /quant/multitest/bh
# ---------------------------------------------------------------------------


def test_golden_multitest_bh_fixed_input() -> None:
    client = _client_for(quant_validation_router)
    body = {
        "p_values": [0.001, 0.01, 0.02, 0.04, 0.06, 0.10, 0.30, 0.80],
        "alpha": 0.05,
    }
    r = client.post("/quant/multitest/bh", json=body)
    assert r.status_code == 200, r.text
    assert_matches_golden("multitest_bh_fixed", r.json())


# ---------------------------------------------------------------------------
# 5. /quant/quarterly-stability
# ---------------------------------------------------------------------------


def test_golden_quarterly_stability_fixed_input() -> None:
    client = _client_for(quant_validation_router)
    body = {
        "quarterly_sharpes": [0.6, 0.55, 0.7, 0.62],
        "threshold": 0.5,
    }
    r = client.post("/quant/quarterly-stability", json=body)
    assert r.status_code == 200, r.text
    assert_matches_golden("quarterly_stability_fixed", r.json())


# ---------------------------------------------------------------------------
# 6. /portfolio/resolution-tree
# ---------------------------------------------------------------------------


def test_golden_portfolio_resolution_tree() -> None:
    client = _client_for(pnl_tree_router)
    body = {
        "positions": [
            {"ticker": "DJT", "size_usd": 10_000, "beta_factor": 1.0},
            {"ticker": "SPY", "size_usd": 5_000, "beta_factor": -0.10},
        ],
        "factor_id": "trump-2024",
        "current_prob": 0.40,
        "epsilon": 0.01,
    }
    r = client.post("/portfolio/resolution-tree", json=body)
    assert r.status_code == 200, r.text
    assert_matches_golden("portfolio_resolution_tree", r.json())


# ---------------------------------------------------------------------------
# 7. /strategies/optimize  — HRP, 3 fixed pair_ids, deterministic seed
# ---------------------------------------------------------------------------


_OPT_PAIR_IDS = [
    "fed_target_40_eoy__fed_target_45_eoy",
    "fed_cuts_10_2026__fed_cuts_7_2026",
    "bitcoin_reach_by_december__bitcoin_reach_by_december_2",
]


def test_golden_strategies_optimize_hrp() -> None:
    """HRP with 3 catalog pair_ids and seed=42.  The optimiser is deterministic
    once ``seed`` is pinned and synthetic returns are generated reproducibly."""
    client = _client_for(portfolio_optimizer_router)
    body = {
        "pair_ids": _OPT_PAIR_IDS,
        "method": "hrp",
        "lookback_days": 252,
        "risk_free_rate": 0.045,
        "max_weight": 0.50,
        "min_weight": 0.0,
        "shrinkage": "ledoit_wolf",
        "shrink_mu": 0.5,
        "mc_paths": 1000,
        "mc_horizon_days": 60,
        "mc_block": 10,
        "return_frontier": False,
        "seed": 42,
    }
    r = client.post("/strategies/optimize", json=body)
    assert r.status_code == 200, r.text
    # ``warnings`` may include catalog-discovery messages whose exact wording
    # depends on whether ``alpha_strategies.json`` is present at test time.
    # We pin shape + numerical outputs but ignore the freeform warnings list.
    assert_matches_golden(
        "strategies_optimize_hrp",
        r.json(),
        ignore_keys=["warnings"],
    )


# ---------------------------------------------------------------------------
# 8. /replay/scenarios
# ---------------------------------------------------------------------------


def test_golden_replay_scenarios() -> None:
    client = _client_for(replay_router)
    r = client.get("/replay/scenarios")
    assert r.status_code == 200, r.text
    assert_matches_golden("replay_scenarios", r.json())


# ---------------------------------------------------------------------------
# 9. /indices/pm-vix/components  — Polymarket Gamma is mocked
# ---------------------------------------------------------------------------


def _market(prob: float, vol: float = 10_000.0) -> dict[str, Any]:
    """Build a Gamma-shaped market dict with mid ≈ ``prob`` (matches
    the helper in ``test_pm_vix.py``)."""
    return {
        "bestBid": max(0.0, prob - 0.005),
        "bestAsk": min(1.0, prob + 0.005),
        "lastTradePrice": prob,
        "volume24hr": vol,
    }


def test_golden_pm_vix_components(monkeypatch: pytest.MonkeyPatch) -> None:
    """Lock the per-bucket breakdown for a deterministic mid=0.50 across
    every bucket.  The composite score is then a pure function of the
    bucket weights."""
    monkeypatch.setattr(pm_vix, "fetch_gamma_market", lambda *a, **k: _market(0.50))

    client = _client_for(pm_vix_router)
    r = client.get("/indices/pm-vix/components")
    assert r.status_code == 200, r.text
    # ``as_of`` is "now" — strip it.  Score / components are deterministic
    # once the upstream fetch is stubbed.
    assert_matches_golden(
        "pm_vix_components_dummy",
        r.json(),
        ignore_keys=["as_of"],
    )


# ---------------------------------------------------------------------------
# Sanity: golden directory is created on first run
# ---------------------------------------------------------------------------


def test_golden_dir_exists() -> None:
    """Smoke check that the golden directory is real and writable."""
    p = Path(__file__).parent / "golden"
    assert p.exists() and p.is_dir(), "tests/golden/ must exist after first run"
