"""Tests for ``pfm.terminal_peer_scanner`` — /terminal/peers/{slug}.

The on-disk caches are monkeypatched to fully synthetic fixtures so the
tests are hermetic — no IO against ``/tmp`` or the project root. The
router is mounted on a bare :class:`FastAPI` app to avoid pulling the
full ``pfm.main`` lifespan.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pfm import terminal_peer_scanner
from pfm.terminal_peer_scanner import clear_cache, find_peers, router

# --- fixtures ---------------------------------------------------------------


def _hits_fixture() -> list[dict[str, Any]]:
    """Synthetic alpha-hunter hits — three peers around ``alpha_slug``.

    - ``alpha_slug`` ↔ ``beta_slug``  : same theme (macro), high Sharpe, A_GOLD
    - ``alpha_slug`` ↔ ``gamma_slug`` : cross-theme (crypto), mid Sharpe, untiered
    - ``alpha_slug`` ↔ ``delta_slug`` : cross-theme (politics), low Sharpe (filtered
      when ``min_sharpe`` >= 1)
    - ``beta_slug``  ↔ ``gamma_slug`` : pair NOT involving ``alpha_slug``
    """
    return [
        {
            "a_id": "alpha_slug",
            "b_id": "beta_slug",
            "verdict": "REAL_ALPHA",
            "n_obs": 120,
            "adf_pvalue": 0.001,
            "half_life_days": 2.5,
            "beta_hedge": 0.42,
            "oos_sharpe": 4.1,
            "full_sharpe": 3.0,
            "perm_p": 0.0,
            "perm_real_sharpe": 3.0,
            "sweep": "macro",
        },
        {
            "a_id": "gamma_slug",
            "b_id": "alpha_slug",
            "verdict": "REAL_ALPHA",
            "n_obs": 90,
            "adf_pvalue": 0.01,
            "half_life_days": 5.0,
            "beta_hedge": -0.15,
            "oos_sharpe": 2.2,
            "full_sharpe": 1.8,
            "perm_p": 0.04,
            "perm_real_sharpe": 1.8,
            "sweep": "crypto",
        },
        {
            "a_id": "alpha_slug",
            "b_id": "delta_slug",
            "verdict": "REAL_ALPHA",
            "n_obs": 60,
            "adf_pvalue": 0.04,
            "half_life_days": 8.0,
            "beta_hedge": 0.05,
            "oos_sharpe": 0.7,
            "full_sharpe": 0.6,
            "perm_p": 0.09,
            "perm_real_sharpe": 0.6,
            "sweep": "politics",
        },
        {
            "a_id": "beta_slug",
            "b_id": "gamma_slug",
            "verdict": "REAL_ALPHA",
            "n_obs": 80,
            "adf_pvalue": 0.02,
            "half_life_days": 3.0,
            "beta_hedge": 0.10,
            "oos_sharpe": 1.5,
            "full_sharpe": 1.2,
            "perm_p": 0.05,
            "perm_real_sharpe": 1.2,
            "sweep": "macro",
        },
    ]


def _factors_fixture() -> dict[str, dict[str, str]]:
    return {
        "alpha_slug": {"name": "Alpha Market", "theme": "macro", "slug": "alpha-mkt"},
        "beta_slug": {"name": "Beta Market", "theme": "macro", "slug": "beta-mkt"},
        "gamma_slug": {"name": "Gamma Market", "theme": "crypto", "slug": "gamma-mkt"},
        "delta_slug": {"name": "Delta Market", "theme": "politics", "slug": "delta-mkt"},
    }


def _tiers_fixture() -> dict[str, str]:
    # Sorted-pair-id keys, matching the production loader's convention.
    return {
        "__".join(sorted(["alpha_slug", "beta_slug"])): "A_GOLD",
        "__".join(sorted(["alpha_slug", "delta_slug"])): "C_TENTATIVE",
    }


@pytest.fixture(autouse=True)
def _patch_loaders(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Replace the on-disk loaders with in-memory fixtures."""
    clear_cache()
    monkeypatch.setattr(terminal_peer_scanner, "_load_hits", _hits_fixture)
    monkeypatch.setattr(terminal_peer_scanner, "_load_factors", _factors_fixture)
    monkeypatch.setattr(terminal_peer_scanner, "_load_tiers", _tiers_fixture)
    yield
    clear_cache()


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


# --- tests ------------------------------------------------------------------


class TestFindPeersUnit:
    """Direct unit tests against ``find_peers`` (no HTTP layer)."""

    def test_returns_only_pairs_involving_slug_and_sorts_by_sharpe(self) -> None:
        out = find_peers("alpha_slug", top=20, min_sharpe=0.0)

        # Three of four fixture pairs involve alpha_slug.
        assert out["n_peers"] == 3
        assert out["slug"] == "alpha_slug"

        # Sorted by oos_sharpe desc.
        sharpes = [p["oos_sharpe"] for p in out["peers"]]
        assert sharpes == sorted(sharpes, reverse=True)
        assert out["peers"][0]["peer_slug"] == "beta_slug"
        assert out["peers"][0]["oos_sharpe"] == pytest.approx(4.1)

        # Peer that does NOT touch alpha_slug must be absent.
        assert "gamma_slug" in {p["peer_slug"] for p in out["peers"]}
        assert "alpha_slug" not in {p["peer_slug"] for p in out["peers"]}

        # best_peer mirrors peers[0].
        assert out["best_peer"] == out["peers"][0]

    def test_min_sharpe_filters_low_quality_peers(self) -> None:
        out = find_peers("alpha_slug", top=20, min_sharpe=1.0)

        # delta_slug had Sharpe 0.7 → filtered.
        peer_ids = {p["peer_slug"] for p in out["peers"]}
        assert peer_ids == {"beta_slug", "gamma_slug"}
        assert out["n_peers"] == 2
        assert all(p["oos_sharpe"] >= 1.0 for p in out["peers"])


class TestPeerScannerEndpoint:
    def test_endpoint_returns_full_payload_with_tiers_and_cross_theme(
        self, client: TestClient
    ) -> None:
        r = client.get("/terminal/peers/alpha_slug?top=20&min_sharpe=0.0")
        assert r.status_code == 200, r.text
        body = r.json()

        # Shape contract — all six top-level keys present.
        assert set(body) >= {
            "slug",
            "n_peers",
            "peers",
            "cross_theme_count",
            "tier_summary",
            "best_peer",
        }
        assert body["slug"] == "alpha_slug"
        assert body["n_peers"] == 3

        # Per-peer record contract.
        first = body["peers"][0]
        for k in (
            "peer_slug",
            "peer_name",
            "oos_sharpe",
            "perm_p",
            "half_life_days",
            "beta_hedge",
            "theme_a",
            "theme_b",
            "verdict",
            "tier",
        ):
            assert k in first, f"missing key {k}"
        assert first["peer_slug"] == "beta_slug"
        assert first["peer_name"] == "Beta Market"
        assert first["verdict"] == "REAL_ALPHA"

        # Cross-theme count: alpha is "macro" → gamma (crypto) + delta
        # (politics) are cross-theme, beta is same-theme. So 2.
        assert body["cross_theme_count"] == 2

        # Tier summary: beta=A_GOLD, gamma=D_RAW (untiered fallback),
        # delta=C_TENTATIVE.
        assert body["tier_summary"] == {
            "A_GOLD": 1,
            "D_RAW": 1,
            "C_TENTATIVE": 1,
        }

        # best_peer matches the highest-Sharpe row.
        assert body["best_peer"]["peer_slug"] == "beta_slug"
        assert body["best_peer"]["tier"] == "A_GOLD"

    def test_unknown_slug_returns_empty_peer_list_not_404(self, client: TestClient) -> None:
        # A slug nobody is paired with should yield an empty list with a
        # 200 — only a missing cache file is a 404.
        r = client.get("/terminal/peers/zzz_no_such_slug?top=20&min_sharpe=0.5")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["slug"] == "zzz_no_such_slug"
        assert body["n_peers"] == 0
        assert body["peers"] == []
        assert body["best_peer"] is None
        assert body["cross_theme_count"] == 0
        assert body["tier_summary"] == {}
