"""Tests for the cross-sectional vol-distribution Terminal endpoint."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pfm.factors import FactorConfig
from pfm.terminal_vol_distribution import (
    DEFAULT_WINDOW,
    VolDistributionResult,
    _get_factors_dep,
    _get_history_path_dep,
    compute_vol_distribution,
    router,
)

# --- helpers ----------------------------------------------------------------


def _make_factor(fid: str, slug: str, theme: str) -> FactorConfig:
    return FactorConfig(
        id=fid,
        name=fid.replace("_", " ").title(),
        slug=slug,
        source="polymarket",
        description=f"Synthetic factor {fid}.",
        theme=theme,
    )


def _series_with_vol(seed: int, n: int = 250, scale: float = 0.04, base: float = 0.5) -> pd.Series:
    """Bounded probability series whose Δlogit returns have controlled scale."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-01", periods=n, freq="D", tz="UTC")
    # Random-walk in logit space, squashed to a bounded probability range.
    increments = rng.normal(0.0, scale, n)
    logits = np.cumsum(increments)
    probs = 1.0 / (1.0 + np.exp(-logits))
    # Squash into [0.20, 0.80] so the clip never bites.
    probs = 0.20 + 0.60 * probs
    return pd.Series(probs, index=idx, name="price")


@pytest.fixture
def synthetic_universe() -> tuple[dict[str, FactorConfig], dict[str, pd.Series]]:
    """A 6-factor synthetic universe spread over two themes with known vol order."""
    factors: dict[str, FactorConfig] = {
        "macro_low": _make_factor("macro_low", "slug-macro-low", "macro"),
        "macro_mid": _make_factor("macro_mid", "slug-macro-mid", "macro"),
        "macro_high": _make_factor("macro_high", "slug-macro-high", "macro"),
        "macro_target": _make_factor("macro_target", "slug-macro-target", "macro"),
        "tech_one": _make_factor("tech_one", "slug-tech-one", "tech"),
        "tech_two": _make_factor("tech_two", "slug-tech-two", "tech"),
    }
    # Vol scale grows from low → high; target is middle-ish.
    history: dict[str, pd.Series] = {
        "slug-macro-low": _series_with_vol(seed=1, scale=0.01),
        "slug-macro-mid": _series_with_vol(seed=2, scale=0.03),
        "slug-macro-high": _series_with_vol(seed=3, scale=0.08),
        "slug-macro-target": _series_with_vol(seed=4, scale=0.04),
        "slug-tech-one": _series_with_vol(seed=5, scale=0.05),
        "slug-tech-two": _series_with_vol(seed=6, scale=0.06),
    }
    return factors, history


# --- 1. Pure compute layer --------------------------------------------------


def test_compute_vol_distribution_returns_well_formed_payload(
    synthetic_universe: tuple[dict[str, FactorConfig], dict[str, pd.Series]],
) -> None:
    """Result has aligned shapes, ordered quantiles, finite stats, theme-only peers."""
    factors, history = synthetic_universe

    result = compute_vol_distribution(
        slug="slug-macro-target",
        factors=factors,
        history=history,
        window=DEFAULT_WINDOW,
    )

    assert isinstance(result, VolDistributionResult)
    assert result.slug == "slug-macro-target"
    assert result.theme == "macro"
    # 4 macro factors total, target excluded → 3 peers.
    assert result.n_peers == 3

    # Distribution keys present and ordered.
    d = result.vol_distribution
    assert set(d) == {"p10", "p25", "p50", "p75", "p90"}
    assert d["p10"] <= d["p25"] <= d["p50"] <= d["p75"] <= d["p90"]

    # Top / bottom peer lists must hold only same-theme slugs and be vol-ordered.
    macro_peer_slugs = {"slug-macro-low", "slug-macro-mid", "slug-macro-high"}
    for entry in result.peers_higher_vol + result.peers_lower_vol:
        assert entry["slug"] in macro_peer_slugs
        assert isinstance(entry["vol"], float)
    higher_vols = [e["vol"] for e in result.peers_higher_vol]
    lower_vols = [e["vol"] for e in result.peers_lower_vol]
    assert higher_vols == sorted(higher_vols, reverse=True)
    assert lower_vols == sorted(lower_vols)

    # Percentile ∈ [0, 100], finite vol & z-score.
    assert 0.0 <= result.percentile_in_theme <= 100.0
    assert np.isfinite(result.current_vol)
    assert np.isfinite(result.current_z_score)


# --- 2. Theme isolation -----------------------------------------------------


def test_compute_vol_distribution_excludes_other_themes(
    synthetic_universe: tuple[dict[str, FactorConfig], dict[str, pd.Series]],
) -> None:
    """Tech peers must not influence the macro distribution and vice-versa."""
    factors, history = synthetic_universe

    macro = compute_vol_distribution(
        slug="slug-macro-target",
        factors=factors,
        history=history,
        window=DEFAULT_WINDOW,
    )
    tech = compute_vol_distribution(
        slug="slug-tech-one",
        factors=factors,
        history=history,
        window=DEFAULT_WINDOW,
    )

    # Macro target sees 3 macro peers; tech_one sees only 1 tech peer (tech_two).
    assert macro.n_peers == 3
    assert tech.n_peers == 1
    assert macro.theme == "macro"
    assert tech.theme == "tech"

    # No cross-theme leakage in peer lists.
    macro_peer_slugs = {e["slug"] for e in macro.peers_higher_vol + macro.peers_lower_vol}
    tech_peer_slugs = {e["slug"] for e in tech.peers_higher_vol + tech.peers_lower_vol}
    assert macro_peer_slugs.isdisjoint({"slug-tech-one", "slug-tech-two"})
    assert tech_peer_slugs.isdisjoint(
        {"slug-macro-low", "slug-macro-mid", "slug-macro-high", "slug-macro-target"}
    )


# --- 3. End-to-end through FastAPI -----------------------------------------


def test_endpoint_returns_distribution_payload(
    tmp_path: Path,
    synthetic_universe: tuple[dict[str, FactorConfig], dict[str, pd.Series]],
) -> None:
    """The router serves the full payload over HTTP using DI overrides."""
    factors, history = synthetic_universe

    # Persist the synthetic history to a tmp pickle and point the router at it.
    pkl_path = tmp_path / "factor_history.pkl"
    with pkl_path.open("wb") as fh:
        pickle.dump(history, fh)

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[_get_factors_dep] = lambda: factors
    app.dependency_overrides[_get_history_path_dep] = lambda: pkl_path

    with TestClient(app) as client:
        # Happy path.
        r = client.get("/terminal/vol-distribution/slug-macro-target?window=30")
        assert r.status_code == 200, r.text
        payload = r.json()

        assert payload["slug"] == "slug-macro-target"
        assert payload["theme"] == "macro"
        assert payload["n_peers"] == 3
        assert set(payload["vol_distribution"]) == {"p10", "p25", "p50", "p75", "p90"}
        assert 0.0 <= payload["percentile_in_theme"] <= 100.0
        assert isinstance(payload["current_vol"], float)
        assert isinstance(payload["current_z_score"], float)
        assert isinstance(payload["peers_higher_vol"], list)
        assert isinstance(payload["peers_lower_vol"], list)
        assert len(payload["peers_higher_vol"]) <= 5
        assert len(payload["peers_lower_vol"]) <= 5

        # Unknown slug → 404.
        r404 = client.get("/terminal/vol-distribution/does-not-exist")
        assert r404.status_code == 404
        assert "not in factors.yml" in r404.json()["detail"]
