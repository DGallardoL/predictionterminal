"""Tests for :mod:`pfm.reverse_finder` and the router in
:mod:`pfm.reverse_finder_router`.

Two layers:

1.  Pure-function tests of ``reverse_find_factors`` and
    ``prediction_driven_alpha`` using injected fake fetchers — no FastAPI,
    no network IO.
2.  Router tests wiring the router into the production ``app`` via a fresh
    ``TestClient`` with monkey-patched data layer (yfinance + Polymarket).
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from pfm.reverse_finder import (
    DEFAULT_TICKERS,
    prediction_driven_alpha,
    reverse_find_factors,
)

# --- helpers ---------------------------------------------------------------


def _make_logit_clipped_series(values: np.ndarray, idx: pd.DatetimeIndex) -> pd.DataFrame:
    """Wrap a probability vector into the {price} DataFrame shape the fetchers return."""
    return pd.DataFrame({"price": np.clip(values, 0.05, 0.95)}, index=idx)


def _build_synthetic_universe(
    n: int = 250, seed: int = 0
) -> tuple[pd.DatetimeIndex, dict[str, pd.DataFrame], pd.Series]:
    """Build a synthetic factor universe + ticker returns.

    The ticker is constructed so that
        y_t = 0.5 * Δlogit(p_A_t) + 0.3 * Δlogit(p_B_t) + small-noise
    plus several pure-noise factors that should NOT be picked.
    """
    from pfm.model import delta_logit

    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-01", periods=n, freq="B", tz="UTC")

    # Two informative factors with smooth-ish dynamics so Δlogit has variance.
    t = np.arange(n) / n
    pa = 0.5 + 0.30 * np.sin(2 * np.pi * t * 1.3) + 0.05 * rng.normal(0, 1, n)
    pb = 0.5 + 0.25 * np.cos(2 * np.pi * t * 0.9) + 0.05 * rng.normal(0, 1, n)
    factors: dict[str, pd.DataFrame] = {
        "factor_A": _make_logit_clipped_series(pa, idx),
        "factor_B": _make_logit_clipped_series(pb, idx),
    }
    # Three independent noise factors (uncorrelated with y by construction).
    for name in ("noise_1", "noise_2", "noise_3"):
        v = 0.5 + 0.25 * rng.normal(0, 1, n)
        factors[name] = _make_logit_clipped_series(v, idx)

    dl_a = delta_logit(factors["factor_A"]["price"]).dropna()
    dl_b = delta_logit(factors["factor_B"]["price"]).dropna()
    common = dl_a.index.intersection(dl_b.index)
    eps = rng.normal(0, 1e-3, len(common))
    y_vals = 0.5 * dl_a.loc[common].values + 0.3 * dl_b.loc[common].values + eps
    y = pd.Series(y_vals, index=common, name="r")
    return idx, factors, y


def _make_fetchers(factors: dict[str, pd.DataFrame], y: pd.Series):
    def factor_fetcher(factor_id, start, end):
        df = factors[factor_id]
        mask = (df.index >= start) & (df.index <= end)
        return df.loc[mask]

    def returns_fetcher(ticker, start, end, return_type="log"):
        # Single-ticker fixture: the ticker is whatever the test passes,
        # we always return ``y``. Index is already restricted; trim by window.
        s = y[(y.index >= start) & (y.index <= end)]
        return s

    return returns_fetcher, factor_fetcher


# --- pure-function tests: reverse_find_factors -----------------------------


class TestReverseFindFactors:
    def test_recovers_planted_factors_in_order(self) -> None:
        _, factors, y = _build_synthetic_universe(n=250, seed=42)
        returns_fetcher, factor_fetcher = _make_fetchers(factors, y)

        out = reverse_find_factors(
            ticker="SYNTH",
            candidate_factor_ids=list(factors.keys()),
            start=date(2025, 1, 1),
            end=date(2025, 12, 31),
            k=5,
            returns_fetcher=returns_fetcher,
            factor_fetcher=factor_fetcher,
        )

        assert out["ticker"] == "SYNTH"
        picks = out["top_factors"]
        ids = [p["factor_id"] for p in picks]
        # Both informative factors must be in the top-5 selection.
        assert "factor_A" in ids
        assert "factor_B" in ids
        # The first pick should be the larger-coefficient informative factor —
        # factor_A is the 0.5 loading, so it should land first.
        assert ids[0] == "factor_A"
        # The selected betas should be close to the planted values (HAC SEs
        # are noisy on n=250, so allow a generous band).
        beta_a = next(p["beta"] for p in picks if p["factor_id"] == "factor_A")
        beta_b = next(p["beta"] for p in picks if p["factor_id"] == "factor_B")
        assert beta_a == pytest.approx(0.5, abs=0.1)
        assert beta_b == pytest.approx(0.3, abs=0.1)

        # The two informative factors should account for nearly all R².
        assert out["total_r_squared"] > 0.9
        # Each informative pick should have |t| > 2 — a real signal.
        assert abs(picks[0]["t_stat"]) > 2.0
        # Δr² for the first pick should be larger than for any subsequent pick.
        deltas = [p["delta_r_squared"] for p in picks]
        assert deltas[0] >= max(deltas[1:])

    def test_k_larger_than_candidates_is_capped(self) -> None:
        _, factors, y = _build_synthetic_universe(n=250, seed=1)
        # Only feed 2 candidates while requesting k=5.
        small = {k: factors[k] for k in ("factor_A", "factor_B")}
        returns_fetcher, factor_fetcher = _make_fetchers(small, y)
        out = reverse_find_factors(
            ticker="SYNTH",
            candidate_factor_ids=list(small.keys()),
            start=date(2025, 1, 1),
            end=date(2025, 12, 31),
            k=5,
            returns_fetcher=returns_fetcher,
            factor_fetcher=factor_fetcher,
        )
        assert len(out["top_factors"]) <= 2

    def test_ticker_with_no_data_returns_empty(self) -> None:
        _, factors, _ = _build_synthetic_universe(n=250, seed=2)

        def returns_fetcher(ticker, start, end, return_type="log"):
            return pd.Series(dtype=float, name="r")

        def factor_fetcher(factor_id, start, end):
            return factors[factor_id]

        out = reverse_find_factors(
            ticker="DEAD",
            candidate_factor_ids=list(factors.keys()),
            start=date(2025, 1, 1),
            end=date(2025, 12, 31),
            k=3,
            returns_fetcher=returns_fetcher,
            factor_fetcher=factor_fetcher,
        )
        assert out["top_factors"] == []
        assert out["n_obs"] == 0
        assert "rejected" in out

    def test_factor_with_no_data_is_skipped(self) -> None:
        _, factors, y = _build_synthetic_universe(n=250, seed=3)
        # Make factor_A return an empty frame to verify it gets rejected.
        factors_modded = dict(factors)
        factors_modded["factor_A"] = pd.DataFrame(
            columns=["price"], index=pd.DatetimeIndex([], tz="UTC")
        )
        returns_fetcher, factor_fetcher = _make_fetchers(factors_modded, y)

        out = reverse_find_factors(
            ticker="SYNTH",
            candidate_factor_ids=list(factors_modded.keys()),
            start=date(2025, 1, 1),
            end=date(2025, 12, 31),
            k=4,
            returns_fetcher=returns_fetcher,
            factor_fetcher=factor_fetcher,
        )
        assert "factor_A" in out["rejected"]
        # factor_B (the surviving informative factor) should still be picked.
        assert any(p["factor_id"] == "factor_B" for p in out["top_factors"])

    def test_empty_candidate_list_raises(self) -> None:
        _, factors, y = _build_synthetic_universe(n=120, seed=4)
        returns_fetcher, factor_fetcher = _make_fetchers(factors, y)
        with pytest.raises(ValueError, match="candidate_factor_ids"):
            reverse_find_factors(
                ticker="SYNTH",
                candidate_factor_ids=[],
                start=date(2025, 1, 1),
                end=date(2025, 12, 31),
                k=3,
                returns_fetcher=returns_fetcher,
                factor_fetcher=factor_fetcher,
            )

    def test_invalid_k_raises(self) -> None:
        _, factors, y = _build_synthetic_universe(n=120, seed=5)
        returns_fetcher, factor_fetcher = _make_fetchers(factors, y)
        with pytest.raises(ValueError, match="k must be"):
            reverse_find_factors(
                ticker="SYNTH",
                candidate_factor_ids=["factor_A"],
                start=date(2025, 1, 1),
                end=date(2025, 12, 31),
                k=0,
                returns_fetcher=returns_fetcher,
                factor_fetcher=factor_fetcher,
            )


# --- pure-function tests: prediction_driven_alpha --------------------------


class TestPredictionDrivenAlpha:
    def test_basic_scan_returns_rows_for_each_ticker(self) -> None:
        _, factors, y = _build_synthetic_universe(n=250, seed=6)

        # We need a scan over multiple tickers — fabricate distinct
        # synthetic returns per ticker.
        rng = np.random.default_rng(99)

        def returns_fetcher(ticker, start, end, return_type="log"):
            # Different β per ticker so we can verify ranking behaviour.
            shift = (hash(ticker) % 7) / 10.0
            base = y * (0.5 + shift)  # reuse y's index/timing
            noise = rng.normal(0, 5e-4, len(base))
            return base + pd.Series(noise, index=base.index)

        def factor_fetcher(factor_id, start, end):
            df = factors[factor_id]
            mask = (df.index >= start) & (df.index <= end)
            return df.loc[mask]

        end_d = y.index[-1].date()
        out = prediction_driven_alpha(
            factor_id="factor_A",
            candidate_tickers=["AAA", "BBB", "CCC", "DDD"],
            window_days=300,
            top_n=4,
            end=end_d,
            returns_fetcher=returns_fetcher,
            factor_fetcher=factor_fetcher,
        )

        assert out["factor_id"] == "factor_A"
        assert out["ranked_by"] == "abs(beta)*r_squared"
        assert len(out["tickers"]) == 4
        # Rows are ranked descending by |beta| * r_squared.
        scores = [abs(r["beta"]) * max(r["r_squared"], 0.0) for r in out["tickers"]]
        assert scores == sorted(scores, reverse=True)

    def test_expected_return_pct_populated_when_delta_provided(self) -> None:
        _, factors, y = _build_synthetic_universe(n=250, seed=7)
        returns_fetcher, factor_fetcher = _make_fetchers(factors, y)
        end_d = y.index[-1].date()

        out = prediction_driven_alpha(
            factor_id="factor_A",
            candidate_tickers=["AAA"],
            window_days=300,
            top_n=1,
            delta_logit_assumed=0.5,
            end=end_d,
            returns_fetcher=returns_fetcher,
            factor_fetcher=factor_fetcher,
        )
        row = out["tickers"][0]
        assert row["expected_return_pct"] is not None
        assert row["expected_return_pct"] == pytest.approx(row["beta"] * 0.5 * 100.0, rel=1e-6)

    def test_top_n_caps_response(self) -> None:
        _, factors, y = _build_synthetic_universe(n=250, seed=8)
        returns_fetcher, factor_fetcher = _make_fetchers(factors, y)
        end_d = y.index[-1].date()

        out = prediction_driven_alpha(
            factor_id="factor_A",
            candidate_tickers=["A", "B", "C", "D", "E"],
            window_days=300,
            top_n=2,
            end=end_d,
            returns_fetcher=returns_fetcher,
            factor_fetcher=factor_fetcher,
        )
        assert len(out["tickers"]) == 2

    def test_default_basket_used_when_none(self) -> None:
        _, factors, y = _build_synthetic_universe(n=250, seed=9)
        returns_fetcher, factor_fetcher = _make_fetchers(factors, y)
        end_d = y.index[-1].date()

        out = prediction_driven_alpha(
            factor_id="factor_A",
            candidate_tickers=None,
            window_days=300,
            top_n=50,  # high enough to keep all defaults
            end=end_d,
            returns_fetcher=returns_fetcher,
            factor_fetcher=factor_fetcher,
        )
        # All default tickers were attempted; the scan returns >= 1 row even
        # with the synthetic single-y fetcher (it returns the same series for
        # every ticker, which still produces a valid univariate fit).
        # We just want to verify the basket fell back to the defaults.
        seen = {r["ticker"] for r in out["tickers"]} | {s["ticker"] for s in out.get("skipped", [])}
        assert seen == set(DEFAULT_TICKERS)

    def test_window_too_small_raises(self) -> None:
        _, factors, y = _build_synthetic_universe(n=120, seed=10)
        returns_fetcher, factor_fetcher = _make_fetchers(factors, y)
        with pytest.raises(ValueError, match="window_days"):
            prediction_driven_alpha(
                factor_id="factor_A",
                candidate_tickers=["AAA"],
                window_days=20,
                top_n=1,
                returns_fetcher=returns_fetcher,
                factor_fetcher=factor_fetcher,
            )


# --- router-level integration test -----------------------------------------


@pytest.fixture
def router_app_client(monkeypatch, factors_file, fake_factor_history, fake_log_returns):
    """A TestClient that mounts ``reverse_finder_router`` onto ``main.app``.

    We reuse the same monkey-patches the project-wide ``app_client`` fixture
    uses so external IO is fully stubbed.
    """
    monkeypatch.setenv("FACTORS_FILE", str(factors_file))
    import pfm.config as cfg

    cfg._settings = None

    import pfm.main as main_mod

    monkeypatch.setattr(main_mod, "fetch_factor_history", fake_factor_history)
    monkeypatch.setattr(main_mod, "get_log_returns", fake_log_returns)

    from pfm.cache import NullCache

    monkeypatch.setattr(main_mod, "RedisCache", lambda url: NullCache())

    # Mount the new router. Use a flag attribute so we don't re-mount on a
    # subsequent test fixture invocation in the same process.
    from pfm.reverse_finder_router import router as rff_router

    if not getattr(main_mod.app.state, "_rff_mounted", False):
        main_mod.app.include_router(rff_router)
        main_mod.app.state._rff_mounted = True  # type: ignore[attr-defined]

    with TestClient(main_mod.app) as client:
        yield client


def test_router_reverse_finder_returns_top_factors(router_app_client: TestClient) -> None:
    r = router_app_client.post(
        "/reverse-finder",
        json={
            "ticker": "TEST",
            "start": "2025-06-15",
            "end": "2025-12-15",
            "candidate_factor_ids": ["factor_a", "factor_b"],
            "k": 2,
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ticker"] == "TEST"
    ids = {f["factor_id"] for f in body["top_factors"]}
    assert ids.issubset({"factor_a", "factor_b"})
    assert body["n_obs"] >= 30
    # factor_name should be enriched from factors.yml.
    if body["top_factors"]:
        assert body["top_factors"][0]["factor_name"] in {"Factor A", "Factor B"}


def test_router_reverse_finder_invalid_dates_400(router_app_client: TestClient) -> None:
    r = router_app_client.post(
        "/reverse-finder",
        json={"ticker": "TEST", "start": "2025-12-15", "end": "2025-06-15", "k": 2},
    )
    assert r.status_code == 400


def test_router_prediction_alpha_returns_rows(router_app_client: TestClient) -> None:
    r = router_app_client.post(
        "/alpha/prediction-driven",
        json={
            "factor_id": "factor_a",
            "candidate_tickers": ["AAA", "BBB", "CCC"],
            "window_days": 200,
            "top_n": 3,
            "delta_logit_assumed": 0.25,
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["factor_id"] == "factor_a"
    assert body["factor_name"] == "Factor A"
    assert body["delta_logit_assumed"] == 0.25
    for row in body["tickers"]:
        assert row["expected_return_pct"] is not None


def test_router_prediction_alpha_unknown_factor_404(router_app_client: TestClient) -> None:
    r = router_app_client.post(
        "/alpha/prediction-driven",
        json={"factor_id": "does_not_exist"},
    )
    assert r.status_code == 404


def test_default_pool_uses_curated_200(router_app_client: TestClient) -> None:
    """Default ``pool="curated"`` must report a ``curated_N`` pool with N ≤ 200.

    The conftest factor catalogue only has 2 polymarket factors so we
    expect ``curated_2``; the upper bound is what matters for the
    contract.
    """
    r = router_app_client.post(
        "/reverse-finder",
        json={
            "ticker": "TEST",
            "start": "2025-06-15",
            "end": "2025-12-15",
            # no candidate_factor_ids → curated pool kicks in
            "k": 2,
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["pool_used"].startswith("curated_")
    assert body["n_candidates_evaluated"] <= 200
    # With only 2 fixture factors the count is exactly 2.
    assert body["n_candidates_evaluated"] == 2


def test_pool_all_opt_in(router_app_client: TestClient) -> None:
    """`pool=all` must report an ``all_N`` pool that equals the full catalogue."""
    r = router_app_client.post(
        "/reverse-finder?pool=all",
        json={"ticker": "TEST", "start": "2025-06-15", "end": "2025-12-15", "k": 2},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["pool_used"].startswith("all_")
    assert body["n_candidates_evaluated"] >= 2
