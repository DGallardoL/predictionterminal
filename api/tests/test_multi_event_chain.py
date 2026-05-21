"""Synthetic-DGP tests for ``pfm.multi_event_chain``.

Each test builds a known data-generating process and checks the public
function recovers the planted structure. We do *not* hit any external
API: every test injects pure-Python fetcher callables.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pfm.multi_event_chain import (
    event_macro_correlation,
    extract_systemic_pm_factor,
    find_chains,
    fit_multi_event_lasso,
    sector_attribution,
)

# ───────────────────────────── helpers ────────────────────────────────


def _utc_idx(n: int, start: str = "2024-01-01") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq="D", tz="UTC").normalize()


def _ar1_to_prob(noise: np.ndarray, *, phi: float = 0.92) -> np.ndarray:
    """Build a probability series via AR(1) on the logit then sigmoid."""
    z = np.zeros_like(noise)
    for t in range(1, len(noise)):
        z[t] = phi * z[t - 1] + noise[t]
    p = 1.0 / (1.0 + np.exp(-z))
    return np.clip(p, 0.05, 0.95)


def _make_factor_fetcher(price_bank: dict[str, pd.Series]):
    def _fetch(fid, start, end):
        if fid not in price_bank:
            return pd.Series(dtype=float)
        s = price_bank[fid]
        s = s[(s.index >= start) & (s.index <= end)]
        return s

    return _fetch


def _make_returns_fetcher(returns_bank: dict[str, pd.Series]):
    def _fetch(ticker, start, end):
        if ticker not in returns_bank:
            return pd.Series(dtype=float)
        s = returns_bank[ticker]
        s = s[(s.index >= start) & (s.index <= end)]
        return s

    return _fetch


def _make_macro_fetcher(macro_bank: dict[str, pd.Series]):
    def _fetch(mid, start, end):
        if mid not in macro_bank:
            return pd.Series(dtype=float)
        s = macro_bank[mid]
        s = s[(s.index >= start) & (s.index <= end)]
        return s

    return _fetch


# ───────────────────────────── A. LASSO ────────────────────────────────


class TestMultiEventLasso:
    def test_recovers_3_real_factors_among_50(self) -> None:
        """50 candidate factors but only 3 truly drive the ticker — Lasso
        should leave most coefs at zero and recover the 3 real ones."""
        rng = np.random.default_rng(seed=42)
        n = 240
        idx = _utc_idx(n)
        n_factors = 50

        # Build 50 independent AR(1) probability series.
        price_bank: dict[str, pd.Series] = {}
        deltas: dict[str, np.ndarray] = {}
        for i in range(n_factors):
            noise = rng.normal(0, 0.4, size=n)
            p = _ar1_to_prob(noise, phi=0.85)
            s = pd.Series(p, index=idx, name=f"f{i}")
            price_bank[f"f{i}"] = s
            # Δlogit will be derived inside the function. Save for ground truth.
            from pfm.model import delta_logit

            deltas[f"f{i}"] = delta_logit(s).fillna(0.0).to_numpy()

        # Three "real" drivers with sizable betas.
        real = ["f3", "f17", "f29"]
        true_betas = {"f3": 0.04, "f17": -0.03, "f29": 0.05}
        y = np.zeros(n)
        for fid, b in true_betas.items():
            y += b * deltas[fid]
        y += rng.normal(0, 0.001, size=n)
        returns_bank = {"AAPL": pd.Series(y, index=idx, name="AAPL")}

        result = fit_multi_event_lasso(
            ticker="AAPL",
            factor_ids=[f"f{i}" for i in range(n_factors)],
            start="2024-01-01",
            end=idx[-1].strftime("%Y-%m-%d"),
            alpha=0.0001,  # only used as fallback
            fetch_factor=_make_factor_fetcher(price_bank),
            fetch_returns=_make_returns_fetcher(returns_bank),
        )

        assert result["ticker"] == "AAPL"
        assert result["n_factors_in"] == 50
        # Most coefs should be zeroed by L1.
        assert result["n_factors_nonzero"] <= 30
        # The three real factors should appear among the surviving factors.
        survivors = set(result["sparse_solution_factors"])
        recovered = sum(1 for r in real if r in survivors)
        assert recovered >= 2, f"expected ≥2 of {real} to survive Lasso; got {sorted(survivors)}"
        # And R² should be reasonably high since the DGP is mostly explained.
        assert result["r_squared"] > 0.30

    def test_empty_factor_set_returns_zero(self) -> None:
        result = fit_multi_event_lasso(
            ticker="AAPL",
            factor_ids=[],
            start="2024-01-01",
            end="2024-06-01",
            fetch_factor=_make_factor_fetcher({}),
            fetch_returns=_make_returns_fetcher({}),
        )
        assert result["n_factors_in"] == 0
        assert result["n_factors_nonzero"] == 0


# ───────────────────────── B. Sector attribution ───────────────────────


class TestSectorAttribution:
    def test_synthetic_sectors_load_on_their_drivers(self) -> None:
        """3 sectors, 3 factors. Each sector loads exclusively on one
        factor. Attribution matrix must be near-diagonal (the dominant
        factor for each sector matches the planted driver)."""
        rng = np.random.default_rng(seed=7)
        n = 180
        idx = _utc_idx(n)

        price_bank: dict[str, pd.Series] = {}
        deltas: dict[str, np.ndarray] = {}
        from pfm.model import delta_logit

        for fid, phi in [("growth", 0.88), ("inflation", 0.90), ("election", 0.92)]:
            p = _ar1_to_prob(rng.normal(0, 0.5, n), phi=phi)
            s = pd.Series(p, index=idx, name=fid)
            price_bank[fid] = s
            deltas[fid] = delta_logit(s).fillna(0.0).to_numpy()

        sectors = ["XLK", "XLE", "XLF"]
        drivers = {"XLK": "growth", "XLE": "inflation", "XLF": "election"}
        returns_bank: dict[str, pd.Series] = {}
        for sec in sectors:
            d = deltas[drivers[sec]]
            y = 0.05 * d + rng.normal(0, 0.001, size=n)
            returns_bank[sec] = pd.Series(y, index=idx, name=sec)

        result = sector_attribution(
            sectors_etfs=sectors,
            factor_ids=["growth", "inflation", "election"],
            start="2024-01-01",
            end=idx[-1].strftime("%Y-%m-%d"),
            fetch_factor=_make_factor_fetcher(price_bank),
            fetch_returns=_make_returns_fetcher(returns_bank),
        )

        assert result["sectors"] == sectors
        assert result["factors"] == ["growth", "inflation", "election"]
        # Each sector's dominant factor must be the planted driver.
        for sec in sectors:
            assert result["dominant_factor_per_sector"][sec] == drivers[sec], (
                f"{sec}: dominant={result['dominant_factor_per_sector'][sec]}"
                f" expected={drivers[sec]}"
            )
        # Attribution matrix is 3×3 and largest entry per row is on the diagonal.
        mat = np.array(result["attribution_matrix"])
        for i in range(3):
            assert int(np.argmax(mat[i])) == i

    def test_default_sectors_when_none(self) -> None:
        # Empty bank → no sectors usable; result should be empty but not raise.
        result = sector_attribution(
            sectors_etfs=None,
            factor_ids=["x"],
            start="2024-01-01",
            end="2024-06-01",
            fetch_factor=_make_factor_fetcher({}),
            fetch_returns=_make_returns_fetcher({}),
        )
        assert result["sectors"] == []
        assert result["factors"] == []


# ──────────────────────────── C. Chains ────────────────────────────────


class TestFindChains:
    def test_recovers_synthetic_a_to_b_to_ticker(self) -> None:
        """Build A → B → returns chain via lagged AR copy. Granger should
        find the path A → B → SPY."""
        rng = np.random.default_rng(seed=13)
        n = 260
        idx = _utc_idx(n)
        from pfm.model import delta_logit

        # A: independent AR(1).
        a_logit = np.zeros(n)
        eps_a = rng.normal(0, 0.5, n)
        for t in range(1, n):
            a_logit[t] = 0.9 * a_logit[t - 1] + eps_a[t]
        p_a = np.clip(1.0 / (1.0 + np.exp(-a_logit)), 0.05, 0.95)

        # B: depends on lag-2 of A's logit innovations.
        b_logit = np.zeros(n)
        eps_b = rng.normal(0, 0.2, n)
        for t in range(2, n):
            b_logit[t] = 0.7 * b_logit[t - 1] + 0.6 * eps_a[t - 2] + eps_b[t]
        p_b = np.clip(1.0 / (1.0 + np.exp(-b_logit)), 0.05, 0.95)

        # SPY returns depend on lag-2 of B's Δlogit.
        s_b = pd.Series(p_b, index=idx)
        d_b = delta_logit(s_b).fillna(0.0).to_numpy()
        ret = np.zeros(n)
        for t in range(2, n):
            ret[t] = 0.6 * d_b[t - 2] + rng.normal(0, 0.001)

        price_bank = {
            "A": pd.Series(p_a, index=idx),
            "B": pd.Series(p_b, index=idx),
            "C": pd.Series(_ar1_to_prob(rng.normal(0, 0.5, n)), index=idx),  # noise factor
        }
        returns_bank = {"SPY": pd.Series(ret, index=idx)}

        paths = find_chains(
            start_factor="A",
            end_ticker="SPY",
            candidate_intermediate_factors=["B", "C"],
            max_depth=3,
            start="2024-01-01",
            end=idx[-1].strftime("%Y-%m-%d"),
            fetch_factor=_make_factor_fetcher(price_bank),
            fetch_returns=_make_returns_fetcher(returns_bank),
            p_threshold=0.20,  # generous so synthetic noise still passes
        )

        assert len(paths) >= 1, "expected at least one significant Granger path"
        # The A → B → SPY path must be present.
        path_node_lists = [p["path"] for p in paths]
        assert ["A", "B", "SPY"] in path_node_lists, (
            f"expected ['A','B','SPY'] in {path_node_lists}"
        )

    def test_no_path_when_no_signal(self) -> None:
        rng = np.random.default_rng(seed=99)
        n = 200
        idx = _utc_idx(n)
        bank = {
            "X": pd.Series(_ar1_to_prob(rng.normal(0, 0.4, n)), index=idx),
            "Y": pd.Series(_ar1_to_prob(rng.normal(0, 0.4, n)), index=idx),
        }
        returns_bank = {"QQQ": pd.Series(rng.normal(0, 0.001, n), index=idx)}
        paths = find_chains(
            start_factor="X",
            end_ticker="QQQ",
            candidate_intermediate_factors=["Y"],
            max_depth=3,
            start="2024-01-01",
            end=idx[-1].strftime("%Y-%m-%d"),
            fetch_factor=_make_factor_fetcher(bank),
            fetch_returns=_make_returns_fetcher(returns_bank),
            p_threshold=0.05,
        )
        # Pure noise → typically no significant path. Allow ≤ 1 false positive.
        assert len(paths) <= 1


# ─────────────────────────── D. Macro corr ─────────────────────────────


class TestEventMacroCorrelation:
    def test_recovers_planted_high_correlation(self) -> None:
        """Build a factor whose Δlogit matches Δ(macro) plus small noise.
        Recovered correlation should be large and t-stat significant."""
        rng = np.random.default_rng(seed=21)
        n = 220
        idx = _utc_idx(n)

        # Macro = random walk; Δmacro is the daily innovation.
        d_macro = rng.normal(0, 1.0, n)
        macro_level = pd.Series(np.cumsum(d_macro), index=idx, name="DGS10")

        # Build factor logit so that Δlogit ≈ 0.8·Δmacro + noise.
        target_dlogit = 0.8 * d_macro + rng.normal(0, 0.2, n)
        logit_path = np.cumsum(target_dlogit) - target_dlogit[0]  # start at 0
        # Re-center so prob stays in (0,1).
        logit_path = logit_path - logit_path.mean()
        p = np.clip(1.0 / (1.0 + np.exp(-logit_path / 4.0)), 0.05, 0.95)
        factor_series = pd.Series(p, index=idx, name="rate-cut-2024")

        result = event_macro_correlation(
            factor_id="rate-cut-2024",
            macro_series=["DGS10", "VIXCLS"],  # second one missing
            start="2024-01-01",
            end=idx[-1].strftime("%Y-%m-%d"),
            fetch_factor=_make_factor_fetcher({"rate-cut-2024": factor_series}),
            fetch_macro=_make_macro_fetcher({"DGS10": macro_level}),
            max_lead_lag_days=3,
        )

        assert "DGS10" in result["macro_correlations"]
        assert abs(result["macro_correlations"]["DGS10"]) >= 0.5, (
            f"expected |corr|≥0.5; got {result['macro_correlations']['DGS10']}"
        )
        assert abs(result["t_stats"]["DGS10"]) > 2.0
        # Missing macro silently skipped.
        assert "VIXCLS" not in result["macro_correlations"]


# ────────────────────────── E. Systemic factor ─────────────────────────


class TestExtractSystemicPmFactor:
    def test_first_pc_explains_majority_when_factors_correlated(self) -> None:
        """5 factors driven by a single shared latent → PC1 should
        explain a clear majority of the variance."""
        rng = np.random.default_rng(seed=33)
        n = 260
        idx = _utc_idx(n)

        # Shared latent driver (an AR(1)).
        z = np.zeros(n)
        eps_z = rng.normal(0, 0.5, n)
        for t in range(1, n):
            z[t] = 0.85 * z[t - 1] + eps_z[t]

        price_bank: dict[str, pd.Series] = {}
        for i in range(5):
            # Each factor logit = α_i · z + small idiosyncratic noise.
            alpha = 0.7 + 0.05 * i
            logit_path = alpha * z + rng.normal(0, 0.10, n)
            p = np.clip(1.0 / (1.0 + np.exp(-logit_path)), 0.05, 0.95)
            price_bank[f"sf{i}"] = pd.Series(p, index=idx, name=f"sf{i}")

        result = extract_systemic_pm_factor(
            factor_ids=[f"sf{i}" for i in range(5)],
            n_factors=2,
            start="2024-01-01",
            end=idx[-1].strftime("%Y-%m-%d"),
            fetch_factor=_make_factor_fetcher(price_bank),
        )

        ev = result["explained_variance"]
        assert len(ev) == 2
        assert ev[0] > 0.60, f"PC1 should explain >60%; got {ev[0]:.3f}"
        assert result["can_use_as_factor"] is True
        assert result["n_factors_in"] == 5
        # Loadings: all factors should load with the same sign on the shared
        # latent (after sign-orientation in the implementation, the largest
        # |loading| is positive).
        loadings = list(result["loadings"].values())
        assert all(np.sign(loadings[0]) == np.sign(L) for L in loadings if abs(L) > 0.05)
        # Component scores should match the index length.
        assert len(result["component_scores"]) == len(result["dates"])

    def test_too_few_factors_returns_unusable(self) -> None:
        result = extract_systemic_pm_factor(
            factor_ids=["only_one"],
            n_factors=1,
            start="2024-01-01",
            end="2024-06-01",
            fetch_factor=_make_factor_fetcher({}),
        )
        assert result["can_use_as_factor"] is False
        assert result["component_scores"] == []


# ───────────────────────── Router smoke test ───────────────────────────


class TestRouterRegistration:
    def test_router_has_five_endpoints(self) -> None:
        from pfm.multi_event_chain_router import router

        paths = {r.path for r in router.routes}
        assert "/multi-event/lasso" in paths
        assert "/multi-event/sector-attribution" in paths
        assert "/multi-event/chains" in paths
        assert "/multi-event/macro-correlation" in paths
        assert "/multi-event/systemic-factor" in paths


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-xvs"])
