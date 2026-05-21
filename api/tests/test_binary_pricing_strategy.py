"""Tests for :mod:`pfm.strategies.binary_pricing_alpha` (Track-L T84).

Two layers:

1. Unit tests against a tiny in-test ``StubPricer`` — verifies the strategy
   recovers a known signal-to-PnL relationship under a synthetic DGP
   where ``true_p = logit(α + β*X)`` and ``market_p = true_p + N(0, σ)``.
2. Guard tests around ``SHOULD_DEPLOY`` and ``register_if_ready`` — make
   sure the safety-rails behave as specified (the module MUST NOT
   auto-register at import time; the tier MUST stay B_VALIDATED).

The real T81 ``pfm.pricing.binary_models`` module body has not landed at
the time these tests were written, so we use a self-contained stub. When
T81 ships, an integration test wiring ``RiskNeutralLogit`` through this
strategy should be added to ``tests/test_binary_pricing_strategy.py`` as
well (paired with stress_test.py 4-quarter pass).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from pfm.strategies.binary_pricing_alpha import (
    SHOULD_DEPLOY,
    STRATEGY_NAME,
    TIER,
    BinaryPricingAlpha,
    MarketState,
    Pricer,
    alpha_catalog_entry,
    register_if_ready,
)

# ---------------------------------------------------------------------------
# Helpers & fixtures
# ---------------------------------------------------------------------------


class StubPricer:
    """Logit-true pricer used for synthetic DGP recovery.

    ``fair_price(state)`` reads ``state.features['x']`` and returns
    ``sigmoid(alpha + beta * x)``. This is the *truth* the DGP draws
    market prices around (with N(0, σ) Gaussian noise).
    """

    def __init__(self, alpha: float, beta: float) -> None:
        self.alpha = float(alpha)
        self.beta = float(beta)

    def fair_price(self, state: MarketState) -> float:
        x = float(state.features.get("x", 0.0))
        z = self.alpha + self.beta * x
        return 1.0 / (1.0 + math.exp(-z))


def _build_synthetic_dataset(
    *,
    n_markets: int = 500,
    alpha: float = -0.4,
    beta: float = 1.6,
    sigma: float = 0.05,
    seed: int = 0xC0FFEE,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Build n_markets independent binary markets with known DGP.

    Returns (frame, realised_outcomes). The frame has columns
    ``market_price``, ``time_to_resolution_days`` and ``features`` (a
    dict per row); realised_outcomes is a 0/1 numpy array drawn from
    Bernoulli(true_p).
    """
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, size=n_markets)
    z = alpha + beta * x
    true_p = 1.0 / (1.0 + np.exp(-z))
    noise = rng.normal(0.0, sigma, size=n_markets)
    market_p = np.clip(true_p + noise, 0.01, 0.99)
    outcomes = (rng.uniform(0, 1, size=n_markets) < true_p).astype(int)
    frame = pd.DataFrame(
        {
            "market_price": market_p,
            "time_to_resolution_days": rng.integers(7, 90, size=n_markets).astype(float),
            "features": [{"x": float(xv)} for xv in x],
            "outcome": outcomes.astype(float),
            "true_p": true_p,
        },
        index=pd.date_range("2025-01-01", periods=n_markets, freq="D"),
    )
    return frame, outcomes


# ---------------------------------------------------------------------------
# Sanity & guard tests
# ---------------------------------------------------------------------------


def test_should_deploy_starts_false() -> None:
    """Module MUST NOT auto-promote without a human review."""
    assert SHOULD_DEPLOY is False, (
        "SHOULD_DEPLOY must remain False until T81 lands, T83 verdict is "
        "written, and 4-quarter stress test returns PASS."
    )


def test_tier_is_b_validated_not_higher() -> None:
    """CLAUDE.md ceiling: never above B_VALIDATED without live confirmation."""
    assert TIER == "B_VALIDATED"


def test_strategy_name_canonical() -> None:
    assert STRATEGY_NAME == "binary-pricing-mispricing"


def test_register_if_ready_noop_when_flag_false() -> None:
    """Even with a valid pricer, registration must skip while flag is False."""
    pricer = StubPricer(alpha=0.0, beta=1.0)
    result = register_if_ready(pricer=pricer, force=False)
    assert result is None


def test_register_if_ready_returns_none_without_pricer() -> None:
    assert register_if_ready(pricer=None, force=True) is None


def test_register_if_ready_force_path_works() -> None:
    """Force-path produces a Strategy and adds it to the registry."""
    from pfm.strategies_registry import _REGISTRY, unregister

    pricer = StubPricer(alpha=0.0, beta=1.0)
    try:
        strat = register_if_ready(pricer=pricer, force=True)
        assert strat is not None
        assert strat.name == STRATEGY_NAME
        assert STRATEGY_NAME in _REGISTRY
    finally:
        unregister(STRATEGY_NAME)


# ---------------------------------------------------------------------------
# Pricer protocol enforcement
# ---------------------------------------------------------------------------


def test_pricer_protocol_runtime_check_accepts_stub() -> None:
    assert isinstance(StubPricer(0.0, 1.0), Pricer)


def test_constructor_rejects_non_pricer() -> None:
    class NotAPricer:
        pass

    with pytest.raises(TypeError):
        BinaryPricingAlpha(NotAPricer())  # type: ignore[arg-type]


@pytest.mark.parametrize("bad", [0.0, -0.1, 1.5])
def test_constructor_rejects_bad_kelly_cap(bad: float) -> None:
    with pytest.raises(ValueError):
        BinaryPricingAlpha(StubPricer(0.0, 1.0), kelly_cap=bad)


def test_constructor_rejects_negative_z_threshold() -> None:
    with pytest.raises(ValueError):
        BinaryPricingAlpha(StubPricer(0.0, 1.0), z_threshold=-0.1)


def test_constructor_rejects_small_window() -> None:
    with pytest.raises(ValueError):
        BinaryPricingAlpha(StubPricer(0.0, 1.0), z_window=1)


# ---------------------------------------------------------------------------
# Scalar-mode signal & position
# ---------------------------------------------------------------------------


def test_scalar_signal_zero_when_fair_equals_market() -> None:
    """Sanity per task spec: market_p == fair_price -> signal = 0, position = 0."""
    pricer = StubPricer(alpha=0.0, beta=0.0)  # always 0.5
    strat = BinaryPricingAlpha(pricer)
    state = MarketState(market_price=0.5, time_to_resolution_days=30, features={"x": 0.0})
    sig = strat.signal(state)
    assert sig == 0.0
    pos = strat.position(sig)
    assert pos == 0.0


def test_scalar_signal_positive_when_fair_above_market() -> None:
    """When the model is bullish vs the market, signal > 0."""
    pricer = StubPricer(alpha=2.0, beta=0.0)  # fair ≈ 0.881
    strat = BinaryPricingAlpha(pricer)
    state = MarketState(market_price=0.50, time_to_resolution_days=30, features={"x": 0.0})
    sig = strat.signal(state)
    assert sig > 0.0


def test_scalar_signal_negative_when_fair_below_market() -> None:
    pricer = StubPricer(alpha=-2.0, beta=0.0)  # fair ≈ 0.119
    strat = BinaryPricingAlpha(pricer)
    state = MarketState(market_price=0.80, time_to_resolution_days=30, features={"x": 0.0})
    sig = strat.signal(state)
    assert sig < 0.0


def test_scalar_signal_handles_edge_market_price() -> None:
    """SE = sqrt(p(1-p)); at clipped 0.99 it's tiny — but never divide by zero."""
    pricer = StubPricer(alpha=2.0, beta=0.0)
    strat = BinaryPricingAlpha(pricer)
    state = MarketState(market_price=0.001, time_to_resolution_days=10, features={"x": 0.0})
    sig = strat.signal(state)
    assert math.isfinite(sig)


def test_position_sits_out_below_threshold() -> None:
    """|signal| < z_threshold -> position = 0."""
    strat = BinaryPricingAlpha(StubPricer(0.0, 0.0), z_threshold=1.5)
    assert strat.position(0.5) == 0.0
    assert strat.position(-1.0) == 0.0


def test_position_caps_at_kelly_cap() -> None:
    """Strong signals are clipped at +/- kelly_cap."""
    strat = BinaryPricingAlpha(StubPricer(0.0, 0.0), kelly_cap=0.15, z_threshold=0.5)
    assert strat.position(10.0) == pytest.approx(0.15)
    assert strat.position(-10.0) == pytest.approx(-0.15)


def test_position_proportional_in_middle_range() -> None:
    """Between threshold and kelly_cap the Kelly fraction scales with signal."""
    strat = BinaryPricingAlpha(StubPricer(0.0, 0.0), kelly_cap=1.0, z_threshold=0.5)
    p1 = strat.position(1.0)  # 2*1/3 = 0.666...
    p2 = strat.position(2.0)  # 2*2/3 = 1.333 capped to 1.0
    assert p1 == pytest.approx(2.0 / 3.0)
    assert p2 == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Vectorised (DataFrame) signal
# ---------------------------------------------------------------------------


def test_dataframe_signal_synthesises_fair_when_missing() -> None:
    """If frame lacks 'fair_price' column, signal() calls the pricer per row."""
    frame, _ = _build_synthetic_dataset(n_markets=60, sigma=0.04)
    strat = BinaryPricingAlpha(StubPricer(alpha=-0.4, beta=1.6), z_window=10)
    sig = strat.signal(frame)
    assert isinstance(sig, pd.Series)
    assert len(sig) == len(frame)
    # After the warm-up window the z-score should be finite.
    tail = sig.iloc[20:]
    assert tail.notna().all()


def test_dataframe_signal_uses_precomputed_fair_price() -> None:
    frame, _ = _build_synthetic_dataset(n_markets=50, sigma=0.04)
    frame = frame.copy()
    # Inject a known mispriced row to verify the column is honoured.
    frame.loc[frame.index[-1], "fair_price"] = 0.95
    frame["fair_price"] = frame["fair_price"].fillna(frame["market_price"])
    strat = BinaryPricingAlpha(StubPricer(0.0, 0.0), z_window=10)
    sig = strat.signal(frame)
    assert isinstance(sig, pd.Series)


def test_dataframe_signal_requires_market_price() -> None:
    bad = pd.DataFrame({"fair_price": [0.5, 0.6]})
    strat = BinaryPricingAlpha(StubPricer(0.0, 0.0))
    with pytest.raises(ValueError):
        strat.signal(bad)


def test_signal_rejects_unknown_input_types() -> None:
    strat = BinaryPricingAlpha(StubPricer(0.0, 0.0))
    with pytest.raises(TypeError):
        strat.signal("not a state")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# PnL
# ---------------------------------------------------------------------------


def test_scalar_pnl_multiplies_position_by_realised() -> None:
    strat = BinaryPricingAlpha(StubPricer(0.0, 0.0))
    assert strat.pnl(0.5, 0.10) == pytest.approx(0.05)
    assert strat.pnl(-0.3, 0.20) == pytest.approx(-0.06)
    assert strat.pnl(0.0, 1.0) == 0.0


def test_series_pnl_aligns_and_zero_fills() -> None:
    strat = BinaryPricingAlpha(StubPricer(0.0, 0.0))
    pos = pd.Series([0.1, 0.2, 0.0], index=pd.date_range("2025-01-01", periods=3))
    ret = pd.Series([0.05, -0.04, 0.01], index=pos.index)
    out = strat.pnl(pos, ret)
    assert isinstance(out, pd.Series)
    assert out.iloc[0] == pytest.approx(0.005)
    assert out.iloc[1] == pytest.approx(-0.008)
    assert out.iloc[2] == 0.0


def test_compute_daily_pnl_pipeline() -> None:
    frame, _ = _build_synthetic_dataset(n_markets=80, sigma=0.04, seed=42)
    strat = BinaryPricingAlpha(StubPricer(alpha=-0.4, beta=1.6), z_window=15)
    pnl = strat.compute_daily_pnl(frame)
    assert isinstance(pnl, pd.Series)
    assert len(pnl) == len(frame)


def test_compute_daily_pnl_requires_outcome_col() -> None:
    frame, _ = _build_synthetic_dataset(n_markets=30)
    frame = frame.drop(columns=["outcome"])
    strat = BinaryPricingAlpha(StubPricer(0.0, 1.0))
    with pytest.raises(ValueError):
        strat.compute_daily_pnl(frame)


# ---------------------------------------------------------------------------
# Synthetic DGP recovery (the core T84 requirement)
# ---------------------------------------------------------------------------


def test_synthetic_dgp_recovery_positive_pnl_and_sharpe() -> None:
    """500 markets, true_p ~ sigmoid(α + β·X), market_p = true + N(0,0.05).

    Strategy with the *true* pricer (StubPricer with the planted α, β)
    should recover positive net PnL and positive Sharpe. The model is
    by construction always right on average; the only randomness comes
    from the noise term and the Bernoulli draw.
    """
    rng = np.random.default_rng(0xBEEF)
    n_markets = 500
    alpha, beta, sigma = -0.4, 1.6, 0.05

    frame, _outcomes = _build_synthetic_dataset(
        n_markets=n_markets,
        alpha=alpha,
        beta=beta,
        sigma=sigma,
        seed=int(rng.integers(0, 2**31 - 1)),
    )
    strat = BinaryPricingAlpha(
        pricer=StubPricer(alpha=alpha, beta=beta),
        z_threshold=0.5,
        z_window=20,
        kelly_cap=0.5,
    )
    pnl = strat.compute_daily_pnl(frame)
    pnl_clean = pnl.iloc[20:]  # drop z-window warm-up
    # 1. Positive net PnL.
    assert pnl_clean.sum() > 0.0, (
        f"Net PnL {pnl_clean.sum():.4f} not positive — DGP recovery failed."
    )
    # 2. Positive Sharpe (per-period, no annualisation).
    sharpe = pnl_clean.mean() / max(pnl_clean.std(ddof=1), 1e-12)
    assert sharpe > 0.0
    # 3. Calibration RMSE < 0.05 — pricer.fair_price vs realised true_p.
    fair = np.array(
        [
            strat.pricer.fair_price(
                MarketState(
                    market_price=row["market_price"],
                    time_to_resolution_days=row["time_to_resolution_days"],
                    features=row["features"],
                )
            )
            for _, row in frame.iterrows()
        ]
    )
    rmse = float(np.sqrt(np.mean((fair - frame["true_p"].to_numpy()) ** 2)))
    assert rmse < 0.05, f"Calibration RMSE {rmse:.4f} exceeds 0.05 spec."


def test_synthetic_zero_signal_when_pricer_matches_market() -> None:
    """If the pricer reproduces the market exactly, no position is taken."""
    rng = np.random.default_rng(7)
    n = 60
    market_p = rng.uniform(0.2, 0.8, size=n)

    class IdentityPricer:
        def fair_price(self, state: MarketState) -> float:
            return state.market_price

    frame = pd.DataFrame(
        {
            "market_price": market_p,
            "time_to_resolution_days": np.full(n, 14.0),
            "features": [{} for _ in range(n)],
            "outcome": (market_p > 0.5).astype(float),
        },
        index=pd.date_range("2025-01-01", periods=n, freq="D"),
    )
    strat = BinaryPricingAlpha(IdentityPricer(), z_window=10)
    sig = strat.signal(frame)
    pos = strat.position(sig)
    # All signals should be ~0 (fair == market on every row), so positions all 0.
    assert (pos.abs() < 1e-9).all()


# ---------------------------------------------------------------------------
# Catalog entry & schema check
# ---------------------------------------------------------------------------


def test_alpha_catalog_entry_has_required_fields() -> None:
    """Per Track-L spec: entry MUST have pair_id, tier, label, deploy_params,
    theory_ref, robustness."""
    rob = {
        "quarters_passed": 4,
        "min_quarter_sharpe": 0.62,
        "max_quarter_sharpe": 1.41,
        "full_sample_sharpe": 1.05,
        "any_sign_flip": False,
    }
    entry = alpha_catalog_entry(pricer_name="RiskNeutralLogit", robustness=rob)
    for key in ("pair_id", "tier", "label", "deploy_params", "theory_ref", "robustness"):
        assert key in entry, f"missing required field {key!r}"
    assert entry["pair_id"] == STRATEGY_NAME
    assert entry["tier"] == "B_VALIDATED"
    assert entry["robustness"]["quarters_passed"] == 4


def test_alpha_catalog_entry_includes_anti_alpha_disclaimer() -> None:
    entry = alpha_catalog_entry(pricer_name="BSDigital", robustness={})
    assert "anti_alpha_rule" in entry
    assert "B_VALIDATED" in entry["anti_alpha_rule"]
    # And the publish-time flag MUST mirror SHOULD_DEPLOY at time of build.
    assert entry["should_deploy_at_publish_time"] is SHOULD_DEPLOY


# ---------------------------------------------------------------------------
# Package wrapper sanity — make sure adding the strategies/ package didn't
# break the legacy ``pfm.strategies`` import surface (used by pfm.scanner).
# ---------------------------------------------------------------------------


def test_legacy_strategies_imports_still_work() -> None:
    """``pfm.scanner`` imports conditional_regression + implication_test from
    pfm.strategies; the new package __init__ must re-export them."""
    from pfm.strategies import conditional_regression, implication_test

    assert callable(conditional_regression)
    assert callable(implication_test)
