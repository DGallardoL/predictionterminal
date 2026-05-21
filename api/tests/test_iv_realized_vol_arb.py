"""Tests for :mod:`pfm.strategies.iv_realized_vol_arb` (W12-26).

Coverage layers:

1. Sanity / guard tests — module MUST stay B_VALIDATED with
   :data:`SHOULD_DEPLOY` ``False`` and MUST NOT auto-register in the
   :mod:`pfm.strategies_registry`.
2. Math helpers — :func:`raw_divergence`, :func:`vol_budget`,
   :func:`raw_signal` over edge cases (NaN, T=0, σ=0, extreme probs).
3. Synthetic-DGP recovery — under a planted divergence DGP (``p_poly``
   systematically richer than ``p_options``), the strategy must take
   net-negative ("short YES") positions and produce a meaningfully
   positive Sharpe (≥ 0.4) over a long simulation.
4. No-divergence DGP — when ``p_poly ≈ p_options`` the strategy must
   sit out (|signal| small, |position| ~ 0).
5. Reproducibility — identical seeds give identical PnL series.
6. Edge cases — 1-observation frame, NaN inputs, T=0, missing columns.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from pfm.strategies.iv_realized_vol_arb import (
    SHOULD_DEPLOY,
    STRATEGY_NAME,
    TIER,
    IVRealizedVolArb,
    IVRealizedVolState,
    alpha_catalog_entry,
    raw_divergence,
    raw_signal,
    register_if_ready,
    vol_budget,
)

# ---------------------------------------------------------------------------
# Synthetic-DGP helpers
# ---------------------------------------------------------------------------


def _build_planted_divergence_dataset(
    *,
    n_days: int = 600,
    poly_bias: float = 0.08,
    spike_amp: float = 0.10,
    spike_prob: float = 0.10,
    noise: float = 0.02,
    realized_vol: float = 0.20,
    days_to_event: float = 60.0,
    seed: int = 0xBEEF,
) -> pd.DataFrame:
    """Build a DGP where ``p_poly`` is systematically richer than ``p_options``.

    The structure has two layers:

    1. A *baseline* positive bias ``poly_bias`` (constant). The strategy's
       rolling z-score subtracts a trailing mean, so the baseline alone is
       invisible — but it makes the **z-score=0 case still profitable**
       on the short side via the captured spikes.
    2. *Episodic positive spikes* (with probability ``spike_prob`` per day,
       amplitude ``spike_amp``). These are what the rolling-z detects:
       extra exuberance days where ``p_poly`` jumps above its trailing
       baseline. The strategy should short YES on those days.

    The true resolution is drawn from ``Bernoulli(p_options)`` (i.e.
    options are the unbiased prior). Because ``p_poly`` is systematically
    richer than ``p_options`` AND has episodic over-shoots, the correct
    trade is to **short YES on Polymarket** — position should be negative
    on average, and PnL (= position × (outcome - p_poly)) should be
    positive in expectation.
    """
    rng = np.random.default_rng(seed)
    # Slow random walk in the underlying probability so z-scores are non-zero.
    base = np.clip(0.40 + 0.10 * np.cumsum(rng.normal(0, 0.02, n_days)), 0.10, 0.90)
    p_options = base
    # Episodic positive spikes on top of the constant bias.
    spike_mask = rng.uniform(0, 1, n_days) < spike_prob
    spikes = spike_mask.astype(float) * spike_amp
    p_poly = np.clip(
        p_options + poly_bias + spikes + rng.normal(0.0, noise, n_days),
        0.01,
        0.99,
    )
    outcomes = (rng.uniform(0, 1, n_days) < p_options).astype(float)
    frame = pd.DataFrame(
        {
            "implied_prob_poly": p_poly,
            "implied_prob_options": p_options,
            "days_to_event": float(days_to_event),
            "realized_vol": float(realized_vol),
            "outcome": outcomes,
        },
        index=pd.date_range("2025-01-01", periods=n_days, freq="D"),
    )
    return frame


def _build_no_divergence_dataset(
    *,
    n_days: int = 600,
    noise: float = 0.01,
    realized_vol: float = 0.20,
    days_to_event: float = 60.0,
    seed: int = 0xCAFE,
) -> pd.DataFrame:
    """DGP with ``p_poly ≈ p_options``; signal should fluctuate around 0."""
    rng = np.random.default_rng(seed)
    base = np.clip(0.50 + 0.10 * np.cumsum(rng.normal(0, 0.015, n_days)), 0.10, 0.90)
    p_options = base
    p_poly = np.clip(p_options + rng.normal(0.0, noise, n_days), 0.01, 0.99)
    outcomes = (rng.uniform(0, 1, n_days) < p_options).astype(float)
    return pd.DataFrame(
        {
            "implied_prob_poly": p_poly,
            "implied_prob_options": p_options,
            "days_to_event": float(days_to_event),
            "realized_vol": float(realized_vol),
            "outcome": outcomes,
        },
        index=pd.date_range("2025-01-01", periods=n_days, freq="D"),
    )


# ---------------------------------------------------------------------------
# Sanity & guard tests
# ---------------------------------------------------------------------------


def test_tier_is_b_validated_and_not_auto_deployed() -> None:
    """Per CLAUDE.md anti-alpha rule: B_VALIDATED + SHOULD_DEPLOY=False."""
    assert TIER == "B_VALIDATED"
    assert SHOULD_DEPLOY is False
    assert STRATEGY_NAME == "iv-realized-vol-arb"


def test_register_if_ready_is_gated_off_by_default() -> None:
    """``register_if_ready()`` MUST return None while SHOULD_DEPLOY is False."""
    assert register_if_ready() is None


def test_alpha_catalog_entry_reflects_b_validated_gate() -> None:
    entry = alpha_catalog_entry()
    assert entry["tier"] == "B_VALIDATED"
    assert entry["should_deploy_at_publish_time"] is False
    assert "anti_alpha_gate" in entry


def test_constructor_rejects_bad_params() -> None:
    with pytest.raises(ValueError):
        IVRealizedVolArb(kelly_cap=0.0)
    with pytest.raises(ValueError):
        IVRealizedVolArb(kelly_cap=1.5)
    with pytest.raises(ValueError):
        IVRealizedVolArb(z_threshold=-0.1)
    with pytest.raises(ValueError):
        IVRealizedVolArb(z_window=1)
    with pytest.raises(ValueError):
        IVRealizedVolArb(clip_eps=0.0)
    with pytest.raises(ValueError):
        IVRealizedVolArb(clip_eps=0.5)
    with pytest.raises(ValueError):
        IVRealizedVolArb(fade_sign=0)


# ---------------------------------------------------------------------------
# Math helper tests
# ---------------------------------------------------------------------------


def test_raw_divergence_clips_and_subtracts() -> None:
    assert raw_divergence(0.7, 0.5) == pytest.approx(0.2, abs=1e-12)
    # Clipping protects ends.
    assert raw_divergence(0.0, 0.5, eps=0.01) == pytest.approx(0.01 - 0.5, abs=1e-12)
    assert raw_divergence(1.0, 0.5, eps=0.01) == pytest.approx(0.99 - 0.5, abs=1e-12)


def test_raw_divergence_handles_nan() -> None:
    val = raw_divergence(float("nan"), 0.5)
    # NaN clips to 0.5, so divergence becomes 0.
    assert val == pytest.approx(0.0, abs=1e-12)


def test_vol_budget_basic_and_edges() -> None:
    # σ=0.20, T=60d → budget = sqrt(0.20 * 60/365) ≈ sqrt(0.03287) ≈ 0.1813.
    assert vol_budget(0.20, 60.0) == pytest.approx(math.sqrt(0.20 * 60.0 / 365.0))
    assert vol_budget(0.0, 60.0) == 0.0
    assert vol_budget(0.20, 0.0) == 0.0
    assert vol_budget(-0.10, 60.0) == 0.0
    assert vol_budget(float("nan"), 60.0) == 0.0
    assert vol_budget(0.20, float("nan")) == 0.0


def test_raw_signal_sign_and_zero_budget() -> None:
    # poly > options, positive budget → positive raw signal.
    s = raw_signal(p_poly=0.60, p_options=0.50, realized_vol=0.20, days_to_event=60.0)
    assert s > 0.0
    # poly < options → negative.
    s_neg = raw_signal(p_poly=0.40, p_options=0.50, realized_vol=0.20, days_to_event=60.0)
    assert s_neg < 0.0
    # Zero budget → zero signal.
    assert raw_signal(0.60, 0.50, 0.0, 60.0) == 0.0
    assert raw_signal(0.60, 0.50, 0.20, 0.0) == 0.0


# ---------------------------------------------------------------------------
# Scalar-mode signal / position / pnl
# ---------------------------------------------------------------------------


def test_signal_scalar_mode_returns_float() -> None:
    alpha = IVRealizedVolArb()
    state = IVRealizedVolState(
        implied_prob_poly=0.70,
        implied_prob_options=0.50,
        days_to_event=30.0,
        realized_vol=0.25,
    )
    sig = alpha.signal(state)
    assert isinstance(sig, float)
    assert sig > 0.0  # poly is richer → positive signal


def test_position_scalar_below_threshold_is_zero() -> None:
    alpha = IVRealizedVolArb(z_threshold=10.0)
    assert alpha.position(0.5) == 0.0
    assert alpha.position(-0.5) == 0.0


def test_position_scalar_obeys_kelly_cap_and_sign() -> None:
    # Use a wide cap so the Kelly formula is observable; then a second
    # check with a tight cap to confirm clipping kicks in.
    wide = IVRealizedVolArb(kelly_cap=0.90, z_threshold=1.0, fade_sign=1)
    # In-band Kelly = -z/3, no cap.
    assert wide.position(2.0) == pytest.approx(-2.0 / 3.0, abs=1e-12)
    assert wide.position(-1.5) == pytest.approx(0.5, abs=1e-12)

    tight = IVRealizedVolArb(kelly_cap=0.10, z_threshold=1.0, fade_sign=1)
    # Big positive signal → big negative position (short YES), capped at -0.10.
    assert tight.position(9.0) == pytest.approx(-0.10, abs=1e-12)
    # Big negative signal → positive position, capped at +0.10.
    assert tight.position(-9.0) == pytest.approx(0.10, abs=1e-12)
    # Signal of 2.0 would yield Kelly -0.667 — clipped to -0.10.
    assert tight.position(2.0) == pytest.approx(-0.10, abs=1e-12)


def test_position_fade_sign_inverts_direction() -> None:
    alpha = IVRealizedVolArb(kelly_cap=0.20, z_threshold=1.0, fade_sign=-1)
    assert alpha.position(3.0) > 0.0
    assert alpha.position(-3.0) < 0.0


def test_pnl_scalar_and_series_modes() -> None:
    alpha = IVRealizedVolArb()
    # Scalar.
    assert alpha.pnl(0.1, 0.05) == pytest.approx(0.005, abs=1e-12)
    # Series.
    idx = pd.date_range("2026-01-01", periods=5, freq="D")
    pos = pd.Series([0.0, -0.1, -0.1, 0.0, 0.1], index=idx)
    ret = pd.Series([0.02, -0.03, 0.01, 0.0, 0.04], index=idx)
    out = alpha.pnl(pos, ret)
    assert isinstance(out, pd.Series)
    assert out.name == "pnl"
    assert out.iloc[1] == pytest.approx(-0.1 * -0.03, abs=1e-12)


# ---------------------------------------------------------------------------
# DataFrame-mode signal
# ---------------------------------------------------------------------------


def test_signal_dataframe_mode_missing_columns_raises() -> None:
    alpha = IVRealizedVolArb()
    df = pd.DataFrame({"implied_prob_poly": [0.5, 0.6]})
    with pytest.raises(ValueError):
        alpha.signal(df)


def test_signal_dataframe_mode_rejects_non_frame() -> None:
    alpha = IVRealizedVolArb()
    with pytest.raises(TypeError):
        alpha.signal("not a frame")  # type: ignore[arg-type]


def test_signal_dataframe_returns_zero_in_warmup_window() -> None:
    """First (z_window - 1) rows should be 0 due to rolling z min_periods."""
    alpha = IVRealizedVolArb(z_window=30, z_threshold=1.0)
    frame = _build_planted_divergence_dataset(n_days=60)
    sig = alpha.signal(frame)
    # First 29 rows must be zero (NaN filled via .fillna(0.0)).
    assert (sig.iloc[: alpha.z_window - 1].abs() == 0.0).all()


def test_signal_dataframe_per_market_grouping() -> None:
    """When ``market_id`` is present, z-scores are computed per group."""
    alpha = IVRealizedVolArb(z_window=10, z_threshold=1.0)
    a = _build_planted_divergence_dataset(n_days=60, seed=1)
    b = _build_no_divergence_dataset(n_days=60, seed=2)
    a["market_id"] = "A"
    b["market_id"] = "B"
    # Reset indices so concat doesn't collide on the date index.
    a = a.reset_index(drop=True)
    b = b.reset_index(drop=True)
    b.index = b.index + 100  # disjoint integer index
    frame = pd.concat([a, b])
    sig = alpha.signal(frame)
    assert isinstance(sig, pd.Series)
    assert len(sig) == len(frame)
    assert sig.index.equals(frame.index)


# ---------------------------------------------------------------------------
# Synthetic-DGP recovery — the headline test of this module.
# ---------------------------------------------------------------------------


def test_planted_divergence_strategy_shorts_poly_on_average() -> None:
    """When ``p_poly > p_options`` consistently, positions must be net-negative."""
    alpha = IVRealizedVolArb(kelly_cap=0.20, z_threshold=1.0, z_window=30)
    frame = _build_planted_divergence_dataset(n_days=600, poly_bias=0.10)
    sig = alpha.signal(frame)
    pos = alpha.position(sig)

    # After warmup, the average position should be materially negative.
    warm = pos.iloc[alpha.z_window :]
    assert warm.mean() < -0.005, f"Expected net-short YES positions, got mean={warm.mean():.4f}"
    # And at least 20% of warm-period days should fire (|pos| > 0).
    active_share = float((warm.abs() > 0).mean())
    assert active_share > 0.20, f"Too few active days: {active_share:.2%}"


def test_planted_divergence_sharpe_at_least_0p4() -> None:
    """Sharpe of strategy PnL on planted DGP must be ≥ 0.4 (annualised)."""
    alpha = IVRealizedVolArb(kelly_cap=0.20, z_threshold=1.0, z_window=30)
    frame = _build_planted_divergence_dataset(n_days=800, poly_bias=0.10, noise=0.015, seed=0xABCD)
    daily_pnl = alpha.compute_daily_pnl(frame)
    warm = daily_pnl.iloc[alpha.z_window :]
    mu = float(warm.mean())
    sigma = float(warm.std(ddof=1))
    assert sigma > 0.0, "Degenerate PnL — should have non-zero variance"
    sharpe = (mu / sigma) * math.sqrt(252.0)
    assert sharpe >= 0.4, f"Planted-DGP Sharpe too low: {sharpe:.3f}"


def test_no_divergence_signal_centred_near_zero() -> None:
    """With ``p_poly ≈ p_options`` the raw signal mean should be small."""
    alpha = IVRealizedVolArb(kelly_cap=0.20, z_threshold=1.8, z_window=30)
    frame = _build_no_divergence_dataset(n_days=600, noise=0.005)
    sig = alpha.signal(frame)
    warm = sig.iloc[alpha.z_window :]
    # z-scores have mean ~0 by construction; check it's well-bounded.
    assert abs(float(warm.mean())) < 0.30
    # And almost no days should hit the threshold (z>1.8 is rare under N(0,1)).
    active = float((warm.abs() >= alpha.z_threshold).mean())
    assert active < 0.20, f"Too many active days in no-divergence DGP: {active:.2%}"


def test_no_divergence_positions_have_small_abs_mean() -> None:
    """Mean |position| in a no-divergence world should be small."""
    alpha = IVRealizedVolArb(kelly_cap=0.20, z_threshold=1.8, z_window=30)
    frame = _build_no_divergence_dataset(n_days=600, noise=0.005, seed=0xFEED)
    sig = alpha.signal(frame)
    pos = alpha.position(sig)
    warm = pos.iloc[alpha.z_window :]
    assert float(warm.abs().mean()) < 0.03


def test_reproducibility_same_seed_same_pnl() -> None:
    """Identical seeds & params → identical PnL series (no hidden randomness)."""
    alpha = IVRealizedVolArb(kelly_cap=0.20, z_threshold=1.0, z_window=30)
    frame1 = _build_planted_divergence_dataset(n_days=200, seed=7)
    frame2 = _build_planted_divergence_dataset(n_days=200, seed=7)
    p1 = alpha.compute_daily_pnl(frame1)
    p2 = alpha.compute_daily_pnl(frame2)
    pd.testing.assert_series_equal(p1, p2, check_names=False)


def test_reproducibility_independent_of_strategy_instance() -> None:
    """Two freshly-constructed strategies with identical params behave identically."""
    a1 = IVRealizedVolArb(kelly_cap=0.20, z_threshold=1.0, z_window=30)
    a2 = IVRealizedVolArb(kelly_cap=0.20, z_threshold=1.0, z_window=30)
    frame = _build_planted_divergence_dataset(n_days=200, seed=11)
    pd.testing.assert_series_equal(
        a1.compute_daily_pnl(frame),
        a2.compute_daily_pnl(frame),
        check_names=False,
    )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_edge_single_observation_returns_zero_signal() -> None:
    """1-row frame: rolling window cannot fill → signal = 0."""
    alpha = IVRealizedVolArb(z_window=30, z_threshold=1.0)
    frame = pd.DataFrame(
        {
            "implied_prob_poly": [0.70],
            "implied_prob_options": [0.50],
            "days_to_event": [30.0],
            "realized_vol": [0.20],
            "outcome": [1.0],
        },
        index=pd.date_range("2026-01-01", periods=1, freq="D"),
    )
    sig = alpha.signal(frame)
    assert len(sig) == 1
    assert float(sig.iloc[0]) == 0.0
    pos = alpha.position(sig)
    assert float(pos.iloc[0]) == 0.0


def test_edge_nan_inputs_in_scalar_path_are_safe() -> None:
    alpha = IVRealizedVolArb()
    # NaN realized_vol -> raw_signal returns 0.
    state = IVRealizedVolState(
        implied_prob_poly=0.7,
        implied_prob_options=0.5,
        days_to_event=30.0,
        realized_vol=float("nan"),
    )
    assert alpha.signal(state) == 0.0
    # NaN days_to_event -> raw_signal returns 0.
    state2 = IVRealizedVolState(
        implied_prob_poly=0.7,
        implied_prob_options=0.5,
        days_to_event=float("nan"),
        realized_vol=0.2,
    )
    assert alpha.signal(state2) == 0.0


def test_edge_t_zero_returns_zero_signal_and_zero_position() -> None:
    """Time-to-event == 0 must yield a zero signal (and hence zero position)."""
    alpha = IVRealizedVolArb(z_threshold=1.0)
    state = IVRealizedVolState(
        implied_prob_poly=0.95,
        implied_prob_options=0.05,
        days_to_event=0.0,
        realized_vol=0.20,
    )
    assert alpha.signal(state) == 0.0
    assert alpha.position(alpha.signal(state)) == 0.0


def test_edge_compute_daily_pnl_requires_outcome_column() -> None:
    alpha = IVRealizedVolArb()
    frame = _build_planted_divergence_dataset(n_days=40).drop(columns=["outcome"])
    with pytest.raises(ValueError):
        alpha.compute_daily_pnl(frame)


def test_position_series_zero_below_threshold_and_capped() -> None:
    alpha = IVRealizedVolArb(kelly_cap=0.10, z_threshold=2.0)
    sig = pd.Series([0.0, 1.0, -1.5, 3.0, -3.0, 9.0, -9.0])
    pos = alpha.position(sig)
    # |z|<2 → 0; |z|>=2 → Kelly = -sign(z) * |z|/3, clipped to ±0.10.
    assert float(pos.iloc[0]) == 0.0
    assert float(pos.iloc[1]) == 0.0  # |1.0| < 2.0
    assert float(pos.iloc[2]) == 0.0  # |-1.5| < 2.0
    # All of 3.0/-3.0/9.0/-9.0 hit the ±0.10 cap because raw Kelly is ≥1.0.
    assert float(pos.iloc[3]) == pytest.approx(-0.10, abs=1e-12)
    assert float(pos.iloc[4]) == pytest.approx(0.10, abs=1e-12)
    assert float(pos.iloc[5]) == pytest.approx(-0.10, abs=1e-12)
    assert float(pos.iloc[6]) == pytest.approx(0.10, abs=1e-12)

    # Wider cap exposes pre-cap Kelly values.
    wide = IVRealizedVolArb(kelly_cap=0.90, z_threshold=2.0)
    wpos = wide.position(sig)
    assert float(wpos.iloc[3]) == pytest.approx(-1.0, abs=1e-12) or float(
        wpos.iloc[3]
    ) == pytest.approx(-0.90, abs=1e-12)
    # Specifically: -3/3 = -1.0, clipped to -0.90.
    assert float(wpos.iloc[3]) == pytest.approx(-0.90, abs=1e-12)
    assert float(wpos.iloc[4]) == pytest.approx(0.90, abs=1e-12)
