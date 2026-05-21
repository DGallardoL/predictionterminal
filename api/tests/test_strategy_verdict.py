"""Tests for the verdict layer that converts raw stats to actions."""

from __future__ import annotations

from pfm.strategy_verdict import (
    alpha_card_verdict,
    bollinger_verdict,
    cointegration_verdict,
    pair_trade_verdict,
    regression_verdict,
)

# ---------------------------------------------------------------------------
# Cointegration verdicts
# ---------------------------------------------------------------------------


def test_cointegration_users_specific_case_skips() -> None:
    """The exact case the user reported should land on SKIP."""
    v = cointegration_verdict(
        adf_p=0.131,
        half_life_days=0.9,
        rho_ar1=0.454,
        n_obs=82,
        beta_hedge=2.189,
    )
    assert v["action"] == "SKIP"
    # Reasoning must mention BOTH the suspicious half-life and the ADF threshold.
    joined = " ".join(v["reasoning"]).lower()
    assert "suspicious" in joined
    assert "0.9" in joined
    assert v["trade_spec"] is None
    assert v["monitoring_rules"]


def test_cointegration_clean_signal_opens_pair() -> None:
    v = cointegration_verdict(
        adf_p=0.01,
        half_life_days=5.0,
        rho_ar1=0.65,
        n_obs=200,
        beta_hedge=1.5,
    )
    assert v["action"] == "OPEN_PAIR"
    assert v["trade_spec"] is not None
    assert v["trade_spec"]["b_size"] == 1.5
    assert v["trade_spec"]["entry_z"] == 2.0
    assert v["confidence"] == "high"


def test_cointegration_borderline_p_value_watches() -> None:
    v = cointegration_verdict(
        adf_p=0.10,
        half_life_days=4.0,
        rho_ar1=0.5,
        n_obs=150,
        beta_hedge=1.0,
    )
    assert v["action"] == "WATCH"
    assert v["trade_spec"] is None


def test_cointegration_in_position_flatten() -> None:
    v = cointegration_verdict(
        adf_p=0.01,
        half_life_days=5.0,
        rho_ar1=0.65,
        n_obs=200,
        beta_hedge=1.5,
        current_z=0.2,
        in_position=True,
    )
    assert v["action"] == "FLATTEN"
    assert v["confidence"] == "high"


def test_cointegration_clean_pair_but_low_z_waits() -> None:
    v = cointegration_verdict(
        adf_p=0.01,
        half_life_days=5.0,
        rho_ar1=0.65,
        n_obs=200,
        beta_hedge=1.5,
        current_z=0.7,
    )
    assert v["action"] == "WAIT"


def test_cointegration_small_n_skips() -> None:
    v = cointegration_verdict(
        adf_p=0.04,
        half_life_days=4.0,
        rho_ar1=0.5,
        n_obs=20,
        beta_hedge=1.0,
    )
    assert v["action"] == "SKIP"


# ---------------------------------------------------------------------------
# Pair trade verdicts
# ---------------------------------------------------------------------------


def test_pair_trade_open_when_z_extreme() -> None:
    v = pair_trade_verdict(current_z=-2.5, in_position=False, beta_hedge=1.5)
    assert v["action"] == "OPEN_PAIR"
    assert v["trade_spec"]["direction"] == "long_a_short_b"


def test_pair_trade_flatten_at_exit_when_in_position() -> None:
    v = pair_trade_verdict(current_z=0.3, in_position=True)
    assert v["action"] == "FLATTEN"


def test_pair_trade_stop_out_blowout() -> None:
    v = pair_trade_verdict(current_z=4.5, in_position=True, stop_z=4.0)
    assert v["action"] == "FLATTEN"


def test_pair_trade_skip_when_not_cointegrated() -> None:
    v = pair_trade_verdict(current_z=-3.0, cointegration_passed=False)
    assert v["action"] == "SKIP"


# ---------------------------------------------------------------------------
# Bollinger
# ---------------------------------------------------------------------------


def test_bollinger_skips_when_trending() -> None:
    v = bollinger_verdict(current_z=-2.5, hurst=0.65)
    assert v["action"] == "SKIP"


# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------


def test_regression_deploy_on_strong_fit() -> None:
    v = regression_verdict(r2=0.65, n_obs=300, n_factors=5, vif_max=2.0, f_pvalue=0.001)
    assert v["action"] == "DEPLOY"


def test_regression_reject_on_low_r2() -> None:
    v = regression_verdict(r2=0.05, n_obs=300, n_factors=5, vif_max=2.0)
    assert v["action"] == "REJECT"


def test_regression_reject_on_high_vif() -> None:
    v = regression_verdict(r2=0.7, n_obs=300, n_factors=5, vif_max=15.0)
    assert v["action"] == "REJECT"


def test_regression_watch_on_marginal_r2() -> None:
    v = regression_verdict(r2=0.30, n_obs=300, n_factors=5, vif_max=2.0)
    assert v["action"] == "WATCH"


def test_regression_reject_on_too_few_obs_per_factor() -> None:
    v = regression_verdict(r2=0.6, n_obs=30, n_factors=5, vif_max=2.0)
    assert v["action"] == "REJECT"


# ---------------------------------------------------------------------------
# Alpha card
# ---------------------------------------------------------------------------


def test_alpha_card_gold_deploys_live() -> None:
    v = alpha_card_verdict({"name": "btc_arb", "tier": "A_GOLD", "allocation_pct": 0.5})
    assert v["action"] == "DEPLOY_LIVE_SMALL_SIZE"
    assert v["confidence"] == "high"


def test_alpha_card_validated_paper_trades() -> None:
    v = alpha_card_verdict({"name": "pair_x", "tier": "B_VALIDATED"})
    assert v["action"] == "PAPER_TRADE_FIRST"


def test_alpha_card_tentative_does_not_deploy() -> None:
    v = alpha_card_verdict({"name": "z", "tier": "C_TENTATIVE"})
    assert v["action"] == "WATCH_DO_NOT_DEPLOY"


def test_alpha_card_rejected_archives() -> None:
    v = alpha_card_verdict({"name": "old_strat", "tier": "D_REJECTED"})
    assert v["action"] == "ARCHIVE"
