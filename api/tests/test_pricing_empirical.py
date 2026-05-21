"""Tests for T82 empirical calibration harness (``pfm.pricing.empirical_calibration``).

The metric functions (Brier, log-loss, calibration RMSE, lead-time, PnL)
do not depend on T81 — we test those directly. The pricer-coupled paths
(``score_model``, ``_resolve_predictions``) are exercised via a tiny
in-test stub pricer so the suite passes even if T81's ``binary_models``
module has not landed yet.

Run (after main.py import has been fixed by T80):

    cd api && PYTHONPATH=src .venv/bin/python -m pytest \
        tests/test_pricing_empirical.py -q

If main.py import is still broken, run with ``--noconftest``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import httpx
import numpy as np
import pytest

from pfm.pricing import empirical_calibration as ec
from pfm.pricing.empirical_calibration import (
    MarketEpisode,
    brier_score,
    calibration_rmse,
    compute_pnl,
    early_warning_lead_time,
    log_loss,
    pull_resolved_markets,
    score_model,
    write_report,
)

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "binary_pricing_fixtures.json"


# ---------------------------------------------------------------------------
# Stub pricer (decouples T82 tests from T81's implementation)
# ---------------------------------------------------------------------------


class _StubPricer:
    """A trivial pricer that returns a configured constant or a per-episode lookup."""

    def __init__(
        self,
        name: str,
        fixed_price: float | None = None,
        episode_overrides: dict[str, list[float]] | None = None,
    ) -> None:
        self.name = name
        self.fixed_price = fixed_price
        self.episode_overrides = episode_overrides or {}

    def calibrate(self, history):
        return {"fitted": True}

    def theoretical_price(self, state, params):
        if self.fixed_price is not None:
            return self.fixed_price
        return float(state.market_price)


def _attach_predictions(episode: MarketEpisode, name: str, values: list[float]) -> None:
    """Helper to seed cached predictions on an episode."""
    meta = episode.metadata or {}
    preds = meta.setdefault("predictions", {})
    preds[name] = list(values)
    episode.metadata = meta


# ---------------------------------------------------------------------------
# Fixture-loading tests
# ---------------------------------------------------------------------------


def test_fixture_file_exists_and_has_five_markets():
    data = json.loads(FIXTURE_PATH.read_text())
    assert len(data["markets"]) == 5
    yes_count = sum(1 for m in data["markets"] if m["resolved"])
    no_count = sum(1 for m in data["markets"] if not m["resolved"])
    assert yes_count == 2
    assert no_count == 3


def test_pull_resolved_markets_from_fixture():
    eps = pull_resolved_markets(limit=50, fixture_path=FIXTURE_PATH)
    assert len(eps) == 5
    assert all(isinstance(e, MarketEpisode) for e in eps)
    assert sum(1 for e in eps if e.resolved) == 2
    # First episode has the expected trajectory length
    assert len(eps[0].trajectory) == 9
    assert eps[0].trajectory[0][1] == pytest.approx(0.55)


def test_pull_resolved_markets_limit_applies():
    eps = pull_resolved_markets(limit=2, fixture_path=FIXTURE_PATH)
    assert len(eps) == 2


def test_episode_market_prices_sorted_descending_time():
    eps = pull_resolved_markets(fixture_path=FIXTURE_PATH)
    ep = eps[0]
    prices = ep.market_prices()
    times = ep.times()
    # Times are monotone decreasing from largest days-to-resolution to 0
    assert np.all(np.diff(times) <= 0)
    assert prices[0] == pytest.approx(0.55)
    assert prices[-1] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Network-mocked Gamma fetch (uses respx)
# ---------------------------------------------------------------------------


def test_pull_resolved_markets_mocked_gamma_returns_expected_count():
    pytest.importorskip("respx")
    import respx

    gamma_payload = json.loads(FIXTURE_PATH.read_text())["gamma_response"]
    with respx.mock(assert_all_called=False) as mock:
        mock.get("https://gamma-api.polymarket.com/markets").mock(
            return_value=httpx.Response(200, json=gamma_payload)
        )
        eps = pull_resolved_markets(limit=10, respx_mock=mock)
    # Two markets in the gamma_response payload, both closed, both have priceHistory
    assert len(eps) == 2
    # The first is YES (outcomePrices=["1.0","0.0"])
    assert eps[0].resolved is True
    assert eps[1].resolved is False


def test_pull_resolved_markets_handles_http_error():
    pytest.importorskip("respx")
    import respx

    with respx.mock(assert_all_called=False) as mock:
        mock.get("https://gamma-api.polymarket.com/markets").mock(
            return_value=httpx.Response(500, text="boom")
        )
        eps = pull_resolved_markets(limit=10, respx_mock=mock)
    assert eps == []


# ---------------------------------------------------------------------------
# Metric tests — Brier score
# ---------------------------------------------------------------------------


def test_brier_score_known_input():
    # All predictions equal outcome -> Brier = 0
    assert brier_score([1.0, 0.0, 1.0], [1, 0, 1]) == pytest.approx(0.0)
    # All predictions = 0.5 -> Brier = 0.25 regardless of outcome
    assert brier_score([0.5, 0.5, 0.5, 0.5], [1, 0, 1, 0]) == pytest.approx(0.25)
    # Hand-checked: preds [0.8, 0.3], outcomes [1, 0]
    # diffs^2 = (0.2)^2 + (0.3)^2 = 0.04 + 0.09 = 0.13; mean = 0.065
    assert brier_score([0.8, 0.3], [1, 0]) == pytest.approx(0.065)


def test_brier_score_empty_returns_nan():
    assert math.isnan(brier_score([], []))


def test_brier_score_shape_mismatch_raises():
    with pytest.raises(ValueError, match="shape mismatch"):
        brier_score([0.5], [1, 0])


# ---------------------------------------------------------------------------
# Metric tests — Log-loss
# ---------------------------------------------------------------------------


def test_log_loss_all_correct_is_near_zero():
    # Predictions clipped to 1-eps so log-loss is small but positive
    loss = log_loss([1.0, 0.0, 1.0, 0.0], [1, 0, 1, 0])
    assert loss < 1e-5
    assert loss >= 0.0


def test_log_loss_all_wrong_is_large_finite_with_clipping():
    # Predictions clipped to eps so log-loss is bounded ~ -log(eps)
    loss = log_loss([0.0, 1.0, 0.0, 1.0], [1, 0, 1, 0])
    assert loss > 10.0  # large but finite, not inf
    assert math.isfinite(loss)


def test_log_loss_uniform_half():
    # All preds = 0.5 -> loss = ln(2) for every point
    loss = log_loss([0.5, 0.5, 0.5], [1, 0, 1])
    assert loss == pytest.approx(math.log(2.0))


def test_log_loss_shape_mismatch_raises():
    with pytest.raises(ValueError, match="shape mismatch"):
        log_loss([0.5, 0.5], [1])


# ---------------------------------------------------------------------------
# Metric tests — Calibration RMSE
# ---------------------------------------------------------------------------


def test_calibration_rmse_perfectly_calibrated_is_zero():
    # Build a perfectly calibrated dataset: 100 points at p=0.7 where 70 are YES.
    n = 100
    preds = [0.7] * n
    outcomes = [1] * 70 + [0] * 30
    # All 100 land in the same bucket [0.7, 0.8): mean(pred)=0.7, mean(y)=0.7
    val = calibration_rmse(preds, outcomes, n_bins=10)
    assert val == pytest.approx(0.0, abs=1e-9)


def test_calibration_rmse_maximally_miscalibrated():
    # Predictions all 0.9 but outcomes all 0 — single populated bucket
    # mean(pred)=0.9, mean(y)=0, RMSE = sqrt((0.9-0)^2) = 0.9
    val = calibration_rmse([0.9] * 10, [0] * 10, n_bins=10)
    assert val == pytest.approx(0.9)


def test_calibration_rmse_multi_bucket():
    # Two buckets: low bucket 0.1 with realized 0.1 (calibrated), high bucket
    # 0.9 with realized 0.5 (miscalibrated by 0.4)
    preds = [0.1] * 10 + [0.9] * 10
    outcomes = [1] + [0] * 9 + [1] * 5 + [0] * 5
    val = calibration_rmse(preds, outcomes, n_bins=10)
    # RMSE of (0.0, 0.4) = sqrt((0 + 0.16)/2) = sqrt(0.08) ≈ 0.2828
    assert val == pytest.approx(math.sqrt(0.08), abs=1e-6)


def test_calibration_rmse_empty():
    assert math.isnan(calibration_rmse([], []))


# ---------------------------------------------------------------------------
# Metric tests — Early-warning lead time
# ---------------------------------------------------------------------------


def test_early_warning_lead_time_yes_market_model_ahead():
    # YES-resolved market; model price is 0.20 above market at all times
    ep = MarketEpisode(
        market_id="ew-1",
        title="EW yes",
        resolved=True,
        trajectory=[(30.0, 0.4), (20.0, 0.5), (10.0, 0.6), (0.0, 1.0)],
    )
    # Predictions correspond to trajectory sorted DESCENDING by t:
    # [(30,0.4), (20,0.5), (10,0.6), (0,1.0)]
    preds = [0.6, 0.7, 0.8, 1.0]  # all >= market + 0.10 except the last (1.0==1.0)
    lead = early_warning_lead_time(ep, preds, threshold=0.10)
    # First three qualify; mean of times [30,20,10] = 20.0
    # (note: lead requires STRICT > threshold so 0.2>0.1 qualifies)
    assert lead == pytest.approx(20.0)


def test_early_warning_lead_time_no_market_model_below():
    ep = MarketEpisode(
        market_id="ew-2",
        title="EW no",
        resolved=False,
        trajectory=[(40.0, 0.6), (10.0, 0.3), (0.0, 0.0)],
    )
    # NO-resolved; model "below" market by >0.10 means correct early signal
    preds = [0.4, 0.1, 0.0]  # deltas: 0.2 below, 0.2 below, 0 — first two qualify
    lead = early_warning_lead_time(ep, preds, threshold=0.10)
    assert lead == pytest.approx((40.0 + 10.0) / 2)


def test_early_warning_lead_time_no_qualifying_returns_zero():
    ep = MarketEpisode(
        market_id="ew-3",
        title="EW none",
        resolved=True,
        trajectory=[(10.0, 0.5), (0.0, 1.0)],
    )
    # Model tracks market exactly
    preds = [0.5, 1.0]
    assert early_warning_lead_time(ep, preds) == 0.0


# ---------------------------------------------------------------------------
# Metric tests — Economic PnL
# ---------------------------------------------------------------------------


def test_compute_pnl_trivially_predictable_market_positive():
    # Market priced at 0.5 throughout, model says 0.9, resolves YES
    ep = MarketEpisode(
        market_id="pnl-1",
        title="Easy money",
        resolved=True,
        trajectory=[(10.0, 0.5), (5.0, 0.5), (0.0, 0.5)],
    )
    _attach_predictions(ep, "stub", [0.9, 0.9, 0.9])
    pricer = _StubPricer("stub", fixed_price=0.9)
    pnl = compute_pnl(pricer, [ep], fee_bps=10, kelly_cap=0.25)
    assert pnl > 0.0


def test_compute_pnl_wrong_direction_negative():
    # Model says 0.9 but market resolves NO at price 0.5 → buying YES loses
    ep = MarketEpisode(
        market_id="pnl-2",
        title="Bad bet",
        resolved=False,
        trajectory=[(0.0, 0.5)],
    )
    _attach_predictions(ep, "stub", [0.9])
    pricer = _StubPricer("stub", fixed_price=0.9)
    pnl = compute_pnl(pricer, [ep], fee_bps=10, kelly_cap=0.25)
    assert pnl < 0.0


def test_compute_pnl_no_edge_no_trade():
    # Model agrees with market; should not trade and incur no fee
    ep = MarketEpisode(
        market_id="pnl-3",
        title="No edge",
        resolved=True,
        trajectory=[(0.0, 0.5)],
    )
    _attach_predictions(ep, "stub", [0.5])
    pricer = _StubPricer("stub", fixed_price=0.5)
    pnl = compute_pnl(pricer, [ep], fee_bps=100, kelly_cap=0.25)
    assert pnl == pytest.approx(0.0)


def test_compute_pnl_kelly_cap_respected():
    # Huge edge would Kelly-size to 1.0, but cap should clamp to 0.25
    ep = MarketEpisode(
        market_id="pnl-4",
        title="Huge edge",
        resolved=True,
        trajectory=[(0.0, 0.5)],
    )
    _attach_predictions(ep, "stub", [0.99])
    pricer = _StubPricer("stub", fixed_price=0.99)
    pnl_cap25 = compute_pnl(pricer, [ep], fee_bps=0, kelly_cap=0.25)
    pnl_cap10 = compute_pnl(pricer, [ep], fee_bps=0, kelly_cap=0.10)
    # PnL scales linearly with the bet fraction up to the cap; cap0.10
    # must be less than cap0.25 (since the bet wins)
    assert 0.0 < pnl_cap10 < pnl_cap25


def test_compute_pnl_short_no_side():
    # Market priced 0.6 but model thinks 0.2 → NO is mispriced, resolves NO
    ep = MarketEpisode(
        market_id="pnl-5",
        title="Short NO",
        resolved=False,
        trajectory=[(0.0, 0.6)],
    )
    _attach_predictions(ep, "stub", [0.2])
    pricer = _StubPricer("stub", fixed_price=0.2)
    pnl = compute_pnl(pricer, [ep], fee_bps=10, kelly_cap=0.25)
    assert pnl > 0.0


# ---------------------------------------------------------------------------
# Score-model integration
# ---------------------------------------------------------------------------


def test_score_model_full_panel_against_fixture():
    eps = pull_resolved_markets(fixture_path=FIXTURE_PATH)
    # Use a pricer that mirrors the market (cached predictions = market prices)
    for ep in eps:
        _attach_predictions(ep, "market_mirror", ep.market_prices().tolist())
    pricer = _StubPricer("market_mirror")
    scores = score_model(pricer, eps, fee_bps=100, kelly_cap=0.25)
    assert {
        "brier",
        "log_loss",
        "calibration_rmse",
        "early_warning_days",
        "pnl",
        "n_episodes",
        "n_points",
    } <= set(scores.keys())
    assert scores["n_episodes"] == 5.0
    # Mirror predictions trade nothing -> pnl == 0
    assert scores["pnl"] == pytest.approx(0.0)
    # Brier should be well-defined and finite
    assert 0.0 <= scores["brier"] <= 1.0
    assert scores["early_warning_days"] == 0.0  # mirror never deviates


def test_score_model_better_oracle_beats_mirror():
    eps = pull_resolved_markets(fixture_path=FIXTURE_PATH)
    oracle_scores = []
    mirror_scores = []
    for ep in eps:
        n = len(ep.trajectory)
        oracle_pred = [1.0 if ep.resolved else 0.0] * n
        _attach_predictions(ep, "oracle", oracle_pred)
        _attach_predictions(ep, "mirror", ep.market_prices().tolist())
    oracle = _StubPricer("oracle")
    mirror = _StubPricer("mirror")
    oracle_scores = score_model(oracle, eps)
    mirror_scores = score_model(mirror, eps)
    # The oracle should have a lower Brier than the market mirror because
    # it sees the eventual outcome directly.
    assert oracle_scores["brier"] < mirror_scores["brier"]


def test_score_model_empty_episode_list():
    pricer = _StubPricer("any", fixed_price=0.5)
    scores = score_model(pricer, [])
    assert scores["n_episodes"] == 0
    assert math.isnan(scores["brier"])


# ---------------------------------------------------------------------------
# T81 dependency error
# ---------------------------------------------------------------------------


def test_resolve_predictions_without_t81_raises_clear_error(monkeypatch):
    """If T81 is missing and there's no cached prediction, raise a clear ImportError.

    Even when T81 *is* present, we can simulate its absence by monkeypatching
    ``_require_t81`` to raise. Pricer-cached predictions are NOT exercised so
    the fallback path triggers.
    """
    ep = MarketEpisode(
        market_id="t81-test",
        title="needs t81",
        resolved=True,
        trajectory=[(5.0, 0.5)],
    )

    def _boom():
        raise ImportError(
            "T82 empirical_calibration requires the T81 module "
            "`pfm.pricing.binary_models` to be present."
        )

    monkeypatch.setattr(ec, "_require_t81", _boom)
    pricer = _StubPricer("uncached")  # no cached prediction
    # Bypass the cache by ensuring the predictions cache is missing
    ep.metadata = {}
    with pytest.raises(ImportError, match="T81 module"):
        ec._resolve_predictions(pricer, ep)


def test_require_t81_error_message_includes_actionable_pointer():
    """The error message should reference TASK-BOARD T81 explicitly."""
    # Sanity check that the canonical error text contains a useful pointer
    # (we don't import T81 here — the test only validates the wording).
    try:
        ec._require_t81()
    except ImportError as e:
        msg = str(e)
        assert "binary_models" in msg
        assert "T81" in msg
    except Exception:
        # T81 is available — skip the wording check; it's not the contract under test
        pytest.skip("T81 is present; cannot exercise the missing-import path")


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------


def test_write_report_creates_file(tmp_path):
    scores = {
        "logit": {"brier": 0.18, "log_loss": 0.55, "pnl": 0.03},
        "bsm": {"brier": 0.22, "log_loss": 0.62, "pnl": -0.01},
    }
    out = tmp_path / "report.json"
    written = write_report(scores, out, extra={"date": "2026-05-16"})
    assert written.exists()
    data = json.loads(written.read_text())
    assert data["schema"] == "binary-pricing-report/v1"
    assert data["models"]["logit"]["brier"] == pytest.approx(0.18)
    assert data["meta"]["date"] == "2026-05-16"


def test_write_report_creates_parent_dirs(tmp_path):
    out = tmp_path / "nested" / "deeply" / "report.json"
    write_report({"m": {"brier": 0.1}}, out)
    assert out.exists()
