"""Tests for the disagrees-jump paper-PnL backtester.

Two layers exercised, mirroring the test design of ``test_jumps.py``:

1. :func:`simulate_disagrees_pnl` is a pure function — we hand-build
   jump dicts + price series where we KNOW the expected sign of the PnL
   (reverting vs continuing prices) and assert the aggregate stats.
2. The endpoint joins three subsystems (Polymarket meta, hourly prices,
   multi-source news). We mock the price + GDELT fetches so the test
   runs offline and deterministically.
"""

from __future__ import annotations

import math
from unittest.mock import patch

import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pfm.terminal.jumps_backtest import (
    DEFAULT_HOLD_HOURS,
    TRADING_HOURS_PER_YEAR,
    router,
    simulate_disagrees_pnl,
)
from pfm.terminal_gdelt_news import GDELTArticle

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _hourly_series(values: list[float], start: str = "2026-05-01T00:00:00Z") -> pd.Series:
    """UTC-indexed hourly probability series helper."""
    idx = pd.date_range(start=start, periods=len(values), freq="h", tz="UTC")
    return pd.Series(values, index=idx, name="price", dtype=float)


def _jump(
    *,
    ts: str,
    price_after: float,
    alignment: str,
    sentiment: float,
    direction: str = "up",
    price_before: float = 0.40,
) -> dict:
    return {
        "ts_iso": ts,
        "price_before": price_before,
        "price_after": price_after,
        "direction": direction,
        "news_sentiment_score": sentiment,
        "sentiment_alignment": alignment,
    }


# ---------------------------------------------------------------------------
# Pure-function tests
# ---------------------------------------------------------------------------


def test_simulate_returns_zeros_on_empty_inputs() -> None:
    result = simulate_disagrees_pnl([], _hourly_series([0.5] * 10))
    assert result["n_disagrees"] == 0
    assert result["n_agrees"] == 0
    assert result["disagrees_pnl"]["mean_return"] == 0.0
    assert result["disagrees_pnl"]["sharpe_naive"] == 0.0
    assert result["equity_curve"] == []


def test_simulate_returns_zeros_when_prices_empty() -> None:
    j = _jump(
        ts="2026-05-01T05:00:00Z",
        price_after=0.60,
        alignment="disagrees",
        sentiment=0.6,
        direction="down",
    )
    result = simulate_disagrees_pnl([j], pd.Series(dtype=float))
    assert result["n_disagrees"] == 0
    assert result["equity_curve"] == []


def test_disagrees_jumps_with_reverting_prices_show_positive_pnl() -> None:
    """Three disagrees jumps where the price reverts toward the news
    direction over the next 6h. Bullish news + price fell → we long YES,
    price recovers → positive PnL."""
    # Build a series where each jump is at a specific hour and the
    # price at hour+6 is the reverted value.
    # Spike #1: at hour 5 price drops to 0.40, by hour 11 it's back to 0.50.
    # Spike #2: at hour 20 price drops to 0.45, by hour 26 it's back to 0.55.
    # Spike #3: at hour 35 price drops to 0.50, by hour 41 it's back to 0.58.
    vals = [0.50] * 50
    vals[5] = 0.40
    vals[11] = 0.50
    vals[20] = 0.45
    vals[26] = 0.55
    vals[35] = 0.50
    vals[41] = 0.58
    prices = _hourly_series(vals)

    jumps = [
        _jump(
            ts="2026-05-01T05:00:00Z",
            price_after=0.40,
            alignment="disagrees",
            sentiment=0.5,
            direction="down",
        ),
        _jump(
            ts="2026-05-01T20:00:00Z",
            price_after=0.45,
            alignment="disagrees",
            sentiment=0.6,
            direction="down",
        ),
        _jump(
            ts="2026-05-02T11:00:00Z",  # hour 35
            price_after=0.50,
            alignment="disagrees",
            sentiment=0.4,
            direction="down",
        ),
    ]
    result = simulate_disagrees_pnl(jumps, prices, hold_hours=6)
    assert result["n_disagrees"] == 3
    assert result["disagrees_pnl"]["mean_return"] > 0.0
    assert result["disagrees_pnl"]["total_return"] > 0.0
    assert result["disagrees_pnl"]["hit_rate"] == 1.0
    assert result["disagrees_pnl"]["avg_win"] > 0.0
    assert result["disagrees_pnl"]["avg_loss"] == 0.0
    # Equity curve must be monotonically increasing with three points.
    eq = result["equity_curve"]
    assert len(eq) == 3
    assert eq[0]["cum_return"] < eq[1]["cum_return"] < eq[2]["cum_return"]
    # Drawdown is zero (no losing trade ever pulled equity down).
    assert result["disagrees_pnl"]["max_drawdown"] == 0.0


def test_disagrees_jumps_with_continuing_prices_show_negative_pnl() -> None:
    """Three disagrees jumps where the price CONTINUES against the news
    (no reversion) — paper PnL is negative."""
    vals = [0.50] * 50
    # Bullish-news jumps where price dropped and *kept dropping*.
    vals[5] = 0.40
    vals[11] = 0.30  # continued down
    vals[20] = 0.45
    vals[26] = 0.35  # continued down
    vals[35] = 0.50
    vals[41] = 0.40  # continued down
    prices = _hourly_series(vals)

    jumps = [
        _jump(
            ts="2026-05-01T05:00:00Z",
            price_after=0.40,
            alignment="disagrees",
            sentiment=0.5,
            direction="down",
        ),
        _jump(
            ts="2026-05-01T20:00:00Z",
            price_after=0.45,
            alignment="disagrees",
            sentiment=0.6,
            direction="down",
        ),
        _jump(
            ts="2026-05-02T11:00:00Z",
            price_after=0.50,
            alignment="disagrees",
            sentiment=0.4,
            direction="down",
        ),
    ]
    result = simulate_disagrees_pnl(jumps, prices, hold_hours=6)
    assert result["n_disagrees"] == 3
    assert result["disagrees_pnl"]["mean_return"] < 0.0
    assert result["disagrees_pnl"]["total_return"] < 0.0
    assert result["disagrees_pnl"]["hit_rate"] == 0.0
    assert result["disagrees_pnl"]["avg_loss"] < 0.0
    assert result["disagrees_pnl"]["avg_win"] == 0.0
    assert result["disagrees_pnl"]["max_drawdown"] < 0.0


def test_agrees_jumps_do_not_pollute_disagrees_bucket() -> None:
    """Mix agrees + disagrees jumps; assert each lands in its bucket and
    the agrees PnL is separate. Trivially the agrees bucket should hold
    the agrees-tagged trades only."""
    vals = [0.50] * 30
    vals[5] = 0.60  # disagrees: bearish news but price went up; we short YES
    vals[11] = 0.55  # short YES: exit lower than entry → +PnL for shorts
    vals[15] = 0.40  # agrees: bearish news + price went down; we short YES
    vals[21] = 0.30
    prices = _hourly_series(vals)

    jumps = [
        # Disagrees, bearish news (sentiment<0), price went up
        _jump(
            ts="2026-05-01T05:00:00Z",
            price_after=0.60,
            alignment="disagrees",
            sentiment=-0.4,
            direction="up",
        ),
        # Agrees, bearish news + price went down (control)
        _jump(
            ts="2026-05-01T15:00:00Z",
            price_after=0.40,
            alignment="agrees",
            sentiment=-0.4,
            direction="down",
        ),
    ]
    result = simulate_disagrees_pnl(jumps, prices, hold_hours=6)
    assert result["n_disagrees"] == 1
    assert result["n_agrees"] == 1
    # Disagrees trade was a winning short (0.60 → 0.55, short YES)
    assert result["disagrees_pnl"]["mean_return"] > 0.0
    # Equity curve contains only the disagrees trade.
    assert len(result["equity_curve"]) == 1


def test_neutral_alignment_is_ignored() -> None:
    """Jumps with sentiment_alignment='neutral' must not contribute to
    either bucket — they're simply skipped."""
    prices = _hourly_series([0.50] * 30)
    jumps = [
        _jump(
            ts="2026-05-01T05:00:00Z",
            price_after=0.60,
            alignment="neutral",
            sentiment=0.0,
            direction="up",
        ),
        _jump(
            ts="2026-05-01T10:00:00Z",
            price_after=0.40,
            alignment="neutral",
            sentiment=0.0,
            direction="down",
        ),
    ]
    result = simulate_disagrees_pnl(jumps, prices, hold_hours=6)
    assert result["n_disagrees"] == 0
    assert result["n_agrees"] == 0


def test_jump_at_end_of_series_is_skipped_not_crashed() -> None:
    """If a jump's timestamp + hold_hours is past the last observation,
    the trade is silently skipped — not an error."""
    # Series ends at hour 5. Jump is at hour 4 with 6h holding period → no exit.
    vals = [0.50, 0.50, 0.50, 0.50, 0.50, 0.40]
    prices = _hourly_series(vals)
    j = _jump(
        ts="2026-05-01T04:00:00Z",
        price_after=0.40,
        alignment="disagrees",
        sentiment=0.5,
        direction="down",
    )
    result = simulate_disagrees_pnl([j], prices, hold_hours=6)
    assert result["n_disagrees"] == 0
    assert result["disagrees_pnl"]["mean_return"] == 0.0


def test_sharpe_annualisation_matches_formula() -> None:
    """Pin down the naive Sharpe formula: mean / std × √(252×24 / hold)."""
    # Construct prices that yield exact returns for two trades. We
    # want returns [0.10, 0.05] (long YES) → mean=0.075, pop-std=0.025.
    # Build a long flat series and patch two windows.
    vals = [0.50] * 40
    # Trade 1: entry hour 5 @ 0.40, exit hour 11 @ 0.44 → (0.44-0.40)/0.40 = 0.10
    vals[5] = 0.40
    vals[11] = 0.44
    # Trade 2: entry hour 20 @ 0.40, exit hour 26 @ 0.42 → 0.05
    vals[20] = 0.40
    vals[26] = 0.42
    prices = _hourly_series(vals)
    jumps = [
        _jump(
            ts="2026-05-01T05:00:00Z",
            price_after=0.40,
            alignment="disagrees",
            sentiment=0.5,
            direction="down",
        ),
        _jump(
            ts="2026-05-01T20:00:00Z",
            price_after=0.40,
            alignment="disagrees",
            sentiment=0.5,
            direction="down",
        ),
    ]
    hold = 6
    result = simulate_disagrees_pnl(jumps, prices, hold_hours=hold)
    assert result["n_disagrees"] == 2
    mean = result["disagrees_pnl"]["mean_return"]
    std = result["disagrees_pnl"]["std_return"]
    expected = (mean / std) * math.sqrt(TRADING_HOURS_PER_YEAR / hold)
    assert result["disagrees_pnl"]["sharpe_naive"] == pytest.approx(expected, rel=1e-6)
    # Sanity: mean ≈ 0.075, pop-std ≈ 0.025, ratio = 3.0 — strongly positive
    # before annualisation.
    assert mean == pytest.approx(0.075, abs=1e-6)
    assert std == pytest.approx(0.025, abs=1e-6)


def test_single_trade_sharpe_is_zero_not_nan() -> None:
    """With n=1 there's no std → Sharpe must collapse to 0, not NaN."""
    vals = [0.50] * 20
    vals[5] = 0.40
    vals[11] = 0.45
    prices = _hourly_series(vals)
    j = _jump(
        ts="2026-05-01T05:00:00Z",
        price_after=0.40,
        alignment="disagrees",
        sentiment=0.5,
        direction="down",
    )
    result = simulate_disagrees_pnl([j], prices, hold_hours=6)
    assert result["n_disagrees"] == 1
    assert result["disagrees_pnl"]["sharpe_naive"] == 0.0
    assert math.isfinite(result["disagrees_pnl"]["sharpe_naive"])


def test_short_side_inverts_pnl_sign() -> None:
    """Bearish news (sentiment<0) on an up-jump → short YES. If YES
    keeps rising (no reversion) → negative PnL."""
    vals = [0.50] * 30
    vals[5] = 0.60  # up-jump
    vals[11] = 0.70  # kept rising
    prices = _hourly_series(vals)
    j = _jump(
        ts="2026-05-01T05:00:00Z",
        price_after=0.60,
        alignment="disagrees",
        sentiment=-0.4,
        direction="up",
    )
    result = simulate_disagrees_pnl([j], prices, hold_hours=6)
    # Short YES: entry 0.60, exit 0.70 → -0.166...
    assert result["disagrees_pnl"]["mean_return"] == pytest.approx(-1 / 6, abs=1e-4)


def test_alignment_fallback_when_sentiment_score_zero() -> None:
    """When sentiment_score=0 but alignment is set, side is recovered
    from alignment + direction (disagrees + up-jump → short YES)."""
    vals = [0.50] * 30
    vals[5] = 0.60
    vals[11] = 0.55  # YES dropped → shorts win
    prices = _hourly_series(vals)
    j = _jump(
        ts="2026-05-01T05:00:00Z",
        price_after=0.60,
        alignment="disagrees",
        sentiment=0.0,  # no signed sentiment — alignment must be used
        direction="up",
    )
    result = simulate_disagrees_pnl([j], prices, hold_hours=6)
    assert result["n_disagrees"] == 1
    assert result["disagrees_pnl"]["mean_return"] > 0.0


def test_hold_hours_changes_exit() -> None:
    """The same jump with different hold periods yields different PnLs
    when the price path is non-monotone — pins the exit-selection
    behaviour."""
    vals = [0.50] * 30
    vals[5] = 0.40  # entry after jump
    vals[8] = 0.45  # +3h: partial revert
    vals[11] = 0.50  # +6h: full revert
    prices = _hourly_series(vals)
    j = _jump(
        ts="2026-05-01T05:00:00Z",
        price_after=0.40,
        alignment="disagrees",
        sentiment=0.5,
        direction="down",
    )
    r3 = simulate_disagrees_pnl([j], prices, hold_hours=3)
    r6 = simulate_disagrees_pnl([j], prices, hold_hours=6)
    assert r6["disagrees_pnl"]["mean_return"] > r3["disagrees_pnl"]["mean_return"] > 0.0


def test_default_hold_hours_is_six() -> None:
    """Pin the default so changes have to be explicit."""
    assert DEFAULT_HOLD_HOURS == 6


# ---------------------------------------------------------------------------
# Endpoint integration test (mocked fetches)
# ---------------------------------------------------------------------------


class _StubMeta:
    def __init__(self) -> None:
        self.question = "Will Trump win the 2024 election?"
        self.yes_token_id = "1234567890"


class _StubPoly:
    def __init__(self) -> None:
        self._client = None
        self.clob_url = "https://clob.example"

    def get_market_metadata(self, slug: str) -> _StubMeta:
        return _StubMeta()


def _stub_prices(*_args, **_kwargs) -> pd.Series:
    """Flat at 0.50 with a drop to 0.40 at hour 6 and a revert to 0.48
    by hour 12 — so the disagrees long should be profitable."""
    vals = [0.50] * 24
    vals[6] = 0.40
    vals[12] = 0.48
    return _hourly_series(vals)


def _stub_bullish_articles(*_args, **_kwargs) -> list[GDELTArticle]:
    """A clearly-bullish headline right before the dip — paired with
    the down-jump this should yield sentiment_alignment='disagrees'."""
    return [
        GDELTArticle(
            ts="2026-05-01T05:30:00Z",
            title="Trump surges in new poll, election odds rally on positive news",
            source="reuters.com",
            country="us",
            language="english",
            tone=3.5,
            url="https://reuters.example/x",
        ),
    ]


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(router)
    app.state.poly = _StubPoly()  # type: ignore[assignment]
    return TestClient(app)


def test_endpoint_runs_end_to_end_with_disagrees_jump() -> None:
    """Wire the endpoint with mocked price + news fetches and verify
    the response carries the expected keys + a non-empty disagrees
    bucket (the synthetic series was designed to fire one)."""
    with (
        patch("pfm.terminal.jumps_backtest._fetch_hourly_prices", _stub_prices),
        patch("pfm.terminal.jumps_backtest._gather_all_news", _stub_bullish_articles),
    ):
        c = _client()
        r = c.get("/terminal/jumps/trump-2024/backtest?days=2&hold_hours=6&mad_k=2.0&min_jump_pp=3")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["slug"] == "trump-2024"
    assert body["hold_hours"] == 6
    # Schema contract
    for k in (
        "n_disagrees",
        "n_agrees",
        "disagrees_pnl",
        "agrees_pnl",
        "equity_curve",
        "interpretation",
    ):
        assert k in body
    for k in (
        "n_trades",
        "mean_return",
        "std_return",
        "sharpe_naive",
        "hit_rate",
        "avg_win",
        "avg_loss",
        "max_drawdown",
        "total_return",
    ):
        assert k in body["disagrees_pnl"]
        assert k in body["agrees_pnl"]
    # The synthetic spike should fire as a disagrees jump (bullish news,
    # price fell) and revert toward 0.48 by hour 12 → positive mean.
    assert body["n_disagrees"] >= 1
    assert body["disagrees_pnl"]["mean_return"] > 0.0


def test_endpoint_validates_hold_hours_range() -> None:
    c = _client()
    # hold_hours below 1 rejected
    r = c.get("/terminal/jumps/x/backtest?hold_hours=0")
    assert r.status_code == 422
    # hold_hours above 48 rejected
    r = c.get("/terminal/jumps/x/backtest?hold_hours=72")
    assert r.status_code == 422


def test_endpoint_validates_days_range() -> None:
    c = _client()
    r = c.get("/terminal/jumps/x/backtest?days=0")
    assert r.status_code == 422
    r = c.get("/terminal/jumps/x/backtest?days=100")
    assert r.status_code == 422


def test_endpoint_returns_404_when_market_not_found() -> None:
    class _BadPoly:
        _client = None
        clob_url = "https://clob.example"

        def get_market_metadata(self, slug: str):
            raise ValueError("no such slug")

    app = FastAPI()
    app.include_router(router)
    app.state.poly = _BadPoly()
    c = TestClient(app)
    r = c.get("/terminal/jumps/does-not-exist/backtest")
    assert r.status_code == 404
    assert "market not found" in r.json()["detail"]


def test_endpoint_503_when_polymarket_client_missing() -> None:
    app = FastAPI()
    app.include_router(router)
    # NOTE: deliberately no app.state.poly
    c = TestClient(app)
    r = c.get("/terminal/jumps/anything/backtest")
    assert r.status_code == 503


# ---------------------------------------------------------------------------
# Additional edge cases (filling coverage gaps in jumps_backtest.py)
# ---------------------------------------------------------------------------


def test_simulate_skips_jumps_when_hold_hours_exceeds_series_duration() -> None:
    """``hold_hours`` past the end of the series → every jump is silently
    skipped (no exit price). Covers the `_exit_price_at` None branch en masse."""
    # 12-hour series; jumps at hours 2 and 4; hold 48h → no exits.
    prices = _hourly_series([0.50] * 12)
    jumps = [
        _jump(
            ts="2026-05-01T02:00:00Z",
            price_after=0.40,
            alignment="disagrees",
            sentiment=0.6,
            direction="down",
        ),
        _jump(
            ts="2026-05-01T04:00:00Z",
            price_after=0.45,
            alignment="agrees",
            sentiment=0.5,
            direction="up",
        ),
    ]
    result = simulate_disagrees_pnl(jumps, prices, hold_hours=48)
    assert result["n_disagrees"] == 0
    assert result["n_agrees"] == 0
    assert result["equity_curve"] == []
    assert result["disagrees_pnl"]["sharpe_naive"] == 0.0
    assert math.isfinite(result["disagrees_pnl"]["mean_return"])


def test_simulate_rejects_non_positive_hold_hours() -> None:
    """``hold_hours <= 0`` is a programmer error — raise ValueError so the
    caller doesn't silently produce nonsense stats."""
    with pytest.raises(ValueError):
        simulate_disagrees_pnl([], _hourly_series([0.5] * 5), hold_hours=0)
    with pytest.raises(ValueError):
        simulate_disagrees_pnl([], _hourly_series([0.5] * 5), hold_hours=-3)


def test_mixed_long_and_short_within_disagrees_bucket() -> None:
    """A single bucket can hold both directions. Pin the math: PnL contributions
    are signed by the trade side, not the bucket label."""
    vals = [0.50] * 30
    # Jump 1: bullish news + price fell → long YES, then reverts → +PnL
    vals[5] = 0.40
    vals[11] = 0.50  # revert: long YES wins
    # Jump 2: bearish news + price rose → short YES, then keeps rising → -PnL
    vals[15] = 0.60
    vals[21] = 0.65  # keeps going: short YES loses
    prices = _hourly_series(vals)

    jumps = [
        _jump(
            ts="2026-05-01T05:00:00Z",
            price_after=0.40,
            alignment="disagrees",
            sentiment=0.5,  # bullish news
            direction="down",
        ),
        _jump(
            ts="2026-05-01T15:00:00Z",
            price_after=0.60,
            alignment="disagrees",
            sentiment=-0.5,  # bearish news
            direction="up",
        ),
    ]
    result = simulate_disagrees_pnl(jumps, prices, hold_hours=6)
    assert result["n_disagrees"] == 2
    # First trade ≈ +0.25; second trade ≈ -0.0833. Both signed correctly.
    # Mean is positive, hit_rate is 0.5 (one win, one loss).
    assert result["disagrees_pnl"]["hit_rate"] == 0.5
    assert result["disagrees_pnl"]["avg_win"] > 0
    assert result["disagrees_pnl"]["avg_loss"] < 0
    # Equity curve has two points, second one below the first peak.
    eq = result["equity_curve"]
    assert len(eq) == 2
    # The cumulative-sum dip after trade 2 must produce a non-zero drawdown.
    assert result["disagrees_pnl"]["max_drawdown"] < 0


def test_single_trade_n_eq_1_has_zero_sharpe_and_zero_drawdown() -> None:
    """n=1 disagrees, profitable: Sharpe should be exactly 0 (no std), and
    max_drawdown should be 0 (one trade can't draw down)."""
    vals = [0.50] * 20
    vals[5] = 0.40
    vals[11] = 0.50
    prices = _hourly_series(vals)
    j = _jump(
        ts="2026-05-01T05:00:00Z",
        price_after=0.40,
        alignment="disagrees",
        sentiment=0.5,
        direction="down",
    )
    result = simulate_disagrees_pnl([j], prices, hold_hours=6)
    assert result["n_disagrees"] == 1
    assert result["disagrees_pnl"]["sharpe_naive"] == 0.0
    # mean_return > 0 (profitable trade) but max_drawdown is exactly 0.
    assert result["disagrees_pnl"]["mean_return"] > 0
    assert result["disagrees_pnl"]["max_drawdown"] == 0.0


def test_jump_with_missing_ts_or_price_is_skipped() -> None:
    """Malformed jump rows (missing ``ts_iso`` or ``price_after``) must not
    crash the simulator — they're silently skipped."""
    prices = _hourly_series([0.50] * 20)
    jumps = [
        # No ts
        {
            "price_after": 0.40,
            "sentiment_alignment": "disagrees",
            "news_sentiment_score": 0.5,
            "direction": "down",
        },
        # No price_after
        {
            "ts_iso": "2026-05-01T05:00:00Z",
            "sentiment_alignment": "disagrees",
            "news_sentiment_score": 0.5,
            "direction": "down",
        },
        # Unparseable ts
        {
            "ts_iso": "definitely-not-a-date",
            "price_after": 0.40,
            "sentiment_alignment": "disagrees",
            "news_sentiment_score": 0.5,
            "direction": "down",
        },
    ]
    result = simulate_disagrees_pnl(jumps, prices, hold_hours=6)
    assert result["n_disagrees"] == 0
    assert result["n_agrees"] == 0


def test_jump_with_non_positive_entry_price_is_skipped() -> None:
    """A degenerate ``entry=0`` row cannot produce a valid return; skip it."""
    vals = [0.50] * 20
    vals[11] = 0.55
    prices = _hourly_series(vals)
    j = _jump(
        ts="2026-05-01T05:00:00Z",
        price_after=0.0,  # degenerate
        alignment="disagrees",
        sentiment=0.5,
        direction="down",
    )
    result = simulate_disagrees_pnl([j], prices, hold_hours=6)
    assert result["n_disagrees"] == 0


def test_alignment_fallback_disagrees_down_jump_recovers_long_side() -> None:
    """Mirror of the existing alignment fallback test for the *opposite*
    direction: disagrees + down-jump implies bullish news (price fell
    against bullish wires) → long YES. Pin the (disagrees, down) branch."""
    vals = [0.50] * 30
    vals[5] = 0.40
    vals[11] = 0.50  # YES recovered → longs win
    prices = _hourly_series(vals)
    j = _jump(
        ts="2026-05-01T05:00:00Z",
        price_after=0.40,
        alignment="disagrees",
        sentiment=0.0,  # no signed sentiment; alignment+direction must be used
        direction="down",
    )
    result = simulate_disagrees_pnl([j], prices, hold_hours=6)
    assert result["n_disagrees"] == 1
    # Long YES profitable: (0.50 - 0.40) / 0.40 = +0.25
    assert result["disagrees_pnl"]["mean_return"] == pytest.approx(0.25, abs=1e-4)


def test_alignment_fallback_agrees_down_jump_recovers_short_side() -> None:
    """Agrees + down-jump → bearish news + price falling → short YES. Mirror
    of the existing (agrees, up → long) implicit coverage."""
    vals = [0.50] * 30
    vals[5] = 0.40
    vals[11] = 0.35  # YES kept falling → shorts win
    prices = _hourly_series(vals)
    j = _jump(
        ts="2026-05-01T05:00:00Z",
        price_after=0.40,
        alignment="agrees",
        sentiment=0.0,
        direction="down",
    )
    result = simulate_disagrees_pnl([j], prices, hold_hours=6)
    assert result["n_agrees"] == 1
    # Short YES: (0.40 - 0.35) / 0.40 = +0.125
    assert result["agrees_pnl"]["mean_return"] == pytest.approx(0.125, abs=1e-4)


def test_news_side_no_signal_when_alignment_neutral_and_score_zero() -> None:
    """If both the sentiment_score is ~0 AND alignment is 'neutral', the
    trade has no actionable side and is skipped entirely."""
    from pfm.terminal.jumps_backtest import _news_side

    j = {
        "news_sentiment_score": 0.0,
        "sentiment_alignment": "neutral",
        "direction": "up",
    }
    assert _news_side(j) == 0


def test_news_side_uses_explicit_sentiment_score_over_alignment() -> None:
    """When the sentiment_score is non-zero, it wins regardless of alignment.
    Pins the priority order documented in the module docstring."""
    from pfm.terminal.jumps_backtest import _news_side

    # Positive score → +1 even if alignment hints differently.
    j_pos = {
        "news_sentiment_score": 0.4,
        "sentiment_alignment": "agrees",
        "direction": "down",
    }
    assert _news_side(j_pos) == 1
    # Negative score → -1.
    j_neg = {
        "news_sentiment_score": -0.7,
        "sentiment_alignment": "disagrees",
        "direction": "up",
    }
    assert _news_side(j_neg) == -1


def test_simulate_returns_404_path_unaffected_by_empty_jumps() -> None:
    """Empty jump list + non-empty prices: schema is unchanged, totals are 0."""
    result = simulate_disagrees_pnl([], _hourly_series([0.5] * 24), hold_hours=6)
    for key in (
        "n_trades",
        "mean_return",
        "std_return",
        "sharpe_naive",
        "hit_rate",
        "avg_win",
        "avg_loss",
        "max_drawdown",
        "total_return",
    ):
        assert result["disagrees_pnl"][key] == 0 or result["disagrees_pnl"][key] == 0.0
        assert result["agrees_pnl"][key] == 0 or result["agrees_pnl"][key] == 0.0


def test_simulate_with_tz_naive_price_index_normalises_to_utc() -> None:
    """A tz-naive price series must be coerced to UTC internally — assert by
    feeding a naive index and verifying jumps still find their exit price."""
    # Build tz-naive series (no `tz=` arg).
    idx = pd.date_range(start="2026-05-01T00:00:00", periods=20, freq="h")
    vals = [0.50] * 20
    vals[5] = 0.40
    vals[11] = 0.48
    prices = pd.Series(vals, index=idx, name="price", dtype=float)
    j = _jump(
        ts="2026-05-01T05:00:00Z",
        price_after=0.40,
        alignment="disagrees",
        sentiment=0.5,
        direction="down",
    )
    result = simulate_disagrees_pnl([j], prices, hold_hours=6)
    # Trade was evaluable despite the tz-naive series — proves the localize
    # branch ran without error.
    assert result["n_disagrees"] == 1
    assert result["disagrees_pnl"]["mean_return"] > 0
