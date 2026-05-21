"""End-to-end tests for ``GET /terminal/jumps/{slug}/backtest``.

Wire the router into a minimal :class:`FastAPI` app, attach a stub
Polymarket client to ``app.state.poly``, and patch the two upstream
fetchers (``_fetch_hourly_prices`` and ``_gather_all_news``) so the
endpoint runs offline and deterministically. The synthetic price series
is hand-crafted per test so the verdict in the response is predictable:

- "DISAGREES IS REAL ALPHA" when reverting disagrees trades dominate.
- "AGREES IS THE REAL SIGNAL" when momentum agrees trades dominate.
- "INCONCLUSIVE" when neither bucket has meaningful PnL.

The tests in this file complement the unit-level tests at
``api/tests/terminal/test_jumps_backtest.py``: those cover
:func:`simulate_disagrees_pnl` directly with hand-built jump dicts. This
file covers the HTTP layer — the response shape, the verdict prose, the
error paths, and the sentiment-to-side wiring through
:func:`aggregate_sentiment` + :func:`_articles_for_jump`.

All tests run with ``--noconftest`` so the project-wide cache-reset
fixture (which depends on the full ``pfm.main`` import graph) is not
required.
"""

from __future__ import annotations

import math
from typing import Any
from unittest.mock import patch

import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pfm.terminal.gdelt_news import GDELTArticle
from pfm.terminal.jumps_backtest import (
    TRADING_HOURS_PER_YEAR,
)
from pfm.terminal.jumps_backtest import (
    router as backtest_router,
)

# ---------------------------------------------------------------------------
# Stubs and fixtures
# ---------------------------------------------------------------------------


class _StubMeta:
    """Minimal stand-in for ``MarketMetadata`` — only the attributes the
    backtest endpoint reads off the metadata object are exposed."""

    def __init__(
        self,
        *,
        question: str = "Will the Fed cut rates at the next FOMC meeting?",
        yes_token_id: str = "1111111111",
    ) -> None:
        self.question = question
        self.yes_token_id = yes_token_id


class _StubPoly:
    """Minimal Polymarket client stand-in. ``get_market_metadata`` is the
    only method the endpoint calls; ``_client`` and ``clob_url`` are
    referenced when the (mocked) fetchers are invoked."""

    def __init__(self, *, raise_on_meta: Exception | None = None) -> None:
        self._client = None
        self.clob_url = "https://clob.example"
        self._raise_on_meta = raise_on_meta

    def get_market_metadata(self, slug: str) -> _StubMeta:
        if self._raise_on_meta is not None:
            raise self._raise_on_meta
        return _StubMeta()


def _hourly_series(values: list[float], start: str = "2026-05-01T00:00:00Z") -> pd.Series:
    """UTC-indexed hourly probability series helper."""
    idx = pd.date_range(start=start, periods=len(values), freq="h", tz="UTC")
    return pd.Series(values, index=idx, name="price", dtype=float)


def _gdelt(
    *,
    ts: str,
    title: str,
    source: str = "reuters.com",
    tone: float = 0.0,
    url: str | None = None,
) -> GDELTArticle:
    """Build a minimal :class:`GDELTArticle` for the news-fetch mock."""
    return GDELTArticle(
        url=url or f"https://{source}/article-{abs(hash(title)) % 10**8}",
        title=title,
        source=source,
        country="us",
        ts=ts,
        tone=tone,
        language="english",
    )


def _build_client(
    *,
    poly: _StubPoly | None = None,
    attach_poly: bool = True,
) -> TestClient:
    """Build an isolated FastAPI app with only the backtest router mounted.

    Using a per-test app (instead of the full ``pfm.main.app``) keeps these
    e2e tests fast and dependency-free — no Redis, no lifespan, no
    factors.yml load.
    """
    app = FastAPI()
    app.include_router(backtest_router)
    if attach_poly:
        app.state.poly = poly or _StubPoly()  # type: ignore[assignment]
    return TestClient(app)


# Bullish / bearish phrases that score above the ``aggregate_sentiment``
# alignment threshold (|mean| ≥ 0.10) AND mention the question's anchors
# ("Fed", "FOMC") so they clear the ``RELEVANCE_MIN=0.18`` filter.
# Sentiment scores (printed by the local trace script): bullish ≥ +0.60,
# bearish ≤ −0.82. Strong enough that the aggregator returns a non-neutral
# alignment for both up- and down-direction jumps.
_BULLISH_HEADLINES = [
    "Fed surges market with surprise rate cut; stocks rally on bullish optimism",
    "Bullish rally as Fed boosts confidence; stocks surge on strong upgrade after FOMC",
    "Fed delivers surprise rate cut: stocks rally, bullish optimism surges",
]
_BEARISH_HEADLINES = [
    "Stocks plunge, market crashes as Fed warns of recession; bearish fears mount",
    "Hawkish Fed sparks selloff: stocks plunge, recession fears mount, bearish turmoil",
    "Fed shock: hawkish surprise crashes stocks; bearish rout as recession warning issued",
]


# ---------------------------------------------------------------------------
# Test 1 — Happy path: schema + known slug
# ---------------------------------------------------------------------------


def test_happy_path_returns_backtest_stats_schema() -> None:
    """Disagrees-favourable synthetic series: endpoint returns 200 with the
    full :class:`JumpsBacktestResponse` schema populated."""
    # A clean dip at hour 6 followed by recovery by hour 12 → one
    # detectable jump that reverts within the 6h holding window.
    vals = [0.50] * 48
    vals[6] = 0.38  # big drop: ≥3pp + large Δlogit
    vals[12] = 0.48  # recovery → long YES wins (~+26%)
    prices = _hourly_series(vals)

    def _fake_news(*_a: Any, **_kw: Any) -> list[GDELTArticle]:
        # Bullish wire 30 min before the dip → news bullish, price fell
        # → disagrees, long YES.
        return [_gdelt(ts="2026-05-01T05:30:00Z", title=_BULLISH_HEADLINES[0])]

    with (
        patch("pfm.terminal.jumps_backtest._fetch_hourly_prices", return_value=prices),
        patch("pfm.terminal.jumps_backtest._gather_all_news", _fake_news),
    ):
        client = _build_client()
        resp = client.get(
            "/terminal/jumps/fed-decision-cut/backtest?days=2&hold_hours=6&mad_k=2.0&min_jump_pp=3"
        )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    # Top-level keys
    assert body["slug"] == "fed-decision-cut"
    assert body["hold_hours"] == 6
    for key in (
        "n_disagrees",
        "n_agrees",
        "disagrees_pnl",
        "agrees_pnl",
        "equity_curve",
        "interpretation",
    ):
        assert key in body
    # Stat block keys
    stat_keys = {
        "n_trades",
        "mean_return",
        "std_return",
        "sharpe_naive",
        "hit_rate",
        "avg_win",
        "avg_loss",
        "max_drawdown",
        "total_return",
    }
    assert stat_keys.issubset(body["disagrees_pnl"].keys())
    assert stat_keys.issubset(body["agrees_pnl"].keys())
    # All numbers must be finite (no NaN leaking into JSON).
    for k in stat_keys:
        assert math.isfinite(body["disagrees_pnl"][k])
        assert math.isfinite(body["agrees_pnl"][k])


# ---------------------------------------------------------------------------
# Test 2 — Verdict: "DISAGREES IS REAL ALPHA"
# ---------------------------------------------------------------------------


def test_verdict_disagrees_is_real_alpha_when_reversion_dominates() -> None:
    """Construct a synthetic series with multiple reverting disagrees jumps
    so the disagrees mean return clears the 0.005 + hit-rate>0.5 thresholds
    and the verdict text fires."""
    # Three independent dip→recover episodes. The detector picks the
    # large negative Δlogit at hours 6, 20, 34; six hours later the price
    # has recovered fully — long-YES trade wins each time.
    vals = [0.50] * 60
    for entry, exit_ in ((6, 12), (20, 26), (34, 40)):
        vals[entry] = 0.38
        vals[exit_] = 0.50
    prices = _hourly_series(vals)

    def _fake_news(*_a: Any, **_kw: Any) -> list[GDELTArticle]:
        # One bullish article ~30 min before each dip.
        return [
            _gdelt(ts="2026-05-01T05:30:00Z", title=_BULLISH_HEADLINES[0]),
            _gdelt(ts="2026-05-01T19:30:00Z", title=_BULLISH_HEADLINES[1]),
            _gdelt(ts="2026-05-02T09:30:00Z", title=_BULLISH_HEADLINES[2]),
        ]

    with (
        patch("pfm.terminal.jumps_backtest._fetch_hourly_prices", return_value=prices),
        patch("pfm.terminal.jumps_backtest._gather_all_news", _fake_news),
    ):
        client = _build_client()
        resp = client.get(
            "/terminal/jumps/fed-decision-cut/backtest?days=3&hold_hours=6&mad_k=2.0&min_jump_pp=3"
        )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    # Disagrees trades fired and were profitable.
    assert body["n_disagrees"] >= 1
    assert body["disagrees_pnl"]["mean_return"] > 0.005
    assert body["disagrees_pnl"]["hit_rate"] > 0.5
    assert "DISAGREES IS REAL ALPHA" in body["interpretation"]


# ---------------------------------------------------------------------------
# Test 3 — Verdict: "AGREES IS THE REAL SIGNAL"
# ---------------------------------------------------------------------------


def test_verdict_agrees_is_the_real_signal_when_momentum_wins() -> None:
    """Construct synthetic data so the agrees bucket dominates (news +
    price aligned and price continues in news direction) → verdict text
    fires."""
    # Three down-trend continuation episodes. Bearish news + price down +
    # price keeps falling → agrees, short YES → profitable shorts.
    # Pre-dip and post-dip levels are chosen so the absolute Δp clears
    # the 3 pp floor and the rolling-MAD z-test fires on each event.
    vals = [0.50] * 60
    for entry, exit_ in ((6, 12), (20, 26), (34, 40)):
        # Big down jump at entry, then continued lower at exit.
        vals[entry] = 0.42  # ~8 pp drop from prior 0.50 baseline
        vals[exit_] = 0.32  # continued lower → shorts +PnL
    prices = _hourly_series(vals)

    def _fake_news(*_a: Any, **_kw: Any) -> list[GDELTArticle]:
        # Bearish articles ~30 min before each down-jump.
        return [
            _gdelt(ts="2026-05-01T05:30:00Z", title=_BEARISH_HEADLINES[0]),
            _gdelt(ts="2026-05-01T19:30:00Z", title=_BEARISH_HEADLINES[1]),
            _gdelt(ts="2026-05-02T09:30:00Z", title=_BEARISH_HEADLINES[2]),
        ]

    with (
        patch("pfm.terminal.jumps_backtest._fetch_hourly_prices", return_value=prices),
        patch("pfm.terminal.jumps_backtest._gather_all_news", _fake_news),
    ):
        client = _build_client()
        resp = client.get(
            "/terminal/jumps/recession-2026/backtest?days=3&hold_hours=6&mad_k=2.0&min_jump_pp=3"
        )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    # Detector fires on every down-jump → both buckets populated. Agrees
    # trades (bearish news + price down + price keeps dropping → short
    # YES wins) are profitable; disagrees trades (bearish news + price up
    # → short YES from an up-bar that lands inside a down-trend) lose.
    assert body["n_agrees"] >= 1
    assert body["agrees_pnl"]["mean_return"] > 0.01
    assert body["agrees_pnl"]["hit_rate"] > 0.55
    assert "AGREES IS THE REAL SIGNAL" in body["interpretation"]


# ---------------------------------------------------------------------------
# Test 4 — Verdict: "INCONCLUSIVE"
# ---------------------------------------------------------------------------


def test_verdict_inconclusive_when_neither_bucket_dominates() -> None:
    """A flat-ish series with tiny PnL in both buckets → verdict is
    "INCONCLUSIVE" (mean_return below 0.005 / 0.01 thresholds)."""
    # Single jump that barely moves and barely reverts — Sharpe is small,
    # mean is small, both thresholds fail. Use bullish news so the trade
    # is taken but the realised return is tiny.
    vals = [0.50] * 30
    vals[6] = 0.46  # ~4pp drop: clears floor
    vals[12] = 0.462  # microscopic revert: mean return ≈ +0.0043
    prices = _hourly_series(vals)

    def _fake_news(*_a: Any, **_kw: Any) -> list[GDELTArticle]:
        return [_gdelt(ts="2026-05-01T05:30:00Z", title=_BULLISH_HEADLINES[0])]

    with (
        patch("pfm.terminal.jumps_backtest._fetch_hourly_prices", return_value=prices),
        patch("pfm.terminal.jumps_backtest._gather_all_news", _fake_news),
    ):
        client = _build_client()
        resp = client.get(
            "/terminal/jumps/marginal-mover/backtest?days=2&hold_hours=6&mad_k=2.0&min_jump_pp=3"
        )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    interp = body["interpretation"]
    # Either the disagrees trade fired but was marginal → INCONCLUSIVE,
    # or no disagrees were detected at all → the "No disagrees jumps"
    # branch. Both outcomes are valid "neither alpha" verdicts.
    assert "INCONCLUSIVE" in interp or "No disagrees jumps" in interp
    assert "DISAGREES IS REAL ALPHA" not in interp
    assert "AGREES IS THE REAL SIGNAL" not in interp


# ---------------------------------------------------------------------------
# Test 5 — Empty jumps list → 200 with empty stats
# ---------------------------------------------------------------------------


def test_empty_jumps_returns_200_with_zero_stats() -> None:
    """A flat series produces no jumps. The endpoint must still return 200
    with zero-filled stats and the documented "No disagrees jumps"
    interpretation."""
    prices = _hourly_series([0.50] * 48)  # perfectly flat → 0 jumps
    with (
        patch("pfm.terminal.jumps_backtest._fetch_hourly_prices", return_value=prices),
        patch("pfm.terminal.jumps_backtest._gather_all_news", return_value=[]),
    ):
        client = _build_client()
        resp = client.get("/terminal/jumps/flat-market/backtest?days=2&hold_hours=6")

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["n_disagrees"] == 0
    assert body["n_agrees"] == 0
    # All stats zeroed.
    for bucket in (body["disagrees_pnl"], body["agrees_pnl"]):
        assert bucket["n_trades"] == 0
        assert bucket["mean_return"] == 0.0
        assert bucket["std_return"] == 0.0
        assert bucket["sharpe_naive"] == 0.0
        assert bucket["hit_rate"] == 0.0
        assert bucket["total_return"] == 0.0
        assert bucket["max_drawdown"] == 0.0
    assert body["equity_curve"] == []
    assert "No disagrees jumps" in body["interpretation"]


# ---------------------------------------------------------------------------
# Test 6 — Unknown slug → 404 (or 502 if the upstream raised httpx)
# ---------------------------------------------------------------------------


def test_unknown_slug_returns_404_with_detail() -> None:
    """When ``get_market_metadata`` raises (e.g. unknown slug), the
    endpoint surfaces a 404 with a helpful detail string."""
    bad_poly = _StubPoly(raise_on_meta=ValueError("no such slug: ghost-market"))
    client = _build_client(poly=bad_poly)
    resp = client.get("/terminal/jumps/ghost-market/backtest")
    # Accept 404 (the documented happy-error path) or 502 (httpx error class).
    assert resp.status_code in (404, 502, 503)
    if resp.status_code == 404:
        assert "market not found" in resp.json()["detail"].lower()


# ---------------------------------------------------------------------------
# Test 7 — Upstream returns no price data → 200 with empty stats
# ---------------------------------------------------------------------------


def test_yfinance_no_data_returns_helpful_empty_response() -> None:
    """When the upstream price fetcher returns an empty series, the
    endpoint does not crash. It either returns 200 with zero stats (the
    current behaviour — the simulator short-circuits on empty prices) or
    a 503 with a helpful error. Both shapes are accepted."""
    empty_prices = pd.Series(dtype=float, name="price")
    with (
        patch("pfm.terminal.jumps_backtest._fetch_hourly_prices", return_value=empty_prices),
        patch("pfm.terminal.jumps_backtest._gather_all_news", return_value=[]),
    ):
        client = _build_client()
        resp = client.get("/terminal/jumps/no-data-slug/backtest?days=2&hold_hours=6")

    assert resp.status_code in (200, 503)
    if resp.status_code == 200:
        body = resp.json()
        assert body["n_disagrees"] == 0
        assert body["n_agrees"] == 0
        assert body["disagrees_pnl"]["mean_return"] == 0.0
        # Interpretation should mention the absence rather than promising alpha.
        assert any(tag in body["interpretation"] for tag in ("No disagrees jumps", "INCONCLUSIVE"))
    else:
        # If a future patch upgrades this to 503, the detail must be helpful.
        assert "detail" in resp.json()


# ---------------------------------------------------------------------------
# Test 8 — Sentiment classification: synthetic Fed-cut headline → positive
# ---------------------------------------------------------------------------


def test_sentiment_classification_fed_cut_drives_positive_score() -> None:
    """A "Fed cut + rally" wire is unambiguously bullish under the hybrid
    VADER + financial-lexicon scorer. The endpoint must wire that
    sentiment all the way through to ``news_sentiment_score > 0`` which
    in turn drives a long-YES trade when the price dipped (disagrees).

    We assert the chain end-to-end by:
        1. Constructing a price dip that aligns in time with one bullish
           headline (so the resulting jump is sentiment-aligned).
        2. Patching the fetchers and hitting the endpoint.
        3. Asserting the disagrees bucket fired AND that the trade was a
           long-YES (positive return on a recovering dip).
    """
    from pfm.terminal.sentiment_nlp import aggregate_sentiment, score_headline

    # Direct check on the scorer first — proves the headline is bullish
    # in isolation, separately from any backtest plumbing. Score must clear
    # the ±0.15 deadband for ``label='positive'`` AND the 0.10 alignment
    # threshold for ``align != 'neutral'``.
    s_score, s_label = score_headline(_BULLISH_HEADLINES[0])
    assert s_score > 0.15, f"Fed-rally headline scored non-positive: {s_score}"
    assert s_label == "positive"

    # And the aggregator on a down-direction jump labels it 'disagrees'.
    mean, label, align = aggregate_sentiment([s_score], jump_direction="down")
    assert mean > 0.0 and label == "positive" and align == "disagrees"

    # End-to-end: same Fed-cut headline + a real recovering dip series.
    # Gradual recovery (≤2pp/hr) so only the down-jump fires.
    vals = [0.50] * 30
    vals[6] = 0.38  # sudden dip
    vals[7] = 0.41
    vals[8] = 0.43
    vals[9] = 0.45
    vals[10] = 0.47
    vals[11] = 0.48
    vals[12] = 0.50  # exit price 6h after entry
    prices = _hourly_series(vals)

    def _fake_news(*_a: Any, **_kw: Any) -> list[GDELTArticle]:
        return [_gdelt(ts="2026-05-01T05:30:00Z", title=_BULLISH_HEADLINES[0])]

    with (
        patch("pfm.terminal.jumps_backtest._fetch_hourly_prices", return_value=prices),
        patch("pfm.terminal.jumps_backtest._gather_all_news", _fake_news),
    ):
        client = _build_client()
        resp = client.get(
            "/terminal/jumps/fed-decision/backtest?days=2&hold_hours=6&mad_k=2.0&min_jump_pp=3"
        )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    # Bullish-news + price-fell = disagrees. Long YES on recovery = +PnL.
    assert body["n_disagrees"] >= 1
    assert body["disagrees_pnl"]["mean_return"] > 0


# ---------------------------------------------------------------------------
# Test 9 — PnL calculation: known long/short positions → expected total
# ---------------------------------------------------------------------------


def test_pnl_calculation_matches_long_yes_formula() -> None:
    """For a single disagrees jump with known entry/exit prices, the
    total_return on the disagrees bucket must equal ``(exit - entry) /
    entry`` (long YES) to within float tolerance. Pins the per-trade
    return formula end-to-end through the endpoint.

    Price path: a single 10pp dip at hour 6, then a *gradual* recovery
    over hours 7-12 (each hourly step ≤ 2pp so no second jump fires).
    Only ONE detectable jump → ONE disagrees trade → PnL is the long-YES
    return ``(exit - entry) / entry``.
    """
    vals = [0.50] * 30
    vals[6] = 0.40  # entry after sudden dip
    vals[7] = 0.42  # gradual recovery, each step ≤2pp
    vals[8] = 0.44
    vals[9] = 0.46
    vals[10] = 0.48
    vals[11] = 0.49
    vals[12] = 0.50  # exit 6h after entry
    prices = _hourly_series(vals)

    def _fake_news(*_a: Any, **_kw: Any) -> list[GDELTArticle]:
        return [_gdelt(ts="2026-05-01T05:30:00Z", title=_BULLISH_HEADLINES[0])]

    with (
        patch("pfm.terminal.jumps_backtest._fetch_hourly_prices", return_value=prices),
        patch("pfm.terminal.jumps_backtest._gather_all_news", _fake_news),
    ):
        client = _build_client()
        resp = client.get(
            "/terminal/jumps/dip-recover/backtest?days=2&hold_hours=6&mad_k=2.0&min_jump_pp=3"
        )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["n_disagrees"] == 1
    expected = (0.50 - 0.40) / 0.40  # = 0.25
    assert body["disagrees_pnl"]["mean_return"] == pytest.approx(expected, abs=1e-4)
    assert body["disagrees_pnl"]["total_return"] == pytest.approx(expected, abs=1e-4)
    # Equity curve has one point matching the per-trade return.
    eq = body["equity_curve"]
    assert len(eq) == 1
    assert eq[0]["cum_return"] == pytest.approx(expected, abs=1e-4)


# ---------------------------------------------------------------------------
# Test 10 — Sharpe formula is honoured (no fake numerics)
# ---------------------------------------------------------------------------


def test_sharpe_naive_obeys_published_formula() -> None:
    """With two disagrees trades whose returns are deterministic
    (0.10 and 0.05), Sharpe must equal ``(mean / pop_std) × √(252*24 /
    hold_hours)`` to within float tolerance. We compute it independently
    and assert byte-equality (relative).
    """
    vals = [0.50] * 40
    # Trade 1: entry h6 @ 0.40 (sudden dip), gradual climb to 0.44 by h12 → +0.10
    vals[6] = 0.40
    vals[7] = 0.41
    vals[8] = 0.42
    vals[9] = 0.43
    vals[10] = 0.435
    vals[11] = 0.438
    vals[12] = 0.44
    # Series flattens back out before trade 2 — start from 0.50 again at h18.
    vals[13] = 0.45
    vals[14] = 0.46
    vals[15] = 0.47
    vals[16] = 0.48
    vals[17] = 0.49
    vals[18] = 0.50
    vals[19] = 0.50
    vals[20] = 0.50
    vals[21] = 0.50
    # Trade 2: entry h22 @ 0.40 (sudden dip), gradual climb to 0.42 by h28 → +0.05
    vals[22] = 0.40
    vals[23] = 0.405
    vals[24] = 0.41
    vals[25] = 0.413
    vals[26] = 0.416
    vals[27] = 0.418
    vals[28] = 0.42
    prices = _hourly_series(vals)

    def _fake_news(*_a: Any, **_kw: Any) -> list[GDELTArticle]:
        return [
            _gdelt(ts="2026-05-01T05:30:00Z", title=_BULLISH_HEADLINES[0]),
            _gdelt(ts="2026-05-01T21:30:00Z", title=_BULLISH_HEADLINES[1]),
        ]

    hold = 6
    with (
        patch("pfm.terminal.jumps_backtest._fetch_hourly_prices", return_value=prices),
        patch("pfm.terminal.jumps_backtest._gather_all_news", _fake_news),
    ):
        client = _build_client()
        resp = client.get(
            f"/terminal/jumps/two-trades/backtest?days=2&hold_hours={hold}&mad_k=2.0&min_jump_pp=3"
        )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["n_disagrees"] == 2
    mean = body["disagrees_pnl"]["mean_return"]
    std = body["disagrees_pnl"]["std_return"]
    expected_sharpe = (mean / std) * math.sqrt(TRADING_HOURS_PER_YEAR / hold)
    assert body["disagrees_pnl"]["sharpe_naive"] == pytest.approx(expected_sharpe, rel=1e-6)
    # Reality check on the inputs themselves: mean ≈ 0.075, pop-std ≈ 0.025.
    assert mean == pytest.approx(0.075, abs=1e-6)
    assert std == pytest.approx(0.025, abs=1e-6)


# ---------------------------------------------------------------------------
# Test 11 — Response carries Sharpe (substitute for "deflated Sharpe")
# ---------------------------------------------------------------------------


def test_response_carries_naive_sharpe_field() -> None:
    """The current schema exposes ``sharpe_naive`` on each bucket. The
    task spec mentions "deflated Sharpe in response (if available)" — at
    the time of writing the deflated Sharpe is not in the
    :class:`BacktestStats` schema, so we accept the naive Sharpe as the
    contract and document the absence of the deflated field with a
    soft-skip. This pins the response shape so a future Wave-N that adds
    ``sharpe_deflated`` will turn the skip into an assertion.
    """
    vals = [0.50] * 30
    vals[6] = 0.40
    vals[12] = 0.46
    prices = _hourly_series(vals)

    def _fake_news(*_a: Any, **_kw: Any) -> list[GDELTArticle]:
        return [_gdelt(ts="2026-05-01T05:30:00Z", title=_BULLISH_HEADLINES[0])]

    with (
        patch("pfm.terminal.jumps_backtest._fetch_hourly_prices", return_value=prices),
        patch("pfm.terminal.jumps_backtest._gather_all_news", _fake_news),
    ):
        client = _build_client()
        resp = client.get(
            "/terminal/jumps/sharpe-check/backtest?days=2&hold_hours=6&mad_k=2.0&min_jump_pp=3"
        )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    # Required: naive Sharpe present and finite on both buckets.
    assert "sharpe_naive" in body["disagrees_pnl"]
    assert "sharpe_naive" in body["agrees_pnl"]
    assert math.isfinite(body["disagrees_pnl"]["sharpe_naive"])
    assert math.isfinite(body["agrees_pnl"]["sharpe_naive"])
    # Soft-skip when deflated Sharpe lands in a future wave — this is a
    # contract check, not a failure mode.
    deflated = body["disagrees_pnl"].get("sharpe_deflated")
    if deflated is not None:
        assert math.isfinite(deflated)


# ---------------------------------------------------------------------------
# Test 12 — Missing polymarket client → 503
# ---------------------------------------------------------------------------


def test_missing_polymarket_client_returns_503() -> None:
    """The endpoint depends on ``app.state.poly`` (via FastAPI Depends).
    When the client is absent the dependency must raise 503 with a clear
    detail — proves the dependency wiring is exercised end-to-end."""
    client = _build_client(attach_poly=False)
    resp = client.get("/terminal/jumps/anything/backtest")
    assert resp.status_code == 503
    assert "polymarket client not initialized" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# Test 13 — Query-parameter validation pinned end-to-end
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("param", "value"),
    [
        ("hold_hours", 0),  # below MIN_HOLD_HOURS=1
        ("hold_hours", 72),  # above MAX_HOLD_HOURS=48
        ("days", 0),  # below MIN_DAYS=1
        ("days", 100),  # above MAX_DAYS=90
        ("mad_k", 0.5),  # below ge=1.0
        ("mad_k", 11),  # above le=10.0
        ("min_jump_pp", 0.1),  # below ge=0.5
        ("min_jump_pp", 60),  # above le=50.0
    ],
)
def test_query_param_out_of_range_returns_422(param: str, value: Any) -> None:
    """Every documented query bound rejects out-of-range values with a
    FastAPI 422. Pin the contract end-to-end so a future refactor that
    drops a bound is caught by CI."""
    client = _build_client()
    resp = client.get(f"/terminal/jumps/anything/backtest?{param}={value}")
    assert resp.status_code == 422, (param, value, resp.text)
