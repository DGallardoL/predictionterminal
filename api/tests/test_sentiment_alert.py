"""Tests for :mod:`pfm.alpha_hub.sentiment_alert` (Task W11-54).

We avoid the live Polymarket / GDELT stack — every test builds synthetic
backtest payloads or monkeypatches the I/O seams. The pure-function
detector :func:`check_sentiment_regression` is exercised exhaustively
(threshold edges, sample-size floor, frozenness, percentage rounding),
then the async collector is covered with a stubbed ``_one_backtest``.

Test inventory (>=10):
    1. ``test_below_threshold_returns_none``
    2. ``test_above_threshold_returns_alert``
    3. ``test_too_few_markets_returns_none_even_at_100_pct``
    4. ``test_exactly_at_threshold_does_not_trigger``
    5. ``test_empty_input_returns_none``
    6. ``test_none_input_returns_none``
    7. ``test_dataclass_is_frozen``
    8. ``test_percentage_calculation_rounds_to_one_decimal``
    9. ``test_slug_field_populated_from_results``
   10. ``test_interpretation_substring_match_case_insensitive``
   11. ``test_slugified_marker_also_matches``
   12. ``test_collect_recent_backtests_returns_expected_count``
   13. ``test_collect_recent_backtests_drops_failures``
   14. ``test_collect_handles_missing_poly_client``
   15. ``test_to_alert_row_severity_buckets``
   16. ``test_build_digest_rows_emits_one_row_on_trigger``
   17. ``test_build_digest_rows_empty_on_no_trigger``
"""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime
from typing import Any

import pytest

from pfm.alpha_hub.sentiment_alert import (
    DEFAULT_MIN_MARKETS,
    DEFAULT_THRESHOLD_PCT,
    SentimentRegressionAlert,
    build_digest_rows,
    check_sentiment_regression,
    collect_recent_backtests,
)

# ─────────────────────────────────────────────────────────────────────────────
# Fixtures: synthetic backtest payloads
# ─────────────────────────────────────────────────────────────────────────────


def _disagrees(slug: str) -> dict[str, Any]:
    """Backtest payload that carries the winning verdict."""
    return {
        "slug": slug,
        "hold_hours": 6,
        "n_disagrees": 8,
        "n_agrees": 3,
        "interpretation": (
            "8 disagrees + 3 agrees over 6h holding · disagrees: +0.0210 mean, "
            "hit 75%, Sharpe +1.4 · agrees: -0.0050 mean, hit 40%, Sharpe -0.3. → "
            "DISAGREES IS REAL ALPHA — news direction predicts reversal at 6h horizon."
        ),
    }


def _agrees(slug: str) -> dict[str, Any]:
    """Backtest payload with the *opposite* verdict (control)."""
    return {
        "slug": slug,
        "hold_hours": 6,
        "interpretation": "→ AGREES IS THE REAL SIGNAL — news predicts continuation.",
    }


def _inconclusive(slug: str) -> dict[str, Any]:
    return {
        "slug": slug,
        "hold_hours": 6,
        "interpretation": "→ INCONCLUSIVE — neither bucket has meaningful PnL.",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Pure-function detector
# ─────────────────────────────────────────────────────────────────────────────


def test_below_threshold_returns_none() -> None:
    # 2/10 = 20 % is well below 40 %.
    results = [
        *(_disagrees(f"m{i}") for i in range(2)),
        *(_inconclusive(f"n{i}") for i in range(8)),
    ]
    assert check_sentiment_regression(results) is None


def test_above_threshold_returns_alert() -> None:
    # 6/10 = 60 % strictly exceeds 40 %.
    results = [
        *(_disagrees(f"m{i}") for i in range(6)),
        *(_inconclusive(f"n{i}") for i in range(4)),
    ]
    alert = check_sentiment_regression(results)
    assert alert is not None
    assert alert.market_count == 10
    assert alert.disagrees_count == 6
    assert alert.disagrees_pct == 60.0
    assert alert.threshold_pct == DEFAULT_THRESHOLD_PCT
    assert alert.min_markets == DEFAULT_MIN_MARKETS


def test_too_few_markets_returns_none_even_at_100_pct() -> None:
    # 4 disagrees out of 4 = 100 %, but sample below the 5-floor.
    results = [_disagrees(f"m{i}") for i in range(4)]
    assert check_sentiment_regression(results) is None


def test_exactly_at_threshold_does_not_trigger() -> None:
    # 2/5 = 40.0 % — must NOT fire (strict greater-than).
    results = [
        *(_disagrees(f"m{i}") for i in range(2)),
        *(_inconclusive(f"n{i}") for i in range(3)),
    ]
    assert check_sentiment_regression(results) is None


def test_empty_input_returns_none() -> None:
    assert check_sentiment_regression([]) is None


def test_none_input_returns_none() -> None:
    assert check_sentiment_regression(None) is None


def test_dataclass_is_frozen() -> None:
    alert = SentimentRegressionAlert(
        triggered_at=datetime(2026, 5, 16, tzinfo=UTC),
        market_count=10,
        disagrees_count=6,
        disagrees_pct=60.0,
    )
    # dataclasses.FrozenInstanceError is the canonical exception.
    with pytest.raises(dataclasses.FrozenInstanceError):
        alert.market_count = 11  # type: ignore[misc]


def test_percentage_calculation_rounds_to_one_decimal() -> None:
    # 7/9 ≈ 77.777... → 77.8 after rounding.
    results = [
        *(_disagrees(f"m{i}") for i in range(7)),
        *(_inconclusive(f"n{i}") for i in range(2)),
    ]
    alert = check_sentiment_regression(results)
    assert alert is not None
    assert alert.disagrees_pct == 77.8


def test_slug_field_populated_from_results() -> None:
    results = [
        _disagrees("trump-2024-presidential-election"),
        _disagrees("fed-rate-cut-march-2026"),
        _disagrees("us-recession-2026"),
        _disagrees("nvda-earnings-q1-2026"),
        _disagrees("bitcoin-100k-2026"),
        _inconclusive("apple-earnings-beat-q1-2026"),
    ]
    alert = check_sentiment_regression(results)
    assert alert is not None
    # 5 disagrees / 6 markets = 83.3 % > 40 %.
    assert alert.disagrees_pct == 83.3
    assert "trump-2024-presidential-election" in alert.slugs
    assert "apple-earnings-beat-q1-2026" not in alert.slugs


def test_interpretation_substring_match_case_insensitive() -> None:
    # Mixed-case marker buried inside the interpretation should still
    # trigger the count.
    payload = {
        "slug": "x",
        "interpretation": "blah blah Disagrees Is Real Alpha — keep going.",
    }
    results = [
        payload,
        *(_disagrees(f"d{i}") for i in range(4)),
        _inconclusive("inc"),
    ]
    alert = check_sentiment_regression(results)
    # 5/6 = 83.3 % → fires.
    assert alert is not None
    assert alert.disagrees_count == 5


def test_slugified_marker_also_matches() -> None:
    payload = {"slug": "x", "verdict": "disagrees-is-real-alpha"}
    results = [
        payload,
        *(_disagrees(f"d{i}") for i in range(3)),
        *(_inconclusive(f"i{i}") for i in range(2)),
    ]
    alert = check_sentiment_regression(results)
    # 4/6 ≈ 66.7 % → fires.
    assert alert is not None
    assert alert.disagrees_count == 4


def test_threshold_custom_override() -> None:
    # 3/5 = 60 %; below a custom 70 % threshold → no fire.
    results = [
        *(_disagrees(f"m{i}") for i in range(3)),
        *(_inconclusive(f"n{i}") for i in range(2)),
    ]
    assert check_sentiment_regression(results, threshold_pct=70.0) is None
    # Same data, default 40 % → fires.
    assert check_sentiment_regression(results) is not None


def test_min_markets_custom_override() -> None:
    # 3 disagrees in 3 with floor=10 → no fire even at 100 %.
    results = [_disagrees(f"m{i}") for i in range(3)]
    assert check_sentiment_regression(results, min_markets=10) is None


def test_to_alert_row_severity_buckets() -> None:
    # High: 100% vs 40 % threshold → ratio 2.5 → high
    high = SentimentRegressionAlert(
        triggered_at=datetime(2026, 5, 16, tzinfo=UTC),
        market_count=5,
        disagrees_count=5,
        disagrees_pct=100.0,
    )
    assert high.to_alert_row()["severity"] == "high"
    # Med: 55% / 40% = 1.375 → med
    med = SentimentRegressionAlert(
        triggered_at=datetime(2026, 5, 16, tzinfo=UTC),
        market_count=20,
        disagrees_count=11,
        disagrees_pct=55.0,
    )
    assert med.to_alert_row()["severity"] == "med"
    # Low: 45% / 40% = 1.125 → low
    low = SentimentRegressionAlert(
        triggered_at=datetime(2026, 5, 16, tzinfo=UTC),
        market_count=20,
        disagrees_count=9,
        disagrees_pct=45.0,
    )
    assert low.to_alert_row()["severity"] == "low"


def test_to_alert_row_shape() -> None:
    alert = SentimentRegressionAlert(
        triggered_at=datetime(2026, 5, 16, 10, 30, tzinfo=UTC),
        market_count=10,
        disagrees_count=6,
        disagrees_pct=60.0,
    )
    row = alert.to_alert_row()
    assert row["kind"] == "sentiment-regression"
    assert row["market_count"] == 10
    assert row["disagrees_count"] == 6
    assert row["disagrees_pct"] == 60.0
    assert row["threshold_pct"] == DEFAULT_THRESHOLD_PCT
    assert row["triggered_at"].endswith("Z")
    assert "DISAGREES" in row["label"]


# ─────────────────────────────────────────────────────────────────────────────
# Async collector — patched I/O
# ─────────────────────────────────────────────────────────────────────────────


class _StubApp:
    """Minimal app shape: ``app.state.poly`` and ``app.state.warm_jumps``."""

    class _State:
        poly: Any = None
        warm_jumps: dict | None = None

    def __init__(
        self,
        *,
        warm: dict | None = None,
        poly: Any = object(),  # sentinel that's *not* None
    ) -> None:
        self.state = self._State()
        self.state.poly = poly
        self.state.warm_jumps = warm


@pytest.mark.asyncio
async def test_collect_recent_backtests_returns_expected_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    warm = {
        "computed_at": 0,
        "slugs": {f"slug-{i}": 0.5 + i * 0.1 for i in range(25)},
    }
    app = _StubApp(warm=warm)
    fake_results = {f"slug-{i}": _disagrees(f"slug-{i}") for i in range(25)}

    async def _stub(app_, slug, *, semaphore):
        return fake_results.get(slug)

    monkeypatch.setattr("pfm.alpha_hub.sentiment_alert._one_backtest", _stub)

    rows = await collect_recent_backtests(app, top_n=10)
    assert len(rows) == 10
    # Every row carries the disagrees verdict, so the detector should fire.
    alert = check_sentiment_regression(rows)
    assert alert is not None
    assert alert.disagrees_pct == 100.0


@pytest.mark.asyncio
async def test_collect_recent_backtests_drops_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    warm = {"slugs": {f"slug-{i}": 1.0 for i in range(6)}}
    app = _StubApp(warm=warm)

    async def _stub(app_, slug, *, semaphore):
        # Even slugs succeed, odd slugs return None.
        idx = int(slug.split("-")[1])
        return _disagrees(slug) if idx % 2 == 0 else None

    monkeypatch.setattr("pfm.alpha_hub.sentiment_alert._one_backtest", _stub)
    rows = await collect_recent_backtests(app, top_n=6)
    # 3 of 6 succeeded; failures silently dropped.
    assert len(rows) == 3


@pytest.mark.asyncio
async def test_collect_handles_missing_poly_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # With poly=None each _one_backtest call short-circuits to None;
    # the collector returns [].  We assert via the real (unpatched)
    # _one_backtest by leaving it alone.
    warm = {"slugs": {f"slug-{i}": 1.0 for i in range(3)}}
    app = _StubApp(warm=warm, poly=None)
    rows = await collect_recent_backtests(app, top_n=3)
    assert rows == []


@pytest.mark.asyncio
async def test_collect_falls_back_to_curated_when_warm_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # No warm_jumps → curated list. We patch _one_backtest to return
    # a marker dict per slug so we can confirm the curated path fired.
    app = _StubApp(warm=None)

    seen_slugs: list[str] = []

    async def _stub(app_, slug, *, semaphore):
        seen_slugs.append(slug)
        return {"slug": slug, "interpretation": "INCONCLUSIVE"}

    monkeypatch.setattr("pfm.alpha_hub.sentiment_alert._one_backtest", _stub)
    rows = await collect_recent_backtests(app, top_n=3)
    assert len(rows) == 3
    # All three slugs should come from CURATED_TOP_SLUGS (non-empty list).
    from pfm.terminal.jumps_prewarm import CURATED_TOP_SLUGS

    assert set(seen_slugs).issubset(set(CURATED_TOP_SLUGS))


# ─────────────────────────────────────────────────────────────────────────────
# Digest integration seam
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_build_digest_rows_emits_one_row_on_trigger(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    warm = {"slugs": {f"slug-{i}": 1.0 for i in range(8)}}
    app = _StubApp(warm=warm)

    async def _stub(app_, slug, *, semaphore):
        return _disagrees(slug)

    monkeypatch.setattr("pfm.alpha_hub.sentiment_alert._one_backtest", _stub)
    rows = await build_digest_rows(app, top_n=8)
    assert len(rows) == 1
    assert rows[0]["kind"] == "sentiment-regression"


@pytest.mark.asyncio
async def test_build_digest_rows_empty_on_no_trigger(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    warm = {"slugs": {f"slug-{i}": 1.0 for i in range(8)}}
    app = _StubApp(warm=warm)

    async def _stub(app_, slug, *, semaphore):
        return _inconclusive(slug)

    monkeypatch.setattr("pfm.alpha_hub.sentiment_alert._one_backtest", _stub)
    rows = await build_digest_rows(app, top_n=8)
    assert rows == []


@pytest.mark.asyncio
async def test_build_digest_rows_swallows_collector_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _boom(app, top_n=20):
        raise RuntimeError("upstream is on fire")

    monkeypatch.setattr("pfm.alpha_hub.sentiment_alert.collect_recent_backtests", _boom)
    rows = await build_digest_rows(_StubApp(), top_n=5)
    assert rows == []
