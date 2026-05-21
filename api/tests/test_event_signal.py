"""Tests for ``pfm.vol.event_signal`` — B3 live signal composition.

All HTTP is mocked. The Polymarket path is exercised via :mod:`respx`
because :func:`pfm.vol.event_signal._polymarket_midpoint` reaches
through the client's underlying ``httpx.Client`` (it cannot use the
``MarketMetadata`` dataclass because that does not expose live
``bestBid``/``bestAsk``).

The Kalshi path is exercised via a small stub class that mimics the
relevant slice of :class:`pfm.sources.kalshi.KalshiClient`.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import httpx
import pandas as pd
import pytest
import respx

from pfm.sources.polymarket import PolymarketClient
from pfm.vol.event_calendar import (
    get_event,
)
from pfm.vol.event_signal import (
    EventSignal,
    compute_all_upcoming_signals,
    compute_event_signal,
    fetch_outcome_probabilities,
)

# --------------------------------------------------------------------------- #
# Helpers / stubs                                                              #
# --------------------------------------------------------------------------- #

GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"

# A representative FOMC event from the curated calendar. We pick the
# all-Polymarket June 2026 entry so the basic path needs no Kalshi.
FOMC_JUN = "fomc-2026-06"
# Mixed-venue: December 2026 FOMC mixes Polymarket and Kalshi slugs.
FOMC_DEC = "fomc-2026-12"


def _make_polymarket_client() -> PolymarketClient:
    """Real :class:`PolymarketClient` over a real httpx.Client so respx can mock."""
    return PolymarketClient(gamma_url=GAMMA, clob_url=CLOB, client=httpx.Client())


def _gamma_market_payload(slug: str, midpoint: float) -> list[dict[str, Any]]:
    """Build a Gamma /markets response with the given midpoint baked in."""
    bb = max(0.0, midpoint - 0.005)
    ba = min(1.0, midpoint + 0.005)
    return [
        {
            "id": f"id-{slug}",
            "slug": slug,
            "question": f"Q: {slug}?",
            "clobTokenIds": json.dumps(["111", "222"]),
            "startDate": "2025-12-01T00:00:00Z",
            "endDate": "2026-12-31T00:00:00Z",
            "closed": False,
            "active": True,
            "bestBid": bb,
            "bestAsk": ba,
            "lastTradePrice": midpoint,
        }
    ]


def _mock_polymarket_slug(slug: str, midpoint: float) -> None:
    """Register a respx mock for one Polymarket slug."""
    respx.get(f"{GAMMA}/markets", params={"slug": slug}).mock(
        return_value=httpx.Response(200, json=_gamma_market_payload(slug, midpoint))
    )


class _StubKalshiClient:
    """Minimal stand-in for :class:`KalshiClient`.

    Returns a one-row candlestick DataFrame per ticker. Tickers not in
    the supplied map raise — simulating "market not found".
    """

    def __init__(self, midpoints: dict[str, float]) -> None:
        self.midpoints = midpoints

    def get_candlesticks(
        self,
        ticker: str,
        start_ts: int,
        end_ts: int,
        period_interval: int = 1440,
        series_ticker: str | None = None,
    ) -> pd.DataFrame:
        if ticker not in self.midpoints:
            raise RuntimeError(f"stub-kalshi: unknown ticker {ticker!r}")
        mid = self.midpoints[ticker]
        return pd.DataFrame(
            {
                "price": [mid],
                "volume": [1000.0],
                "open_interest": [500.0],
                "yes_bid": [max(0.0, mid - 0.005)],
                "yes_ask": [min(1.0, mid + 0.005)],
                "spread": [0.01],
            },
            index=pd.DatetimeIndex([pd.Timestamp.now(tz="UTC").normalize()], name="date"),
        )


# --------------------------------------------------------------------------- #
# Test 1 — Polymarket-only fetch happy path                                    #
# --------------------------------------------------------------------------- #


@respx.mock
def test_fetch_outcome_probabilities_polymarket_only() -> None:
    event = get_event(FOMC_JUN)
    assert event is not None
    assert all(s.venue == "polymarket" for s in event.outcome_slugs)

    # Roughly uniform across 5 outcomes (0.20 each).
    target_probs = [0.18, 0.21, 0.22, 0.20, 0.19]
    for slug_def, p in zip(event.outcome_slugs, target_probs, strict=True):
        _mock_polymarket_slug(slug_def.slug, p)

    client = _make_polymarket_client()
    outcomes, completeness, warnings = fetch_outcome_probabilities(
        event,
        polymarket_client=client,
        kalshi_client=None,
    )

    assert len(outcomes) == 5
    assert completeness == pytest.approx(1.0)
    assert warnings == []
    # Each midpoint should match the planted probability (± 1e-9).
    for o, p in zip(outcomes, target_probs, strict=True):
        assert o.probability == pytest.approx(p, abs=1e-9)


# --------------------------------------------------------------------------- #
# Test 2 — per-slug fetch failure is dropped, warning emitted                  #
# --------------------------------------------------------------------------- #


@respx.mock
def test_fetch_outcome_probabilities_handles_per_slug_failure() -> None:
    event = get_event(FOMC_JUN)
    assert event is not None
    # Mock 4 slugs to succeed; the 3rd one (index 2, "no_change") returns
    # an empty list both on the default filter and on the closed=true
    # fallback — which surfaces as LookupError → handled as fetch_failed.
    target_probs = [0.20, 0.20, None, 0.20, 0.20]
    failing_slug: str = ""
    for slug_def, p in zip(event.outcome_slugs, target_probs, strict=True):
        if p is None:
            failing_slug = slug_def.slug
            respx.get(f"{GAMMA}/markets", params={"slug": slug_def.slug}).mock(
                return_value=httpx.Response(200, json=[])
            )
            # The closed=true fallback also returns empty so fetch_gamma_market
            # raises LookupError, which our midpoint helper swallows as None.
            respx.get(f"{GAMMA}/markets", params={"slug": slug_def.slug, "closed": "true"}).mock(
                return_value=httpx.Response(200, json=[])
            )
        else:
            _mock_polymarket_slug(slug_def.slug, p)

    client = _make_polymarket_client()
    outcomes, completeness, warnings = fetch_outcome_probabilities(
        event,
        polymarket_client=client,
        kalshi_client=None,
    )

    assert len(outcomes) == 4
    assert completeness == pytest.approx(0.8)
    assert any(w.startswith("fetch_failed:") and failing_slug in w for w in warnings)


# --------------------------------------------------------------------------- #
# Test 3 — Kalshi slugs are skipped when kalshi_client is None                 #
# --------------------------------------------------------------------------- #


@respx.mock
def test_fetch_outcome_probabilities_skips_kalshi_when_client_missing() -> None:
    event = get_event(FOMC_DEC)
    assert event is not None
    pm_slugs = [s for s in event.outcome_slugs if s.venue == "polymarket"]
    kalshi_slugs = [s for s in event.outcome_slugs if s.venue == "kalshi"]
    assert pm_slugs and kalshi_slugs, "FOMC Dec entry should be mixed-venue"

    for s in pm_slugs:
        _mock_polymarket_slug(s.slug, 0.50)

    client = _make_polymarket_client()
    outcomes, completeness, warnings = fetch_outcome_probabilities(
        event,
        polymarket_client=client,
        kalshi_client=None,
    )

    # Only the Polymarket slugs should come back.
    assert len(outcomes) == len(pm_slugs)
    assert completeness == pytest.approx(len(pm_slugs) / len(event.outcome_slugs))
    assert "kalshi_client_missing" in warnings
    # Exactly one ``kalshi_client_missing`` warning, not one per skipped slug.
    assert warnings.count("kalshi_client_missing") == 1


# --------------------------------------------------------------------------- #
# Test 4 — full signal: uniform FOMC → entropy-proxy EM around 0.5%             #
# --------------------------------------------------------------------------- #


@respx.mock
def test_compute_event_signal_fomc_returns_em_forecast() -> None:
    event = get_event(FOMC_JUN)
    assert event is not None
    for slug_def in event.outcome_slugs:
        _mock_polymarket_slug(slug_def.slug, 0.20)  # uniform

    client = _make_polymarket_client()
    sig = compute_event_signal(
        FOMC_JUN,
        polymarket_client=client,
        kalshi_client=None,
    )

    assert isinstance(sig, EventSignal)
    assert sig.event_id == FOMC_JUN
    assert sig.fetch_completeness == pytest.approx(1.0)
    assert sig.forecast.em_method == "entropy_proxy"
    # k_fomc == 0.50; uniform 5-outcome → entropy_normalized = 1.0,
    # tail_pct = 0.4 → dispersion_factor = 0 → em_pct = 0.5.
    assert sig.forecast.em_pct == pytest.approx(0.50, abs=0.05)
    # options_em_pct stays None in v1.
    assert sig.options_em_pct is None
    assert sig.em_gap_pct is None


# --------------------------------------------------------------------------- #
# Test 5 — thin-book degrades to entropy-proxy even with calibration           #
# --------------------------------------------------------------------------- #


@respx.mock
def test_compute_event_signal_thin_book_falls_back_to_entropy_proxy() -> None:
    from pfm.vol.event_vol_engine import EMCalibration

    event = get_event(FOMC_JUN)
    assert event is not None
    # Succeed for only 2 of 5 slugs → completeness = 0.4 < 0.5.
    successes = list(event.outcome_slugs[:2])
    failures = list(event.outcome_slugs[2:])
    for s in successes:
        _mock_polymarket_slug(s.slug, 0.30)
    for s in failures:
        respx.get(f"{GAMMA}/markets", params={"slug": s.slug}).mock(
            return_value=httpx.Response(200, json=[])
        )
        respx.get(f"{GAMMA}/markets", params={"slug": s.slug, "closed": "true"}).mock(
            return_value=httpx.Response(200, json=[])
        )

    calib = EMCalibration(
        event_kind="fomc",
        underlying_ticker="SPY",
        coefficients={
            "entropy_normalized": 0.8,
            "dispersion": 0.1,
            "asymmetric_mass": 0.0,
            "tail_pct": 0.2,
        },
        intercept=0.1,
        r_squared=0.55,
        sample_size=30,
        sigma_residual=0.15,
    )

    client = _make_polymarket_client()
    sig = compute_event_signal(
        FOMC_JUN,
        polymarket_client=client,
        kalshi_client=None,
        calibration=calib,
    )

    assert sig.fetch_completeness < 0.5
    assert sig.forecast.em_method == "entropy_proxy"
    assert "thin_book" in sig.forecast.warnings


# --------------------------------------------------------------------------- #
# Test 6 — unknown event_id raises KeyError                                    #
# --------------------------------------------------------------------------- #


def test_compute_event_signal_unknown_event_raises_keyerror() -> None:
    client = _make_polymarket_client()
    with pytest.raises(KeyError, match="unknown event_id"):
        compute_event_signal(
            "fomc-2099-99",
            polymarket_client=client,
            kalshi_client=None,
        )


# --------------------------------------------------------------------------- #
# Test 7 — compute_all_upcoming_signals: partial failure yields partial list   #
# --------------------------------------------------------------------------- #


@respx.mock
def test_compute_all_upcoming_signals_partial_failure_yields_partial_list() -> None:
    # Pick a date that exposes ≥2 upcoming events in the curated calendar.
    now = datetime(2026, 5, 15, tzinfo=UTC)
    from pfm.vol.event_calendar import list_upcoming

    upcoming = list_upcoming(now, lookahead_days=120)
    assert len(upcoming) >= 2, f"Expected ≥2 upcoming events from {now}"

    # Make ALL slugs from the first event fail (both polymarket and kalshi),
    # and make every other event's polymarket slugs succeed.
    failing_event = upcoming[0]
    successful_events = upcoming[1:]

    for slug_def in failing_event.outcome_slugs:
        if slug_def.venue == "polymarket":
            respx.get(f"{GAMMA}/markets", params={"slug": slug_def.slug}).mock(
                return_value=httpx.Response(200, json=[])
            )
            respx.get(f"{GAMMA}/markets", params={"slug": slug_def.slug, "closed": "true"}).mock(
                return_value=httpx.Response(200, json=[])
            )

    for ev in successful_events:
        for slug_def in ev.outcome_slugs:
            if slug_def.venue == "polymarket":
                _mock_polymarket_slug(slug_def.slug, 0.20)

    # Build a Kalshi stub that satisfies any Kalshi slug a successful event
    # might need. We populate it lazily so every requested ticker resolves.
    kalshi_mids: dict[str, float] = {}
    for ev in upcoming:
        for slug_def in ev.outcome_slugs:
            if slug_def.venue == "kalshi":
                kalshi_mids[slug_def.slug] = 0.30
    kalshi = _StubKalshiClient(kalshi_mids)

    # Make the failing event also fail on its Kalshi slugs by removing them.
    for slug_def in failing_event.outcome_slugs:
        if slug_def.venue == "kalshi":
            kalshi.midpoints.pop(slug_def.slug, None)

    client = _make_polymarket_client()
    signals = compute_all_upcoming_signals(
        now_utc=now,
        lookahead_days=120,
        polymarket_client=client,
        kalshi_client=kalshi,
    )

    returned_ids = {s.event_id for s in signals}
    assert failing_event.event_id not in returned_ids
    # The successful events that have ≥1 polymarket leg should show up.
    expected = {
        ev.event_id
        for ev in successful_events
        if any(s.venue == "polymarket" for s in ev.outcome_slugs)
    } | {
        ev.event_id
        for ev in successful_events
        if any(s.venue == "kalshi" and s.slug in kalshi.midpoints for s in ev.outcome_slugs)
    }
    # At least one successful event should have made it into the result.
    assert returned_ids & expected, f"Expected some of {expected} in {returned_ids}"
    # And the count is strictly less than the upcoming count.
    assert len(signals) < len(upcoming)
