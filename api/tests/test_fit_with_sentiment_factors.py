"""End-to-end integration tests for the sentiment factor flow through ``/fit``.

Verifies the 10 curated ``sentiment_*`` factor ids registered in
``app.state.factors`` by the FastAPI lifespan (see ``pfm.main.lifespan``
and ``pfm.sources.sentiment_factor.CURATED_QUERIES``) survive the full
``POST /fit`` pipeline:

    factor-id resolver → fetch dispatcher → design assembly → OLS+HAC fit
    → response serialisation

All external IO is mocked:

  * ``pfm.sources.sentiment_factor.fetch_sentiment_history`` is replaced
    via ``monkeypatch`` so no GDELT / Reddit / HN traffic is generated
    (the dispatcher imports it lazily inside ``pfm.factors.
    fetch_factor_history_dispatch``, so patching the module attribute is
    sufficient — FastAPI ``dependency_overrides`` is not used because
    the source is *not* a FastAPI dependency).
  * yfinance is mocked through the ``app_client`` fixture in
    ``tests/conftest.py`` (it patches ``pfm.main.get_log_returns``).
  * Polymarket history is mocked through the same fixture's
    ``fetch_factor_history`` patch.

The ``app_client`` fixture's temporary ``factors.yml`` carries only
``factor_a`` / ``factor_b``, but the lifespan startup ALSO injects the
ten curated sentiment ids into ``app.state.factors`` from
``CURATED_QUERIES``, so they're resolvable without further setup.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import pfm.main as main_mod
from pfm.sources import sentiment_factor as sf


# These tests assert against the lifespan-injected curated sentiment
# factors, so opt out of the conftest autouse that suppresses them.
@pytest.fixture(autouse=True)
def _restore_curated_sentiment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PFM_SUPPRESS_CURATED_SENTIMENT", raising=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _synthetic_sentiment_frame(
    start: pd.Timestamp = pd.Timestamp("2025-06-01", tz="UTC"),
    end: pd.Timestamp = pd.Timestamp("2025-12-31", tz="UTC"),
    *,
    amplitude: float = 0.4,
    period_days: float = 30.0,
    phase: float = 0.0,
) -> pd.DataFrame:
    """Build a deterministic ``DataFrame[price]`` matching what
    ``fetch_sentiment_history`` returns.

    Series oscillates smoothly in ``[-amplitude, +amplitude]`` so
    ``delta_level`` produces non-zero variation each day (essential for
    the regression to identify a coefficient).
    """
    idx = pd.date_range(start, end, freq="D", tz="UTC")
    t = np.arange(len(idx), dtype=float)
    values = amplitude * np.sin(2.0 * np.pi * t / period_days + phase)
    df = pd.DataFrame({"price": values}, index=idx)
    df.index.name = "date"
    return df


def _install_sentiment_mock(
    monkeypatch: pytest.MonkeyPatch,
    builder: Callable[[str, pd.Timestamp, pd.Timestamp], pd.DataFrame] | None = None,
) -> dict[str, int]:
    """Patch the lazy-imported ``fetch_sentiment_history`` symbol on
    ``pfm.sources.sentiment_factor``.

    The dispatcher in ``pfm.factors`` re-imports this module attribute
    on every call, so monkeypatching the module-level binding is the
    right hook (a stand-alone ``SentimentFactorSource`` injection won't
    reach the dispatcher path).

    Returns a small ``calls`` counter dict the caller can inspect.
    """
    calls: dict[str, int] = {"n": 0}

    def _default_builder(
        query: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> pd.DataFrame:
        # Vary phase by query so different sentiment factors produce
        # distinct columns (the regression is mathematically degenerate
        # if every sentiment column is identical).
        phase = float(abs(hash(query)) % 1000) / 1000.0 * 2.0 * np.pi
        return _synthetic_sentiment_frame(start, end, phase=phase)

    impl = builder or _default_builder

    def _patched(query: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        calls["n"] += 1
        return impl(query, start, end)

    monkeypatch.setattr(sf, "fetch_sentiment_history", _patched)
    return calls


# ---------------------------------------------------------------------------
# 1. Single curated sentiment factor → 200 + the id surfaces in `factors[0]`
# ---------------------------------------------------------------------------


def test_fit_with_single_sentiment_factor_returns_200_and_correct_id(
    app_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A /fit with ``factors=["sentiment_trump"]`` returns 200 and the
    response's lone factor estimate carries id ``sentiment_trump``."""
    _install_sentiment_mock(monkeypatch)

    # Sanity-check: the lifespan must have injected the curated sentiment
    # ids into app.state.factors. If this fails, the rest of the suite is
    # meaningless because the resolver would 400 before reaching the
    # dispatcher.
    assert "sentiment_trump" in main_mod.app.state.factors
    assert main_mod.app.state.factors["sentiment_trump"].source == "sentiment"

    r = app_client.post(
        "/fit",
        json={
            "ticker": "TEST",
            "factors": ["sentiment_trump"],
            "start": "2025-06-15",
            "end": "2025-12-15",
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ticker"] == "TEST"
    assert len(body["factors"]) == 1
    assert body["factors"][0]["id"] == "sentiment_trump"
    # Sanity: the beta is a finite float (sin-curve sentiment is
    # non-degenerate, so OLS must produce *some* coefficient).
    beta = body["factors"][0]["beta"]
    assert isinstance(beta, float)
    # n_obs must be large enough that HAC inference is at least defined.
    assert body["n_obs"] >= 20
    # factor_metadata is populated with the sentiment-source provenance.
    meta = body.get("factor_metadata", {}).get("sentiment_trump")
    assert meta is not None
    assert meta["source"] == "sentiment"
    assert meta["is_probability"] is False


# ---------------------------------------------------------------------------
# 2. Three sentiment factors all appear in the response.factors list
# ---------------------------------------------------------------------------


def test_fit_with_three_sentiment_factors_all_in_coefficients(
    app_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """All three requested sentiment factors must surface as separate
    coefficients in the response (no silent dedup, no swap-with-other-id)."""
    calls = _install_sentiment_mock(monkeypatch)

    requested = ["sentiment_bitcoin", "sentiment_fed", "sentiment_china"]
    r = app_client.post(
        "/fit",
        json={
            "ticker": "TEST",
            "factors": requested,
            "start": "2025-06-15",
            "end": "2025-12-15",
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    returned_ids = [f["id"] for f in body["factors"]]
    assert set(returned_ids) == set(requested), (
        f"missing/extra factors: expected={requested} got={returned_ids}"
    )
    # Three distinct upstream fetches (one per factor — the mock is per-call).
    # The thread-pool dispatch may interleave but exactly N calls must have
    # been issued.
    assert calls["n"] == len(requested)
    # All three carry the sentiment source in their metadata.
    for fid in requested:
        meta = body["factor_metadata"][fid]
        assert meta["source"] == "sentiment", fid


# ---------------------------------------------------------------------------
# 3. Mixing a sentiment factor with a polymarket factor must work
# ---------------------------------------------------------------------------


def test_fit_mixing_sentiment_and_polymarket_factors(
    app_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A request mixing sources must dispatch each one through its own
    fetcher and surface both in the response."""
    sent_calls = _install_sentiment_mock(monkeypatch)

    r = app_client.post(
        "/fit",
        json={
            "ticker": "TEST",
            "factors": ["sentiment_oil", "factor_a"],
            "start": "2025-06-15",
            "end": "2025-12-15",
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    ids = {f["id"] for f in body["factors"]}
    assert ids == {"sentiment_oil", "factor_a"}
    # Each source got exactly one fetch.
    assert sent_calls["n"] == 1
    # factor_metadata distinguishes the sources.
    fm = body["factor_metadata"]
    assert fm["sentiment_oil"]["source"] == "sentiment"
    assert fm["factor_a"]["source"] == "polymarket"
    # is_probability differs by source: sentiment is a level (False),
    # polymarket is a probability (True).
    assert fm["sentiment_oil"]["is_probability"] is False
    assert fm["factor_a"]["is_probability"] is True


# ---------------------------------------------------------------------------
# 4. User-typed ``sentiment:<query>`` prefix → on-the-fly synthesis
# ---------------------------------------------------------------------------


def test_fit_with_sentiment_prefix_synthesises_factor_on_the_fly(
    app_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``sentiment:<query>`` is *not* in the yaml catalog — the resolver
    must synthesise a FactorConfig on the fly, route it through the same
    dispatcher, and echo the user-typed id back in the response."""
    captured: dict[str, str] = {}

    def _capture(query: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        # Record the query string the dispatcher hands to the source —
        # it should match the prefix-stripped, whitespace-trimmed query.
        captured["query"] = query
        return _synthetic_sentiment_frame(start, end, phase=1.23)

    _install_sentiment_mock(monkeypatch, builder=_capture)

    # The id is NOT in the curated catalog.
    assert "sentiment:custom-query" not in main_mod.app.state.factors

    r = app_client.post(
        "/fit",
        json={
            "ticker": "TEST",
            "factors": ["sentiment:custom-query"],
            "start": "2025-06-15",
            "end": "2025-12-15",
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    # The user-typed id round-trips verbatim.
    assert len(body["factors"]) == 1
    assert body["factors"][0]["id"] == "sentiment:custom-query"
    # The dispatcher was asked for the prefix-stripped query.
    assert captured["query"] == "custom-query"
    # And the source provenance is still 'sentiment' even though the id
    # wasn't in the yaml catalog.
    assert body["factor_metadata"]["sentiment:custom-query"]["source"] == "sentiment"


def test_fit_with_empty_sentiment_prefix_returns_400(
    app_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``sentiment:`` with no query body must 400 with a helpful detail —
    this guards against accidentally creating a no-op factor."""
    _install_sentiment_mock(monkeypatch)

    r = app_client.post(
        "/fit",
        json={
            "ticker": "TEST",
            "factors": ["sentiment:"],
            "start": "2025-06-15",
            "end": "2025-12-15",
        },
    )
    assert r.status_code == 400, r.text
    detail = r.json()["detail"]
    # Detail is a dict with the bad query echoed back; the human message
    # is in detail["error"].
    assert isinstance(detail, dict)
    assert "sentiment" in detail.get("error", "").lower()


# ---------------------------------------------------------------------------
# 5. Empty sentiment series must degrade gracefully (no 500 uncaught crash)
# ---------------------------------------------------------------------------


def test_fit_empty_sentiment_series_degrades_gracefully(
    app_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the upstream returns no rows (rate-limited GDELT, blank query,
    weekend-only window…), the API must NOT 500 with an uncaught
    exception. The current implementation surfaces a structured 502 with
    a JSON detail; an alternative future implementation might return 200
    with a NaN coefficient. Either is acceptable — the only forbidden
    behaviour is a 500-with-traceback.
    """

    def _empty(query: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        # Return the empty-shape frame the production code produces for a
        # no-data upstream (matches ``fetch_sentiment_history``'s
        # ``return pd.DataFrame(columns=["price"])`` path).
        return pd.DataFrame(columns=["price"])

    _install_sentiment_mock(monkeypatch, builder=_empty)

    r = app_client.post(
        "/fit",
        json={
            "ticker": "TEST",
            "factors": ["sentiment_recession"],
            "start": "2025-06-15",
            "end": "2025-12-15",
        },
    )
    # Critical invariant: no uncaught 500.
    assert r.status_code != 500, f"uncaught crash: {r.text}"
    # The response is structured JSON either way (200 with NaN beta or
    # 4xx/5xx with a detail field).
    body = r.json()
    if r.status_code == 200:
        # Future-proofing branch: a 200 must carry the factor id with a
        # NaN (or otherwise non-finite) coefficient — never a silently
        # zeroed-out beta that callers would mistake for a real signal.
        assert any(
            f["id"] == "sentiment_recession" and (np.isnan(f["beta"]) or not np.isfinite(f["beta"]))
            for f in body.get("factors", [])
        )
    else:
        # Current implementation: 502 with a structured detail message
        # telling the caller which factor had no data.
        assert r.status_code in (400, 422, 502), (
            f"unexpected status for empty-sentiment: {r.status_code}: {r.text}"
        )
        # The detail must reference the failing factor so callers can act.
        detail = body.get("detail")
        assert detail, "expected a non-empty detail on graceful failure"
        detail_text = detail if isinstance(detail, str) else str(detail)
        assert "sentiment" in detail_text.lower() or "sentiment_recession" in detail_text
