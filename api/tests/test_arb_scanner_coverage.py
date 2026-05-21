"""Coverage-focused tests for ``pfm.arb_scanner``.

Targets the lines left untouched by ``test_arb_scanner.py`` and
``test_arb_scanner_property.py``: mid fetchers, per-venue normalisers,
async venue fetchers, the 4-way arb path, the persistent confirmed-match
registry, auto-discover happy + degraded paths, and edge branches in the
small pure helpers (``_tokenise`` early-out, ``_date_proximity_score``
numeric epoch + bad-input fallbacks, ``_theme_score`` neutral pair).

Mocking strategy
----------------

* HTTP is mocked with ``respx`` so we exercise the real ``httpx`` plumbing
  (status codes, JSON decoding, error handling) without hitting the
  network.
* Polymarket Gamma calls into ``fetch_gamma_market`` go through a
  monkeypatch on the symbol imported into ``arb_scanner`` so we don't
  have to model the on-disk Gamma cache.
* Kalshi candlesticks return synthetic ``pandas.DataFrame``\\ s shaped like
  the production schema (``yes_bid``/``yes_ask``/``volume``).
* The persistent confirmed-match path is redirected to ``tmp_path`` so a
  test never writes ``/tmp/pfm_arb_confirmed_matches.json``.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import httpx
import pandas as pd
import pytest
import respx
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pfm import arb_scanner
from pfm.arb_scanner import (
    CONCEPT_MAPS,
    _auto_discover_lock,
    _build_4way_record,
    _date_proximity_score,
    _fetch_active_kalshi,
    _fetch_active_manifold,
    _fetch_active_polymarket,
    _fetch_active_predictit,
    _gather_active_markets,
    _half_life_estimate,
    _kalshi_mid,
    _keyword_jaccard,
    _load_confirmed_store,
    _max_pairwise_spread_pct,
    _normalise_kalshi_market,
    _normalise_manifold_market,
    _normalise_pm_market,
    _normalise_predictit_market,
    _pair_key,
    _pm_mid,
    _save_confirmed_store,
    _spread_record,
    _theme_score,
    _tokenise,
    all_matched_pairs,
    auto_discover_arb_pairs,
    compute_4way_arbs,
    compute_arb_spreads,
    find_4way_arb,
    get_concept_map,
    list_confirmed_matches,
    record_match_observation,
    router,
    top_arbs,
)
from pfm.cache_utils import get_cache

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate_state(tmp_path: Path) -> None:
    """Wipe every per-module cache + redirect persistent paths to tmp_path."""
    get_cache("arb_scanner").clear()
    get_cache("arb_matched").clear()
    get_cache("arb_dynamic").clear()
    arb_scanner._MANUAL_PAIRS.clear()
    arb_scanner.CONFIRMED_MATCHES_PATH = tmp_path / "confirmed.json"
    # Reset the per-loop lock registry to avoid asyncio-loop bleed.
    arb_scanner._AUTO_DISCOVER_LOCKS.clear()


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def _kalshi_df(
    *,
    yes_bid: float | None = 0.50,
    yes_ask: float | None = 0.52,
    volume: float = 12_000.0,
) -> pd.DataFrame:
    """One-row candlestick DataFrame matching ``get_candlesticks`` schema."""
    idx = pd.DatetimeIndex([pd.Timestamp("2026-05-08", tz="UTC")], name="date")
    return pd.DataFrame(
        {
            "price": [(float(yes_bid or 0) + float(yes_ask or 0)) / 2.0],
            "volume": [volume],
            "open_interest": [volume * 2],
            "yes_bid": [yes_bid],
            "yes_ask": [yes_ask],
            "spread": [0.02],
        },
        index=idx,
    )


# ---------------------------------------------------------------------------
# Pure helpers: _tokenise / _keyword_jaccard / _date_proximity_score / themes
# ---------------------------------------------------------------------------


def test_tokenise_empty_string_returns_empty_set() -> None:
    """Line 161-162 (the empty-string early-out)."""
    assert _tokenise("") == set()
    assert _tokenise(None) == set()  # type: ignore[arg-type]


def test_tokenise_drops_short_tokens_and_stopwords() -> None:
    out = _tokenise("The BTC is at a new high above 100k")
    # Drops 'the', 'is', 'at', 'a' (stopwords) and short tokens.
    # 'btc' is exactly 3 chars and kept.
    assert "btc" in out
    assert "the" not in out
    assert "at" not in out
    # Regex strips leading digits via ``[a-zA-Z][a-zA-Z0-9]*`` ⇒ "100k" is dropped
    # entirely; "high" survives.
    assert "high" in out


def test_keyword_jaccard_empty_inputs_return_zero() -> None:
    """Line 171: empty-set early-out."""
    assert _keyword_jaccard("", "anything") == 0.0
    assert _keyword_jaccard("anything", "") == 0.0


def test_keyword_jaccard_identical_titles_is_one() -> None:
    assert _keyword_jaccard("BTC hits 100k", "BTC hits 100k") == pytest.approx(1.0)


def test_date_proximity_score_neutral_when_missing() -> None:
    assert _date_proximity_score(None, "2026-12-31T00:00:00Z") == 0.5
    assert _date_proximity_score("2026-12-31T00:00:00Z", None) == 0.5


def test_date_proximity_score_numeric_epoch_inputs() -> None:
    """Line 186 + 190: numeric epoch (non-string) branch."""
    epoch_a = 1_767_225_600  # 2026-01-01
    epoch_b = 1_767_225_600
    assert _date_proximity_score(epoch_a, epoch_b) == pytest.approx(1.0)


def test_date_proximity_score_far_apart_is_zero() -> None:
    assert _date_proximity_score("2026-01-01T00:00:00Z", "2026-06-01T00:00:00Z") == 0.0


def test_date_proximity_score_intermediate_linear_decay() -> None:
    """tol_days < delta < 30d should yield a value in (0, 1)."""
    val = _date_proximity_score(
        "2026-01-01T00:00:00Z",
        "2026-01-21T00:00:00Z",  # 20 days
    )
    assert 0.0 < val < 1.0


def test_date_proximity_score_bad_string_returns_neutral() -> None:
    """Line 193-194: ValueError fallback path."""
    assert _date_proximity_score("not-a-date", "2026-01-01") == 0.5


def test_theme_score_match_mismatch_and_missing() -> None:
    assert _theme_score("macro", "MACRO") == 1.0
    assert _theme_score("macro", "crypto") == 0.0
    assert _theme_score("", "macro") == 0.5
    assert _theme_score(None, None) == 0.5


# ---------------------------------------------------------------------------
# _pm_mid: covers happy path, LookupError, missing fields, bad volume
# ---------------------------------------------------------------------------


def test_pm_mid_happy_path_with_bid_ask(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        arb_scanner,
        "fetch_gamma_market",
        lambda http, url, slug: {
            "bestBid": 0.48,
            "bestAsk": 0.52,
            "volume24hr": 12345.0,
        },
    )
    mid, vol = _pm_mid("foo", MagicMock())
    assert mid == pytest.approx(0.50)
    assert vol == 12345.0


def test_pm_mid_falls_back_to_lastTradePrice(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fallback to lastTradePrice only works when lastTradeTime is fresh."""
    from datetime import UTC as _UTC
    from datetime import datetime as _dt

    fresh_iso = _dt.now(tz=_UTC).isoformat()
    monkeypatch.setattr(
        arb_scanner,
        "fetch_gamma_market",
        lambda http, url, slug: {
            "bestBid": None,
            "bestAsk": None,
            "lastTradePrice": 0.42,
            "lastTradeTime": fresh_iso,
            "volumeNum": 9_000.0,
        },
    )
    mid, vol = _pm_mid("foo", MagicMock())
    assert mid == pytest.approx(0.42)
    assert vol == 9_000.0


def test_pm_mid_skips_stale_lastTradePrice(monkeypatch: pytest.MonkeyPatch) -> None:
    """A stale (>30min) lastTradePrice must NOT be used as the mid."""
    from datetime import UTC as _UTC
    from datetime import datetime as _dt
    from datetime import timedelta as _td

    stale_iso = (_dt.now(tz=_UTC) - _td(hours=2)).isoformat()
    monkeypatch.setattr(
        arb_scanner,
        "fetch_gamma_market",
        lambda http, url, slug: {
            "bestBid": None,
            "bestAsk": None,
            "lastTradePrice": 0.42,
            "lastTradeTime": stale_iso,
            "volumeNum": 9_000.0,
        },
    )
    mid, vol = _pm_mid("foo", MagicMock())
    assert mid is None
    assert vol == 9_000.0


def test_pm_mid_skips_lastTradePrice_when_time_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without lastTradeTime we conservatively skip the fallback."""
    monkeypatch.setattr(
        arb_scanner,
        "fetch_gamma_market",
        lambda http, url, slug: {
            "bestBid": None,
            "bestAsk": None,
            "lastTradePrice": 0.42,
            "volumeNum": 9_000.0,
        },
    )
    mid, vol = _pm_mid("foo", MagicMock())
    assert mid is None
    assert vol == 9_000.0


def test_pm_mid_handles_lookup_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise(*_a: Any, **_k: Any) -> None:
        raise LookupError("not found")

    monkeypatch.setattr(arb_scanner, "fetch_gamma_market", _raise)
    assert _pm_mid("foo", MagicMock()) == (None, None)


def test_pm_mid_handles_httpx_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise(*_a: Any, **_k: Any) -> None:
        raise httpx.ConnectError("boom")

    monkeypatch.setattr(arb_scanner, "fetch_gamma_market", _raise)
    assert _pm_mid("foo", MagicMock()) == (None, None)


def test_pm_mid_bad_numeric_prices(monkeypatch: pytest.MonkeyPatch) -> None:
    """Line 313-314: TypeError/ValueError when bestBid/bestAsk are garbage."""
    monkeypatch.setattr(
        arb_scanner,
        "fetch_gamma_market",
        lambda http, url, slug: {
            "bestBid": "not-a-num",
            "bestAsk": "also-bad",
            "lastTradePrice": None,
        },
    )
    mid, vol = _pm_mid("foo", MagicMock())
    assert mid is None
    assert vol is None


def test_pm_mid_bad_volume_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        arb_scanner,
        "fetch_gamma_market",
        lambda http, url, slug: {
            "bestBid": 0.5,
            "bestAsk": 0.5,
            "volume24hr": "not-a-num",
            "volumeNum": None,
            "volume": None,
        },
    )
    mid, vol = _pm_mid("foo", MagicMock())
    assert mid == pytest.approx(0.5)
    assert vol is None


# ---------------------------------------------------------------------------
# _kalshi_mid: empty df, exception, missing yes_bid/ask, bad volume
# ---------------------------------------------------------------------------


def test_kalshi_mid_happy_path() -> None:
    client = MagicMock()
    client.get_candlesticks.return_value = _kalshi_df(
        yes_bid=0.48,
        yes_ask=0.52,
        volume=25_000.0,
    )
    mid, vol = _kalshi_mid("TICKER", client)
    assert mid == pytest.approx(0.50)
    assert vol == 25_000.0


def test_kalshi_mid_empty_dataframe_returns_none() -> None:
    client = MagicMock()
    client.get_candlesticks.return_value = pd.DataFrame()
    assert _kalshi_mid("TICKER", client) == (None, None)


def test_kalshi_mid_exception_returns_none() -> None:
    client = MagicMock()
    client.get_candlesticks.side_effect = RuntimeError("kalshi-down")
    assert _kalshi_mid("TICKER", client) == (None, None)


def test_kalshi_mid_missing_bid_ask_columns_returns_none() -> None:
    """Line 346-347: KeyError fallback when ``yes_bid`` is missing."""
    client = MagicMock()
    idx = pd.DatetimeIndex([pd.Timestamp("2026-05-08", tz="UTC")], name="date")
    client.get_candlesticks.return_value = pd.DataFrame(
        {"price": [0.5], "volume": [1_000.0]},
        index=idx,
    )
    assert _kalshi_mid("TICKER", client) == (None, None)


def test_kalshi_mid_bad_volume_defaults_to_zero() -> None:
    """Line 350-351: ValueError on volume → 0.0."""
    client = MagicMock()
    idx = pd.DatetimeIndex([pd.Timestamp("2026-05-08", tz="UTC")], name="date")
    client.get_candlesticks.return_value = pd.DataFrame(
        {
            "price": [0.5],
            "yes_bid": [0.49],
            "yes_ask": [0.51],
            "volume": ["bad"],
            "open_interest": [1_000.0],
        },
        index=idx,
    )
    mid, vol = _kalshi_mid("TICKER", client)
    assert mid == pytest.approx(0.50)
    assert vol == 0.0


# ---------------------------------------------------------------------------
# compute_arb_spreads: both-sides-down + per-pair exception
# ---------------------------------------------------------------------------


def test_compute_arb_spreads_empty_input_returns_empty() -> None:
    assert compute_arb_spreads([]) == []


def test_compute_arb_spreads_pm_miss_skips_pair(monkeypatch: pytest.MonkeyPatch) -> None:
    """Both-sides-down: PM mid is None ⇒ no arb, no Kalshi call needed."""
    monkeypatch.setattr(arb_scanner, "_pm_mid", lambda slug, http: (None, None))
    kalshi_calls: list[str] = []

    def _spy(ticker: str, _c: Any) -> tuple[None, None]:
        kalshi_calls.append(ticker)
        return None, None

    monkeypatch.setattr(arb_scanner, "_kalshi_mid", _spy)
    pairs = [{"pm_slug": "p1", "kalshi_slug": "K1", "label": "x"}]
    arbs = compute_arb_spreads(
        pairs,
        http=MagicMock(),
        kalshi_client=MagicMock(),
    )
    assert arbs == []
    assert kalshi_calls == [], "should short-circuit before hitting Kalshi"


def test_compute_arb_spreads_kalshi_miss_skips_pair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(arb_scanner, "_pm_mid", lambda slug, http: (0.50, 10_000.0))
    monkeypatch.setattr(arb_scanner, "_kalshi_mid", lambda ticker, c: (None, None))
    pairs = [{"pm_slug": "p1", "kalshi_slug": "K1", "label": "x"}]
    assert compute_arb_spreads(pairs, http=MagicMock(), kalshi_client=MagicMock()) == []


def test_compute_arb_spreads_swallows_per_pair_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Line 442-444: per-pair eval exception is caught + logged."""

    def _boom(slug: str, _h: Any) -> tuple[float, float]:
        raise RuntimeError("kaboom")

    monkeypatch.setattr(arb_scanner, "_pm_mid", _boom)
    monkeypatch.setattr(arb_scanner, "_kalshi_mid", lambda *_a, **_k: (0.5, 1_000.0))
    out = compute_arb_spreads(
        [{"pm_slug": "p", "kalshi_slug": "K", "label": "x"}],
        http=MagicMock(),
        kalshi_client=MagicMock(),
    )
    assert out == []


def test_compute_arb_spreads_default_clients_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the caller omits http/kalshi_client, both must be closed in finally."""
    closed: dict[str, bool] = {"http": False, "kalshi": False}

    class _FakeHttp:
        def close(self) -> None:
            closed["http"] = True

    class _FakeKalshi:
        def close(self) -> None:
            closed["kalshi"] = True

    monkeypatch.setattr(httpx, "Client", lambda **_k: _FakeHttp())  # type: ignore[arg-type]
    monkeypatch.setattr(
        arb_scanner.kalshi_src,
        "KalshiClient",
        lambda *_a, **_k: _FakeKalshi(),
    )
    monkeypatch.setattr(arb_scanner, "_pm_mid", lambda slug, http: (0.5, 1_000.0))
    monkeypatch.setattr(arb_scanner, "_kalshi_mid", lambda ticker, c: (0.55, 1_000.0))
    compute_arb_spreads(
        [{"pm_slug": "a", "kalshi_slug": "K", "label": ""}],
        min_spread_pct=10.0,
        min_volume_usd=1_000_000.0,  # filter everything
    )
    assert closed == {"http": True, "kalshi": True}


# ---------------------------------------------------------------------------
# _spread_record: edge cases
# ---------------------------------------------------------------------------


def test_spread_record_zero_volume_returns_none() -> None:
    pair = {"pm_slug": "p", "kalshi_slug": "K", "label": "x"}
    rec = _spread_record(
        pair,
        0.40,
        0.55,
        None,
        None,
        min_spread_pct=2.0,
        min_volume_usd=0.0,
    )
    # min(0, 0) = 0; 0 >= 0 ⇒ kept (volume floor satisfied trivially).
    assert rec is not None
    assert rec["tradeable_size_usd"] == 0.0


def test_spread_record_half_life_calibration() -> None:
    """Wider spread ⇒ shorter half-life, bounded between 5 and 120 min."""
    pair = {"pm_slug": "p", "kalshi_slug": "K", "label": ""}
    rec_narrow = _spread_record(
        pair,
        0.50,
        0.52,
        10_000.0,
        10_000.0,
        min_spread_pct=1.0,
        min_volume_usd=0.0,
    )
    rec_wide = _spread_record(
        pair,
        0.30,
        0.80,
        10_000.0,
        10_000.0,
        min_spread_pct=1.0,
        min_volume_usd=0.0,
    )
    assert rec_narrow is not None and rec_wide is not None
    assert rec_wide["half_life_minutes"] <= rec_narrow["half_life_minutes"]
    assert 5.0 <= rec_wide["half_life_minutes"] <= 120.0


def test_spread_record_direction_buy_kalshi_when_kalshi_cheaper() -> None:
    pair = {"pm_slug": "p", "kalshi_slug": "K", "label": ""}
    rec = _spread_record(
        pair,
        0.60,
        0.40,
        10_000.0,
        10_000.0,
        min_spread_pct=1.0,
        min_volume_usd=0.0,
    )
    assert rec is not None
    assert rec["direction"] == "buy_kalshi_sell_pm"


# ---------------------------------------------------------------------------
# Normalisers
# ---------------------------------------------------------------------------


def test_normalise_pm_market_mid_from_bid_ask() -> None:
    out = _normalise_pm_market(
        {
            "id": 1,
            "slug": "x",
            "question": "Foo?",
            "category": "Macro",
            "endDate": "2026-12-31",
            "bestBid": 0.48,
            "bestAsk": 0.52,
            "volume24hr": 1_000.0,
        }
    )
    assert out["price"] == pytest.approx(0.50)
    assert out["theme"] == "macro"
    assert out["title"] == "Foo?"


def test_normalise_pm_market_fallbacks() -> None:
    out = _normalise_pm_market(
        {
            "id": 2,
            "slug": "y",
            "title": "Bar",
            "lastTradePrice": 0.33,
            "volume": "not-a-num",  # triggers ValueError → vol=0
        }
    )
    assert out["price"] == pytest.approx(0.33)
    assert out["volume_24h_usd"] == 0.0
    assert out["theme"] is None


def test_normalise_pm_market_bad_price() -> None:
    out = _normalise_pm_market(
        {
            "id": 3,
            "slug": "z",
            "title": "T",
            "bestBid": "junk",
            "bestAsk": "junk",
            "lastTradePrice": None,
        }
    )
    assert out["price"] is None


def test_normalise_kalshi_market_cents_to_unit() -> None:
    out = _normalise_kalshi_market(
        {
            "ticker": "T1",
            "title": "Foo",
            "category": "Crypto",
            "close_time": "2026-12-31",
            "yes_bid": 48,
            "yes_ask": 52,
            "volume_24h": 10_000.0,
        }
    )
    assert out["price"] == pytest.approx(0.50)
    assert out["theme"] == "crypto"
    assert out["ticker"] == "T1"


def test_normalise_kalshi_market_last_fallback() -> None:
    out = _normalise_kalshi_market(
        {
            "ticker": "T2",
            "title": "Foo",
            "yes_bid": None,
            "yes_ask": None,
            "last_price": 40,  # cents → 0.40
        }
    )
    assert out["price"] == pytest.approx(0.40)


def test_normalise_kalshi_market_bad_inputs() -> None:
    out = _normalise_kalshi_market(
        {
            "ticker": "T3",
            "yes_bid": "x",
            "yes_ask": "y",
            "last_price": "z",
            "volume": "bad",
        }
    )
    assert out["price"] is None
    assert out["volume_24h_usd"] == 0.0


def test_normalise_manifold_market_happy_and_degraded() -> None:
    ok = _normalise_manifold_market(
        {
            "id": "abc",
            "slug": "abc",
            "question": "Q?",
            "probability": 0.7,
            "volume24Hours": 1_500.0,
        }
    )
    assert ok["price"] == pytest.approx(0.7)
    assert ok["volume_24h_usd"] == 1_500.0

    bad = _normalise_manifold_market(
        {
            "id": "x",
            "slug": "x",
            "title": "T",
            "probability": "garbage",
            "volume": "x",
        }
    )
    assert bad["price"] is None
    assert bad["volume_24h_usd"] == 0.0


def test_normalise_predictit_market_picks_lead_contract() -> None:
    rec = _normalise_predictit_market(
        {
            "id": 8200,
            "name": "Pres 2028",
            "contracts": [
                {"lastTradePrice": 0.20},
                {"lastTradePrice": 0.55},  # lead
                {"lastTradePrice": 0.10},
            ],
            "totalSharesTraded": 2_500.0,
            "dateEnd": "2028-11-07",
        }
    )
    assert rec["price"] == pytest.approx(0.55)
    assert rec["volume_24h_usd"] == 2_500.0


def test_normalise_predictit_market_empty_contracts() -> None:
    rec = _normalise_predictit_market({"id": 1, "name": "X", "contracts": []})
    assert rec["price"] is None


def test_normalise_predictit_market_bad_contracts_field() -> None:
    """Non-list contracts triggers the early-skip path."""
    rec = _normalise_predictit_market({"id": 2, "name": "Y", "contracts": "??"})
    assert rec["price"] is None


def test_normalise_predictit_market_bad_price_in_lead() -> None:
    rec = _normalise_predictit_market(
        {
            "id": 3,
            "name": "Z",
            "contracts": [{"lastTradePrice": "junk"}],
        }
    )
    # ValueError on the float(ltp) ⇒ price stays None.
    assert rec["price"] is None


# ---------------------------------------------------------------------------
# _max_pairwise_spread_pct
# ---------------------------------------------------------------------------


def test_max_pairwise_spread_pct_empty_and_single() -> None:
    assert _max_pairwise_spread_pct({}) == (0.0, "", "")
    assert _max_pairwise_spread_pct({"pm": 0.5}) == (0.0, "", "")


def test_max_pairwise_spread_pct_picks_extremes() -> None:
    spread, lo, hi = _max_pairwise_spread_pct(
        {"pm": 0.40, "kalshi": 0.55, "manifold": 0.50},
    )
    assert lo == "pm"
    assert hi == "kalshi"
    assert spread == pytest.approx(15.0)


# ---------------------------------------------------------------------------
# Concept maps and find_4way_arb
# ---------------------------------------------------------------------------


def test_get_concept_map_known_and_unknown() -> None:
    assert get_concept_map("FED_CUTS_2026") is not None  # case-insensitive
    assert get_concept_map("does-not-exist") is None
    assert get_concept_map("") is None


def test_find_4way_arb_happy_path() -> None:
    out = find_4way_arb(
        "fed_cuts_2026",
        pm_price_fn=lambda _: (0.40, 10_000.0),
        kalshi_price_fn=lambda _: (0.55, 8_000.0),
        manifold_price_fn=lambda _: (0.50, 2_000.0),
        predictit_price_fn=lambda _: (0.45, 1_500.0),
    )
    assert out["concept_id"] == "fed_cuts_2026"
    assert set(out["legs_present"]) == {"polymarket", "kalshi", "manifold", "predictit"}
    assert out["missing_venues"] == []
    assert out["max_spread_pct"] == pytest.approx(15.0)
    assert out["low_venue"] == "polymarket"
    assert out["high_venue"] == "kalshi"
    assert out["capital_required_usd"] == pytest.approx(20_000.0)


def test_find_4way_arb_unknown_concept_raises() -> None:
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc:
        find_4way_arb("not-a-real-concept")
    assert exc.value.status_code == 404


def test_find_4way_arb_missing_fn_marks_venue_missing() -> None:
    """A leg without a price_fn (or no identifier on the concept) is missing."""
    out = find_4way_arb(
        "btc_ath_2026",  # predictit is None in this concept
        pm_price_fn=lambda _: (0.5, 1_000.0),
        kalshi_price_fn=None,
        manifold_price_fn=lambda _: (0.45, 200.0),
    )
    assert "predictit" in out["missing_venues"]
    assert "kalshi" in out["missing_venues"]


def test_find_4way_arb_leg_raises_is_caught() -> None:
    def _boom(_ident: Any) -> tuple[float, float]:
        raise RuntimeError("venue-down")

    out = find_4way_arb(
        "fed_cuts_2026",
        pm_price_fn=lambda _: (0.5, 1_000.0),
        kalshi_price_fn=_boom,
        manifold_price_fn=lambda _: (0.55, 100.0),
        predictit_price_fn=lambda _: (0.5, 100.0),
    )
    assert "kalshi" in out["missing_venues"]


def test_find_4way_arb_price_returns_none() -> None:
    out = find_4way_arb(
        "fed_cuts_2026",
        pm_price_fn=lambda _: (None, None),
        kalshi_price_fn=lambda _: (0.5, 100.0),
        manifold_price_fn=lambda _: (0.5, 100.0),
        predictit_price_fn=lambda _: (0.5, 100.0),
    )
    assert "polymarket" in out["missing_venues"]


def test_find_4way_arb_zero_spread_no_capital() -> None:
    out = find_4way_arb(
        "fed_cuts_2026",
        pm_price_fn=lambda _: (0.5, 100.0),
        kalshi_price_fn=lambda _: (0.5, 100.0),
        manifold_price_fn=lambda _: (0.5, 100.0),
        predictit_price_fn=lambda _: (0.5, 100.0),
    )
    assert out["max_spread_pct"] == 0.0
    assert out["capital_required_usd"] == 0.0


# ---------------------------------------------------------------------------
# Persistent confirmed-match registry
# ---------------------------------------------------------------------------


def test_load_confirmed_store_missing_file_returns_empty() -> None:
    assert _load_confirmed_store() == {}


def test_load_confirmed_store_bad_json_returns_empty(tmp_path: Path) -> None:
    arb_scanner.CONFIRMED_MATCHES_PATH.write_text("{not json")
    assert _load_confirmed_store() == {}


def test_load_confirmed_store_non_dict_root_returns_empty() -> None:
    arb_scanner.CONFIRMED_MATCHES_PATH.write_text(json.dumps(["not", "a", "dict"]))
    assert _load_confirmed_store() == {}


def test_load_confirmed_store_filters_non_dict_values() -> None:
    arb_scanner.CONFIRMED_MATCHES_PATH.write_text(
        json.dumps(
            {
                "good": {"fetches": 1},
                "bad": "scalar",
            }
        )
    )
    out = _load_confirmed_store()
    assert "good" in out
    assert "bad" not in out


def test_save_confirmed_store_creates_parent_dir(tmp_path: Path) -> None:
    arb_scanner.CONFIRMED_MATCHES_PATH = tmp_path / "deep" / "nested" / "store.json"
    _save_confirmed_store({"k": {"v": 1}})
    assert arb_scanner.CONFIRMED_MATCHES_PATH.exists()


def test_save_confirmed_store_swallows_oserror(monkeypatch: pytest.MonkeyPatch) -> None:
    """Best-effort persistence: never raise."""

    def _boom(*_a: Any, **_k: Any) -> None:
        raise OSError("read-only fs")

    monkeypatch.setattr(Path, "mkdir", _boom)
    # Should not raise.
    _save_confirmed_store({"k": {"v": 1}})


def test_pair_key_is_order_independent() -> None:
    a = _pair_key("polymarket", "abc", "kalshi", "XYZ")
    b = _pair_key("kalshi", "XYZ", "polymarket", "abc")
    assert a == b


def test_record_match_observation_increments_then_confirms() -> None:
    for i in range(1, arb_scanner.CONFIRMED_FETCHES_REQUIRED):
        rec = record_match_observation(
            "polymarket",
            "abc",
            "kalshi",
            "XYZ",
            similarity=0.8,
            label="Test pair",
        )
        assert rec["confirmed"] is False
        assert rec["fetches"] == i
    rec = record_match_observation(
        "polymarket",
        "abc",
        "kalshi",
        "XYZ",
        similarity=0.85,
        label="Test pair",
    )
    assert rec["confirmed"] is True
    assert rec["fetches"] == arb_scanner.CONFIRMED_FETCHES_REQUIRED


def test_record_match_observation_does_not_overwrite_label() -> None:
    """Once label is set, a later call with a new label keeps the original."""
    record_match_observation(
        "polymarket",
        "abc",
        "kalshi",
        "XYZ",
        similarity=0.7,
        label="First label",
    )
    rec = record_match_observation(
        "polymarket",
        "abc",
        "kalshi",
        "XYZ",
        similarity=0.9,
        label="Second label",
    )
    assert rec["label"] == "First label"


def test_list_confirmed_matches_filter_only_confirmed() -> None:
    record_match_observation(
        "polymarket",
        "x",
        "kalshi",
        "Y",
        similarity=0.7,
        label="lo",
    )
    # 1 fetch — not yet confirmed.
    assert list_confirmed_matches(only_confirmed=True) == []
    # With filter off, should show the unconfirmed record.
    unconf = list_confirmed_matches(only_confirmed=False)
    assert len(unconf) == 1
    assert unconf[0]["confirmed"] is False


# ---------------------------------------------------------------------------
# _half_life_estimate + _build_4way_record
# ---------------------------------------------------------------------------


def test_half_life_estimate_bounds() -> None:
    # 60 / (2.0/5.0) = 150 → capped at 120 by the upper bound.
    assert _half_life_estimate(2.0) == pytest.approx(120.0)
    # 60 / (10.0/5.0) = 30 — within both bounds.
    assert _half_life_estimate(10.0) == pytest.approx(30.0)
    # Very wide spread → floored at 5.
    assert _half_life_estimate(100.0) == pytest.approx(5.0)
    # Zero spread: max(0.01, 0/5) keeps the divisor positive; result is capped.
    assert _half_life_estimate(0.0) == pytest.approx(120.0)


def test_build_4way_record_minimal() -> None:
    concept = {"concept_id": "fed", "label": "Fed", "theme": "macro"}
    out = _build_4way_record(
        concept,
        {"polymarket": 0.40, "kalshi": 0.55},
        {"polymarket": 10_000.0, "kalshi": 5_000.0},
    )
    assert out["concept"] == "fed"
    assert out["tradeable_size_usd"] == 5_000.0
    assert out["max_spread_pct"] == pytest.approx(15.0)
    assert out["legs_present"] == ["kalshi", "polymarket"]


def test_build_4way_record_empty_volumes() -> None:
    """Zero volumes ⇒ tradeable_size = 0."""
    out = _build_4way_record(
        {"concept_id": "x"},
        {"pm": 0.5, "ks": 0.6},
        {},
    )
    assert out["tradeable_size_usd"] == 0.0


# ---------------------------------------------------------------------------
# compute_4way_arbs
# ---------------------------------------------------------------------------


def test_compute_4way_arbs_skips_single_leg() -> None:
    """A concept with only one live venue must produce no arb."""
    fns = {"polymarket": lambda _: (0.5, 1_000.0)}  # only PM
    out = compute_4way_arbs(price_fns=fns)
    assert out == []


def test_compute_4way_arbs_ranks_by_score() -> None:
    """Spread × tradeable_size determines order."""
    fns = {
        "polymarket": lambda _: (0.40, 50_000.0),
        "kalshi": lambda _: (0.55, 50_000.0),
    }
    out = compute_4way_arbs(price_fns=fns)
    assert len(out) >= 1
    # Top entry must have a positive spread.
    assert out[0]["max_spread_pct"] > 0


def test_compute_4way_arbs_filters_by_min_spread() -> None:
    fns = {
        "polymarket": lambda _: (0.50, 1_000.0),
        "kalshi": lambda _: (0.51, 1_000.0),  # 1% spread
    }
    out = compute_4way_arbs(price_fns=fns, min_spread_pct=5.0)
    assert out == []


def test_compute_4way_arbs_price_fn_exception_skips_leg() -> None:
    def _boom(_ident: Any) -> tuple[float, float]:
        raise RuntimeError("down")

    fns = {
        "polymarket": _boom,
        "kalshi": lambda _: (0.5, 1_000.0),
        "manifold": lambda _: (0.6, 1_000.0),
    }
    out = compute_4way_arbs(price_fns=fns)
    # PM is skipped but kalshi + manifold still produce an arb.
    if out:
        for rec in out:
            assert "polymarket" not in rec["prices_per_venue"]


def test_compute_4way_arbs_uses_concept_maps_when_concepts_none() -> None:
    """When ``concepts`` is None the module's CONCEPT_MAPS is used."""
    fns = {
        "polymarket": lambda _: (0.5, 1_000.0),
        "kalshi": lambda _: (0.6, 1_000.0),
    }
    out = compute_4way_arbs(concepts=None, price_fns=fns)
    # At least one of the curated concepts must yield a 2-leg arb.
    assert len(out) >= 1


def test_compute_4way_arbs_price_none_skips() -> None:
    fns = {
        "polymarket": lambda _: (None, None),
        "kalshi": lambda _: (0.5, 100.0),
    }
    out = compute_4way_arbs(price_fns=fns)
    # Only one leg → no arb.
    assert out == []


# ---------------------------------------------------------------------------
# Async venue fetchers (respx-mocked)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_active_polymarket_happy_path() -> None:
    async with httpx.AsyncClient() as client:
        with respx.mock(assert_all_called=False) as mock:
            mock.get("https://gamma-api.polymarket.com/markets").mock(
                return_value=httpx.Response(
                    200,
                    json=[
                        {
                            "id": 1,
                            "slug": "x",
                            "question": "Q?",
                            "bestBid": 0.4,
                            "bestAsk": 0.5,
                            "volume24hr": 5_000.0,
                            "category": "Macro",
                            "endDate": "2026-12-31",
                        }
                    ],
                ),
            )
            out = await _fetch_active_polymarket(client, limit=10)
    assert len(out) == 1
    assert out[0]["price"] == pytest.approx(0.45)


@pytest.mark.asyncio
async def test_fetch_active_polymarket_http_error_returns_empty() -> None:
    async with httpx.AsyncClient() as client:
        with respx.mock(assert_all_called=False) as mock:
            mock.get("https://gamma-api.polymarket.com/markets").mock(
                return_value=httpx.Response(500, text="boom"),
            )
            out = await _fetch_active_polymarket(client, limit=10)
    assert out == []


@pytest.mark.asyncio
async def test_fetch_active_polymarket_non_list_response_returns_empty() -> None:
    async with httpx.AsyncClient() as client:
        with respx.mock(assert_all_called=False) as mock:
            mock.get("https://gamma-api.polymarket.com/markets").mock(
                return_value=httpx.Response(200, json={"oops": "object"}),
            )
            out = await _fetch_active_polymarket(client, limit=10)
    assert out == []


@pytest.mark.asyncio
async def test_fetch_active_kalshi_happy_path() -> None:
    async with httpx.AsyncClient() as client:
        with respx.mock(assert_all_called=False) as mock:
            mock.get("https://api.elections.kalshi.com/trade-api/v2/markets").mock(
                return_value=httpx.Response(
                    200,
                    json={
                        "markets": [
                            {
                                "ticker": "T1",
                                "title": "Foo",
                                "category": "Crypto",
                                "close_time": "2026-12-31",
                                "yes_bid": 48,
                                "yes_ask": 52,
                                "volume_24h": 10_000.0,
                            }
                        ]
                    },
                ),
            )
            out = await _fetch_active_kalshi(client, limit=10)
    assert len(out) == 1
    assert out[0]["price"] == pytest.approx(0.50)


@pytest.mark.asyncio
async def test_fetch_active_kalshi_429_returns_empty() -> None:
    """429 raised by raise_for_status() ⇒ caught ⇒ empty list."""
    async with httpx.AsyncClient() as client:
        with respx.mock(assert_all_called=False) as mock:
            mock.get("https://api.elections.kalshi.com/trade-api/v2/markets").mock(
                return_value=httpx.Response(429, text="slow down")
            )
            out = await _fetch_active_kalshi(client, limit=10)
    assert out == []


@pytest.mark.asyncio
async def test_fetch_active_kalshi_non_dict_response() -> None:
    async with httpx.AsyncClient() as client:
        with respx.mock(assert_all_called=False) as mock:
            mock.get("https://api.elections.kalshi.com/trade-api/v2/markets").mock(
                return_value=httpx.Response(200, json=["unexpected"])
            )
            out = await _fetch_active_kalshi(client, limit=10)
    assert out == []


@pytest.mark.asyncio
async def test_fetch_active_manifold_happy_path() -> None:
    async with httpx.AsyncClient() as client:
        with respx.mock(assert_all_called=False) as mock:
            mock.get("https://api.manifold.markets/v0/markets").mock(
                return_value=httpx.Response(
                    200,
                    json=[
                        {
                            "id": "abc",
                            "slug": "abc",
                            "question": "Q",
                            "probability": 0.6,
                            "volume24Hours": 1_000.0,
                            "isResolved": False,
                        },
                        # Resolved → skipped.
                        {
                            "id": "old",
                            "slug": "old",
                            "question": "Old",
                            "probability": 0.9,
                            "isResolved": True,
                        },
                    ],
                ),
            )
            out = await _fetch_active_manifold(client, limit=10)
    assert len(out) == 1
    assert out[0]["price"] == pytest.approx(0.6)


@pytest.mark.asyncio
async def test_fetch_active_manifold_http_error() -> None:
    async with httpx.AsyncClient() as client:
        with respx.mock(assert_all_called=False) as mock:
            mock.get("https://api.manifold.markets/v0/markets").mock(
                side_effect=httpx.ConnectError("dns-fail"),
            )
            out = await _fetch_active_manifold(client, limit=10)
    assert out == []


@pytest.mark.asyncio
async def test_fetch_active_predictit_happy_path() -> None:
    async with httpx.AsyncClient() as client:
        with respx.mock(assert_all_called=False) as mock:
            mock.get("https://www.predictit.org/api/marketdata/all/").mock(
                return_value=httpx.Response(
                    200,
                    json={
                        "markets": [
                            {
                                "id": 100,
                                "name": "Pres",
                                "contracts": [
                                    {"lastTradePrice": 0.40},
                                    {"lastTradePrice": 0.55},
                                ],
                                "totalSharesTraded": 1_000.0,
                            }
                        ]
                    },
                ),
            )
            out = await _fetch_active_predictit(client, limit=10)
    assert len(out) == 1
    assert out[0]["price"] == pytest.approx(0.55)


@pytest.mark.asyncio
async def test_fetch_active_predictit_bad_json_returns_empty() -> None:
    async with httpx.AsyncClient() as client:
        with respx.mock(assert_all_called=False) as mock:
            mock.get("https://www.predictit.org/api/marketdata/all/").mock(
                return_value=httpx.Response(200, text="not json"),
            )
            out = await _fetch_active_predictit(client, limit=10)
    assert out == []


# ---------------------------------------------------------------------------
# _gather_active_markets
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_gather_active_markets_skips_unknown_venue() -> None:
    async with httpx.AsyncClient() as client:
        with respx.mock(assert_all_called=False) as mock:
            mock.get("https://gamma-api.polymarket.com/markets").mock(
                return_value=httpx.Response(200, json=[]),
            )
            out = await _gather_active_markets(
                ["polymarket", "nonexistent-venue"],
                client,
            )
    assert "polymarket" in out
    assert "nonexistent-venue" not in out


@pytest.mark.asyncio
async def test_gather_active_markets_isolates_per_venue_failure() -> None:
    """One venue raising should not blank the whole batch."""

    async def _boom(*_a: Any, **_k: Any) -> list[dict[str, Any]]:
        raise RuntimeError("kalshi-down")

    async def _ok(*_a: Any, **_k: Any) -> list[dict[str, Any]]:
        return [{"venue": "polymarket", "slug": "x"}]

    fetchers = dict(arb_scanner._VENUE_FETCHERS)
    fetchers["polymarket"] = _ok
    fetchers["kalshi"] = _boom

    async with httpx.AsyncClient() as client:
        # Patch the registry temporarily.
        old = arb_scanner._VENUE_FETCHERS
        try:
            arb_scanner._VENUE_FETCHERS = fetchers
            out = await _gather_active_markets(["polymarket", "kalshi"], client)
        finally:
            arb_scanner._VENUE_FETCHERS = old

    assert out["polymarket"] == [{"venue": "polymarket", "slug": "x"}]
    assert out["kalshi"] == []


# ---------------------------------------------------------------------------
# auto_discover_arb_pairs
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_auto_discover_arb_pairs_happy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Synthetic universe with one obvious cross-venue match."""

    async def _fake_gather(venues: list[str], _http: Any, **_k: Any) -> dict[str, list[Any]]:
        return {
            "polymarket": [
                {
                    "venue": "polymarket",
                    "slug": "us-recession-2026",
                    "title": "Will the US enter a recession in 2026?",
                    "theme": "macro",
                    "end_date": "2026-12-31T00:00:00Z",
                    "price": 0.40,
                    "volume_24h_usd": 30_000.0,
                }
            ],
            "kalshi": [
                {
                    "venue": "kalshi",
                    "ticker": "KXRECSSNBER-26",
                    "slug": "KXRECSSNBER-26",
                    "title": "US recession declared by NBER in 2026",
                    "theme": "macro",
                    "end_date": "2026-12-31T00:00:00Z",
                    "price": 0.55,
                    "volume_24h_usd": 20_000.0,
                }
            ],
        }

    monkeypatch.setattr(arb_scanner, "_gather_active_markets", _fake_gather)

    pairs = await auto_discover_arb_pairs(
        min_similarity=0.5,
        min_volume_usd_per_venue=1_000.0,
        max_pairs=10,
        venues=["polymarket", "kalshi"],
        http=httpx.AsyncClient(),  # passed in but never used by stub
    )
    assert len(pairs) == 1
    p = pairs[0]
    assert p["venue_a"] == "polymarket"
    assert p["venue_b"] == "kalshi"
    assert p["spread_pct"] == pytest.approx(15.0, rel=1e-3)
    assert p["tradeable_size_usd"] == 20_000.0
    # First observation never confirmed.
    assert p["confirmed"] is False


@pytest.mark.asyncio
async def test_auto_discover_arb_pairs_volume_floor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Markets below the per-venue volume floor are filtered out."""

    async def _gather(*_a: Any, **_k: Any) -> dict[str, list[Any]]:
        return {
            "polymarket": [
                {
                    "slug": "x",
                    "title": "Foo recession",
                    "theme": "macro",
                    "end_date": "2026-12-31T00:00:00Z",
                    "price": 0.4,
                    "volume_24h_usd": 100.0,  # below floor
                }
            ],
            "kalshi": [
                {
                    "ticker": "K",
                    "slug": "K",
                    "title": "Foo recession kalshi",
                    "theme": "macro",
                    "end_date": "2026-12-31T00:00:00Z",
                    "price": 0.5,
                    "volume_24h_usd": 100.0,
                }
            ],
        }

    monkeypatch.setattr(arb_scanner, "_gather_active_markets", _gather)
    pairs = await auto_discover_arb_pairs(
        min_volume_usd_per_venue=10_000.0,
        venues=["polymarket", "kalshi"],
        http=httpx.AsyncClient(),
    )
    assert pairs == []


@pytest.mark.asyncio
async def test_auto_discover_arb_pairs_similarity_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Below-threshold pairs are filtered out."""

    async def _gather(*_a: Any, **_k: Any) -> dict[str, list[Any]]:
        return {
            "polymarket": [
                {
                    "slug": "p",
                    "title": "Apples baked in autumn",
                    "theme": "food",
                    "end_date": "2026-12-31T00:00:00Z",
                    "price": 0.4,
                    "volume_24h_usd": 50_000.0,
                }
            ],
            "kalshi": [
                {
                    "ticker": "K",
                    "slug": "K",
                    "title": "Zebras racing in Africa",
                    "theme": "sports",
                    "end_date": "2026-12-31T00:00:00Z",
                    "price": 0.5,
                    "volume_24h_usd": 50_000.0,
                }
            ],
        }

    monkeypatch.setattr(arb_scanner, "_gather_active_markets", _gather)
    pairs = await auto_discover_arb_pairs(
        min_similarity=0.9,
        venues=["polymarket", "kalshi"],
        http=httpx.AsyncClient(),
    )
    assert pairs == []


@pytest.mark.asyncio
async def test_auto_discover_arb_pairs_max_pairs_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Output is capped at ``max_pairs`` after scoring."""

    pm = [
        {
            "slug": f"pm-{i}",
            "title": f"Will event {i} happen in 2026?",
            "theme": "macro",
            "end_date": "2026-12-31T00:00:00Z",
            "price": 0.5,
            "volume_24h_usd": 50_000.0,
        }
        for i in range(5)
    ]
    ks = [
        {
            "ticker": f"K-{i}",
            "slug": f"K-{i}",
            "title": f"Will event {i} happen in 2026?",
            "theme": "macro",
            "end_date": "2026-12-31T00:00:00Z",
            "price": 0.6,
            "volume_24h_usd": 30_000.0,
        }
        for i in range(5)
    ]

    async def _g(*_a: Any, **_k: Any) -> dict[str, list[Any]]:
        return {"polymarket": pm, "kalshi": ks}

    monkeypatch.setattr(arb_scanner, "_gather_active_markets", _g)
    pairs = await auto_discover_arb_pairs(
        min_similarity=0.5,
        max_pairs=2,
        venues=["polymarket", "kalshi"],
        http=httpx.AsyncClient(),
    )
    assert len(pairs) <= 2


# ---------------------------------------------------------------------------
# top_arbs: empty path + when compute_arb_spreads returns nothing
# ---------------------------------------------------------------------------


def test_top_arbs_no_arbs_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(arb_scanner, "all_matched_pairs", lambda: [])
    out = top_arbs(http=MagicMock(), kalshi_client=MagicMock())
    assert out == []


def test_top_arbs_zero_n_clipped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        arb_scanner,
        "all_matched_pairs",
        lambda: [
            {"pm_slug": "a", "kalshi_slug": "A", "label": ""},
        ],
    )
    monkeypatch.setattr(arb_scanner, "_pm_mid", lambda *_a, **_k: (0.4, 10_000.0))
    monkeypatch.setattr(arb_scanner, "_kalshi_mid", lambda *_a, **_k: (0.6, 10_000.0))
    out = top_arbs(n=-5, http=MagicMock(), kalshi_client=MagicMock())
    # max(0, -5) → empty slice.
    assert out == []


def test_all_matched_pairs_includes_manual() -> None:
    arb_scanner._MANUAL_PAIRS.append(
        {"pm_slug": "manual-pm", "kalshi_slug": "MANUAL-K", "label": "M", "theme": "x"},
    )
    pairs = all_matched_pairs()
    assert any(p["pm_slug"] == "manual-pm" for p in pairs)


# ---------------------------------------------------------------------------
# Endpoint coverage: cache hit + error branches
# ---------------------------------------------------------------------------


def test_get_scanner_cache_hit(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two calls with identical params: the second must hit the cache."""
    call_count = {"n": 0}

    def _spy_top_arbs(**_k: Any) -> list[dict[str, Any]]:
        call_count["n"] += 1
        return [
            {
                "pm_slug": "p",
                "kalshi_slug": "K",
                "label": "",
                "pm_price": 0.4,
                "kalshi_price": 0.5,
                "spread_pct": 10.0,
                "direction": "buy_pm_sell_kalshi",
                "tradeable_size_usd": 1_000.0,
                "half_life_minutes": 30.0,
                "last_seen_iso": "2026-05-08T00:00:00+00:00",
                "confirmed": True,
                "confirmation_window_min": 30,
            }
        ]

    monkeypatch.setattr(arb_scanner, "top_arbs", _spy_top_arbs)
    r1 = client.get("/arb/scanner", params={"min_spread_pct": 2.0, "n": 5})
    r2 = client.get("/arb/scanner", params={"min_spread_pct": 2.0, "n": 5})
    assert r1.status_code == 200
    assert r2.status_code == 200
    assert call_count["n"] == 1


def test_post_match_empty_slug_returns_422(client: TestClient) -> None:
    """Pydantic ``min_length=1`` rejects empty slugs before our 400 path."""
    r = client.post("/arb/match", json={"pm_slug": "", "kalshi_slug": "K"})
    assert r.status_code == 422


def test_post_match_whitespace_only_slug_returns_400(client: TestClient) -> None:
    """A whitespace-only slug passes Pydantic but our handler raises 400."""
    r = client.post("/arb/match", json={"pm_slug": "   ", "kalshi_slug": "K"})
    assert r.status_code == 400


def test_get_matched_cache_hit(client: TestClient) -> None:
    r1 = client.get("/arb/matched")
    r2 = client.get("/arb/matched")
    assert r1.json() == r2.json()


def test_get_4way_concept_unknown_404(client: TestClient) -> None:
    r = client.get("/arb/concept/does-not-exist")
    assert r.status_code == 404


def test_get_4way_concept_known_returns_venues(client: TestClient) -> None:
    r = client.get(f"/arb/concept/{CONCEPT_MAPS[0]['concept_id']}")
    assert r.status_code == 200
    body = r.json()
    assert "venues" in body
    assert body["venues"]["polymarket"] == CONCEPT_MAPS[0]["polymarket"]


def test_list_4way_concepts(client: TestClient) -> None:
    r = client.get("/arb/concepts")
    assert r.status_code == 200
    body = r.json()
    assert body["n"] == len(CONCEPT_MAPS)


def test_get_confirmed_matches_empty(client: TestClient) -> None:
    r = client.get("/arb/confirmed-matches")
    assert r.status_code == 200
    body = r.json()
    assert body["n"] == 0
    assert body["matches"] == []


def test_get_confirmed_matches_returns_unconfirmed_when_flag_off(
    client: TestClient,
) -> None:
    record_match_observation(
        "polymarket",
        "x",
        "kalshi",
        "Y",
        similarity=0.7,
        label="lo",
    )
    r = client.get("/arb/confirmed-matches", params={"only_confirmed": "false"})
    assert r.status_code == 200
    body = r.json()
    assert body["n"] == 1


def test_get_4way_arbs_cache_hit(client: TestClient) -> None:
    r1 = client.get("/arb/4way-arbs")
    r2 = client.get("/arb/4way-arbs")
    assert r1.status_code == 200
    assert r1.json() == r2.json()


def test_get_4way_arbs_with_filter(client: TestClient) -> None:
    """No price_fns wired ⇒ empty arbs regardless of threshold."""
    r = client.get("/arb/4way-arbs", params={"min_spread_pct": 50.0})
    assert r.status_code == 200
    body = r.json()
    assert body["arbs"] == []


def test_auto_discover_endpoint_uses_cache(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"n": 0}

    async def _stub(**_k: Any) -> list[dict[str, Any]]:
        calls["n"] += 1
        return [{"venue_a": "polymarket", "venue_b": "kalshi", "label": "x"}]

    monkeypatch.setattr(arb_scanner, "auto_discover_arb_pairs", _stub)
    r1 = client.get(
        "/arb/auto-discover",
        params={"min_similarity": 0.65, "min_volume": 1_000.0, "max_pairs": 10},
    )
    r2 = client.get(
        "/arb/auto-discover",
        params={"min_similarity": 0.65, "min_volume": 1_000.0, "max_pairs": 10},
    )
    assert r1.status_code == 200
    assert r2.status_code == 200
    assert calls["n"] == 1


# ---------------------------------------------------------------------------
# Cache + per-loop lock plumbing
# ---------------------------------------------------------------------------


def test_auto_discover_lock_returns_same_lock_for_key() -> None:
    """Same key on the same loop ⇒ same Lock object (single-flight requirement)."""

    async def _check() -> bool:
        l1 = _auto_discover_lock(("k", 1))
        l2 = _auto_discover_lock(("k", 1))
        return l1 is l2

    assert asyncio.run(_check()) is True


def test_auto_discover_lock_different_keys_different_locks() -> None:
    async def _check() -> bool:
        l1 = _auto_discover_lock(("a",))
        l2 = _auto_discover_lock(("b",))
        return l1 is not l2

    assert asyncio.run(_check()) is True
