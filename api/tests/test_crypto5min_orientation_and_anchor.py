"""Tests for the safety-critical pieces: Up/Down orientation, market-anchor
blending, and Binance aggTrades VWAP. These are the bits that — if wrong —
would silently flip every signal."""

from __future__ import annotations

import httpx
import pytest
import respx

from pfm.crypto5min.comparator import (
    DEFAULT_EDGE_THRESHOLD,
    MARKET_ANCHOR_WEIGHT,
    anchor_to_market,
    compare_market_vs_model,
)
from pfm.crypto5min.market_fetcher import (
    fetch_binance_price_at,
    parse_active_market,
)
from pfm.crypto5min.predictor import predict_for_window

# ---------------------------------------------------------------------------
# Up/Down token orientation
# ---------------------------------------------------------------------------


def test_active_market_picks_token_0_as_up() -> None:
    """clobTokenIds[0] = Up, clobTokenIds[1] = Down. Match the Polymarket
    gamma convention verified against the live CLOB endpoint."""
    import json

    payload = {
        "slug": "btc-updown-5m-1700000000",
        "id": "1",
        "closed": False,
        "active": True,
        "clobTokenIds": json.dumps(["TOK_UP_ID", "TOK_DOWN_ID"]),
    }
    parsed = parse_active_market(
        payload,
        asset="BTC",
        window_minutes=5,
        binance_symbol="BTCUSDT",
        end_unix=1_700_000_000,
        now_unix=1_700_000_000 - 100,
    )
    assert parsed is not None
    assert parsed.up_token_id == "TOK_UP_ID"
    assert parsed.down_token_id == "TOK_DOWN_ID"


# ---------------------------------------------------------------------------
# anchor_to_market — math
# ---------------------------------------------------------------------------


def test_anchor_returns_gbm_when_market_is_none() -> None:
    assert anchor_to_market(0.80, None) == pytest.approx(0.80)


def test_anchor_full_weight_on_market_returns_market_exactly() -> None:
    assert anchor_to_market(0.80, 0.45, weight=1.0) == pytest.approx(0.45)


def test_anchor_zero_weight_returns_raw_gbm() -> None:
    assert anchor_to_market(0.80, 0.45, weight=0.0) == pytest.approx(0.80)


def test_anchor_default_weight_blends_correctly() -> None:
    """final = w·market + (1-w)·gbm at the module default weight."""
    out = anchor_to_market(0.80, 0.45)
    expected = MARKET_ANCHOR_WEIGHT * 0.45 + (1.0 - MARKET_ANCHOR_WEIGHT) * 0.80
    assert out == pytest.approx(expected)


def test_anchor_invalid_weight_raises() -> None:
    with pytest.raises(ValueError):
        anchor_to_market(0.50, 0.50, weight=-0.1)
    with pytest.raises(ValueError):
        anchor_to_market(0.50, 0.50, weight=1.5)


def test_anchor_preserves_endpoints() -> None:
    """gbm=market ⇒ final=market for any weight."""
    for w in (0.0, 0.25, 0.50, 0.75, 1.0):
        assert anchor_to_market(0.42, 0.42, weight=w) == pytest.approx(0.42)


def test_anchor_signed_drift_in_correct_direction() -> None:
    """gbm > market ⇒ anchored > market (small positive tilt)."""
    out = anchor_to_market(0.99, 0.40)
    assert out > 0.40
    assert out < 0.99


# ---------------------------------------------------------------------------
# compare_market_vs_model — uses anchor internally
# ---------------------------------------------------------------------------


def _atm_pred():
    return predict_for_window(
        spot_t=60_000.0,
        spot_0=60_000.0,
        seconds_remaining=300.0,
        sigma_long_annual=0.65,
    )


def test_compare_exposes_both_raw_and_anchored_probs() -> None:
    pred = _atm_pred()
    out = compare_market_vs_model(
        slug="x",
        asset="BTC",
        window_minutes=5,
        market_prob_up=0.30,
        prediction=pred,
    )
    d = out.as_dict()
    assert d["model_prob_gbm_raw"] == pytest.approx(pred.prob_up)
    expected_anchored = MARKET_ANCHOR_WEIGHT * 0.30 + (1.0 - MARKET_ANCHOR_WEIGHT) * pred.prob_up
    assert d["model_prob_up"] == pytest.approx(expected_anchored)


def test_compare_edge_uses_anchored_not_raw() -> None:
    pred = _atm_pred()
    out = compare_market_vs_model(
        slug="x",
        asset="BTC",
        window_minutes=5,
        market_prob_up=0.30,
        prediction=pred,
    )
    # edge = anchored - market = (1-w)·(gbm - market)
    raw_gap = pred.prob_up - 0.30
    expected_edge = (1.0 - MARKET_ANCHOR_WEIGHT) * raw_gap
    assert out.edge == pytest.approx(expected_edge)


def test_compare_anchor_shrinks_extreme_edges_to_realistic_range() -> None:
    """Anchored edge is shrunk by (1-w) vs the raw gbm-vs-market gap."""
    pred = _atm_pred()  # gbm ≈ 0.50
    market_p = 0.05
    raw_gap = pred.prob_up - market_p
    out = compare_market_vs_model(
        slug="x",
        asset="BTC",
        window_minutes=5,
        market_prob_up=market_p,  # huge gap from 0.50
        prediction=pred,
    )
    expected = (1.0 - MARKET_ANCHOR_WEIGHT) * raw_gap
    assert abs(out.edge) <= abs(expected) + 1e-9
    assert abs(out.edge) < abs(raw_gap)  # anchor strictly shrinks


def test_compare_signal_fires_only_above_threshold() -> None:
    pred = _atm_pred()  # gbm near 0.50
    # Anchored edge = (1-w)·raw_gap; pick gaps that straddle the threshold
    # for any reasonable weight in [0.5, 0.9].
    wait_gap = DEFAULT_EDGE_THRESHOLD / (1.0 - MARKET_ANCHOR_WEIGHT) * 0.5
    fire_gap = DEFAULT_EDGE_THRESHOLD / (1.0 - MARKET_ANCHOR_WEIGHT) * 2.0
    out_below = compare_market_vs_model(
        slug="x",
        asset="BTC",
        window_minutes=5,
        market_prob_up=pred.prob_up - wait_gap,
        prediction=pred,
    )
    out_above = compare_market_vs_model(
        slug="x",
        asset="BTC",
        window_minutes=5,
        market_prob_up=pred.prob_up - fire_gap,
        prediction=pred,
    )
    assert out_below.signal == "WAIT"
    assert out_above.signal == "BUY_YES"
    # Threshold value sanity-check
    assert DEFAULT_EDGE_THRESHOLD < 0.05  # we shrunk it post-anchor


# ---------------------------------------------------------------------------
# Binance aggTrades VWAP — accurate strike fallback
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_aggtrades_vwap_uniform_trades() -> None:
    """All trades at same price + same size → VWAP equals that price."""
    rows = [{"p": "60000.0", "q": "1.0", "T": 1_700_000_000_000} for _ in range(5)]
    async with respx.mock:
        respx.get("https://api.binance.com/api/v3/aggTrades").mock(
            return_value=httpx.Response(200, json=rows)
        )
        async with httpx.AsyncClient() as client:
            price = await fetch_binance_price_at(client, "BTCUSDT", 1_700_000_000)
    assert price == pytest.approx(60_000.0)


@pytest.mark.asyncio
async def test_aggtrades_vwap_weighted_correctly() -> None:
    """VWAP = Σ(p·q) / Σq. Test with skewed sizes."""
    rows = [
        {"p": "60000.0", "q": "1.0"},
        {"p": "60100.0", "q": "9.0"},  # 9× size → should dominate
    ]
    async with respx.mock:
        respx.get("https://api.binance.com/api/v3/aggTrades").mock(
            return_value=httpx.Response(200, json=rows)
        )
        async with httpx.AsyncClient() as client:
            price = await fetch_binance_price_at(client, "BTCUSDT", 1_700_000_000)
    expected = (60000.0 * 1.0 + 60100.0 * 9.0) / 10.0
    assert price == pytest.approx(expected)


@pytest.mark.asyncio
async def test_aggtrades_handles_empty_response() -> None:
    async with respx.mock:
        respx.get("https://api.binance.com/api/v3/aggTrades").mock(
            return_value=httpx.Response(200, json=[])
        )
        async with httpx.AsyncClient() as client:
            price = await fetch_binance_price_at(client, "BTCUSDT", 1_700_000_000)
    assert price is None


@pytest.mark.asyncio
async def test_aggtrades_handles_5xx() -> None:
    async with respx.mock:
        respx.get("https://api.binance.com/api/v3/aggTrades").mock(return_value=httpx.Response(503))
        async with httpx.AsyncClient() as client:
            price = await fetch_binance_price_at(client, "BTCUSDT", 1_700_000_000)
    assert price is None


@pytest.mark.asyncio
async def test_aggtrades_skips_bad_rows() -> None:
    """Malformed individual rows should be skipped; good ones contribute."""
    rows = [
        {"p": "60000.0", "q": "1.0"},
        {"p": "oops", "q": "2.0"},  # bad price
        {"p": "60100.0"},  # missing qty
        {"p": "60050.0", "q": "1.0"},
    ]
    async with respx.mock:
        respx.get("https://api.binance.com/api/v3/aggTrades").mock(
            return_value=httpx.Response(200, json=rows)
        )
        async with httpx.AsyncClient() as client:
            price = await fetch_binance_price_at(client, "BTCUSDT", 1_700_000_000)
    expected = (60000.0 * 1.0 + 60050.0 * 1.0) / 2.0
    assert price == pytest.approx(expected)


@pytest.mark.asyncio
async def test_aggtrades_rejects_non_positive_values() -> None:
    rows = [
        {"p": "0", "q": "1.0"},  # zero price → skip
        {"p": "60000.0", "q": "0"},  # zero qty → skip
        {"p": "-1", "q": "1.0"},  # negative → skip
        {"p": "60100.0", "q": "1.0"},  # only valid one
    ]
    async with respx.mock:
        respx.get("https://api.binance.com/api/v3/aggTrades").mock(
            return_value=httpx.Response(200, json=rows)
        )
        async with httpx.AsyncClient() as client:
            price = await fetch_binance_price_at(client, "BTCUSDT", 1_700_000_000)
    assert price == pytest.approx(60_100.0)


@pytest.mark.asyncio
async def test_aggtrades_uses_correct_time_window() -> None:
    """Verify that startTime/endTime params are sent correctly."""
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["startTime"] = int(req.url.params["startTime"])
        seen["endTime"] = int(req.url.params["endTime"])
        seen["symbol"] = req.url.params["symbol"]
        return httpx.Response(200, json=[{"p": "60000.0", "q": "1.0"}])

    async with respx.mock:
        respx.get("https://api.binance.com/api/v3/aggTrades").mock(side_effect=handler)
        async with httpx.AsyncClient() as client:
            await fetch_binance_price_at(
                client,
                "BTCUSDT",
                1_700_000_000,
                window_seconds=3,
            )
    assert seen["startTime"] == 1_700_000_000 * 1000
    assert seen["endTime"] == 1_700_000_000 * 1000 + 3000
    assert seen["symbol"] == "BTCUSDT"
