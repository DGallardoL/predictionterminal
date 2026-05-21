"""Tests for ``pfm.portfolio_rebalance_router``.

We mount both the import router (to seed real portfolios) and the
rebalance router on a fresh FastAPI per test. The store
(``app.state.portfolios``) is therefore per-test and we don't need to
mock it explicitly — but we DO exercise the pure-function
``_propose_trades`` path directly for finer-grained assertions and to
mock the portfolio state without going through CSV parsing.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterator
from datetime import UTC, datetime

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pfm.portfolio_import_router import (
    Portfolio,
    PortfolioRow,
)
from pfm.portfolio_import_router import (
    router as import_router,
)
from pfm.portfolio_rebalance_router import (
    HOLD_TOLERANCE_SHARES,
    MAX_TICKERS_PER_REQUEST,
    _propose_trades,
)
from pfm.portfolio_rebalance_router import (
    router as rebalance_router,
)

# ---- fixtures -------------------------------------------------------------


@pytest.fixture()
def app() -> Iterator[FastAPI]:
    a = FastAPI()
    a.include_router(import_router)
    a.include_router(rebalance_router)
    yield a


@pytest.fixture()
def client(app: FastAPI) -> Iterator[TestClient]:
    with TestClient(app) as c:
        yield c


def _seed_portfolio(
    app: FastAPI,
    handle: str,
    rows: list[tuple[str, float]],
) -> Portfolio:
    """Inject a portfolio directly into ``app.state.portfolios`` —
    bypasses CSV parsing so each test can craft an exact state."""
    store: OrderedDict[str, Portfolio] = getattr(app.state, "portfolios", None) or OrderedDict()
    pf = Portfolio(
        handle=handle,
        rows=[PortfolioRow(ticker=t, shares=s) for t, s in rows],
        created_at=datetime.now(UTC),
    )
    store[handle] = pf
    app.state.portfolios = store
    return pf


# ---- pure-function unit tests --------------------------------------------


class TestProposeTrades:
    """Direct tests of the rebalance math, mocking the Portfolio state."""

    def test_simple_two_ticker_rebalance(self) -> None:
        # Current: 100 NVDA @ $1000 = $100k, 50 TSLA @ $200 = $10k.
        # Total = $110k. Target 50/50 => $55k each.
        # NVDA target = 55 shares (sell 45). TSLA target = 275 (buy 225).
        pf = Portfolio(
            handle="pf_test_1",
            rows=[
                PortfolioRow(ticker="NVDA", shares=100.0),
                PortfolioRow(ticker="TSLA", shares=50.0),
            ],
            created_at=datetime.now(UTC),
        )
        trades, total, warnings = _propose_trades(
            pf,
            target_weights={"NVDA": 0.5, "TSLA": 0.5},
            current_prices={"NVDA": 1000.0, "TSLA": 200.0},
        )
        assert total == pytest.approx(110_000.0)
        assert warnings == []
        by_t = {t.ticker: t for t in trades}
        assert by_t["NVDA"].action == "sell"
        assert by_t["NVDA"].target_shares == pytest.approx(55.0)
        assert by_t["NVDA"].delta == pytest.approx(-45.0)
        assert by_t["TSLA"].action == "buy"
        assert by_t["TSLA"].target_shares == pytest.approx(275.0)
        assert by_t["TSLA"].delta == pytest.approx(225.0)

    def test_hold_when_already_at_target(self) -> None:
        # Current weights are already 50/50; rebalance proposes hold.
        pf = Portfolio(
            handle="pf_hold",
            rows=[
                PortfolioRow(ticker="AAPL", shares=100.0),
                PortfolioRow(ticker="MSFT", shares=50.0),
            ],
            created_at=datetime.now(UTC),
        )
        # 100 * 100 = 10_000 ; 50 * 200 = 10_000 ; total = 20_000
        trades, total, _ = _propose_trades(
            pf,
            target_weights={"AAPL": 0.5, "MSFT": 0.5},
            current_prices={"AAPL": 100.0, "MSFT": 200.0},
        )
        assert total == pytest.approx(20_000.0)
        for t in trades:
            assert t.action == "hold"
            assert abs(t.delta) <= HOLD_TOLERANCE_SHARES

    def test_new_ticker_in_target_only_is_a_buy(self) -> None:
        pf = Portfolio(
            handle="pf_new",
            rows=[PortfolioRow(ticker="NVDA", shares=10.0)],
            created_at=datetime.now(UTC),
        )
        # Total = 10 * 100 = $1000. Target 50% GOOG => $500 / $250 = 2.0.
        trades, _, _ = _propose_trades(
            pf,
            target_weights={"NVDA": 0.5, "GOOG": 0.5},
            current_prices={"NVDA": 100.0, "GOOG": 250.0},
        )
        by_t = {t.ticker: t for t in trades}
        assert by_t["GOOG"].current_shares == 0.0
        assert by_t["GOOG"].action == "buy"
        assert by_t["GOOG"].target_shares == pytest.approx(2.0)
        assert by_t["GOOG"].delta == pytest.approx(2.0)

    def test_held_ticker_missing_from_target_is_full_sell_with_warning(
        self,
    ) -> None:
        pf = Portfolio(
            handle="pf_drop",
            rows=[
                PortfolioRow(ticker="NVDA", shares=10.0),
                PortfolioRow(ticker="TSLA", shares=20.0),
            ],
            created_at=datetime.now(UTC),
        )
        trades, _, warnings = _propose_trades(
            pf,
            target_weights={"NVDA": 1.0},
            current_prices={"NVDA": 100.0, "TSLA": 50.0},
        )
        by_t = {t.ticker: t for t in trades}
        assert by_t["TSLA"].action == "sell"
        assert by_t["TSLA"].target_shares == pytest.approx(0.0)
        assert by_t["TSLA"].delta == pytest.approx(-20.0)
        assert any("TSLA" in w for w in warnings)

    def test_duplicate_holdings_aggregated_into_one_trade(self) -> None:
        # The importer warns on dup tickers but keeps both rows; the
        # rebalancer must aggregate them so only one trade is emitted.
        pf = Portfolio(
            handle="pf_dup",
            rows=[
                PortfolioRow(ticker="NVDA", shares=10.0),
                PortfolioRow(ticker="NVDA", shares=5.0),
            ],
            created_at=datetime.now(UTC),
        )
        trades, total, _ = _propose_trades(
            pf,
            target_weights={"NVDA": 1.0},
            current_prices={"NVDA": 100.0},
        )
        assert len(trades) == 1
        assert trades[0].ticker == "NVDA"
        assert trades[0].current_shares == pytest.approx(15.0)
        assert total == pytest.approx(1500.0)

    def test_missing_price_for_held_ticker_raises_422(self) -> None:
        from fastapi import HTTPException

        pf = Portfolio(
            handle="pf_missing",
            rows=[
                PortfolioRow(ticker="NVDA", shares=10.0),
                PortfolioRow(ticker="TSLA", shares=20.0),
            ],
            created_at=datetime.now(UTC),
        )
        with pytest.raises(HTTPException) as ei:
            _propose_trades(
                pf,
                target_weights={"NVDA": 1.0},
                current_prices={"NVDA": 100.0},  # TSLA price missing
            )
        assert ei.value.status_code == 422
        assert "TSLA" in str(ei.value.detail)

    def test_missing_price_for_target_only_ticker_raises_422(self) -> None:
        from fastapi import HTTPException

        pf = Portfolio(
            handle="pf_missing2",
            rows=[PortfolioRow(ticker="NVDA", shares=10.0)],
            created_at=datetime.now(UTC),
        )
        with pytest.raises(HTTPException) as ei:
            _propose_trades(
                pf,
                target_weights={"NVDA": 0.5, "GOOG": 0.5},
                current_prices={"NVDA": 100.0},  # GOOG price missing
            )
        assert ei.value.status_code == 422
        assert "GOOG" in str(ei.value.detail)


# ---- endpoint integration tests ------------------------------------------


class TestEndpoint:
    def test_unknown_handle_returns_404(self, client: TestClient) -> None:
        r = client.post(
            "/portfolio/pf_does_not_exist/rebalance",
            json={
                "target_weights": {"NVDA": 1.0},
                "current_prices": {"NVDA": 100.0},
            },
        )
        assert r.status_code == 404

    def test_happy_path_returns_trades(self, app: FastAPI, client: TestClient) -> None:
        _seed_portfolio(
            app,
            "pf_happy",
            [("NVDA", 100.0), ("TSLA", 50.0)],
        )
        r = client.post(
            "/portfolio/pf_happy/rebalance",
            json={
                "target_weights": {"NVDA": 0.5, "TSLA": 0.5},
                "current_prices": {"NVDA": 1000.0, "TSLA": 200.0},
            },
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["handle"] == "pf_happy"
        assert body["total_value"] == pytest.approx(110_000.0)
        assert body["cash_weight"] == pytest.approx(0.0)
        tickers = {t["ticker"] for t in body["trades"]}
        assert tickers == {"NVDA", "TSLA"}
        actions = {t["ticker"]: t["action"] for t in body["trades"]}
        assert actions["NVDA"] == "sell"
        assert actions["TSLA"] == "buy"

    def test_cash_weight_reported_when_weights_sum_below_one(
        self, app: FastAPI, client: TestClient
    ) -> None:
        _seed_portfolio(app, "pf_cash", [("NVDA", 10.0)])
        r = client.post(
            "/portfolio/pf_cash/rebalance",
            json={
                "target_weights": {"NVDA": 0.6},
                "current_prices": {"NVDA": 100.0},
            },
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["cash_weight"] == pytest.approx(0.4)

    def test_negative_weight_rejected_at_validation(self, app: FastAPI, client: TestClient) -> None:
        _seed_portfolio(app, "pf_neg", [("NVDA", 10.0)])
        r = client.post(
            "/portfolio/pf_neg/rebalance",
            json={
                "target_weights": {"NVDA": -0.1},
                "current_prices": {"NVDA": 100.0},
            },
        )
        assert r.status_code == 422

    def test_weight_above_one_rejected(self, app: FastAPI, client: TestClient) -> None:
        _seed_portfolio(app, "pf_big", [("NVDA", 10.0)])
        r = client.post(
            "/portfolio/pf_big/rebalance",
            json={
                "target_weights": {"NVDA": 1.5},
                "current_prices": {"NVDA": 100.0},
            },
        )
        assert r.status_code == 422

    def test_zero_price_rejected(self, app: FastAPI, client: TestClient) -> None:
        _seed_portfolio(app, "pf_zero", [("NVDA", 10.0)])
        r = client.post(
            "/portfolio/pf_zero/rebalance",
            json={
                "target_weights": {"NVDA": 1.0},
                "current_prices": {"NVDA": 0.0},
            },
        )
        assert r.status_code == 422

    def test_lowercase_ticker_keys_normalized(self, app: FastAPI, client: TestClient) -> None:
        # Importer always uppercases; rebalance request should too so a
        # caller mixing case doesn't get spurious 422s.
        _seed_portfolio(app, "pf_case", [("NVDA", 10.0)])
        r = client.post(
            "/portfolio/pf_case/rebalance",
            json={
                "target_weights": {"nvda": 1.0},
                "current_prices": {"nvda": 100.0},
            },
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["trades"][0]["ticker"] == "NVDA"
        assert body["trades"][0]["action"] == "hold"

    def test_warning_emitted_for_dropped_holding(self, app: FastAPI, client: TestClient) -> None:
        _seed_portfolio(app, "pf_drop2", [("NVDA", 10.0), ("TSLA", 20.0)])
        r = client.post(
            "/portfolio/pf_drop2/rebalance",
            json={
                "target_weights": {"NVDA": 1.0},
                "current_prices": {"NVDA": 100.0, "TSLA": 50.0},
            },
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert any("TSLA" in w for w in body["warnings"])
        tsla = next(t for t in body["trades"] if t["ticker"] == "TSLA")
        assert tsla["action"] == "sell"
        assert tsla["target_shares"] == pytest.approx(0.0)

    def test_too_many_target_tickers_rejected(self, app: FastAPI, client: TestClient) -> None:
        _seed_portfolio(app, "pf_huge", [("NVDA", 10.0)])
        weights = {f"T{i:04d}": 0.0 for i in range(MAX_TICKERS_PER_REQUEST + 1)}
        weights["NVDA"] = 1.0
        prices = dict.fromkeys(weights, 1.0)
        r = client.post(
            "/portfolio/pf_huge/rebalance",
            json={"target_weights": weights, "current_prices": prices},
        )
        assert r.status_code == 422

    def test_full_round_trip_via_import_then_rebalance(self, client: TestClient) -> None:
        # End-to-end: real CSV import path, then rebalance the returned handle.
        csv = "ticker,shares,cost_basis\nNVDA,10,1000\nTSLA,20,2000\n"
        r = client.post(
            "/portfolio/import",
            content=csv,
            headers={"Content-Type": "text/csv"},
        )
        assert r.status_code == 200, r.text
        handle = r.json()["handle"]

        r2 = client.post(
            f"/portfolio/{handle}/rebalance",
            json={
                "target_weights": {"NVDA": 0.5, "TSLA": 0.5},
                "current_prices": {"NVDA": 200.0, "TSLA": 100.0},
            },
        )
        assert r2.status_code == 200, r2.text
        body = r2.json()
        # 10 * 200 = 2000 ; 20 * 100 = 2000 ; total = 4000 -> already 50/50
        assert body["total_value"] == pytest.approx(4000.0)
        for t in body["trades"]:
            assert t["action"] == "hold"
