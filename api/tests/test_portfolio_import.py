"""Tests for ``pfm.portfolio_import_router``.

We mount the router on a *fresh* ``FastAPI`` instance per test rather
than the full ``pfm.main`` app — that keeps the suite fast (no real
upstream client init) and isolates the ``app.state.portfolios`` store.

Note: T33 pivoted from ``pfm/portfolio/import_router.py`` to a flat
``pfm/portfolio_import_router.py`` module to avoid shadowing the
existing ``pfm/portfolio.py`` (which exports ``vol_targeted_combiner``).
"""

from __future__ import annotations

import importlib.util
from collections import OrderedDict
from collections.abc import Iterator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

# ``python-multipart`` is an optional install for Starlette/FastAPI; the
# multipart-upload tests need it. Skip cleanly when absent rather than
# hard-failing — the raw text/csv path is the primary contract.
_HAS_MULTIPART = (
    importlib.util.find_spec("multipart") is not None
    or importlib.util.find_spec("python_multipart") is not None
)
requires_multipart = pytest.mark.skipif(
    not _HAS_MULTIPART,
    reason="python-multipart not installed",
)

from pfm.portfolio_import_router import (
    MAX_HANDLES,
    MAX_ROWS,
    Portfolio,
    get_portfolio,
    router,
)

# ---- fixtures -------------------------------------------------------------


@pytest.fixture()
def app() -> Iterator[FastAPI]:
    a = FastAPI()
    a.include_router(router)
    yield a


@pytest.fixture()
def client(app: FastAPI) -> Iterator[TestClient]:
    with TestClient(app) as c:
        yield c


def _csv(*lines: str) -> str:
    return "\n".join(lines) + "\n"


def _post_csv(client: TestClient, body: str):
    return client.post(
        "/portfolio/import",
        content=body,
        headers={"Content-Type": "text/csv"},
    )


def _letter_ticker(i: int) -> str:
    """Generate ``i``-th letter-only ticker (e.g. AA, AB, ... ZZ, AAA, ...)."""
    chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    out = ""
    n = i
    while True:
        out = chars[n % 26] + out
        n = n // 26 - 1
        if n < 0:
            break
    # Guarantee 1-5 chars upper-case letters.
    return out


# ---- happy path -----------------------------------------------------------


class TestValidImport:
    def test_five_rows_returns_handle_and_count(self, client: TestClient) -> None:
        body = _csv(
            "ticker,shares,cost_basis",
            "NVDA,100,12450.50",
            "TSLA,50,15000",
            "AAPL,200,32000.00",
            "GOOG,10,1500.25",
            "MSFT,80,30400",
        )
        r = _post_csv(client, body)
        assert r.status_code == 200, r.text
        body_json = r.json()
        assert body_json["row_count"] == 5
        assert body_json["tickers"] == ["NVDA", "TSLA", "AAPL", "GOOG", "MSFT"]
        assert body_json["total_cost_basis"] == pytest.approx(91350.75)
        assert body_json["warnings"] == []
        handle = body_json["handle"]
        assert handle.startswith("pf_")
        # pf_YYYY-MM-DD_<hex6>
        parts = handle.split("_")
        assert len(parts) == 3
        assert len(parts[1]) == 10  # ISO date
        assert len(parts[2]) == 6

    def test_cost_basis_column_optional(self, client: TestClient) -> None:
        body = _csv(
            "ticker,shares",
            "NVDA,100",
            "TSLA,50",
        )
        r = _post_csv(client, body)
        assert r.status_code == 200, r.text
        out = r.json()
        assert out["row_count"] == 2
        # No cost_basis column → every row is a "missing" warning.
        assert len(out["warnings"]) == 2
        assert all("missing cost_basis, assumed market value" in w for w in out["warnings"])
        assert out["total_cost_basis"] == 0.0

    def test_header_is_case_insensitive(self, client: TestClient) -> None:
        body = _csv(
            "Ticker,Shares,Cost_Basis",
            "NVDA,100,12450.50",
        )
        r = _post_csv(client, body)
        assert r.status_code == 200, r.text
        assert r.json()["tickers"] == ["NVDA"]

    def test_ticker_is_uppercased(self, client: TestClient) -> None:
        body = _csv(
            "ticker,shares,cost_basis",
            "nvda,100,12450.50",
        )
        r = _post_csv(client, body)
        assert r.status_code == 200, r.text
        assert r.json()["tickers"] == ["NVDA"]

    def test_fractional_shares_allowed(self, client: TestClient) -> None:
        body = _csv(
            "ticker,shares,cost_basis",
            "NVDA,0.5,62.25",
        )
        r = _post_csv(client, body)
        assert r.status_code == 200, r.text
        out = r.json()
        assert out["row_count"] == 1
        assert out["total_cost_basis"] == pytest.approx(62.25)

    def test_blank_lines_are_skipped(self, client: TestClient) -> None:
        body = "ticker,shares,cost_basis\nNVDA,100,12450.50\n\nTSLA,50,15000\n"
        r = _post_csv(client, body)
        assert r.status_code == 200, r.text
        assert r.json()["row_count"] == 2

    def test_missing_cost_basis_per_row_produces_warning(self, client: TestClient) -> None:
        body = _csv(
            "ticker,shares,cost_basis",
            "NVDA,100,12450.50",
            "TSLA,50,15000",
            "AAPL,200,32000.00",
            "GOOG,10,1500.25",
            "MSFT,80,30400",
            "AMZN,5,",  # blank cost_basis on row 7
            "META,15,4500",
            "ORCL,12,",  # blank cost_basis on row 9
        )
        r = _post_csv(client, body)
        assert r.status_code == 200, r.text
        out = r.json()
        assert out["row_count"] == 8
        # The CSV has header on row 1 → row 7 and row 9 trigger warnings.
        assert any("row 7" in w for w in out["warnings"])
        assert any("row 9" in w for w in out["warnings"])


# ---- transport -----------------------------------------------------------


class TestTransports:
    @requires_multipart
    def test_multipart_upload_works(self, client: TestClient) -> None:
        body = _csv(
            "ticker,shares,cost_basis",
            "NVDA,100,12450.50",
            "TSLA,50,15000",
        )
        files = {"file": ("port.csv", body.encode("utf-8"), "text/csv")}
        r = client.post("/portfolio/import", files=files)
        assert r.status_code == 200, r.text
        assert r.json()["tickers"] == ["NVDA", "TSLA"]

    @requires_multipart
    def test_multipart_without_file_field_rejected(self, client: TestClient) -> None:
        r = client.post(
            "/portfolio/import",
            files={"not_file": ("x.csv", b"ticker,shares\nNVDA,1\n", "text/csv")},
        )
        assert r.status_code == 400
        assert "file" in r.json()["detail"].lower()

    def test_raw_text_csv_body_works(self, client: TestClient) -> None:
        body = _csv(
            "ticker,shares,cost_basis",
            "NVDA,100,12450.50",
        )
        r = client.post(
            "/portfolio/import",
            content=body,
            headers={"Content-Type": "text/csv"},
        )
        assert r.status_code == 200, r.text


# ---- validation failures -------------------------------------------------


class TestValidationFailures:
    def test_missing_required_column_returns_400(self, client: TestClient) -> None:
        body = _csv(
            "ticker,cost_basis",  # no 'shares' column
            "NVDA,12450.50",
        )
        r = _post_csv(client, body)
        assert r.status_code == 400
        detail = r.json()["detail"].lower()
        assert "shares" in detail
        assert "missing" in detail

    def test_negative_shares_returns_400(self, client: TestClient) -> None:
        body = _csv(
            "ticker,shares,cost_basis",
            "NVDA,-100,12450.50",
        )
        r = _post_csv(client, body)
        assert r.status_code == 400
        detail = r.json()["detail"]
        assert "row 2" in detail
        assert "shares" in detail
        assert "positive" in detail.lower()

    def test_zero_shares_returns_400(self, client: TestClient) -> None:
        body = _csv(
            "ticker,shares,cost_basis",
            "NVDA,0,12450.50",
        )
        r = _post_csv(client, body)
        assert r.status_code == 400
        assert "shares" in r.json()["detail"]

    def test_non_numeric_shares_returns_400(self, client: TestClient) -> None:
        body = _csv(
            "ticker,shares,cost_basis",
            "NVDA,not-a-number,12450.50",
        )
        r = _post_csv(client, body)
        assert r.status_code == 400
        detail = r.json()["detail"]
        assert "row 2" in detail
        assert "shares" in detail
        assert "not a number" in detail.lower()

    def test_negative_cost_basis_returns_400(self, client: TestClient) -> None:
        body = _csv(
            "ticker,shares,cost_basis",
            "NVDA,100,-12450.50",
        )
        r = _post_csv(client, body)
        assert r.status_code == 400
        detail = r.json()["detail"]
        assert "cost_basis" in detail
        assert "non-negative" in detail.lower()

    def test_bad_ticker_returns_400(self, client: TestClient) -> None:
        body = _csv(
            "ticker,shares,cost_basis",
            "TOOLONG,100,12450.50",
        )
        r = _post_csv(client, body)
        assert r.status_code == 400
        detail = r.json()["detail"]
        assert "ticker" in detail
        assert "row 2" in detail

    def test_ticker_with_digits_rejected(self, client: TestClient) -> None:
        body = _csv(
            "ticker,shares,cost_basis",
            "NV1,100,12450.50",
        )
        r = _post_csv(client, body)
        assert r.status_code == 400
        assert "ticker" in r.json()["detail"]

    def test_empty_body_returns_400(self, client: TestClient) -> None:
        r = client.post(
            "/portfolio/import",
            content="",
            headers={"Content-Type": "text/csv"},
        )
        assert r.status_code == 400
        assert "empty" in r.json()["detail"].lower()

    def test_header_only_returns_400(self, client: TestClient) -> None:
        body = _csv("ticker,shares,cost_basis")
        r = _post_csv(client, body)
        assert r.status_code == 400
        assert "no data" in r.json()["detail"].lower()

    def test_whitespace_only_body_returns_400(self, client: TestClient) -> None:
        r = client.post(
            "/portfolio/import",
            content="   \n   \n",
            headers={"Content-Type": "text/csv"},
        )
        assert r.status_code == 400

    def test_max_rows_plus_one_rejected(self, client: TestClient) -> None:
        lines = ["ticker,shares,cost_basis"]
        # Generate MAX_ROWS + 1 valid rows with unique letter-only tickers.
        for i in range(MAX_ROWS + 1):
            lines.append(f"{_letter_ticker(i)},10,100")
        r = _post_csv(client, "\n".join(lines) + "\n")
        assert r.status_code == 413
        assert "max" in r.json()["detail"].lower()


# ---- app.state store behavior --------------------------------------------


class TestAppStateStore:
    def test_handle_retrievable_via_app_state(self, app: FastAPI, client: TestClient) -> None:
        body = _csv(
            "ticker,shares,cost_basis",
            "NVDA,100,12450.50",
            "TSLA,50,15000",
        )
        r = _post_csv(client, body)
        handle = r.json()["handle"]
        store = app.state.portfolios
        assert isinstance(store, OrderedDict)
        assert handle in store
        pf = store[handle]
        assert isinstance(pf, Portfolio)
        assert pf.tickers == ["NVDA", "TSLA"]
        assert pf.total_cost_basis == pytest.approx(27450.50)

    def test_get_portfolio_helper_returns_handle(self, app: FastAPI, client: TestClient) -> None:
        r = _post_csv(client, _csv("ticker,shares,cost_basis", "NVDA,100,12450.50"))
        handle = r.json()["handle"]
        pf = get_portfolio(app.state, handle)
        assert pf.handle == handle
        assert pf.tickers == ["NVDA"]

    def test_get_portfolio_unknown_handle_raises(self, app: FastAPI, client: TestClient) -> None:
        # Trigger lazy initialisation of the store via one import.
        _post_csv(client, _csv("ticker,shares,cost_basis", "NVDA,1,1"))
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc:
            get_portfolio(app.state, "pf_does_not_exist")
        assert exc.value.status_code == 404

    def test_101st_handle_evicts_oldest_fifo(self, app: FastAPI, client: TestClient) -> None:
        handles: list[str] = []
        for i in range(MAX_HANDLES + 1):
            ticker = _letter_ticker(i)
            r = _post_csv(
                client,
                _csv("ticker,shares,cost_basis", f"{ticker},10,100"),
            )
            assert r.status_code == 200, r.text
            handles.append(r.json()["handle"])

        store = app.state.portfolios
        # Store stays capped at MAX_HANDLES.
        assert len(store) == MAX_HANDLES
        # Oldest (handles[0]) evicted; newest still present.
        assert handles[0] not in store
        assert handles[-1] in store
        # Insertion order preserved for the survivors.
        assert list(store.keys()) == handles[1:]

    def test_each_handle_is_unique(self, app: FastAPI, client: TestClient) -> None:
        seen: set[str] = set()
        for _ in range(20):
            r = _post_csv(client, _csv("ticker,shares,cost_basis", "NVDA,1,1"))
            assert r.status_code == 200
            seen.add(r.json()["handle"])
        # 20 unique handles in a row — 6 hex chars give 16.7M combinations,
        # collision probability is negligible.
        assert len(seen) == 20
