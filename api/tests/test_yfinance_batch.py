"""Tests for ``pfm.sources.yfinance_batch``.

All tests monkeypatch ``yfinance.download`` so no network calls happen.
Run with ``--noconftest`` to skip the project conftest (which imports the
full FastAPI app and is far too heavy for an isolated unit test):

    cd api && PYTHONPATH=src .venv/bin/python -m pytest \\
        tests/test_yfinance_batch.py -q --noconftest
"""

from __future__ import annotations

import asyncio
import threading
import time
from datetime import date

import pandas as pd
import pytest

from pfm.sources import yfinance_batch as ybatch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_df(ticker: str, n: int = 5) -> pd.DataFrame:
    """Build a fake yfinance-style closes DataFrame."""
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {
            "Open": [100.0 + i for i in range(n)],
            "High": [101.0 + i for i in range(n)],
            "Low": [99.0 + i for i in range(n)],
            "Close": [100.5 + i for i in range(n)],
            "Volume": [1_000_000 + i * 100 for i in range(n)],
        },
        index=idx,
    )


@pytest.fixture(autouse=True)
def _patch_jitter(monkeypatch):
    """Disable jitter sleeps in tests so the suite is fast (default ms is 50–200)."""
    monkeypatch.setattr(ybatch, "_JITTER_MS_MIN", 0)
    monkeypatch.setattr(ybatch, "_JITTER_MS_MAX", 0)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_empty_input_returns_empty_dict(monkeypatch):
    """An empty ticker list should short-circuit (no downloads called)."""
    calls: list[str] = []

    def fake_download(ticker, **kwargs):
        calls.append(ticker)
        return _make_df(ticker)

    monkeypatch.setattr(ybatch.yf, "download", fake_download)

    result = ybatch.fetch_tickers_batch([], start="2024-01-01")
    assert result == {}
    assert calls == []


def test_batch_of_four_returns_four_dataframes(monkeypatch):
    """4 tickers in → 4 DataFrames out, each non-empty and keyed by ticker."""
    tickers = ["AAPL", "MSFT", "GOOG", "NVDA"]

    def fake_download(ticker, **kwargs):
        return _make_df(ticker)

    monkeypatch.setattr(ybatch.yf, "download", fake_download)

    result = ybatch.fetch_tickers_batch(tickers, start="2024-01-01", workers=4)

    assert set(result.keys()) == set(tickers)
    for t in tickers:
        assert isinstance(result[t], pd.DataFrame)
        assert not result[t].empty
        assert len(result[t]) == 5


def test_one_ticker_failing_others_succeed(monkeypatch, caplog):
    """A failing ticker maps to empty DF; siblings unaffected; warning logged."""
    tickers = ["AAPL", "BADTICK", "MSFT"]

    def fake_download(ticker, **kwargs):
        if ticker == "BADTICK":
            raise ValueError("simulated yfinance failure")
        return _make_df(ticker)

    monkeypatch.setattr(ybatch.yf, "download", fake_download)

    with caplog.at_level("WARNING", logger="pfm.sources.yfinance_batch"):
        result = ybatch.fetch_tickers_batch(tickers, start="2024-01-01", workers=3)

    assert set(result.keys()) == set(tickers)
    assert result["BADTICK"].empty
    assert not result["AAPL"].empty
    assert not result["MSFT"].empty

    # Warning must mention the ticker and the exception type
    msg = " ".join(r.message for r in caplog.records)
    assert "BADTICK" in msg
    assert "ValueError" in msg


def test_date_parsing_accepts_str_and_date(monkeypatch):
    """Both ``date`` and ISO string should be accepted for start/end."""
    captured: dict[str, object] = {}

    def fake_download(ticker, **kwargs):
        captured["start"] = kwargs.get("start")
        captured["end"] = kwargs.get("end")
        return _make_df(ticker)

    monkeypatch.setattr(ybatch.yf, "download", fake_download)

    # String form
    ybatch.fetch_tickers_batch(["AAPL"], start="2024-03-15", end="2024-04-15")
    assert captured["start"] == date(2024, 3, 15)
    assert captured["end"] == date(2024, 4, 15)

    # date form
    ybatch.fetch_tickers_batch(["AAPL"], start=date(2023, 1, 2), end=date(2023, 6, 30))
    assert captured["start"] == date(2023, 1, 2)
    assert captured["end"] == date(2023, 6, 30)


def test_date_parsing_end_optional(monkeypatch):
    """``end=None`` should omit the kwarg entirely (yfinance default = today)."""
    captured: dict[str, object] = {}

    def fake_download(ticker, **kwargs):
        captured.update(kwargs)
        return _make_df(ticker)

    monkeypatch.setattr(ybatch.yf, "download", fake_download)
    ybatch.fetch_tickers_batch(["AAPL"], start="2024-01-01")

    assert "end" not in captured


def test_date_parsing_rejects_garbage():
    """A non-date, non-str ``start`` raises TypeError."""
    with pytest.raises(TypeError):
        ybatch.fetch_tickers_batch(["AAPL"], start=12345)  # type: ignore[arg-type]


def test_concurrency_bounded_to_workers(monkeypatch):
    """With workers=2 and 10 tickers, max-in-flight must never exceed 2."""
    barrier_lock = threading.Lock()
    in_flight = {"current": 0, "max": 0}

    def slow_download(ticker, **kwargs):
        with barrier_lock:
            in_flight["current"] += 1
            in_flight["max"] = max(in_flight["max"], in_flight["current"])
        # Hold each call for long enough that the executor would
        # certainly start more if the semaphore weren't bounding it.
        time.sleep(0.05)
        with barrier_lock:
            in_flight["current"] -= 1
        return _make_df(ticker)

    monkeypatch.setattr(ybatch.yf, "download", slow_download)

    tickers = [f"T{i}" for i in range(10)]
    result = ybatch.fetch_tickers_batch(tickers, start="2024-01-01", workers=2)

    assert len(result) == 10
    assert in_flight["max"] <= 2, f"max in-flight was {in_flight['max']}, exceeds workers=2"

    # And the module-level introspection counter should match
    assert ybatch._last_max_in_flight <= 2


def test_concurrency_actually_parallel(monkeypatch):
    """With workers=4 and 4 slow tickers, wall time must be ~1 slot, not 4."""
    call_starts: list[float] = []
    call_lock = threading.Lock()

    def slow_download(ticker, **kwargs):
        with call_lock:
            call_starts.append(time.monotonic())
        time.sleep(0.2)
        return _make_df(ticker)

    monkeypatch.setattr(ybatch.yf, "download", slow_download)

    t0 = time.monotonic()
    result = ybatch.fetch_tickers_batch(["A", "B", "C", "D"], start="2024-01-01", workers=4)
    elapsed = time.monotonic() - t0

    assert len(result) == 4
    # Serial wall-time would be 4 * 0.2 = 0.8 s.
    # With workers=4 we expect close to 0.2 s. Allow generous headroom.
    assert elapsed < 0.6, f"elapsed={elapsed:.3f}s suggests serial execution"

    # All 4 calls should have started within a small window of each other
    span = max(call_starts) - min(call_starts)
    assert span < 0.15, f"call-start span {span:.3f}s too wide for parallel exec"


def test_duplicate_tickers_deduplicated(monkeypatch):
    """Duplicate tickers in input should be fetched once."""
    calls: list[str] = []
    lock = threading.Lock()

    def fake_download(ticker, **kwargs):
        with lock:
            calls.append(ticker)
        return _make_df(ticker)

    monkeypatch.setattr(ybatch.yf, "download", fake_download)
    result = ybatch.fetch_tickers_batch(["AAPL", "MSFT", "AAPL", "AAPL"], start="2024-01-01")

    assert set(result.keys()) == {"AAPL", "MSFT"}
    assert sorted(calls) == ["AAPL", "MSFT"]


def test_yfinance_returns_none_handled(monkeypatch, caplog):
    """If yf.download returns None, ticker gets empty DF and a warning."""

    def fake_download(ticker, **kwargs):
        return None

    monkeypatch.setattr(ybatch.yf, "download", fake_download)
    with caplog.at_level("WARNING", logger="pfm.sources.yfinance_batch"):
        result = ybatch.fetch_tickers_batch(["AAPL"], start="2024-01-01")

    assert result["AAPL"].empty
    assert any("AAPL" in r.message for r in caplog.records)


def test_multiindex_flattened(monkeypatch):
    """If yfinance hands back a MultiIndex column frame, we flatten it."""
    ticker = "AAPL"

    def fake_download(t, **kwargs):
        # Build a MultiIndex-columns frame like yfinance does for list input
        base = _make_df(t)
        base.columns = pd.MultiIndex.from_product([base.columns, [t]])
        return base

    monkeypatch.setattr(ybatch.yf, "download", fake_download)
    result = ybatch.fetch_tickers_batch([ticker], start="2024-01-01")

    df = result[ticker]
    assert not df.empty
    assert not isinstance(df.columns, pd.MultiIndex)
    assert "Close" in df.columns


def test_interval_passed_through(monkeypatch):
    """Custom interval should reach yf.download."""
    captured: dict[str, object] = {}

    def fake_download(ticker, **kwargs):
        captured.update(kwargs)
        return _make_df(ticker)

    monkeypatch.setattr(ybatch.yf, "download", fake_download)
    ybatch.fetch_tickers_batch(["AAPL"], start="2024-01-01", interval="1h")
    assert captured["interval"] == "1h"


def test_threads_false_always_passed(monkeypatch):
    """We MUST pass threads=False so yfinance doesn't multiply our concurrency."""
    captured: list[dict[str, object]] = []

    def fake_download(ticker, **kwargs):
        captured.append(dict(kwargs))
        return _make_df(ticker)

    monkeypatch.setattr(ybatch.yf, "download", fake_download)
    ybatch.fetch_tickers_batch(["AAPL", "MSFT"], start="2024-01-01", workers=2)

    assert len(captured) == 2
    for kwargs in captured:
        assert kwargs.get("threads") is False


def test_async_wrapper_returns_same_dict(monkeypatch):
    """The async wrapper must return the same dict as the sync version."""

    def fake_download(ticker, **kwargs):
        return _make_df(ticker)

    monkeypatch.setattr(ybatch.yf, "download", fake_download)

    tickers = ["AAPL", "MSFT", "GOOG"]
    sync_result = ybatch.fetch_tickers_batch(tickers, start="2024-01-01")
    async_result = asyncio.run(ybatch.fetch_tickers_batch_async(tickers, start="2024-01-01"))

    assert set(sync_result.keys()) == set(async_result.keys())
    for t in tickers:
        pd.testing.assert_frame_equal(sync_result[t], async_result[t])


def test_workers_clamped_to_at_least_one(monkeypatch):
    """workers=0 or negative should be clamped to 1 (don't deadlock)."""

    def fake_download(ticker, **kwargs):
        return _make_df(ticker)

    monkeypatch.setattr(ybatch.yf, "download", fake_download)

    result = ybatch.fetch_tickers_batch(["AAPL"], start="2024-01-01", workers=0)
    assert "AAPL" in result
    assert not result["AAPL"].empty


def test_warning_log_includes_exception_type(monkeypatch, caplog):
    """Structured warning must include the exception type name."""

    def fake_download(ticker, **kwargs):
        raise ConnectionError("timeout from yahoo")

    monkeypatch.setattr(ybatch.yf, "download", fake_download)
    with caplog.at_level("WARNING", logger="pfm.sources.yfinance_batch"):
        result = ybatch.fetch_tickers_batch(["AAPL"], start="2024-01-01")

    assert result["AAPL"].empty
    found = False
    for rec in caplog.records:
        if "ConnectionError" in rec.message and "AAPL" in rec.message:
            found = True
            break
    assert found, "warning log must include exception type and ticker"
