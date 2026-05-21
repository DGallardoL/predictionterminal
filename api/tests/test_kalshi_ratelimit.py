"""Tests for Kalshi rate-limit handling (throttling + 429 retry).

All HTTP is mocked via respx. Sleep is injected so tests run in
milliseconds — we verify the *durations* the client requests, not
real wall-clock.
"""

from __future__ import annotations

import time

import httpx
import pytest
import respx

from pfm.sources.kalshi import (
    KalshiClient,
    KalshiRateLimitError,
    _MinIntervalLimiter,
    _parse_retry_after,
)

BASE = KalshiClient.BASE_URL


def _empty_candles_payload() -> dict:
    return {"candlesticks": []}


def _success_market_payload(ticker: str = "KXFOO-26MAY") -> dict:
    return {
        "market": {
            "ticker": ticker,
            "event_ticker": ticker,
            "title": "Foo",
            "status": "active",
            "open_time": None,
            "close_time": None,
        }
    }


# ─────────────────────────── _MinIntervalLimiter ───────────────────────────


class TestMinIntervalLimiter:
    def test_zero_interval_is_no_op(self) -> None:
        lim = _MinIntervalLimiter(0.0)
        t0 = time.monotonic()
        for _ in range(100):
            lim.acquire()
        assert time.monotonic() - t0 < 0.1

    def test_first_acquire_is_immediate(self) -> None:
        lim = _MinIntervalLimiter(0.5)
        t0 = time.monotonic()
        lim.acquire()
        assert time.monotonic() - t0 < 0.05

    def test_second_acquire_waits(self) -> None:
        lim = _MinIntervalLimiter(0.10)
        lim.acquire()
        t0 = time.monotonic()
        lim.acquire()
        elapsed = time.monotonic() - t0
        # Should wait ~0.10s; allow generous slop on slow CI.
        assert elapsed >= 0.08, f"expected ≥0.08s wait, got {elapsed:.3f}s"
        assert elapsed < 0.30


# ───────────────────────────── _parse_retry_after ──────────────────────────


class TestParseRetryAfter:
    def test_none(self) -> None:
        assert _parse_retry_after(None) is None
        assert _parse_retry_after("") is None

    def test_integer_seconds(self) -> None:
        assert _parse_retry_after("5") == 5.0
        assert _parse_retry_after("0") == 0.0
        assert _parse_retry_after("  3.5  ") == 3.5

    def test_negative_clamped(self) -> None:
        assert _parse_retry_after("-1") == 0.0

    def test_garbage_returns_none(self) -> None:
        assert _parse_retry_after("not-a-date") is None

    def test_http_date(self) -> None:
        # Far-future date → big positive number
        v = _parse_retry_after("Wed, 01 Jan 2099 00:00:00 GMT")
        assert v is not None and v > 1e8


# ──────────────────────────── 429 retry behaviour ──────────────────────────


class TestRetryOn429:
    @respx.mock
    def test_single_429_then_success(self) -> None:
        slept: list[float] = []
        ticker = "KXFOO-26MAY"
        url = f"{BASE}/markets/{ticker}"
        # First 429, then 200.
        route = respx.get(url).mock(
            side_effect=[
                httpx.Response(429, headers={"Retry-After": "0"}),
                httpx.Response(200, json=_success_market_payload(ticker)),
            ]
        )
        c = KalshiClient(min_interval_s=0.0, max_retries=3, sleep=slept.append)
        m = c.get_market(ticker)
        assert m.ticker == ticker
        assert route.call_count == 2
        # Slept once, with Retry-After=0 → wait should be 0.0
        assert slept == [0.0]

    @respx.mock
    def test_429_with_integer_retry_after(self) -> None:
        slept: list[float] = []
        ticker = "KXFOO-26MAY"
        url = f"{BASE}/markets/{ticker}"
        respx.get(url).mock(
            side_effect=[
                httpx.Response(429, headers={"Retry-After": "2"}),
                httpx.Response(200, json=_success_market_payload(ticker)),
            ]
        )
        c = KalshiClient(min_interval_s=0.0, max_retries=3, sleep=slept.append)
        c.get_market(ticker)
        assert slept == [2.0]

    @respx.mock
    def test_429_no_retry_after_uses_exponential_backoff(self) -> None:
        slept: list[float] = []
        ticker = "KXFOO-26MAY"
        url = f"{BASE}/markets/{ticker}"
        respx.get(url).mock(
            side_effect=[
                httpx.Response(429),  # no Retry-After
                httpx.Response(429),  # no Retry-After
                httpx.Response(200, json=_success_market_payload(ticker)),
            ]
        )
        c = KalshiClient(
            min_interval_s=0.0,
            max_retries=4,
            backoff_base_s=1.0,
            backoff_cap_s=30.0,
            sleep=slept.append,
        )
        c.get_market(ticker)
        assert len(slept) == 2
        # Attempt 0 backoff: 1.0 ± 25%   → [0.75, 1.25]
        # Attempt 1 backoff: 2.0 ± 25%   → [1.50, 2.50]
        assert 0.75 <= slept[0] <= 1.25, slept[0]
        assert 1.50 <= slept[1] <= 2.50, slept[1]

    @respx.mock
    def test_429_exhaustion_raises_rate_limit_error(self) -> None:
        slept: list[float] = []
        ticker = "KXFOO-26MAY"
        url = f"{BASE}/markets/{ticker}"
        respx.get(url).mock(return_value=httpx.Response(429, headers={"Retry-After": "0"}))
        c = KalshiClient(min_interval_s=0.0, max_retries=2, sleep=slept.append)
        with pytest.raises(KalshiRateLimitError) as ei:
            c.get_market(ticker)
        # 1 initial attempt + 2 retries = 3 total; 2 sleeps in between.
        assert len(slept) == 2
        assert "Retry-After=" in str(ei.value)

    @respx.mock
    def test_retry_after_above_cap_falls_back_to_backoff(self) -> None:
        """If server says wait 600s but our cap is 5s, we use backoff math
        instead (otherwise a malicious / misconfigured upstream could pin us)."""
        slept: list[float] = []
        ticker = "KXFOO-26MAY"
        url = f"{BASE}/markets/{ticker}"
        respx.get(url).mock(
            side_effect=[
                httpx.Response(429, headers={"Retry-After": "600"}),
                httpx.Response(200, json=_success_market_payload(ticker)),
            ]
        )
        c = KalshiClient(
            min_interval_s=0.0,
            max_retries=3,
            backoff_base_s=1.0,
            backoff_cap_s=5.0,
            sleep=slept.append,
        )
        c.get_market(ticker)
        # Should fall back to backoff (~1s ± 25%), capped at 5s.
        assert 0.75 <= slept[0] <= 5.0


# ─────────────────────────── candlesticks integration ──────────────────────


class TestCandlesticksUseRateLimitWrapper:
    @respx.mock
    def test_candlesticks_429_then_200(self) -> None:
        slept: list[float] = []
        ticker = "KXFOO-26MAY"
        series = "KXFOO"
        url = f"{BASE}/series/{series}/markets/{ticker}/candlesticks"
        respx.get(url).mock(
            side_effect=[
                httpx.Response(429, headers={"Retry-After": "0"}),
                httpx.Response(200, json=_empty_candles_payload()),
            ]
        )
        c = KalshiClient(min_interval_s=0.0, max_retries=3, sleep=slept.append)
        df = c.get_candlesticks(ticker, start_ts=1, end_ts=100)
        assert df.empty
        assert slept == [0.0]

    @respx.mock
    def test_400_does_not_retry(self) -> None:
        """Non-429 errors should NOT trigger the retry loop."""
        slept: list[float] = []
        ticker = "KXFOO-26MAY"
        url = f"{BASE}/markets/{ticker}"
        route = respx.get(url).mock(return_value=httpx.Response(400))
        c = KalshiClient(min_interval_s=0.0, max_retries=3, sleep=slept.append)
        with pytest.raises(httpx.HTTPStatusError):
            c.get_market(ticker)
        assert route.call_count == 1
        assert slept == []


# ────────────────────────── min-interval throttling ────────────────────────


class TestThrottlingDuringRequests:
    @respx.mock
    def test_two_requests_respect_min_interval(self) -> None:
        # Two distinct tickers — the 2026-05-15 hardening pass added a
        # process-local 1h cache on get_market(ticker), so identical-ticker
        # calls now short-circuit the upstream limiter. Throttling is still
        # a real behaviour we want to verify across consecutive cold-paths.
        t1 = "KXFOO-26MAY"
        t2 = "KXFOO-26JUN"
        for t in (t1, t2):
            respx.get(f"{BASE}/markets/{t}").mock(
                return_value=httpx.Response(200, json=_success_market_payload(t))
            )
        c = KalshiClient(min_interval_s=0.10, max_retries=0, sleep=time.sleep)
        t0 = time.monotonic()
        c.get_market(t1)
        c.get_market(t2)
        elapsed = time.monotonic() - t0
        # Two requests with 0.10s min interval → at least ~0.10s elapsed.
        assert elapsed >= 0.08, f"expected ≥0.08s, got {elapsed:.3f}s"


# ──────────── 2026-05-15: process-local get_market cache (1 h TTL) ──────────


class TestMarketMetadataCache:
    @respx.mock
    def test_second_get_market_call_hits_cache(self) -> None:
        """Identical ticker fetches share one upstream call.

        Cache key is ticker → KalshiMarket. The cache is bounded and
        cleared between tests via the conftest autouse fixture, so this
        test sees a clean cold path.
        """
        ticker = "KXCACHE-26MAY"
        route = respx.get(f"{BASE}/markets/{ticker}").mock(
            return_value=httpx.Response(200, json=_success_market_payload(ticker))
        )
        c = KalshiClient(min_interval_s=0.0, max_retries=0, sleep=lambda *_a: None)
        m1 = c.get_market(ticker)
        m2 = c.get_market(ticker)
        assert m1 == m2
        assert route.call_count == 1, (
            f"expected cache hit on second call; got {route.call_count} upstream calls"
        )
