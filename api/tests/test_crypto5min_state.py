"""Tests for the rolling spot buffer / window-anchor resolver."""

from __future__ import annotations

import pytest

from pfm.crypto5min.state import CryptoFiveMinState, get_state, reset_state


def test_record_and_latest() -> None:
    s = CryptoFiveMinState()
    s.record_spot("BTCUSDT", 1_700_000_000.0, 60_000.0)
    s.record_spot("BTCUSDT", 1_700_000_001.0, 60_010.0)
    latest = s.latest("BTCUSDT")
    assert latest == (1_700_000_001.0, 60_010.0)


def test_record_is_case_insensitive() -> None:
    s = CryptoFiveMinState()
    s.record_spot("btcusdt", 1_700_000_000.0, 60_000.0)
    assert s.latest("BTCUSDT") == (1_700_000_000.0, 60_000.0)


def test_record_drops_out_of_order_samples() -> None:
    s = CryptoFiveMinState()
    s.record_spot("BTC", 100.0, 50.0)
    s.record_spot("BTC", 99.0, 51.0)  # earlier — must be dropped
    s.record_spot("BTC", 101.0, 52.0)
    assert s.n_samples("BTC") == 2
    assert s.latest("BTC") == (101.0, 52.0)


def test_record_drops_non_positive_prices() -> None:
    s = CryptoFiveMinState()
    s.record_spot("BTC", 100.0, 0.0)
    s.record_spot("BTC", 101.0, -5.0)
    assert s.n_samples("BTC") == 0


def test_buffer_is_bounded() -> None:
    s = CryptoFiveMinState(max_samples=5)
    for i in range(10):
        s.record_spot("X", float(i), 100.0 + i)
    assert s.n_samples("X") == 5
    # Oldest retained sample is i=5
    assert s.latest("X") == (9.0, 109.0)


def test_anchor_returns_none_when_empty() -> None:
    s = CryptoFiveMinState()
    assert s.anchor("BTC", period_seconds=300) is None


def test_anchor_picks_sample_at_window_start() -> None:
    s = CryptoFiveMinState()
    # Window starts at 1500 (period=300). Samples right before should be the anchor.
    for ts, mid in [(1499.0, 60_000.0), (1500.0, 60_005.0), (1700.0, 60_050.0)]:
        s.record_spot("BTCUSDT", ts, mid)
    anchor = s.anchor("BTCUSDT", period_seconds=300, now_unix=1700.0)
    assert anchor is not None
    # sample at exactly 1500 is the boundary anchor
    assert anchor.spot_at_start == 60_005.0
    assert anchor.spot_now == 60_050.0
    assert anchor.start_unix == 1500
    assert anchor.end_unix == 1800
    assert anchor.seconds_remaining == pytest.approx(100.0)


def test_anchor_falls_back_to_earliest_when_boot_mid_window() -> None:
    s = CryptoFiveMinState()
    # Now = 1750; window start = 1500. No sample <= 1500.
    s.record_spot("BTC", 1600.0, 60_000.0)
    s.record_spot("BTC", 1700.0, 60_010.0)
    anchor = s.anchor("BTC", period_seconds=300, now_unix=1750.0)
    assert anchor is not None
    assert anchor.spot_at_start == 60_000.0  # earliest fallback


def test_anchor_seconds_remaining_zero_at_boundary() -> None:
    s = CryptoFiveMinState()
    s.record_spot("BTC", 1799.0, 60_000.0)
    s.record_spot("BTC", 1800.0, 60_010.0)
    anchor = s.anchor("BTC", period_seconds=300, now_unix=1800.0)
    assert anchor is not None
    # at exactly 1800 we're at the *next* window's start, so seconds_remaining
    # is for THAT window, not the previous one.
    assert anchor.seconds_remaining == pytest.approx(300.0)


def test_anchor_period_must_be_positive() -> None:
    s = CryptoFiveMinState()
    s.record_spot("BTC", 1.0, 60_000.0)
    with pytest.raises(ValueError):
        s.anchor("BTC", period_seconds=0)


def test_snapshot_returns_n_samples_zero_for_unknown() -> None:
    s = CryptoFiveMinState()
    snap = s.snapshot("NEVER_SEEN")
    assert snap["n_samples"] == 0


def test_snapshot_returns_diagnostics() -> None:
    s = CryptoFiveMinState()
    s.record_spot("BTC", 100.0, 60_000.0)
    s.record_spot("BTC", 200.0, 60_300.0)
    snap = s.snapshot("BTC")
    assert snap["n_samples"] == 2
    assert snap["span_seconds"] == 100.0
    assert snap["drift_pct"] == pytest.approx(0.005, rel=1e-6)


def test_singleton_is_stable() -> None:
    a = get_state()
    b = get_state()
    assert a is b


def test_reset_state_replaces_singleton() -> None:
    a = get_state()
    a.record_spot("X", 1.0, 99.0)
    reset_state()
    b = get_state()
    assert a is not b
    assert b.n_samples("X") == 0


def test_clear_drops_all_samples() -> None:
    s = CryptoFiveMinState()
    s.record_spot("A", 1.0, 1.0)
    s.record_spot("B", 1.0, 1.0)
    s.clear()
    assert s.n_samples("A") == 0
    assert s.n_samples("B") == 0
