"""Tests for the per-channel throttle (digest mode)."""

from __future__ import annotations

import asyncio

import pytest

from pfm.alerts.channels import (
    _channel_key,
    fanout,
    flush_pending_digests,
)
from pfm.alerts.storage import AlertStore


class _Recorder:
    """Mock channel that records every call."""

    def __init__(self, label: str = "rec") -> None:
        self.type = label
        self.calls: list[tuple[dict, str]] = []

    async def send(self, event: dict, target: str) -> dict:
        self.calls.append((event, target))
        return {
            "channel": self.type,
            "target": target,
            "ok": True,
            "status_code": 200,
            "error": None,
            "ts": 0.0,
        }


@pytest.fixture()
def store() -> AlertStore:
    return AlertStore(":memory:")


def _event(i: int, user_id: str = "u1") -> dict:
    return {
        "event_id": f"evt_{i}",
        "rule_id": "rule_1",
        "user_id": user_id,
        "kind": "price_cross",
        "fired_at": float(i),
        "payload": {
            "rule_name": f"r{i}",
            "rule_kind": "price_cross",
            "message": f"event {i}",
        },
        "delivered": [],
        "acked": False,
    }


def test_throttle_allows_first_10_buffers_eleventh(store: AlertStore) -> None:
    rec = _Recorder(label="rec")
    registry = {"rec": rec}
    channels = [{"type": "rec", "target": "tgt-A", "enabled": True}]

    delivered_ok = 0
    buffered = 0
    for i in range(11):
        results = asyncio.run(
            fanout(
                _event(i),
                channels,
                registry,
                throttle_store=store,
                max_per_minute=10,
            )
        )
        assert len(results) == 1
        r = results[0]
        if r.get("status_code") == 202 and "throttled-buffered" in (r.get("error") or ""):
            buffered += 1
        else:
            delivered_ok += 1

    assert delivered_ok == 10
    assert buffered == 1
    # Mock channel only saw 10 actual sends.
    assert len(rec.calls) == 10


def test_throttle_flush_emits_single_digest(store: AlertStore) -> None:
    """11 events → 10 delivered + 1 buffered → flush yields one digest."""
    rec = _Recorder(label="rec")
    registry = {"rec": rec}
    channels = [{"type": "rec", "target": "tgt-A", "enabled": True}]

    for i in range(11):
        asyncio.run(
            fanout(
                _event(i),
                channels,
                registry,
                throttle_store=store,
                max_per_minute=10,
            )
        )

    pre_calls = len(rec.calls)
    assert pre_calls == 10

    # Buffer non-empty BEFORE the quiet window passes → no flush.
    flushed_now = asyncio.run(
        flush_pending_digests(
            store,
            quiet_seconds=60.0,
            registry=registry,
            now=0.0,  # virtually no time elapsed since last_event_at
        )
    )
    assert flushed_now == [], "should not flush within quiet window"

    # Force the quiet window to elapse via large `now`.
    flushed_results = asyncio.run(
        flush_pending_digests(
            store,
            quiet_seconds=60.0,
            registry=registry,
            now=1e12,
        )
    )
    assert len(flushed_results) == 1
    digest = flushed_results[0]
    assert digest.get("digest") is True
    assert digest.get("count") == 1
    # Mock channel saw 10 + 1 (the digest) deliveries.
    assert len(rec.calls) == pre_calls + 1
    digest_event, _target = rec.calls[-1]
    assert digest_event["kind"] == "digest"
    assert "buffered" in digest_event["payload"]["summary"]


def test_throttle_resets_on_new_minute(store: AlertStore) -> None:
    """Counter resets when window_start is older than 60s."""
    ckey = _channel_key("u1", "rec", "tgt-A")
    # Fill the bucket at t=0.
    for i in range(10):
        ok, _ = store.throttle_check_and_record(ckey, _event(i), max_per_minute=10, now=0.0)
        assert ok
    # 11th at t=0 → buffered.
    ok, _ = store.throttle_check_and_record(ckey, _event(99), max_per_minute=10, now=0.0)
    assert ok is False

    # 12th at t=70 (new window) → allowed.
    ok2, count = store.throttle_check_and_record(ckey, _event(100), max_per_minute=10, now=70.0)
    assert ok2 is True
    assert count == 1


def test_throttle_independent_buckets_per_channel_target(
    store: AlertStore,
) -> None:
    """Different (user/channel/target) combos do not share quota."""
    rec = _Recorder(label="rec")
    registry = {"rec": rec}
    channels_a = [{"type": "rec", "target": "tgt-A", "enabled": True}]
    channels_b = [{"type": "rec", "target": "tgt-B", "enabled": True}]

    for i in range(10):
        asyncio.run(fanout(_event(i), channels_a, registry, throttle_store=store))
    for i in range(10):
        asyncio.run(fanout(_event(100 + i), channels_b, registry, throttle_store=store))

    # Both channels delivered all 10 (independent buckets).
    targets = [t for _, t in rec.calls]
    assert targets.count("tgt-A") == 10
    assert targets.count("tgt-B") == 10


def test_throttle_no_store_means_no_throttling(store: AlertStore) -> None:
    """Without throttle_store the legacy fanout path delivers everything."""
    rec = _Recorder(label="rec")
    registry = {"rec": rec}
    channels = [{"type": "rec", "target": "tgt-A", "enabled": True}]
    for i in range(20):
        asyncio.run(fanout(_event(i), channels, registry))
    assert len(rec.calls) == 20
