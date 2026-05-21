"""Tests for :mod:`pfm.arb.confirmed_store` — durable observed-arb store.

All tests are no-network and use ``tmp_path`` so nothing touches the real
``arbstuff/`` volume.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

from pfm.arb.confirmed_store import (
    DEFAULT_STORE_PATH,
    ENV_STORE_PATH,
    RECENT_PROFIT_MAXLEN,
    ConfirmedArb,
    ConfirmedArbStore,
)


@pytest.fixture
def store(tmp_path):
    """A fresh store backed by a file inside ``tmp_path``."""
    return ConfirmedArbStore(tmp_path / "confirmed_arbs.json")


def test_first_record_creates_entry(store):
    arb = store.record("k1:p1", kalshi_ticker="K1", poly_slug="p1", profit_pct=2.5)
    assert arb.count == 1
    assert arb.first_seen
    assert arb.first_seen == arb.last_seen
    assert arb.max_profit_pct == 2.5
    assert arb.recent_profit_pct == [2.5]
    assert store.get("k1:p1") is arb
    assert len(store) == 1


def test_repeat_record_bumps_count_and_updates(store):
    t0 = datetime(2026, 5, 1, tzinfo=UTC)
    t1 = datetime(2026, 5, 2, tzinfo=UTC)
    store.record("k1:p1", kalshi_ticker="K1", poly_slug="p1", profit_pct=2.0, now=t0)
    arb = store.record("k1:p1", kalshi_ticker="K1", poly_slug="p1", profit_pct=3.5, now=t1)
    assert arb.count == 2
    assert arb.first_seen == t0.isoformat()
    assert arb.last_seen == t1.isoformat()
    assert arb.max_profit_pct == 3.5
    assert arb.recent_profit_pct == [2.0, 3.5]


def test_max_profit_does_not_decrease(store):
    store.record("k:p", kalshi_ticker="K", poly_slug="p", profit_pct=5.0)
    arb = store.record("k:p", kalshi_ticker="K", poly_slug="p", profit_pct=1.0)
    assert arb.max_profit_pct == 5.0


def test_recent_profit_rolling_window_capped(store):
    for i in range(RECENT_PROFIT_MAXLEN + 10):
        store.record("k:p", kalshi_ticker="K", poly_slug="p", profit_pct=float(i))
    arb = store.get("k:p")
    assert len(arb.recent_profit_pct) == RECENT_PROFIT_MAXLEN
    # newest observations retained (last value is the final i)
    assert arb.recent_profit_pct[-1] == float(RECENT_PROFIT_MAXLEN + 10 - 1)
    assert arb.count == RECENT_PROFIT_MAXLEN + 10


def test_confirmed_filters_by_min_count(store):
    store.record("a", kalshi_ticker="A", poly_slug="a", profit_pct=1.0)
    for _ in range(3):
        store.record("b", kalshi_ticker="B", poly_slug="b", profit_pct=1.0)
    confirmed_default = store.confirmed()  # min_count=3
    assert [a.arb_key for a in confirmed_default] == ["b"]
    assert {a.arb_key for a in store.confirmed(min_count=1)} == {"a", "b"}
    assert store.confirmed(min_count=99) == []


def test_all_sorted_by_last_seen_desc(store):
    store.record(
        "old",
        kalshi_ticker="O",
        poly_slug="o",
        profit_pct=1.0,
        now=datetime(2026, 1, 1, tzinfo=UTC),
    )
    store.record(
        "new",
        kalshi_ticker="N",
        poly_slug="n",
        profit_pct=1.0,
        now=datetime(2026, 6, 1, tzinfo=UTC),
    )
    assert [a.arb_key for a in store.all()] == ["new", "old"]


def test_top_by_profit(store):
    store.record("low", kalshi_ticker="L", poly_slug="l", profit_pct=1.0)
    store.record("hi", kalshi_ticker="H", poly_slug="h", profit_pct=9.0)
    store.record("mid", kalshi_ticker="M", poly_slug="m", profit_pct=5.0)
    top2 = store.top_by_profit(2)
    assert [a.arb_key for a in top2] == ["hi", "mid"]
    assert store.top_by_profit(0) == []


def test_persistence_round_trips_across_instances(tmp_path):
    path = tmp_path / "store.json"
    s1 = ConfirmedArbStore(path)
    s1.record(
        "k:p",
        kalshi_ticker="K",
        poly_slug="p",
        profit_pct=4.0,
        volume=1234.0,
        confidence="high",
        extra={"venue": "kalshi"},
    )
    s1.record("k:p", kalshi_ticker="K", poly_slug="p", profit_pct=6.0)

    s2 = ConfirmedArbStore(path)
    arb = s2.get("k:p")
    assert arb is not None
    assert arb.count == 2
    assert arb.max_profit_pct == 6.0
    assert arb.volume == 1234.0
    assert arb.confidence == "high"
    assert arb.extra == {"venue": "kalshi"}
    assert arb.recent_profit_pct == [4.0, 6.0]


def test_missing_file_yields_empty(tmp_path):
    store = ConfirmedArbStore(tmp_path / "does_not_exist.json")
    assert store.all() == []
    assert len(store) == 0
    assert store.stats()["n_markets"] == 0


def test_corrupt_file_yields_empty(tmp_path):
    path = tmp_path / "corrupt.json"
    path.write_text("{ this is not valid json ]]]")
    store = ConfirmedArbStore(path)
    assert store.all() == []
    # store still usable after a corrupt load
    store.record("k:p", kalshi_ticker="K", poly_slug="p", profit_pct=1.0)
    assert len(store) == 1


def test_partial_corrupt_entries_skipped(tmp_path):
    path = tmp_path / "partial.json"
    payload = {
        "version": 1,
        "entries": [
            {
                "arb_key": "good",
                "kalshi_ticker": "K",
                "poly_slug": "p",
                "count": 2,
                "first_seen": "2026-01-01T00:00:00+00:00",
                "last_seen": "2026-01-02T00:00:00+00:00",
                "max_profit_pct": 3.0,
            },
            {"no_key": "bad"},
            "not even a dict",
        ],
    }
    path.write_text(json.dumps(payload))
    store = ConfirmedArbStore(path)
    keys = {a.arb_key for a in store.all()}
    assert keys == {"good"}


def test_atomic_write_does_not_clobber_on_reload(tmp_path):
    path = tmp_path / "store.json"
    s1 = ConfirmedArbStore(path)
    s1.record("a", kalshi_ticker="A", poly_slug="a", profit_pct=1.0)

    # A second instance writes a different key; first instance reload sees both
    # only after reload (independent caches, shared file).
    s2 = ConfirmedArbStore(path)
    s2.record("b", kalshi_ticker="B", poly_slug="b", profit_pct=2.0)

    # File on disk must contain at least the most-recent atomic write and never
    # be left half-written / unparseable.
    on_disk = json.loads(path.read_text())
    assert isinstance(on_disk, dict)
    assert "entries" in on_disk
    fresh = ConfirmedArbStore(path)
    assert fresh.get("b") is not None


def test_env_var_path_override(tmp_path, monkeypatch):
    target = tmp_path / "env_store.json"
    monkeypatch.setenv(ENV_STORE_PATH, str(target))
    store = ConfirmedArbStore()  # no explicit path
    assert store.path == target
    store.record("k:p", kalshi_ticker="K", poly_slug="p", profit_pct=1.0)
    assert target.exists()


def test_explicit_path_beats_env(tmp_path, monkeypatch):
    monkeypatch.setenv(ENV_STORE_PATH, str(tmp_path / "env.json"))
    explicit = tmp_path / "explicit.json"
    store = ConfirmedArbStore(explicit)
    assert store.path == explicit


def test_default_path_when_no_override(monkeypatch):
    monkeypatch.delenv(ENV_STORE_PATH, raising=False)
    store = ConfirmedArbStore()
    assert store.path == DEFAULT_STORE_PATH


def test_stats_correct(store):
    store.record(
        "a",
        kalshi_ticker="A",
        poly_slug="a",
        profit_pct=1.0,
        now=datetime(2026, 1, 1, tzinfo=UTC),
    )
    for _ in range(3):
        store.record(
            "b",
            kalshi_ticker="B",
            poly_slug="b",
            profit_pct=1.0,
            now=datetime(2026, 5, 1, tzinfo=UTC),
        )
    stats = store.stats()
    assert stats["total_seen"] == 4  # 1 + 3
    assert stats["n_confirmed"] == 1  # only "b" has count >= 3
    assert stats["n_markets"] == 2
    assert stats["oldest"] == datetime(2026, 1, 1, tzinfo=UTC).isoformat()
    assert stats["newest"] == datetime(2026, 5, 1, tzinfo=UTC).isoformat()


def test_unbounded_retains_many_keys(tmp_path):
    store = ConfirmedArbStore(tmp_path / "big.json")
    for i in range(1000):
        store.record(f"key-{i}", kalshi_ticker=f"K{i}", poly_slug=f"p{i}", profit_pct=float(i))
    assert len(store) == 1000
    # survives a reload from disk
    reloaded = ConfirmedArbStore(tmp_path / "big.json")
    assert len(reloaded) == 1000
    assert reloaded.stats()["n_markets"] == 1000


def test_prune_drops_old_entries(store):
    now = datetime(2026, 6, 1, tzinfo=UTC)
    store.record(
        "old",
        kalshi_ticker="O",
        poly_slug="o",
        profit_pct=1.0,
        now=now - timedelta(days=40),
    )
    store.record(
        "fresh",
        kalshi_ticker="F",
        poly_slug="f",
        profit_pct=1.0,
        now=now - timedelta(days=1),
    )
    removed = store.prune(max_age_days=30, now=now)
    assert removed == 1
    assert store.get("old") is None
    assert store.get("fresh") is not None


def test_prune_keeps_everything_when_young(store):
    now = datetime(2026, 6, 1, tzinfo=UTC)
    store.record("a", kalshi_ticker="A", poly_slug="a", profit_pct=1.0, now=now)
    assert store.prune(max_age_days=1, now=now) == 0
    assert len(store) == 1


def test_dataclass_to_from_dict_round_trip():
    arb = ConfirmedArb(
        arb_key="k:p",
        kalshi_ticker="K",
        poly_slug="p",
        count=5,
        first_seen="2026-01-01T00:00:00+00:00",
        last_seen="2026-02-01T00:00:00+00:00",
        max_profit_pct=7.5,
        recent_profit_pct=[1.0, 2.0, 7.5],
        confidence="high",
        volume=100.0,
        extra={"foo": "bar"},
    )
    restored = ConfirmedArb.from_dict(arb.to_dict())
    assert restored == arb


def test_naive_datetime_normalized_to_utc(store):
    naive = datetime(2026, 5, 1, 12, 0, 0)  # no tzinfo
    arb = store.record("k:p", kalshi_ticker="K", poly_slug="p", profit_pct=1.0, now=naive)
    assert arb.first_seen.endswith("+00:00")
