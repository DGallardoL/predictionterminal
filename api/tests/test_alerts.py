"""Tests for pfm.alerts: schemas, storage, engine, channels, and router."""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import TypeAdapter, ValidationError

from pfm.alerts.channels import (
    InAppChannel,
    SlackChannel,
    WebhookChannel,
    fanout,
)
from pfm.alerts.engine import evaluate_all
from pfm.alerts.router import get_alert_store
from pfm.alerts.router import router as alerts_router
from pfm.alerts.schemas import (
    AlertRule,
    ChannelRef,
    PriceChangePctRule,
    PriceCrossRule,
    VolumeSpikeRule,
    ZScorePairRule,
)
from pfm.alerts.storage import AlertStore

# ---------------------------------------------------------------- schemas


def test_schema_roundtrip_each_kind():
    """Each rule kind must (de)serialize through the discriminated union."""
    adapter = TypeAdapter(AlertRule)

    rules = [
        PriceCrossRule(
            user_id="u1",
            name="r1",
            slug="aapl-up",
            op=">",
            threshold=0.5,
            channels=[ChannelRef(type="inapp", target="u1")],
        ),
        PriceChangePctRule(
            user_id="u1",
            name="r2",
            slug="aapl-up",
            window="4h",
            pct_abs=0.1,
            channels=[ChannelRef(type="slack", target="https://example/s")],
        ),
        ZScorePairRule(
            user_id="u1",
            name="r3",
            slug_a="a",
            slug_b="b",
            beta=1.2,
            window=10,
            z_threshold=2.5,
            channels=[],
        ),
        VolumeSpikeRule(
            user_id="u1",
            name="r4",
            slug="x",
            lookback_days=14,
            n_sigma=3.0,
            channels=[],
        ),
    ]
    for r in rules:
        dumped = r.model_dump(mode="json")
        rebuilt = adapter.validate_python(dumped)
        assert rebuilt.kind == r.kind
        assert rebuilt.name == r.name


def test_schema_rejects_bad_threshold():
    with pytest.raises(ValidationError):
        PriceCrossRule(
            user_id="u",
            name="bad",
            slug="x",
            op=">",
            threshold=1.5,  # > 1
            channels=[],
        )


# ---------------------------------------------------------------- storage


@pytest.fixture()
def store() -> AlertStore:
    return AlertStore(":memory:")


def _mk_cross(user="u1", name="cross", threshold=0.5, slug="aapl") -> PriceCrossRule:
    return PriceCrossRule(
        user_id=user,
        name=name,
        slug=slug,
        op=">",
        threshold=threshold,
        cooldown_seconds=60,
        channels=[ChannelRef(type="inapp", target=user)],
    )


def test_storage_crud(store: AlertStore):
    rid = store.save_rule(_mk_cross())
    assert rid.startswith("rule_")
    rows = store.list_rules("u1")
    assert len(rows) == 1
    assert rows[0]["id"] == rid
    one = store.get_rule(rid)
    assert one is not None
    assert one["kind"] == "price_cross"
    assert store.delete_rule(rid)
    assert store.get_rule(rid) is None
    assert not store.delete_rule(rid)


def test_storage_patch(store: AlertStore):
    rid = store.save_rule(_mk_cross(name="orig"))
    upd = store.patch_rule(rid, name="renamed", cooldown_seconds=120)
    assert upd is not None
    assert upd["name"] == "renamed"
    assert upd["cooldown_seconds"] == 120


def test_storage_events(store: AlertStore):
    rid = store.save_rule(_mk_cross())
    ev = store.record_event(rid, {"price": 0.6, "rule_kind": "price_cross"})
    assert ev["acked"] is False
    evs = store.list_events("u1")
    assert len(evs) == 1
    assert store.ack_event(ev["event_id"])
    unacked = store.list_events("u1", unack_only=True)
    assert unacked == []


# ---------------------------------------------------------------- engine


def test_engine_price_cross_fires_then_cooldown(store: AlertStore):
    rule = _mk_cross(threshold=0.5)
    store.save_rule(rule)
    ctx = {"snapshot": {"aapl": 0.55}, "now": 1000.0}

    fired = evaluate_all(store, ctx, user_id="u1")
    assert len(fired) == 1
    assert fired[0]["payload"]["price"] == 0.55

    # Second eval immediately: cooldown blocks even though price still > thr
    # (and last_state == 'fired' would block edge anyway).
    ctx2 = {"snapshot": {"aapl": 0.6}, "now": 1010.0}
    fired2 = evaluate_all(store, ctx2, user_id="u1")
    assert fired2 == []


def test_engine_edge_trigger_no_refire_until_rearm(store: AlertStore):
    """Once fired, must dip below (threshold - hysteresis) to re-arm."""
    rule = PriceCrossRule(
        user_id="u1",
        name="edge",
        slug="x",
        op=">",
        threshold=0.5,
        hysteresis=0.05,
        cooldown_seconds=0,  # disable cooldown so we test edge purely
        channels=[],
    )
    rid = store.save_rule(rule)

    fired_a = evaluate_all(store, {"snapshot": {"x": 0.6}, "now": 1.0}, "u1")
    assert len(fired_a) == 1

    # Still above thr → no re-fire even with cooldown=0 because last_state='fired'.
    fired_b = evaluate_all(store, {"snapshot": {"x": 0.7}, "now": 2.0}, "u1")
    assert fired_b == []

    # Dip below threshold but NOT below threshold - hysteresis: still no re-arm.
    fired_c = evaluate_all(store, {"snapshot": {"x": 0.48}, "now": 3.0}, "u1")
    assert fired_c == []
    after = store.get_rule(rid)
    assert after["last_state"] == "fired"

    # Dip below hysteresis band → re-arm.
    fired_d = evaluate_all(store, {"snapshot": {"x": 0.40}, "now": 4.0}, "u1")
    assert fired_d == []
    after2 = store.get_rule(rid)
    assert after2["last_state"] == "armed"

    # Now cross again → fire.
    fired_e = evaluate_all(store, {"snapshot": {"x": 0.6}, "now": 5.0}, "u1")
    assert len(fired_e) == 1


def test_engine_zscore_pair_fires(store: AlertStore):
    rule = ZScorePairRule(
        user_id="u1",
        name="z",
        slug_a="a",
        slug_b="b",
        beta=1.0,
        window=20,
        z_threshold=2.0,
        cooldown_seconds=0,
        channels=[],
    )
    store.save_rule(rule)
    # 19 points where spread ≈ 0, then 1 huge spread.
    ha = [(float(i), 0.5) for i in range(19)] + [(20.0, 0.9)]
    hb = [(float(i), 0.5) for i in range(20)]
    fired = evaluate_all(store, {"history": {"a": ha, "b": hb}, "now": 100.0}, "u1")
    assert len(fired) == 1
    assert fired[0]["payload"]["rule_kind"] == "zscore_pair"


def test_engine_volume_spike_fires(store: AlertStore):
    rule = VolumeSpikeRule(
        user_id="u1",
        name="v",
        slug="x",
        lookback_days=7,
        n_sigma=2.0,
        cooldown_seconds=0,
        channels=[],
    )
    store.save_rule(rule)
    # 8 lookback days with mild variation (mean~100, sd>0), then a huge spike.
    base = [95.0, 105.0, 100.0, 102.0, 98.0, 101.0, 99.0, 100.0]
    vol_hist = [(float(i), v) for i, v in enumerate(base)] + [(8.0, 1000.0)]
    fired = evaluate_all(store, {"volume_history": {"x": vol_hist}, "now": 200.0}, "u1")
    assert len(fired) == 1


def test_engine_ignores_disabled_rules(store: AlertStore):
    rule = _mk_cross()
    rid = store.save_rule(rule)
    store.patch_rule(rid, enabled=False)
    fired = evaluate_all(store, {"snapshot": {"aapl": 0.99}, "now": 1.0}, "u1")
    assert fired == []


# ---------------------------------------------------------------- channels


class MockChannel:
    """In-test channel that records every call."""

    def __init__(self, label: str = "mock", raise_on_call: bool = False, ok: bool = True) -> None:
        self.type = label
        self.raise_on_call = raise_on_call
        self.ok = ok
        self.calls: list[tuple[dict, str]] = []

    async def send(self, event: dict, target: str) -> dict:
        self.calls.append((event, target))
        if self.raise_on_call:
            raise RuntimeError("simulated channel failure")
        return {
            "channel": self.type,
            "target": target,
            "ok": self.ok,
            "status_code": 200 if self.ok else 500,
            "error": None,
            "ts": time.time(),
        }


def test_channels_inapp_always_ok():
    ch = InAppChannel()
    res = asyncio.run(ch.send({"event_id": "e1"}, "user1"))
    assert res["ok"] is True
    assert res["channel"] == "inapp"


def test_fanout_failure_isolation():
    """If one channel raises, siblings still receive the event."""
    good = MockChannel(label="good")
    bad = MockChannel(label="bad", raise_on_call=True)
    other = MockChannel(label="other")
    registry = {"good": good, "bad": bad, "other": other}
    channels = [
        {"type": "good", "target": "g1", "enabled": True},
        {"type": "bad", "target": "b1", "enabled": True},
        {"type": "other", "target": "o1", "enabled": True},
    ]
    results = asyncio.run(fanout({"event_id": "e1"}, channels, registry))
    assert len(results) == 3
    # All three were attempted — the bad one was captured as a non-ok result.
    assert len(good.calls) == 1
    assert len(bad.calls) == 1
    assert len(other.calls) == 1
    bad_res = next(r for r in results if r["channel"] == "bad")
    assert bad_res["ok"] is False
    assert "simulated" in (bad_res.get("error") or "")
    good_res = next(r for r in results if r["channel"] == "good")
    assert good_res["ok"] is True


def test_fanout_skips_disabled_channels():
    ch = MockChannel(label="mock")
    registry = {"mock": ch}
    channels = [{"type": "mock", "target": "x", "enabled": False}]
    results = asyncio.run(fanout({"event_id": "e1"}, channels, registry))
    assert results == []
    assert ch.calls == []


def test_fanout_unknown_channel_recorded_as_error():
    results = asyncio.run(
        fanout(
            {"event_id": "e1"},
            [{"type": "doesnotexist", "target": "x", "enabled": True}],
            {},
        )
    )
    assert len(results) == 1
    assert results[0]["ok"] is False
    assert "unknown channel" in (results[0]["error"] or "")


def test_slack_channel_dry_run(monkeypatch):
    monkeypatch.setenv("PFM_ALERTS_DRY_RUN", "1")
    res = asyncio.run(
        SlackChannel().send(
            {"event_id": "e1", "kind": "price_cross", "payload": {"rule_name": "rn"}},
            "https://hooks.slack.example/x",
        )
    )
    assert res["ok"] is True
    assert res["error"] == "dry-run"


def test_webhook_signature_when_secret_set(monkeypatch):
    monkeypatch.setenv("PFM_ALERTS_WEBHOOK_SECRET", "shh")
    sig = WebhookChannel._sign(b'{"x":1}')
    assert sig is not None and len(sig) == 64  # sha256 hex


def test_webhook_no_signature_when_no_secret(monkeypatch):
    monkeypatch.delenv("PFM_ALERTS_WEBHOOK_SECRET", raising=False)
    assert WebhookChannel._sign(b'{"x":1}') is None


# ---------------------------------------------------------------- router


@pytest.fixture()
def app_client(store: AlertStore) -> TestClient:
    app = FastAPI()
    app.include_router(alerts_router)
    app.dependency_overrides[get_alert_store] = lambda: store
    return TestClient(app)


def test_endpoint_create_each_kind(app_client: TestClient):
    bodies: list[dict[str, Any]] = [
        {
            "kind": "price_cross",
            "user_id": "u1",
            "name": "pc",
            "slug": "x",
            "op": ">",
            "threshold": 0.5,
            "channels": [{"type": "inapp", "target": "u1"}],
        },
        {
            "kind": "price_change_pct",
            "user_id": "u1",
            "name": "pcp",
            "slug": "x",
            "window": "1h",
            "pct_abs": 0.1,
            "channels": [],
        },
        {
            "kind": "zscore_pair",
            "user_id": "u1",
            "name": "zs",
            "slug_a": "a",
            "slug_b": "b",
            "channels": [],
        },
        {
            "kind": "volume_spike",
            "user_id": "u1",
            "name": "vs",
            "slug": "x",
            "channels": [],
        },
    ]
    for body in bodies:
        r = app_client.post("/alerts", json=body)
        assert r.status_code == 200, r.text
        assert r.json()["kind"] == body["kind"]

    listing = app_client.get("/alerts", params={"user_id": "u1"})
    assert listing.status_code == 200
    assert len(listing.json()) == 4


def test_endpoint_patch_and_delete(app_client: TestClient):
    r = app_client.post(
        "/alerts",
        json={
            "kind": "price_cross",
            "user_id": "u1",
            "name": "n",
            "slug": "x",
            "op": ">",
            "threshold": 0.5,
            "channels": [],
        },
    )
    rid = r.json()["id"]

    pr = app_client.patch(f"/alerts/{rid}", json={"name": "renamed", "enabled": False})
    assert pr.status_code == 200
    assert pr.json()["name"] == "renamed"
    assert pr.json()["enabled"] is False

    dr = app_client.delete(f"/alerts/{rid}")
    assert dr.status_code == 200
    g = app_client.get(f"/alerts/{rid}")
    assert g.status_code == 404


def test_endpoint_test_dry_run(app_client: TestClient, monkeypatch):
    monkeypatch.setenv("PFM_ALERTS_DRY_RUN", "1")
    r = app_client.post(
        "/alerts",
        json={
            "kind": "price_cross",
            "user_id": "u1",
            "name": "n",
            "slug": "x",
            "op": ">",
            "threshold": 0.5,
            "channels": [
                {"type": "inapp", "target": "u1"},
                {"type": "slack", "target": "https://hooks.slack/x"},
            ],
        },
    )
    rid = r.json()["id"]
    tr = app_client.post(f"/alerts/{rid}/test")
    assert tr.status_code == 200
    body = tr.json()
    assert body["dry_run"] is True
    deliveries = body["deliveries"]
    assert len(deliveries) == 2
    assert all(d["ok"] for d in deliveries)


def test_endpoint_events_and_ack(app_client: TestClient, store: AlertStore):
    r = app_client.post(
        "/alerts",
        json={
            "kind": "price_cross",
            "user_id": "u1",
            "name": "n",
            "slug": "x",
            "op": ">",
            "threshold": 0.5,
            "cooldown_seconds": 0,
            "channels": [],
        },
    )
    rid = r.json()["id"]
    # Manually fire by recording an event.
    ev = store.record_event(rid, {"price": 0.99})

    listing = app_client.get("/alerts/events", params={"user_id": "u1", "unack": 1})
    assert listing.status_code == 200
    assert len(listing.json()) == 1

    ack = app_client.post(f"/alerts/events/{ev['event_id']}/ack")
    assert ack.status_code == 200

    after = app_client.get("/alerts/events", params={"user_id": "u1", "unack": 1})
    assert after.json() == []


def test_concurrent_save_no_race() -> None:
    """100 concurrent save_rule calls all succeed without sqlite errors."""
    import threading

    store = AlertStore(":memory:")
    store.init_schema()
    errors: list[Exception] = []

    def save_one(i: int) -> None:
        try:
            rule = PriceCrossRule(
                user_id=f"u{i}",
                name=f"rule_{i}",
                kind="price_cross",
                slug="test",
                op=">",
                threshold=0.5,
                channels=[ChannelRef(type="inapp", target="_")],
            )
            store.save_rule(rule)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=save_one, args=(i,)) for i in range(100)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == [], f"Got {len(errors)} errors: {errors[:3]}"
    all_users: set[str] = set()
    for i in range(100):
        rs = store.list_rules(user_id=f"u{i}")
        all_users.update(r["user_id"] for r in rs)
    assert len(all_users) == 100
