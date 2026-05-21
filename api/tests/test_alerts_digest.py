"""Tests for ``GET /alerts/digest`` (:mod:`pfm.alerts.digest_router`).

The digest assembles alerts from several producers (jumps, sentiment
disagreements, arb opportunities). To keep these tests offline and
deterministic we either:

* monkeypatch ``app.state.alerts`` with a hand-rolled list (preferred —
  this is the production fast path); OR
* inject synthetic ``jumps_source`` / ``arbs_source`` callables into
  :func:`build_digest` directly.

We never call the live Polymarket / Kalshi / arb-scanner stack here.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import pfm.alerts.digest_router as digest_mod
from pfm.alerts.digest_router import (
    DigestResponse,
    _cache_clear,
    _parse_since,
    build_digest,
    router,
)

# ─────────────────────────────────────────────────────────────── fixtures


@pytest.fixture(autouse=True)
def _clear_cache() -> None:
    """Every test starts with a cold cache — TTLs must not leak across tests."""
    _cache_clear()
    yield
    _cache_clear()


def _mk_app(state_alerts: Any = None) -> TestClient:
    app = FastAPI()
    app.include_router(router)
    app.state.alerts = state_alerts
    return TestClient(app)


def _sample_alerts() -> list[dict[str, Any]]:
    """A representative cross-kind alert set for digest tests."""
    return [
        # 3 jumps: 2 high, 1 low
        {"kind": "jump", "label": "trump-2028", "delta_pp": 12.5},
        {"kind": "jump", "label": "fed-cut-may", "delta_pp": 11.0},
        {"kind": "jump", "label": "btc-200k", "delta_pp": 1.4},
        # 2 sentiment-disagree
        {"kind": "sentiment-disagree", "label": "nvda-beats-q1", "delta_pp": 8.0},
        {"kind": "sentiment-disagree", "label": "tsla-down-recall", "delta_pp": 3.2},
        # 4 arb-opportunity: 1 high, 2 med, 1 low
        {"kind": "arb-opportunity", "label": "biden-2024", "spread_pct": 6.1},
        {"kind": "arb-opportunity", "label": "fed-hike-q2", "spread_pct": 3.8},
        {"kind": "arb-opportunity", "label": "sup-court-ruling", "spread_pct": 3.0},
        {"kind": "arb-opportunity", "label": "election-pa", "spread_pct": 2.1},
    ]


# ─────────────────────────────────────────────────────────── _parse_since


class TestParseSince:
    def test_canonical_24h_default(self) -> None:
        assert _parse_since("24h") == 24

    def test_canonical_1h(self) -> None:
        assert _parse_since("1h") == 1

    def test_canonical_7d(self) -> None:
        assert _parse_since("7d") == 24 * 7

    def test_caps_above_7d_at_7d(self) -> None:
        # 30d → capped at 7d (168h), not rejected.
        assert _parse_since("30d") == 24 * 7

    def test_case_insensitive(self) -> None:
        assert _parse_since("24H") == 24
        assert _parse_since("7D") == 24 * 7

    def test_invalid_string_raises_400(self) -> None:
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            _parse_since("forever")
        assert exc_info.value.status_code == 400

    def test_negative_zero_raises_400(self) -> None:
        from fastapi import HTTPException

        with pytest.raises(HTTPException):
            _parse_since("0h")


# ─────────────────────────────────────────────────────────── build_digest


class TestBuildDigest:
    def test_empty_state_returns_zero_total_with_three_buckets(self) -> None:
        # With no app.state.alerts and the default no-op adapters,
        # everything is empty. But we still emit the canonical three
        # buckets so the UI can render a stable layout.
        payload = build_digest(
            "24h", state_alerts=None, jumps_source=lambda h: [], arbs_source=lambda h: []
        )
        assert payload.summary.total == 0
        assert payload.summary.high == 0
        assert payload.summary.med == 0
        assert payload.summary.low == 0
        kinds = [b.kind for b in payload.buckets]
        assert "jump" in kinds
        assert "sentiment-disagree" in kinds
        assert "arb-opportunity" in kinds
        for bucket in payload.buckets:
            assert bucket.count == 0
            assert bucket.examples == []

    def test_bucket_aggregation_from_state_alerts(self) -> None:
        payload = build_digest("24h", state_alerts=_sample_alerts())
        by_kind = {b.kind: b for b in payload.buckets}
        assert by_kind["jump"].count == 3
        assert by_kind["sentiment-disagree"].count == 2
        assert by_kind["arb-opportunity"].count == 4
        # Total = 9
        assert payload.summary.total == 9

    def test_severity_totals_derived_from_magnitudes(self) -> None:
        payload = build_digest("24h", state_alerts=_sample_alerts())
        # Jumps: 12.5 → high, 11.0 → high, 1.4 → low
        # Disagrees: 8.0 → med, 3.2 → low
        # Arbs: 6.1 → high, 3.8 → med, 3.0 → med, 2.1 → low
        # high: 2 + 0 + 1 = 3
        # med:  0 + 1 + 2 = 3
        # low:  1 + 1 + 1 = 3
        assert payload.summary.high == 3
        assert payload.summary.med == 3
        assert payload.summary.low == 3
        assert payload.summary.total == 9

    def test_examples_capped_at_three_per_bucket(self) -> None:
        # 5 jumps → only 3 examples returned.
        many = [{"kind": "jump", "label": f"slug-{i}", "delta_pp": 8.0} for i in range(5)]
        payload = build_digest("24h", state_alerts=many)
        jump_bucket = next(b for b in payload.buckets if b.kind == "jump")
        assert jump_bucket.count == 5
        assert len(jump_bucket.examples) == 3
        # And they're in input order (top-N stable).
        assert jump_bucket.examples == ["slug-0", "slug-1", "slug-2"]

    def test_examples_dedup(self) -> None:
        dupes = [
            {"kind": "jump", "label": "trump-2028", "delta_pp": 8.0},
            {"kind": "jump", "label": "trump-2028", "delta_pp": 9.0},
            {"kind": "jump", "label": "fed-cut-may", "delta_pp": 10.0},
        ]
        payload = build_digest("24h", state_alerts=dupes)
        jump_bucket = next(b for b in payload.buckets if b.kind == "jump")
        # Count still reflects raw alerts (deduplication only affects display).
        assert jump_bucket.count == 3
        assert jump_bucket.examples == ["trump-2028", "fed-cut-may"]

    def test_canonical_window_echo(self) -> None:
        # 7d cap echoes "7d" even when client passed something larger.
        payload = build_digest("30d", state_alerts=[])
        assert payload.since == "7d"

        payload2 = build_digest("1h", state_alerts=[])
        assert payload2.since == "1h"

    def test_injection_seam_used_when_state_alerts_absent(self) -> None:
        # When state_alerts is None, the injected source callables drive
        # the aggregation. This is the test pathway that gives us
        # deterministic non-state behaviour.
        def fake_jumps(hours: int) -> list[dict[str, Any]]:
            return [{"kind": "jump", "label": "fed-hike", "delta_pp": 7.0}]

        def fake_arbs(hours: int) -> list[dict[str, Any]]:
            return [{"kind": "arb-opportunity", "label": "x", "spread_pct": 4.0}]

        payload = build_digest(
            "24h",
            state_alerts=None,
            jumps_source=fake_jumps,
            arbs_source=fake_arbs,
        )
        assert payload.summary.total == 2
        by_kind = {b.kind: b for b in payload.buckets}
        assert by_kind["jump"].count == 1
        assert by_kind["arb-opportunity"].count == 1
        assert by_kind["sentiment-disagree"].count == 0

    def test_source_failure_does_not_blank_other_buckets(self) -> None:
        def bad_jumps(hours: int) -> list[dict[str, Any]]:
            raise RuntimeError("boom")

        def good_arbs(hours: int) -> list[dict[str, Any]]:
            return [{"kind": "arb-opportunity", "label": "y", "spread_pct": 6.0}]

        payload = build_digest(
            "24h", state_alerts=None, jumps_source=bad_jumps, arbs_source=good_arbs
        )
        by_kind = {b.kind: b for b in payload.buckets}
        assert by_kind["jump"].count == 0
        assert by_kind["arb-opportunity"].count == 1


# ─────────────────────────────────────────────────────────── HTTP endpoint


class TestEndpoint:
    def test_default_since_returns_200(self) -> None:
        client = _mk_app(state_alerts=_sample_alerts())
        resp = client.get("/alerts/digest")
        assert resp.status_code == 200
        body = resp.json()
        assert body["since"] == "24h"
        assert "checked_at" in body
        assert body["checked_at"].endswith("Z")
        assert body["summary"]["total"] == 9
        assert len(body["buckets"]) == 3

    def test_explicit_since_1h(self) -> None:
        client = _mk_app(state_alerts=_sample_alerts())
        resp = client.get("/alerts/digest", params={"since": "1h"})
        assert resp.status_code == 200
        assert resp.json()["since"] == "1h"

    def test_invalid_since_returns_400(self) -> None:
        client = _mk_app(state_alerts=[])
        resp = client.get("/alerts/digest", params={"since": "invalid"})
        assert resp.status_code == 400
        assert "since" in resp.json()["detail"].lower()

    def test_empty_state_returns_count_zero(self) -> None:
        # No state.alerts and the default lazy adapters return [] when
        # the upstream modules aren't reachable / fail. The endpoint
        # must still 200 with three zero-count buckets.
        client = _mk_app(state_alerts=[])
        resp = client.get("/alerts/digest")
        assert resp.status_code == 200
        body = resp.json()
        assert body["summary"]["total"] == 0
        for bucket in body["buckets"]:
            assert bucket["count"] == 0
            assert bucket["examples"] == []

    def test_since_param_caps_at_7d(self) -> None:
        client = _mk_app(state_alerts=[])
        resp = client.get("/alerts/digest", params={"since": "30d"})
        assert resp.status_code == 200
        assert resp.json()["since"] == "7d"

    def test_bucket_examples_top3_appears_in_response(self) -> None:
        client = _mk_app(state_alerts=_sample_alerts())
        resp = client.get("/alerts/digest")
        body = resp.json()
        arb_bucket = next(b for b in body["buckets"] if b["kind"] == "arb-opportunity")
        assert arb_bucket["count"] == 4
        # Examples capped at 3.
        assert len(arb_bucket["examples"]) == 3
        assert arb_bucket["examples"][0] == "biden-2024"


# ─────────────────────────────────────────────────────────── TTL cache


class TestTTLCache:
    def test_repeat_request_hits_cache(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Two back-to-back requests within 60 s share the same response object."""
        # Freeze the perf counter so cache entries never expire mid-test.
        t = [1000.0]
        monkeypatch.setattr(digest_mod, "_PERF_COUNTER", lambda: t[0])

        client = _mk_app(state_alerts=_sample_alerts())
        first = client.get("/alerts/digest").json()
        # Mutate state.alerts AFTER the first hit. A cached response must
        # ignore this change (until TTL expires) — that's the contract.
        client.app.state.alerts = []
        second = client.get("/alerts/digest").json()
        assert first["summary"]["total"] == second["summary"]["total"] == 9

    def test_cache_expires_after_ttl(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Drive the cache clock forward past the 60 s TTL.
        t = [1000.0]
        monkeypatch.setattr(digest_mod, "_PERF_COUNTER", lambda: t[0])

        client = _mk_app(state_alerts=_sample_alerts())
        first = client.get("/alerts/digest").json()
        assert first["summary"]["total"] == 9

        # 61 s later, swap the state and request again — should re-compute.
        client.app.state.alerts = []
        t[0] = 1061.0
        second = client.get("/alerts/digest").json()
        assert second["summary"]["total"] == 0

    def test_cache_ttl_is_60s(self) -> None:
        """The configured TTL is exactly 60 s per the task spec."""
        assert digest_mod._CACHE_TTL_S == 60.0


# ─────────────────────────────────────────────────────────── shape checks


class TestResponseShape:
    def test_response_model_round_trip(self) -> None:
        # build_digest returns a DigestResponse; pydantic must validate it.
        payload = build_digest("24h", state_alerts=_sample_alerts())
        assert isinstance(payload, DigestResponse)
        dumped = payload.model_dump()
        assert set(dumped.keys()) == {"since", "checked_at", "summary", "buckets"}
        assert set(dumped["summary"].keys()) == {"total", "high", "med", "low"}
        for bucket in dumped["buckets"]:
            assert set(bucket.keys()) == {"kind", "count", "examples"}

    def test_checked_at_is_iso_z_format(self) -> None:
        payload = build_digest("24h", state_alerts=[])
        assert payload.checked_at.endswith("Z")
        # Must parse as ISO8601 — strict round-trip via fromisoformat.
        from datetime import datetime

        # fromisoformat in py3.11+ handles trailing Z; for older versions
        # we strip it ourselves.
        ts = payload.checked_at.rstrip("Z")
        datetime.fromisoformat(ts)
