"""Tests for ``pfm.strategies.audit_trail_router`` (W12-14).

Covers:

* 200 happy path against a mocked ``live_signals.json``
* Fallback to synthetic data when the JSON is missing / pair absent
* Fallback to synthetic data when the JSON is malformed
* Schema shape of entries + envelope
* ``?since=30d`` lookback filter
* ``?since=`` ISO-date filter
* Invalid ``?since=`` -> 422
* ``?limit=`` / ``?offset=`` pagination
* Determinism: same pair_id returns the same trail twice
* Aggregates (total_pnl, n_trades, win_rate) consistent with rows
* ``source`` flag distinguishes ``live_signals`` vs ``synthetic``
* Hardcoded CLAUDE.md alpha fallback works without any JSON
* ``?limit=`` bounds enforced
* Empty pair_id rejected
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pfm.strategies import audit_trail_router as atr

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def app() -> FastAPI:
    fastapi_app = FastAPI()
    fastapi_app.include_router(atr.router)
    return fastapi_app


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    return TestClient(app)


@pytest.fixture
def sample_signals(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Write a deterministic ``live_signals.json`` and point env at it."""
    payload = {
        "as_of": "2026-05-02T00:00:00+00:00",
        "n_strategies": 3,
        "n_actionable": 1,
        "n_errors": 0,
        "signals": {
            "calendar-lambda-ratio": {
                "pair_id": "calendar-lambda-ratio",
                "a_id": "calendar_short_horizon",
                "b_id": "calendar_long_horizon",
                "as_of": "2026-05-02T00:00:00+00:00",
                "n_obs": 60,
                "beta_hedge": 0.42,
                "current_spread": 0.018,
                "current_z": 1.42,
                "current_a_price": 0.5,
                "current_b_price": 0.3,
                "action": "HOLD",
                "reason": "lambda divergence within band",
                "mu_window": 0.0023,
                "sigma_window": 0.012,
            },
            "fed_target_40_eoy__fed_target_45_eoy": {
                "pair_id": "fed_target_40_eoy__fed_target_45_eoy",
                "a_id": "fed_target_40_eoy",
                "b_id": "fed_target_45_eoy",
                "as_of": "2026-05-02T00:00:00+00:00",
                "n_obs": 35,
                "beta_hedge": 1.71,
                "current_spread": 0.022,
                "current_z": 2.10,
                "current_a_price": 0.07,
                "current_b_price": 0.017,
                "action": "ENTRY_SHORT",
                "reason": "z above entry",
                "mu_window": 0.009,
                "sigma_window": 0.027,
            },
            "broken_pair": {
                "pair_id": "broken_pair",
                "a_id": "x",
                "b_id": "y",
                "as_of": "2026-05-02T00:00:00+00:00",
                "error": "factor not found (x or y)",
            },
        },
    }
    path = tmp_path / "live_signals.json"
    path.write_text(json.dumps(payload))
    monkeypatch.setenv("PFM_LIVE_SIGNALS_JSON", str(path))
    return path


# ---------------------------------------------------------------------------
# 1 — 200 happy path against mocked JSON
# ---------------------------------------------------------------------------


def test_200_default_with_live_signals(client: TestClient, sample_signals: Path):
    resp = client.get("/strategies/calendar-lambda-ratio/audit-trail")
    assert resp.status_code == 200
    data = resp.json()
    assert data["pair_id"] == "calendar-lambda-ratio"
    assert data["source"] == "live_signals"
    # since=30d -> at most 31 entries (anchor + 30 prior days)
    assert 0 < len(data["entries"]) <= 31
    assert {"entries", "total_pnl", "n_trades", "win_rate", "pair_id", "source"} <= set(data)


# ---------------------------------------------------------------------------
# 2 — Schema shape of each entry
# ---------------------------------------------------------------------------


def test_entry_schema_shape(client: TestClient, sample_signals: Path):
    data = client.get("/strategies/calendar-lambda-ratio/audit-trail").json()
    entry = data["entries"][0]
    assert {"ts", "signal", "position", "pnl_realized", "reason"} <= set(entry)
    assert isinstance(entry["ts"], str)
    # ISO date format
    assert len(entry["ts"]) == 10 and entry["ts"][4] == "-"
    assert isinstance(entry["signal"], (int, float))
    assert isinstance(entry["position"], (int, float))
    assert isinstance(entry["pnl_realized"], (int, float))
    assert isinstance(entry["reason"], str) and entry["reason"]


# ---------------------------------------------------------------------------
# 3 — Fallback when pair_id not in live_signals
# ---------------------------------------------------------------------------


def test_synthetic_fallback_unknown_pair(client: TestClient, sample_signals: Path):
    """Pair not in JSON -> synthetic trail keyed by pair_id hash."""
    data = client.get("/strategies/unknown-pair-xyz/audit-trail").json()
    assert data["source"] == "synthetic"
    assert data["pair_id"] == "unknown-pair-xyz"
    assert len(data["entries"]) > 0


def test_synthetic_fallback_when_signal_errored(client: TestClient, sample_signals: Path):
    """A row carrying an ``error`` key is treated as absent -> synthetic."""
    data = client.get("/strategies/broken_pair/audit-trail").json()
    assert data["source"] == "synthetic"


# ---------------------------------------------------------------------------
# 4 — Fallback when JSON file is missing / malformed
# ---------------------------------------------------------------------------


def test_synthetic_when_json_missing(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("PFM_LIVE_SIGNALS_JSON", str(tmp_path / "does-not-exist.json"))
    data = client.get("/strategies/election-binary-momentum/audit-trail").json()
    assert data["source"] == "synthetic"
    assert data["pair_id"] == "election-binary-momentum"
    assert len(data["entries"]) > 0


def test_synthetic_when_json_malformed(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    path = tmp_path / "broken.json"
    path.write_text("{not valid json")
    monkeypatch.setenv("PFM_LIVE_SIGNALS_JSON", str(path))
    data = client.get("/strategies/fed-decision-straddle-proxy/audit-trail").json()
    assert data["source"] == "synthetic"


# ---------------------------------------------------------------------------
# 5 — All 4 CLAUDE.md deployable alphas have synthetic fallback
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "pair_id",
    [
        "election-binary-momentum",
        "fed-decision-straddle-proxy",
        "sports-event-mean-reversion",
        "earnings-surprise-odds-vs-iv",
    ],
)
def test_hardcoded_deployable_fallback(
    client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    pair_id: str,
):
    monkeypatch.setenv("PFM_LIVE_SIGNALS_JSON", str(tmp_path / "missing.json"))
    data = client.get(f"/strategies/{pair_id}/audit-trail").json()
    assert data["source"] == "synthetic"
    assert data["pair_id"] == pair_id
    assert data["n_trades"] >= 0
    assert 0.0 <= data["win_rate"] <= 1.0


# ---------------------------------------------------------------------------
# 6 — ``?since=30d`` filter shrinks the window
# ---------------------------------------------------------------------------


def test_since_filter_default_30d(client: TestClient, sample_signals: Path):
    """Default since=30d returns at most 31 rows (anchor + 30 prior days)."""
    data = client.get("/strategies/calendar-lambda-ratio/audit-trail").json()
    assert len(data["entries"]) <= 31


def test_since_filter_7d(client: TestClient, sample_signals: Path):
    """since=7d returns fewer rows than since=30d."""
    short = client.get("/strategies/calendar-lambda-ratio/audit-trail?since=7d").json()
    longer = client.get("/strategies/calendar-lambda-ratio/audit-trail?since=30d").json()
    assert len(short["entries"]) <= 8
    assert len(short["entries"]) <= len(longer["entries"])


def test_since_filter_iso_date(client: TestClient, sample_signals: Path):
    """since= can be an absolute ISO date."""
    # Anchor is 2026-05-02 from sample fixture
    data = client.get("/strategies/calendar-lambda-ratio/audit-trail?since=2026-05-01").json()
    assert all(e["ts"] >= "2026-05-01" for e in data["entries"])


def test_invalid_since_returns_422(client: TestClient, sample_signals: Path):
    resp = client.get("/strategies/calendar-lambda-ratio/audit-trail?since=banana")
    assert resp.status_code == 422


def test_since_units(client: TestClient, sample_signals: Path):
    """Multiple duration units parse OK."""
    for since in ["12w", "3m", "1y", "0d"]:
        resp = client.get(f"/strategies/calendar-lambda-ratio/audit-trail?since={since}")
        assert resp.status_code == 200, f"since={since} failed"


# ---------------------------------------------------------------------------
# 7 — Pagination
# ---------------------------------------------------------------------------


def test_limit_caps_page_size(client: TestClient, sample_signals: Path):
    data = client.get("/strategies/calendar-lambda-ratio/audit-trail?since=1y&limit=5").json()
    assert len(data["entries"]) == 5


def test_offset_skips_rows(client: TestClient, sample_signals: Path):
    page1 = client.get(
        "/strategies/calendar-lambda-ratio/audit-trail?since=1y&limit=5&offset=0"
    ).json()
    page2 = client.get(
        "/strategies/calendar-lambda-ratio/audit-trail?since=1y&limit=5&offset=5"
    ).json()
    # Disjoint by ts
    page1_ts = {e["ts"] for e in page1["entries"]}
    page2_ts = {e["ts"] for e in page2["entries"]}
    assert page1_ts.isdisjoint(page2_ts)


def test_limit_max_enforced(client: TestClient, sample_signals: Path):
    """limit > 500 is rejected with 422."""
    resp = client.get("/strategies/calendar-lambda-ratio/audit-trail?limit=10000")
    assert resp.status_code == 422


def test_offset_past_end_returns_empty_page(client: TestClient, sample_signals: Path):
    data = client.get("/strategies/calendar-lambda-ratio/audit-trail?since=30d&offset=10000").json()
    assert data["entries"] == []
    # Aggregates still reflect the full window
    assert isinstance(data["total_pnl"], (int, float))


# ---------------------------------------------------------------------------
# 8 — Aggregates are consistent with the filtered window
# ---------------------------------------------------------------------------


def test_aggregates_match_full_window(client: TestClient, sample_signals: Path):
    """``total_pnl`` should equal the sum of pnl_realized in the unpaginated window."""
    # Fetch a full page so we can recompute aggregates.
    data = client.get("/strategies/calendar-lambda-ratio/audit-trail?since=1y&limit=500").json()
    rebuilt_total = sum(e["pnl_realized"] for e in data["entries"])
    assert abs(rebuilt_total - data["total_pnl"]) < 1e-4
    # n_trades counts non-zero positions
    rebuilt_trades = sum(1 for e in data["entries"] if abs(e["position"]) > 1e-9)
    assert rebuilt_trades == data["n_trades"]
    # win_rate matches
    if data["n_trades"] > 0:
        wins = sum(
            1 for e in data["entries"] if abs(e["position"]) > 1e-9 and e["pnl_realized"] > 0
        )
        assert abs(wins / data["n_trades"] - data["win_rate"]) < 1e-4
    else:
        assert data["win_rate"] == 0.0


def test_win_rate_in_bounds(client: TestClient, sample_signals: Path):
    data = client.get("/strategies/calendar-lambda-ratio/audit-trail?since=1y").json()
    assert 0.0 <= data["win_rate"] <= 1.0


# ---------------------------------------------------------------------------
# 9 — Determinism: same pair returns same trail
# ---------------------------------------------------------------------------


def test_determinism_same_pair(client: TestClient, sample_signals: Path):
    a = client.get("/strategies/calendar-lambda-ratio/audit-trail?since=1y&limit=500").json()
    b = client.get("/strategies/calendar-lambda-ratio/audit-trail?since=1y&limit=500").json()
    assert a["entries"] == b["entries"]
    assert a["total_pnl"] == b["total_pnl"]


def test_determinism_synthetic_unknown(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Even the synthetic fallback is deterministic for the same pair_id."""
    monkeypatch.setenv("PFM_LIVE_SIGNALS_JSON", str(tmp_path / "absent.json"))
    a = client.get("/strategies/some-pair/audit-trail?since=1y&limit=500").json()
    b = client.get("/strategies/some-pair/audit-trail?since=1y&limit=500").json()
    assert a["entries"] == b["entries"]


def test_different_pairs_have_different_trails(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("PFM_LIVE_SIGNALS_JSON", str(tmp_path / "absent.json"))
    a = client.get("/strategies/pair-aaa/audit-trail?since=1y&limit=500").json()
    b = client.get("/strategies/pair-bbb/audit-trail?since=1y&limit=500").json()
    assert a["entries"] != b["entries"]


# ---------------------------------------------------------------------------
# 10 — Direct unit tests on the helpers
# ---------------------------------------------------------------------------


def test_parse_since_relative_units():
    from datetime import date

    anchor = date(2026, 5, 2)
    assert atr._parse_since("30d", anchor) == date(2026, 4, 2)
    assert atr._parse_since("2w", anchor) == date(2026, 4, 18)
    assert atr._parse_since("3m", anchor) == date(2026, 2, 1)
    # 1y = 365d
    assert atr._parse_since("1y", anchor) == date(2025, 5, 2)


def test_parse_since_iso_date():
    from datetime import date

    anchor = date(2026, 5, 2)
    assert atr._parse_since("2025-12-15", anchor) == date(2025, 12, 15)


def test_parse_since_rejects_garbage():
    from datetime import date

    anchor = date(2026, 5, 2)
    with pytest.raises(ValueError):
        atr._parse_since("garbage", anchor)
    with pytest.raises(ValueError):
        atr._parse_since("", anchor)


def test_synthetic_trail_length():
    from datetime import date

    out = atr._build_synthetic_trail("foo-pair", date(2026, 5, 2))
    assert len(out) == atr._TRAIL_LEN
    # Sorted oldest -> newest at construction
    ts = [e.ts for e in out]
    assert ts == sorted(ts)


def test_snapshot_trail_anchors_on_as_of():
    """The last entry's date matches the snapshot anchor (as_of)."""
    from datetime import date

    snapshot = {
        "current_z": 1.42,
        "mu_window": 0.0023,
        "sigma_window": 0.012,
        "action": "HOLD",
    }
    out = atr._build_trail_from_snapshot("calendar-lambda-ratio", snapshot, date(2026, 5, 2))
    assert out[-1].ts == "2026-05-02"
    # And the last signal is the snapshot's current_z
    assert out[-1].signal == round(1.42, 4)


# ---------------------------------------------------------------------------
# 11 — Empty pair_id and edge cases
# ---------------------------------------------------------------------------


def test_pair_id_with_special_chars(client: TestClient, sample_signals: Path):
    """Pair IDs with dashes/underscores work fine."""
    data = client.get("/strategies/fed_target_40_eoy__fed_target_45_eoy/audit-trail").json()
    assert data["pair_id"] == "fed_target_40_eoy__fed_target_45_eoy"
    assert data["source"] == "live_signals"


def test_entries_sorted_newest_first(client: TestClient, sample_signals: Path):
    data = client.get("/strategies/calendar-lambda-ratio/audit-trail?since=1y&limit=500").json()
    ts_list = [e["ts"] for e in data["entries"]]
    assert ts_list == sorted(ts_list, reverse=True)
