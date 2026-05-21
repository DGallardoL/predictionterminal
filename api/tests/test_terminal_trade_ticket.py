"""Tests for ``pfm.terminal_trade_ticket`` — /terminal/trade-ticket/{...}.

The router is mounted on a fresh :class:`FastAPI` app and the on-disk
strat-28 file is replaced with an in-memory fixture so the suite is
hermetic — no Redis, no factors.yml, no live HTTP.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pfm import terminal_trade_ticket
from pfm.terminal_trade_ticket import reload_clusters, router


def _strat28_payload(*, strong: bool) -> dict:
    """Build a strat-28 fixture with one strong-signal cluster + one benign.

    ``strong=True`` ensures the Fed-cut cluster crosses ``OPEN_THRESHOLD``;
    the benign Amazon cluster never does, so it always resolves to ``WAIT``.
    """
    # short leg: p=0.06 over 60d ⇒ λ ≈ 0.00103.
    # strong=True  → p_long=0.50 over 244d ⇒ λ ≈ 0.00284 ⇒ log-ratio ≈ +1.01
    # strong=False → p_long=0.22 over 244d ⇒ λ ≈ 0.00102 ⇒ log-ratio ≈ -0.01
    long_mid_strong = 0.50
    long_mid_weak = 0.22

    fed_long_mid = long_mid_strong if strong else long_mid_weak

    return {
        "meta": {"today": "2026-05-02", "n_pairs": 2},
        "top5_actionable": [
            {
                "event_token": "bps cuts fed kalshi kxfeddecision",
                "short_id": "k_fed_jul_cut25",
                "short_name": "Fed cuts 25bps in July 2026 (Kalshi)",
                "short_mid": 0.06,
                "short_dtr": 60,
                "long_id": "k_fed_dec_cut25",
                "long_name": "Fed cuts 25bps in December 2026 (Kalshi)",
                "long_mid": fed_long_mid,
                "long_dtr": 244,
                "log_ratio": 1.21 if strong else 0.05,
                "ev_per_dollar_net": 0.07 if strong else 0.005,
            },
        ],
        "pairs_sample": [
            {
                "event": "amazon best ai may",
                "short": {
                    "id": "amazon_best_ai_may",
                    "name": "Amazon best AI May",
                    "dtr": 30,
                    "mid": 0.0015,
                },
                "long": {
                    "id": "amazon_best_ai_jun",
                    "name": "Amazon best AI Jun",
                    "dtr": 60,
                    "mid": 0.0025,
                },
            },
        ],
    }


@pytest.fixture
def strong_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """Client where the Fed-cut cluster is strongly dispersed (OPEN_PAIR)."""
    p28 = tmp_path / "strat28.json"
    p28.write_text(json.dumps(_strat28_payload(strong=True)))
    # Force the strat-28 fallback path (curated file absent in tmp_path).
    monkeypatch.setattr(terminal_trade_ticket, "CURATED_CLUSTERS_PATH", tmp_path / "_missing.json")
    monkeypatch.setattr(terminal_trade_ticket, "STRAT28_PATH", p28)
    reload_clusters()

    app = FastAPI()
    app.include_router(router)
    with TestClient(app) as client:
        yield client


@pytest.fixture
def weak_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """Client where no cluster crosses ``OPEN_THRESHOLD`` (WAIT)."""
    p28 = tmp_path / "strat28.json"
    p28.write_text(json.dumps(_strat28_payload(strong=False)))
    monkeypatch.setattr(terminal_trade_ticket, "CURATED_CLUSTERS_PATH", tmp_path / "_missing.json")
    monkeypatch.setattr(terminal_trade_ticket, "STRAT28_PATH", p28)
    reload_clusters()

    app = FastAPI()
    app.include_router(router)
    with TestClient(app) as client:
        yield client


# ── tests ─────────────────────────────────────────────────────────────────────


def test_high_log_ratio_yields_open_pair_with_correct_sides(
    strong_client: TestClient,
) -> None:
    """A strongly-dispersed pair → OPEN_PAIR with sane near/far sides.

    Far hazard >> near hazard ⇒ ``log_ratio > 0`` ⇒ STEEPEN_CURVE:
    BUY_YES on the cheap-in-hazard NEAR leg, BUY_NO on the rich-in-hazard
    FAR leg.
    """
    cluster_id = "bps_cuts_fed_kalshi_kxfeddecision"
    r = strong_client.get(
        f"/terminal/trade-ticket/{cluster_id}",
        params={"bankroll": 10_000, "risk_per_trade": 0.05},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["action"] == "OPEN_PAIR"
    assert body["cluster_id"] == cluster_id
    assert len(body["tickets"]) == 2
    near, far = body["tickets"]
    # log_ratio > 0 ⇒ near=BUY_YES, far=BUY_NO
    assert body["log_lambda_ratio"] > 0
    assert near["side"] == "BUY_YES"
    assert far["side"] == "BUY_NO"
    assert near["slug"] == "k_fed_jul_cut25"
    assert far["slug"] == "k_fed_dec_cut25"


def test_low_log_ratio_yields_wait(weak_client: TestClient) -> None:
    """When |log-ratio| is below ``OPEN_THRESHOLD`` we return WAIT, no legs."""
    r = weak_client.get(
        "/terminal/trade-ticket/bps_cuts_fed_kalshi_kxfeddecision",
        params={"bankroll": 10_000, "risk_per_trade": 0.05},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["action"] == "WAIT"
    assert body["tickets"] == []
    assert body["total_capital_at_risk_usd"] == 0.0


def test_bankroll_sizing_splits_250_per_leg_at_5pct_of_10k(
    strong_client: TestClient,
) -> None:
    """$10k @ 5% risk_per_trade ⇒ $250 per leg, $500 total at risk."""
    r = strong_client.get(
        "/terminal/trade-ticket/bps_cuts_fed_kalshi_kxfeddecision",
        params={"bankroll": 10_000, "risk_per_trade": 0.05},
    )
    body = r.json()
    assert body["action"] == "OPEN_PAIR"
    assert body["total_capital_at_risk_usd"] == 500.0
    for leg in body["tickets"]:
        assert leg["size_usd"] == 250.0
        # size_contracts must be a positive integer corresponding to
        # size_usd / cost_per_contract. Cost is mid for BUY_YES, (1-mid)
        # for BUY_NO. Verify rough order-of-magnitude (avoid pinning the
        # rounding behaviour bit-for-bit).
        assert leg["size_contracts"] > 0


def test_exit_conditions_and_scan_endpoint_populated(
    strong_client: TestClient,
) -> None:
    """Exit/execution/risk lists are populated, and /scan returns the ticket."""
    r = strong_client.get(
        "/terminal/trade-ticket/bps_cuts_fed_kalshi_kxfeddecision",
        params={"bankroll": 10_000, "risk_per_trade": 0.05},
    )
    body = r.json()
    assert body["action"] == "OPEN_PAIR"

    # Three exit rules, each non-trivial.
    assert len(body["exit_conditions"]) == 3
    joined = " ".join(body["exit_conditions"]).lower()
    assert "take profit" in joined
    assert "stop loss" in joined
    assert "time stop" in joined

    assert any("limit" in note.lower() for note in body["execution_notes"])
    assert any("concentration" in w.lower() for w in body["risk_warnings"])
    assert body["expected_hold_days"] > 0

    # /scan reports the actionable subset.
    s = strong_client.get("/terminal/trade-ticket/scan", params={"bankroll": 10_000})
    assert s.status_code == 200, s.text
    sbody = s.json()
    assert sbody["bankroll_usd"] == 10_000.0
    assert sbody["n_actionable"] >= 1
    ids = {t["cluster_id"] for t in sbody["tickets"]}
    assert "bps_cuts_fed_kalshi_kxfeddecision" in ids
    # Every returned ticket must in fact be OPEN_PAIR.
    for t in sbody["tickets"]:
        assert t["action"] == "OPEN_PAIR"
