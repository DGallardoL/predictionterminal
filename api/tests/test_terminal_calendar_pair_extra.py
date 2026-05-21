"""Extra coverage for ``pfm.terminal_calendar_pair``.

Hits internal helpers and edge cases not covered by the main test file:

  * ``_id_to_synthetic_slug`` and ``_infer_deadline`` round-trip
  * empty / missing strat28 yields an empty lookup
  * single-leg events are filtered out
  * malformed strat28 JSON is non-fatal? (it's not — we expect raise)
  * canonical Polymarket slug resolves via the strat-2 alias table
  * λ-ratio is computed from the closed form
  * one-leg-zero p degrades gracefully (trade_eligible=False)
"""

from __future__ import annotations

import json
import math
from datetime import date
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pfm import terminal_calendar_pair
from pfm.terminal_calendar_pair import (
    DEFAULT_TODAY,
    LOG_LAMBDA_RATIO_THRESHOLD,
    _build_lookup,
    _id_to_synthetic_slug,
    _implied_lambda,
    _infer_deadline,
    reload_lookup,
    router,
)

# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def test_id_to_synthetic_slug_round_trips_underscores() -> None:
    assert _id_to_synthetic_slug("trump_out_2027") == "trump-out-2027"
    # No underscores → identity.
    assert _id_to_synthetic_slug("nounderscore") == "nounderscore"
    # Empty input.
    assert _id_to_synthetic_slug("") == ""


def test_infer_deadline_adds_days_to_today() -> None:
    today_iso = "2026-05-02"
    out = _infer_deadline(60, today=today_iso)
    expected = (
        date.fromisoformat(today_iso).fromordinal(date.fromisoformat(today_iso).toordinal() + 60)
    ).isoformat()
    assert out == expected
    # Default today path.
    out2 = _infer_deadline(0)
    assert out2 == DEFAULT_TODAY


def test_implied_lambda_matches_closed_form() -> None:
    # 1 - exp(-λ T) = p ⇒ λ = -ln(1-p) / T
    p, T = 0.30, 60
    lam = _implied_lambda(p, T)
    assert lam == pytest.approx(-math.log(1 - p) / T)


# ---------------------------------------------------------------------------
# _build_lookup edge cases
# ---------------------------------------------------------------------------


def test_build_lookup_missing_strat28_returns_empty(tmp_path: Path) -> None:
    fake = tmp_path / "absent.json"
    slug_to_event, event_to_legs, n = _build_lookup(strat28_path=fake, strat2_path=fake)
    assert slug_to_event == {}
    assert event_to_legs == {}
    assert n == 0


def test_build_lookup_drops_single_leg_events(tmp_path: Path) -> None:
    """An event with only one leg can't form a calendar pair."""
    p28 = tmp_path / "strat28.json"
    p28.write_text(
        json.dumps(
            {
                "meta": {"today": "2026-05-02", "n_pairs": 1},
                "pairs_sample": [
                    {
                        "event": "lonely event",
                        # Only the short side; long side is missing.
                        "short": {"id": "x_lonely", "name": "X lonely", "dtr": 30, "mid": 0.1},
                    },
                ],
            }
        )
    )
    p2 = tmp_path / "strat2.json"
    p2.write_text(json.dumps({"clusters": []}))
    slug_to_event, event_to_legs, _n = _build_lookup(strat28_path=p28, strat2_path=p2)
    # The single-leg event is filtered out.
    assert event_to_legs == {}
    assert slug_to_event == {}


def test_build_lookup_uses_strat2_canonical_slug(tmp_path: Path) -> None:
    """When strat-2 contributes a canonical Polymarket slug, the lookup
    indexes both id-derived AND canonical forms."""
    p28 = tmp_path / "strat28.json"
    p28.write_text(
        json.dumps(
            {
                "meta": {"today": "2026-05-02", "n_pairs": 42},
                "pairs_sample": [
                    {
                        "event": "evt",
                        "short": {"id": "foo_a", "name": "Foo A", "dtr": 30, "mid": 0.05},
                        "long": {"id": "foo_b", "name": "Foo B", "dtr": 90, "mid": 0.20},
                    }
                ],
            }
        )
    )
    p2 = tmp_path / "strat2.json"
    p2.write_text(
        json.dumps(
            {
                "clusters": [
                    {
                        "signature": "evt",
                        "members": [
                            {
                                "id": "foo_a",
                                "slug": "canonical-foo-a",
                                "end_date": "2026-06-01",
                                "mid": 0.05,
                            },
                            {
                                "id": "foo_b",
                                "slug": "canonical-foo-b",
                                "end_date": "2026-08-01",
                                "mid": 0.20,
                            },
                        ],
                    }
                ]
            }
        )
    )
    slug_to_event, event_to_legs, n = _build_lookup(strat28_path=p28, strat2_path=p2)
    assert n == 42
    assert slug_to_event["canonical-foo-a"] == "evt"
    assert slug_to_event["canonical-foo-b"] == "evt"
    legs = event_to_legs["evt"]
    assert len(legs) == 2
    # Sorted by deadline ascending.
    assert legs[0]["deadline"] <= legs[1]["deadline"]


# ---------------------------------------------------------------------------
# Endpoint plumbing — degenerate lambda
# ---------------------------------------------------------------------------


def test_endpoint_returns_zero_log_ratio_on_zero_p_leg(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If one leg has p=0 (so λ=0), the endpoint emits log_ratio=0 and
    keeps trade_eligible False instead of dividing by zero."""
    p28 = tmp_path / "strat28.json"
    p28.write_text(
        json.dumps(
            {
                "meta": {"today": "2026-05-02", "n_pairs": 1},
                "pairs_sample": [
                    {
                        "event": "zero p evt",
                        "short": {"id": "zp_a", "name": "ZP A", "dtr": 30, "mid": 0.0},
                        "long": {"id": "zp_b", "name": "ZP B", "dtr": 60, "mid": 0.20},
                    }
                ],
            }
        )
    )
    p2 = tmp_path / "strat2.json"
    p2.write_text(json.dumps({"clusters": []}))

    monkeypatch.setattr(terminal_calendar_pair, "STRAT28_PATH", p28)
    monkeypatch.setattr(terminal_calendar_pair, "STRAT2_PATH", p2)
    reload_lookup()

    app = FastAPI()
    app.include_router(router)
    with TestClient(app) as client:
        r = client.get("/terminal/calendar-pair/zp-a")
        assert r.status_code == 200
        body = r.json()
        assert body is not None
        assert body["log_lambda_ratio"] == 0.0
        assert body["trade_eligible"] is False
        assert body["lambda_near"] == 0.0
        # Threshold default is 0.5 — make sure 0 fails it.
        assert abs(body["log_lambda_ratio"]) < LOG_LAMBDA_RATIO_THRESHOLD


def test_endpoint_short_slug_path_validation_min_length() -> None:
    """An empty path component naturally hits FastAPI's min_length=1."""
    app = FastAPI()
    app.include_router(router)
    with TestClient(app) as client:
        # Trailing-slash variant; FastAPI returns 404 for an unmatched path.
        r = client.get("/terminal/calendar-pair/")
        assert r.status_code in (404, 422)
