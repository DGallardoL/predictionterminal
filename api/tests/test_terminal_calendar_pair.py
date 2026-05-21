"""Tests for ``pfm.terminal_calendar_pair`` — /terminal/calendar-pair/{slug}.

The router is mounted on a fresh :class:`FastAPI` app (no Redis / no
factors.yml lifespan) and the on-disk strat-28 / strat-2 inputs are
replaced with in-memory fixtures so the suite is hermetic.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pfm import terminal_calendar_pair
from pfm.terminal_calendar_pair import (
    LOG_LAMBDA_RATIO_THRESHOLD,
    _implied_lambda,
    reload_lookup,
    router,
)

# --- fixtures ---------------------------------------------------------------


def _write_fixture_files(tmp_path: Path, *, strong_signal: bool) -> tuple[Path, Path]:
    """Produce strat-28 + strat-2 fixtures with two calendar pairs.

    ``strong_signal=True`` builds a pair whose far/near hazard ratio
    exceeds the Strategy-24 threshold so we can exercise the
    ``trade_eligible`` flag in both states.
    """
    # strong_signal=True → p_long=0.50 over 244d ⇒ λ_far/λ_near ratio crosses
    # the Strategy-24 threshold. False → p_long=0.08 ⇒ benign log-ratio (~0.07).
    long_mid = 0.50 if strong_signal else 0.08

    strat28 = {
        "meta": {"today": "2026-05-02", "n_pairs": 42},
        "pairs_sample": [
            {
                "event": "out president trump",
                "short": {
                    "id": "trump_out_jun30",
                    "name": "Trump out by Jun 30",
                    "dtr": 60,
                    "mid": 0.02,
                },
                "long": {
                    "id": "trump_out_2027",
                    "name": "Trump out before 2027",
                    "dtr": 244,
                    "mid": long_mid,
                },
                "log_ratio": 1.0,
            },
            {
                "event": "amazon best has model",
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
                "log_ratio": 0.2,
            },
        ],
    }
    strat2 = {
        "clusters": [
            {
                "signature": "out president trump",
                "members": [
                    {
                        "id": "trump_out_jun30",
                        "slug": "trump-out-as-president-by-june-30",
                        "end_date": "2026-06-30",
                        "mid": 0.02,
                    },
                    {
                        "id": "trump_out_2027",
                        "slug": "trump-out-as-president-before-2027",
                        "end_date": "2026-12-31",
                        "mid": long_mid,
                    },
                ],
            },
        ],
    }

    p28 = tmp_path / "strat28.json"
    p2 = tmp_path / "strat2.json"
    p28.write_text(json.dumps(strat28))
    p2.write_text(json.dumps(strat2))
    return p28, p2


@pytest.fixture
def eligible_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """Client where the Trump-out calendar pair crosses the trade threshold."""
    p28, p2 = _write_fixture_files(tmp_path, strong_signal=True)
    monkeypatch.setattr(terminal_calendar_pair, "STRAT28_PATH", p28)
    monkeypatch.setattr(terminal_calendar_pair, "STRAT2_PATH", p2)
    reload_lookup()

    app = FastAPI()
    app.include_router(router)
    with TestClient(app) as client:
        yield client


@pytest.fixture
def benign_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """Client where no pair crosses the threshold (trade_eligible=False)."""
    p28, p2 = _write_fixture_files(tmp_path, strong_signal=False)
    monkeypatch.setattr(terminal_calendar_pair, "STRAT28_PATH", p28)
    monkeypatch.setattr(terminal_calendar_pair, "STRAT2_PATH", p2)
    reload_lookup()

    app = FastAPI()
    app.include_router(router)
    with TestClient(app) as client:
        yield client


# --- tests ------------------------------------------------------------------


def test_known_calendar_slug_returns_full_surface_and_flags_eligible(
    eligible_client: TestClient,
) -> None:
    """A canonical slug returns its surface, the λ-ratio, and the trade flag."""
    slug = "trump-out-as-president-by-june-30"
    r = eligible_client.get(f"/terminal/calendar-pair/{slug}")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body is not None
    assert body["slug"] == slug
    assert body["event_token"] == "out president trump"

    # The surface must contain BOTH legs sorted by deadline.
    surface = body["surface"]
    assert len(surface) == 2
    deadlines = [leg["deadline"] for leg in surface]
    assert deadlines == sorted(deadlines)
    assert deadlines[0] == "2026-06-30"
    assert deadlines[-1] == "2026-12-31"

    # Implied λ must match the closed-form formula at every leg.
    for leg in surface:
        expected = _implied_lambda(leg["current_p"], leg["days_to_resolution"])
        assert math.isclose(leg["implied_lambda"], expected, rel_tol=1e-9)

    # With strong dispersion the |log-ratio| crosses the threshold.
    assert body["trade_eligible"] is True
    assert abs(body["log_lambda_ratio"]) >= LOG_LAMBDA_RATIO_THRESHOLD
    assert body["lambda_near"] < body["lambda_far"]


def test_unknown_slug_returns_null(eligible_client: TestClient) -> None:
    """Slugs without a calendar partner return JSON ``null`` (not 404)."""
    r = eligible_client.get("/terminal/calendar-pair/some-random-non-calendar-slug")
    assert r.status_code == 200
    assert r.json() is None


def test_below_threshold_pair_is_not_trade_eligible(benign_client: TestClient) -> None:
    """A pair whose log-λ-ratio is small is NOT flagged as Strategy-24 eligible.

    The endpoint still returns the full surface (so the Terminal can render
    the term-structure chart) — only the boolean differs.
    """
    # Synthetic id-slug path: the strat-2 cluster for "amazon best has model"
    # was *not* written, so the endpoint must fall back to the id-derived slug.
    slug = "amazon-best-ai-may"
    r = benign_client.get(f"/terminal/calendar-pair/{slug}")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body is not None
    assert body["event_token"] == "amazon best has model"
    assert len(body["surface"]) == 2
    assert body["trade_eligible"] is False
    assert abs(body["log_lambda_ratio"]) < LOG_LAMBDA_RATIO_THRESHOLD


# --- additional coverage ----------------------------------------------------


def test_implied_lambda_handles_degenerate_inputs() -> None:
    """The closed-form λ helper must short-circuit on p<=0 / T<=0."""
    assert _implied_lambda(0.0, 30) == 0.0
    assert _implied_lambda(-0.1, 30) == 0.0
    assert _implied_lambda(0.5, 0) == 0.0
    assert _implied_lambda(0.5, -10) == 0.0
    # And it stays finite at p == 1 (clipping kicks in).
    assert math.isfinite(_implied_lambda(1.0, 30))
    # Monotone in p for fixed T.
    assert _implied_lambda(0.1, 30) < _implied_lambda(0.5, 30) < _implied_lambda(0.9, 30)


def test_response_schema_keys_match_pydantic_contract(eligible_client: TestClient) -> None:
    """The wire JSON has exactly the documented top-level keys + leg keys."""
    slug = "trump-out-as-president-by-june-30"
    body = eligible_client.get(f"/terminal/calendar-pair/{slug}").json()
    expected_top = {
        "slug",
        "event_token",
        "surface",
        "lambda_near",
        "lambda_far",
        "log_lambda_ratio",
        "trade_eligible",
    }
    assert set(body.keys()) == expected_top
    expected_leg = {
        "slug",
        "deadline",
        "current_p",
        "days_to_resolution",
        "implied_lambda",
    }
    for leg in body["surface"]:
        assert set(leg.keys()) == expected_leg


def test_path_too_long_is_rejected(eligible_client: TestClient) -> None:
    """Per the FPath constraint, slugs > 200 chars must 422 before any lookup."""
    long_slug = "x" * 300
    r = eligible_client.get(f"/terminal/calendar-pair/{long_slug}")
    assert r.status_code == 422
