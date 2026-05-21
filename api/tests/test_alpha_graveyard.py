"""Tests for the public Alpha Graveyard registry, library and router."""

from __future__ import annotations

import json

from fastapi import FastAPI
from fastapi.testclient import TestClient

from pfm.alpha_graveyard import (
    GraveyardEntry,
    filter_by_cause,
    get_graveyard_path,
    load_graveyard,
)
from pfm.alpha_graveyard_router import router as graveyard_router

# ---------------------------------------------------------------------------
# Required schema fields — keep in sync with GraveyardEntry.
# ---------------------------------------------------------------------------

REQUIRED_FIELDS: set[str] = {
    "pair_id",
    "name",
    "killed_iso",
    "killed_in_wave",
    "cause",
    "claimed_sharpe",
    "post_mortem_sharpe",
    "thesis_original",
    "lesson",
    "could_resurrect_if",
    "tags",
    "death_certificate_md",
}

VALID_CAUSES: set[str] = {
    "regime",
    "TC",
    "single-episode",
    "grid-search",
    "tautology",
    "capacity",
    "non-portable",
}


# ---------------------------------------------------------------------------
# JSON registry tests
# ---------------------------------------------------------------------------


def test_graveyard_json_exists_and_is_valid_list() -> None:
    path = get_graveyard_path()
    assert path.exists(), f"alpha_graveyard.json missing at {path}"
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(raw, list)
    assert len(raw) >= 6, f"expected ≥6 graveyard entries, got {len(raw)}"


def test_load_graveyard_returns_list_of_dicts() -> None:
    entries = load_graveyard()
    assert isinstance(entries, list)
    assert len(entries) >= 6
    for e in entries:
        assert isinstance(e, dict)


def test_every_entry_has_required_fields() -> None:
    entries = load_graveyard()
    for e in entries:
        missing = REQUIRED_FIELDS - set(e.keys())
        assert not missing, f"entry {e.get('pair_id')} missing fields: {missing}"


def test_every_entry_validates_against_pydantic_model() -> None:
    entries = load_graveyard()
    for e in entries:
        # Will raise pydantic.ValidationError on schema mismatch.
        GraveyardEntry.model_validate(e)


def test_pair_ids_are_unique() -> None:
    entries = load_graveyard()
    ids = [e["pair_id"] for e in entries]
    assert len(ids) == len(set(ids)), "duplicate pair_id found in graveyard"


def test_causes_are_in_closed_vocabulary() -> None:
    entries = load_graveyard()
    for e in entries:
        assert e["cause"] in VALID_CAUSES, f"entry {e['pair_id']} has invalid cause '{e['cause']}'"


def test_known_anti_alphas_are_present() -> None:
    """The 6 anti-alphas mentioned in CLAUDE.md must all be in the graveyard."""
    entries = load_graveyard()
    ids = {e["pair_id"] for e in entries}
    must_have = {
        "recession_odds_defensive_long",
        "crypto_etf_approval_drift",
        "senate_control_short_vol",
        "geopolitical_conflict_oil_long",
        "btc_latency_arb",
        "favorites_bias",
    }
    missing = must_have - ids
    assert not missing, f"required anti-alphas missing: {missing}"


# ---------------------------------------------------------------------------
# Filter helper tests
# ---------------------------------------------------------------------------


def test_filter_by_cause_all_returns_full_list() -> None:
    entries = load_graveyard()
    assert filter_by_cause(entries, "all") == entries


def test_filter_by_cause_regime_keeps_only_regime() -> None:
    entries = load_graveyard()
    out = filter_by_cause(entries, "regime")
    assert len(out) >= 1
    assert all(e["cause"] == "regime" for e in out)


def test_filter_by_cause_unknown_returns_empty() -> None:
    entries = load_graveyard()
    # tautology has no current entries; the function must still return a list.
    out = filter_by_cause(entries, "tautology")
    assert out == []


# ---------------------------------------------------------------------------
# Router tests
# ---------------------------------------------------------------------------


def _build_test_client() -> TestClient:
    app = FastAPI()
    app.include_router(graveyard_router)
    return TestClient(app)


def test_router_lists_graveyard() -> None:
    client = _build_test_client()
    r = client.get("/alpha-hub/graveyard")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["cause_filter"] == "all"
    assert body["n_entries"] >= 6
    assert len(body["entries"]) == body["n_entries"]
    # Spot-check a known entry.
    ids = {e["pair_id"] for e in body["entries"]}
    assert "btc_latency_arb" in ids


def test_router_filters_by_cause() -> None:
    client = _build_test_client()
    r = client.get("/alpha-hub/graveyard", params={"cause": "regime"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["cause_filter"] == "regime"
    assert body["n_entries"] >= 1
    assert all(e["cause"] == "regime" for e in body["entries"])


def test_router_rejects_unknown_cause() -> None:
    client = _build_test_client()
    r = client.get("/alpha-hub/graveyard", params={"cause": "not_a_real_cause"})
    # Pydantic / FastAPI rejects values outside the Literal vocabulary.
    assert r.status_code == 422


def test_router_detail_endpoint_returns_entry() -> None:
    client = _build_test_client()
    r = client.get("/alpha-hub/graveyard/btc_latency_arb")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["pair_id"] == "btc_latency_arb"
    assert body["cause"] == "non-portable"
    assert "thesis_original" in body
    assert "lesson" in body


def test_router_detail_endpoint_404_on_missing() -> None:
    client = _build_test_client()
    r = client.get("/alpha-hub/graveyard/does_not_exist")
    assert r.status_code == 404
    assert "not found" in r.json()["detail"].lower()
