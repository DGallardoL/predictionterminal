"""Tests for ``GET /strategies/anti-alpha-list`` (W11-23).

The router (``pfm.strategies.anti_alpha_router``) is not wired into the
running app — ``main.py:routes`` is held by another active coordination
claim. Each test therefore mounts the router into a fresh ``FastAPI``
instance and overrides the source-file paths via the module's environment
variables (``PFM_ANTI_ALPHA_STRATEGIES_PATH`` /
``PFM_ANTI_ALPHA_GRAVEYARD_PATH`` / ``PFM_ANTI_ALPHA_REPORTS_DIR``).

These tests deliberately avoid the real ``web/data/*.json`` files so the
suite is hermetic and doesn't break if the catalog churns.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pfm.strategies import anti_alpha_router as mod

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_app() -> FastAPI:
    """Mount the anti-alpha router into a fresh FastAPI instance."""
    app = FastAPI()
    app.include_router(mod.router)
    return app


@pytest.fixture(autouse=True)
def _clear_cache() -> Iterator[None]:
    """Clear the module's in-process cache before AND after each test."""
    mod.cache_clear()
    yield
    mod.cache_clear()


@pytest.fixture
def synthetic_sources(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    """Build synthetic alpha_strategies.json + alpha_graveyard.json fixtures.

    Lays down:
        * 3 graveyard entries (treated as ANTI by the router)
        * 5 strategies entries — 3 should pass the C_*/D_*/ANTI filter,
          2 are B_VALIDATED (deployable) and must be filtered out
        * a reports dir with ``alpha-report-v16.md`` so ``report_version``
          resolves to 16
    """
    grave_path = tmp_path / "alpha_graveyard.json"
    strat_path = tmp_path / "alpha_strategies.json"
    reports_dir = tmp_path / "alpha-reports"
    reports_dir.mkdir()
    (reports_dir / "alpha-report-v16.md").write_text("# v16\n")
    (reports_dir / "alpha-report-v15.md").write_text("# v15\n")

    grave_path.write_text(
        json.dumps(
            [
                {
                    "pair_id": "recession_odds_defensive_long",
                    "name": "Recession-odds → Defensive-sector long",
                    "killed_in_wave": 3,
                    "cause": "regime",
                    "lesson": (
                        "Backtest dominated by Q4-2024 risk-off regime. Sign flipped in Q1-2025."
                    ),
                },
                {
                    "pair_id": "crypto_etf_approval_drift",
                    "name": "Crypto-ETF approval drift",
                    "killed_in_wave": 4,
                    "cause": "single-episode",
                    "lesson": "All gross PnL came from one ETF approval.",
                },
                {
                    "pair_id": "favorites_bias",
                    "name": "Favorites-bias",
                    "killed_in_wave": 5,
                    "cause": "regime",
                    "lesson": "Wave-5 stress test killed it.",
                },
            ]
        )
    )

    strat_path.write_text(
        json.dumps(
            {
                "generated": "2026-05-16",
                "strategies": [
                    {
                        "pair_id": "tentative_pair_a",
                        "a_name": "Tentative A",
                        "tier": "C_TENTATIVE",
                        "v17_reclassification_reason": "FDR-q10 only",
                    },
                    {
                        "pair_id": "raw_pair_b",
                        "a_name": "Raw B",
                        "tier": "D_RAW",
                        "rationale": "speculative raw hit",
                    },
                    {
                        "pair_id": "explicit_anti_c",
                        "a_name": "Explicit Anti C",
                        "tier": "ANTI",
                        "rationale": "manually flagged",
                    },
                    {
                        "pair_id": "deployable_validated",
                        "a_name": "Deployable B-validated",
                        "tier": "B_VALIDATED",
                        "rationale": "should NOT appear",
                    },
                    {
                        "pair_id": "deployable_gold",
                        "a_name": "Deployable Gold",
                        "tier": "A_GOLD",
                        "rationale": "should NOT appear",
                    },
                ],
            }
        )
    )

    monkeypatch.setenv(mod._ALPHA_GRAVEYARD_PATH_ENV, str(grave_path))
    monkeypatch.setenv(mod._ALPHA_STRATEGIES_PATH_ENV, str(strat_path))
    monkeypatch.setenv(mod._ALPHA_REPORTS_DIR_ENV, str(reports_dir))

    return {
        "graveyard": grave_path,
        "strategies": strat_path,
        "reports": reports_dir,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_200_default_returns_envelope(
    synthetic_sources: dict[str, Path],
) -> None:
    """Endpoint returns 200 with count + items[] envelope."""
    client = TestClient(_make_app())
    r = client.get("/strategies/anti-alpha-list")
    assert r.status_code == 200
    body = r.json()
    assert "count" in body
    assert "items" in body
    assert isinstance(body["items"], list)
    # 3 from graveyard + 3 C/D/ANTI from strategies = 6
    assert body["count"] == 6
    assert len(body["items"]) == 6
    # Every item has the required schema fields.
    for it in body["items"]:
        assert {"pair_id", "tier", "label", "reason"}.issubset(it.keys())


def test_only_anti_and_c_d_strategies_pass_filter(
    synthetic_sources: dict[str, Path],
) -> None:
    """B_VALIDATED + A_GOLD entries from strategies.json must be excluded."""
    client = TestClient(_make_app())
    body = client.get("/strategies/anti-alpha-list").json()
    pair_ids = {it["pair_id"] for it in body["items"]}
    assert "deployable_validated" not in pair_ids
    assert "deployable_gold" not in pair_ids
    # And the C/D/ANTI strategies DO appear.
    assert "tentative_pair_a" in pair_ids
    assert "raw_pair_b" in pair_ids
    assert "explicit_anti_c" in pair_ids


def test_graveyard_entries_marked_as_anti_tier(
    synthetic_sources: dict[str, Path],
) -> None:
    """Every graveyard entry must surface with tier='ANTI'."""
    client = TestClient(_make_app())
    body = client.get("/strategies/anti-alpha-list").json()
    by_id = {it["pair_id"]: it for it in body["items"]}
    assert by_id["recession_odds_defensive_long"]["tier"] == "ANTI"
    assert by_id["crypto_etf_approval_drift"]["tier"] == "ANTI"
    assert by_id["favorites_bias"]["tier"] == "ANTI"
    # Wave number flows through.
    assert by_id["recession_odds_defensive_long"]["demoted_in_wave"] == 3
    assert by_id["crypto_etf_approval_drift"]["demoted_in_wave"] == 4
    # report_version is the latest report file (v16, since we put v15+v16
    # into reports_dir).
    assert by_id["recession_odds_defensive_long"]["report_version"] == 16


def test_missing_source_files_uses_claude_md_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When BOTH source files are missing, the CLAUDE.md fallback fires.

    The fallback list has the 4 entries documented in CLAUDE.md.
    """
    grave = tmp_path / "no_grave.json"
    strat = tmp_path / "no_strat.json"
    reports = tmp_path / "no_reports"
    monkeypatch.setenv(mod._ALPHA_GRAVEYARD_PATH_ENV, str(grave))
    monkeypatch.setenv(mod._ALPHA_STRATEGIES_PATH_ENV, str(strat))
    monkeypatch.setenv(mod._ALPHA_REPORTS_DIR_ENV, str(reports))

    client = TestClient(_make_app())
    r = client.get("/strategies/anti-alpha-list")
    assert r.status_code == 200
    body = r.json()
    # All fallback rows are tier=ANTI.
    assert body["count"] == 4
    assert all(it["tier"] == "ANTI" for it in body["items"])
    # Diagnostic note must signal the fallback path.
    notes = " ".join(body["source_notes"])
    assert "fallback" in notes.lower()
    assert "claude" in notes.lower()


def test_caching_within_ttl_avoids_disk_reload(
    synthetic_sources: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two back-to-back calls hit the cache (no second disk read)."""
    calls = {"n": 0}
    real_safe_read = mod._safe_read_json

    def counting_read(path: Path, notes: list[str]):  # type: ignore[no-untyped-def]
        calls["n"] += 1
        return real_safe_read(path, notes)

    monkeypatch.setattr(mod, "_safe_read_json", counting_read)

    client = TestClient(_make_app())
    r1 = client.get("/strategies/anti-alpha-list")
    assert r1.status_code == 200
    n_after_first = calls["n"]
    assert n_after_first >= 2  # graveyard + strategies were both read

    r2 = client.get("/strategies/anti-alpha-list")
    assert r2.status_code == 200
    # Second call MUST be served from cache — no new disk reads.
    assert calls["n"] == n_after_first
    # And the payload is byte-identical.
    assert r1.json() == r2.json()


def test_cache_clear_forces_reload(
    synthetic_sources: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """``cache_clear()`` invalidates the cache and re-reads disk."""
    calls = {"n": 0}
    real_safe_read = mod._safe_read_json

    def counting_read(path: Path, notes: list[str]):  # type: ignore[no-untyped-def]
        calls["n"] += 1
        return real_safe_read(path, notes)

    monkeypatch.setattr(mod, "_safe_read_json", counting_read)

    client = TestClient(_make_app())
    client.get("/strategies/anti-alpha-list")
    n_first = calls["n"]
    mod.cache_clear()
    client.get("/strategies/anti-alpha-list")
    assert calls["n"] > n_first


def test_tier_filter_exact_match(
    synthetic_sources: dict[str, Path],
) -> None:
    """``?tier=ANTI`` returns only graveyard + explicit ANTI strategy rows."""
    client = TestClient(_make_app())
    r = client.get("/strategies/anti-alpha-list", params={"tier": "ANTI"})
    assert r.status_code == 200
    body = r.json()
    # 3 graveyard + 1 explicit_anti_c = 4
    assert body["count"] == 4
    assert all(it["tier"] == "ANTI" for it in body["items"])
    pair_ids = {it["pair_id"] for it in body["items"]}
    assert "explicit_anti_c" in pair_ids
    assert "tentative_pair_a" not in pair_ids


def test_tier_filter_wildcard_prefix(
    synthetic_sources: dict[str, Path],
) -> None:
    """``?tier=C_*`` returns only ``C_TENTATIVE`` (no D or ANTI)."""
    client = TestClient(_make_app())
    body = client.get("/strategies/anti-alpha-list", params={"tier": "C_*"}).json()
    assert body["count"] == 1
    assert body["items"][0]["tier"] == "C_TENTATIVE"
    assert body["items"][0]["pair_id"] == "tentative_pair_a"


def test_dedupe_by_pair_id_graveyard_wins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """If the same pair_id appears in both files, graveyard wins (tier=ANTI)."""
    grave_path = tmp_path / "g.json"
    strat_path = tmp_path / "s.json"
    grave_path.write_text(
        json.dumps(
            [
                {
                    "pair_id": "shared_id",
                    "name": "Shared (graveyard)",
                    "killed_in_wave": 7,
                    "cause": "regime",
                    "lesson": "killed for cause.",
                }
            ]
        )
    )
    strat_path.write_text(
        json.dumps(
            {
                "strategies": [
                    {
                        "pair_id": "shared_id",
                        "a_name": "Shared (strategies)",
                        "tier": "C_TENTATIVE",
                        "rationale": "watchlist only",
                    }
                ]
            }
        )
    )
    monkeypatch.setenv(mod._ALPHA_GRAVEYARD_PATH_ENV, str(grave_path))
    monkeypatch.setenv(mod._ALPHA_STRATEGIES_PATH_ENV, str(strat_path))
    monkeypatch.setenv(mod._ALPHA_REPORTS_DIR_ENV, str(tmp_path / "nonexistent"))

    client = TestClient(_make_app())
    body = client.get("/strategies/anti-alpha-list").json()
    assert body["count"] == 1
    assert body["items"][0]["tier"] == "ANTI"
    assert body["items"][0]["label"] == "Shared (graveyard)"


def test_malformed_strategies_json_falls_back_gracefully(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Corrupt JSON on the strategies file → router still serves graveyard."""
    grave_path = tmp_path / "g.json"
    strat_path = tmp_path / "s.json"
    grave_path.write_text(
        json.dumps(
            [
                {
                    "pair_id": "alive_grave",
                    "name": "Alive in graveyard",
                    "killed_in_wave": 2,
                    "cause": "regime",
                    "lesson": "ok.",
                }
            ]
        )
    )
    strat_path.write_text("{this is not valid json")
    monkeypatch.setenv(mod._ALPHA_GRAVEYARD_PATH_ENV, str(grave_path))
    monkeypatch.setenv(mod._ALPHA_STRATEGIES_PATH_ENV, str(strat_path))
    monkeypatch.setenv(mod._ALPHA_REPORTS_DIR_ENV, str(tmp_path / "nonexistent"))

    client = TestClient(_make_app())
    r = client.get("/strategies/anti-alpha-list")
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 1
    assert body["items"][0]["pair_id"] == "alive_grave"
    notes_joined = " ".join(body["source_notes"]).lower()
    assert "unreadable" in notes_joined or "jsondecodeerror" in notes_joined
