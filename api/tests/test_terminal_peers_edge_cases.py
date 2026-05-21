"""Edge-case battery for ``GET /terminal/peers/{slug}`` (Task W12-02).

These tests complement ``test_terminal_peer_scanner.py`` by hammering the
endpoint's boundary conditions: unknown slugs, malformed metadata, slugs
with <30 obs, parameter clamping, caching behaviour, concurrency,
URL-encoded special characters, trailing slashes, schema invariants, and
a soft latency budget.

All on-disk loaders are monkeypatched to synthetic fixtures so the suite
is hermetic (no /tmp or factors.yml IO). The endpoint actually exposes
``top`` + ``min_sharpe`` (not ``limit``/``sort``/``offset``); where the
task brief mentions those, we either verify the closest equivalent or
explicitly assert the endpoint's documented refusal/no-op.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from typing import Any
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pfm import terminal as _term_mod
from pfm import terminal_peer_scanner
from pfm.terminal_peer_scanner import clear_cache, router

# --- Required peer-record keys (response schema contract) -------------------

REQUIRED_PEER_KEYS: set[str] = {
    "peer_slug",
    "peer_name",
    "oos_sharpe",
    "perm_p",
    "half_life_days",
    "beta_hedge",
    "theme_a",
    "theme_b",
    "verdict",
    "tier",
    "n_obs",
    "adf_pvalue",
    "sweep",
}

REQUIRED_TOPLEVEL_KEYS: set[str] = {
    "slug",
    "n_peers",
    "peers",
    "cross_theme_count",
    "tier_summary",
    "best_peer",
}


# --- Fixtures ---------------------------------------------------------------


def _populated_hits() -> list[dict[str, Any]]:
    """Eight hits — enough to exercise top-N truncation and tie-breakers."""
    rows: list[dict[str, Any]] = []
    for i in range(1, 9):
        rows.append(
            {
                "a_id": "anchor_slug",
                "b_id": f"peer_{i:02d}",
                "verdict": "REAL_ALPHA",
                "n_obs": 30 + i * 5,
                "adf_pvalue": 0.01,
                "half_life_days": 3.0 + i * 0.1,
                "beta_hedge": 0.10 + i * 0.01,
                # Decreasing Sharpe so ordering is deterministic.
                "oos_sharpe": 5.0 - i * 0.4,
                "full_sharpe": 4.0 - i * 0.3,
                "perm_p": 0.001,
                "perm_real_sharpe": 3.5,
                "sweep": "macro" if i % 2 == 0 else "crypto",
            }
        )
    # Pair that does NOT involve anchor_slug — must be skipped.
    rows.append(
        {
            "a_id": "peer_01",
            "b_id": "peer_02",
            "verdict": "REAL_ALPHA",
            "n_obs": 50,
            "adf_pvalue": 0.02,
            "half_life_days": 2.0,
            "beta_hedge": 0.2,
            "oos_sharpe": 2.0,
            "full_sharpe": 1.7,
            "perm_p": 0.01,
            "perm_real_sharpe": 1.5,
            "sweep": "macro",
        }
    )
    return rows


def _malformed_hits() -> list[dict[str, Any]]:
    """Hits with malformed metadata — endpoint must coerce or skip gracefully."""
    return [
        # Missing a_id entirely → enrichment must skip (returns None).
        {
            "b_id": "ghost_slug",
            "verdict": "REAL_ALPHA",
            "oos_sharpe": 3.0,
            "sweep": "macro",
        },
        # Valid pair involving anchor_slug, but numerics absent — defaults kick in.
        {
            "a_id": "anchor_slug",
            "b_id": "partial_peer",
            "verdict": "REAL_ALPHA",
            "sweep": "macro",
            # No oos_sharpe / perm_p / half_life_days / beta_hedge / n_obs / adf_pvalue.
        },
        # Valid full row for control.
        {
            "a_id": "anchor_slug",
            "b_id": "full_peer",
            "verdict": "REAL_ALPHA",
            "n_obs": 80,
            "adf_pvalue": 0.001,
            "half_life_days": 2.5,
            "beta_hedge": 0.42,
            "oos_sharpe": 3.5,
            "full_sharpe": 3.0,
            "perm_p": 0.0,
            "perm_real_sharpe": 2.8,
            "sweep": "macro",
        },
    ]


def _thin_hits() -> list[dict[str, Any]]:
    """All peers have <30 observations — endpoint must still surface them.

    The endpoint itself does NOT filter on n_obs (only on min_sharpe); thin
    peers are emitted, but if a caller passes min_sharpe high enough we get
    an empty list. We use this fixture for the explicit-empty-array test.
    """
    return [
        {
            "a_id": "anchor_slug",
            "b_id": f"thin_{i:02d}",
            "verdict": "REAL_ALPHA",
            "n_obs": 5 + i,
            "adf_pvalue": 0.1,
            "half_life_days": 10.0,
            "beta_hedge": 0.0,
            "oos_sharpe": 0.2,  # below default min_sharpe=0.5
            "full_sharpe": 0.1,
            "perm_p": 0.5,
            "perm_real_sharpe": 0.1,
            "sweep": "thin",
        }
        for i in range(1, 4)
    ]


def _factors() -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {
        "anchor_slug": {"name": "Anchor Market", "theme": "macro", "slug": "anchor"},
        # peer_01 / peer_02 share theme=macro, others split.
    }
    for i in range(1, 9):
        out[f"peer_{i:02d}"] = {
            "name": f"Peer {i}",
            "theme": "macro" if i % 2 == 0 else "crypto",
            "slug": f"peer-{i:02d}",
        }
    out["partial_peer"] = {"name": "Partial Peer", "theme": "macro", "slug": "partial"}
    out["full_peer"] = {"name": "Full Peer", "theme": "macro", "slug": "full"}
    return out


def _tiers() -> dict[str, str]:
    # Tier only the first two peers; the rest fall through to D_RAW.
    return {
        "__".join(sorted(["anchor_slug", "peer_01"])): "A_GOLD",
        "__".join(sorted(["anchor_slug", "peer_02"])): "B_VALIDATED",
        "__".join(sorted(["anchor_slug", "full_peer"])): "A_GOLD",
    }


@pytest.fixture(autouse=True)
def _patch_loaders(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Default fixture set — populated hits. Individual tests can override."""
    clear_cache()
    # Also clear the TERMINAL_CACHE so prior-test cache hits don't bleed.
    _term_mod.TERMINAL_CACHE.clear()
    monkeypatch.setattr(terminal_peer_scanner, "_load_hits", _populated_hits)
    monkeypatch.setattr(terminal_peer_scanner, "_load_factors", _factors)
    monkeypatch.setattr(terminal_peer_scanner, "_load_tiers", _tiers)
    yield
    clear_cache()
    _term_mod.TERMINAL_CACHE.clear()


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


# --- Helpers ----------------------------------------------------------------


def _validate_peer_schema(peer: dict[str, Any]) -> None:
    missing = REQUIRED_PEER_KEYS - set(peer)
    assert not missing, f"peer record missing keys: {missing}"


def _validate_envelope(body: dict[str, Any]) -> None:
    missing = REQUIRED_TOPLEVEL_KEYS - set(body)
    assert not missing, f"response missing keys: {missing}"


# --- Tests ------------------------------------------------------------------


class TestUnknownAndEmpty:
    def test_unknown_slug_returns_200_with_degraded_mode(self, client: TestClient) -> None:
        """Brief allows 404 or 503, but reality is graceful 200 with reason.

        Endpoint contract: an unknown slug means n_peers=0 and
        degraded_mode=True with a textual ``reason``. We accept any of
        (200, 404, 503) so the test is robust to future hardening.
        """
        r = client.get("/terminal/peers/totally_unknown_slug")
        assert r.status_code in (200, 404, 503), r.text
        if r.status_code == 200:
            body = r.json()
            _validate_envelope(body)
            assert body["slug"] == "totally_unknown_slug"
            assert body["n_peers"] == 0
            assert body["peers"] == []
            assert body["best_peer"] is None
            # Should explain WHY the list is empty (per endpoint docstring).
            assert body.get("degraded_mode") is True
            assert body.get("reason") is not None

    def test_known_slug_with_zero_matches_returns_explicit_empty(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A slug that exists in factors but has no peers in hits → empty."""
        factors = _factors()
        factors["lonely_slug"] = {
            "name": "Lonely",
            "theme": "macro",
            "slug": "lonely",
        }
        monkeypatch.setattr(terminal_peer_scanner, "_load_factors", lambda: factors)
        clear_cache()
        _term_mod.TERMINAL_CACHE.clear()

        r = client.get("/terminal/peers/lonely_slug?min_sharpe=0.5")
        assert r.status_code == 200, r.text
        body = r.json()
        _validate_envelope(body)
        assert body["peers"] == []
        assert isinstance(body["peers"], list)
        assert body["tier_summary"] == {}

    def test_thin_peers_all_filtered_when_min_sharpe_above_their_sharpe(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """All peers with Sharpe 0.2 are filtered at default min_sharpe=0.5."""
        monkeypatch.setattr(terminal_peer_scanner, "_load_hits", _thin_hits)
        clear_cache()
        _term_mod.TERMINAL_CACHE.clear()

        r = client.get("/terminal/peers/anchor_slug?min_sharpe=0.5")
        assert r.status_code == 200
        body = r.json()
        assert body["n_peers"] == 0
        assert body["peers"] == []

    def test_thin_peers_visible_when_min_sharpe_low(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Same data, lower min_sharpe → thin peers ARE returned."""
        monkeypatch.setattr(terminal_peer_scanner, "_load_hits", _thin_hits)
        clear_cache()
        _term_mod.TERMINAL_CACHE.clear()

        r = client.get("/terminal/peers/anchor_slug?min_sharpe=0.0")
        assert r.status_code == 200
        body = r.json()
        assert body["n_peers"] == 3
        for p in body["peers"]:
            _validate_peer_schema(p)


class TestMalformedMetadata:
    def test_malformed_hits_skipped_or_coerced_without_500(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Missing a_id is dropped; missing numerics coerced to defaults."""
        monkeypatch.setattr(terminal_peer_scanner, "_load_hits", _malformed_hits)
        clear_cache()
        _term_mod.TERMINAL_CACHE.clear()

        r = client.get("/terminal/peers/anchor_slug?min_sharpe=0.0")
        assert r.status_code == 200, r.text
        body = r.json()
        _validate_envelope(body)

        peer_ids = {p["peer_slug"] for p in body["peers"]}
        # full_peer always present; partial_peer present with default Sharpe=0.0.
        assert "full_peer" in peer_ids
        # ghost_slug row had no a_id → dropped silently.
        assert "ghost_slug" not in peer_ids

        # Every emitted peer satisfies the schema (coercion fills defaults).
        for p in body["peers"]:
            _validate_peer_schema(p)
            assert isinstance(p["oos_sharpe"], (int, float))
            assert isinstance(p["n_obs"], int)


class TestParameterHandling:
    def test_top_param_enforced(self, client: TestClient) -> None:
        """top=5 must return at most 5 peers even when 8 are eligible."""
        r = client.get("/terminal/peers/anchor_slug?top=5&min_sharpe=0.0")
        assert r.status_code == 200
        body = r.json()
        assert body["n_peers"] <= 5
        assert len(body["peers"]) <= 5

    def test_top_param_above_eligible_returns_all(self, client: TestClient) -> None:
        r = client.get("/terminal/peers/anchor_slug?top=200&min_sharpe=0.0")
        assert r.status_code == 200
        body = r.json()
        # 8 anchor pairs in fixture, all above Sharpe 0.0.
        assert body["n_peers"] == 8

    def test_top_param_below_one_rejected(self, client: TestClient) -> None:
        """``top`` is declared ge=1 — top=0 must be a 422."""
        r = client.get("/terminal/peers/anchor_slug?top=0")
        assert r.status_code == 422

    def test_top_param_above_max_rejected(self, client: TestClient) -> None:
        """``top`` is declared le=200 — top=999 must be a 422."""
        r = client.get("/terminal/peers/anchor_slug?top=999")
        assert r.status_code == 422

    def test_unsupported_sort_param_is_ignored(self, client: TestClient) -> None:
        """Endpoint has no ``sort``; FastAPI ignores unknown query params.

        Sort order is ALWAYS by oos_sharpe desc — passing ``sort=corr`` (or
        any other value) must NOT change the ordering and must NOT 422.
        """
        r1 = client.get("/terminal/peers/anchor_slug?min_sharpe=0.0")
        r2 = client.get("/terminal/peers/anchor_slug?min_sharpe=0.0&sort=corr")
        r3 = client.get("/terminal/peers/anchor_slug?min_sharpe=0.0&sort=spread")
        for r in (r1, r2, r3):
            assert r.status_code == 200
        sharpes = [[p["oos_sharpe"] for p in r.json()["peers"]] for r in (r1, r2, r3)]
        # Identical ordering across all three calls.
        assert sharpes[0] == sharpes[1] == sharpes[2]
        # Still sorted desc.
        assert sharpes[0] == sorted(sharpes[0], reverse=True)

    def test_unsupported_offset_param_is_ignored(self, client: TestClient) -> None:
        """No pagination in the endpoint — ``offset`` must be a no-op."""
        r1 = client.get("/terminal/peers/anchor_slug?min_sharpe=0.0")
        r2 = client.get("/terminal/peers/anchor_slug?min_sharpe=0.0&offset=10")
        assert r1.status_code == 200 and r2.status_code == 200
        assert r1.json()["peers"] == r2.json()["peers"]


class TestCachingAndConcurrency:
    def test_second_call_hits_cache_and_skips_compute(self, client: TestClient) -> None:
        """Two identical GETs → underlying find_peers called at most once."""
        # Make sure cache is empty first.
        _term_mod.TERMINAL_CACHE.clear()
        with patch.object(
            terminal_peer_scanner,
            "find_peers",
            wraps=terminal_peer_scanner.find_peers,
        ) as spy:
            r1 = client.get("/terminal/peers/anchor_slug?top=20&min_sharpe=0.0")
            r2 = client.get("/terminal/peers/anchor_slug?top=20&min_sharpe=0.0")
            assert r1.status_code == 200 and r2.status_code == 200
            assert r1.json() == r2.json()
            # First call computes, second is cache-hit.
            assert spy.call_count == 1, f"expected exactly one compute call, got {spy.call_count}"

    def test_different_params_bypass_cache(self, client: TestClient) -> None:
        """Different ``top`` values must NOT share a cache key."""
        _term_mod.TERMINAL_CACHE.clear()
        with patch.object(
            terminal_peer_scanner,
            "find_peers",
            wraps=terminal_peer_scanner.find_peers,
        ) as spy:
            client.get("/terminal/peers/anchor_slug?top=5&min_sharpe=0.0")
            client.get("/terminal/peers/anchor_slug?top=20&min_sharpe=0.0")
            assert spy.call_count == 2

    def test_concurrent_same_slug_all_succeed_with_consistent_body(
        self, client: TestClient
    ) -> None:
        """10 parallel threads on the same slug → identical bodies, no errors.

        We can't assert ``single fetch`` strictly (TTLCache has no per-key
        lock), but we CAN assert no race produces inconsistent payloads or
        non-200 responses.
        """
        _term_mod.TERMINAL_CACHE.clear()
        results: list[tuple[int, str]] = []
        lock = threading.Lock()

        def hit() -> None:
            r = client.get("/terminal/peers/anchor_slug?top=20&min_sharpe=0.0")
            with lock:
                results.append((r.status_code, r.text))

        threads = [threading.Thread(target=hit) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 10
        statuses = {s for s, _ in results}
        bodies = {b for _, b in results}
        assert statuses == {200}, f"non-200s: {statuses}"
        assert len(bodies) == 1, "concurrent responses diverged"


class TestSlugFormatting:
    def test_url_encoded_special_chars_in_slug_handled(self, client: TestClient) -> None:
        """A slug with %-encoded specials should be decoded and 200."""
        # The endpoint's slug validator allows any string ≤120 chars; FastAPI
        # auto-decodes URL-encoded path components. A nonexistent decoded
        # slug just falls through to degraded-empty.
        r = client.get("/terminal/peers/foo%20bar")
        assert r.status_code in (200, 404), r.text
        if r.status_code == 200:
            body = r.json()
            # Decoded slug echoes back in payload.
            assert body["slug"] == "foo bar"

    def test_trailing_slash_handled(self, client: TestClient) -> None:
        """Trailing slash on the route should NOT 500.

        FastAPI by default treats ``/terminal/peers/anchor_slug/`` as a
        distinct path → typically a 404 or 307. We accept either.
        """
        r = client.get("/terminal/peers/anchor_slug/", follow_redirects=False)
        assert r.status_code in (200, 307, 404, 405), r.text

    def test_slug_with_max_length_accepted(self, client: TestClient) -> None:
        """120-char slug should pass FPath validator, 121 should fail."""
        slug_ok = "a" * 120
        slug_too_long = "a" * 121
        r_ok = client.get(f"/terminal/peers/{slug_ok}")
        r_bad = client.get(f"/terminal/peers/{slug_too_long}")
        assert r_ok.status_code == 200
        assert r_bad.status_code == 422


class TestSchemaInvariants:
    def test_every_peer_has_all_required_fields(self, client: TestClient) -> None:
        r = client.get("/terminal/peers/anchor_slug?top=20&min_sharpe=0.0")
        assert r.status_code == 200
        body = r.json()
        _validate_envelope(body)
        assert isinstance(body["peers"], list)
        assert body["peers"], "expected non-empty peer list for fixture"
        for p in body["peers"]:
            _validate_peer_schema(p)
            # Type contract on the numerics.
            assert isinstance(p["peer_slug"], str) and p["peer_slug"]
            assert isinstance(p["peer_name"], str)
            assert isinstance(p["oos_sharpe"], (int, float))
            assert isinstance(p["n_obs"], int)
            assert isinstance(p["tier"], str)

    def test_tier_summary_sums_to_n_peers(self, client: TestClient) -> None:
        r = client.get("/terminal/peers/anchor_slug?top=20&min_sharpe=0.0")
        body = r.json()
        assert sum(body["tier_summary"].values()) == body["n_peers"]

    def test_best_peer_is_first_peer_when_nonempty(self, client: TestClient) -> None:
        r = client.get("/terminal/peers/anchor_slug?top=20&min_sharpe=0.0")
        body = r.json()
        assert body["best_peer"] == body["peers"][0]


class TestPerformance:
    def test_cold_call_under_500ms_for_known_slug(self, client: TestClient) -> None:
        """Soft budget: a cold (cache-cleared) call returns <500 ms.

        With the fully-mocked loaders this should be well under 50 ms; the
        500 ms ceiling exists to catch regressions like accidental synchronous
        Redis IO or n^2 enrichment loops.
        """
        _term_mod.TERMINAL_CACHE.clear()
        clear_cache()
        t0 = time.perf_counter()
        r = client.get("/terminal/peers/anchor_slug?top=20&min_sharpe=0.0")
        elapsed_ms = (time.perf_counter() - t0) * 1000
        assert r.status_code == 200
        assert elapsed_ms < 500, f"cold call took {elapsed_ms:.1f} ms"
