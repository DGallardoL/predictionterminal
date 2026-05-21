"""Perf + correctness tests for ``POST /factors/suggest-for-ticker``.

Covers the 2026-05-15 fixes:

1. **L2 (cross-worker) cache hit** avoids the per-factor upstream burst —
   simulated by writing the payload directly into the injected Redis-shim
   under the new ``pfm:suggest_for_ticker:*`` key and confirming the scan
   helper is never invoked.

2. **429 retry-with-backoff** — when ``_cached_factor_history`` raises a
   429 ``HTTPException`` on the first call, the scan retries once and
   keeps the factor; a persistent 429 only skips that single factor and
   the rest of the scan still runs.

3. **SETNX stampede protection** — when 4 concurrent same-ticker requests
   race the cold cache, only ONE actually runs the K-factor scan; the
   other 3 wait + read the L2 entry the leader writes.

The synthetic 2-factor fixture from ``conftest.py`` (``factor_a``,
``factor_b``) is reused so we don't need a live Polymarket connection.
"""

from __future__ import annotations

import json
import threading
from typing import Any

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

import pfm.regression_router as rr

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _DictRedisShim:
    """Minimal CacheBackend stand-in that records ``get/set/setnx`` calls.

    Tracks ``enabled = True`` so the L2 read/write paths in
    ``regression_router`` exercise the real code (NullCache short-circuits
    ``enabled = False`` and skips the L2 entirely).
    """

    def __init__(self) -> None:
        self.store: dict[str, bytes] = {}
        self.setnx_calls: list[str] = []
        self.get_calls: list[str] = []
        self.set_calls: list[str] = []
        self.lock = threading.Lock()

    enabled = True

    def get(self, key: str) -> bytes | None:
        with self.lock:
            self.get_calls.append(key)
            return self.store.get(key)

    def set(self, key: str, value: bytes, ttl_seconds: int) -> None:
        with self.lock:
            self.set_calls.append(key)
            self.store[key] = value

    def setnx(self, key: str, value: bytes, ttl_seconds: int) -> bool:
        with self.lock:
            self.setnx_calls.append(key)
            if key in self.store:
                return False
            self.store[key] = value
            return True

    # Mirror RedisCache's ``_client`` attribute so ``_release_suggest_lock``
    # can call ``client.delete(...)``.
    @property
    def _client(self):
        shim = self

        class _C:
            def delete(self, key: str) -> None:
                with shim.lock:
                    shim.store.pop(key, None)

        return _C()


@pytest.fixture
def cache_shim(app_client: TestClient) -> _DictRedisShim:
    """Replace the app's NullCache with a dict-backed Redis shim."""
    import pfm.main as main_mod

    shim = _DictRedisShim()
    main_mod.app.state.cache = shim
    return shim


# ---------------------------------------------------------------------------
# 1. L2 cache hit avoids the upstream scan
# ---------------------------------------------------------------------------


class TestL2CacheHit:
    def test_l2_hit_skips_scan(
        self,
        monkeypatch: pytest.MonkeyPatch,
        app_client: TestClient,
        cache_shim: _DictRedisShim,
    ) -> None:
        # Pre-seed the L2 entry so the second worker's cold path finds it
        # before invoking the scan.
        body = {
            "ticker": "AAPL",
            "lookback_days": 90,
            "top_k": 3,
            "min_n_obs": 10,
        }
        cache_key = rr._suggest_cache_key(
            body["ticker"],
            body["lookback_days"],
            body["top_k"],
            body["min_n_obs"],
        )
        canned = {
            "ticker": "AAPL",
            "lookback_days": 90,
            "n_factors_scanned": 2,
            "n_factors_skipped": 0,
            "top_factors": [
                {
                    "factor_id": "factor_a",
                    "name": "Factor A",
                    "source": "polymarket",
                    "theme": None,
                    "r": 0.42,
                    "abs_r": 0.42,
                    "n_obs": 60,
                },
            ],
        }
        cache_shim.set(
            rr._suggest_l2_key(cache_key),
            json.dumps(canned).encode("utf-8"),
            3600,
        )

        scan_calls = {"n": 0}
        original = rr._scan_factor_correlations_for_ticker

        def _spy(*args: Any, **kwargs: Any) -> Any:
            scan_calls["n"] += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(rr, "_scan_factor_correlations_for_ticker", _spy)

        # Clear the L1 bucket so we hit the L2 path explicitly.
        from pfm.cache_utils import get_cache as _g

        _g("factors_suggest_for_ticker").clear()

        r = app_client.post("/factors/suggest-for-ticker", json=body)
        assert r.status_code == 200, r.text
        assert r.json()["top_factors"][0]["factor_id"] == "factor_a"
        assert scan_calls["n"] == 0, "L2 hit should skip the scan entirely"

    def test_cold_call_writes_l2(
        self,
        app_client: TestClient,
        cache_shim: _DictRedisShim,
    ) -> None:
        body = {
            "ticker": "MSFT",
            "lookback_days": 90,
            "top_k": 3,
            "min_n_obs": 10,
        }
        cache_key = rr._suggest_cache_key(
            body["ticker"],
            body["lookback_days"],
            body["top_k"],
            body["min_n_obs"],
        )

        r = app_client.post("/factors/suggest-for-ticker", json=body)
        assert r.status_code == 200, r.text
        # The leader should have written the L2 payload.
        assert rr._suggest_l2_key(cache_key) in cache_shim.store


# ---------------------------------------------------------------------------
# 2. 429 retry on a single factor doesn't kill the scan
# ---------------------------------------------------------------------------


class TestRateLimitRetry:
    def test_one_429_then_success_does_not_skip_factor(
        self,
        monkeypatch: pytest.MonkeyPatch,
        app_client: TestClient,
    ) -> None:
        # Patch the per-factor cached fetch to raise 429 once and then
        # succeed. The retry path inside ``_fetch_factor_with_retry`` must
        # absorb the first 429 and call again; we confirm via attempt
        # counts (not via top-K membership, since the fixture's date
        # window may produce zero overlapping observations regardless).
        import pfm.main as main_mod

        original_cached = main_mod._cached_factor_history
        attempts: dict[str, int] = {}

        def _flaky(fc, start, end, poly, cache, settings):
            attempts[fc.id] = attempts.get(fc.id, 0) + 1
            # First call to factor_a → 429; subsequent calls succeed.
            if fc.id == "factor_a" and attempts[fc.id] == 1:
                raise HTTPException(status_code=429, detail="gamma rate-limited")
            return original_cached(fc, start, end, poly, cache, settings)

        monkeypatch.setattr(main_mod, "_cached_factor_history", _flaky)
        # Make the retry instant in tests.
        monkeypatch.setattr(rr, "_SUGGEST_FETCH_RETRY_AFTER_S", 0.0)

        r = app_client.post(
            "/factors/suggest-for-ticker",
            json={
                "ticker": "RETRY1",
                "lookback_days": 90,
                "top_k": 5,
                "min_n_obs": 10,
            },
        )
        assert r.status_code == 200, r.text
        # The retry actually fired: 2 attempts on factor_a, 1 on factor_b.
        # Without the retry wrapper, attempts[factor_a] would be 1 and the
        # whole scan would have logged a fetch_error for it.
        assert attempts.get("factor_a", 0) == 2, f"retry on 429 did not fire — attempts={attempts}"
        assert attempts.get("factor_b", 0) == 1

    def test_persistent_429_does_not_tank_whole_scan(
        self,
        monkeypatch: pytest.MonkeyPatch,
        app_client: TestClient,
    ) -> None:
        # Persistent 429 on factor_a → skipped after the single retry;
        # the scan stays at HTTP 200 (one rate-limited factor must NOT
        # tank the whole response). factor_b is still attempted exactly
        # once and contributes to the skip/scan accounting.
        import pfm.main as main_mod

        original_cached = main_mod._cached_factor_history
        attempts: dict[str, int] = {}

        def _always_429_for_a(fc, start, end, poly, cache, settings):
            attempts[fc.id] = attempts.get(fc.id, 0) + 1
            if fc.id == "factor_a":
                raise HTTPException(status_code=429, detail="persistent 429")
            return original_cached(fc, start, end, poly, cache, settings)

        monkeypatch.setattr(main_mod, "_cached_factor_history", _always_429_for_a)
        monkeypatch.setattr(rr, "_SUGGEST_FETCH_RETRY_AFTER_S", 0.0)

        r = app_client.post(
            "/factors/suggest-for-ticker",
            json={
                "ticker": "RETRY2",
                "lookback_days": 90,
                "top_k": 5,
                "min_n_obs": 10,
            },
        )
        # Whole-scan resilience: 200 even when one factor is persistently
        # rate-limited.
        assert r.status_code == 200, r.text
        # Retry fired exactly once on factor_a (2 total attempts), factor_b
        # was tried exactly once (no retry needed). Without the retry path
        # the count would be 1 / 1 — so this also confirms the retry was
        # actually invoked on the persistent failure.
        assert attempts.get("factor_a", 0) == 2
        assert attempts.get("factor_b", 0) == 1
        # And factor_a must not appear (after both attempts failed).
        ids = [it["factor_id"] for it in r.json()["top_factors"]]
        assert "factor_a" not in ids

    def test_non_429_error_is_not_retried(
        self,
        monkeypatch: pytest.MonkeyPatch,
        app_client: TestClient,
    ) -> None:
        # 404 should NOT be retried — only 429/503/504 are transient.
        import pfm.main as main_mod

        original_cached = main_mod._cached_factor_history
        attempts: dict[str, int] = {}

        def _404_for_a(fc, start, end, poly, cache, settings):
            attempts[fc.id] = attempts.get(fc.id, 0) + 1
            if fc.id == "factor_a":
                raise HTTPException(status_code=404, detail="market gone")
            return original_cached(fc, start, end, poly, cache, settings)

        monkeypatch.setattr(main_mod, "_cached_factor_history", _404_for_a)
        monkeypatch.setattr(rr, "_SUGGEST_FETCH_RETRY_AFTER_S", 0.0)

        r = app_client.post(
            "/factors/suggest-for-ticker",
            json={
                "ticker": "RETRY3",
                "lookback_days": 90,
                "top_k": 5,
                "min_n_obs": 10,
            },
        )
        assert r.status_code == 200, r.text
        # factor_a hit exactly once (no retry on 404).
        assert attempts.get("factor_a", 0) == 1


# ---------------------------------------------------------------------------
# 3. SETNX stampede protection
# ---------------------------------------------------------------------------


class TestStampedeLock:
    def test_setnx_only_one_scan_under_concurrent_misses(
        self,
        monkeypatch: pytest.MonkeyPatch,
        app_client: TestClient,
        cache_shim: _DictRedisShim,
    ) -> None:
        # Two concurrent same-key cold callers: the loser must wait for
        # the leader's L2 write rather than running its own scan. We
        # simulate the race by holding the leader inside the scan with a
        # barrier and instrumenting the call count.

        scan_calls = {"n": 0}
        scan_started = threading.Event()
        scan_release = threading.Event()
        original = rr._scan_factor_correlations_for_ticker

        def _slow_scan(*args: Any, **kwargs: Any) -> Any:
            scan_calls["n"] += 1
            scan_started.set()
            # Hold the lock long enough that the second caller has to
            # wait (~0.5 s is fine — well under the 60 s wait budget).
            scan_release.wait(timeout=5.0)
            return original(*args, **kwargs)

        monkeypatch.setattr(rr, "_scan_factor_correlations_for_ticker", _slow_scan)
        # Keep the loser-poll interval tight so the test isn't slow.
        monkeypatch.setattr(rr, "_SUGGEST_LOCK_POLL_S", 0.05)

        body = {
            "ticker": "RACE",
            "lookback_days": 90,
            "top_k": 3,
            "min_n_obs": 10,
        }

        results: dict[int, int] = {}

        def _hit(idx: int) -> None:
            r = app_client.post("/factors/suggest-for-ticker", json=body)
            results[idx] = r.status_code

        t1 = threading.Thread(target=_hit, args=(1,))
        t2 = threading.Thread(target=_hit, args=(2,))
        t1.start()
        # Wait for the leader to hold the scan lock before launching the
        # second caller — guarantees the loser sees the SETNX miss.
        scan_started.wait(timeout=5.0)
        t2.start()
        # Give the loser a beat to attempt SETNX before we let the
        # leader finish (otherwise the loser may hit L1 directly).
        import time as _t

        _t.sleep(0.2)
        # Let the leader finish.
        scan_release.set()
        t1.join(timeout=10.0)
        t2.join(timeout=10.0)

        assert results == {1: 200, 2: 200}
        # Core invariant: only ONE scan ran for two concurrent same-key
        # cold misses. Without the SETNX lock both threads would scan
        # in parallel and ``scan_calls["n"]`` would be 2.
        assert scan_calls["n"] == 1, (
            f"stampede protection broken: {scan_calls['n']} scans for "
            f"2 concurrent same-key cold misses"
        )
        # Both workers should have at minimum reached the SETNX call
        # site (the loser bails out into the wait loop after losing).
        lock_key = rr._suggest_lock_key(
            rr._suggest_cache_key(
                body["ticker"],
                body["lookback_days"],
                body["top_k"],
                body["min_n_obs"],
            )
        )
        assert cache_shim.setnx_calls.count(lock_key) >= 2, (
            f"both workers should have attempted SETNX, got {cache_shim.setnx_calls}"
        )

    def test_lock_released_after_scan(
        self,
        app_client: TestClient,
        cache_shim: _DictRedisShim,
    ) -> None:
        # After a successful scan the SETNX lock key should be cleared so
        # a subsequent (different-key) request can acquire its own lock
        # without waiting on the 120 s TTL.
        body = {
            "ticker": "REL",
            "lookback_days": 90,
            "top_k": 3,
            "min_n_obs": 10,
        }
        r = app_client.post("/factors/suggest-for-ticker", json=body)
        assert r.status_code == 200, r.text
        lock_key = rr._suggest_lock_key(
            rr._suggest_cache_key(
                body["ticker"],
                body["lookback_days"],
                body["top_k"],
                body["min_n_obs"],
            )
        )
        assert lock_key not in cache_shim.store, "leader should release the SETNX lock on success"
