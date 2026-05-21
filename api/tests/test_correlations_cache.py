"""Tests for :mod:`pfm.terminal.correlations_cache`.

Run in isolation via:

    cd api && PYTHONPATH=src .venv/bin/python -m pytest \
        tests/test_correlations_cache.py -q --noconftest
"""

from __future__ import annotations

import threading
import time
from datetime import UTC, datetime

import numpy as np
import pytest

from pfm.terminal.correlations_cache import (
    DEFAULT_CACHE_SIZE,
    CorrMatrix,
    _LRUCache,
    default_cache,
    get_or_compute_corr,
    pair_corr_cache_key,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_corr_fn(call_counter: list[int]) -> callable:
    """Return a compute_fn that records call count and produces a valid
    symmetric correlation matrix derived from a deterministic seed."""

    def _fn(slugs: list[str]) -> np.ndarray:
        call_counter.append(1)
        n = len(slugs)
        # Build a symmetric matrix with ones on the diagonal, deterministic
        # off-diagonals derived from sorted-slug hashing so two calls with
        # the same slug set produce identical matrices.
        rng = np.random.default_rng(seed=abs(hash("|".join(slugs))) % (2**32))
        a = rng.uniform(-0.9, 0.9, size=(n, n))
        m = (a + a.T) / 2.0
        np.fill_diagonal(m, 1.0)
        return m

    return _fn


@pytest.fixture(autouse=True)
def _reset_default_cache():
    """Clear the module-level default cache before/after each test."""
    cache = default_cache()
    cache.clear()
    yield
    cache.clear()


# ---------------------------------------------------------------------------
# Key derivation
# ---------------------------------------------------------------------------


def test_cache_key_order_independent() -> None:
    """frozenset key derivation must not depend on insertion order."""
    a = pair_corr_cache_key(frozenset({"a", "b", "c"}), 30)
    b = pair_corr_cache_key(frozenset({"c", "b", "a"}), 30)
    assert a == b


def test_cache_key_window_sensitive() -> None:
    k30 = pair_corr_cache_key(frozenset({"a", "b"}), 30)
    k60 = pair_corr_cache_key(frozenset({"a", "b"}), 60)
    assert k30 != k60


def test_cache_key_rejects_non_frozenset() -> None:
    with pytest.raises(TypeError):
        pair_corr_cache_key({"a", "b"}, 30)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        pair_corr_cache_key(["a", "b"], 30)  # type: ignore[arg-type]


def test_cache_key_rejects_bad_window() -> None:
    with pytest.raises(ValueError):
        pair_corr_cache_key(frozenset({"a"}), 0)
    with pytest.raises(ValueError):
        pair_corr_cache_key(frozenset({"a"}), -1)
    with pytest.raises(TypeError):
        pair_corr_cache_key(frozenset({"a"}), 30.0)  # type: ignore[arg-type]
    # bool is a subclass of int — reject explicitly
    with pytest.raises(TypeError):
        pair_corr_cache_key(frozenset({"a"}), True)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# get_or_compute_corr — basic memoization
# ---------------------------------------------------------------------------


def test_compute_called_once_for_repeated_call() -> None:
    counter: list[int] = []
    fn = _make_corr_fn(counter)
    slugs = ["btc", "eth", "spy"]
    r1 = get_or_compute_corr(slugs, fn, window_days=30)
    r2 = get_or_compute_corr(slugs, fn, window_days=30)
    assert len(counter) == 1
    # Cache hit returns the *same* instance.
    assert r1 is r2


def test_different_slug_order_same_cache_key() -> None:
    counter: list[int] = []
    fn = _make_corr_fn(counter)
    r1 = get_or_compute_corr(["a", "b", "c"], fn, window_days=30)
    r2 = get_or_compute_corr(["c", "b", "a"], fn, window_days=30)
    r3 = get_or_compute_corr(["b", "a", "c"], fn, window_days=30)
    assert len(counter) == 1
    assert r1 is r2 is r3
    # Slug ordering inside the result is normalised (sorted ascending).
    assert r1.slugs == ["a", "b", "c"]


def test_different_window_different_cache_key() -> None:
    counter: list[int] = []
    fn = _make_corr_fn(counter)
    r30 = get_or_compute_corr(["a", "b"], fn, window_days=30)
    r60 = get_or_compute_corr(["a", "b"], fn, window_days=60)
    assert len(counter) == 2
    assert r30 is not r60
    assert r30.window == 30
    assert r60.window == 60


def test_duplicate_slugs_collapsed() -> None:
    counter: list[int] = []
    fn = _make_corr_fn(counter)
    r1 = get_or_compute_corr(["a", "b", "a", "b"], fn, window_days=30)
    r2 = get_or_compute_corr(["a", "b"], fn, window_days=30)
    assert len(counter) == 1
    assert r1 is r2
    assert r1.slugs == ["a", "b"]


# ---------------------------------------------------------------------------
# LRU eviction
# ---------------------------------------------------------------------------


def test_cache_eviction_at_max_size() -> None:
    cache = _LRUCache(maxsize=DEFAULT_CACHE_SIZE)
    counter: list[int] = []
    fn = _make_corr_fn(counter)

    # Fill exactly to capacity.
    for i in range(DEFAULT_CACHE_SIZE):
        get_or_compute_corr([f"slug_{i}_a", f"slug_{i}_b"], fn, window_days=30, cache=cache)
    assert len(cache) == DEFAULT_CACHE_SIZE
    assert len(counter) == DEFAULT_CACHE_SIZE

    # Insert one more — oldest is evicted.
    get_or_compute_corr(["evictor_a", "evictor_b"], fn, window_days=30, cache=cache)
    assert len(cache) == DEFAULT_CACHE_SIZE
    assert len(counter) == DEFAULT_CACHE_SIZE + 1

    # Hitting the evicted entry triggers a recompute.
    get_or_compute_corr(["slug_0_a", "slug_0_b"], fn, window_days=30, cache=cache)
    assert len(counter) == DEFAULT_CACHE_SIZE + 2


def test_lru_recency_moves_on_get() -> None:
    """Re-reading an entry should refresh its position so it survives eviction."""
    cache = _LRUCache(maxsize=3)
    counter: list[int] = []
    fn = _make_corr_fn(counter)
    get_or_compute_corr(["x"], fn, window_days=30, cache=cache)
    get_or_compute_corr(["y"], fn, window_days=30, cache=cache)
    get_or_compute_corr(["z"], fn, window_days=30, cache=cache)
    # Touch the oldest so it becomes most-recent.
    get_or_compute_corr(["x"], fn, window_days=30, cache=cache)
    # Inserting a 4th should evict y, not x.
    get_or_compute_corr(["w"], fn, window_days=30, cache=cache)
    # x still cached (no recompute).
    before = len(counter)
    get_or_compute_corr(["x"], fn, window_days=30, cache=cache)
    assert len(counter) == before


# ---------------------------------------------------------------------------
# Failure isolation
# ---------------------------------------------------------------------------


def test_compute_fn_exception_does_not_poison_cache() -> None:
    attempts = {"n": 0}

    def flaky(slugs: list[str]) -> np.ndarray:
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise RuntimeError("upstream blew up")
        n = len(slugs)
        return np.eye(n)

    with pytest.raises(RuntimeError):
        get_or_compute_corr(["a", "b"], flaky, window_days=30)
    # Cache must be empty so the next call retries cleanly.
    assert len(default_cache()) == 0
    result = get_or_compute_corr(["a", "b"], flaky, window_days=30)
    assert attempts["n"] == 2
    assert result.matrix.shape == (2, 2)


def test_compute_fn_bad_shape_rejected() -> None:
    def bad(slugs: list[str]) -> np.ndarray:
        return np.zeros((1, 2))

    with pytest.raises(ValueError):
        get_or_compute_corr(["a", "b"], bad, window_days=30)
    assert len(default_cache()) == 0


def test_compute_fn_bad_type_rejected() -> None:
    def bad(slugs: list[str]):
        return [[1.0, 0.0], [0.0, 1.0]]  # type: ignore[return-value]

    with pytest.raises(TypeError):
        get_or_compute_corr(["a", "b"], bad, window_days=30)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Concurrency / single-flight
# ---------------------------------------------------------------------------


def test_concurrent_calls_single_flight() -> None:
    """Many threads racing on the same (slugs, window) compute exactly once."""
    barrier = threading.Barrier(8)
    counter: list[int] = []
    counter_lock = threading.Lock()

    def slow(slugs: list[str]) -> np.ndarray:
        with counter_lock:
            counter.append(1)
        # Hold the compute lock long enough that every thread must contend.
        time.sleep(0.05)
        return np.eye(len(slugs))

    results: list[CorrMatrix] = []
    results_lock = threading.Lock()

    def worker() -> None:
        barrier.wait()
        r = get_or_compute_corr(["alpha", "beta", "gamma"], slow, window_days=30)
        with results_lock:
            results.append(r)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(counter) == 1, f"compute_fn ran {len(counter)} times, expected 1"
    assert len(results) == 8
    # Every thread received the same cached instance.
    first = results[0]
    assert all(r is first for r in results)


def test_concurrent_different_keys_do_not_serialize() -> None:
    """Two threads on distinct keys both run compute_fn (no false contention)."""
    counter: list[int] = []
    counter_lock = threading.Lock()
    barrier = threading.Barrier(2)

    def fn(slugs: list[str]) -> np.ndarray:
        with counter_lock:
            counter.append(1)
        barrier.wait(timeout=2.0)  # both threads must be inside compute_fn
        return np.eye(len(slugs))

    out: dict[str, CorrMatrix] = {}

    def worker(slug: str) -> None:
        out[slug] = get_or_compute_corr([slug], fn, window_days=30)

    t1 = threading.Thread(target=worker, args=("k1",))
    t2 = threading.Thread(target=worker, args=("k2",))
    t1.start()
    t2.start()
    t1.join(timeout=3.0)
    t2.join(timeout=3.0)
    assert not t1.is_alive() and not t2.is_alive()
    assert len(counter) == 2
    assert out["k1"] is not out["k2"]


# ---------------------------------------------------------------------------
# Shape / symmetry invariants
# ---------------------------------------------------------------------------


def test_matrix_shape_matches_slug_count() -> None:
    counter: list[int] = []
    fn = _make_corr_fn(counter)
    slugs = [f"f{i}" for i in range(7)]
    r = get_or_compute_corr(slugs, fn, window_days=30)
    assert r.matrix.shape == (7, 7)
    assert len(r.slugs) == 7


def test_matrix_symmetry_preserved() -> None:
    counter: list[int] = []
    fn = _make_corr_fn(counter)
    r = get_or_compute_corr(["a", "b", "c", "d"], fn, window_days=30)
    assert np.allclose(r.matrix, r.matrix.T)
    # Diagonal is exactly 1.0.
    assert np.allclose(np.diag(r.matrix), 1.0)


def test_returned_matrix_is_read_only() -> None:
    counter: list[int] = []
    fn = _make_corr_fn(counter)
    r = get_or_compute_corr(["a", "b"], fn, window_days=30)
    assert not r.matrix.flags.writeable
    with pytest.raises(ValueError):
        r.matrix[0, 0] = 99.0


def test_corrmatrix_metadata() -> None:
    counter: list[int] = []
    fn = _make_corr_fn(counter)
    before = datetime.now(UTC)
    r = get_or_compute_corr(["a", "b"], fn, window_days=42)
    after = datetime.now(UTC)
    assert isinstance(r, CorrMatrix)
    assert r.window == 42
    assert r.slugs == ["a", "b"]
    assert before <= r.computed_at <= after


# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------


def test_empty_slugs_rejected() -> None:
    with pytest.raises(ValueError):
        get_or_compute_corr([], lambda _s: np.zeros((0, 0)), window_days=30)


def test_bad_window_rejected() -> None:
    fn = _make_corr_fn([])
    with pytest.raises(ValueError):
        get_or_compute_corr(["a"], fn, window_days=0)
    with pytest.raises(ValueError):
        get_or_compute_corr(["a"], fn, window_days=-7)
    with pytest.raises(TypeError):
        get_or_compute_corr(["a"], fn, window_days=30.0)  # type: ignore[arg-type]


def test_non_callable_compute_fn_rejected() -> None:
    with pytest.raises(TypeError):
        get_or_compute_corr(["a"], "not-a-callable", window_days=30)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Custom cache instance honoured
# ---------------------------------------------------------------------------


def test_custom_cache_isolated_from_default() -> None:
    """Passing a custom cache must not poison the module-level default."""
    counter: list[int] = []
    fn = _make_corr_fn(counter)
    custom = _LRUCache(maxsize=4)
    get_or_compute_corr(["x", "y"], fn, window_days=30, cache=custom)
    assert len(custom) == 1
    assert len(default_cache()) == 0
