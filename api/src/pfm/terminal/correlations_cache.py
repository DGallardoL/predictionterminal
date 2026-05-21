"""Memoization layer for pairwise correlation matrices.

With the 1228-factor catalogue (post-Wave-9), a single all-pairs correlation
call computes O(n²) ≈ 1.5M correlations. Users typically request the same
basket (cluster theme, watchlist, top-k by liquidity) repeatedly across
sessions, so an LRU cache fronting the expensive ``compute_fn`` makes the
warm path effectively free.

Design notes
------------
- The cache key is a stable hex digest of ``frozenset(slugs)`` plus the
  ``window_days`` integer.  Two callers passing the *same* slugs in different
  order produce identical keys (frozenset is order-insensitive).
- ``compute_fn`` is kept abstract: this module knows nothing about how
  factor histories are loaded.  Callers (typically ``terminal/correlations``
  or ``terminal/factor_clusters``) wire in a closure that does the heavy
  lifting and returns an ``np.ndarray`` aligned with the *sorted* slug list.
- Thread-safety: a single module-level ``threading.Lock`` guards both the
  default cache and the single-flight semantics — a concurrent second caller
  on the same key blocks on the lock, then finds the result already in the
  cache and returns it without re-computing.
- ``compute_fn`` exceptions are NOT cached.  A failure leaves the cache
  untouched, so the next caller retries (the alternative — negative caching —
  would mask transient upstream errors and is explicitly out of scope).
- The returned ``CorrMatrix`` is shared across cache hits: callers must
  treat it as immutable.  The ``np.ndarray`` is set read-only via
  ``arr.flags.writeable = False`` as a belt-and-braces guard.

The cache mirrors the ``cachetools.LRUCache`` API surface we actually use
(``__contains__``, ``__getitem__``, ``__setitem__``, ``__len__``,
``popitem``) so callers can pass either our default or a ``cachetools``
instance interchangeably.
"""

from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np

__all__ = [
    "DEFAULT_CACHE_SIZE",
    "CorrMatrix",
    "default_cache",
    "get_or_compute_corr",
    "pair_corr_cache_key",
]


DEFAULT_CACHE_SIZE: int = 256


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CorrMatrix:
    """Square symmetric correlation matrix bundled with provenance metadata.

    Attributes
    ----------
    slugs:
        Ordered list of factor slugs.  Always sorted ascending so that two
        callers passing the same ``frozenset`` produce identical row/column
        indexing.
    matrix:
        ``(n, n)`` ``np.ndarray`` of Pearson (or whatever the ``compute_fn``
        returns) correlations.  The array is set read-only at construction
        time; mutating it via ``matrix[i, j] = ...`` raises.
    window:
        Lookback window in days used by the underlying ``compute_fn``.
    computed_at:
        UTC timestamp captured at the time the matrix was filled in.
    """

    slugs: list[str]
    matrix: np.ndarray
    window: int
    computed_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self) -> None:
        # Lock the array down so accidental mutation in one caller does not
        # poison every other holder of the cached instance.
        if isinstance(self.matrix, np.ndarray) and self.matrix.flags.writeable:
            self.matrix.flags.writeable = False


# ---------------------------------------------------------------------------
# Minimal LRU implementation (drop-in for cachetools.LRUCache.maxsize+items)
# ---------------------------------------------------------------------------


class _LRUCache:
    """Tiny LRU cache backed by ``collections.OrderedDict``.

    Implements just enough of the ``cachetools.LRUCache`` API for this
    module's needs.  Operations are O(1).
    """

    def __init__(self, maxsize: int = DEFAULT_CACHE_SIZE) -> None:
        if maxsize <= 0:
            raise ValueError(f"maxsize must be positive, got {maxsize!r}")
        self.maxsize: int = maxsize
        self._data: OrderedDict[str, Any] = OrderedDict()

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def __getitem__(self, key: str) -> Any:
        value = self._data[key]
        # mark as recently used
        self._data.move_to_end(key)
        return value

    def __setitem__(self, key: str, value: Any) -> None:
        if key in self._data:
            self._data.move_to_end(key)
            self._data[key] = value
            return
        self._data[key] = value
        if len(self._data) > self.maxsize:
            self._data.popitem(last=False)  # drop oldest

    def __len__(self) -> int:
        return len(self._data)

    def popitem(self) -> tuple[str, Any]:
        return self._data.popitem(last=False)

    def clear(self) -> None:
        self._data.clear()

    def keys(self) -> list[str]:  # convenience for tests
        return list(self._data.keys())


_DEFAULT_CACHE_LOCK = threading.Lock()
_DEFAULT_CACHE: _LRUCache = _LRUCache(maxsize=DEFAULT_CACHE_SIZE)
# Per-key single-flight locks live next to the cache so two callers racing
# on the same key serialise without blocking unrelated keys.
_KEY_LOCKS: dict[str, threading.Lock] = {}


def default_cache() -> _LRUCache:
    """Return the module-level default LRU cache (256 entries)."""
    return _DEFAULT_CACHE


# ---------------------------------------------------------------------------
# Key derivation
# ---------------------------------------------------------------------------


def pair_corr_cache_key(slugs: frozenset[str], window_days: int) -> str:
    """Derive a stable cache key for ``(slugs, window_days)``.

    The slug set is hashed via SHA-1 of the sorted, NUL-joined slug list
    (Python's built-in ``hash()`` is process-local and unsuitable as a cache
    key across reloads or for debugging).  ``window_days`` is appended in
    plain decimal so cache misses on different windows are obvious from the
    key alone.
    """
    if not isinstance(slugs, frozenset):
        raise TypeError(f"slugs must be a frozenset, got {type(slugs).__name__}")
    if not isinstance(window_days, int) or isinstance(window_days, bool):
        raise TypeError(
            f"window_days must be int, got {type(window_days).__name__}",
        )
    if window_days <= 0:
        raise ValueError(f"window_days must be positive, got {window_days}")

    payload = "\x00".join(sorted(slugs)).encode("utf-8")
    digest = hashlib.sha1(payload, usedforsecurity=False).hexdigest()
    return f"{digest}:{window_days}"


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def _acquire_key_lock(key: str) -> threading.Lock:
    """Return (creating if needed) the single-flight lock for ``key``."""
    with _DEFAULT_CACHE_LOCK:
        lock = _KEY_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _KEY_LOCKS[key] = lock
        return lock


def _validate_matrix(arr: np.ndarray, n: int) -> None:
    if not isinstance(arr, np.ndarray):
        raise TypeError(
            f"compute_fn must return np.ndarray, got {type(arr).__name__}",
        )
    if arr.shape != (n, n):
        raise ValueError(
            f"compute_fn returned shape {arr.shape!r}, expected ({n}, {n})",
        )


def get_or_compute_corr(
    slugs: list[str],
    compute_fn: Callable[[list[str]], np.ndarray],
    *,
    window_days: int = 30,
    cache: Any | None = None,
) -> CorrMatrix:
    """Return the memoized ``CorrMatrix`` for ``slugs`` (computing if absent).

    Parameters
    ----------
    slugs:
        Iterable of factor slugs.  Order is *not* significant — the key is
        derived from ``frozenset(slugs)`` and the returned matrix is always
        indexed by ``sorted(set(slugs))`` to give callers a deterministic
        layout.  Duplicates are collapsed silently.
    compute_fn:
        Callable that receives the ordered slug list and returns an
        ``(n, n)`` ``np.ndarray``.  Called at most once per cache key when
        invoked concurrently (single-flight via per-key lock).
    window_days:
        Lookback window passed to ``compute_fn``.  Distinct windows produce
        distinct cache keys.
    cache:
        Optional LRU-like cache instance.  Must support
        ``__contains__/__getitem__/__setitem__``.  Defaults to the
        module-level 256-entry cache.

    Returns
    -------
    CorrMatrix
        Frozen, read-only correlation matrix payload.  Cache hits return the
        *same* instance across calls.
    """
    if not isinstance(window_days, int) or isinstance(window_days, bool):
        raise TypeError(
            f"window_days must be int, got {type(window_days).__name__}",
        )
    if window_days <= 0:
        raise ValueError(f"window_days must be positive, got {window_days}")
    if not callable(compute_fn):
        raise TypeError("compute_fn must be callable")

    ordered = sorted({str(s) for s in slugs})
    if not ordered:
        raise ValueError("slugs must contain at least one entry")

    key = pair_corr_cache_key(frozenset(ordered), window_days)
    store = cache if cache is not None else _DEFAULT_CACHE

    # Fast-path: cache hit under the default lock (cheap snapshot).
    with _DEFAULT_CACHE_LOCK:
        if key in store:
            return store[key]

    # Single-flight: only one thread per key reaches compute_fn.
    key_lock = _acquire_key_lock(key)
    with key_lock:
        # Re-check inside the per-key lock — another thread may have filled
        # the cache while we waited.
        with _DEFAULT_CACHE_LOCK:
            if key in store:
                return store[key]

        # Compute outside the global lock so the heavy correlation work does
        # not block unrelated keys.  Failures propagate; nothing is cached.
        matrix = compute_fn(ordered)
        _validate_matrix(matrix, len(ordered))

        result = CorrMatrix(
            slugs=list(ordered),
            matrix=matrix,
            window=window_days,
        )

        with _DEFAULT_CACHE_LOCK:
            store[key] = result
            # If the caller is using the *default* cache, also reap any
            # orphan per-key locks for keys that have since been evicted so
            # the dict does not grow unboundedly.  Heuristic: only sweep
            # when the lock table is much larger than the cache itself.
            if store is _DEFAULT_CACHE and len(_KEY_LOCKS) > 4 * _DEFAULT_CACHE.maxsize:
                live = set(_DEFAULT_CACHE.keys())
                for stale in [k for k in _KEY_LOCKS if k not in live and k != key]:
                    _KEY_LOCKS.pop(stale, None)

        return result
