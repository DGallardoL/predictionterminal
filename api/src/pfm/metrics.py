"""Operational latency tracker for the per-endpoint audit endpoint.

This module is intentionally tiny and dependency-free (no Prometheus, no
Redis) so it can be imported very early in :mod:`pfm.main` without adding to
startup latency. It backs the ``GET /metrics/audit`` endpoint registered by
:mod:`pfm.metrics_router`, which exposes per-endpoint latency percentiles
(p50/p95/p99) and 5xx error rate observed since process start.

Design notes
------------

* The buffer is a single :class:`collections.deque` with ``maxlen=10000``;
  push/pop are O(1) and the deque itself is thread-safe for ``append``,
  which is the only mutation we perform here. A :class:`threading.Lock`
  still guards iteration so :meth:`LatencyTracker.snapshot` sees a
  consistent view even under contention.
* Paths are passed through verbatim — the middleware in :mod:`pfm.main`
  is responsible for collapsing templated paths (``/foo/{slug}`` etc.)
  before calling :meth:`record`.
* Percentiles are computed with :func:`numpy.percentile` when numpy is
  importable; otherwise we fall back to a manual sort+interpolate. Both
  paths satisfy the ``p50 <= p95 <= p99`` invariant by construction.
"""

from __future__ import annotations

import threading
from collections import defaultdict, deque
from typing import Any

try:  # numpy is a hard dep elsewhere in the project, but keep this defensive.
    import numpy as _np

    _HAS_NUMPY = True
except ImportError:  # pragma: no cover - numpy is available in this project.
    _np = None  # type: ignore[assignment]
    _HAS_NUMPY = False


_DEFAULT_MAXLEN = 10_000


class LatencyTracker:
    """Thread-safe ring buffer of recent request observations.

    Each observation is a ``(path, ms, status_code)`` tuple. ``record`` is
    called from the FastAPI middleware on every non-``/metrics/*`` request.
    """

    def __init__(self, maxlen: int = _DEFAULT_MAXLEN) -> None:
        self._buf: deque[tuple[str, float, int]] = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def record(self, path: str, ms: float, code: int) -> None:
        """Append one observation. Cheap; safe to call on the hot path."""
        # ``deque.append`` is atomic under CPython's GIL, but we still take
        # the lock to keep an invariant: ``snapshot`` sees either the new
        # element or not, never a half-mutated container.
        with self._lock:
            self._buf.append((str(path), float(ms), int(code)))

    def clear(self) -> None:
        """Drop all observations. Useful for tests."""
        with self._lock:
            self._buf.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._buf)

    # ------------------------------------------------------------------
    # Read-side
    # ------------------------------------------------------------------

    def snapshot(self) -> dict[str, dict[str, Any]]:
        """Aggregate the buffer into a ``{path: stats}`` dict.

        Each ``stats`` dict contains ``count``, ``p50``, ``p95``, ``p99``
        (all in milliseconds, rounded to 3 dp), and ``err_rate`` (the
        fraction of observations with ``status_code >= 500``, rounded to
        4 dp).
        """
        with self._lock:
            # Copy so we can release the lock before doing the heavy lifting.
            obs = list(self._buf)

        if not obs:
            return {}

        grouped: dict[str, list[tuple[float, int]]] = defaultdict(list)
        for path, ms, code in obs:
            grouped[path].append((ms, code))

        out: dict[str, dict[str, Any]] = {}
        for path, rows in grouped.items():
            latencies = [r[0] for r in rows]
            err_count = sum(1 for r in rows if r[1] >= 500)
            p50, p95, p99 = _percentiles(latencies, (50.0, 95.0, 99.0))
            out[path] = {
                "count": len(rows),
                "p50_ms": round(p50, 3),
                "p95_ms": round(p95, 3),
                "p99_ms": round(p99, 3),
                "err_rate": round(err_count / len(rows), 4),
                "errors_5xx": err_count,
            }
        return out

    def total_requests(self) -> int:
        """Total observations in the buffer (capped at ``maxlen``)."""
        return len(self)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _percentiles(values: list[float], qs: tuple[float, ...]) -> tuple[float, ...]:
    """Return percentiles in ``qs`` (0-100) for ``values``.

    Uses :func:`numpy.percentile` with linear interpolation when available,
    otherwise a manual nearest-rank-with-interpolation fallback. Returns
    zeros for an empty list so callers don't need to guard.
    """
    if not values:
        return tuple(0.0 for _ in qs)
    if _HAS_NUMPY:
        arr = _np.asarray(values, dtype=float)
        out = _np.percentile(arr, list(qs))
        return tuple(float(x) for x in out)
    # Manual fallback: sort once, linear-interp per quantile.
    s = sorted(values)
    n = len(s)
    res: list[float] = []
    for q in qs:
        if n == 1:
            res.append(s[0])
            continue
        # numpy "linear" interpolation: position = q/100 * (n-1).
        pos = (q / 100.0) * (n - 1)
        lo = int(pos)
        hi = min(lo + 1, n - 1)
        frac = pos - lo
        res.append(s[lo] * (1.0 - frac) + s[hi] * frac)
    return tuple(res)


# ----------------------------------------------------------------------
# Process-wide singleton
# ----------------------------------------------------------------------

#: Shared tracker used by the FastAPI middleware and the audit router.
TRACKER = LatencyTracker()


def get_tracker() -> LatencyTracker:
    """Return the module-level :class:`LatencyTracker` singleton."""
    return TRACKER
