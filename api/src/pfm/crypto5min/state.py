"""Rolling state for the 5m / 15m crypto predictor.

The predictor needs ``spot_0`` — the Binance spot price *at the start of the
current Polymarket window* — to compute ``log(spot_t / spot_0)``. Polymarket
btc-updown markets resolve on the Chainlink reference at the window boundary
and the boundary is deterministic (``(now // period) * period`` in unix
seconds), so we can derive ``spot_0`` ourselves: just keep a rolling buffer
of (timestamp, mid) samples and pick the sample whose timestamp is closest
to (but not after) the window's start boundary.

This module is a small singleton helper that:

* Ingests spot samples (``record_spot(symbol, ts_unix, mid)``).
* Resolves the anchor for a given (symbol, period) at request time.
* Bounds memory — keeps at most the last ``MAX_SAMPLES`` per symbol.

It's deliberately *separate* from ``crypto_events_engine`` because the WS
engine isn't always running (tests, prod-without-PFM_CRYPTO_WS_ENABLED) and
the 5m predictor still has to work in those cases via REST polling.
"""

from __future__ import annotations

import bisect
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from threading import Lock

#: How many (ts, mid) tuples to retain per symbol. At 1 sample/sec the
#: buffer covers ~33 minutes which is enough for the 5m and 15m windows
#: plus generous slack for the discovery sweep at startup.
MAX_SAMPLES: int = 2000


@dataclass(frozen=True, slots=True)
class WindowAnchor:
    """Resolved boundary-spot pair for a single (symbol, period)."""

    symbol: str
    period_seconds: int
    start_unix: int
    end_unix: int
    spot_at_start: float
    spot_now: float
    seconds_remaining: float


class CryptoFiveMinState:
    """Thread-safe rolling buffer + anchor resolver for short-window crypto."""

    def __init__(self, max_samples: int = MAX_SAMPLES) -> None:
        self._max = max_samples
        self._samples: dict[str, deque[tuple[float, float]]] = defaultdict(
            lambda: deque(maxlen=self._max)
        )
        self._lock = Lock()

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def record_spot(self, symbol: str, ts_unix: float, mid: float) -> None:
        """Append a spot sample. No-ops on non-positive prices."""
        if mid <= 0:
            return
        sym = symbol.upper()
        with self._lock:
            buf = self._samples[sym]
            # Reject out-of-order samples — they'd break the bisect in
            # ``anchor()`` which assumes ascending timestamps.
            if buf and ts_unix <= buf[-1][0]:
                return
            buf.append((float(ts_unix), float(mid)))

    def latest(self, symbol: str) -> tuple[float, float] | None:
        sym = symbol.upper()
        with self._lock:
            buf = self._samples.get(sym)
            if not buf:
                return None
            return buf[-1]

    def n_samples(self, symbol: str) -> int:
        sym = symbol.upper()
        with self._lock:
            return len(self._samples.get(sym, ()))

    def all_symbols(self) -> list[str]:
        with self._lock:
            return list(self._samples.keys())

    # ------------------------------------------------------------------
    # Anchor resolution
    # ------------------------------------------------------------------

    def anchor(
        self,
        symbol: str,
        period_seconds: int,
        *,
        now_unix: float | None = None,
    ) -> WindowAnchor | None:
        """Resolve ``(spot_at_window_start, spot_now, seconds_remaining)``.

        Uses the *current* Polymarket window: it starts at the previous
        ``period_seconds`` boundary and ends at the next one. If we don't
        have a sample close enough to the start boundary we fall back to
        the first sample on or after the boundary (acceptable when the
        process just booted; predictor will then over-weight the drift).

        Returns ``None`` when the buffer has no samples for ``symbol``.
        """
        if period_seconds <= 0:
            raise ValueError("period_seconds must be positive")
        sym = symbol.upper()
        now = time.time() if now_unix is None else float(now_unix)
        start = (int(now) // period_seconds) * period_seconds
        end = start + period_seconds
        with self._lock:
            buf = self._samples.get(sym)
            if not buf:
                return None
            samples = list(buf)
        timestamps = [t for t, _ in samples]
        # Find the latest sample <= window start.
        idx = bisect.bisect_right(timestamps, start) - 1
        if idx >= 0:
            spot_at_start = samples[idx][1]
        else:
            # Process booted mid-window — use the earliest sample as proxy.
            spot_at_start = samples[0][1]
        spot_now = samples[-1][1]
        seconds_remaining = max(0.0, end - now)
        return WindowAnchor(
            symbol=sym,
            period_seconds=period_seconds,
            start_unix=int(start),
            end_unix=int(end),
            spot_at_start=spot_at_start,
            spot_now=spot_now,
            seconds_remaining=seconds_remaining,
        )

    def snapshot(self, symbol: str) -> dict[str, object]:
        """Return diagnostics for ``/strategies/crypto/5min/diag``."""
        sym = symbol.upper()
        with self._lock:
            buf = list(self._samples.get(sym, []))
        if not buf:
            return {"symbol": sym, "n_samples": 0}
        first_ts, first_p = buf[0]
        last_ts, last_p = buf[-1]
        return {
            "symbol": sym,
            "n_samples": len(buf),
            "first_ts_unix": first_ts,
            "last_ts_unix": last_ts,
            "span_seconds": last_ts - first_ts,
            "first_mid": first_p,
            "last_mid": last_p,
            "drift_pct": (last_p / first_p - 1.0) if first_p else None,
        }

    def clear(self) -> None:
        with self._lock:
            self._samples.clear()


_STATE: CryptoFiveMinState | None = None


def get_state() -> CryptoFiveMinState:
    """Process-wide singleton accessor."""
    global _STATE
    if _STATE is None:
        _STATE = CryptoFiveMinState()
    return _STATE


def reset_state() -> None:
    """Test hook — replace the singleton with an empty buffer."""
    global _STATE
    _STATE = CryptoFiveMinState()
