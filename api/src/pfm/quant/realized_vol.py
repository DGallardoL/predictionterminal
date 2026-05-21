"""Realized-volatility estimators for option-pricing factors.

Implements a small library of well-known volatility estimators used in option
pricing and risk modelling, plus a harmonic-mean aggregator across factors.

References
----------
- Parkinson, M. (1980). "The Extreme Value Method for Estimating the Variance
  of the Rate of Return."
- Garman, M. B., & Klass, M. J. (1980). "On the Estimation of Security Price
  Volatilities from Historical Data."
- Rogers, L. C. G., & Satchell, S. E. (1991). "Estimating Variance from High,
  Low and Closing Prices."
- Yang, D., & Zhang, Q. (2000). "Drift-Independent Volatility Estimation Based
  on High, Low, Open, and Close Prices."

Conventions
-----------
- Inputs are price series for OHLC methods. The ``returns`` argument is named
  for the close-to-close case where log returns are passed directly. For OHLC
  methods, pass a structured ``np.ndarray`` with named fields
  ``open``/``high``/``low``/``close`` OR a 2-D array shaped (n, 4) in that
  column order.
- Annualisation factor defaults to 252 trading days. Daily-frequency vol is
  multiplied by sqrt(ann_factor).
- All log operations clip nonpositive prices defensively; callers are expected
  to pre-clean their data.

Public API
----------
- :func:`realized_vol` — dispatches to the chosen estimator.
- :func:`realized_vol_harmonic_mean` — harmonic mean of independent realised
  vol estimates across a basket of factors.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

Method = Literal[
    "close-to-close",
    "parkinson",
    "garman-klass",
    "rogers-satchell",
    "yang-zhang",
]

VALID_METHODS: tuple[str, ...] = (
    "close-to-close",
    "parkinson",
    "garman-klass",
    "rogers-satchell",
    "yang-zhang",
)


def _coerce_ohlc(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (open, high, low, close) as 1-D float arrays.

    Accepts either a structured array with fields o/h/l/c, or a 2-D array of
    shape (n, 4) with columns [open, high, low, close].
    """
    if arr.dtype.names is not None:
        names = {n.lower() for n in arr.dtype.names}
        if {"open", "high", "low", "close"}.issubset(names):
            return (
                np.asarray(arr["open"], dtype=float),
                np.asarray(arr["high"], dtype=float),
                np.asarray(arr["low"], dtype=float),
                np.asarray(arr["close"], dtype=float),
            )
        raise ValueError("structured array must have open/high/low/close fields")

    arr2 = np.asarray(arr, dtype=float)
    if arr2.ndim != 2 or arr2.shape[1] != 4:
        raise ValueError(
            "OHLC array must be 2-D with shape (n, 4) in [open, high, low, close] order"
        )
    return arr2[:, 0], arr2[:, 1], arr2[:, 2], arr2[:, 3]


def _annualize(daily_sigma: float | np.ndarray, ann_factor: int, do: bool) -> float | np.ndarray:
    if not do:
        return daily_sigma
    return daily_sigma * np.sqrt(ann_factor)


def _close_to_close(returns: np.ndarray) -> float:
    """Stdev of log returns (sample std, ddof=1). Returns 0 for length < 2."""
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    if r.size < 2:
        return 0.0
    return float(np.std(r, ddof=1))


def _parkinson(high: np.ndarray, low: np.ndarray) -> float:
    """Parkinson (1980) estimator.

    sigma^2 = (1 / (4 ln 2)) * mean( (ln(H/L))^2 )
    """
    h = np.asarray(high, dtype=float)
    l = np.asarray(low, dtype=float)
    mask = (h > 0) & (l > 0) & ~np.isnan(h) & ~np.isnan(l)
    h, l = h[mask], l[mask]
    if h.size == 0:
        return 0.0
    rng = np.log(h / l)
    var = np.mean(rng**2) / (4.0 * np.log(2.0))
    return float(np.sqrt(max(var, 0.0)))


def _garman_klass(open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> float:
    """Garman-Klass (1980).

    sigma^2 = mean( 0.5 * (ln(H/L))^2 - (2 ln 2 - 1) * (ln(C/O))^2 )
    """
    h = np.asarray(high, dtype=float)
    l = np.asarray(low, dtype=float)
    c = np.asarray(close, dtype=float)
    o = np.asarray(open_, dtype=float)
    mask = (h > 0) & (l > 0) & (c > 0) & (o > 0)
    mask &= ~(np.isnan(h) | np.isnan(l) | np.isnan(c) | np.isnan(o))
    h, l, c, o = h[mask], l[mask], c[mask], o[mask]
    if h.size == 0:
        return 0.0
    hl = np.log(h / l)
    co = np.log(c / o)
    var = np.mean(0.5 * hl**2 - (2.0 * np.log(2.0) - 1.0) * co**2)
    return float(np.sqrt(max(var, 0.0)))


def _rogers_satchell(
    open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray
) -> float:
    """Rogers-Satchell (1991), drift-independent.

    sigma^2 = mean( ln(H/C)*ln(H/O) + ln(L/C)*ln(L/O) )
    """
    h = np.asarray(high, dtype=float)
    l = np.asarray(low, dtype=float)
    c = np.asarray(close, dtype=float)
    o = np.asarray(open_, dtype=float)
    mask = (h > 0) & (l > 0) & (c > 0) & (o > 0)
    mask &= ~(np.isnan(h) | np.isnan(l) | np.isnan(c) | np.isnan(o))
    h, l, c, o = h[mask], l[mask], c[mask], o[mask]
    if h.size == 0:
        return 0.0
    term = np.log(h / c) * np.log(h / o) + np.log(l / c) * np.log(l / o)
    var = np.mean(term)
    return float(np.sqrt(max(var, 0.0)))


def _yang_zhang(open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> float:
    """Yang-Zhang (2000): overnight + open-to-close + Rogers-Satchell.

    sigma^2 = sigma_o^2 + k * sigma_c^2 + (1 - k) * sigma_rs^2
    where sigma_o^2 = var(ln O_t / C_{t-1}) (overnight),
          sigma_c^2 = var(ln C_t / O_t)     (open-to-close),
          sigma_rs^2 = Rogers-Satchell estimator,
          k = 0.34 / (1.34 + (n+1)/(n-1)).
    """
    h = np.asarray(high, dtype=float)
    l = np.asarray(low, dtype=float)
    c = np.asarray(close, dtype=float)
    o = np.asarray(open_, dtype=float)
    mask = (h > 0) & (l > 0) & (c > 0) & (o > 0)
    mask &= ~(np.isnan(h) | np.isnan(l) | np.isnan(c) | np.isnan(o))
    h, l, c, o = h[mask], l[mask], c[mask], o[mask]
    n = h.size
    if n < 2:
        return 0.0

    overnight = np.log(o[1:] / c[:-1])
    open_close = np.log(c / o)
    sigma_o2 = float(np.var(overnight, ddof=1)) if overnight.size >= 2 else 0.0
    sigma_c2 = float(np.var(open_close, ddof=1)) if open_close.size >= 2 else 0.0

    # Rogers-Satchell variance (not sigma)
    term = np.log(h / c) * np.log(h / o) + np.log(l / c) * np.log(l / o)
    sigma_rs2 = float(np.mean(term))

    k = 0.34 / (1.34 + (n + 1) / (n - 1))
    var = sigma_o2 + k * sigma_c2 + (1.0 - k) * sigma_rs2
    return float(np.sqrt(max(var, 0.0)))


def _rolling(
    fn,
    arr: np.ndarray,
    window: int,
    is_ohlc: bool,
) -> np.ndarray:
    """Apply ``fn`` on rolling windows of length ``window``.

    For 1-D return series, slides over the first axis. For OHLC arrays, slides
    over rows. Output length is ``len(arr) - window + 1``.
    """
    n = arr.shape[0]
    if window <= 0:
        raise ValueError("window must be a positive integer")
    if window > n:
        return np.array([], dtype=float)
    out = np.empty(n - window + 1, dtype=float)
    for i in range(out.size):
        chunk = arr[i : i + window]
        out[i] = fn(chunk) if not is_ohlc else fn(*_coerce_ohlc(chunk))
    return out


def realized_vol(
    returns: np.ndarray,
    *,
    method: str = "close-to-close",
    annualize: bool = True,
    ann_factor: int = 252,
    window: int | None = None,
) -> float | np.ndarray:
    """Compute realised volatility under the chosen estimator.

    Parameters
    ----------
    returns
        For ``close-to-close``: 1-D array of daily log returns.
        For OHLC methods: structured array with fields open/high/low/close, or
        2-D ``(n, 4)`` ndarray with columns ``[open, high, low, close]``.
    method
        One of ``close-to-close`` / ``parkinson`` / ``garman-klass`` /
        ``rogers-satchell`` / ``yang-zhang``.
    annualize
        If ``True``, multiply the per-period stdev by ``sqrt(ann_factor)``.
    ann_factor
        Annualisation factor. Defaults to 252.
    window
        If provided, computes rolling vol over windows of this length and
        returns a 1-D array of shape ``(n - window + 1,)``. If ``None``, the
        estimator runs on the full series and returns a scalar.

    Returns
    -------
    float or np.ndarray
        Scalar realised vol if ``window is None``, else a 1-D rolling series.
    """
    if method not in VALID_METHODS:
        raise ValueError(f"unknown method '{method}'. Valid: {VALID_METHODS}")
    if ann_factor <= 0:
        raise ValueError("ann_factor must be positive")

    arr = np.asarray(returns)
    if arr.size == 0:
        if window is None:
            return 0.0
        return np.array([], dtype=float)

    if method == "close-to-close":
        flat = arr.astype(float).ravel() if arr.ndim > 1 else arr.astype(float)
        if window is None:
            sigma = _close_to_close(flat)
            return float(_annualize(sigma, ann_factor, annualize))
        roll = _rolling(_close_to_close, flat, window, is_ohlc=False)
        return _annualize(roll, ann_factor, annualize)

    # OHLC methods
    if method == "parkinson":

        def fn(o, h, l, c):
            return _parkinson(h, l)
    elif method == "garman-klass":
        fn = _garman_klass
    elif method == "rogers-satchell":
        fn = _rogers_satchell
    else:  # yang-zhang
        fn = _yang_zhang

    if window is None:
        o, h, l, c = _coerce_ohlc(arr)
        sigma = fn(o, h, l, c)
        return float(_annualize(sigma, ann_factor, annualize))

    roll = _rolling(fn, arr, window, is_ohlc=True)
    return _annualize(roll, ann_factor, annualize)


def realized_vol_harmonic_mean(returns_series_list: list[np.ndarray]) -> float:
    """Harmonic mean of close-to-close realised vols across a basket.

    Useful as a basket aggregator that downweights outliers. NaN and zero
    series are skipped (a zero vol would make the harmonic mean collapse to 0,
    which is rarely what callers want for a basket of factors).

    Parameters
    ----------
    returns_series_list
        List of 1-D arrays of log returns. Each series is treated independently
        and run through the close-to-close estimator (annualised, ann_factor
        =252).

    Returns
    -------
    float
        Harmonic mean of the per-series annualised vols. Returns 0.0 if the
        list is empty or every series collapses to zero / NaN.
    """
    if not returns_series_list:
        return 0.0
    vols: list[float] = []
    for s in returns_series_list:
        if s is None:
            continue
        arr = np.asarray(s, dtype=float)
        if arr.size < 2:
            continue
        v = _close_to_close(arr) * np.sqrt(252)
        if np.isfinite(v) and v > 0:
            vols.append(v)
    if not vols:
        return 0.0
    return float(len(vols) / np.sum(1.0 / np.array(vols)))
