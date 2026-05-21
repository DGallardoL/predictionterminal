"""Reverse Factor Finder + Prediction-Driven Alpha Scanner.

Two complementary alpha-discovery primitives that ride on top of the existing
PM↔equity factor graph:

1.  :func:`reverse_find_factors` — given a ticker, find the top-k Polymarket /
    Kalshi markets whose Δlogit best explains its returns. Uses forward
    stepwise selection on R² (greedy add the candidate with the largest
    marginal ΔR² until ``k`` factors are picked).

2.  :func:`prediction_driven_alpha` — given a single PM market, scan a basket
    of equities and return per-ticker univariate β / R² / t-stat. If the
    caller provides a hypothetical ``Δlogit`` move, returns expected returns
    so users can see the cross-sectional impact of a single PM event.

Both functions take pre-fetched price/return inputs as injected fetcher
callables so they're trivially unit-testable without network IO. The router
in :mod:`pfm.reverse_finder_router` wires the production fetchers (cached
yfinance + cached Polymarket).
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date
from typing import Literal, Protocol

import numpy as np
import pandas as pd

from pfm.model import DEFAULT_EPSILON, delta_logit, fit_ols_hac

logger = logging.getLogger(__name__)

ReturnType = Literal["log", "simple"]

# Default basket for the prediction-driven scanner — broad-market ETFs +
# sector ETFs that span most macro factor exposures. Caller can override.
DEFAULT_TICKERS: tuple[str, ...] = (
    "SPY",
    "QQQ",
    "TLT",
    "XLU",
    "XLF",
    "XLE",
    "XLV",
    "XLY",
    "XLP",
    "KRE",
    "SMH",
    "IWM",
)


# --- protocols for injected fetchers ---------------------------------------


class ReturnsFetcher(Protocol):
    """Callable returning a pandas Series of returns for ``(ticker, start, end)``."""

    def __call__(
        self,
        ticker: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
        return_type: ReturnType = "log",
    ) -> pd.Series: ...


class FactorHistoryFetcher(Protocol):
    """Callable returning a DataFrame indexed by UTC-normalised dates with a ``price`` column."""

    def __call__(
        self,
        factor_id: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> pd.DataFrame: ...


# --- internal helpers -------------------------------------------------------


def _to_utc_ts(d: date | pd.Timestamp) -> pd.Timestamp:
    if isinstance(d, pd.Timestamp):
        return d.tz_convert("UTC") if d.tzinfo else d.tz_localize("UTC")
    return pd.Timestamp(d, tz="UTC")


def _univariate_r2(y: pd.Series, x: pd.Series) -> float:
    """Plain univariate R² of ``y`` regressed on ``x`` (centred). NaN-safe."""
    common = y.index.intersection(x.index)
    if len(common) < 5:
        return float("nan")
    yj = y.loc[common].values.astype(float)
    xj = x.loc[common].values.astype(float)
    if np.std(xj) < 1e-12:
        return 0.0
    yc = yj - yj.mean()
    xc = xj - xj.mean()
    beta = float(np.sum(xc * yc) / max(np.sum(xc * xc), 1e-12))
    pred = beta * xc
    ss_res = float(np.sum((yc - pred) ** 2))
    ss_tot = float(np.sum(yc * yc))
    if ss_tot < 1e-12:
        return 0.0
    return 1.0 - ss_res / ss_tot


# --- public API: Reverse Factor Finder --------------------------------------


@dataclass(frozen=True)
class ReverseFactorPick:
    """One factor selected by the reverse finder."""

    factor_id: str
    delta_r_squared: float
    beta: float
    t_stat: float
    vif: float


@dataclass(frozen=True)
class ReverseStreamStep:
    """One streamed event yielded by :func:`iter_reverse_find_factors`.

    ``kind`` discriminates the payload:

    - ``"meta"`` — first event; ``extra`` carries ``n_candidates`` (after
      min-obs filtering) and ``n_obs`` (length of the return series).
    - ``"factor"`` — one per forward-selection step. ``rank`` is 1-based.
    - ``"done"`` — final event with ``total_r_squared`` and ``rejected``
      packaged into ``extra``.

    For ``"factor"`` events the per-pick fields are populated. For
    ``"meta"`` and ``"done"`` the payload lives in ``extra``.
    """

    kind: Literal["meta", "factor", "done"]
    rank: int | None = None
    factor_id: str | None = None
    beta: float | None = None
    t_stat: float | None = None
    delta_r2: float | None = None
    cumulative_r2: float | None = None
    vif: float | None = None
    extra: dict | None = None


def _prepare_reverse_state(
    ticker: str,
    candidate_factor_ids: list[str],
    start: date,
    end: date,
    *,
    return_type: ReturnType,
    epsilon: float,
    min_obs: int,
    returns_fetcher: ReturnsFetcher,
    factor_fetcher: FactorHistoryFetcher,
) -> tuple[pd.Series | None, dict[str, pd.Series], list[str], str | None]:
    """Fetch returns + each candidate's Δlogit and pre-filter on min_obs.

    Returns ``(y, delta_by_id, rejected, note)``. ``y`` is ``None`` when
    the ticker has no usable return history. ``note`` is set to a
    human-readable diagnostic when the caller should short-circuit to an
    empty response.
    """
    start_ts = _to_utc_ts(start)
    end_ts = _to_utc_ts(end)

    y = returns_fetcher(ticker, start_ts, end_ts, return_type=return_type)
    if y is None or y.empty or len(y) < min_obs:
        return y, {}, list(candidate_factor_ids), "ticker has insufficient return history"

    delta_by_id: dict[str, pd.Series] = {}
    rejected: list[str] = []
    seen: set[str] = set()
    for fid in candidate_factor_ids:
        if fid in seen:
            continue
        seen.add(fid)
        try:
            df = factor_fetcher(fid, start_ts, end_ts)
        except Exception as e:  # pragma: no cover - defensive
            logger.debug("reverse_finder: factor %s fetch failed: %s", fid, e)
            rejected.append(fid)
            continue
        if df is None or df.empty or "price" not in df.columns:
            rejected.append(fid)
            continue
        s = delta_logit(df["price"], epsilon=epsilon).rename(fid).dropna()
        common = s.index.intersection(y.index)
        if len(common) < min_obs:
            rejected.append(fid)
            continue
        delta_by_id[fid] = s

    if not delta_by_id:
        return y, delta_by_id, rejected, "no candidate factor has enough overlapping observations"
    return y, delta_by_id, rejected, None


def iter_reverse_find_factors(
    ticker: str,
    candidate_factor_ids: list[str],
    start: date,
    end: date,
    *,
    k: int = 5,
    return_type: ReturnType = "log",
    epsilon: float = DEFAULT_EPSILON,
    min_obs: int = 30,
    returns_fetcher: ReturnsFetcher,
    factor_fetcher: FactorHistoryFetcher,
) -> Iterator[ReverseStreamStep]:
    """Generator variant: yield meta, one event per selected factor, then done.

    Same algorithm as :func:`reverse_find_factors` but emits intermediate
    progress so a streaming endpoint can render bars as picks are made.

    Always yields a ``"meta"`` event first and a ``"done"`` event last,
    even when no candidate survives.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if not candidate_factor_ids:
        raise ValueError("candidate_factor_ids must be non-empty")

    y, delta_by_id, rejected, note = _prepare_reverse_state(
        ticker,
        candidate_factor_ids,
        start,
        end,
        return_type=return_type,
        epsilon=epsilon,
        min_obs=min_obs,
        returns_fetcher=returns_fetcher,
        factor_fetcher=factor_fetcher,
    )

    n_obs = 0 if y is None or y.empty else len(y)
    yield ReverseStreamStep(
        kind="meta",
        extra={
            "ticker": ticker,
            "n_candidates": len(delta_by_id),
            "n_obs": n_obs,
        },
    )

    if note is not None or not delta_by_id or y is None or y.empty:
        yield ReverseStreamStep(
            kind="done",
            extra={
                "total_r_squared": 0.0,
                "n_obs": n_obs,
                "rejected": rejected,
                "note": note,
            },
        )
        return

    selected: list[str] = []
    deltas: list[float] = []
    last_r2 = 0.0
    target_k = min(k, len(delta_by_id))

    for _ in range(target_k):
        remaining = [fid for fid in delta_by_id if fid not in selected]
        if not remaining:
            break
        best_fid: str | None = None
        best_r2 = last_r2
        for fid in remaining:
            try_set = [*selected, fid]
            X = pd.concat([delta_by_id[k_] for k_ in try_set], axis=1).dropna()
            common = X.index.intersection(y.index)
            if len(common) <= len(try_set) + 1 or len(common) < min_obs:
                continue
            try:
                fit = fit_ols_hac(y.loc[common], X.loc[common])
            except (ValueError, RuntimeError, np.linalg.LinAlgError):
                continue
            r2 = fit.stats.r_squared
            if r2 > best_r2 + 1e-9:
                best_r2 = r2
                best_fid = fid
        if best_fid is None:
            break

        selected.append(best_fid)
        deltas.append(best_r2 - last_r2)
        last_r2 = best_r2

        # Re-fit on the cumulative selection so we can report β / t / VIF
        # for the freshly-added factor as part of this step's event.
        X_cum = pd.concat([delta_by_id[fid] for fid in selected], axis=1).dropna()
        common_cum = X_cum.index.intersection(y.index)
        try:
            fit_cum = fit_ols_hac(y.loc[common_cum], X_cum.loc[common_cum])
        except (ValueError, RuntimeError, np.linalg.LinAlgError):
            # Should not happen — we just succeeded with this exact set in
            # the inner loop — but stay defensive.
            yield ReverseStreamStep(
                kind="done",
                extra={
                    "total_r_squared": float(last_r2),
                    "n_obs": len(common_cum),
                    "rejected": rejected,
                    "note": "final fit on selected set failed",
                },
            )
            return

        beta_by_id = {f.factor_id: f.beta for f in fit_cum.factors}
        tstat_by_id = {f.factor_id: f.t_stat for f in fit_cum.factors}
        vif_by_id = dict(fit_cum.diagnostics.vif)

        yield ReverseStreamStep(
            kind="factor",
            rank=len(selected),
            factor_id=best_fid,
            beta=float(beta_by_id.get(best_fid, float("nan"))),
            t_stat=float(tstat_by_id.get(best_fid, float("nan"))),
            delta_r2=float(deltas[-1]),
            cumulative_r2=float(last_r2),
            vif=float(vif_by_id.get(best_fid, 1.0)),
        )

    if not selected:
        yield ReverseStreamStep(
            kind="done",
            extra={
                "total_r_squared": 0.0,
                "n_obs": 0,
                "rejected": rejected,
                "note": "no factor improved R² above zero",
            },
        )
        return

    X_final = pd.concat([delta_by_id[fid] for fid in selected], axis=1).dropna()
    common_final = X_final.index.intersection(y.index)
    yield ReverseStreamStep(
        kind="done",
        extra={
            "total_r_squared": float(last_r2),
            "n_obs": len(common_final),
            "rejected": rejected,
        },
    )


def reverse_find_factors(
    ticker: str,
    candidate_factor_ids: list[str],
    start: date,
    end: date,
    *,
    k: int = 5,
    return_type: ReturnType = "log",
    epsilon: float = DEFAULT_EPSILON,
    min_obs: int = 30,
    returns_fetcher: ReturnsFetcher,
    factor_fetcher: FactorHistoryFetcher,
) -> dict:
    """Forward-stepwise: top-k PM markets that best explain ``ticker`` returns.

    Args:
        ticker: equity symbol (e.g. ``"NVDA"``).
        candidate_factor_ids: pool of factor ids to consider.
        start: inclusive start date.
        end: inclusive end date.
        k: max factors to return (the actual list may be shorter if no further
            factor improves R²).
        return_type: ``"log"`` or ``"simple"`` returns.
        epsilon: clipping for the logit transform.
        min_obs: minimum overlapping observations between the ticker and a
            candidate's Δlogit; candidates below this are dropped.
        returns_fetcher: injected callable that returns a return series.
        factor_fetcher: injected callable that returns a factor price DataFrame.

    Returns:
        dict ready for JSON serialisation:

        ``{"ticker", "top_factors": [{"factor_id", "delta_r_squared", "beta",
        "t_stat", "vif"}], "total_r_squared", "n_obs", "rejected": [...]}``.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if not candidate_factor_ids:
        raise ValueError("candidate_factor_ids must be non-empty")

    start_ts = _to_utc_ts(start)
    end_ts = _to_utc_ts(end)

    y = returns_fetcher(ticker, start_ts, end_ts, return_type=return_type)
    if y is None or y.empty or len(y) < min_obs:
        return {
            "ticker": ticker,
            "top_factors": [],
            "total_r_squared": 0.0,
            "n_obs": 0 if y is None else len(y),
            "rejected": list(candidate_factor_ids),
            "note": "ticker has insufficient return history",
        }

    # Pre-pull each factor's Δlogit on the [start, end] window. Drop anything
    # that doesn't reach min_obs overlap with the return series so the
    # stepwise loop is cheap.
    delta_by_id: dict[str, pd.Series] = {}
    rejected: list[str] = []
    seen: set[str] = set()
    for fid in candidate_factor_ids:
        if fid in seen:
            continue
        seen.add(fid)
        try:
            df = factor_fetcher(fid, start_ts, end_ts)
        except Exception as e:  # pragma: no cover - defensive
            logger.debug("reverse_finder: factor %s fetch failed: %s", fid, e)
            rejected.append(fid)
            continue
        if df is None or df.empty or "price" not in df.columns:
            rejected.append(fid)
            continue
        s = delta_logit(df["price"], epsilon=epsilon).rename(fid).dropna()
        common = s.index.intersection(y.index)
        if len(common) < min_obs:
            rejected.append(fid)
            continue
        delta_by_id[fid] = s

    if not delta_by_id:
        return {
            "ticker": ticker,
            "top_factors": [],
            "total_r_squared": 0.0,
            "n_obs": 0,
            "rejected": rejected,
            "note": "no candidate factor has enough overlapping observations",
        }

    # Forward stepwise: at each step add the candidate that maximises ΔR²
    # vs the current selection. Stop once we hit ``k`` or no candidate
    # improves the in-sample R² above the current value.
    selected: list[str] = []
    deltas: list[float] = []
    last_r2 = 0.0

    target_k = min(k, len(delta_by_id))
    for _ in range(target_k):
        remaining = [fid for fid in delta_by_id if fid not in selected]
        if not remaining:
            break
        best_fid: str | None = None
        best_r2 = last_r2
        for fid in remaining:
            try_set = [*selected, fid]
            X = pd.concat([delta_by_id[k_] for k_ in try_set], axis=1).dropna()
            common = X.index.intersection(y.index)
            if len(common) <= len(try_set) + 1 or len(common) < min_obs:
                continue
            try:
                fit = fit_ols_hac(y.loc[common], X.loc[common])
            except (ValueError, RuntimeError, np.linalg.LinAlgError):
                continue
            r2 = fit.stats.r_squared
            if r2 > best_r2 + 1e-9:
                best_r2 = r2
                best_fid = fid
        if best_fid is None:
            break
        selected.append(best_fid)
        deltas.append(best_r2 - last_r2)
        last_r2 = best_r2

    if not selected:
        return {
            "ticker": ticker,
            "top_factors": [],
            "total_r_squared": 0.0,
            "n_obs": 0,
            "rejected": rejected,
            "note": "no factor improved R² above zero",
        }

    # Final fit on the chosen subset to recover β / t / VIF for the response.
    X_final = pd.concat([delta_by_id[fid] for fid in selected], axis=1).dropna()
    common = X_final.index.intersection(y.index)
    y_final = y.loc[common]
    X_final = X_final.loc[common]
    fit = fit_ols_hac(y_final, X_final)
    beta_by_id = {f.factor_id: f.beta for f in fit.factors}
    tstat_by_id = {f.factor_id: f.t_stat for f in fit.factors}
    vif_by_id = dict(fit.diagnostics.vif)

    top_factors = [
        {
            "factor_id": fid,
            "delta_r_squared": float(deltas[i]),
            "beta": float(beta_by_id.get(fid, float("nan"))),
            "t_stat": float(tstat_by_id.get(fid, float("nan"))),
            "vif": float(vif_by_id.get(fid, 1.0)),
        }
        for i, fid in enumerate(selected)
    ]

    return {
        "ticker": ticker,
        "top_factors": top_factors,
        "total_r_squared": float(fit.stats.r_squared),
        "n_obs": len(common),
        "rejected": rejected,
    }


# --- public API: Prediction-Driven Alpha Scanner ----------------------------


def prediction_driven_alpha(
    factor_id: str,
    candidate_tickers: list[str] | None = None,
    *,
    window_days: int = 252,
    top_n: int = 12,
    delta_logit_assumed: float | None = None,
    return_type: ReturnType = "log",
    epsilon: float = DEFAULT_EPSILON,
    min_obs: int = 30,
    end: date | None = None,
    factor_name: str | None = None,
    returns_fetcher: ReturnsFetcher,
    factor_fetcher: FactorHistoryFetcher,
) -> dict:
    """Univariate β / R² scan: which equities load on a single PM factor.

    For each ticker we fit ``r_t = α + β · Δlogit(p_t) + ε_t`` (HAC SEs) on the
    last ``window_days`` calendar days. Tickers are ranked by
    ``|β| · R²`` (signal strength × explanatory power).

    Args:
        factor_id: id of the PM/Kalshi market to scan.
        candidate_tickers: equities to scan. If ``None``, uses
            :data:`DEFAULT_TICKERS`.
        window_days: lookback window in calendar days.
        top_n: cap the response at this many tickers (keeping the best-ranked
            ones).
        delta_logit_assumed: optional hypothetical Δlogit (e.g. ``+0.5`` if
            the user wants to know "what happens to each ticker if the market
            jumps from 50% to 65%"). When provided, populates
            ``expected_return_pct = β · Δlogit_assumed * 100``.
        return_type: ``"log"`` or ``"simple"`` returns.
        epsilon: logit clip.
        min_obs: skip tickers with fewer overlapping observations than this.
        end: inclusive upper bound. Defaults to today (UTC).
        factor_name: human-readable name for the factor. Falls back to
            ``factor_id``.
        returns_fetcher: injected returns fetcher.
        factor_fetcher: injected factor-history fetcher.

    Returns:
        dict ready for JSON serialisation.
    """
    if window_days < min_obs + 5:
        raise ValueError(f"window_days ({window_days}) too small for min_obs ({min_obs})")
    tickers = list(candidate_tickers) if candidate_tickers else list(DEFAULT_TICKERS)
    if not tickers:
        raise ValueError("candidate_tickers must be non-empty")

    end_d = end or date.today()
    end_ts = _to_utc_ts(end_d)
    start_ts = end_ts - pd.Timedelta(days=window_days)

    # Pull the factor once and pre-compute Δlogit.
    try:
        df = factor_fetcher(factor_id, start_ts, end_ts)
    except Exception as e:  # pragma: no cover - defensive
        return {
            "factor_id": factor_id,
            "factor_name": factor_name or factor_id,
            "tickers": [],
            "ranked_by": "abs(beta)*r_squared",
            "delta_logit_assumed": delta_logit_assumed,
            "note": f"factor fetch failed: {e!r}",
        }
    if df is None or df.empty or "price" not in df.columns:
        return {
            "factor_id": factor_id,
            "factor_name": factor_name or factor_id,
            "tickers": [],
            "ranked_by": "abs(beta)*r_squared",
            "delta_logit_assumed": delta_logit_assumed,
            "note": "factor has no price history",
        }
    dl = delta_logit(df["price"], epsilon=epsilon).rename(factor_id).dropna()
    if len(dl) < min_obs:
        return {
            "factor_id": factor_id,
            "factor_name": factor_name or factor_id,
            "tickers": [],
            "ranked_by": "abs(beta)*r_squared",
            "delta_logit_assumed": delta_logit_assumed,
            "note": f"factor has only {len(dl)} usable Δlogit obs (< {min_obs})",
        }

    rows: list[dict] = []
    skipped: list[dict] = []
    for ticker in tickers:
        try:
            y = returns_fetcher(ticker, start_ts, end_ts, return_type=return_type)
        except Exception as e:
            skipped.append({"ticker": ticker, "reason": f"returns fetch: {e!r}"})
            continue
        if y is None or y.empty:
            skipped.append({"ticker": ticker, "reason": "no return data"})
            continue
        common = y.index.intersection(dl.index)
        if len(common) < min_obs:
            skipped.append({"ticker": ticker, "reason": f"only {len(common)} overlap"})
            continue
        X = dl.loc[common].to_frame()
        y_sub = y.loc[common]
        try:
            fit = fit_ols_hac(y_sub, X)
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as e:
            skipped.append({"ticker": ticker, "reason": f"fit failed: {e!r}"})
            continue
        coef = fit.factors[0]
        row = {
            "ticker": ticker,
            "beta": float(coef.beta),
            "r_squared": float(fit.stats.r_squared),
            "t_stat": float(coef.t_stat),
            "n_obs": len(common),
            "expected_return_pct": (
                float(coef.beta * delta_logit_assumed * 100.0)
                if delta_logit_assumed is not None
                else None
            ),
        }
        rows.append(row)

    rows.sort(key=lambda r: abs(r["beta"]) * max(r["r_squared"], 0.0), reverse=True)
    rows = rows[: max(top_n, 0)]

    return {
        "factor_id": factor_id,
        "factor_name": factor_name or factor_id,
        "tickers": rows,
        "ranked_by": "abs(beta)*r_squared",
        "delta_logit_assumed": delta_logit_assumed,
        "window_days": window_days,
        "skipped": skipped,
    }


__all__ = [
    "DEFAULT_TICKERS",
    "FactorHistoryFetcher",
    "ReturnsFetcher",
    "ReverseFactorPick",
    "ReverseStreamStep",
    "iter_reverse_find_factors",
    "prediction_driven_alpha",
    "reverse_find_factors",
]
